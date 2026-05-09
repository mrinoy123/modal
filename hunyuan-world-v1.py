import modal
import os
import sys
import io
import gc

# =========================================================
# 1. IMAGE CONFIGURATION
# =========================================================
# We explicitly set PYTHONPATH so the container knows where the 'models' folder is.
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .env({
        "TORCH_CUDA_ARCH_LIST": "8.9",       
        "FORCE_CUDA": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1",
        "CUDA_HOME": "/usr/local/cuda",
        "CC": "clang",                       
        "CXX": "clang++",
        "GIT_TERMINAL_PROMPT": "0",
        "PYTHONPATH": "/root/HunyuanWorld"   # CRITICAL FIX: Ensures 'models' is findable
    })
    .apt_install(
        "git", "build-essential", "cmake", "libgl1-mesa-glx", 
        "libglib2.0-0", "wget", "libdraco-dev", 
        "ninja-build", "clang", "llvm"
    )
    .pip_install("pip>=24.0", "wheel", "setuptools", "ninja")
    .pip_install("torch==2.3.1", "torchvision", "torchaudio")
    .pip_install(
        "transformers", "diffusers==0.30.0", "accelerate", "sentencepiece", 
        "huggingface_hub", "hf-transfer", "opencv-python", "trimesh", 
        "pillow", "einops", "omegaconf", "scipy", "onnxruntime-gpu", 
        "boto3", "segment-anything", "plyfile", "pycocotools"
    )
    .run_commands(
        "export PATH=/usr/local/cuda/bin:$PATH && "
        "pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git",
        "git clone --depth 1 https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git /root/HunyuanWorld",
        "touch /root/HunyuanWorld/__init__.py" # Ensure it's treated as a package
    )
)

app = modal.App("hunyuan-world-1-hallucination", image=image)
weights_vol = modal.Volume.from_name("weights-hy-world-1")

# =========================================================
# 2. GENERATION LOGIC
# =========================================================
def generate_hallucinated_world(input_image, prompt, base_name):
    import torch
    from PIL import Image
    
    ROOT = "/root/HunyuanWorld"
    # Force this path to the top of the search list
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    
    os.chdir(ROOT)

    # Weights Linker
    def resolve_and_link_weights(volume_path="/weights", project_path=f"{ROOT}/weights"):
        print(f"🔍 Linking weights from Volume...")
        os.makedirs(project_path, exist_ok=True)
        mapping = ["flux", "hunyuan_world", "text_encoders", "annotators"]
        for folder in mapping:
            src = os.path.join(volume_path, folder)
            dst = os.path.join(project_path, folder)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)
                print(f"🔗 Linked: {folder}")

    resolve_and_link_weights()

    # Deferred imports to ensure sys.path is respected
    try:
        from models.pano_gen_pipeline import HunyuanWorldPanoGenPipeline
        from models.scene_gen_pipeline import HunyuanWorldSceneGenPipeline
    except ImportError as e:
        print(f"❌ Import Failure: {e}")
        print(f"Current sys.path: {sys.path}")
        print(f"Files in ROOT: {os.listdir(ROOT)}")
        raise e

    # STAGE 1: PanoGen
    print("🚀 [Stage 1] Generating 360 Panorama...")
    pano_pipe = HunyuanWorldPanoGenPipeline.from_pretrained(
        f"{ROOT}/weights/hunyuan_world",
        flux_model_path=f"{ROOT}/weights/flux",
        text_encoder_path=f"{ROOT}/weights/text_encoders",
        torch_dtype=torch.float16,
        device="cuda"
    )
    pano_pipe.enable_model_cpu_offload()

    with torch.inference_mode():
        hallucinated_pano = pano_pipe(
            image=input_image,
            prompt=prompt,
            num_inference_steps=30,
            height=512,
            width=1024
        ).images[0]

    pano_path = f"/tmp/{base_name}_pano.png"
    hallucinated_pano.save(pano_path)

    # Cleanup VRAM
    del pano_pipe
    gc.collect()
    torch.cuda.empty_cache()

    # STAGE 2: SceneGen
    print("🚀 [Stage 2] Building 3D Scene...")
    scene_pipe = HunyuanWorldSceneGenPipeline(
        model_root=f"{ROOT}/weights/hunyuan_world",
        moge_path=f"{ROOT}/weights/annotators/moge",
        sam_path=f"{ROOT}/weights/annotators/sam/sam_vit_h_4b8939.pth",
        grounding_dino_path=f"{ROOT}/weights/annotators/groundingdino",
        device="cuda"
    )

    labels = "floor, walls, ceiling, sky, furniture, windows"
    with torch.inference_mode():
        world_mesh = scene_pipe(
            panorama_path=pano_path,
            labels=labels,
            output_type="mesh"
        )

    output_ply = f"/tmp/{base_name}_world.ply"
    world_mesh.export(output_ply)

    del scene_pipe
    gc.collect()
    torch.cuda.empty_cache()

    return pano_path, output_ply

# =========================================================
# 3. R2 QUEUE WORKER
# =========================================================
@app.function(
    gpu="L4",
    timeout=7200,
    volumes={"/weights": weights_vol},
    scaledown_window=180
)
def process_cloudflare_queue(cfg: dict):
    import boto3
    from PIL import Image

    print("☁️ Initializing Cloudflare R2 Connection...")
    s3 = boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"]
    )

    response = s3.list_objects_v2(Bucket=cfg["bucket"], Prefix="queue/")
    if "Contents" not in response:
        print("📭 Queue is empty.")
        return

    for obj in response["Contents"]:
        key = obj["Key"]
        if key == "queue/" or not key.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        print(f"📥 Found Image: {key}")
        try:
            data = s3.get_object(Bucket=cfg["bucket"], Key=key)
            img = Image.open(io.BytesIO(data["Body"].read())).convert("RGB")
            
            metadata = data.get("Metadata", {})
            prompt = metadata.get("prompt", "a cinematic photorealistic 360 panoramic world")
            base_name = os.path.splitext(os.path.basename(key))[0]

            local_pano, local_mesh = generate_hallucinated_world(img, prompt, base_name)
            
            print(f"📤 Uploading results for {base_name}...")
            with open(local_pano, "rb") as f:
                s3.put_object(Bucket=cfg["bucket"], Key=f"output/{base_name}_panorama.png", Body=f)
            with open(local_mesh, "rb") as f:
                s3.put_object(Bucket=cfg["bucket"], Key=f"output/{base_name}_world.ply", Body=f)

            s3.delete_object(Bucket=cfg["bucket"], Key=key)
            print(f"✅ COMPLETED: {base_name}")

        except Exception as e:
            print(f"❌ FAILED {key}: {str(e)}")
            try:
                s3.copy_object(Bucket=cfg["bucket"], CopySource={"Bucket": cfg["bucket"], "Key": key}, Key=key.replace("queue/", "failed/"))
                s3.delete_object(Bucket=cfg["bucket"], Key=key)
            except: pass

# =========================================================
# 4. ENTRYPOINT
# =========================================================
@app.local_entrypoint()
def main():
    config = {
        "endpoint": "https://4d91f4d3d0366568a54ffa32ffcb7bf4.r2.cloudflarestorage.com",
        "access_key": "3c33425ba6e5abbd3e63afab14dc8866",
        "secret_key": "d65f107bb61093843c6dd980c764443fdf50924a7701078b99f007d3060e25a8",
        "bucket": "video-asset-files-storage-workflow"
    }
    process_cloudflare_queue.remote(config)
