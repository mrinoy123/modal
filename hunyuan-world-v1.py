import modal
import os
import sys
import io
import gc
import shutil
import glob
import subprocess

# =========================================================
# 1. IMAGE CONFIGURATION (Optimized for HW-1 + L4)
# =========================================================
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .env({
        "TORCH_CUDA_ARCH_LIST": "8.9", # L4 optimization
        "FORCE_CUDA": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1"
    })
    .apt_install(
        "git", "build-essential", "cmake", "libgl1-mesa-glx", 
        "libglib2.0-0", "wget", "libdraco-dev"
    )
    .pip_install(
        "torch==2.3.1", "torchvision", "transformers", "diffusers==0.30.0",
        "accelerate", "sentencepiece", "huggingface_hub", "hf-transfer",
        "opencv-python", "trimesh", "pillow", "einops", "omegaconf",
        "scipy", "onnxruntime-gpu", "boto3", "segment-anything", "plyfile"
    )
    # Install GroundingDINO from source (Crucial for WorldGen)
    .run_commands(
        "pip install git+https://github.com/IDEA-Research/GroundingDINO.git",
        "rm -rf /root/HunyuanWorld && git clone --depth 1 https://github.com/tencent/HunyuanWorld-1.git /root/HunyuanWorld"
    )
)

app = modal.App("hyworld1-hallucination-engine", image=image)
weights_vol = modal.Volume.from_name("weights-hy-world-1")

# =========================================================
# 2. GENERATION ENGINE (Hallucination + 3D Mapping)
# =========================================================
def generate_hallucinated_world(input_image, prompt, base_name):
    import torch
    from PIL import Image
    
    ROOT = "/root/HunyuanWorld"
    sys.path.append(ROOT)

    # 🕵️ THE RESILIENT LINKER (V1 Optimized)
    # This finds your weights on the Modal Volume and symlinks them to /root/HunyuanWorld/weights
    def resolve_and_link_weights(volume_path="/weights", project_path=f"{ROOT}/weights"):
        print(f"🔍 Linker: Scanning {volume_path} for HunyuanWorld-1 weights...")
        os.makedirs(project_path, exist_ok=True)
        
        # We need to map the subfolders you created in the downloader script
        mapping = {
            "flux": "flux",
            "hunyuan_world": "hunyuan_world",
            "text_encoders": "text_encoders",
            "annotators": "annotators"
        }
        
        for vol_sub, proj_sub in mapping.items():
            src = os.path.join(volume_path, vol_sub)
            dst = os.path.join(project_path, proj_sub)
            if os.path.exists(src) and not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                    print(f"🔗 Linked: {vol_sub} -> {proj_sub}")
                except Exception as e:
                    print(f"⚠️ Link failed for {vol_sub}: {e}")

    # Initialize paths
    resolve_and_link_weights()
    os.chdir(ROOT)

    from models.pano_gen_pipeline import HunyuanWorldPanoGenPipeline
    from models.scene_gen_pipeline import HunyuanWorldSceneGenPipeline

    # STAGE 1: PanoGen (Hallucinate the 360-degree environment)
    print("🚀 [Stage 1] Hallucinating 360 Panorama (FP8 Mode)...")
    
    pano_pipe = HunyuanWorldPanoGenPipeline.from_pretrained(
        f"{ROOT}/weights/hunyuan_world",
        flux_model_path=f"{ROOT}/weights/flux",
        text_encoder_path=f"{ROOT}/weights/text_encoders",
        torch_dtype=torch.float16,
        device="cuda"
    )
    
    # L4 24GB Memory Management
    pano_pipe.enable_model_cpu_offload()

    with torch.inference_mode():
        # The model uses the reference image + prompt to hallucinate the rest of the world
        hallucinated_pano = pano_pipe(
            image=input_image,
            prompt=prompt,
            num_inference_steps=30,
            height=512,
            width=1024
        ).images[0]

    pano_path = f"/tmp/{base_name}_pano.png"
    hallucinated_pano.save(pano_path)

    # Clean up PanoGen VRAM to make room for SceneGen
    del pano_pipe
    gc.collect()
    torch.cuda.empty_cache()

    # STAGE 2: SceneGen (Convert Panorama to 3D Scene)
    print("🚀 [Stage 2] Building 3D Geometric Scene...")
    
    scene_pipe = HunyuanWorldSceneGenPipeline(
        model_root=f"{ROOT}/weights/hunyuan_world",
        moge_path=f"{ROOT}/weights/annotators/moge",
        sam_path=f"{ROOT}/weights/annotators/sam/sam_vit_h_4b8939.pth",
        grounding_dino_path=f"{ROOT}/weights/annotators/groundingdino",
        device="cuda"
    )

    # Labels guide the segmentation of the hallucinated world
    labels = "floor, walls, ceiling, sky, furniture, windows"

    with torch.inference_mode():
        world_mesh = scene_pipe(
            panorama_path=pano_path,
            labels=labels,
            output_type="mesh"
        )

    output_ply = f"/tmp/{base_name}_world.ply"
    world_mesh.export(output_ply)

    # Final cleanup
    del scene_pipe
    gc.collect()
    torch.cuda.empty_cache()

    return pano_path, output_ply

# =========================================================
# 3. R2 WORKER & QUEUE LOGIC
# =========================================================
@app.function(
    gpu="L4",
    timeout=7200,
    volumes={"/weights": weights_vol},
    container_idle_timeout=180
)
def process_cloudflare_queue(cfg: dict):
    import boto3
    from PIL import Image

    print("☁️ Connecting to Cloudflare R2...")
    s3 = boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"]
    )

    response = s3.list_objects_v2(Bucket=cfg["bucket"], Prefix="queue/")
    if "Contents" not in response:
        print("📭 Queue empty.")
        return

    for obj in response["Contents"]:
        key = obj["Key"]
        if key == "queue/" or not key.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        print(f"📥 Processing Image: {key}")
        try:
            # 1. Download
            data = s3.get_object(Bucket=cfg["bucket"], Key=key)
            img = Image.open(io.BytesIO(data["Body"].read())).convert("RGB")
            
            # Check for custom prompt in metadata or use default hallucination prompt
            metadata = data.get("Metadata", {})
            prompt = metadata.get("prompt", "a cinematic photorealistic high-resolution 360 panoramic world")
            base_name = os.path.splitext(os.path.basename(key))[0]

            # 2. Run Hallucination Pipeline
            local_pano, local_mesh = generate_hallucinated_world(img, prompt, base_name)
            
            # 3. Upload Results
            print(f"📤 Uploading results for {base_name}...")
            
            # Upload Panorama
            with open(local_pano, "rb") as f:
                s3.put_object(Bucket=cfg["bucket"], Key=f"output/{base_name}_panorama.png", Body=f)
            
            # Upload 3D PLY Scene
            with open(local_mesh, "rb") as f:
                s3.put_object(Bucket=cfg["bucket"], Key=f"output/{base_name}_world.ply", Body=f)

            # 4. Clean up queue
            s3.delete_object(Bucket=cfg["bucket"], Key=key)
            print(f"✅ SUCCESS: Generated world for {base_name}")

        except Exception as e:
            print(f"❌ ERROR processing {key}: {str(e)}")
            # Fail-safe move to failed folder
            try:
                failed_key = key.replace("queue/", "failed/", 1)
                s3.copy_object(Bucket=cfg["bucket"], CopySource={"Bucket": cfg["bucket"], "Key": key}, Key=failed_key)
                s3.delete_object(Bucket=cfg["bucket"], Key=key)
            except: pass

# =========================================================
# 4. ENTRYPOINT
# =========================================================
@app.local_entrypoint()
def main():
    # Credentials from your R2 Storage configuration
    config = {
        "endpoint": "https://4d91f4d3d0366568a54ffa32ffcb7bf4.r2.cloudflarestorage.com",
        "access_key": "3c33425ba6e5abbd3e63afab14dc8866",
        "secret_key": "d65f107bb61093843c6dd980c764443fdf50924a7701078b99f007d3060e25a8",
        "bucket": "video-asset-files-storage-workflow"
    }
    process_cloudflare_queue.remote(config)
