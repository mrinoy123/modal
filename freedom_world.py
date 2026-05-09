import modal
import os
import sys
import io
import gc
import shutil
import glob

# =========================================================
# APP CONFIGURATION
# =========================================================
app = modal.App("hyworld2-hallucination-production")

weights_vol = modal.Volume.from_name("weights-hy-world-2", create_if_missing=True)
cache_vol = modal.Volume.from_name("hyworld2-cache", create_if_missing=True)

# =========================================================
# IMAGE BUILD
# =========================================================
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10"
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "FORCE_CUDA": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/cache/huggingface",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    })
    .apt_install(
        "git", "git-lfs", "wget", "ffmpeg", "libgl1-mesa-glx", 
        "libglib2.0-0", "build-essential", "ninja-build", "cmake"
    )
    .pip_install(
        "pip==25.0.1", "setuptools==70.0.0", "packaging", "wheel", "ninja"
    )
    .pip_install(
        "torch==2.4.0", "torchvision==0.19.0", 
        index_url="https://download.pytorch.org/whl/cu124"
    )
    .run_commands(
        "pip install flash-attn --no-build-isolation",
        "pip install gsplat==1.5.3"
    )
    .pip_install(
        "transformers==4.46.3", "accelerate==1.1.1", "diffusers==0.31.0",
        "einops", "omegaconf", "timm", "scipy", "trimesh",
        "opencv-python-headless==4.9.0.80",
        "pygltflib", "jaxtyping", "boto3", "roma", "pyquaternion", "plyfile"
    )
    .run_commands(
        "git clone https://github.com/Tencent-Hunyuan/HY-World-2.0.git /root/HYWorld",
        "cd /root/HYWorld && pip install -r requirements.txt || true"
    )
)

# =========================================================
# GENERATION ENGINE (L4 Optimized)
# =========================================================
def generate_hallucinated_city(input_image, prompt, output_name):
    import torch
    from PIL import Image
    
    ROOT = "/root/HYWorld"
    
    # 1. Resilient Linker
    def setup_weights(volume_path="/weights", target_path="/root/HYWorld/weights"):
        os.makedirs(target_path, exist_ok=True)
        if not os.path.exists(volume_path): return
        for item in os.listdir(volume_path):
            src, dst = os.path.join(volume_path, item), os.path.join(target_path, item)
            if not os.path.exists(dst):
                try: os.symlink(src, dst)
                except: pass

    setup_weights(volume_path="/weights", target_path=f"{ROOT}/weights")
    if ROOT not in sys.path: sys.path.insert(0, ROOT)
    os.chdir(ROOT)

    # 2. Pipeline Import
    try:
        from hyworld2.pipelines.pipeline_world import HunyuanWorldPipeline
    except ImportError:
        from hyworld2.world.pipeline import HunyuanWorldPipeline

    # 3. Model Loading (VRAM Optimization)
    model_path = f"{ROOT}/weights/HY-World-2.0"
    print(f"Loading Hallucination Engine for L4 (24GB Mode)...")
    
    # Use float16 to save 50% VRAM immediately
    pipe = HunyuanWorldPipeline.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    # CRITICAL: L4 Optimizations
    if torch.cuda.is_available():
        # Moves model parts to GPU only when needed
        pipe.enable_model_cpu_offload() 
        # Prevents VRAM spikes during image decoding
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()

    # 4. Run Hallucination
    print(f"--- Hallucinating City: {prompt} ---")
    with torch.inference_mode():
        output = pipe(
            image=input_image,
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=512,
            width=512
        )

    # 5. Extract Result
    output_dir = "/tmp/final_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Search for output (the pipeline usually returns a directory path)
    search_path = output if isinstance(output, str) else ROOT
    all_files = glob.glob(os.path.join(search_path, "**/*"), recursive=True)
    glb_files = [f for f in all_files if f.endswith(".glb")]
    ply_files = [f for f in all_files if f.endswith("gaussians.ply")]

    if glb_files:
        final_path = os.path.join(output_dir, f"{output_name}.glb")
        shutil.copy(glb_files[0], final_path)
    elif ply_files:
        final_path = os.path.join(output_dir, f"{output_name}.ply")
        shutil.copy(ply_files[0], final_path)
    elif hasattr(output, "export_glb"):
        final_path = os.path.join(output_dir, f"{output_name}.glb")
        output.export_glb(final_path)
    else:
        raise RuntimeError("No 3D asset found in generation output.")

    # Cleanup VRAM before next task
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return final_path

# =========================================================
# MODAL WORKER & R2 LOGIC
# =========================================================
@app.function(
    image=image,
    gpu="L4", # Switched from A100 to L4 24GB
    timeout=7200,
    volumes={"/weights": weights_vol, "/cache": cache_vol}
)
def process_cloudflare_queue(cfg: dict):
    import boto3
    from PIL import Image

    s3 = boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"]
    )

    response = s3.list_objects_v2(Bucket=cfg["bucket"], Prefix="queue/")
    if "Contents" not in response: return

    for obj in response["Contents"]:
        key = obj["Key"]
        if key == "queue/" or not key.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        try:
            data = s3.get_object(Bucket=cfg["bucket"], Key=key)
            img = Image.open(io.BytesIO(data["Body"].read())).convert("RGB")
            
            metadata = data.get("Metadata", {})
            prompt = metadata.get("prompt", "a detailed cinematic 3d city, futuristic style, 360 degree environment")
            base_name = os.path.splitext(os.path.basename(key))[0]

            local_file = generate_hallucinated_city(img, prompt, base_name)
            
            ext = os.path.splitext(local_file)[1]
            output_key = f"output/{base_name}{ext}"
            
            with open(local_file, "rb") as f:
                s3.put_object(
                    Bucket=cfg["bucket"],
                    Key=output_key,
                    Body=f,
                    ContentType="model/gltf-binary" if ext == ".glb" else "application/octet-stream"
                )

            s3.delete_object(Bucket=cfg["bucket"], Key=key)
            print(f"SUCCESS: {output_key}")

        except Exception as e:
            print(f"ERROR: {str(e)}")
            try:
                failed_key = key.replace("queue/", "failed/", 1)
                s3.copy_object(Bucket=cfg["bucket"], CopySource={"Bucket": cfg["bucket"], "Key": key}, Key=failed_key)
                s3.delete_object(Bucket=cfg["bucket"], Key=key)
            except: pass

@app.local_entrypoint()
def main():
    config = {
        "endpoint": "https://4d91f4d3d0366568a54ffa32ffcb7bf4.r2.cloudflarestorage.com",
        "access_key": "3c33425ba6e5abbd3e63afab14dc8866",
        "secret_key": "d65f107bb61093843c6dd980c764443fdf50924a7701078b99f007d3060e25a8",
        "bucket": "video-asset-files-storage-workflow"
    }
    process_cloudflare_queue.remote(config)
