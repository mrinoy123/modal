import modal
import os
import sys
import io
import gc

# =========================================================
# IMAGE DEFINITION
# =========================================================

# We use the official CUDA devel image as the foundation
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10"
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.9", # Optimized for L4/A10G GPUs
        "FORCE_CUDA": "1",
        "MAX_JOBS": "4", # Prevents build crashes
        "PYTHONUNBUFFERED": "1",
    })
    .apt_install(
        "git", "wget", "build-essential", "clang", "cmake", 
        "ninja-build", "libgl1-mesa-glx", "libglib2.0-0", 
        "libopenblas-dev", "libsm6", "libxext6", "libxrender1", "libgomp1"
    )
    .pip_install(
        "setuptools", "wheel", "ninja", "packaging"
    )
    # Install PyTorch with specific CUDA 12.4 support
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        index_url="https://download.pytorch.org/whl/cu124"
    )
    # Core AI and 3D Processing packages
    .pip_install(
        "numpy==1.26.4",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "diffusers==0.31.0",
        "safetensors", "sentencepiece", "einops", "omegaconf",
        "opencv-python", "imageio", "scipy", "pandas", "tqdm", "pillow",
        "open3d==0.18.0", "trimesh", "pymeshlab==2022.2.post3", "pygltflib",
        "boto3==1.34.131", "botocore==1.34.131"
    )
    # FIXED: Install Flash Attention and GSplat from official compiled wheels 
    # instead of local paths that don't exist during build.
    .run_commands(
        "pip install flash-attn --no-build-isolation",
        "pip install gsplat==1.5.3"
    )
    .run_commands(
        "pip install --force-reinstall numpy==1.26.4",
        "pip cache purge"
    )
)

# =========================================================
# APP CONFIGURATION
# =========================================================

app = modal.App("hyworld-production")

# Ensure these match the names of the volumes you created in Modal
weights_vol = modal.Volume.from_name("weights-hy-world-2", create_if_missing=True)
cache_vol = modal.Volume.from_name("hyworld-cache", create_if_missing=True)

# =========================================================
# GENERATOR LOGIC
# =========================================================

def generate_world(input_img, base_name, prompt):
    import torch
    
    # Verify hardware acceleration
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not detected by Torch")

    # Optimization settings for HY-World
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # 1. Resolve Code Paths
    # We look specifically in /weights/ where the model code should be uploaded
    CODE_ROOT = "/weights/HY-World-2.0"
    HELPER_ROOT = "/weights/helpers"

    if not os.path.exists(CODE_ROOT):
        # Fallback search if path is different
        search_paths = ["/weights", "/root", "/mnt"]
        found = False
        for p in search_paths:
            candidate = os.path.join(p, "HY-World-2.0")
            if os.path.exists(candidate):
                CODE_ROOT = candidate
                HELPER_ROOT = os.path.join(p, "helpers")
                found = True
                break
        if not found:
            raise FileNotFoundError(f"HY-World-2.0 not found in /weights. Please check your Volume content.")

    print(f"Using Code Root: {CODE_ROOT}")
    sys.path.insert(0, CODE_ROOT)
    os.chdir(CODE_ROOT)

    # 2. Imports (Lazy loading to avoid build-time issues)
    from pipelines import HYWorld2Pipeline
    from utils.meshing import splat_to_glb

    # 3. Load Pipeline
    print("Initializing HY-World-2.0 Pipeline...")
    pipeline = HYWorld2Pipeline.from_pretrained(
        CODE_ROOT,
        llm_path=os.path.join(HELPER_ROOT, "llm"),
        vision_path=os.path.join(HELPER_ROOT, "siglip"),
        device="cuda",
        torch_dtype=torch.bfloat16
    )

    # 4. Run Inference
    print(f"Generating 3D assets for prompt: {prompt}")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = pipeline(
            image=input_img,
            prompt=prompt,
            negative_prompt="blurry, messy, low quality, photorealistic, realistic texture",
            target_resolution=1024,
            output_type="3dgs"
        )

    # 5. Export to GLB
    output_dir = "/tmp/hyworld_output"
    os.makedirs(output_dir, exist_ok=True)
    glb_path = os.path.join(output_dir, f"{base_name}.glb")

    print("Converting Splat to GLB (this may take a moment)...")
    splat_to_glb(
        result,
        output_path=glb_path,
        simplify_ratio=0.8,
        texture_size=2048
    )

    # 6. Memory Cleanup
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return glb_path

# =========================================================
# MAIN WORKER FUNCTION
# =========================================================

@app.function(
    image=image,
    gpu="L4", # L4 is cost-effective and powerful for this task
    timeout=3600,
    max_containers=1,
    volumes={
        "/weights": weights_vol,
        "/cache": cache_vol
    }
)
def process_cloudflare_queue(cfg: dict):
    import boto3
    from PIL import Image

    # Connect to R2
    s3 = boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"]
    )

    print("Polling Cloudflare R2 for new tasks...")
    response = s3.list_objects_v2(Bucket=cfg["bucket"], Prefix="queue/")

    if "Contents" not in response:
        print("Queue is empty.")
        return

    for obj in response["Contents"]:
        key = obj["Key"]
        # Skip directories and non-image files
        if key.endswith("/") or not key.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        print(f"Processing Task: {key}")
        
        try:
            # Download Image
            data = s3.get_object(Bucket=cfg["bucket"], Key=key)
            img = Image.open(io.BytesIO(data["Body"].read())).convert("RGB")
            
            # Get prompt from metadata or use default
            prompt = data.get("Metadata", {}).get("prompt", "comic cinematic city world")
            base_name = os.path.splitext(os.path.basename(key))[0]

            # Generate 3D World
            glb_file = generate_world(img, base_name, prompt)

            # Upload Result
            print(f"Uploading result: {base_name}.glb")
            with open(glb_file, "rb") as f:
                s3.put_object(
                    Bucket=cfg["bucket"],
                    Key=f"output/{base_name}.glb",
                    Body=f,
                    ContentType="model/gltf-binary"
                )

            # Clean up processed item
            s3.delete_object(Bucket=cfg["bucket"], Key=key)
            print("Successfully processed and deleted from queue.")

        except Exception as e:
            print(f"CRITICAL ERROR processing {key}: {str(e)}")
            # Move to failed folder
            failed_key = key.replace("queue/", "failed/", 1)
            s3.copy_object(
                Bucket=cfg["bucket"],
                CopySource={"Bucket": cfg["bucket"], "Key": key},
                Key=failed_key
            )
            s3.delete_object(Bucket=cfg["bucket"], Key=key)

    # Persist cache changes
    cache_vol.commit()

# =========================================================
# LOCAL ENTRYPOINT
# =========================================================

@app.local_entrypoint()
def main():
    # It is recommended to use modal.Secret for these in production
    config = {
        "endpoint": "https://4d91f4d3d0366568a54ffa32ffcb7bf4.r2.cloudflarestorage.com",
        "access_key": "3c33425ba6e5abbd3e63afab14dc8866",
        "secret_key": "d65f107bb61093843c6dd980c764443fdf50924a7701078b99f007d3060e25a8",
        "bucket": "video-asset-files-storage-workflow"
    }
    
    process_cloudflare_queue.remote(config)
