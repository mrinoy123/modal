import modal
import os
import sys
import io
import gc

# =========================================================
# 1. IMAGE DEFINITION (Optimized & Error-Proofed)
# =========================================================

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10"
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.9", # L4/A10G/A100 optimized
        "FORCE_CUDA": "1",
        "MAX_JOBS": "4",
        "PYTHONUNBUFFERED": "1",
        "PIP_NO_CACHE_DIR": "1" # Explicitly tell pip not to worry about cache
    })
    .apt_install(
        "git", "wget", "build-essential", "clang", "cmake", 
        "ninja-build", "libgl1-mesa-glx", "libglib2.0-0", 
        "libopenblas-dev", "libsm6", "libxext6", "libxrender1", "libgomp1"
    )
    .pip_install(
        "setuptools", "wheel", "ninja", "packaging"
    )
    # Install Torch 2.4 first to provide the base for CUDA extensions
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        index_url="https://download.pytorch.org/whl/cu124"
    )
    # Core requirements. Pinning Numpy early to avoid 2.0+ conflicts
    .pip_install(
        "numpy==1.26.4", 
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "diffusers==0.31.0",
        "safetensors", "sentencepiece", "einops", "omegaconf",
        "opencv-python", "imageio", "scipy", "pandas", "tqdm", "pillow",
        "open3d==0.18.0", "trimesh", "pymeshlab==2022.2.post3", "pygltflib",
        "boto3==1.34.131", "botocore==1.34.131", "jaxtyping", "rich"
    )
    # Install Flash-Attn and GSplat. 
    # Note: Removed the local file path as it causes build failures if volume isn't present
    .run_commands(
        "pip install flash-attn --no-build-isolation",
        "pip install gsplat==1.5.3"
    )
    # Final cleanup - removed 'pip cache purge' as it's not supported in this environment
)

# =========================================================
# 2. APP & STORAGE
# =========================================================

app = modal.App("hyworld-production")

# Volumes for persistent data
weights_vol = modal.Volume.from_name("weights-hy-world-2", create_if_missing=True)
cache_vol = modal.Volume.from_name("hyworld-cache", create_if_missing=True)

# =========================================================
# 3. GENERATOR CORE
# =========================================================

def generate_world(input_img, base_name, prompt):
    import torch
    
    # Pre-flight check
    if not torch.cuda.is_available():
        raise RuntimeError("GPU requested but not accessible by PyTorch.")

    # Performance optimizations
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Resolve paths (Mounted Volumes are in /weights)
    CODE_ROOT = "/weights/HY-World-2.0"
    HELPER_ROOT = "/weights/helpers"

    if not os.path.exists(CODE_ROOT):
        print(f"Directory {CODE_ROOT} not found. Checking root...")
        CODE_ROOT = "/HY-World-2.0" # Fallback if direct volume mount differs

    print(f"--> Loading code from: {CODE_ROOT}")
    
    # Setup python environment for the external repo
    if CODE_ROOT not in sys.path:
        sys.path.insert(0, CODE_ROOT)
    os.chdir(CODE_ROOT)

    # Lazy imports to prevent issues if the volume isn't ready during cold start
    try:
        from pipelines import HYWorld2Pipeline
        from utils.meshing import splat_to_glb
    except ImportError as e:
        raise ImportError(f"Failed to import HY-World modules. Is the code uploaded to the volume? Error: {e}")

    # Initialize Pipeline
    print("--> Initializing HYWorld2 Pipeline (Device: CUDA)")
    pipeline = HYWorld2Pipeline.from_pretrained(
        CODE_ROOT,
        llm_path=os.path.join(HELPER_ROOT, "llm"),
        vision_path=os.path.join(HELPER_ROOT, "siglip"),
        device="cuda",
        torch_dtype=torch.bfloat16
    )

    # Inference logic
    print(f"--> Generating: {prompt}")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        result = pipeline(
            image=input_img,
            prompt=prompt,
            negative_prompt="blurry, messy, low quality, photorealistic, realistic texture",
            target_resolution=1024,
            output_type="3dgs"
        )

    # Export Process
    output_dir = "/tmp/hyworld_output"
    os.makedirs(output_dir, exist_ok=True)
    glb_path = os.path.join(output_dir, f"{base_name}.glb")

    print("--> Converting Splat to GLB...")
    splat_to_glb(
        result,
        output_path=glb_path,
        simplify_ratio=0.8,
        texture_size=2048
    )

    # Cleanup memory immediately
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return glb_path

# =========================================================
# 4. QUEUE WORKER
# =========================================================

@app.function(
    image=image,
    gpu="L4", 
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

    # S3 / R2 Connection
    s3 = boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"]
    )

    print("Checking Queue...")
    response = s3.list_objects_v2(Bucket=cfg["bucket"], Prefix="queue/")

    if "Contents" not in response:
        print("No pending tasks in 'queue/'.")
        return

    for obj in response["Contents"]:
        key = obj["Key"]
        
        # Validation
        if key.endswith("/") or not key.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        print(f"--- STARTING TASK: {key} ---")
        
        try:
            # 1. Fetch
            data = s3.get_object(Bucket=cfg["bucket"], Key=key)
            img = Image.open(io.BytesIO(data["Body"].read())).convert("RGB")
            
            # 2. Metadata / Prompt extraction
            prompt = data.get("Metadata", {}).get("prompt", "comic cinematic city world")
            base_name = os.path.splitext(os.path.basename(key))[0]

            # 3. Process
            glb_file = generate_world(img, base_name, prompt)

            # 4. Upload result
            output_key = f"output/{base_name}.glb"
            print(f"Uploading to {output_key}")
            with open(glb_file, "rb") as f:
                s3.put_object(
                    Bucket=cfg["bucket"],
                    Key=output_key,
                    Body=f,
                    ContentType="model/gltf-binary"
                )

            # 5. Cleanup
            s3.delete_object(Bucket=cfg["bucket"], Key=key)
            print(f"--- FINISHED TASK: {key} ---")

        except Exception as e:
            print(f"!!! ERROR ON {key}: {e}")
            # Move to failed folder so it doesn't block the queue
            failed_key = key.replace("queue/", "failed/", 1)
            try:
                s3.copy_object(
                    Bucket=cfg["bucket"],
                    CopySource={"Bucket": cfg["bucket"], "Key": key},
                    Key=failed_key
                )
                s3.delete_object(Bucket=cfg["bucket"], Key=key)
            except:
                pass

    # Ensure cache volume is flushed
    cache_vol.commit()

# =========================================================
# 5. LOCAL ENTRYPOINT
# =========================================================

@app.local_entrypoint()
def main():
    # Recommended: Use Modal Secrets for these in the future
    config = {
        "endpoint": "https://4d91f4d3d0366568a54ffa32ffcb7bf4.r2.cloudflarestorage.com",
        "access_key": "3c33425ba6e5abbd3e63afab14dc8866",
        "secret_key": "d65f107bb61093843c6dd980c764443fdf50924a7701078b99f007d3060e25a8",
        "bucket": "video-asset-files-storage-workflow"
    }
    
    process_cloudflare_queue.remote(config)
