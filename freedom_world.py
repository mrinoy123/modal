import modal
import os
import sys
import io
import urllib.request
import uuid
import gc

# ==========================================
# 1. IMAGE (STABLE TORCH 2.4 + L4 OPTIMIZED)
# ==========================================
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")

    .env({
        "TORCH_CUDA_ARCH_LIST": "8.9", # L4 Architecture
        "CUDA_HOME": "/usr/local/cuda",
        "FORCE_CUDA": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    })

    .apt_install("git", "build-essential", "libgl1-mesa-glx", "libglib2.0-0", "wget")
    .pip_install("setuptools", "wheel")

    # HY-World 2.0 requires Torch 2.4.0 for Flash-Attention 2.8 compatibility
    .pip_install(
        "torch==2.4.0", "torchvision", "torchaudio", 
        index_url="https://download.pytorch.org/whl/cu124"
    )

    .pip_install("transformers", "accelerate", "diffusers", "boto3", "pillow", "einops", "omegaconf")

    # 🔥 INSTALLING THE 2026 WHEELS FROM YOUR VERIFIED VOLUME
    .run_commands(
        "pip install /weights/dependencies/gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl",
        "pip install /weights/dependencies/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    )
)

app = modal.App("freedom-force-world", image=image)
hy_volume = modal.Volume.from_name("weights-hy-world-2")

# ==========================================
# 2. GENERATOR PIPELINE
# ==========================================
def generate_3d_stage(input_img, base_name, prompt):
    import torch
    import gc
    import numpy as np

    assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"
    print("CUDA verified! Initializing 3D World generation...")

    torch.backends.cuda.preferred_linalg_library("default")

    # =================================================================
    # 🕵️ THE RESILIENT LINKER (HY-World 2.0 Adaptation)
    # =================================================================
    def resolve_paths():
        candidate_mounts = ["/weights", "/mnt", "/root"]
        base_path = None
        
        for mount in candidate_mounts:
            # We look for the main core folder downloaded by the migrator
            if os.path.exists(os.path.join(mount, "HY-World-2.0")):
                base_path = mount
                break
                
        if not base_path: 
            raise FileNotFoundError("Linker Failed: Missing HY-World-2.0 in volumes.")
            
        code_root = os.path.join(base_path, "HY-World-2.0")
        helper_root = os.path.join(base_path, "helpers")
        
        return code_root, helper_root

    CODE_ROOT, HELPER_ROOT = resolve_paths()

    # Add the repository to system path so we can import the pipelines natively
    sys.path.insert(0, CODE_ROOT)
    os.chdir(CODE_ROOT)

    from pipelines import HYWorld2Pipeline # Native import from the mounted repo

    # =================================================================
    # STAGE 1: MULTI-MODAL WORLD GENERATION (BFloat16 Boosted)
    # =================================================================
    print(f"Loading HY-World 2.0 Pipeline from {CODE_ROOT}...")
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)

    # Initialize the engine using the Qwen Brain and SigLIP Eyes
    pipeline = HYWorld2Pipeline.from_pretrained(
        CODE_ROOT,
        llm_path=os.path.join(HELPER_ROOT, "llm"),
        vision_path=os.path.join(HELPER_ROOT, "siglip"),
        device="cuda",
        torch_dtype=torch.bfloat16 # Saves heavy VRAM on the L4
    )

    print(f"Generating 3D Stage for prompt: {prompt[:50]}...")
    
    # The image + prompt combo ensures comic accuracy in the 360 view
    result = pipeline(
        image=input_img,
        prompt=prompt,
        negative_prompt="photorealistic, blurry, messy, organic, trees, realistic textures",
        target_resolution=1024,
        output_type="3dgs" # Output as Gaussian Splatting point cloud
    )

    # Save to the specific output folder created during migration
    output_dir = "/weights/outputs/mesh"
    ply_path = os.path.join(output_dir, f"{base_name}.ply")
    result.save(ply_path)

    print("Clearing HY-World Model from VRAM...")
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return ply_path

# ==========================================
# 3. AUTOMATION WORKER
# ==========================================
@app.function(
    volumes={"/weights": hy_volume}, 
    gpu="L4", 
    timeout=3600,
    container_idle_timeout=60, # 🛡️ Kills GPU after 1 min of idle
    concurrency_limit=1        # 🛡️ Max 1 GPU to protect credits
)
def process_cloudflare_queue(cfg: dict):
    import boto3
    from PIL import Image

    print("Connecting to Cloudflare R2...")
    s3 = boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"]
    )

    # Looking at the exact same queue folder as your object generator
    res = s3.list_objects_v2(Bucket=cfg["bucket"], Prefix="queue/")
    if "Contents" not in res:
        print("📭 Queue is empty.")
        return

    processed_count = 0

    for obj in res["Contents"]:
        key = obj["Key"]
        
        # Ensure it only processes actual images, like "city_skyline_raw.png"
        if len(key.split("/")) != 2 or not key.lower().endswith((".png",".jpg",".jpeg")):
            continue

        processed_count += 1
        print(f"\n📥 Processing World Image: {key}")
        
        data = s3.get_object(Bucket=cfg["bucket"], Key=key)
        # Using RGB since World Generation doesn't need an Alpha channel (backgrounds are solid)
        img = Image.open(io.BytesIO(data["Body"].read())).convert("RGB")

        # 🕵️ METADATA EXTRACTION
        prompt = data.get('Metadata', {}).get('prompt', "A bold comic book newsroom city, ink outlines, sunset")

        try:
            base_name = key.split("/")[-1].split(".")[0]
            ply_file = generate_3d_stage(img, base_name, prompt)

            print(f"Uploading 3D stage {ply_file} back to R2...")
            with open(ply_file, "rb") as f:
                s3.put_object(
                    Bucket=cfg["bucket"],
                    Key=f"output/{os.path.basename(ply_file)}",
                    Body=f
                )
            print("✅ Successfully uploaded! Removing original from queue...")
            s3.delete_object(Bucket=cfg["bucket"], Key=key)

        except Exception as e:
            print(f"❌ ERROR processing {key}: {e}")
            # The identical fail-safe from your factory.py
            failed_key = key.replace("queue/", "queue/failed/", 1)
            s3.copy_object(
                Bucket=cfg["bucket"],
                CopySource={'Bucket': cfg["bucket"], 'Key': key},
                Key=failed_key
            )
            s3.delete_object(Bucket=cfg["bucket"], Key=key)

    if processed_count == 0:
        print("📭 Checked R2, but found no valid images waiting in the queue/")

# ==========================================
# 4. ENTRYPOINT
# ==========================================
@app.local_entrypoint()
def main():
    process_cloudflare_queue.remote({
        "endpoint": "https://4d91f4d3d0366568a54ffa32ffcb7bf4.r2.cloudflarestorage.com",
        "access_key": "3c33425ba6e5abbd3e63afab14dc8866",
        "secret_key": "d65f107bb61093843c6dd980c764443fdf50924a7701078b99f007d3060e25a8",
        "bucket": "video-asset-files-storage-workflow"
    })
