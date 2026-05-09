import modal
import os
import sys
import io
import urllib.request
import uuid
import gc

# ==========================================
# 1. IMAGE (STABLE TORCH 2.4 + L4 OPTIMIZED + MESH TOOLS)
# ==========================================
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")

    .env({
        "TORCH_CUDA_ARCH_LIST": "8.9", # L4 Architecture
        "CUDA_HOME": "/usr/local/cuda",
        "FORCE_CUDA": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    })

    # 🏗️ ADDED libopenblas-dev for Open3D mesh mathematics
    .apt_install("git", "build-essential", "libgl1-mesa-glx", "libglib2.0-0", "wget", "libopenblas-dev")
    .pip_install("setuptools", "wheel")

    # HY-World 2.0 requires Torch 2.4.0 for Flash-Attention 2.8 compatibility
    .pip_install(
        "torch==2.4.0", "torchvision", "torchaudio", 
        index_url="https://download.pytorch.org/whl/cu124"
    )

    # 🏗️ ADDED open3d and pymeshlab to sculpt the GLB mesh
    .pip_install(
        "transformers", "accelerate", "diffusers", "boto3", "pillow", 
        "einops", "omegaconf", "open3d", "pymeshlab"
    )

    # 🔥 INSTALLING THE 2026 WHEELS FROM YOUR VERIFIED VOLUME
    .run_commands(
        "pip install /weights/dependencies/gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl",
        "pip install /weights/dependencies/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    )
)

app = modal.App("freedom-force-world", image=image)
hy_volume = modal.Volume.from_name("weights-hy-world-2")

# ==========================================
# 2. GENERATOR PIPELINE (SPLAT TO MESH CONVERTER)
# ==========================================
def generate_3d_stage(input_img, base_name, prompt):
    import torch
    import gc
    import numpy as np

    assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"
    print("CUDA verified! Initializing 3D World to GLB generation...")

    torch.backends.cuda.preferred_linalg_library("default")

    # =================================================================
    # 🕵️ THE RESILIENT LINKER (HY-World 2.0 Adaptation)
    # =================================================================
    def resolve_paths():
        candidate_mounts = ["/weights", "/mnt", "/root"]
        base_path = None
        
        for mount in candidate_mounts:
            if os.path.exists(os.path.join(mount, "HY-World-2.0")):
                base_path = mount
                break
                
        if not base_path: 
            raise FileNotFoundError("Linker Failed: Missing HY-World-2.0 in volumes.")
            
        code_root = os.path.join(base_path, "HY-World-2.0")
        helper_root = os.path.join(base_path, "helpers")
        
        return code_root, helper_root

    CODE_ROOT, HELPER_ROOT = resolve_paths()

    # Add the repository to system path
    sys.path.insert(0, CODE_ROOT)
    os.chdir(CODE_ROOT)

    from pipelines import HYWorld2Pipeline 
    # 🪄 IMPORT THE MESH CONVERTER
    from utils.meshing import splat_to_glb 

    # =================================================================
    # STAGE 1: MULTI-MODAL WORLD GENERATION 
    # =================================================================
    print(f"Loading HY-World 2.0 Pipeline from {CODE_ROOT}...")
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)

    pipeline = HYWorld2Pipeline.from_pretrained(
        CODE_ROOT,
        llm_path=os.path.join(HELPER_ROOT, "llm"),
        vision_path=os.path.join(HELPER_ROOT, "siglip"),
        device="cuda",
        torch_dtype=torch.bfloat16
    )

    print(f"🚀 Step 1: Generating high-fidelity 3D Splat for: {prompt[:50]}...")
    
    result = pipeline(
        image=input_img,
        prompt=prompt,
        negative_prompt="photorealistic, blurry, messy, organic, trees, realistic textures",
        target_resolution=1024,
        output_type="3dgs" 
    )

    # =================================================================
    # STAGE 2: SHRINK-WRAP CONVERSION (SPLAT -> GLB)
    # =================================================================
    output_dir = "/weights/outputs/mesh"
    glb_path = os.path.join(output_dir, f"{base_name}.glb")
    
    print(f"🪄 Step 2: Converting Splat into a destruction-ready .glb Mesh...")
    splat_to_glb(
        result, 
        output_path=glb_path,
        simplify_ratio=0.8,  # Keeps it light for Blender physics
        texture_size=2048    # High quality comic textures
    )

    print(f"✅ Success: 3D Mesh created at {glb_path}")

    # Clean up VRAM
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return glb_path

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

    res = s3.list_objects_v2(Bucket=cfg["bucket"], Prefix="queue/")
    if "Contents" not in res:
        print("📭 Queue is empty.")
        return

    processed_count = 0

    for obj in res["Contents"]:
        key = obj["Key"]
        
        if len(key.split("/")) != 2 or not key.lower().endswith((".png",".jpg",".jpeg")):
            continue

        processed_count += 1
        print(f"\n📥 Processing World Image: {key}")
        
        data = s3.get_object(Bucket=cfg["bucket"], Key=key)
        img = Image.open(io.BytesIO(data["Body"].read())).convert("RGB")

        # 🕵️ METADATA EXTRACTION
        prompt = data.get('Metadata', {}).get('prompt', "A bold comic book newsroom city, ink outlines, sunset")

        try:
            base_name = key.split("/")[-1].split(".")[0]
            # Gets the .glb file instead of .ply
            glb_file = generate_3d_stage(img, base_name, prompt)

            print(f"Uploading 3D Mesh {glb_file} back to R2...")
            with open(glb_file, "rb") as f:
                s3.put_object(
                    Bucket=cfg["bucket"],
                    Key=f"output/{os.path.basename(glb_file)}",
                    Body=f
                )
            print("✅ Successfully uploaded! Removing original from queue...")
            s3.delete_object(Bucket=cfg["bucket"], Key=key)

        except Exception as e:
            print(f"❌ ERROR processing {key}: {e}")
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
