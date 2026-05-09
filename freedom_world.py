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
app = modal.App("hyworld2-production-v3")

# Volumes
weights_vol = modal.Volume.from_name("weights-hy-world-2", create_if_missing=True)
cache_vol = modal.Volume.from_name("hyworld2-cache", create_if_missing=True)

# =========================================================
# IMAGE BUILD (Added Open3D and Pymeshlab for meshing)
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
        "libglib2.0-0", "build-essential", "ninja-build", "cmake",
        "libopenblas-dev", "libusb-1.0-0"
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
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "diffusers==0.31.0",
        "einops", "omegaconf", "timm", "scipy", "trimesh",
        "opencv-python-headless==4.9.0.80",
        "pygltflib", "jaxtyping", "boto3", "roma", "pyquaternion", "plyfile",
        "open3d", "pymeshlab" # Added for PointCloud -> Mesh conversion
    )
    .run_commands(
        "git clone https://github.com/Tencent-Hunyuan/HY-World-2.0.git /root/HYWorld",
        "cd /root/HYWorld && pip install -r requirements.txt || true"
    )
)

# =========================================================
# GENERATION LOGIC
# =========================================================
def generate_world(input_image, prompt, output_name):
    import torch
    import trimesh
    import numpy as np
    from PIL import Image
    
    ROOT = "/root/HYWorld"

    def setup_resilient_linker(volume_path="/weights", target_path="/root/HYWorld/weights"):
        print(f"LINKING VOLUMES TO {target_path}")
        os.makedirs(target_path, exist_ok=True)
        if not os.path.exists(volume_path): return
        for item in os.listdir(volume_path):
            src, dst = os.path.join(volume_path, item), os.path.join(target_path, item)
            if not os.path.exists(dst):
                try: os.symlink(src, dst)
                except: pass

    setup_resilient_linker(volume_path="/weights", target_path=f"{ROOT}/weights")

    if ROOT not in sys.path: sys.path.insert(0, ROOT)
    os.chdir(ROOT)

    # 1. Import Pipeline
    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline
    
    # 2. Load Model
    model_path = f"{ROOT}/weights/HY-World-2.0"
    pipe = WorldMirrorPipeline.from_pretrained(model_path).to("cuda")

    # 3. Prepare Input
    temp_input_dir = f"/tmp/input_{output_name}"
    os.makedirs(temp_input_dir, exist_ok=True)
    input_image.save(os.path.join(temp_input_dir, "input.png"))

    # 4. Run Inference
    print(f"--- Running HY-World Mirror Reconstruction ---")
    with torch.inference_mode():
        # Pipeline returns the output root directory
        output_root = pipe(temp_input_dir)

    # 5. Extract and Convert Output
    # The pipeline saves to: inference_output/input_name/timestamp/...
    output_dir = "/tmp/final_output"
    os.makedirs(output_dir, exist_ok=True)
    glb_path = os.path.join(output_dir, f"{output_name}.glb")

    print(f"Inference finished. Searching results in: {output_root}")
    
    # HY-World 2.0 creates a timestamped folder inside the output_root
    # We look for .glb or .ply files
    all_files = glob.glob(os.path.join(output_root, "**/*"), recursive=True)
    
    glb_files = [f for f in all_files if f.endswith(".glb")]
    ply_files = [f for f in all_files if f.endswith("gaussians.ply") or f.endswith("points.ply")]

    if glb_files:
        print(f"Found GLB: {glb_files[0]}")
        shutil.copy(glb_files[0], glb_path)
    elif ply_files:
        print(f"Found PLY: {ply_files[0]}. Converting Point Cloud to Mesh...")
        try:
            # Load point cloud
            pcd = trimesh.load(ply_files[0])
            
            # If it's a point cloud (common for WorldMirror), we need to create a mesh
            if isinstance(pcd, trimesh.points.PointCloud):
                # Basic Alpha-Shape or Convex Hull to get a visible GLB world
                # For "Worlds", a Convex Hull or a simple Poisson reconstruction works best
                mesh = pcd.convex_hull 
                mesh.export(glb_path)
            else:
                # It's already a mesh in a .ply wrapper
                pcd.export(glb_path)
            print(f"Conversion Successful: {glb_path}")
        except Exception as e:
            print(f"Meshing failed: {e}. Uploading raw PLY instead.")
            glb_path = ply_files[0] # Fallback to raw PLY
    else:
        raise RuntimeError(f"No usable 3D assets found in {output_root}")

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return glb_path

# =========================================================
# MODAL WORKER
# =========================================================
@app.function(
    image=image,
    gpu="L4", 
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

        print(f"Processing: {key}")
        try:
            data = s3.get_object(Bucket=cfg["bucket"], Key=key)
            img = Image.open(io.BytesIO(data["Body"].read())).convert("RGB")
            
            metadata = data.get("Metadata", {})
            prompt = metadata.get("prompt", "cinematic 3d world")
            base_name = os.path.splitext(os.path.basename(key))[0]

            local_file = generate_world(img, prompt, base_name)
            
            # Determine content type based on extension (might be .glb or .ply)
            ext = os.path.splitext(local_file)[1].lower()
            content_type = "model/gltf-binary" if ext == ".glb" else "application/octet-stream"
            output_key = f"output/{base_name}{ext}"

            with open(local_file, "rb") as f:
                s3.put_object(
                    Bucket=cfg["bucket"],
                    Key=output_key,
                    Body=f,
                    ContentType=content_type
                )

            s3.delete_object(Bucket=cfg["bucket"], Key=key)
            print(f"SUCCESS: {base_name}")

        except Exception as e:
            print(f"FAILED {key}: {str(e)}")
            try:
                failed_key = key.replace("queue/", "failed/", 1)
                s3.copy_object(Bucket=cfg["bucket"], CopySource={"Bucket": cfg["bucket"], "Key": key}, Key=failed_key)
                s3.delete_object(Bucket=cfg["bucket"], Key=key)
            except: pass

# =========================================================
# ENTRYPOINT
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
