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
app = modal.App("hyworld2-production-v5")

# Volumes
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
        "open3d==0.18.0"
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
    import importlib
    
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

    # 1. Import the GENERATION Pipeline (Not Mirror Reconstruction)
    pipeline_loaded = False
    pipeline_class = None
    
    # We dynamically check for the World Generation pipeline (hallucination)
    candidate_imports = [
        ("hyworld2.pipeline", "HunyuanWorldPipeline"),
        ("hyworld2.world.pipeline", "HunyuanWorldPipeline"),
        ("hyworld2.inference", "HunyuanWorldPipeline"),
        ("hyworld.pipelines", "HYWorld2Pipeline")
    ]

    for module_name, class_name in candidate_imports:
        try:
            module = importlib.import_module(module_name)
            pipeline_class = getattr(module, class_name)
            print(f"SUCCESS: Loaded GENERATION Pipeline from {module_name}")
            pipeline_loaded = True
            break
        except Exception as e:
            continue

    if not pipeline_loaded:
        raise RuntimeError("Could not locate the World Generation pipeline in the repository.")
    
    # 2. Load Model
    model_path = f"{ROOT}/weights/HY-World-2.0"
    print(f"Loading Model from {model_path}...")
    pipe = pipeline_class.from_pretrained(model_path)

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    # 3. Run Inference (Image to 360 Panorama to 3D World)
    print(f"--- Hallucinating 3D World from Image and Prompt ---")
    print(f"Prompt: {prompt}")
    
    with torch.inference_mode():
        # The generation pipeline takes the image as a seed, and the prompt to guide the hallucination
        output = pipe(
            image=input_image,
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=5.0,
            generator=torch.Generator("cuda").manual_seed(42)
        )

    # 4. Locate Result
    output_dir = "/tmp/final_output"
    os.makedirs(output_dir, exist_ok=True)
    final_glb_path = os.path.join(output_dir, f"{output_name}.glb")

    # If the pipeline returns a directory path (like WorldMirror does)
    if isinstance(output, str) and os.path.isdir(output):
        print(f"Searching for 3D assets in: {output}")
        all_files = glob.glob(os.path.join(output, "**/*"), recursive=True)
        glb_files = [f for f in all_files if f.endswith(".glb")]
        ply_files = [f for f in all_files if f.endswith("gaussians.ply") or f.endswith("points.ply")]

        if glb_files:
            shutil.copy(glb_files[0], final_glb_path)
        elif ply_files:
            print("Safely copying raw PLY file to preserve splat properties...")
            final_ply_path = final_glb_path.replace(".glb", ".ply")
            shutil.copy(ply_files[0], final_ply_path)
            final_glb_path = final_ply_path 
        else:
            raise RuntimeError("No 3D file (.glb or .ply) found in output directory.")

    # If the pipeline returns an object with an export method (Standard Generation Pipeline)
    elif hasattr(output, "export_glb"):
        output.export_glb(final_glb_path)
    elif isinstance(output, dict) and "mesh" in output:
        output["mesh"].export(final_glb_path)
    else:
        try:
            output[0].export(final_glb_path)
        except:
            output.export(final_glb_path)

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return final_glb_path

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

        print(f"Processing File: {key}")
        try:
            data = s3.get_object(Bucket=cfg["bucket"], Key=key)
            img = Image.open(io.BytesIO(data["Body"].read())).convert("RGB")
            
            prompt = data.get("Metadata", {}).get("prompt", "a sprawling, dense cyberpunk city at night, highly detailed, cinematic lighting")
            base_name = os.path.splitext(os.path.basename(key))[0]

            local_file = generate_world(img, prompt, base_name)
            
            ext = os.path.splitext(local_file)[1]
            output_key = f"output/{base_name}{ext}"
            content_type = "model/gltf-binary" if ext == ".glb" else "application/octet-stream"

            with open(local_file, "rb") as f:
                s3.put_object(
                    Bucket=cfg["bucket"],
                    Key=output_key,
                    Body=f,
                    ContentType=content_type
                )

            s3.delete_object(Bucket=cfg["bucket"], Key=key)
            print(f"SUCCESS: {base_name}{ext}")

        except Exception as e:
            print(f"ERROR processing {key}: {str(e)}")
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
