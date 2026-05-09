import modal
import os
import sys
import io
import gc

# =========================================================
# APP CONFIGURATION
# =========================================================
app = modal.App("hy-world-v2-production")

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
        "XFORMERS_FORCE_STAGES": "1"
    })
    .apt_install(
        "git", "git-lfs", "wget", "ffmpeg", "libgl1-mesa-glx", 
        "libglib2.0-0", "build-essential", "ninja-build"
    )
    .pip_install(
        "torch==2.4.0", "torchvision==0.19.0", 
        index_url="https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "diffusers==0.31.0",
        "einops",
        "omegaconf",
        "timm",
        "scipy",
        "opencv-python-headless",
        "trimesh",
        "pygltflib",
        "gsplat==1.5.3",
        "jaxtyping",
        "boto3",
        "roma",
        "pyquaternion"
    )
    .run_commands(
        "git clone https://github.com/Tencent-Hunyuan/HY-World-2.0.git /root/HYWorld",
        "cd /root/HYWorld && pip install -e ." 
    )
)

# =========================================================
# GENERATION LOGIC
# =========================================================
def generate_world(input_image, prompt, output_name):
    import torch
    # Import the specific v2.0 pipeline
    # The repo uses 'hyworld' as the base package
    from hyworld.pipelines.pipeline_world import HunyuanWorldPipeline
    
    ROOT = "/root/HYWorld"
    WEIGHTS_DIR = "/weights" # Where the Modal volume is mounted

    # Ensure the code can find its own modules
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.chdir(ROOT)

    print(f"--- Loading Hunyuan World 2.0 ---")
    
    # Load Pipeline
    # NOTE: Ensure your Volume contains the folders: 'ckpts', 't2v_model', etc.
    # We point to the directory containing the config files.
    pipe = HunyuanWorldPipeline.from_pretrained(
        WEIGHTS_DIR, 
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    # Enable memory optimizations
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload() # Use if L4 (24GB) is tight, otherwise just .to("cuda")

    print(f"--- Generating: {prompt} ---")
    
    with torch.inference_mode():
        # HY-World 2.0 __call__ expects image and prompt
        output = pipe(
            image=input_image,
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=5.0,
            generator=torch.Generator("cuda").manual_seed(42)
        )

    # Output directory
    output_dir = "/tmp/output"
    os.makedirs(output_dir, exist_ok=True)
    glb_path = os.path.join(output_dir, f"{output_name}.glb")

    # HY-World returns a result object with an export method or mesh attribute
    # Checking for common return types in Tencent's 3D repos
    if hasattr(output, "export_glb"):
        output.export_glb(glb_path)
    elif isinstance(output, dict) and "mesh" in output:
        output["mesh"].export(glb_path)
    else:
        # Fallback for standard trimesh/scene objects
        output[0].export(glb_path)

    # Memory Cleanup
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
    volumes={
        "/weights": weights_vol,
        "/cache": cache_vol
    }
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
    
    if "Contents" not in response:
        print("Queue Empty.")
        return

    for obj in response["Contents"]:
        key = obj["Key"]
        if key == "queue/" or not key.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        print(f"Processing: {key}")
        try:
            # Download image
            data = s3.get_object(Bucket=cfg["bucket"], Key=key)
            img = Image.open(io.BytesIO(data["Body"].read())).convert("RGB")
            
            # Extract Metadata
            metadata = data.get("Metadata", {})
            prompt = metadata.get("prompt", "a beautiful highly detailed 3d world")
            base_name = os.path.splitext(os.path.basename(key))[0]

            # Generate
            glb_local_path = generate_world(img, prompt, base_name)

            # Upload result
            output_key = f"output/{base_name}.glb"
            with open(glb_local_path, "rb") as f:
                s3.put_object(
                    Bucket=cfg["bucket"],
                    Key=output_key,
                    Body=f,
                    ContentType="model/gltf-binary"
                )

            # Cleanup S3 queue
            s3.delete_object(Bucket=cfg["bucket"], Key=key)
            print(f"Successfully processed {base_name}")

        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            # Move to failed folder
            failed_key = key.replace("queue/", "failed/", 1)
            s3.copy_object(Bucket=cfg["bucket"], CopySource={"Bucket": cfg["bucket"], "Key": key}, Key=failed_key)
            s3.delete_object(Bucket=cfg["bucket"], Key=key)

# =========================================================
# LOCAL ENTRYPOINT
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
