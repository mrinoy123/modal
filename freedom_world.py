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
        "pyquaternion",
        "plyfile" # Added to prevent missing dependency errors for 3D outputs
    )
    .run_commands(
        "rm -rf /root/HYWorld",
        "git clone https://github.com/Tencent-Hunyuan/HY-World-2.0.git /root/HYWorld",
        # FIX: Replaced `pip install -e .` with requirements install
        "cd /root/HYWorld && pip install -r requirements.txt || true" 
    )
)

# =========================================================
# GENERATION LOGIC
# =========================================================
def generate_world(input_image, prompt, output_name):
    import torch
    
    ROOT = "/root/HYWorld"

    # =====================================================
    # RESILIENT LINKER
    # =====================================================
    def setup_resilient_linker(volume_path="/weights", target_path="/root/HYWorld/weights"):
        print(f"EXECUTING RESILIENT LINKER FROM {volume_path} TO {target_path}")
        if not os.path.exists(volume_path):
            print(f"WARNING: Volume path {volume_path} not found.")
            return

        os.makedirs(target_path, exist_ok=True)
        for item in os.listdir(volume_path):
            src = os.path.join(volume_path, item)
            dst = os.path.join(target_path, item)
            if not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                    print(f"Resilient Linker: Symlinked {src} -> {dst}")
                except Exception as e:
                    print(f"Resilient Linker Error for {item}: {e}")

    # Link the files from the mounted volume to the local model folder
    setup_resilient_linker(volume_path="/weights", target_path=f"{ROOT}/weights")

    # Ensure the code can find its own modules
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.chdir(ROOT)

    print(f"--- Loading Hunyuan World 2.0 ---")

    # =====================================================
    # ROBUST PIPELINE IMPORT
    # =====================================================
    pipeline_loaded = False
    import importlib

    # Tencent changes import paths frequently. This tries the known active paths.
    candidate_imports = [
        ("hyworld.pipelines.pipeline_world", "HunyuanWorldPipeline"),
        ("hyworld2.worldrecon.pipeline", "WorldMirrorPipeline"),
        ("hyworld2.inference", "HunyuanWorldPipeline"),
        ("pipelines", "HYWorld2Pipeline"),
    ]

    for module_name, class_name in candidate_imports:
        try:
            module = importlib.import_module(module_name)
            pipeline_class = getattr(module, class_name)
            print(f"SUCCESS IMPORT: {module_name}.{class_name}")
            pipeline_loaded = True
            break
        except Exception as e:
            print(f"FAILED IMPORT: {module_name} -> {e}")

    if not pipeline_loaded:
        raise RuntimeError("Could not locate HYWorld pipeline.")
    
    # Load Pipeline
    pipe = pipeline_class.from_pretrained(
        f"{ROOT}/weights", # Points to the resiliently linked directory
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    # Enable memory optimizations
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload() 

    print(f"--- Generating: {prompt} ---")
    
    with torch.inference_mode():
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

    # Export Logic
    if hasattr(output, "export_glb"):
        output.export_glb(glb_path)
    elif isinstance(output, dict) and "mesh" in output:
        output["mesh"].export(glb_path)
    else:
        # Fallback for standard trimesh/scene objects
        try:
            output[0].export(glb_path)
        except:
            output.export(glb_path)

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
            # Move to failed folder to avoid infinite loop
            failed_key = key.replace("queue/", "failed/", 1)
            try:
                s3.copy_object(Bucket=cfg["bucket"], CopySource={"Bucket": cfg["bucket"], "Key": key}, Key=failed_key)
                s3.delete_object(Bucket=cfg["bucket"], Key=key)
            except Exception as move_e:
                print(f"Failed to move object to failed folder: {move_e}")

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
