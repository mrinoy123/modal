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
# IMAGE BUILD (Fixed Module Dependencies)
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
        "rm -rf /root/HYWorld",
        "git clone https://github.com/Tencent-Hunyuan/HY-World-2.0.git /root/HYWorld",
        "cd /root/HYWorld && pip install -r requirements.txt || true"
    )
)

# =========================================================
# GENERATION ENGINE (L4 Optimized with Smart Import)
# =========================================================
def generate_hallucinated_city(input_image, prompt, output_name):
    import torch
    from PIL import Image
    import importlib
    
    ROOT = "/root/HYWorld"
    
    # 1. Resilient Linker
    def setup_weights(volume_path="/weights", target_path="/root/HYWorld/weights"):
        print(f"Mapping Weights to {target_path}")
        os.makedirs(target_path, exist_ok=True)
        if not os.path.exists(volume_path): return
        for item in os.listdir(volume_path):
            src, dst = os.path.join(volume_path, item), os.path.join(target_path, item)
            if not os.path.exists(dst):
                try: os.symlink(src, dst)
                except: pass

    setup_weights(volume_path="/weights", target_path=f"{ROOT}/weights")
    
    # Add ROOT and its subfolders to path to fix "ModuleNotFound"
    if ROOT not in sys.path: sys.path.insert(0, ROOT)
    
    # Add potential package locations (Tencent changes these between hyworld and hyworld2)
    for folder in os.listdir(ROOT):
        folder_path = os.path.join(ROOT, folder)
        if os.path.isdir(folder_path) and folder.startswith("hyworld"):
            if folder_path not in sys.path: sys.path.insert(0, folder_path)

    os.chdir(ROOT)

    # 2. Smart Pipeline Import
    # This tries every possible Tencent import path for the Generative World pipeline
    pipeline_class = None
    import_errors = []
    
    candidate_imports = [
        ("hyworld.pipelines.pipeline_world", "HunyuanWorldPipeline"),
        ("hyworld2.pipelines.pipeline_world", "HunyuanWorldPipeline"),
        ("hyworld.inference", "HunyuanWorldPipeline"),
        ("hyworld2.inference", "HunyuanWorldPipeline"),
        ("pipelines.pipeline_world", "HunyuanWorldPipeline")
    ]

    for mod_path, cls_name in candidate_imports:
        try:
            module = importlib.import_module(mod_path)
            pipeline_class = getattr(module, cls_name)
            print(f"SUCCESS: Loaded pipeline from {mod_path}")
            break
        except Exception as e:
            import_errors.append(f"{mod_path}: {str(e)}")

    if not pipeline_class:
        raise ImportError(f"Could not find HunyuanWorldPipeline. Errors: {'; '.join(import_errors)}")

    # 3. Model Loading (VRAM Optimized for L4)
    model_path = f"{ROOT}/weights/HY-World-2.0"
    print(f"Loading Hallucination Engine (L4 24GB Optimization)...")
    
    # Use float16 to fit the 50GB engine into the L4
    pipe = pipeline_class.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    # L4 Memory Management
    if torch.cuda.is_available():
        # Sequential offloading is required for 50GB models on 24GB VRAM
        pipe.enable_model_cpu_offload() 
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

    # 4. Run Hallucination
    print(f"--- Hallucinating 360-Degree City: {prompt} ---")
    with torch.inference_mode():
        output = pipe(
            image=input_image,
            prompt=prompt,
            num_inference_steps=30, # Reduced steps for speed on L4
            guidance_scale=7.5,
            height=512,
            width=512
        )

    # 5. Extract Result
    output_dir = "/tmp/final_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Locate assets (Recursive search for any produced 3D files)
    search_path = output if isinstance(output, str) and os.path.exists(output) else ROOT
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
        # Check standard output location
        fallback_output = os.path.join(ROOT, "outputs")
        if os.path.exists(fallback_output):
            fb_files = glob.glob(os.path.join(fallback_output, "**/*.glb"), recursive=True)
            if fb_files:
                final_path = os.path.join(output_dir, f"{output_name}.glb")
                shutil.copy(fb_files[0], final_path)
                return final_path
        raise RuntimeError(f"No 3D asset found. Scanned: {len(all_files)} files.")

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return final_path

# =========================================================
# MODAL WORKER & R2 LOGIC
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
            
            metadata = data.get("Metadata", {})
            prompt = metadata.get("prompt", "a cinematic photorealistic 3d city, 360 environment")
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
            print(f"SUCCESS: Generated {output_key}")

        except Exception as e:
            print(f"ERROR processing {key}: {str(e)}")
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
