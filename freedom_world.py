import modal
import os
import sys
import io
import gc
import shutil

# =========================================================
# APP CONFIGURATION
# =========================================================
app = modal.App("hyworld2-production-fixed")

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
        "MAX_JOBS": "4",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    })
    .apt_install(
        "git", "git-lfs", "wget", "ffmpeg", "libgl1-mesa-glx", 
        "libglib2.0-0", "build-essential", "ninja-build", "cmake"
    )
    .pip_install(
        "pip==25.0.1",
        "setuptools==70.0.0",
        "packaging",
        "wheel",
        "ninja"
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
        "pygltflib", "jaxtyping", "boto3", "roma", "pyquaternion", "plyfile"
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
    from PIL import Image
    
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

    setup_resilient_linker(volume_path="/weights", target_path=f"{ROOT}/weights")

    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.chdir(ROOT)

    # =====================================================
    # PIPELINE IMPORT
    # =====================================================
    import importlib
    candidate_imports = [
        ("hyworld2.worldrecon.pipeline", "WorldMirrorPipeline"),
        ("hyworld2.pipelines.pipeline_world", "HunyuanWorldPipeline"),
    ]

    pipeline_class = None
    class_name = None
    for module_name, c_name in candidate_imports:
        try:
            module = importlib.import_module(module_name)
            pipeline_class = getattr(module, c_name)
            class_name = c_name
            print(f"SUCCESS IMPORT: {module_name}.{class_name}")
            break
        except Exception as e:
            print(f"Skipping import {module_name}: {e}")

    if not pipeline_class:
        raise RuntimeError("Failed to import any HY-World pipeline.")

    # =====================================================
    # MODEL LOADING
    # =====================================================
    # The logs show weights are in /weights/HY-World-2.0
    model_path = f"{ROOT}/weights/HY-World-2.0"
    
    print(f"--- Loading Model from: {model_path} ---")
    
    # WorldMirror 2.0 from_pretrained typically takes the root folder
    pipe = pipeline_class.from_pretrained(model_path)
    
    if hasattr(pipe, "to"):
        pipe = pipe.to("cuda")

    # =====================================================
    # INFERENCE
    # =====================================================
    print(f"--- Generating World: {output_name} ---")
    
    with torch.inference_mode():
        if class_name == "WorldMirrorPipeline":
            # WorldMirror is Image-to-World Reconstruction. 
            # It does NOT accept 'prompt'. It accepts image path or directory.
            temp_input_dir = f"/tmp/input_{output_name}"
            os.makedirs(temp_input_dir, exist_ok=True)
            img_path = os.path.join(temp_input_dir, "input.png")
            input_image.save(img_path)
            
            # The pipeline usually returns the output directory path
            output = pipe(temp_input_dir)
        else:
            # Generic generator pipeline
            output = pipe(image=input_image, prompt=prompt)

    # =====================================================
    # EXPORT & RETRIEVAL
    # =====================================================
    output_dir = "/tmp/final_glb"
    os.makedirs(output_dir, exist_ok=True)
    target_glb_path = os.path.join(output_dir, f"{output_name}.glb")

    # If the pipeline returned a path to a directory (standard for HY-World WorldMirror)
    if isinstance(output, str) and os.path.isdir(output):
        print(f"Searching for GLB in output dir: {output}")
        found = False
        # WorldMirror often creates subfolders like 'mesh' or 'reconstruction'
        for root, dirs, files in os.walk(output):
            for f in files:
                if f.endswith(".glb"):
                    shutil.copy(os.path.join(root, f), target_glb_path)
                    found = True
                    break
            if found: break
            
        if not found:
            raise RuntimeError(f"Pipeline finished but no .glb was found in {output}")
    
    # If the pipeline returned an object with export capabilities
    elif hasattr(output, "export_glb"):
        output.export_glb(target_glb_path)
    elif isinstance(output, dict) and "mesh" in output:
        output["mesh"].export(target_glb_path)
    else:
        # Fallback for trimesh/list objects
        try:
            output[0].export(target_glb_path)
        except:
            output.export(target_glb_path)

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return target_glb_path

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

        print(f"Processing File: {key}")
        try:
            data = s3.get_object(Bucket=cfg["bucket"], Key=key)
            img = Image.open(io.BytesIO(data["Body"].read())).convert("RGB")
            
            metadata = data.get("Metadata", {})
            prompt = metadata.get("prompt", "cinematic 3d world")
            base_name = os.path.splitext(os.path.basename(key))[0]

            local_glb = generate_world(img, prompt, base_name)

            output_key = f"output/{base_name}.glb"
            with open(local_glb, "rb") as f:
                s3.put_object(
                    Bucket=cfg["bucket"],
                    Key=output_key,
                    Body=f,
                    ContentType="model/gltf-binary"
                )

            s3.delete_object(Bucket=cfg["bucket"], Key=key)
            print(f"SUCCESS: {base_name}")

        except Exception as e:
            print(f"FAILED {key}: {str(e)}")
            failed_key = key.replace("queue/", "failed/", 1)
            try:
                s3.copy_object(Bucket=cfg["bucket"], CopySource={"Bucket": cfg["bucket"], "Key": key}, Key=failed_key)
                s3.delete_object(Bucket=cfg["bucket"], Key=key)
            except:
                pass

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
