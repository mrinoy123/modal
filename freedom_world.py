import modal
import os
import sys
import io
import gc

# =========================================================
# MODAL APP & STORAGE
# =========================================================

app = modal.App("hyworld-2-0-production")

weights_vol = modal.Volume.from_name("weights-hy-world-2", create_if_missing=True)
cache_vol = modal.Volume.from_name("hyworld-cache", create_if_missing=True)

# =========================================================
# IMAGE DEFINITION (FIXED VERSION MISMATCH)
# =========================================================

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10"
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", ""),
        "TORCH_CUDA_ARCH_LIST": "8.9", # Optimized for L4/A10G
        "FORCE_CUDA": "1",
        "MAX_JOBS": "4",
        "PIP_NO_CACHE_DIR": "1",
        "PYTHONUNBUFFERED": "1",
    })
    .apt_install(
        "git", "wget", "ffmpeg", "build-essential", "clang", "cmake", 
        "ninja-build", "pkg-config", "libgl1-mesa-glx", "libglib2.0-0", 
        "libsm6", "libxext6", "libxrender1", "libgomp1", "libopenblas-dev"
    )
    .pip_install("setuptools", "wheel", "packaging", "ninja")
    # 1. INSTALL STABLE TORCH FOR CUDA 12.4 FIRST
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        index_url="https://download.pytorch.org/whl/cu124"
    )
    # 2. INSTALL PRE-COMPILED FLASH-ATTN (Bypasses the "HTTP 404" and "CUDA mismatch" error)
    .run_commands(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu124torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    )
    # 3. INSTALL PRE-COMPILED GSPLAT
    .run_commands(
        "pip install gsplat==1.4.0 --index-url https://docs.gsplat.ai/whl/pt24cu124"
    )
    # 4. INSTALL REMAINING AI TOOLS
    .pip_install(
        "numpy==1.26.4", "scipy", "pandas", "tqdm", "pillow", "opencv-python",
        "imageio", "einops", "sentencepiece", "omegaconf", "safetensors",
        "accelerate==1.1.1", "transformers==4.46.3", "diffusers==0.31.0",
        "huggingface_hub", "timm", "rich", "jaxtyping", "open3d==0.18.0",
        "trimesh", "pygltflib", "pymeshlab==2022.2.post3", "boto3==1.34.131"
    )
    # 5. CLONE REPO
    .run_commands(
        "git clone https://github.com/Tencent-Hunyuan/HY-World-2.0.git /root/HunyuanWorld || true"
    )
)

# =========================================================
# GENERATOR LOGIC
# =========================================================

def generate_world(input_img, base_name, prompt):
    import torch

    # FIRE LINKER: Map volume weights to where the repo expects them
    def setup_resilient_linker(volume_path="/weights"):
        target_path = "/root/HunyuanWorld/weights"
        if os.path.exists(volume_path):
            os.makedirs(target_path, exist_ok=True)
            for item in os.listdir(volume_path):
                src = os.path.join(volume_path, item)
                dst = os.path.join(target_path, item)
                if not os.path.exists(dst):
                    try:
                        os.symlink(src, dst)
                        print(f"Linked: {item}")
                    except: pass

    setup_resilient_linker()

    if not torch.cuda.is_available():
        raise RuntimeError("GPU hardware not found.")

    # REPO SETUP
    CODE_ROOT = "/root/HunyuanWorld"
    if CODE_ROOT not in sys.path:
        sys.path.insert(0, CODE_ROOT)
    os.chdir(CODE_ROOT)

    # DYNAMIC PIPELINE LOADING
    try:
        # HunyuanWorld 2.0 structure
        from hyworld.pipelines import HYWorldPipeline
        print("Loaded HYWorldPipeline (2.0)")
    except ImportError:
        try:
            from hyworld2.worldrecon.pipeline import WorldMirrorPipeline as HYWorldPipeline
            print("Loaded WorldMirrorPipeline (2.0 Fallback)")
        except ImportError:
            raise Exception("Could not locate HY-World-2.0 Pipeline classes. Check your repo structure.")

    from hyworld.utils.meshing import splat_to_glb

    # LOAD MODEL
    print("Loading Weights...")
    pipeline = HYWorldPipeline.from_pretrained(
        CODE_ROOT, # Assumes weights are linked into /weights subfolder of repo
        device="cuda",
        torch_dtype=torch.bfloat16
    )

    # INFERENCE
    print(f"Generating World: {prompt}")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        result = pipeline(
            image=input_img,
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted",
            target_resolution=1024,
            output_type="3dgs"
        )

    # EXPORT
    output_dir = "/tmp/hyworld_output"
    os.makedirs(output_dir, exist_ok=True)
    glb_path = os.path.join(output_dir, f"{base_name}.glb")

    print("Converting Splat to GLB...")
    splat_to_glb(result, output_path=glb_path, simplify_ratio=0.8, texture_size=2048)

    # MEMORY CLEANUP
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return glb_path

# =========================================================
# WORKER
# =========================================================

@app.function(
    image=image,
    gpu="L4",
    timeout=3600,
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
    if "Contents" not in response:
        print("Queue Empty")
        return

    for obj in response["Contents"]:
        key = obj["Key"]
        if key.endswith("/") or not key.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        print(f"--- Processing: {key} ---")
        try:
            data = s3.get_object(Bucket=cfg["bucket"], Key=key)
            img = Image.open(io.BytesIO(data["Body"].read())).convert("RGB")
            
            prompt = data.get("Metadata", {}).get("prompt", "cinematic nature world")
            base_name = os.path.splitext(os.path.basename(key))[0]

            glb_path = generate_world(img, base_name, prompt)

            # Upload Result
            with open(glb_path, "rb") as f:
                s3.put_object(
                    Bucket=cfg["bucket"],
                    Key=f"output/{base_name}.glb",
                    Body=f,
                    ContentType="model/gltf-binary"
                )

            s3.delete_object(Bucket=cfg["bucket"], Key=key)
            print(f"--- Success: {key} ---")

        except Exception as e:
            print(f"Error on {key}: {e}")
            # Optional: move to failed/ folder
            s3.copy_object(
                Bucket=cfg["bucket"],
                CopySource={'Bucket': cfg["bucket"], 'Key': key},
                Key=key.replace("queue/", "failed/", 1)
            )
            s3.delete_object(Bucket=cfg["bucket"], Key=key)

    cache_vol.commit()

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
