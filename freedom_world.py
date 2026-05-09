import modal
import os
import sys
import io
import gc

# =========================================================
# APP
# =========================================================

app = modal.App("hyworld-2-production")

# =========================================================
# VOLUMES
# =========================================================

weights_vol = modal.Volume.from_name(
    "weights-hy-world-2",
    create_if_missing=True
)

cache_vol = modal.Volume.from_name(
    "hyworld-cache",
    create_if_missing=True
)

# =========================================================
# IMAGE
# =========================================================

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10"
    )

    # -----------------------------------------------------
    # ENV
    # -----------------------------------------------------
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "MAX_JOBS": "4",
        "PYTHONUNBUFFERED": "1",
        "PIP_NO_CACHE_DIR": "1",
        "HF_HOME": "/cache/huggingface",
        "TRANSFORMERS_CACHE": "/cache/huggingface",
    })

    # -----------------------------------------------------
    # SYSTEM
    # -----------------------------------------------------
    .apt_install(
        "git",
        "git-lfs",
        "wget",
        "ffmpeg",
        "build-essential",
        "clang",
        "cmake",
        "ninja-build",
        "pkg-config",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgomp1",
        "libopenblas-dev"
    )

    # -----------------------------------------------------
    # PYTHON BUILD TOOLS
    # -----------------------------------------------------
    .pip_install(
        "setuptools==69.5.1",
        "wheel",
        "packaging",
        "ninja"
    )

    # -----------------------------------------------------
    # TORCH FIRST
    # -----------------------------------------------------
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        index_url="https://download.pytorch.org/whl/cu124"
    )

    # -----------------------------------------------------
    # FIX NUMPY ABI
    # -----------------------------------------------------
    .run_commands(
        "pip uninstall -y numpy",
        "pip install numpy==1.26.4"
    )

    # -----------------------------------------------------
    # FLASH ATTN
    # -----------------------------------------------------
    .run_commands(
        "pip install flash-attn --no-build-isolation"
    )

    # -----------------------------------------------------
    # GSPLAT
    # -----------------------------------------------------
    .run_commands(
        "pip install gsplat==1.5.3"
    )

    # -----------------------------------------------------
    # AI LIBRARIES
    # -----------------------------------------------------
    .pip_install(
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "diffusers==0.31.0",
        "safetensors",
        "sentencepiece",
        "einops",
        "omegaconf",
        "opencv-python",
        "imageio",
        "scipy",
        "pandas",
        "tqdm",
        "pillow",
        "open3d==0.18.0",
        "trimesh",
        "pygltflib",
        "pymeshlab==2022.2.post3",
        "boto3==1.34.131",
        "botocore==1.34.131",
        "jaxtyping",
        "rich",
        "huggingface_hub"
    )

    # -----------------------------------------------------
    # CLONE HY-WORLD 2.0 (WORLD CREATOR, NOT OBJECT CREATOR)
    # -----------------------------------------------------
    .run_commands(
        "git clone https://github.com/Tencent-Hunyuan/HY-World-2.0.git /root/HunyuanWorld || true"
    )
)

# =========================================================
# VALIDATE REPO
# =========================================================

def validate_repo():
    root = "/root/HunyuanWorld"
    if not os.path.exists(root):
        raise Exception("HY-World-2.0 repo missing.")
    print("Repo validated.")
    print("Repo files:")
    print(os.listdir(root))

# =========================================================
# GENERATOR
# =========================================================

def generate_world(input_img, base_name, prompt):

    import torch

    # -----------------------------------------------------
    # RESILIENT LINKER
    # -----------------------------------------------------
    def setup_resilient_linker(volume_path="/weights", target_path="/root/HunyuanWorld/weights"):
        """
        Dynamically links downloaded model weights from your attached Modal Volume
        into the execution directory so the model doesn't re-download them.
        """
        print(f"EXECUTING RESILIENT LINKER FROM {volume_path}")
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

    # Fire the linker to map /weights to /root/HunyuanWorld/weights
    setup_resilient_linker(volume_path="/weights")

    validate_repo()

    # -----------------------------------------------------
    # CUDA CHECK
    # -----------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    # -----------------------------------------------------
    # PERFORMANCE
    # -----------------------------------------------------
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # -----------------------------------------------------
    # PATHS
    # -----------------------------------------------------
    CODE_ROOT = "/root/HunyuanWorld"

    if CODE_ROOT not in sys.path:
        sys.path.insert(0, CODE_ROOT)

    os.chdir(CODE_ROOT)

    # -----------------------------------------------------
    # IMPORTS & LOAD PIPELINE (HY-World-2.0 Architectures)
    # -----------------------------------------------------
    pipeline_loaded = False
    
    # Attempt to load the HY-World-2.0 world generation pipeline
    try:
        from pipelines import HYWorld2Pipeline
        pipeline_class = HYWorld2Pipeline
        pipeline_loaded = True
        print("LOADED HYWorld2Pipeline")
    except Exception as e:
        try:
            from hyworld2.worldrecon.pipeline import WorldMirrorPipeline
            pipeline_class = WorldMirrorPipeline
            pipeline_loaded = True
            print("LOADED WorldMirrorPipeline")
        except Exception as e2:
            raise Exception(f"Pipeline import failed: {e} | {e2}")

    try:
        from utils.meshing import splat_to_glb
    except Exception as e:
        raise Exception(f"Meshing import failed: {e}")

    print("Loading HY-World 2.0 pipeline...")

    pipeline = pipeline_class.from_pretrained(
        CODE_ROOT, # Assumes weights are mapped to this directory
        device="cuda",
        torch_dtype=torch.bfloat16
    )

    # -----------------------------------------------------
    # GENERATE
    # -----------------------------------------------------
    print(f"Generating World for: {prompt}")

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = pipeline(
                image=input_img,
                prompt=prompt,
                target_resolution=1024,
                output_type="3dgs" # Outputting 3D Gaussian Splats
            )

    # -----------------------------------------------------
    # EXPORT SPLAT TO GLB
    # -----------------------------------------------------
    output_dir = "/tmp/output"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{base_name}.glb"
    )

    print("CONVERTING SPLAT TO GLB")
    splat_to_glb(
        result,
        output_path=output_path,
        simplify_ratio=0.8,
        texture_size=2048
    )

    # -----------------------------------------------------
    # CLEANUP
    # -----------------------------------------------------
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return output_path

# =========================================================
# WORKER
# =========================================================

@app.function(
    image=image,
    gpu="L4",
    timeout=3600,
    max_containers=1,
    volumes={
        "/weights": weights_vol, # Mounts your volume to /weights
        "/cache": cache_vol
    }
)
def process_cloudflare_queue(cfg: dict):

    import boto3
    from PIL import Image

    # -----------------------------------------------------
    # S3
    # -----------------------------------------------------
    s3 = boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"]
    )

    # -----------------------------------------------------
    # QUEUE
    # -----------------------------------------------------
    response = s3.list_objects_v2(
        Bucket=cfg["bucket"],
        Prefix="queue/"
    )

    if "Contents" not in response:
        print("Queue empty.")
        return

    # -----------------------------------------------------
    # LOOP
    # -----------------------------------------------------
    for obj in response["Contents"]:

        key = obj["Key"]

        if key.endswith("/"):
            continue

        if not key.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        print(f"PROCESSING: {key}")

        try:
            # -------------------------------------------------
            # DOWNLOAD
            # -------------------------------------------------
            data = s3.get_object(
                Bucket=cfg["bucket"],
                Key=key
            )

            img = Image.open(
                io.BytesIO(data["Body"].read())
            ).convert("RGB")

            prompt = data.get(
                "Metadata",
                {}
            ).get(
                "prompt",
                "cinematic environment"
            )

            base_name = os.path.splitext(
                os.path.basename(key)
            )[0]

            # -------------------------------------------------
            # GENERATE
            # -------------------------------------------------
            glb_path = generate_world(
                img,
                base_name,
                prompt
            )

            # -------------------------------------------------
            # UPLOAD
            # -------------------------------------------------
            output_key = f"output/{base_name}.glb"
            with open(glb_path, "rb") as f:
                s3.put_object(
                    Bucket=cfg["bucket"],
                    Key=output_key,
                    Body=f,
                    ContentType="model/gltf-binary"
                )

            # -------------------------------------------------
            # DELETE SOURCE
            # -------------------------------------------------
            s3.delete_object(
                Bucket=cfg["bucket"],
                Key=key
            )

            print(f"SUCCESS: {key}")

        except Exception as e:

            print(f"FAILED: {key}")
            print(str(e))

            try:
                failed_key = key.replace(
                    "queue/",
                    "failed/",
                    1
                )

                s3.copy_object(
                    Bucket=cfg["bucket"],
                    CopySource={
                        "Bucket": cfg["bucket"],
                        "Key": key
                    },
                    Key=failed_key
                )

                s3.delete_object(
                    Bucket=cfg["bucket"],
                    Key=key
                )

            except Exception as move_error:
                print(f"Failed moving object: {move_error}")

    try:
        cache_vol.commit()
    except Exception as e:
        print(f"Warning: Cache commit failed: {e}")

# =========================================================
# ENTRYPOINT
# =========================================================

@app.local_entrypoint()
def main():
    # Cloudflare R2 Credentials restored
    config = {
        "endpoint": "https://4d91f4d3d0366568a54ffa32ffcb7bf4.r2.cloudflarestorage.com",
        "access_key": "3c33425ba6e5abbd3e63afab14dc8866",
        "secret_key": "d65f107bb61093843c6dd980c764443fdf50924a7701078b99f007d3060e25a8",
        "bucket": "video-asset-files-storage-workflow"
    }

    process_cloudflare_queue.remote(config)
