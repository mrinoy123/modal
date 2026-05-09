import modal
import os
import sys
import io
import gc
import subprocess

# =========================================================
# APP
# =========================================================

app = modal.App("hyworld-2-production")

# =========================================================
# VOLUMES
# =========================================================

weights_vol = modal.Volume.from_name(
    "weights-hyworld2",
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
    # IMPORTANT
    # FIX NUMPY ABI
    # -----------------------------------------------------

    .run_commands(
        "pip uninstall -y numpy",
        "pip install numpy==1.26.4"
    )

    # -----------------------------------------------------
    # FLASH ATTN
    # BUILD FROM SOURCE
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
    # CLONE HY-WORLD
    # -----------------------------------------------------

    .run_commands(
        """
        cd /root && \
        rm -rf HY-World && \
        git clone https://github.com/Tencent/Hunyuan3D-2.git HY-World
        """
    )
)

# =========================================================
# VALIDATE REPO
# =========================================================

def validate_repo():

    root = "/root/HY-World"

    if not os.path.exists(root):
        raise Exception("HY-World repo missing.")

    print("Repo validated.")

    print("Repo files:")
    print(os.listdir(root))

# =========================================================
# GENERATOR
# =========================================================

def generate_world(input_img, base_name, prompt):

    import torch

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

    CODE_ROOT = "/root/HY-World"

    if CODE_ROOT not in sys.path:
        sys.path.insert(0, CODE_ROOT)

    os.chdir(CODE_ROOT)

    # -----------------------------------------------------
    # IMPORTS
    # -----------------------------------------------------

    try:

        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    except Exception as e:
        raise Exception(f"Pipeline import failed: {e}")

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    print("Loading HY-World pipeline...")

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",
        device="cuda"
    )

    # -----------------------------------------------------
    # GENERATE
    # -----------------------------------------------------

    print(f"Generating: {prompt}")

    with torch.inference_mode():

        result = pipeline(
            image=input_img,
            num_inference_steps=30,
            guidance_scale=5.5
        )

    # -----------------------------------------------------
    # EXPORT
    # -----------------------------------------------------

    output_dir = "/tmp/output"

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{base_name}.glb"
    )

    result[0].export(output_path)

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
        "/weights": weights_vol,
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

            with open(glb_path, "rb") as f:

                s3.put_object(
                    Bucket=cfg["bucket"],
                    Key=f"output/{base_name}.glb",
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

    cache_vol.commit()

# =========================================================
# ENTRYPOINT
# =========================================================

@app.local_entrypoint()
def main():

    config = {
        "endpoint": "https://YOUR-ENDPOINT",
        "access_key": "YOUR-KEY",
        "secret_key": "YOUR-SECRET",
        "bucket": "YOUR-BUCKET"
    }

    process_cloudflare_queue.remote(config)
