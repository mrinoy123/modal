import modal
import os
import sys
import io
import gc

# =========================================================
# MODAL IMAGE
# =========================================================

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10"
    )

    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "FORCE_CUDA": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })

    # =====================================================
    # SYSTEM DEPENDENCIES
    # =====================================================

    .apt_install(
        "git",
        "wget",
        "build-essential",
        "clang",
        "cmake",
        "ninja-build",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libopenblas-dev",
        "libsm6",
        "libxext6",
        "libxrender1"
    )

    # =====================================================
    # BASE PYTHON TOOLS
    # =====================================================

    .pip_install(
        "setuptools",
        "wheel",
        "ninja"
    )

    # =====================================================
    # TORCH 2.4
    # =====================================================

    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        index_url="https://download.pytorch.org/whl/cu124"
    )

    # =====================================================
    # CORE AI
    # =====================================================

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
        "numpy==1.26.4",
        "pandas",
        "tqdm",
        "pillow",
        "boto3",
        "botocore"
    )

    # =====================================================
    # 3D STACK
    # =====================================================

    .pip_install(
        "open3d==0.18.0",
        "trimesh",
        "pymeshlab==2022.2.post3",
        "pygltflib"
    )

    # =====================================================
    # CLEANUP
    # =====================================================

    .run_commands(
        "pip uninstall -y pynvml || true",
        "pip cache purge || true"
    )
)

# =========================================================
# APP
# =========================================================

app = modal.App("hyworld-production")

weights_vol = modal.Volume.from_name(
    "weights-hy-world-2",
    create_if_missing=True
)

cache_vol = modal.Volume.from_name(
    "hyworld-cache",
    create_if_missing=True
)

# =========================================================
# RUNTIME INSTALLER
# =========================================================

def install_runtime_dependencies():

    import subprocess
    import os

    flash_wheel = "/weights/dependencies/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

    gsplat_wheel = "/weights/dependencies/gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl"

    if os.path.exists(flash_wheel):
        print("Installing FlashAttention...")
        subprocess.run(
            ["pip", "install", flash_wheel],
            check=True
        )

    if os.path.exists(gsplat_wheel):
        print("Installing gsplat...")
        subprocess.run(
            ["pip", "install", gsplat_wheel],
            check=True
        )

# =========================================================
# WORLD GENERATION
# =========================================================

def generate_world(input_img, base_name, prompt):

    import torch
    import gc

    assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"

    print("CUDA VERIFIED")

    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # =====================================================
    # FIND REPO
    # =====================================================

    def resolve_paths():

        candidates = [
            "/weights",
            "/cache",
            "/mnt",
            "/root"
        ]

        code_root = None

        for mount in candidates:

            target = os.path.join(
                mount,
                "HY-World-2.0"
            )

            if os.path.exists(target):
                code_root = target
                break

        if not code_root:
            raise Exception(
                "HY-World-2.0 repository missing"
            )

        helper_root = os.path.join(
            os.path.dirname(code_root),
            "helpers"
        )

        return code_root, helper_root

    CODE_ROOT, HELPER_ROOT = resolve_paths()

    print(f"FOUND REPO: {CODE_ROOT}")

    sys.path.insert(0, CODE_ROOT)

    os.chdir(CODE_ROOT)

    # =====================================================
    # IMPORTS
    # =====================================================

    from pipelines import HYWorld2Pipeline
    from utils.meshing import splat_to_glb

    # =====================================================
    # LOAD PIPELINE
    # =====================================================

    print("LOADING PIPELINE")

    pipeline = HYWorld2Pipeline.from_pretrained(
        CODE_ROOT,
        llm_path=os.path.join(
            HELPER_ROOT,
            "llm"
        ),
        vision_path=os.path.join(
            HELPER_ROOT,
            "siglip"
        ),
        device="cuda",
        torch_dtype=torch.bfloat16
    )

    # =====================================================
    # GENERATION
    # =====================================================

    print("GENERATING WORLD")

    with torch.autocast(
        "cuda",
        dtype=torch.bfloat16
    ):

        result = pipeline(
            image=input_img,
            prompt=prompt,
            negative_prompt=(
                "blurry, messy, low quality, "
                "photorealistic"
            ),
            target_resolution=1024,
            output_type="3dgs"
        )

    # =====================================================
    # OUTPUT
    # =====================================================

    output_dir = "/tmp/hyworld_output"

    os.makedirs(output_dir, exist_ok=True)

    glb_path = os.path.join(
        output_dir,
        f"{base_name}.glb"
    )

    print("CONVERTING TO GLB")

    splat_to_glb(
        result,
        output_path=glb_path,
        simplify_ratio=0.8,
        texture_size=2048
    )

    # =====================================================
    # CLEANUP
    # =====================================================

    del pipeline

    gc.collect()

    torch.cuda.empty_cache()

    print(f"FINAL GLB: {glb_path}")

    return glb_path

# =========================================================
# WORKER
# =========================================================

@app.function(
    gpu="L4",

    timeout=3600,

    scaledown_window=60,

    max_containers=1,

    volumes={
        "/weights": weights_vol,
        "/cache": cache_vol
    }
)

def process_cloudflare_queue(cfg: dict):

    # =====================================================
    # INSTALL RUNTIME WHEELS
    # =====================================================

    install_runtime_dependencies()

    import boto3
    from PIL import Image

    print("CONNECTING TO R2")

    s3 = boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"]
    )

    response = s3.list_objects_v2(
        Bucket=cfg["bucket"],
        Prefix="queue/"
    )

    if "Contents" not in response:
        print("QUEUE EMPTY")
        return

    for obj in response["Contents"]:

        key = obj["Key"]

        if (
            len(key.split("/")) != 2
            or
            not key.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ):
            continue

        print(f"PROCESSING: {key}")

        data = s3.get_object(
            Bucket=cfg["bucket"],
            Key=key
        )

        img = Image.open(
            io.BytesIO(
                data["Body"].read()
            )
        ).convert("RGB")

        prompt = data.get(
            "Metadata",
            {}
        ).get(
            "prompt",
            "comic cinematic city world"
        )

        try:

            base_name = (
                key.split("/")[-1]
                .split(".")[0]
            )

            glb_file = generate_world(
                img,
                base_name,
                prompt
            )

            print("UPLOADING GLB")

            with open(glb_file, "rb") as f:

                s3.put_object(
                    Bucket=cfg["bucket"],
                    Key=f"output/{os.path.basename(glb_file)}",
                    Body=f
                )

            print("UPLOAD COMPLETE")

            s3.delete_object(
                Bucket=cfg["bucket"],
                Key=key
            )

        except Exception as e:

            print(f"FAILED: {e}")

            failed_key = key.replace(
                "queue/",
                "queue/failed/",
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

    cache_vol.commit()

# =========================================================
# ENTRYPOINT
# =========================================================

@app.local_entrypoint()

def main():

    process_cloudflare_queue.remote({

        "endpoint":
        "https://4d91f4d3d0366568a54ffa32ffcb7bf4.r2.cloudflarestorage.com",

        "access_key":
        "3c33425ba6e5abbd3e63afab14dc8866",

        "secret_key":
        "d65f107bb61093843c6dd980c764443fdf50924a7701078b99f007d3060e25a8",

        "bucket":
        "video-asset-files-storage-workflow"
    })
