```python
import modal
import os
import sys
import io
import gc

# =========================================================
# IMAGE
# =========================================================

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10"
    )

    # =====================================================
    # ENV
    # =====================================================

    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "CUDA_VISIBLE_DEVICES": "0",
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "FORCE_CUDA": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONUNBUFFERED": "1",
    })

    # =====================================================
    # SYSTEM
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
        "libxrender1",
        "libgomp1"
    )

    # =====================================================
    # BASE
    # =====================================================

    .pip_install(
        "setuptools",
        "wheel",
        "ninja",
        "packaging"
    )

    # =====================================================
    # TORCH 2.4 CUDA 12.4
    # =====================================================

    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        index_url="https://download.pytorch.org/whl/cu124"
    )

    # =====================================================
    # PIN NUMPY FIRST
    # =====================================================

    .pip_install(
        "numpy==1.26.4"
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
        "pandas",
        "tqdm",
        "pillow"
    )

    # =====================================================
    # 3D
    # =====================================================

    .pip_install(
        "open3d==0.18.0",
        "trimesh",
        "pymeshlab==2022.2.post3",
        "pygltflib"
    )

    # =====================================================
    # INSTALL FLASH-ATTN + GSPLAT
    # =====================================================

    .run_commands(
        "pip install /weights/dependencies/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
        "pip install /weights/dependencies/gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl"
    )

    # =====================================================
    # REPAIR NUMPY AFTER GSPLAT
    # =====================================================

    .run_commands(
        "pip install --force-reinstall numpy==1.26.4"
    )

    # =====================================================
    # INSTALL BOTO3 LAST
    # =====================================================

    .pip_install(
        "boto3==1.34.131",
        "botocore==1.34.131"
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
# GENERATOR
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
    # RESOLVE PATHS
    # =====================================================

    def resolve_paths():

        candidate_mounts = [
            "/weights",
            "/cache",
            "/mnt",
            "/root"
        ]

        code_root = None

        for mount in candidate_mounts:

            candidate = os.path.join(
                mount,
                "HY-World-2.0"
            )

            if os.path.exists(candidate):
                code_root = candidate
                break

        if not code_root:
            raise FileNotFoundError(
                "HY-World-2.0 NOT FOUND"
            )

        helper_root = os.path.join(
            os.path.dirname(code_root),
            "helpers"
        )

        return code_root, helper_root

    CODE_ROOT, HELPER_ROOT = resolve_paths()

    print(f"FOUND CODE ROOT: {CODE_ROOT}")

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

    print("LOADING HYWORLD2 PIPELINE")

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
    # GENERATE
    # =====================================================

    print("GENERATING 3D WORLD")

    with torch.autocast(
        "cuda",
        dtype=torch.bfloat16
    ):

        result = pipeline(
            image=input_img,
            prompt=prompt,
            negative_prompt=(
                "blurry, messy, low quality, "
                "photorealistic, realistic texture"
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

    print("CONVERTING SPLAT TO GLB")

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

    import boto3
    from PIL import Image

    print("CONNECTING TO CLOUDFLARE R2")

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
            len(key.split("/")) != 2 or
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
```
