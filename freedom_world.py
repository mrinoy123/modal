import modal
import os
import sys
import io
import uuid
import gc

# =========================================================
# IMAGE (STABLE HY-WORLD 2.0 BUILD)
# =========================================================

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10"
    )

    .env({
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "CUDA_HOME": "/usr/local/cuda",
        "FORCE_CUDA": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })

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

    .pip_install(
        "setuptools",
        "wheel",
        "ninja"
    )

    # =====================================================
    # TORCH 2.4 (Required for HY-World + FA2.8)
    # =====================================================

    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        index_url="https://download.pytorch.org/whl/cu124"
    )

    # =====================================================
    # CORE AI STACK
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
    )

    # =====================================================
    # 3D STACK
    # =====================================================

    .pip_install(
        "open3d==0.18.0",
        "trimesh",
        "pymeshlab==2022.2.post3",
        "pygltflib",
    )

    # =====================================================
    # FLASH ATTENTION + GSPLAT
    # =====================================================

    .run_commands(
        "pip install /weights/dependencies/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
        "pip install /weights/dependencies/gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl"
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
# MODAL APP
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
# GENERATION PIPELINE
# =========================================================

def generate_world(input_img, base_name, prompt):

    import torch
    import numpy as np
    import trimesh
    import gc

    assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"

    print("✅ CUDA AVAILABLE")

    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # =====================================================
    # RESILIENT PATH LINKER
    # =====================================================

    def resolve_paths():

        candidate_mounts = [
            "/weights",
            "/cache",
            "/mnt",
            "/root"
        ]

        code_root = None
        helper_root = None

        for mount in candidate_mounts:

            target = os.path.join(mount, "HY-World-2.0")

            if os.path.exists(target):
                code_root = target
                break

        if not code_root:
            raise FileNotFoundError(
                "HY-World-2.0 repository missing"
            )

        helper_root = os.path.join(
            os.path.dirname(code_root),
            "helpers"
        )

        return code_root, helper_root

    CODE_ROOT, HELPER_ROOT = resolve_paths()

    print(f"✅ CODE ROOT: {CODE_ROOT}")

    # =====================================================
    # PATH SETUP
    # =====================================================

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

    print("🚀 Loading HY-World Pipeline...")

    pipeline = HYWorld2Pipeline.from_pretrained(
        CODE_ROOT,
        llm_path=os.path.join(HELPER_ROOT, "llm"),
        vision_path=os.path.join(HELPER_ROOT, "siglip"),
        device="cuda",
        torch_dtype=torch.bfloat16
    )

    # =====================================================
    # GENERATE SPLAT
    # =====================================================

    print("🎨 Generating Gaussian Splat...")

    with torch.autocast("cuda", dtype=torch.bfloat16):

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
    # OUTPUT PATHS
    # =====================================================

    output_dir = "/tmp/hyworld_output"

    os.makedirs(output_dir, exist_ok=True)

    glb_path = os.path.join(
        output_dir,
        f"{base_name}.glb"
    )

    # =====================================================
    # SPLAT -> GLB
    # =====================================================

    print("🪄 Converting Splat to GLB...")

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

    print(f"✅ FINAL GLB: {glb_path}")

    return glb_path

# =========================================================
# WORKER
# =========================================================

@app.function(
    volumes={
        "/weights": weights_vol,
        "/cache": cache_vol
    },

    gpu="L4",

    timeout=3600,

    container_idle_timeout=60,

    # MODAL 2026 FIX
    max_containers=1
)

def process_cloudflare_queue(cfg: dict):

    import boto3
    from PIL import Image

    print("🌐 Connecting to Cloudflare R2...")

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
        print("📭 Queue Empty")
        return

    processed = 0

    for obj in response["Contents"]:

        key = obj["Key"]

        if (
            len(key.split("/")) != 2 or
            not key.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ):
            continue

        processed += 1

        print(f"📥 Processing: {key}")

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

            print("☁️ Uploading GLB to R2...")

            with open(glb_file, "rb") as f:

                s3.put_object(
                    Bucket=cfg["bucket"],
                    Key=f"output/{os.path.basename(glb_file)}",
                    Body=f
                )

            print("✅ Upload complete")

            s3.delete_object(
                Bucket=cfg["bucket"],
                Key=key
            )

        except Exception as e:

            print(f"❌ FAILED: {e}")

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

    if processed == 0:
        print("📭 No valid images found")

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
