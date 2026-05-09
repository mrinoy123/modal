import modal
import os
import sys
import io
import gc

# =========================================================
# MODAL APP
# =========================================================

app = modal.App("hyworld-production-final")

# =========================================================
# VOLUMES
# =========================================================

# POINTING DIRECTLY TO YOUR SPECIFIC WORKSPACE VOLUME
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

    # =====================================================
    # ENVIRONMENT
    # =====================================================

    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "FORCE_CUDA": "1",
        "MAX_JOBS": "8",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": "/cache/huggingface",
        "TRANSFORMERS_CACHE": "/cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/cache/huggingface"
    })

    # =====================================================
    # SYSTEM PACKAGES
    # =====================================================

    .apt_install(
        "git",
        "wget",
        "ffmpeg",
        "build-essential",
        "clang",
        "cmake",
        "ninja-build",
        "pkg-config",
        "curl",
        "ca-certificates",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgomp1",
        "libopenblas-dev"
    )

    # =====================================================
    # PYTHON TOOLS
    # =====================================================

    .pip_install(
        "pip",
        "setuptools",
        "wheel",
        "packaging",
        "ninja"
    )

    # =====================================================
    # TORCH
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
        "numpy==1.26.4",
        "scipy",
        "pandas",
        "tqdm",
        "pillow",
        "opencv-python",
        "imageio",
        "einops",
        "sentencepiece",
        "omegaconf",
        "safetensors",
        "accelerate==1.1.1",
        "transformers==4.46.3",
        "diffusers==0.31.0",
        "huggingface_hub",
        "timm",
        "xformers",
        "rich",
        "jaxtyping"
    )

    # =====================================================
    # 3D
    # =====================================================

    .pip_install(
        "open3d==0.18.0",
        "trimesh",
        "pygltflib",
        "pymeshlab==2022.2.post3"
    )

    # =====================================================
    # CLOUD
    # =====================================================

    .pip_install(
        "boto3==1.34.131",
        "botocore==1.34.131"
    )

    # =====================================================
    # FLASH ATTENTION
    # =====================================================

    .run_commands(
        "pip install flash-attn --no-build-isolation"
    )

    # =====================================================
    # GSPLAT
    # =====================================================

    .run_commands(
        "pip install gsplat==1.5.3"
    )

    # =====================================================
    # CLONE HYWORLD (UPGRADED TO 2.0)
    # =====================================================

    .run_commands(
        "git clone https://github.com/Tencent-Hunyuan/HY-World-2.0.git /root/HunyuanWorld || true"
    )

    # =====================================================
    # DEBUG
    # =====================================================

    .run_commands(
        "echo 'VERIFYING HYWORLD'",
        "ls -la /root",
        "ls -la /root/HunyuanWorld || true"
    )
)

# =========================================================
# GENERATOR
# =========================================================

def generate_world(input_img, base_name, prompt):

    import torch

    # =====================================================
    # RESILIENT LINKER
    # =====================================================
    
    def setup_resilient_linker(volume_path="/weights", target_paths=["/root/HunyuanWorld/weights", "/root/weights"]):
        """
        Safely links necessary files from the mounted Modal volume to the execution directories
        so models don't need to be downloaded from scratch.
        """
        print(f"EXECUTING RESILIENT LINKER FROM {volume_path}")
        if not os.path.exists(volume_path):
            print(f"WARNING: Volume path {volume_path} not found.")
            return
            
        for target_path in target_paths:
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

    # Fire the linker before loading the model
    setup_resilient_linker(volume_path="/weights")

    # =====================================================
    # CUDA CHECK
    # =====================================================

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA NOT AVAILABLE"
        )

    print("CUDA VERIFIED")

    # =====================================================
    # PERFORMANCE
    # =====================================================

    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # =====================================================
    # REPO
    # =====================================================

    CODE_ROOT = "/root/HunyuanWorld"

    if not os.path.exists(CODE_ROOT):
        raise Exception(
            f"REPO MISSING: {CODE_ROOT}"
        )

    print(f"USING REPO: {CODE_ROOT}")
    print("ROOT CONTENTS:")
    print(os.listdir(CODE_ROOT))

    # =====================================================
    # PYTHON PATH
    # =====================================================

    if CODE_ROOT not in sys.path:
        sys.path.insert(0, CODE_ROOT)

    os.chdir(CODE_ROOT)

    # =====================================================
    # IMPORTS (UPDATED FOR 2.0 ARCHITECTURE)
    # =====================================================

    pipeline_loaded = False

    try:
        from hyworld.pipelines import HYWorldPipeline
        pipeline_loaded = True
        print("LOADED NEW PIPELINE")
    except Exception as e:
        print(f"NEW PIPELINE FAILED: {e}")

    if not pipeline_loaded:
        try:
            from pipelines import HYWorld2Pipeline
            HYWorldPipeline = HYWorld2Pipeline
            pipeline_loaded = True
            print("LOADED OLD PIPELINE")
        except Exception as e:
            try:
                # HunyuanWorld 2.0 Fallback to WorldMirrorPipeline
                from hyworld2.worldrecon.pipeline import WorldMirrorPipeline
                HYWorldPipeline = WorldMirrorPipeline
                pipeline_loaded = True
                print("LOADED WORLDMIRROR 2.0 PIPELINE")
            except Exception as e2:
                raise Exception(
                    f"FAILED IMPORTING PIPELINES: {e} | {e2}"
                )

    # =====================================================
    # MESH EXPORT
    # =====================================================

    try:
        from hyworld.utils.meshing import splat_to_glb
    except:
        try:
            from utils.meshing import splat_to_glb
        except Exception as e:
            raise Exception(
                f"FAILED IMPORTING MESHING: {e}"
            )

    # =====================================================
    # LOAD MODEL
    # =====================================================

    print("LOADING MODEL")

    try:
        pipeline = HYWorldPipeline.from_pretrained(
            CODE_ROOT,
            device="cuda",
            torch_dtype=torch.bfloat16
        )
    except Exception as e:
        raise Exception(
            f"MODEL LOAD FAILURE: {e}"
        )

    # =====================================================
    # GENERATE
    # =====================================================

    print("STARTING GENERATION")

    with torch.inference_mode():
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16
        ):
            result = pipeline(
                image=input_img,
                prompt=prompt,
                negative_prompt=(
                    "blurry, noisy, "
                    "low quality, "
                    "distorted, "
                    "photorealistic"
                ),
                target_resolution=1024,
                output_type="3dgs"
            )

    # =====================================================
    # OUTPUT
    # =====================================================

    output_dir = "/tmp/hyworld_output"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    glb_path = os.path.join(
        output_dir,
        f"{base_name}.glb"
    )

    print("EXPORTING GLB")

    try:
        splat_to_glb(
            result,
            output_path=glb_path,
            simplify_ratio=0.8,
            texture_size=2048
        )
    except Exception as e:
        raise Exception(
            f"GLB EXPORT FAILED: {e}"
        )

    # =====================================================
    # CLEANUP
    # =====================================================

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    print(f"SUCCESS: {glb_path}")

    return glb_path

# =========================================================
# WORKER
# =========================================================

@app.function(
    image=image, # <-- Properly binding your environment to the worker
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
    # IMPORTS
    # =====================================================

    import boto3
    from PIL import Image

    # =====================================================
    # CONNECT R2
    # =====================================================

    print("CONNECTING TO R2")

    s3 = boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"]
    )

    # =====================================================
    # CHECK QUEUE
    # =====================================================

    print("CHECKING QUEUE")

    response = s3.list_objects_v2(
        Bucket=cfg["bucket"],
        Prefix="queue/"
    )

    if "Contents" not in response:
        print("QUEUE EMPTY")
        return

    # =====================================================
    # LOOP
    # =====================================================

    for obj in response["Contents"]:

        key = obj["Key"]

        # =================================================
        # FILTER
        # =================================================

        if key.endswith("/"):
            continue

        if not key.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            continue

        print(f"STARTING TASK: {key}")

        try:
            # =============================================
            # DOWNLOAD
            # =============================================

            data = s3.get_object(
                Bucket=cfg["bucket"],
                Key=key
            )

            img = Image.open(
                io.BytesIO(
                    data["Body"].read()
                )
            ).convert("RGB")

            # =============================================
            # PROMPT
            # =============================================

            prompt = data.get(
                "Metadata",
                {}
            ).get(
                "prompt",
                "comic cinematic city world"
            )

            base_name = os.path.splitext(
                os.path.basename(key)
            )[0]

            print(f"PROMPT: {prompt}")

            # =============================================
            # GENERATE
            # =============================================

            glb_path = generate_world(
                img,
                base_name,
                prompt
            )

            # =============================================
            # UPLOAD
            # =============================================

            output_key = (
                f"output/{base_name}.glb"
            )

            print(f"UPLOADING: {output_key}")

            with open(glb_path, "rb") as f:
                s3.put_object(
                    Bucket=cfg["bucket"],
                    Key=output_key,
                    Body=f,
                    ContentType="model/gltf-binary"
                )

            # =============================================
            # DELETE ORIGINAL
            # =============================================

            s3.delete_object(
                Bucket=cfg["bucket"],
                Key=key
            )

            print(f"TASK COMPLETE: {key}")

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

                print(f"MOVED TO FAILED: {failed_key}")

            except Exception as move_error:
                print(f"FAILED MOVING FILE: {move_error}")

    # =====================================================
    # SAVE CACHE
    # =====================================================
    try:
        cache_vol.commit()
    except Exception as e:
        print(f"Warning: Cache volume commit failed: {e}")


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

    process_cloudflare_queue.remote(
        config
    )
