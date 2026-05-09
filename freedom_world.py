import modal
import os
import sys
import io
import gc

# =========================================================
# APP
# =========================================================

app = modal.App("hyworld2-production")

# =========================================================
# VOLUMES
# =========================================================

weights_vol = modal.Volume.from_name(
    "weights-hy-world-2", # Mapped exactly to your Modal storage
    create_if_missing=True
)

cache_vol = modal.Volume.from_name(
    "hyworld2-cache",
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
        "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "MAX_JOBS": "4",
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/cache/huggingface",
        "TRANSFORMERS_CACHE": "/cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/cache/huggingface",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    })

    # =====================================================
    # SYSTEM PACKAGES
    # =====================================================
    .apt_install(
        "git",
        "git-lfs",
        "wget",
        "curl",
        "ffmpeg",
        "build-essential",
        "clang",
        "cmake",
        "ninja-build",
        "pkg-config",
        "mesa-utils",
        "libgl1-mesa-glx",
        "libglu1-mesa",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgomp1",
        "libopenblas-dev",
        "libeigen3-dev",
        "python3-dev",
        "libx11-6",
        "libxi6",
        "libxxf86vm1",
        "libxrandr2"
    )

    # =====================================================
    # PYTHON CORE
    # =====================================================
    .pip_install(
        "pip==25.1.1",
        "setuptools==70.0.0",
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
    # NUMPY FIX
    # =====================================================
    .run_commands(
        "pip uninstall -y numpy",
        "pip install numpy==1.26.4"
    )

    # =====================================================
    # CORE AI LIBRARIES
    # =====================================================
    .pip_install(
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "diffusers==0.31.0",
        "safetensors",
        "sentencepiece",
        "einops",
        "omegaconf",
        "timm",
        "kornia"
    )

    # =====================================================
    # IMAGE / VIDEO
    # =====================================================
    .pip_install(
        "opencv-python-headless",
        "pillow",
        "imageio",
        "imageio-ffmpeg"
    )

    # =====================================================
    # 3D / MESH
    # =====================================================
    .pip_install(
        "open3d==0.18.0",
        "trimesh",
        "pygltflib",
        "pymeshlab==2022.2.post3",
        "plyfile",
        "PyMCubes"
    )

    # =====================================================
    # GSPLAT + NERF
    # =====================================================
    .run_commands(
        "pip install gsplat==1.5.3",
        "pip install nerfacc"
    )

    # =====================================================
    # UTILITIES
    # =====================================================
    .pip_install(
        "scipy",
        "pandas",
        "tqdm",
        "rich",
        "jaxtyping",
        "typing_extensions",
        "huggingface_hub"
    )

    # =====================================================
    # CLOUD
    # =====================================================
    .pip_install(
        "boto3==1.34.131",
        "botocore==1.34.131"
    )

    # =====================================================
    # EXTRA HYWORLD FIXES
    # =====================================================
    .run_commands(
        "pip install roma",
        "pip install pyquaternion"
    )

    # =====================================================
    # CLONE HY-WORLD
    # =====================================================
    .run_commands(
        "rm -rf /root/HYWorld",
        "git clone https://github.com/Tencent-Hunyuan/HY-World-2.0.git /root/HYWorld",
        "cd /root/HYWorld && pip install -r requirements.txt || true",
        "find /root/HYWorld -maxdepth 3 -type f | head -100"
    )
)

# =========================================================
# GENERATION
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

    # Fire the linker to map the Modal volume files to the local execution folder
    setup_resilient_linker(volume_path="/weights", target_path=f"{ROOT}/weights")

    # =====================================================
    # PATH INJECTION
    # =====================================================
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    os.chdir(ROOT)

    # =====================================================
    # CUDA CHECK
    # =====================================================
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    print("CUDA:", torch.cuda.get_device_name(0))

    # =====================================================
    # FIND REAL PIPELINE
    # =====================================================
    pipeline_loaded = False
    import importlib

    candidate_imports = [
        ("hyworld2.inference", "HunyuanWorldPipeline"),
        ("hyworld2.pipeline", "HunyuanWorldPipeline"),
        ("hyworld2.world.pipeline", "HunyuanWorldPipeline"),
        ("hyworld2.world.inference", "HunyuanWorldPipeline"),
    ]

    for module_name, class_name in candidate_imports:
        try:
            module = importlib.import_module(module_name)
            pipeline_class = getattr(module, class_name)
            print(f"SUCCESS IMPORT: {module_name}")
            pipeline_loaded = True
            break
        except Exception as e:
            print(f"FAILED IMPORT: {module_name} -> {e}")

    if not pipeline_loaded:
        raise RuntimeError("Could not locate HYWorld pipeline.")

    # =====================================================
    # LOAD MODEL
    # =====================================================
    print("LOADING MODEL")

    pipe = pipeline_class.from_pretrained(
        ROOT, # Uses the symlinked weights folder
        torch_dtype=torch.float16
    ).to("cuda")

    # =====================================================
    # GENERATE
    # =====================================================
    print("GENERATING WORLD")

    with torch.inference_mode():
        result = pipe(
            image=input_image,
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=5.0
        )

    # =====================================================
    # OUTPUT
    # =====================================================
    output_dir = "/tmp/output"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{output_name}.glb"
    )

    # =====================================================
    # EXPORT
    # =====================================================
    if isinstance(result, dict):
        if "mesh" in result:
            result["mesh"].export(output_path)
        elif "scene" in result:
            result["scene"].export(output_path)
        else:
            raise RuntimeError("No mesh/scene in result.")
    else:
        result.export(output_path)

    # =====================================================
    # CLEANUP
    # =====================================================
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return output_path

# =========================================================
# WORKER
# =========================================================

@app.function(
    image=image,
    gpu="L4",
    timeout=7200,
    max_containers=1,
    volumes={
        "/weights": weights_vol, # Mounts your Modal Volume to /weights
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

    response = s3.list_objects_v2(
        Bucket=cfg["bucket"],
        Prefix="queue/"
    )

    if "Contents" not in response:
        print("QUEUE EMPTY")
        return

    for obj in response["Contents"]:
        key = obj["Key"]

        if not key.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        print(f"PROCESSING: {key}")

        try:
            data = s3.get_object(
                Bucket=cfg["bucket"],
                Key=key
            )

            img = Image.open(
                io.BytesIO(data["Body"].read())
            ).convert("RGB")

            prompt = (
                data.get("Metadata", {})
                .get("prompt", "cinematic 3d world")
            )

            base_name = os.path.splitext(
                os.path.basename(key)
            )[0]

            glb_path = generate_world(
                img,
                prompt,
                base_name
            )

            output_key = f"output/{base_name}.glb"

            with open(glb_path, "rb") as f:
                s3.put_object(
                    Bucket=cfg["bucket"],
                    Key=output_key,
                    Body=f,
                    ContentType="model/gltf-binary"
                )

            s3.delete_object(
                Bucket=cfg["bucket"],
                Key=key
            )

            print(f"SUCCESS: {key}")

        except Exception as e:
            print("FAILED:")
            print(str(e))
            
            # Move failed files to failed directory to prevent infinite loops
            try:
                failed_key = key.replace("queue/", "queue/failed/", 1)
                s3.copy_object(
                    Bucket=cfg["bucket"],
                    CopySource={"Bucket": cfg["bucket"], "Key": key},
                    Key=failed_key
                )
                s3.delete_object(Bucket=cfg["bucket"], Key=key)
            except Exception as move_error:
                print(f"Failed to move object: {move_error}")

    try:
        cache_vol.commit()
    except Exception as e:
        print(f"Warning: cache volume commit failed: {e}")

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
