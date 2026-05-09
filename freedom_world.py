import modal
import os
import sys
import io
import gc

# =========================================================
# APP
# =========================================================

app = modal.App("hyworld2-world-generator")

# =========================================================
# VOLUMES
# =========================================================

weights_vol = modal.Volume.from_name(
    "hyworld2-weights",
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
    # ENV
    # =====================================================

    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "MAX_JOBS": "6",
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/cache/huggingface",
        "TRANSFORMERS_CACHE": "/cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/cache/huggingface",
        "XFORMERS_FORCE_DISABLE_TRITON": "1",
    })

    # =====================================================
    # SYSTEM
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
        "libxrender-dev",
        "libgomp1",

        "libopenblas-dev",
        "libeigen3-dev",

        "python3-dev",

        # OPEN3D FIXES
        "libusb-1.0-0",
        "libudev1",

        # GSPLAT / RASTERIZATION
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
    # NUMPY ABI FIX
    # =====================================================

    .run_commands(
        "pip uninstall -y numpy",
        "pip install numpy==1.26.4"
    )

    # =====================================================
    # XFORMERS
    # =====================================================

    .run_commands(
        "pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu124"
    )

    # =====================================================
    # FLASH ATTN
    # =====================================================

    .run_commands(
        "pip install flash-attn==2.7.4.post1 --no-build-isolation || true"
    )

    # =====================================================
    # GSPLAT
    # =====================================================

    .run_commands(
        "pip install gsplat==1.5.3"
    )

    # =====================================================
    # HY-WORLD DEPENDENCIES
    # =====================================================

    .pip_install(

        # AI
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "diffusers==0.31.0",
        "safetensors",
        "sentencepiece",
        "einops",
        "omegaconf",

        # Image
        "opencv-python-headless",
        "pillow",
        "imageio",
        "imageio-ffmpeg",

        # Math
        "scipy",
        "numpy==1.26.4",

        # Utilities
        "tqdm",
        "rich",
        "jaxtyping",
        "typing_extensions",

        # 3D
        "open3d==0.18.0",
        "trimesh",
        "pygltflib",
        "pymeshlab==2022.2.post3",
        "plyfile",
        "PyMCubes",
        "pyvista",
        "vtk",

        # ML
        "huggingface_hub",
        "timm",
        "kornia",

        # Cloud
        "boto3==1.34.131",
        "botocore==1.34.131"
    )

    # =====================================================
    # CLONE HY-WORLD
    # =====================================================

    .run_commands(

        "rm -rf /root/HYWorld",

        "git clone https://github.com/Tencent-Hunyuan/HY-World-2.0.git /root/HYWorld",

        "cd /root/HYWorld && pip install -r requirements.txt || true",

        # EXTRA FIXES
        "pip install plyfile",
        "pip install roma",
        "pip install pyquaternion",
        "pip install nerfacc",
        "pip install git+https://github.com/NVlabs/nvdiffrast.git",

        # VERIFY
        "ls -la /root/HYWorld",
        "find /root/HYWorld -maxdepth 2 -type f | head -50"
    )

)

# =========================================================
# GENERATION
# =========================================================

def generate_world(image_input, prompt, output_name):

    import torch
    from PIL import Image

    # =====================================================
    # PATHS
    # =====================================================

    ROOT = "/root/HYWorld"

    sys.path.insert(0, ROOT)

    os.chdir(ROOT)

    # =====================================================
    # CUDA
    # =====================================================

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    print("CUDA DEVICE:", torch.cuda.get_device_name(0))

    # =====================================================
    # IMPORTS
    # =====================================================

    try:

        from hyworld2.inference import HunyuanWorldPipeline

    except Exception as e:

        raise Exception(f"REAL PIPELINE IMPORT FAILED:\n{e}")

    # =====================================================
    # LOAD MODEL
    # =====================================================

    print("LOADING HY-WORLD MODEL")

    pipe = HunyuanWorldPipeline.from_pretrained(

        ROOT,

        torch_dtype=torch.bfloat16

    ).to("cuda")

    # =====================================================
    # OPTIMIZATION
    # =====================================================

    pipe.enable_model_cpu_offload()

    # =====================================================
    # GENERATE
    # =====================================================

    print("STARTING WORLD GENERATION")

    with torch.inference_mode():

        result = pipe(

            image=image_input,
            prompt=prompt,

            num_inference_steps=30,

            guidance_scale=5.0,

            output_type="mesh"

        )

    # =====================================================
    # EXPORT
    # =====================================================

    output_dir = "/tmp/output"

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{output_name}.glb"
    )

    mesh = result["mesh"]

    mesh.export(output_path)

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
                .get("prompt", "cinematic 3d environment")
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

            print("FAILED")
            print(str(e))





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
