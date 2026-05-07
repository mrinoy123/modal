import modal
import os
import sys
import io
import gc
import uuid
import shutil
import traceback
import urllib.request

# =========================================================
# IMAGE
# =========================================================

image = (
    modal.Image
    .from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.10"
    )

    .env({

        # CUDA
        "CUDA_HOME": "/usr/local/cuda",
        "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "CUDA_VISIBLE_DEVICES": "0",

        # stability
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",

        # disable unsupported accelerators
        "ACCELERATE_DISABLE_XPU": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",

        # huggingface cache
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",

        # runtime
        "MAX_JOBS": "4"
    })

    # =====================================================
    # SYSTEM
    # =====================================================

    .apt_install(
        "git",
        "wget",
        "clang",
        "build-essential",
        "cmake",
        "ninja-build",

        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libx11-6",
        "libegl1",
        "libopengl0",
        "libgomp1"
    )

    .pip_install(
        "setuptools",
        "wheel"
    )

    # =====================================================
    # PYTORCH
    # =====================================================

    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu121"
    )

    # =====================================================
    # CORE AI STACK
    # =====================================================

    .pip_install(
        "transformers==4.46.0",
        "diffusers==0.30.0",
        "accelerate==1.1.1",
        "huggingface_hub==0.30.2",
        "deepspeed==0.11.2"
    )

    # =====================================================
    # EXTRA LIBRARIES
    # =====================================================

    .pip_install(
        "boto3",
        "trimesh",
        "pillow",
        "einops",
        "omegaconf",
        "opencv-python",
        "imageio",
        "scikit-image",
        "rembg",
        "realesrgan",
        "basicsr",
        "pyrender",
        "pygltflib",
        "pyyaml",
        "configargparse",
        "hf-transfer",
        "timm",
        "peft",
        "pybind11",
        "safetensors",
        "scipy",
        "pandas",
        "pymeshlab==2022.2.post3"
    )

    # =====================================================
    # OPEN3D FIX
    # =====================================================

    .pip_install(
        "open3d==0.18.0",
        "onnxruntime==1.16.3"
    )

    .run_commands(
        "pip uninstall -y numpy",
        "pip install numpy==1.26.4"
    )

    # =====================================================
    # BLENDER
    # =====================================================

    .run_commands(
        "pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/"
    )

    # =====================================================
    # TORCHMCUBES
    # =====================================================

    .run_commands(
        "git clone https://github.com/tatsy/torchmcubes.git /tmp/torchmcubes",
        'cd /tmp/torchmcubes && TORCH_CUDA_ARCH_LIST="8.9" pip install .'
    )

    # =====================================================
    # U2NET CACHE
    # =====================================================

    .run_commands(
        "mkdir -p ~/.u2net",
        "wget -q https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx -O ~/.u2net/u2net.onnx"
    )

    # =====================================================
    # HUNYUAN
    # =====================================================

    .run_commands(
        "rm -rf /root/hunyuan3d",

        "git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /root/hunyuan3d",

        'cd /root/hunyuan3d/hy3dpaint/custom_rasterizer && TORCH_CUDA_ARCH_LIST="8.9" pip install .',

        'cd /root/hunyuan3d/hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh'
    )
)

# =========================================================
# APP
# =========================================================

app = modal.App(
    "hunyuan3d-stable-final",
    image=image
)

weights_vol = modal.Volume.from_name(
    "weights-hunyuan-21",
    create_if_missing=True
)

cache_vol = modal.Volume.from_name(
    "ai-factory-cache",
    create_if_missing=True
)

# =========================================================
# GLOBAL MODEL CACHE
# =========================================================

shape_pipeline = None
paint_pipeline = None
models_loaded = False

# =========================================================
# LOAD MODELS
# =========================================================

def load_models():

    global shape_pipeline
    global paint_pipeline
    global models_loaded

    if models_loaded:
        return

    import torch
    from omegaconf import OmegaConf

    CODE_ROOT = "/root/hunyuan3d"
    WEIGHT_ROOT = "/weights/Hunyuan3D-2.1-Weights-Dataset"

    if CODE_ROOT not in sys.path:
        sys.path.insert(0, CODE_ROOT)

    if f"{CODE_ROOT}/hy3dshape" not in sys.path:
        sys.path.insert(0, f"{CODE_ROOT}/hy3dshape")

    if f"{CODE_ROOT}/hy3dpaint" not in sys.path:
        sys.path.insert(0, f"{CODE_ROOT}/hy3dpaint")

    os.chdir(CODE_ROOT)

    from hy3dshape.pipelines import (
        Hunyuan3DDiTFlowMatchingPipeline
    )

    from textureGenPipeline import (
        Hunyuan3DPaintPipeline,
        Hunyuan3DPaintConfig
    )

    print("Loading Shape Pipeline...")

    shape_pipeline = (
        Hunyuan3DDiTFlowMatchingPipeline
        .from_pretrained(
            WEIGHT_ROOT,
            subfolder="hunyuan3d-dit-v2-1",
            device="cuda"
        )
    )

    torch.cuda.empty_cache()

    # =====================================================
    # ESRGAN
    # =====================================================

    esrgan_path = "/cache/RealESRGAN_x4plus.pth"

    if not os.path.exists(esrgan_path):

        os.makedirs("/cache", exist_ok=True)

        urllib.request.urlretrieve(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            esrgan_path
        )

    # =====================================================
    # CFG
    # =====================================================

    cfg_path = (
        f"{CODE_ROOT}/hy3dpaint/cfgs/"
        "hunyuan-paint-pbr.yaml"
    )

    cfg = OmegaConf.load(cfg_path)

    cfg.model.pretrained_model_name_or_path = WEIGHT_ROOT

    OmegaConf.save(cfg, cfg_path)

    # =====================================================
    # LOWER VRAM CONFIG
    # =====================================================

    conf = Hunyuan3DPaintConfig(
        max_num_view=4,
        resolution=512
    )

    conf.realesrgan_ckpt_path = esrgan_path
    conf.multiview_cfg_path = cfg_path

    conf.custom_pipeline = (
        f"{CODE_ROOT}/hy3dpaint/hunyuanpaintpbr"
    )

    print("Loading Paint Pipeline...")

    paint_pipeline = Hunyuan3DPaintPipeline(conf)

    models_loaded = True

    print("Pipelines loaded successfully.")

# =========================================================
# GENERATION
# =========================================================

def generate_3d_from_image(input_img, base_name):

    import torch
    import trimesh

    load_models()

    from convert_utils import (
        create_glb_with_pbr_materials
    )

    sid = uuid.uuid4().hex

    tmp_dir = f"/tmp/{sid}"

    os.makedirs(tmp_dir, exist_ok=True)

    base_obj = f"{tmp_dir}/base.obj"
    output_obj = f"{tmp_dir}/painted.obj"
    glb_path = f"{tmp_dir}/{base_name}.glb"

    try:

        with torch.inference_mode():

            print("Generating mesh...")

            outputs = shape_pipeline(
                image=input_img,
                output_type="mesh"
            )

            raw_mesh = (
                outputs[0]
                if isinstance(outputs, list)
                else outputs
            )

            raw_mesh.mesh_f = raw_mesh.mesh_f[:, ::-1]

            mesh = trimesh.Trimesh(
                raw_mesh.mesh_v,
                raw_mesh.mesh_f
            )

            mesh.export(base_obj)

            print("Painting textures...")

            tex_obj = paint_pipeline(
                mesh_path=base_obj,
                image_path=input_img,
                output_mesh_path=output_obj,
                save_glb=False
            )

            textures = {
                "albedo": tex_obj.replace(".obj", ".jpg"),
                "metallic": tex_obj.replace(".obj", "_metallic.jpg"),
                "roughness": tex_obj.replace(".obj", "_roughness.jpg")
            }

            print("Building GLB...")

            create_glb_with_pbr_materials(
                tex_obj,
                textures,
                glb_path
            )

        return glb_path

    finally:

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

# =========================================================
# WORKER
# =========================================================

@app.function(
    gpu="L4",
    timeout=3600,

    # FIXED: renamed from concurrency_limit
    max_containers=1,

    scaledown_window=300,

    volumes={
        "/weights": weights_vol,
        "/cache": cache_vol
    }
)

def process_cloudflare_queue(cfg: dict):

    import boto3
    from PIL import Image

    print("Connecting to Cloudflare R2...")

    s3 = boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"]
    )

    res = s3.list_objects_v2(
        Bucket=cfg["bucket"],
        Prefix="queue/"
    )

    if "Contents" not in res:
        print("Queue empty.")
        return

    print(f"Found {len(res['Contents'])} queue items.")

    for obj in res["Contents"]:

        key = obj["Key"]

        if not key.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            continue

        print(f"Processing: {key}")

        try:

            data = s3.get_object(
                Bucket=cfg["bucket"],
                Key=key
            )

            img = (
                Image.open(
                    io.BytesIO(
                        data["Body"].read()
                    )
                )
                .convert("RGBA")
            )

            base_name = (
                os.path.basename(key)
                .split(".")[0]
            )

            glb_path = generate_3d_from_image(
                img,
                base_name
            )

            output_key = (
                f"output/{base_name}.glb"
            )

            print("Uploading GLB...")

            with open(glb_path, "rb") as f:

                s3.put_object(
                    Bucket=cfg["bucket"],
                    Key=output_key,
                    Body=f,
                    ContentType="model/gltf-binary"
                )

            print("Upload complete.")

            # remove processed queue item
            s3.delete_object(
                Bucket=cfg["bucket"],
                Key=key
            )

            print("Queue item deleted.")

        except Exception as e:

            print("===================================")
            print(f"FAILED: {key}")
            print(str(e))
            traceback.print_exc()
            print("===================================")

        finally:

            gc.collect()

            if os.path.exists("/tmp"):
                try:
                    shutil.rmtree("/tmp", ignore_errors=True)
                    os.makedirs("/tmp", exist_ok=True)
                except:
                    pass

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
