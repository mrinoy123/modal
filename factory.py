import modal
import os
import sys
import io
import urllib.request
import uuid

# ==========================================
# IMAGE
# ==========================================
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")

    .env({
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "CUDA_HOME": "/usr/local/cuda",
        "FORCE_CUDA": "1",
        "MAX_JOBS": "4"
    })

    .apt_install(
        "git","build-essential","clang","cmake","ninja-build",
        "libgl1-mesa-glx","libglib2.0-0","libopengl0","libegl1",
        "libsm6","libxext6","libxrender1","libx11-6","libxi6",
        "libxxf86vm1","libxfixes3","libxkbcommon0"
    )

    # 🔥 CRITICAL FIX
    .pip_install("numpy==1.26.4")

    .pip_install("setuptools","wheel")

    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "torchaudio==2.1.2",
        index_url="https://download.pytorch.org/whl/cu121"
    )

    .pip_install(
        "boto3","transformers","accelerate","trimesh","pillow",
        "einops","omegaconf","xatlas","pyrender","ninja","pybind11",
        "diffusers","pytorch-lightning","huggingface-hub",
        "safetensors","scipy","pandas","opencv-python",
        "imageio","scikit-image","rembg","realesrgan","basicsr",
        "pymeshlab==2022.2.post3","pygltflib","open3d",
        "pyyaml","configargparse","hf-transfer","timm","peft","onnxruntime"
    )

    .run_commands(
        "pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/"
    )

    # 🔥 PATCHED TORCHMCUBES
    .run_commands(
        "git clone https://github.com/tatsy/torchmcubes.git /tmp/torchmcubes",
        "sed -i 's/3.5;//g' /tmp/torchmcubes/CMakeLists.txt || true",
        "sed -i 's/5.0;//g' /tmp/torchmcubes/CMakeLists.txt || true",
        "rm -rf /tmp/torchmcubes/build",
        'cd /tmp/torchmcubes && TORCH_CUDA_ARCH_LIST="8.9" pip install .'
    )

    # HUNYUAN
    .run_commands(
        "rm -rf /root/hunyuan3d && git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /root/hunyuan3d",
        'cd /root/hunyuan3d/hy3dpaint/custom_rasterizer && TORCH_CUDA_ARCH_LIST="8.9" pip install .',
        'cd /root/hunyuan3d/hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh'
    )
)

app = modal.App("fixed-hunyuan-pipeline", image=image)

hunyuan_vol = modal.Volume.from_name("weights-hunyuan-21")
cache_vol = modal.Volume.from_name("ai-factory-cache", create_if_missing=True)

# ==========================================
# PIPELINE
# ==========================================
def generate_3d_from_image(input_img, base_name):
    import torch
    import gc
    from omegaconf import OmegaConf
    import trimesh

    # 🔥 ENSURE CUDA OK
    assert torch.cuda.is_available(), "CUDA not available"
    torch.cuda.init()

    CODE_ROOT = "/root/hunyuan3d"
    WEIGHT_ROOT = "/weights/Hunyuan3D-2.1-Weights-Dataset"

    sys.path.insert(0, CODE_ROOT)
    sys.path.insert(0, os.path.join(CODE_ROOT, 'hy3dshape'))
    sys.path.insert(0, os.path.join(CODE_ROOT, 'hy3dpaint'))
    os.chdir(CODE_ROOT)

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    from convert_utils import create_glb_with_pbr_materials

    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        WEIGHT_ROOT,
        subfolder="hunyuan3d-dit-v2-1",
        device="cuda"
    )

    esrgan_path = "/cache/RealESRGAN.pth"
    if not os.path.exists(esrgan_path):
        os.makedirs("/cache", exist_ok=True)
        urllib.request.urlretrieve(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            esrgan_path
        )

    cfg_path = f"{CODE_ROOT}/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.model.pretrained_model_name_or_path = WEIGHT_ROOT
    OmegaConf.save(cfg, cfg_path)

    conf = Hunyuan3DPaintConfig(max_num_view=8, resolution=768)
    conf.realesrgan_ckpt_path = esrgan_path
    conf.multiview_cfg_path = cfg_path
    conf.custom_pipeline = f"{CODE_ROOT}/hy3dpaint/hunyuanpaintpbr"

    paint_pipeline = Hunyuan3DPaintPipeline(conf)

    outputs = shape_pipeline(image=input_img, output_type='mesh')

    raw_mesh = outputs[0] if isinstance(outputs, list) else outputs
    raw_mesh.mesh_f = raw_mesh.mesh_f[:, ::-1]

    mesh = trimesh.Trimesh(raw_mesh.mesh_v, raw_mesh.mesh_f)

    sid = uuid.uuid4().hex
    base_obj = f"/tmp/{sid}.obj"
    glb = f"/tmp/{base_name}.glb"

    mesh.export(base_obj)

    tex_obj = paint_pipeline(
        mesh_path=base_obj,
        image_path=input_img,
        output_mesh_path=f"/tmp/text_{sid}.obj",
        save_glb=False
    )

    textures = {
        'albedo': tex_obj.replace('.obj', '.jpg'),
        'metallic': tex_obj.replace('.obj', '_metallic.jpg'),
        'roughness': tex_obj.replace('.obj', '_roughness.jpg')
    }

    create_glb_with_pbr_materials(tex_obj, textures, glb)

    del shape_pipeline, paint_pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return glb

# ==========================================
# WORKER
# ==========================================
@app.function(volumes={"/weights": hunyuan_vol, "/cache": cache_vol}, gpu="L4", timeout=3600)
def process_cloudflare_queue(cfg: dict):
    import boto3
    from PIL import Image

    s3 = boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"]
    )

    res = s3.list_objects_v2(Bucket=cfg["bucket"], Prefix="queue/")
    if "Contents" not in res:
        return

    for obj in res["Contents"]:
        key = obj["Key"]
        if not key.endswith((".png",".jpg",".jpeg")):
            continue

        data = s3.get_object(Bucket=cfg["bucket"], Key=key)
        img = Image.open(io.BytesIO(data["Body"].read())).convert("RGBA")

        try:
            glb = generate_3d_from_image(img, key.split("/")[-1].split(".")[0])

            with open(glb, "rb") as f:
                s3.put_object(
                    Bucket=cfg["bucket"],
                    Key=f"output/{os.path.basename(glb)}",
                    Body=f
                )

        except Exception as e:
            print("ERROR:", e)

    cache_vol.commit()


# ==========================================
# ENTRYPOINT (UPDATED WITH YOUR CREDENTIALS)
# ==========================================
@app.local_entrypoint()
def main():
    process_cloudflare_queue.remote({
        "endpoint": "https://4d91f4d3d0366568a54ffa32ffcb7bf4.r2.cloudflarestorage.com",
        "access_key": "3c33425ba6e5abbd3e63afab14dc8866",
        "secret_key": "d65f107bb61093843c6dd980c764443fdf50924a7701078b99f007d3060e25a8",
        "bucket": "video-asset-files-storage-workflow"
    })
