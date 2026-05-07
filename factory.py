import modal
import os
import sys
import io
import urllib.request
import uuid
import gc

# ==========================================
# IMAGE (STABLE TORCH 2.1.2 + XFORMERS FIX)
# ==========================================
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")

    .env({
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "CUDA_HOME": "/usr/local/cuda",
        "FORCE_CUDA": "1",
        "MAX_JOBS": "4",
        "ACCELERATE_DISABLE_XPU": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    })

    # 1. SYSTEM DEPENDENCIES
    .apt_install(
        "git","build-essential","clang","cmake","ninja-build",
        "libgl1-mesa-glx","libglib2.0-0","libopengl0","libegl1",
        "libsm6","libxext6","libxrender1","libx11-6","libxi6",
        "libxxf86vm1","libxfixes3","libxkbcommon0", "wget"
    )

    .pip_install("setuptools","wheel")

    # 2. STABLE TORCH (2.1.2 is REQUIRED for diffusers 0.30.0 stability)
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "torchaudio==2.1.2",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    
    # 3. THE FIX: ADD XFORMERS FOR ATTENTION MECHANISM
    .pip_install("xformers==0.0.23.post1")
    
    # 4. DEEPSPEED
    .pip_install("deepspeed==0.11.2")

    # 5. COMPATIBLE HF STACK
    .pip_install(
        "transformers==4.46.0",
        "accelerate==1.1.1",
        "diffusers==0.30.0",
        "huggingface_hub==0.30.2"
    )

    # 6. OTHER LIBS
    .pip_install(
        "boto3","trimesh","pillow","einops","omegaconf","xatlas",
        "pyrender","pybind11","safetensors","scipy","pandas",
        "opencv-python","imageio","scikit-image","rembg",
        "realesrgan","basicsr","pymeshlab==2022.2.post3",
        "pygltflib","pyyaml","configargparse",
        "hf-transfer","timm","peft",
        "pytorch_lightning", "huggingface_hub[hf_xet]"
    )

    # 7. Install problematic libs BEFORE numpy lock
    .pip_install("open3d==0.18.0", "onnxruntime==1.16.3")

    # 8. FINAL NUMPY FIX (Forces 1.26.4 so Open3D doesn't crash)
    .run_commands(
        "pip uninstall -y numpy",
        "pip install numpy==1.26.4",
        "python -c 'import numpy; print(numpy.__version__)'"
    )

    # 9. BLENDER
    .run_commands(
        "pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/"
    )

    # 10. CACHE U2NET
    .run_commands(
        "mkdir -p ~/.u2net",
        "wget https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx -O ~/.u2net/u2net.onnx"
    )

    # 11. TORCHMCUBES
    .run_commands(
        "git clone https://github.com/tatsy/torchmcubes.git /tmp/torchmcubes",
        "sed -i 's/3.5;//g' /tmp/torchmcubes/CMakeLists.txt || true",
        "sed -i 's/5.0;//g' /tmp/torchmcubes/CMakeLists.txt || true",
        'cd /tmp/torchmcubes && TORCH_CUDA_ARCH_LIST="8.9" pip install .'
    )

    # 12. HUNYUAN BUILD
    .run_commands(
        "rm -rf /root/hunyuan3d && git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /root/hunyuan3d",
        'cd /root/hunyuan3d/hy3dpaint/custom_rasterizer && TORCH_CUDA_ARCH_LIST="8.9" pip install -v .',
        'cd /root/hunyuan3d/hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh'
    )
)

# ==========================================
# APP + VOLUMES
# ==========================================
app = modal.App("hunyuan-final-fixed", image=image)

hunyuan_vol = modal.Volume.from_name("weights-hunyuan-21", create_if_missing=True)
cache_vol = modal.Volume.from_name("ai-factory-cache", create_if_missing=True)

# ==========================================
# PIPELINE
# ==========================================
def generate_3d_from_image(input_img, base_name):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    import torch
    import torch.nn as nn
    import gc
    from omegaconf import OmegaConf
    import trimesh
    import sys

    assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"
    print("CUDA is available! Moving to generation...")

    CODE_ROOT = "/root/hunyuan3d"
    WEIGHT_ROOT = "/weights/Hunyuan3D-2.1-Weights-Dataset"

    sys.path.insert(0, CODE_ROOT)
    sys.path.insert(0, os.path.join(CODE_ROOT, 'hy3dshape'))
    sys.path.insert(0, os.path.join(CODE_ROOT, 'hy3dpaint'))
    os.chdir(CODE_ROOT)

    # =================================================================
    # 🛠️ MOCK 1: Torchvision functional_tensor fix
    # =================================================================
    from torchvision.transforms import functional as TF
    sys.modules["torchvision.transforms.functional_tensor"] = TF

    # =================================================================
    # 🛠️ MOCK 2: RMSNorm Polyfill for PyTorch 2.1.2 (STOPS THE CRASH)
    # =================================================================
    if not hasattr(nn, 'RMSNorm'):
        class RMSNorm(nn.Module):
            def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
                super().__init__()
                if isinstance(normalized_shape, int):
                    normalized_shape = (normalized_shape,)
                self.normalized_shape = tuple(normalized_shape)
                self.eps = eps
                self.elementwise_affine = elementwise_affine
                if self.elementwise_affine:
                    self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
                else:
                    self.register_parameter('weight', None)

            def forward(self, input):
                input_dtype = input.dtype
                input_fp32 = input.to(torch.float32)
                variance = input_fp32.pow(2).mean(-1, keepdim=True)
                input_norm = input_fp32 * torch.rsqrt(variance + self.eps)
                if self.elementwise_affine:
                    return (self.weight * input_norm).to(input_dtype)
                return input_norm.to(input_dtype)
        
        # Inject it into PyTorch's brain
        nn.RMSNorm = RMSNorm
        import torch.nn.modules.normalization as norm_mod
        norm_mod.RMSNorm = RMSNorm
    # =================================================================

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    from convert_utils import create_glb_with_pbr_materials

    print("Loading Shape Pipeline...")
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        WEIGHT_ROOT,
        subfolder="hunyuan3d-dit-v2-1",
        device="cuda"
    )
    
    # Force Xformers for memory-efficient attention (fixes the 0% freeze)
    shape_pipeline.enable_xformers_memory_efficient_attention()

    esrgan_path = "/cache/RealESRGAN.pth"
    if not os.path.exists(esrgan_path):
        print("Downloading RealESRGAN...")
        os.makedirs("/cache", exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            esrgan_path
        )

    # =================================================================
    # 🛠️ MOCK 3: Paint Pipeline Path Fix (STOPS THE 19 FILE DOWNLOAD)
    # =================================================================
    # We must point directly to the paint subfolder so it sees the local files
    paint_weight_path = os.path.join(WEIGHT_ROOT, "hunyuan3d-paintpbr-v2-1")
    
    cfg_path = f"{CODE_ROOT}/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.model.pretrained_model_name_or_path = paint_weight_path
    OmegaConf.save(cfg, cfg_path)

    conf = Hunyuan3DPaintConfig(max_num_view=8, resolution=768)
    conf.realesrgan_ckpt_path = esrgan_path
    conf.multiview_cfg_path = cfg_path
    conf.custom_pipeline = f"{CODE_ROOT}/hy3dpaint/hunyuanpaintpbr"

    print("Loading Paint Pipeline...")
    paint_pipeline = Hunyuan3DPaintPipeline(conf)

    print("Generating base 3D mesh...")
    outputs = shape_pipeline(image=input_img, output_type='mesh')

    raw_mesh = outputs[0] if isinstance(outputs, list) else outputs
    raw_mesh.mesh_f = raw_mesh.mesh_f[:, ::-1]

    mesh = trimesh.Trimesh(raw_mesh.mesh_v, raw_mesh.mesh_f)

    import uuid
    sid = uuid.uuid4().hex
    base_obj = f"/tmp/{sid}.obj"
    glb = f"/tmp/{base_name}.glb"

    mesh.export(base_obj)

    print("Painting textures...")
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

    print("Combining materials into GLB...")
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
    import io
    import os

    print("Connecting to Cloudflare R2...")
    s3 = boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"]
    )

    res = s3.list_objects_v2(Bucket=cfg["bucket"], Prefix="queue/")
    if "Contents" not in res:
        print("Queue is empty.")
        return

    for obj in res["Contents"]:
        key = obj["Key"]
        
        # FIX FOR INFINITE LOOP
        if len(key.split("/")) != 2 or not key.endswith((".png",".jpg",".jpeg")):
            continue

        print(f"Processing image: {key}")
        data = s3.get_object(Bucket=cfg["bucket"], Key=key)
        img = Image.open(io.BytesIO(data["Body"].read())).convert("RGBA")

        try:
            glb = generate_3d_from_image(img, key.split("/")[-1].split(".")[0])

            print(f"Uploading 3D model {glb} back to R2...")
            with open(glb, "rb") as f:
                s3.put_object(
                    Bucket=cfg["bucket"],
                    Key=f"output/{os.path.basename(glb)}",
                    Body=f
                )
            print("Successfully uploaded! Removing original from queue...")
            
            s3.delete_object(Bucket=cfg["bucket"], Key=key)

        except Exception as e:
            print(f"ERROR processing {key}:", e)
            
            failed_key = key.replace("queue/", "queue/failed/", 1)
            print(f"Moving {key} to {failed_key}...")
            s3.copy_object(
                Bucket=cfg["bucket"],
                CopySource={'Bucket': cfg["bucket"], 'Key': key},
                Key=failed_key
            )
            s3.delete_object(Bucket=cfg["bucket"], Key=key)

    cache_vol.commit()

# ==========================================
# ENTRYPOINT
# ==========================================
@app.local_entrypoint()
def main():
    process_cloudflare_queue.remote({
        "endpoint": "https://4d91f4d3d0366568a54ffa32ffcb7bf4.r2.cloudflarestorage.com",
        "access_key": "3c33425ba6e5abbd3e63afab14dc8866",
        "secret_key": "d65f107bb61093843c6dd980c764443fdf50924a7701078b99f007d3060e25a8",
        "bucket": "video-asset-files-storage-workflow"
    })
