import modal
import os
import sys
import io
import urllib.request
import uuid
import gc

# ==========================================
# IMAGE (STABLE TORCH 2.1.2 + L4 OPTIMIZED)
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

    .apt_install(
        "git","build-essential","clang","cmake","ninja-build",
        "libgl1-mesa-glx","libglib2.0-0","libopengl0","libegl1",
        "libsm6","libxext6","libxrender1","libx11-6","libxi6",
        "libxxf86vm1","libxfixes3","libxkbcommon0", "wget"
    )

    .pip_install("setuptools","wheel")

    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "torchaudio==2.1.2",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    
    .pip_install("deepspeed==0.11.2")

    .pip_install(
        "transformers==4.46.0",
        "accelerate==1.1.1",
        "diffusers==0.30.0"
    )

    .pip_install(
        "boto3","trimesh","pillow","einops","omegaconf","xatlas",
        "pyrender","pybind11","safetensors","scipy","pandas",
        "opencv-python","imageio","scikit-image","rembg",
        "realesrgan","basicsr","pymeshlab==2022.2.post3",
        "pygltflib","pyyaml","configargparse",
        "hf-transfer","timm","peft",
        "pytorch_lightning",
        "fast-simplification",
        "cupy-cuda12x",   
        "torchmetrics",   
        "psutil",         
        "tqdm",
        "nvidia-ml-py"    
    )

    .pip_install("open3d==0.18.0", "onnxruntime==1.16.3")

    .run_commands(
        "pip install --force-reinstall huggingface_hub==0.25.2",
        "pip install --force-reinstall numpy==1.26.4",
        "pip uninstall -y pynvml"
    )

    .run_commands(
        "pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/"
    )

    .run_commands(
        "mkdir -p ~/.u2net",
        "wget https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx -O ~/.u2net/u2net.onnx"
    )

    .run_commands(
        "git clone https://github.com/tatsy/torchmcubes.git /tmp/torchmcubes",
        "sed -i 's/3.5;//g' /tmp/torchmcubes/CMakeLists.txt || true",
        "sed -i 's/5.0;//g' /tmp/torchmcubes/CMakeLists.txt || true",
        'cd /tmp/torchmcubes && TORCH_CUDA_ARCH_LIST="8.9" pip install .'
    )

    .run_commands(
        "rm -rf /root/hunyuan3d && git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /root/hunyuan3d",
        'cd /root/hunyuan3d/hy3dpaint/custom_rasterizer && TORCH_CUDA_ARCH_LIST="8.9" pip install -v .',
        'cd /root/hunyuan3d/hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh'
    )
)

app = modal.App("hunyuan-final-optimized", image=image)

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
    import numpy as np
    from types import SimpleNamespace

    assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"
    print("CUDA is available! Moving to generation...")

    torch.backends.cuda.preferred_linalg_library("default")

    # =================================================================
    # 🕵️ THE RESILIENT LINKER
    # =================================================================
    def resolve_paths():
        candidate_mounts = ["/weights", "/cache", "/mnt", "/root"]
        engine_path, weights_path = None, None

        for mount in candidate_mounts:
            for root, dirs, files in os.walk(mount, topdown=True):
                if root.count(os.sep) - mount.count(os.sep) > 3:
                    del dirs[:]
                    continue
                if "hy3dpaint" in dirs and "hy3dshape" in dirs:
                    engine_path = root
                    break
            if engine_path: break

        for mount in candidate_mounts:
            for root, dirs, files in os.walk(mount, topdown=True):
                if root.count(os.sep) - mount.count(os.sep) > 4:
                    del dirs[:]
                    continue
                if "model.fp16.ckpt" in files and "hunyuan3d-dit-v2-1" in root:
                    weights_path = os.path.dirname(root)
                    break
            if weights_path: break

        if not engine_path: engine_path = "/root/hunyuan3d" 
        if not weights_path: raise FileNotFoundError("Linker Failed: Missing weights.")
        return engine_path, weights_path

    CODE_ROOT, WEIGHT_ROOT = resolve_paths()

    sys.path.insert(0, CODE_ROOT)
    sys.path.insert(0, os.path.join(CODE_ROOT, 'hy3dshape'))
    sys.path.insert(0, os.path.join(CODE_ROOT, 'hy3dpaint'))
    os.chdir(CODE_ROOT)

    from torchvision.transforms import functional as TF
    sys.modules["torchvision.transforms.functional_tensor"] = TF

    # =================================================================
    # 🛠️ MOCK 1: FP16 RMSNorm (High Speed)
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
                
                # Using FP16 natively as requested
                if self.elementwise_affine:
                    self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=torch.float16))
                else:
                    self.register_parameter('weight', None)

            def forward(self, input):
                variance = input.pow(2).mean(-1, keepdim=True)
                input_norm = input * torch.rsqrt(variance + self.eps)
                if self.elementwise_affine:
                    return (self.weight.to(input.dtype) * input_norm)
                return input_norm
        
        nn.RMSNorm = RMSNorm
        import torch.nn.modules.normalization as norm_mod
        norm_mod.RMSNorm = RMSNorm

    # =================================================================
    # 🛠️ MOCK 2: Fast-Simplification
    # =================================================================
    import fast_simplification
    _orig_simplify = fast_simplification.simplify

    def patched_simplify(*args, **kwargs):
        if 'target_reduction' in kwargs and kwargs['target_reduction'] > 1.0:
            kwargs['target_reduction'] = 0.9
        new_args = list(args)
        if len(new_args) >= 3 and new_args[2] > 1.0:
            new_args[2] = 0.9
        return _orig_simplify(*new_args, **kwargs)
    fast_simplification.simplify = patched_simplify

    # =================================================================
    # 🛠️ MOCK 3: THE SHIELDED XATLAS BYPASS (Fixed Return Type)
    # =================================================================
    import xatlas
    
    def patched_parametrize(positions, indices, normals=None, uvs=None, **kwargs):
        print("✅ SUCCESS: Xatlas intercepted. Running 4 iterations in FP32...")
        
        # Strip Tensors to CPU Numpy
        if hasattr(positions, 'detach'): positions = positions.detach().cpu().numpy()
        if hasattr(indices, 'detach'): indices = indices.detach().cpu().numpy()
        
        pos_np = np.asarray(positions, dtype=np.float32)
        ind_np = np.asarray(indices, dtype=np.uint32)
        
        norm_np = None
        if normals is not None:
            if hasattr(normals, 'detach'): normals = normals.detach().cpu().numpy()
            norm_np = np.asarray(normals, dtype=np.float32)
        
        uv_np = None
        if uvs is not None:
            if hasattr(uvs, 'detach'): uvs = uvs.detach().cpu().numpy()
            uv_np = np.asarray(uvs, dtype=np.float32)
        
        atlas = xatlas.Atlas()
        atlas.add_mesh(pos_np, ind_np, norm_np, uv_np)
        
        opts = xatlas.ChartOptions()
        opts.max_iterations = 4
        atlas.generate(chart_options=opts)
        
        # 👑 THE FIX: Return a SimpleNamespace so .vertex_mapping doesn't error out
        m = atlas.get_mesh(0)
        return SimpleNamespace(
            vmapping=m.vmapping,
            vertex_mapping=m.vmapping, 
            indices=m.indices,
            uvs=m.uvs
        )
        
    xatlas.parametrize = patched_parametrize

    # =================================================================
    # STAGE 1: SHAPE GENERATION (FP16 Boosted)
    # =================================================================
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    
    print("Loading Shape Pipeline...")
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        WEIGHT_ROOT,
        subfolder="hunyuan3d-dit-v2-1",
        torch_dtype=torch.float16,
        device="cuda"
    )

    print("Generating base 3D mesh...")
    with torch.autocast('cuda', dtype=torch.float16):
        outputs = shape_pipeline(image=input_img, num_inference_steps=20, output_type='mesh')

    raw_mesh = outputs[0] if isinstance(outputs, list) else outputs
    raw_mesh.mesh_f = raw_mesh.mesh_f[:, ::-1]
    mesh = trimesh.Trimesh(raw_mesh.mesh_v, raw_mesh.mesh_f)

    sid = uuid.uuid4().hex
    base_obj = f"/tmp/{sid}.obj"
    glb = f"/tmp/{base_name}.glb"
    mesh.export(base_obj)

    print("Clearing Shape Model from VRAM...")
    del shape_pipeline
    gc.collect()
    torch.cuda.empty_cache()

    # =================================================================
    # STAGE 2: TEXTURE PAINTING (FP32 Guarded)
    # =================================================================
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    from convert_utils import create_glb_with_pbr_materials

    paint_weight_path = os.path.join(WEIGHT_ROOT, "hunyuan3d-paintpbr-v2-1")
    cfg_path = f"{CODE_ROOT}/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.model.pretrained_model_name_or_path = paint_weight_path
    OmegaConf.save(cfg, cfg_path)

    conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    conf.realesrgan_ckpt_path = "/cache/RealESRGAN.pth"
    conf.multiview_cfg_path = cfg_path
    conf.custom_pipeline = f"{CODE_ROOT}/hy3dpaint/hunyuanpaintpbr"

    print("Loading Paint Pipeline...")
    paint_pipeline = Hunyuan3DPaintPipeline(conf)

    print("Painting textures...")
    
    # 🛡️ THE STABILITY FIX: We explicitly disable autocast here.
    # This prevents the C++ Rasterizer from seeing "Half" precision data.
    with torch.cuda.amp.autocast(enabled=False):
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

    del paint_pipeline
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

    processed_count = 0

    for obj in res["Contents"]:
        key = obj["Key"]
        
        if len(key.split("/")) != 2 or not key.lower().endswith((".png",".jpg",".jpeg")):
            continue

        processed_count += 1
        print(f"\n📥 Processing image: {key}")
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
            print(f"❌ ERROR processing {key}:", e)
            failed_key = key.replace("queue/", "queue/failed/", 1)
            s3.copy_object(
                Bucket=cfg["bucket"],
                CopySource={'Bucket': cfg["bucket"], 'Key': key},
                Key=failed_key
            )
            s3.delete_object(Bucket=cfg["bucket"], Key=key)

    if processed_count == 0:
        print("📭 Checked R2, but found no valid images waiting in the queue/")

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
