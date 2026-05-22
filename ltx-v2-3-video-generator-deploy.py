import modal
import subprocess
import time
import os
import json
import shutil
import threading
import aiohttp
import urllib.request
import asyncio
import ctypes
import uuid
from fastapi import Request, Response, HTTPException, Header
from typing import Optional

# ==========================================
# PART 1: Infrastructure Configuration & Base Image
# Purpose: Defines the core operating system, environment variables, system-level dependencies, and the base python libraries needed for video processing and AI model inference.
# ==========================================

R2_ACCOUNT_ID = "4d91f4d3d0366568a54ffa32ffcb7bf4"
R2_ACCESS_KEY_ID = "3c33425ba6e5abbd3e63afab14dc8866"
R2_SECRET_ACCESS_KEY = "d65f107bb61093843c6dd980c764443fdf50924a7701078b99f007d3060e25a8"
R2_ENDPOINT_URL = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

base_image = modal.Image.from_registry(
    "nvidia/cuda:12.5.1-devel-ubuntu24.04", 
    add_python="3.12"
).apt_install(
    "git", "wget", "ffmpeg", "libgl1", "libglib2.0-0", 
    "build-essential", "ninja-build", "cmake", "clang", "llvm"
)

build_image = base_image.env({
    "CUDA_HOME": "/usr/local/cuda",
    "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", ""),
    "FORCE_CUDA": "1",
    "TORCH_CUDA_ARCH_LIST": "8.9", 
    "MAX_JOBS": "4",
    "CC": "gcc",
    "CXX": "g++",
    "R2_ACCOUNT_ID": R2_ACCOUNT_ID,
    "R2_ACCESS_KEY_ID": R2_ACCESS_KEY_ID,
    "R2_SECRET_ACCESS_KEY": R2_SECRET_ACCESS_KEY
}).pip_install(
    "fastapi", "aiohttp", "boto3", "triton>=3.1.0", 
    "ninja", "setuptools>=70.0.0", "wheel", "pip>=24.0"
).pip_install(
    "pandas", "numexpr", "pytz", "python-dateutil", 
    "scipy", "matplotlib", "colorama", "librosa", "soundfile", 
    "decord", "imageio", "scikit-image", "numba", "einops", 
    "transformers", "diffusers", "accelerate", "bitsandbytes"
)

TARGET_UNET = "ltx-2.3-22b-dev-fp8.safetensors"
TARGET_GEMMA = "gemma_3_12B_it_fp8_scaled.safetensors"
TARGET_CONNECTOR = "ltx-2.3_text_projection_bf16.safetensors"
TARGET_VIDEO_VAE = "LTX23_video_vae_bf16.safetensors"
TARGET_AUDIO_VAE = "LTX23_audio_vae_bf16.safetensors"

TARGET_DISTILLED_LORA = "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors"
TARGET_DETAILER_LORA = "ltx-2-19b-ic-lora-detailer.safetensors"

# ==========================================
# PART 2: Topological Graph Analyzer & Build-Time Appliance Baker
# Purpose: Pre-downloads the master workflow, injects the Execution Fence logic, and routes explicit model pathways into the prebaked appliance image.
# ==========================================

def bake_private_workflow_into_image():
    import boto3
    import json
    import os

    print("🏗️ Build Phase: Securely fetching master workflow from private Cloudflare R2...")
    
    s3 = boto3.client(
        service_name='s3', 
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com", 
        aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'], 
        aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], 
        region_name="auto"
    )

    raw_path = "/tmp/raw_workflow.json"
    try:
        s3.download_file(
            "video-asset-files-storage-workflow", 
            "Comfyui-workflows-json/ltx-23-new-workflow-modified(api)new.json", 
            raw_path
        )

        with open(raw_path, "r") as f:
            wf_data = json.load(f)

        if "workflow" in wf_data and not any("class_type" in v for v in wf_data.values() if isinstance(v, dict)):
            wf_data = wf_data["workflow"]

        print("🏗️ Build Phase: Executing Topological Graph Tracing for Multi-LoRA & Execution Fence Injections...")
        
        unet_nodes = []
        lora_nodes = []
        text_loader_id = None
        for node_id, node in wf_data.items():
            if not isinstance(node, dict): continue
            cls = node.get("class_type", "")
            if cls in ["UNETLoader", "UnetLoaderGGUFAdvanced", "CheckpointLoaderSimple", "LowVRAMUNETLoader"]:
                unet_nodes.append(node_id)
            elif cls == "LoraLoader":
                lora_nodes.append(node_id)
            elif cls == "LTXAVTextEncoderLoader":
                text_loader_id = node_id

        assignments = {}
        if unet_nodes and len(lora_nodes) >= 2:
            unet_id = unet_nodes[0]
            first_lora, second_lora = None, None
            for l_id in lora_nodes:
                model_input = wf_data[l_id].get("inputs", {}).get("model")
                if isinstance(model_input, list) and len(model_input) > 0 and str(model_input[0]) == str(unet_id):
                    first_lora = l_id
                    break
            if first_lora:
                for l_id in lora_nodes:
                    if l_id == first_lora: continue
                    model_input = wf_data[l_id].get("inputs", {}).get("model")
                    if isinstance(model_input, list) and len(model_input) > 0 and str(model_input[0]) == str(first_lora):
                        second_lora = l_id
                        break
                        
            if first_lora and second_lora:
                print(f"🎯 Traced Connection Pathway: UNET -> Node {first_lora} (Rank111 Distill) -> Node {second_lora} (19b Detailer)")
                assignments[first_lora] = TARGET_DISTILLED_LORA
                assignments[second_lora] = TARGET_DETAILER_LORA
        
        for l_id in lora_nodes:
            if l_id not in assignments:
                node = wf_data[l_id]
                lora_name = str(node.get("inputs", {}).get("lora_name", "")).lower()
                if "detail" in lora_name or "ic-lora" in lora_name or l_id == "229":
                    assignments[l_id] = TARGET_DETAILER_LORA
                else:
                    assignments[l_id] = TARGET_DISTILLED_LORA

        for node_id, node in wf_data.items():
            if not isinstance(node, dict) or "inputs" not in node:
                continue
                
            cls = node.get("class_type", "")
            inputs = node["inputs"]
            
            # ⚡ Injection of the LowVRAM Execution Fence 
            if cls in ["UNETLoader", "UnetLoaderGGUFAdvanced", "CheckpointLoaderSimple", "LowVRAMUNETLoader"]:
                node["class_type"] = "LowVRAMUNETLoader"
                inputs["unet_name"] = TARGET_UNET
                inputs["ckpt_name"] = TARGET_UNET
                if "weight_dtype" not in inputs:
                    inputs["weight_dtype"] = "default"
                if "widgets_values" in node and isinstance(node["widgets_values"], list) and len(node["widgets_values"]) > 0:
                    node["widgets_values"][0] = TARGET_UNET
                if text_loader_id:
                    inputs["dependencies"] = [text_loader_id, 0] # Forces text encoder to clear before NVMe mapping

            elif cls == "LTXAVTextEncoderLoader":
                inputs["text_encoder"] = TARGET_GEMMA
                inputs["ckpt_name"] = TARGET_CONNECTOR
            elif cls in ["VAELoaderKJ", "VAELoader"]:
                inputs["vae_name"] = TARGET_VIDEO_VAE
                inputs["ckpt_name"] = TARGET_VIDEO_VAE
            elif cls == "LTXVAudioVAELoader":
                inputs["ckpt_name"] = TARGET_AUDIO_VAE
                inputs["vae_name"] = TARGET_AUDIO_VAE
            elif cls == "LoraLoader":
                resolved = assignments.get(node_id, TARGET_DISTILLED_LORA)
                inputs["lora_name"] = resolved
                if "widgets_values" in node and isinstance(node["widgets_values"], list) and len(node["widgets_values"]) > 0:
                    node["widgets_values"][0] = resolved
            elif cls == "DenoMultiImageLoader":
                inputs["image_paths"] = "input/dynamic_guides"

        os.makedirs("/workspace", exist_ok=True)
        with open("/workspace/prebaked_workflow.json", "w") as f:
            json.dump(wf_data, f, indent=2)
            
        print("🏗️ Build Phase: Successfully sealed prebaked_workflow.json to container disk!")
    except Exception as e:
        print(f"⚠️ Build Phase Issue (Fallback Skipped): {e}")

# ==========================================
# PART 3: Advanced Optimization Patches & Custom Node Installation
# Purpose: Applies necessary backwards-compatibility patches, establishes the 2GB VRAM reserve rule, and deploys the Execution Fence node directly into the application space.
# ==========================================

def patch_ltx_video_imports():
    import os
    init_path = "/workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/__init__.py"
    if os.path.exists(init_path):
        with open(init_path, "r") as f:
            content = f.read()
        patch_code = (
            "import sys\n"
            "import torch\n"
            "import math\n"
            "try:\n"
            "    import comfy.ldm.lightricks.model as m\n"
            "    if not hasattr(m, 'precompute_freqs_cis'):\n"
            "        def precompute_freqs_cis(coords, dim, out_dtype):\n"
            "            try:\n"
            "                if hasattr(m, 'LTXBaseModel'):\n"
            "                    for name in ['_precompute_freqs_cis', 'precompute_freqs_cis']:\n"
            "                        if hasattr(m.LTXBaseModel, name):\n"
            "                            return getattr(m.LTXBaseModel, name)(coords, dim, out_dtype)\n"
            "            except Exception: pass\n"
            "            theta = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=coords.device, dtype=torch.float32) / dim))\n"
            "            freqs = torch.einsum('... , d -> ... d', coords.flatten(1, -2).to(torch.float32), theta)\n"
            "            freqs = freqs.view(*coords.shape[:-1], -1)\n"
            "            return torch.polar(torch.ones_like(freqs), freqs.to(out_dtype))\n"
            "        m.precompute_freqs_cis = precompute_freqs_cis\n"
            "except Exception as e:\n"
            "    print(f'⚠️ Warning: LTX-Video import monkeypatch failed: {e}')\n"
        )
        with open(init_path, "w") as f:
            f.write(patch_code + content)
        print("🔧 Successfully applied LTXVideo backwards-compatibility monkey-patch for precompute_freqs_cis.")

def patch_comfy_lightricks_model():
    import os
    model_path = "/workspace/ComfyUI/comfy/ldm/lightricks/model.py"
    if os.path.exists(model_path):
        with open(model_path, "r") as f:
            content = f.read()
        if "def precompute_freqs_cis" not in content:
            print("🔧 Patching comfy/ldm/lightricks/model.py to add backward compatibility helpers...")
            fallback_code = """
# --- LTX-Video/LTX-2.3 Compatibility Patch ---
import torch
import math

def precompute_freqs_cis(coords, dim, out_dtype):
    try:
        if "LTXBaseModel" in globals():
            base_model = globals()["LTXBaseModel"]
            for method_name in ["_precompute_freqs_cis", "precompute_freqs_cis"]:
                if hasattr(base_model, method_name):
                    return getattr(base_model, method_name)(coords, dim, out_dtype)
    except Exception:
        pass
        
    theta = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=coords.device, dtype=torch.float32) / dim))
    freqs = torch.einsum('... , d -> ... d', coords.flatten(1, -2).to(torch.float32), theta)
    freqs = freqs.view(*coords.shape[:-1], -1)
    return torch.polar(torch.ones_like(freqs), freqs.to(out_dtype))

def apply_rotary_emb(x, freqs):
    try:
        if "LTXBaseModel" in globals():
            base_model = globals()["LTXBaseModel"]
            for name in ["_apply_rotary_emb", "apply_rotary_emb"]:
                if hasattr(base_model, name):
                    return getattr(base_model, name)(x, freqs)
    except Exception:
        pass
    from comfy.ldm.flux.layers import apply_rotary_emb as flux_apply_rope
    try:
        return flux_apply_rope(x, freqs)
    except Exception:
        pass
    return x

globals()["precompute_freqs_cis"] = precompute_freqs_cis
globals()["apply_rotary_emb"] = apply_rotary_emb
"""
            with open(model_path, "w") as f:
                f.write(content + fallback_code)
            print("✅ Successfully patched comfy/ldm/lightricks/model.py!")

def patch_ltx_kornia_pad():
    import os
    pyramid_blending_path = "/workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/pyramid_blending.py"
    if os.path.exists(pyramid_blending_path):
        with open(pyramid_blending_path, "r") as f:
            lines = f.readlines()
        print("🔧 Patching pyramid_blending.py to resolve kornia pad Import Error...")
        new_lines = []
        in_pyramid_import = False
        patched = False
        for line in lines:
            if "from kornia.geometry.transform.pyramid import" in line:
                in_pyramid_import = True
                new_lines.append(line)
                continue
            if in_pyramid_import:
                if ")" in line:
                    in_pyramid_import = False
                import re
                if re.search(r'\bpad\b', line):
                    patched = True
                    line = re.sub(r'\bpad\b\s*,?', '', line)
                    if line.strip() == "," or not line.strip():
                        continue
            new_lines.append(line)
        if patched:
            new_lines.insert(0, "from torch.nn.functional import pad\n")
            with open(pyramid_blending_path, "w") as f:
                f.writelines(new_lines)
            print("✅ Successfully patched pyramid_blending.py!")

def patch_comfyui_model_management():
    import os
    mm_path = "/workspace/ComfyUI/comfy/model_management.py"
    if os.path.exists(mm_path):
        with open(mm_path, "a") as f:
            f.write("\n# --- 2.0 GB VRAM Reservation Patch ---\n")
            f.write("import sys\n")
            f.write("_orig_get_free_memory = get_free_memory\n")
            f.write("def get_free_memory(dev=None, torch_free_too=False):\n")
            f.write("    free_mem = _orig_get_free_memory(dev, torch_free_too)\n")
            f.write("    reserve_bytes = 2 * 1024 * 1024 * 1024  # Force strict 2.0 GB Reserve for NVMe mapping safety\n")
            f.write("    return max(0, free_mem - reserve_bytes)\n")
        print("✅ Successfully injected 2.0 GB VRAM Reservation into ComfyUI allocator!")

def install_execution_fence_node():
    import os
    os.makedirs("/workspace/ComfyUI/custom_nodes", exist_ok=True)
    node_path = "/workspace/ComfyUI/custom_nodes/LowVRAM_Execution_Fence.py"
    code = """
import folder_paths
import nodes
import gc
import torch
import ctypes

class LowVRAMUNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                              "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],) },
                "optional": { "dependencies": ("*", )}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "advanced/loaders"

    def load_unet(self, unet_name, weight_dtype="default", dependencies=None):
        print("🛡️ Execution Fence Triggered! Flushing text-encoder memory before NVMe streaming UNet...")
        gc.collect()
        torch.cuda.empty_cache()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        return nodes.UNETLoader().load_unet(unet_name, weight_dtype)

NODE_CLASS_MAPPINGS = { "LowVRAMUNETLoader": LowVRAMUNETLoader }
NODE_DISPLAY_NAME_MAPPINGS = { "LowVRAMUNETLoader": "🛡️ LowVRAM Execution Fence UNET Loader" }
"""
    with open(node_path, "w") as f:
        f.write(code)
    print("✅ Successfully installed custom LowVRAM Execution Fence Node!")

final_image = (
    build_image.pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "numpy==1.26.4", "diffusers", "accelerate", "transformers", 
        "comfyui-workflow-templates", "peft"
    )
    .run_commands(
        "git clone https://github.com/comfyanonymous/ComfyUI /workspace/ComfyUI # cache_bust=2026_05_23_v4",
        "pip install -r /workspace/ComfyUI/requirements.txt # cache_bust=2026_05_23_v4"
    )
    .run_commands(
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite # cache_bust=2026_05_23_v4",
        "git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo # cache_bust=2026_05_23_v4",
        "git clone https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes # cache_bust=2026_05_23_v4",
        "git clone https://github.com/yolain/ComfyUI-Easy-Use.git /workspace/ComfyUI/custom_nodes/ComfyUI-Easy-Use # cache_bust=2026_05_23_v4",
        "git clone https://github.com/Deno2026/comfyui-deno-custom-nodes.git /workspace/ComfyUI/custom_nodes/comfyui-deno-custom-nodes # cache_bust=2026_05_23_v4",
        "git clone https://github.com/cubiq/ComfyUI_essentials.git /workspace/ComfyUI/custom_nodes/ComfyUI_essentials # cache_bust=2026_05_23_v4",
        "git clone https://github.com/FizzleDorf/ComfyUI_FizzNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI_FizzNodes # cache_bust=2026_05_23_v4",
        "git clone https://github.com/SquirrelRat/MultiString-Prompts.git /workspace/ComfyUI/custom_nodes/MultiString-Prompts # cache_bust=2026_05_23_v4",
        "git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git /workspace/ComfyUI/custom_nodes/ComfyUI-Custom-Scripts # cache_bust=2026_05_23_v4",
        "git clone https://github.com/IvanRybakov/comfyui-node-int-to-string-convertor.git /workspace/ComfyUI/custom_nodes/comfyui-node-int-to-string-convertor # cache_bust=2026_05_23_v4",
        "git clone https://github.com/siraxe/ComfyUI-LTX-FDG.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTX-FDG # cache_bust=2026_05_23_v4"
    )
    .run_commands(
        "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/requirements.txt # cache_bust=2026_05_23_v4",
        "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt # cache_bust=2026_05_23_v4"
    )
    .run_commands(
        "pip install kornia==0.6.12 # cache_bust=2026_05_23_v4"
    )
    .run_commands(
        "sed -i 's/final_pooled_output = torch.cat(pooled_out, dim=0)/final_pooled_output = torch.cat([p for p in pooled_out if p is not None], dim=0) if any(p is not None for p in pooled_out) else None/g' /workspace/ComfyUI/custom_nodes/ComfyUI_FizzNodes/BatchFuncs.py"
    )
    .run_function(patch_comfy_lightricks_model)
    .run_function(patch_ltx_video_imports)
    .run_function(patch_ltx_kornia_pad)
    .run_function(install_execution_fence_node)
    .run_function(patch_comfyui_model_management)
    .run_function(bake_private_workflow_into_image)
)

# ==========================================
# PART 4: Production Class Definition & Resource Reclamation Loops
# Purpose: Manages the Modal lifecycle, attaches the dual storage volumes, prevents validation failures, and initiates background garbage collection to maintain stable System RAM thresholds.
# ==========================================

app = modal.App("ltx-2-3-v20-api")

weights_volume_23 = modal.Volume.from_name("LTX-2.3-Model-Weights", create_if_missing=True)
weights_volume_19 = modal.Volume.from_name("ltx-20-19b-weights", create_if_missing=True)

@app.cls(
    gpu="L4", 
    image=final_image, 
    volumes={
        "/mnt/weights_23": weights_volume_23,
        "/mnt/weights_19": weights_volume_19
    },
    secrets=[modal.Secret.from_dict({
        "R2_ACCOUNT_ID": R2_ACCOUNT_ID,
        "R2_ACCESS_KEY_ID": R2_ACCESS_KEY_ID,
        "R2_SECRET_ACCESS_KEY": R2_SECRET_ACCESS_KEY,
        "API_KEY": "secure-video-n8n-workflow-2026"
    })],
    memory=8192,  
    scaledown_window=5,  
    timeout=3600
)
class LTXEngine:
    def _log_reader(self):
        for line in iter(self.process.stdout.readline, ""):
            if line:
                print(f"[ComfyUI] {line.strip()}")

    async def _ram_squeezer(self):
        print("🛡️ RAM Watchdog Active. Forcing Linux to drop page cache...")
        while True:
            try:
                with open('/proc/sys/vm/drop_caches', 'w') as f:
                    f.write('1\n')
            except Exception:
                pass
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
            await asyncio.sleep(2)

    def patch_comfyui_main_ram(self):
        main_path = "/workspace/ComfyUI/main.py"
        if os.path.exists(main_path):
            with open(main_path, "r") as f:
                content = f.read()
            
            mock_code = """
import psutil
def mock_virtual_memory():
    class MockVM:
        def __init__(self):
            self.total = 6 * 1024 * 1024 * 1024  # Force limit to 6 GB
            self.available = 4 * 1024 * 1024 * 1024  # Force limit to 4 GB
            self.percent = 33.3
            self.used = 2 * 1024 * 1024 * 1024
            self.free = 4 * 1024 * 1024 * 1024
    return MockVM()
psutil.virtual_memory = mock_virtual_memory
print("🔧 [Patched] System RAM mocked to 6GB to force aggressive disk-streaming & disable pinned CPU memory allocations!")

# --- Guider raw_conds backward-compatibility patch ---
try:
    import threading
    import time
    def background_patcher():
        try:
            patched = False
            while not patched:
                import sys
                if "comfy.samplers" in sys.modules or "comfy.guiders" in sys.modules:
                    import comfy.samplers
                    import comfy.guiders
                    
                    modules_to_patch = []
                    if "comfy.samplers" in sys.modules:
                        modules_to_patch.append(sys.modules["comfy.samplers"])
                    if "comfy.guiders" in sys.modules:
                        modules_to_patch.append(sys.modules["comfy.guiders"])
                        
                    patched_classes = set()
                    for module in modules_to_patch:
                        for name in dir(module):
                            obj = getattr(module, name)
                            if isinstance(obj, type) and hasattr(obj, "set_conds"):
                                if obj in patched_classes:
                                    continue
                                orig_set_conds = obj.set_conds
                                if getattr(orig_set_conds, "__name__", "") == "patched_set_conds":
                                    continue
                                    
                                def make_patched_set_conds(original_method):
                                    def patched_set_conds(self, *args, **kwargs):
                                        pos = args[0] if len(args) >= 1 else None
                                        neg = args[1] if len(args) >= 2 else None
                                        if "positive" in kwargs:
                                            pos = kwargs["positive"]
                                        if "negative" in kwargs:
                                            neg = kwargs["negative"]
                                        self.raw_conds = (pos, neg)
                                        return original_method(self, *args, **kwargs)
                                    return patched_set_conds
                                    
                                obj.set_conds = make_patched_set_conds(orig_set_conds)
                                patched_classes.add(obj)
                                print(f"🔧 [Background Patched] preserved raw_conds on class {obj.__name__} in module {module.__name__}.")
                    patched = True
                time.sleep(0.1)
        except Exception as e:
            print(f"⚠️ Warning: background patcher failed: {e}")
            
    threading.Thread(target=background_patcher, daemon=True).start()
except Exception as e:
    print(f"⚠️ Warning: failed starting background patcher thread: {e}")
"""
            if "mock_virtual_memory" not in content:
                with open(main_path, "w") as f:
                    f.write(mock_code + "\n" + content)
                print("✅ Successfully injected virtual RAM and raw_conds patches into ComfyUI's main.py!")

    @modal.enter()
    def start_comfy(self):
        import boto3
        print("🔗 Running Atomic Model Folder Linker (Multi-Volume)...")
        base_models_dir = "/workspace/ComfyUI/models"
        
        dirs = ["unet", "vae", "clip", "text_encoders", "text_encoder", "checkpoints", "diffusion_models", "gguf", "loras"]
        for d in dirs:
            os.makedirs(os.path.join(base_models_dir, d), exist_ok=True)

        exact_mapping = {
            "gemma_3_12B_it_fp8_scaled.safetensors": ["text_encoders"],
            "ltx-2.3_text_projection_bf16.safetensors": ["checkpoints"],
            "ltx-2.3-22b-dev-fp8.safetensors": ["unet", "diffusion_models"],
            "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors": ["loras"],
            "ltx-2-19b-ic-lora-detailer.safetensors": ["loras"],
            "LTX23_audio_vae_bf16.safetensors": ["checkpoints", "vae"],
            "LTX23_video_vae_bf16.safetensors": ["vae"]
        }

        for mount_point in ["/mnt/weights_23", "/mnt/weights_19"]:
            if os.path.exists(mount_point):
                for root_dir, _, files in os.walk(mount_point):
                    for filename in files:
                        if filename in exact_mapping:
                            src_path = os.path.join(root_dir, filename)
                            for target_dir in exact_mapping[filename]:
                                dest = os.path.join(base_models_dir, target_dir, filename)
                                os.makedirs(os.path.dirname(dest), exist_ok=True)
                                if not os.path.exists(dest):
                                    try:
                                        os.symlink(src_path, dest)
                                        print(f"🔗 Linked: {filename} -> models/{target_dir}")
                                    except FileExistsError:
                                        pass
                                    except Exception as e:
                                        print(f"⚠️ Failed linking {filename}: {e}")

        self.patch_comfyui_main_ram()

        self.s3 = boto3.client(
            service_name='s3', 
            endpoint_url=R2_ENDPOINT_URL, 
            aws_access_key_id=R2_ACCESS_KEY_ID, 
            aws_secret_access_key=R2_SECRET_ACCESS_KEY, 
            region_name="auto"
        )

        print("🚀 Launching Clean LTX-2.3 NVMe-Streaming Server Engine...")
        os.makedirs("/tmp/comfy_swap", exist_ok=True)
        os.makedirs("/tmp/hf_offload", exist_ok=True)

        env_vars = os.environ.copy()
        env_vars["TORCH_NUM_THREADS"] = "1"
        env_vars["OMP_NUM_THREADS"] = "1"
        env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
        env_vars["CUDA_MODULE_LOADING"] = "LAZY" 
        env_vars["MALLOC_TRIM_THRESHOLD_"] = "65536" 
        env_vars["HF_HUB_OFFLOAD_DIR"] = "/tmp/hf_offload"
        
        # ⚡ Direct Hard Drive mapping configured via flags
        self.process = subprocess.Popen([
            "python", "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--mmap-torch-files", "--cache-none", "--temp-directory", "/tmp/comfy_swap", 
            "--bf16-vae", "--disable-xformers", "--fp8_e4m3fn-text-enc", "--lowvram",
            "--disable-pinned-memory"        
        ], cwd="/workspace/ComfyUI", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env_vars)
        
        self.t = threading.Thread(target=self._log_reader, daemon=True)
        self.t.start()

        asyncio.run_coroutine_threadsafe(self._ram_squeezer(), asyncio.get_event_loop())

        start_time = time.time()
        while time.time() - start_time < 300:
            if self.process.poll() is not None:
                os._exit(1)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8188/", timeout=1) as response:
                    if response.status == 200:
                        print("⚡ LTX-2.3 API ONLINE!")
                        return
            except Exception:
                time.sleep(2)
        os._exit(1)

# ==========================================
# PART 5: Hybrid Endpoint Handler & Dynamic Parameter Override
# Purpose: Intercepts POST requests, modifies the JSON payload dynamically to insert the Execution Fence, and applies 2.3 constraints to maximize performance.
# ==========================================

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, body: dict, x_api_key: str = Header(None)):
        import aiohttp
        import json
        import os
        import uuid
        import shutil
        import torch
        import gc
        from urllib.parse import urlparse

        if x_api_key != os.environ.get("API_KEY"):
            raise HTTPException(status_code=403, detail="Unauthorized")

        image_url = body.get("image_url")
        workflow_str = body.get("workflow")
        requested_length = body.get("length", 65)
        filename_prefix = body.get("filename", "output_video")

        if not workflow_str:
            raise HTTPException(status_code=400, detail="Missing 'workflow' in request body")

        if isinstance(workflow_str, str):
            try:
                wf_data = json.loads(workflow_str)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}")
        else:
            wf_data = workflow_str

        dynamic_guides_dir = "/workspace/ComfyUI/input/dynamic_guides"
        if os.path.exists(dynamic_guides_dir):
            shutil.rmtree(dynamic_guides_dir)
        os.makedirs(dynamic_guides_dir, exist_ok=True)

        if image_url:
            print(f"📥 Downloading dynamic guide image: {image_url}")
            ext = os.path.splitext(image_url.split("?")[0])[1] or ".png"
            target_image_path = os.path.join(dynamic_guides_dir, f"guide_image{ext}")
            
            if "r2.dev" in image_url or R2_ACCOUNT_ID in image_url:
                parsed_url = urlparse(image_url)
                key = parsed_url.path.lstrip("/")
                try:
                    self.s3.download_file("video-asset-files-storage-workflow", key, target_image_path)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"S3 R2 download failure: {e}")
            else:
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url) as resp:
                        if resp.status == 200:
                            with open(target_image_path, "wb") as f:
                                f.write(await resp.read())
                        else:
                            raise HTTPException(status_code=400, detail=f"Failed to download guide image. HTTP {resp.status}")

        unet_nodes = []
        lora_nodes = []
        text_loader_id = None
        for node_id, node in wf_data.items():
            if not isinstance(node, dict): continue
            cls = node.get("class_type", "")
            if cls in ["UNETLoader", "UnetLoaderGGUFAdvanced", "CheckpointLoaderSimple", "LowVRAMUNETLoader"]:
                unet_nodes.append(node_id)
            elif cls == "LoraLoader":
                lora_nodes.append(node_id)
            elif cls == "LTXAVTextEncoderLoader":
                text_loader_id = node_id

        assignments = {}
        if unet_nodes and len(lora_nodes) >= 2:
            unet_id = unet_nodes[0]
            first_lora, second_lora = None, None
            
            for l_id in lora_nodes:
                model_input = wf_data[l_id].get("inputs", {}).get("model")
                if isinstance(model_input, list) and len(model_input) > 0 and str(model_input[0]) == str(unet_id):
                    first_lora = l_id
                    break
            
            if first_lora:
                for l_id in lora_nodes:
                    if l_id == first_lora: continue
                    model_input = wf_data[l_id].get("inputs", {}).get("model")
                    if isinstance(model_input, list) and len(model_input) > 0 and str(model_input[0]) == str(first_lora):
                        second_lora = l_id
                        break
                        
            if first_lora and second_lora:
                assignments[first_lora] = TARGET_DISTILLED_LORA
                assignments[second_lora] = TARGET_DETAILER_LORA
        
        for l_id in lora_nodes:
            if l_id not in assignments:
                node = wf_data[l_id]
                lora_name = str(node.get("inputs", {}).get("lora_name", "")).lower()
                if "detail" in lora_name or "ic-lora" in lora_name or l_id == "229":
                    assignments[l_id] = TARGET_DETAILER_LORA
                else:
                    assignments[l_id] = TARGET_DISTILLED_LORA

        tgt_len = int(requested_length)
        if (tgt_len - 1) % 16 != 0:
            tgt_len = ((tgt_len - 1) // 16) * 16 + 1
            if tgt_len < 17:
                tgt_len = 17

        for node_id, node in wf_data.items():
            if not isinstance(node, dict) or "inputs" not in node:
                continue
                
            cls = node.get("class_type", "")
            inputs = node["inputs"]
            
            # ⚡ The Dynamic API Injection: Intercepts the standard loader, changes the class to our custom Fence, and attaches the text encoder dependency so they stream sequentially.
            if cls in ["UNETLoader", "UnetLoaderGGUFAdvanced", "CheckpointLoaderSimple", "LowVRAMUNETLoader"]:
                node["class_type"] = "LowVRAMUNETLoader"
                inputs["unet_name"] = TARGET_UNET
                inputs["ckpt_name"] = TARGET_UNET
                if "weight_dtype" not in inputs:
                    inputs["weight_dtype"] = "default"
                if "widgets_values" in node and isinstance(node["widgets_values"], list) and len(node["widgets_values"]) > 0:
                    node["widgets_values"][0] = TARGET_UNET
                if text_loader_id:
                    inputs["dependencies"] = [text_loader_id, 0]

            elif cls == "LTXAVTextEncoderLoader":
                inputs["text_encoder"] = TARGET_GEMMA
                inputs["ckpt_name"] = TARGET_CONNECTOR
            elif cls in ["VAELoaderKJ", "VAELoader"]:
                inputs["vae_name"] = TARGET_VIDEO_VAE
                inputs["ckpt_name"] = TARGET_VIDEO_VAE
            elif cls == "LTXVAudioVAELoader":
                inputs["ckpt_name"] = TARGET_AUDIO_VAE
                inputs["vae_name"] = TARGET_AUDIO_VAE
            elif cls == "LoraLoader":
                resolved = assignments.get(node_id, TARGET_DISTILLED_LORA)
                inputs["lora_name"] = resolved
                if "widgets_values" in node and isinstance(node["widgets_values"], list) and len(node["widgets_values"]) > 0:
                    node["widgets_values"][0] = resolved
            elif cls == "DenoMultiImageLoader":
                inputs["image_paths"] = "input/dynamic_guides"
            elif "EmptyLTXVLatentVideo" in cls or "LTXVEmptyLatentVideo" in cls:
                inputs["length"] = tgt_len
            elif "LTXVEmptyLatentAudio" in cls:
                inputs["frames_number"] = tgt_len
            elif cls == "LTXVSpatioTemporalTiledVAEDecode":
                inputs["working_device"] = "auto"
            elif cls in ["VAEDecodeTiled", "VAEDecode"]:
                inputs["tile_size"] = 512
                inputs["overlap"] = 64
                inputs["temporal_tile_length"] = 8
                inputs["temporal_overlap"] = 4
                inputs["working_device"] = "auto"

        sage_node_id = next((k for k, v in wf_data.items() if isinstance(v, dict) and v.get("class_type") == "LTX2MemoryEfficientSageAttentionPatch"), None)
        if sage_node_id:
            sage_input = wf_data[sage_node_id]["inputs"].get("model")
            if sage_input:
                for node_id, node_data in wf_data.items():
                    if isinstance(node_data, dict) and "inputs" in node_data:
                        for k, v in node_data["inputs"].items():
                            if isinstance(v, list) and len(v) > 0 and str(v[0]) == str(sage_node_id):
                                node_data["inputs"][k] = sage_input
            del wf_data[sage_node_id]

        print("⚡ Dispatching NVMe Stream-Optimized LTX-2.3 workflow to local endpoint...")
        comfy_url = "http://127.0.0.1:8188/prompt"
        payload = {"prompt": wf_data}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(comfy_url, json=payload) as resp:
                if resp.status != 200:
                    err_txt = await resp.text()
                    raise HTTPException(status_code=500, detail=f"ComfyUI execution error: {err_txt}")
                
                res_json = await resp.json()
                prompt_id = res_json.get("prompt_id")
                print(f"🎉 Job queued successfully. Prompt ID: {prompt_id}")

            history_url = f"http://127.0.0.1:8188/history/{prompt_id}"
            print("⏳ Polling local ComfyUI history for rendering outputs...")
            
            while True:
                async with session.get(history_url) as resp:
                    if resp.status == 200:
                        hist_data = await resp.json()
                        if prompt_id in hist_data:
                            print("✅ Local generation task completed.")
                            break
                await asyncio.sleep(2)

            outputs = hist_data[prompt_id].get("outputs", {})
            output_file_path = None
            for node_id, node_output in outputs.items():
                if "gifs" in node_output:
                    for gif in node_output["gifs"]:
                        filename = gif.get("filename")
                        output_file_path = os.path.join("/workspace/ComfyUI/output", filename)
                        break
                elif "images" in node_output:
                    for img in node_output["images"]:
                        filename = img.get("filename")
                        output_file_path = os.path.join("/workspace/ComfyUI/output", filename)
                        break
            
            if not output_file_path or not os.path.exists(output_file_path):
                output_dir = "/workspace/ComfyUI/output"
                files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if filename_prefix in f]
                if files:
                    output_file_path = max(files, key=os.path.getmtime)
                else:
                    raise HTTPException(status_code=500, detail="Output video file not found in output directory")

            r2_output_key = f"rendered-videos/{uuid.uuid4()}_{os.path.basename(output_file_path)}"
            print(f"📤 Uploading final rendering to Cloudflare R2 path: {r2_output_key}")
            
            try:
                self.s3.upload_file(
                    output_file_path, 
                    "video-asset-files-storage-workflow", 
                    r2_output_key,
                    ExtraArgs={"ContentType": "video/mp4"}
                )
                
                signed_url = self.s3.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': "video-asset-files-storage-workflow",
                        'Key': r2_output_key
                    },
                    ExpiresIn=86400
                )
                
                # Active VRAM and Garbage collection sweeps to prepare for consecutive low-memory requests
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass
                
                print(f"✨ Task finished successfully. Returning signed asset path: {signed_url}")
                return {"status": "success", "video_url": signed_url}
                
            except Exception as e:
                print(f"❌ Failed to transfer output asset to Cloudflare R2: {e}")
                raise HTTPException(status_code=500, detail=f"R2 asset storage exception: {e}")
