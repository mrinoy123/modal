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
    "transformers", "diffusers", "accelerate", "bitsandbytes",
    "lark", "openpyxl", "blake3", "sqlalchemy", "alembic", "psutil"
)

TARGET_UNET = "ltx-2-19b-dev-fp8.safetensors"
TARGET_GEMMA = "gemma-3-12b-it-FP8.safetensors"
TARGET_CONNECTOR = "ltx-2-19b-embeddings_connector_dev_bf16.safetensors"
TARGET_VIDEO_VAE = "ltx-2-19b-dev_video_vae.safetensors"
TARGET_AUDIO_VAE = "ltx-2-19b-dev_audio_vae.safetensors"

TARGET_DISTILLED_LORA = "ltx-2-19b-distilled-lora-384.safetensors"
TARGET_DETAILER_LORA = "ltx-2-19b-ic-lora-detailer.safetensors"

# ==========================================
# PART 2: Topological Graph Analyzer & Build-Time Appliance Baker
# Purpose: Pre-downloads the master workflow, injects clean model pathways, and seals it into the appliance image.
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
            "Comfyui-workflows-json/ltx2-19b-new-workflow-modified(api)new.json", 
            raw_path
        )

        with open(raw_path, "r") as f:
            wf_data = json.load(f)

        if "workflow" in wf_data and not any("class_type" in v for v in wf_data.values() if isinstance(v, dict)):
            wf_data = wf_data["workflow"]

        print("🏗️ Build Phase: Executing Topological Graph Tracing for Multi-LoRA Injections...")
        
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
                print(f"🎯 Traced Connection Pathway: UNET -> Node {first_lora} (Rank 384 Distill) -> Node {second_lora} (19b Detailer)")
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
            
            # Injection of standard pathways (No LowVRAM modifiers)
            if cls in ["UNETLoader", "UnetLoaderGGUFAdvanced", "CheckpointLoaderSimple", "LowVRAMUNETLoader"]:
                node["class_type"] = "UNETLoader"
                inputs["unet_name"] = TARGET_UNET
                if "weight_dtype" not in inputs:
                    inputs["weight_dtype"] = "default"
                if "widgets_values" in node and isinstance(node["widgets_values"], list) and len(node["widgets_values"]) > 0:
                    node["widgets_values"][0] = TARGET_UNET

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
            
            # Hardware-Optimized Decoding Rules
            elif cls == "LTXVSpatioTemporalTiledVAEDecode":
                inputs["working_device"] = "auto"
                inputs["working_dtype"] = "float16"
            elif cls in ["VAEDecodeTiled", "VAEDecode", "LTXVVAEDecode"]:
                inputs["working_device"] = "cuda"
                inputs["working_dtype"] = "float16"

        os.makedirs("/workspace", exist_ok=True)
        with open("/workspace/prebaked_workflow.json", "w") as f:
            json.dump(wf_data, f, indent=2)
            
        print("🏗️ Build Phase: Successfully sealed prebaked_workflow.json to container disk!")
    except Exception as e:
        print(f"⚠️ Build Phase Issue (Fallback Skipped): {e}")

# ==========================================
# PART 3: Advanced Optimization Patches & Custom Node Installation
# Purpose: Applies necessary backwards-compatibility patches and establishes the 500 MB VRAM reserve rule.
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
# --- LTX-Video/LTX-2.0 Compatibility Patch ---
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

def patch_flux_layers():
    import os
    layers_path = "/workspace/ComfyUI/comfy/ldm/flux/layers.py"
    if os.path.exists(layers_path):
        with open(layers_path, "r") as f:
            content = f.read()
        
        patch_code = (
            "\n# --- LTX-Video Compatibility Fallback (apply_rotary_emb) ---\n"
            "def apply_rotary_emb(x, freqs_cis):\n"
            "    if x.shape[1] == 0: return x\n"
            "    t_ = x.reshape(*x.shape[:-1], -1, 1, 2)\n"
            "    t_out = freqs_cis[..., 0] * t_[..., 0] + freqs_cis[..., 1] * t_[..., 1]\n"
            "    return t_out.reshape(*x.shape).type_as(x)\n"
        )
        
        if "def apply_rotary_emb" not in content:
            with open(layers_path, "w") as f:
                f.write(content + patch_code)
            
            with open(layers_path, "r") as f:
                new_content = f.read()
            if "__all__" in new_content:
                with open(layers_path, "a") as f:
                    f.write("\ntry:\n    if '__all__' in globals():\n        if 'apply_rotary_emb' not in __all__:\n            __all__.append('apply_rotary_emb')\nexcept Exception: pass\n")
        print("✅ Patched comfy/ldm/flux/layers.py with robust apply_rotary_emb fallback!")

def patch_comfyui_model_management_all():
    import os
    mm_path = "/workspace/ComfyUI/comfy/model_management.py"
    if os.path.exists(mm_path):
        with open(mm_path, "r") as f:
            content = f.read()
        
        # 1. Inject the Upgraded Category-Aware Sequential Model Purger Patch at the start of load_models_gpu
        if "### SEQUENTIAL MODEL PURGER PATCH ###" not in content:
            print("🔧 Injecting upgraded category-aware sequential model-purging patch into comfy/model_management.py...")
            target_str = "def load_models_gpu("
            idx = content.find(target_str)
            if idx != -1:
                colon_idx = content.find("):", idx)
                if colon_idx != -1:
                    insert_pos = content.find("\n", colon_idx) + 1
                    patch_code = """    ### SEQUENTIAL MODEL PURGER PATCH ###
    try:
        req_classes = set()
        for m in models:
            if hasattr(m, "model") and m.model is not None:
                req_classes.add(type(m.model).__name__)
            elif hasattr(m, "model_type"):
                req_classes.add(str(m.model_type))
        
        curr_classes = set()
        for lm in current_loaded_models:
            if lm.model is not None and hasattr(lm.model, "model") and lm.model.model is not None:
                curr_classes.add(type(lm.model.model).__name__)
                
        has_conflict = False
        if req_classes and curr_classes:
            # Switch triggers only if the loaded category is NOT a subset of the requested category
            if not curr_classes.issubset(req_classes):
                has_conflict = True
                
        if has_conflict:
            print(f"🛡️ [Memory Purger] Class switch detected: currently loaded {curr_classes} -> requesting {req_classes}. Unloading previous models from VRAM...")
            try:
                unload_all_models()
            except Exception as e:
                pass
            import gc
            import torch
            import ctypes
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
        else:
            print(f"🛡️ [Memory Purger] Reuse path validated: currently loaded {curr_classes} -> requesting {req_classes}. Retaining resident weights.")
    except Exception as e:
        print(f"⚠️ Memory Purger Patch Error: {e}")
    #######################################
"""
                    content = content[:insert_pos] + patch_code + content[insert_pos:]
        
        # 2. Append the 500 MB VRAM Reservation Patch to the end of the file
        if "500 MB VRAM Reservation Patch" not in content:
            print("🔧 Appending 500 MB VRAM Reservation Patch...")
            reservation_code = """
# --- 500 MB VRAM Reservation Patch (Tuple-Safe) ---
_orig_get_free_memory = get_free_memory
def get_free_memory(dev=None, torch_free_too=False):
    res = _orig_get_free_memory(dev, torch_free_too)
    reserve_bytes = 500 * 1024 * 1024  # Force strict 500 MB Reserve
    if isinstance(res, tuple):
        free_mem, torch_free = res
        return (max(0, free_mem - reserve_bytes), torch_free)
    return max(0, res - reserve_bytes)
"""
            content += reservation_code

        with open(mm_path, "w") as f:
            f.write(content)
        print("✅ Successfully patched comfy/model_management.py!")

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
    # ⚡ Using unique cache_bust to force a complete clean rebuild, bypassing stale caches
    .run_commands(
        "git clone https://github.com/comfyanonymous/ComfyUI /workspace/ComfyUI # cache_bust=2026_05_23_19b_v5",
        "pip install -r /workspace/ComfyUI/requirements.txt # cache_bust=2026_05_23_19b_v5"
    )
    .run_commands(
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite # cache_bust=2026_05_23_19b_v5",
        "git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo # cache_bust=2026_05_23_19b_v5",
        "git clone https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes # cache_bust=2026_05_23_19b_v5",
        "git clone https://github.com/yolain/ComfyUI-Easy-Use.git /workspace/ComfyUI/custom_nodes/ComfyUI-Easy-Use # cache_bust=2026_05_23_19b_v5",
        "git clone https://github.com/Deno2026/comfyui-deno-custom-nodes.git /workspace/ComfyUI/custom_nodes/comfyui-deno-custom-nodes # cache_bust=2026_05_23_19b_v5",
        "git clone https://github.com/cubiq/ComfyUI_essentials.git /workspace/ComfyUI/custom_nodes/ComfyUI_essentials # cache_bust=2026_05_23_19b_v5",
        "git clone https://github.com/FizzleDorf/ComfyUI_FizzNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI_FizzNodes # cache_bust=2026_05_23_19b_v5",
        "git clone https://github.com/SquirrelRat/MultiString-Prompts.git /workspace/ComfyUI/custom_nodes/MultiString-Prompts # cache_bust=2026_05_23_19b_v5",
        "git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git /workspace/ComfyUI/custom_nodes/ComfyUI-Custom-Scripts # cache_bust=2026_05_23_19b_v5",
        "git clone https://github.com/siraxe/ComfyUI-LTX-FDG.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTX-FDG # cache_bust=2026_05_23_19b_v5"
    )
    .run_commands(
        "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/requirements.txt # cache_bust=2026_05_23_19b_v5",
        "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt # cache_bust=2026_05_23_19b_v5"
    )
    .run_commands(
        "pip install kornia==0.6.12 # cache_bust=2026_05_23_19b_v5"
    )
    .run_commands(
        "sed -i 's/final_pooled_output = torch.cat(pooled_out, dim=0)/final_pooled_output = torch.cat([p for p in pooled_out if p is not None], dim=0) if any(p is not None for p in pooled_out) else None/g' /workspace/ComfyUI/custom_nodes/ComfyUI_FizzNodes/BatchFuncs.py"
    )
    .run_function(patch_comfy_lightricks_model)
    .run_function(patch_ltx_video_imports)
    .run_function(patch_ltx_kornia_pad)
    .run_function(patch_flux_layers)
    .run_function(patch_comfyui_model_management_all)
    .run_function(bake_private_workflow_into_image)
)

# ==========================================
# PART 4: Production Class Definition & Resource Reclamation Loops
# ==========================================

app = modal.App("ltx-2-19b-v20-api")
weights_volume = modal.Volume.from_name("ltx-20-19b-weights")

@app.cls(
    gpu="L4", 
    image=final_image, 
    volumes={"/mnt/weights": weights_volume},
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
            
            mock_code = """import sys
import os
import psutil
import importlib.abc
import importlib.machinery

# --- Import Redirection Hook to Resolve Module vs Package Conflict (ComfyUI vs comfy/utils.py) ---
class UtilsPackageRedirector(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "utils" or fullname.startswith("utils."):
            comfyui_path = "/workspace/ComfyUI"
            original_path = list(sys.path)
            try:
                # Force dynamic routing away from the file 'comfy/utils.py' back to package directory 'utils/'
                sys.path = [p for p in sys.path if not (p.endswith("comfy") or "comfy/" in p or "comfy\\\\" in p or p.endswith("comfy-cli") or "comfy-cli" in p)]
                if comfyui_path not in sys.path:
                    sys.path.insert(0, comfyui_path)
                
                for finder in sys.meta_path:
                    if finder is self:
                        continue
                    try:
                        spec = finder.find_spec(fullname, path, target)
                        if spec is not None:
                            return spec
                    except Exception:
                        pass
            finally:
                sys.path = original_path
        return None

sys.meta_path.insert(0, UtilsPackageRedirector())
print("🛡️ [Import Guard] Registered 'utils' Package Redirector import hook successfully.")

def mock_virtual_memory():
    class MockVM:
        def __init__(self):
            self.total = 6 * 1024 * 1024 * 1024  # Force limit system RAM to 6 GB
            self.available = 4 * 1024 * 1024 * 1024  # Force limit system RAM to 4 GB
            self.percent = 33.3
            self.used = 2 * 1024 * 1024 * 1024
            self.free = 4 * 1024 * 1024 * 1024
    return MockVM()
psutil.virtual_memory = mock_virtual_memory
print("🔧 [Patched] System RAM mocked to 6GB to force aggressive disk-streaming & disable pinned CPU memory allocations!")

# --- Guider raw_conds resilient background-compatibility patch ---
try:
    import threading
    import time
    def background_patcher():
        try:
            patched = False
            while not patched:
                import sys
                modules_to_patch = []
                
                if "comfy.samplers" in sys.modules:
                    try:
                        import comfy.samplers
                        modules_to_patch.append(sys.modules["comfy.samplers"])
                    except Exception:
                        pass
                        
                if "comfy.guiders" in sys.modules:
                    try:
                        import comfy.guiders
                        modules_to_patch.append(sys.modules["comfy.guiders"])
                    except Exception:
                        pass
                        
                modules_to_patch = [m for m in modules_to_patch if m is not None]
                if not modules_to_patch:
                    time.sleep(0.1)
                    continue
                    
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
            print(f"⚠️ Warning: background patcher loop failed: {e}")
            
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
        print("🔗 Running Atomic Model Folder Linker...")
        base_models_dir = "/workspace/ComfyUI/models"
        
        dirs = ["unet", "vae", "clip", "text_encoders", "text_encoder", "checkpoints", "diffusion_models", "gguf", "loras"]
        for d in dirs:
            os.makedirs(os.path.join(base_models_dir, d), exist_ok=True)

        exact_mapping = {
            "gemma-3-12b-it-FP8.safetensors": ["text_encoders", "text_encoder"],
            "ltx-2-19b-embeddings_connector_dev_bf16.safetensors": ["checkpoints"],
            "ltx-2-19b-dev-fp8.safetensors": ["unet", "diffusion_models"],
            "ltx-2-19b-ic-lora-detailer.safetensors": ["loras"],
            "ltx-2-19b-distilled-lora-384.safetensors": ["loras"],
            "ltx-2-19b-dev_audio_vae.safetensors": ["checkpoints"],
            "ltx-2-19b-dev_video_vae.safetensors": ["vae"]
        }

        if os.path.exists("/mnt/weights"):
            for root_dir, _, files in os.walk("/mnt/weights"):
                for filename in files:
                    if filename in exact_mapping:
                        src_path = os.path.join(root_dir, filename)
                        for target_dir in exact_mapping[filename]:
                            dest = os.path.join(base_models_dir, target_dir, filename)
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

        print("🚀 Launching Clean LTX-19B Normal VRAM Server Engine...")
        os.makedirs("/tmp/comfy_swap", exist_ok=True)
        os.makedirs("/tmp/hf_offload", exist_ok=True)

        env_vars = os.environ.copy()
        env_vars["PYTHONUNBUFFERED"] = "1"  # Force unbuffered stdout streaming for real-time logs
        env_vars["TORCH_NUM_THREADS"] = "1"
        env_vars["OMP_NUM_THREADS"] = "1"
        env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
        env_vars["CUDA_MODULE_LOADING"] = "LAZY" 
        env_vars["MALLOC_TRIM_THRESHOLD_"] = "65536" 
        env_vars["HF_HUB_OFFLOAD_DIR"] = "/tmp/hf_offload"
        
        self.process = subprocess.Popen([
            "python", "-u", "main.py", "--listen", "127.0.0.1", "--port", "8188",  
            "--mmap-torch-files", "--cache-none", "--temp-directory", "/tmp/comfy_swap", 
            "--bf16-vae", "--disable-xformers", "--fp8_e4m3fn-text-enc",
            "--disable-pinned-memory"        
        ], cwd="/workspace/ComfyUI", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env_vars)
        
        self.t = threading.Thread(target=self._log_reader, daemon=True)
        self.t.start()

        asyncio.run_coroutine_threadsafe(self._ram_squeezer(), asyncio.get_event_loop())

        # Safe node registry poll loop: Blocks until node schema loads cleanly
        start_time = time.time()
        node_registration_verified = False
        while time.time() - start_time < 300:
            if self.process.poll() is not None:
                os._exit(1)
            try:
                req = urllib.request.Request("http://127.0.0.1:8188/object_info")
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.status == 200:
                        data = json.loads(response.read().decode())
                        if "LTXVLoopingSampler" in data or "KSamplerSelect" in data:
                            print("⚡ LTX-19B API ONLINE & Node Registry Verified!")
                            node_registration_verified = True
                            break
            except Exception:
                pass
            time.sleep(2)
            
        if not node_registration_verified:
            time.sleep(15)
            print("⚡ LTX-19B API ONLINE (Fallback Mode)!")

# ==========================================
# PART 5: Hybrid Endpoint Handler & Dynamic Parameter Override
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
        
        has_images = False
        if image_url:
            # Supports single image as well as comma-separated image sequences
            urls = [u.strip() for u in image_url.split(",") if u.strip()]
            for idx, url in enumerate(urls):
                print(f"📥 Downloading dynamic guide image {idx}: {url}")
                ext = os.path.splitext(url.split("?")[0])[1] or ".png"
                target_image_path = os.path.join(dynamic_guides_dir, f"guide_image_{idx}{ext}")
                
                if "r2.dev" in url or R2_ACCOUNT_ID in url:
                    parsed_url = urlparse(url)
                    key = parsed_url.path.lstrip("/")
                    try:
                        self.s3.download_file("video-asset-files-storage-workflow", key, target_image_path)
                        has_images = True
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"S3 R2 download failure for {url}: {e}")
                else:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(url) as resp:
                                if resp.status == 200:
                                    with open(target_image_path, "wb") as f:
                                        f.write(await resp.read())
                                    has_images = True
                                else:
                                    raise HTTPException(status_code=400, detail=f"Failed to download guide image. HTTP {resp.status}")
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"Download failure for {url}: {e}")

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
            
            if cls in ["UNETLoader", "UnetLoaderGGUFAdvanced", "CheckpointLoaderSimple", "LowVRAMUNETLoader"]:
                node["class_type"] = "UNETLoader"
                inputs["unet_name"] = TARGET_UNET
                if "weight_dtype" not in inputs:
                    inputs["weight_dtype"] = "default"
                if "widgets_values" in node and isinstance(node["widgets_values"], list) and len(node["widgets_values"]) > 0:
                    node["widgets_values"][0] = TARGET_UNET

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
                if has_images:
                    inputs["image_paths"] = dynamic_guides_dir
            elif "EmptyLTXVLatentVideo" in cls or "LTXVEmptyLatentVideo" in cls:
                inputs["length"] = tgt_len
            elif "LTXVEmptyLatentAudio" in cls:
                inputs["frames_number"] = tgt_len
            
            # Hardware Decoding Rules
            elif cls == "LTXVSpatioTemporalTiledVAEDecode":
                inputs["working_device"] = "auto"
                inputs["working_dtype"] = "float16"
                if "tile_size" in inputs:
                    inputs["tile_size"] = 512
                if "overlap" in inputs:
                    inputs["overlap"] = 64
                if "temporal_tile_length" in inputs:
                    inputs["temporal_tile_length"] = 8
                if "temporal_overlap" in inputs:
                    inputs["temporal_overlap"] = 4
                if "spatial_tiles" in inputs:
                    inputs["spatial_tiles"] = 8
                if "spatial_overlap" in inputs:
                    inputs["spatial_overlap"] = 4
            elif cls in ["VAEDecodeTiled", "VAEDecode", "LTXVVAEDecode"]:
                inputs["working_device"] = "cuda"
                inputs["working_dtype"] = "float16"
                if "tile_size" in inputs:
                    inputs["tile_size"] = 512
                if "overlap" in inputs:
                    inputs["overlap"] = 64
                if "temporal_tile_length" in inputs:
                    inputs["temporal_tile_length"] = 8
                if "temporal_overlap" in inputs:
                    inputs["temporal_overlap"] = 4
                if "spatial_tiles" in inputs:
                    inputs["spatial_tiles"] = 8
                if "spatial_overlap" in inputs:
                    inputs["spatial_overlap"] = 4

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

        print("⚡ Dispatching NVMe Stream-Optimized LTX-19B workflow to local endpoint...")
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
                
                # Active VRAM and Garbage collection sweeps to prepare for consecutive requests
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass
                
                # Clean up local ComfyUI VRAM cache entirely before exiting
                try:
                    free_url = "http://127.0.0.1:8188/free"
                    free_payload = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
                    free_req = urllib.request.Request(
                        free_url, 
                        data=free_payload, 
                        headers={"Content-Type": "application/json"},
                        method="POST"
                    )
                    with urllib.request.urlopen(free_req, timeout=5) as free_resp:
                        if free_resp.status == 200:
                            print("🛡️ [Memory Purger] Local VRAM cleared and returned to 0 resident model state.")
                except Exception as e:
                    print(f"⚠️ Warning: post-process local VRAM cleanup sweep skipped: {e}")

                print(f"✨ Task finished successfully. Returning signed asset path: {signed_url}")
                return {"status": "success", "video_url": signed_url}
                
            except Exception as e:
                print(f"❌ Failed to transfer output asset to Cloudflare R2: {e}")
                raise HTTPException(status_code=500, detail=f"R2 asset storage exception: {e}")
