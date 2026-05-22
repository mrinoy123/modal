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
# ==========================================

# Cloudflare R2 explicit provisioning configurations
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

# Clean, ultra-fast environment setup without slow native compilation tasks
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

# ⚡ LTX-2.3 Specific Weights & Dev Models
TARGET_UNET = "ltx-2.3-22b-dev-fp8.safetensors"
TARGET_GEMMA = "gemma_3_12B_it_fp8_scaled.safetensors"
TARGET_CONNECTOR = "ltx-2.3_text_projection_bf16.safetensors"
TARGET_VIDEO_VAE = "LTX23_video_vae_bf16.safetensors"
TARGET_AUDIO_VAE = "LTX23_audio_vae_bf16.safetensors"
TARGET_TAE = "taeltx2_3.safetensors"

# ⚡ Dual-LoRA Chain Components (Cross-version merging)
TARGET_DISTILLED_LORA = "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors"
TARGET_DETAILER_LORA = "ltx-2-19b-ic-lora-detailer.safetensors"

# ==========================================
# PART 2: Topological Graph Analyzer & Build-Time Appliance Baker
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
        # Pulling the newly modified 2.3 API workflow you uploaded
        s3.download_file(
            "video-asset-files-storage-workflow", 
            "Comfyui-workflows-json/ltx-23-new-workflow-modified(api)new.json", 
            raw_path
        )

        with open(raw_path, "r") as f:
            wf_data = json.load(f)

        if "workflow" in wf_data and not any("class_type" in v for v in wf_data.values() if isinstance(v, dict)):
            wf_data = wf_data["workflow"]

        print("🏗️ Build Phase: Executing Topological Graph Tracing for Multi-LoRA Injections...")
        
        unet_nodes = []
        lora_nodes = []
        for node_id, node in wf_data.items():
            if not isinstance(node, dict): continue
            cls = node.get("class_type", "")
            if cls in ["UNETLoader", "UnetLoaderGGUFAdvanced", "CheckpointLoaderSimple"]:
                unet_nodes.append(node_id)
            elif cls == "LoraLoader":
                lora_nodes.append(node_id)

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
            
            # Injecting LTX 2.3 specific components
            if cls in ["UNETLoader", "UnetLoaderGGUFAdvanced", "CheckpointLoaderSimple"]:
                inputs["unet_name"] = TARGET_UNET
                inputs["ckpt_name"] = TARGET_UNET
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

        os.makedirs("/workspace", exist_ok=True)
        with open("/workspace/prebaked_workflow.json", "w") as f:
            json.dump(wf_data, f, indent=2)
            
        print("🏗️ Build Phase: Successfully sealed prebaked_workflow.json to container disk!")
    except Exception as e:
        print(f"⚠️ Build Phase Issue (Fallback Skipped): {e}")

# ==========================================
# PART 3: Advanced Optimization Patches & Custom Node Installation
# ==========================================

final_image = (
    build_image.pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124"
    )
    .pip_install("numpy==1.26.4", "diffusers", "accelerate", "transformers")
    .run_commands(
        "git clone https://github.com/comfyanonymous/ComfyUI /workspace/ComfyUI",
        "cd /workspace/ComfyUI && git checkout $(git rev-list -n 1 --before='2026-03-01' HEAD)",
        "pip install -r /workspace/ComfyUI/requirements.txt"
    )
    .run_commands(
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        "git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo",
        "git clone https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes",
        "git clone https://github.com/yolain/ComfyUI-Easy-Use.git /workspace/ComfyUI/custom_nodes/ComfyUI-Easy-Use",
        "git clone https://github.com/Deno2026/comfyui-deno-custom-nodes.git /workspace/ComfyUI/custom_nodes/comfyui-deno-custom-nodes",
        "git clone https://github.com/cubiq/ComfyUI_essentials.git /workspace/ComfyUI/custom_nodes/ComfyUI_essentials",
        "git clone https://github.com/FizzleDorf/ComfyUI_FizzNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI_FizzNodes",
        "git clone https://github.com/SquirrelRat/MultiString-Prompts.git /workspace/ComfyUI/custom_nodes/MultiString-Prompts",
        "git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git /workspace/ComfyUI/custom_nodes/ComfyUI-Custom-Scripts",
        "git clone https://github.com/IvanRybakov/comfyui-node-int-to-string-convertor.git /workspace/ComfyUI/custom_nodes/comfyui-node-int-to-string-convertor",
        "git clone https://github.com/siraxe/ComfyUI-LTX-FDG.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTX-FDG"
    )
    .run_commands(
        "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/requirements.txt",
        "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt"
    )
    .run_commands(
        "sed -i 's/final_pooled_output = torch.cat(pooled_out, dim=0)/final_pooled_output = torch.cat([p for p in pooled_out if p is not None], dim=0) if any(p is not None for p in pooled_out) else None/g' /workspace/ComfyUI/custom_nodes/ComfyUI_FizzNodes/BatchFuncs.py"
    )
    # ⚡ FIXED: The destructive regex python patch for looping_sampler.py has been completely removed.
    .run_function(bake_private_workflow_into_image)
)

# ==========================================
# PART 4: Production Class Definition & Resource Reclamation Loops
# ==========================================

app = modal.App("ltx-2-3-v20-api")

# ⚡ MODIFICATION: Mapping TWO separate volumes so it can find both the 2.3 base weights AND the old 19b detailer LoRA
weights_volume_23 = modal.Volume.from_name("ltx-video-weights-23", create_if_missing=True)
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
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass
            await asyncio.sleep(2)

    @modal.enter()
    def start_comfy(self):
        import boto3
        print("🔗 Running Atomic Model Folder Linker (Multi-Volume)...")
        base_models_dir = "/workspace/ComfyUI/models"
        
        dirs = ["unet", "vae", "clip", "text_encoders", "text_encoder", "checkpoints", "diffusion_models", "gguf", "loras"]
        for d in dirs:
            os.makedirs(os.path.join(base_models_dir, d), exist_ok=True)

        # ⚡ MODIFICATION: Exact Mapping for all LTX 2.3 assets + the old 19b IC-Detailer
        exact_mapping = {
            "gemma_3_12B_it_fp8_scaled.safetensors": ["text_encoders", "text_encoder"],
            "ltx-2.3_text_projection_bf16.safetensors": ["checkpoints"],
            "ltx-2.3-22b-dev-fp8.safetensors": ["unet", "diffusion_models"],
            "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors": ["loras"],
            "ltx-2-19b-ic-lora-detailer.safetensors": ["loras"],
            "LTX23_audio_vae_bf16.safetensors": ["checkpoints"],
            "LTX23_video_vae_bf16.safetensors": ["vae"],
            "taeltx2_3.safetensors": ["vae"]
        }

        # Recursively search both mounted volumes to link assets
        for mount_point in ["/mnt/weights_23", "/mnt/weights_19"]:
            if os.path.exists(mount_point):
                for root_dir, _, files in os.walk(mount_point):
                    for filename in files:
                        if filename in exact_mapping:
                            src_path = os.path.join(root_dir, filename)
                            for target_dir in exact_mapping[filename]:
                                dest = os.path.join(base_models_dir, target_dir, filename)
                                if not os.path.exists(dest):
                                    try:
                                        os.symlink(src_path, dest)
                                    except FileExistsError:
                                        pass

        self.s3 = boto3.client(
            service_name='s3', 
            endpoint_url=R2_ENDPOINT_URL, 
            aws_access_key_id=R2_ACCESS_KEY_ID, 
            aws_secret_access_key=R2_SECRET_ACCESS_KEY, 
            region_name="auto"
        )

        print("🚀 Launching Clean LTX-2.3 Server Engine...")
        os.makedirs("/tmp/comfy_swap", exist_ok=True)
        os.makedirs("/tmp/hf_offload", exist_ok=True)

        env_vars = os.environ.copy()
        env_vars["TORCH_NUM_THREADS"] = "1"
        env_vars["OMP_NUM_THREADS"] = "1"
        env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
        env_vars["CUDA_MODULE_LOADING"] = "LAZY" 
        env_vars["MALLOC_TRIM_THRESHOLD_"] = "65536" 
        env_vars["HF_HUB_OFFLOAD_DIR"] = "/tmp/hf_offload"
        
        self.process = subprocess.Popen([
            "python", "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--mmap-torch-files", "--cache-none", "--temp-directory", "/tmp/comfy_swap", 
            "--bf16-vae", "--disable-xformers", "--fp8_e4m3fn-text-enc"        
        ], cwd="/workspace/ComfyUI", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env_vars)
        
        self.t = threading.Thread(target=self._log_reader, daemon=True)
        self.t.start()

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
# ==========================================

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, body: dict, x_api_key: str = Header(None)):
        import aiohttp
        import json
        import os
        import uuid
        import shutil
        from urllib.parse import urlparse

        if x_api_key != os.environ.get("API_KEY"):
            raise HTTPException(status_code=403, detail="Unauthorized")

        image_url = body.get("image_url")
        workflow_str = body.get("workflow")
        requested_length = body.get("length", 65)  # Defaulting to a clean 16n+1 value
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
        for node_id, node in wf_data.items():
            if not isinstance(node, dict): continue
            cls = node.get("class_type", "")
            if cls in ["UNETLoader", "UnetLoaderGGUFAdvanced", "CheckpointLoaderSimple"]:
                unet_nodes.append(node_id)
            elif cls == "LoraLoader":
                lora_nodes.append(node_id)

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

        # ⚡ MODIFICATION: LTX-2.3 strict Frame Math Constraint (16n + 1)
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
            
            if cls in ["UNETLoader", "UnetLoaderGGUFAdvanced", "CheckpointLoaderSimple"]:
                inputs["unet_name"] = TARGET_UNET
                inputs["ckpt_name"] = TARGET_UNET
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
            # Hard-cramming constraints for safety
            elif "EmptyLTXVLatentVideo" in cls or "LTXVEmptyLatentVideo" in cls:
                inputs["length"] = tgt_len
            elif "LTXVEmptyLatentAudio" in cls:
                inputs["frames_number"] = tgt_len
            # ⚡ MODIFICATION: Overriding Decoder limits so 24GB L4 GPU doesn't crash
            elif cls in ["VAEDecodeTiled", "VAEDecode", "LTXVSpatioTemporalTiledVAEDecode"]:
                inputs["tile_size"] = 512
                inputs["overlap"] = 64
                inputs["temporal_tile_length"] = 8
                inputs["temporal_overlap"] = 4
                inputs["working_device"] = "cuda"

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

        print("⚡ Dispatching LTX-2.3 workflow to local endpoint...")
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
                
                print(f"✨ Task finished successfully. Returning signed asset path: {signed_url}")
                return {"status": "success", "video_url": signed_url}
                
            except Exception as e:
                print(f"❌ Failed to transfer output asset to Cloudflare R2: {e}")
                raise HTTPException(status_code=500, detail=f"R2 asset storage exception: {e}")
