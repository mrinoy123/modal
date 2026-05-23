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
from fastapi import Request, Response, HTTPException, Header
from typing import Optional

# ==========================================
# PART 1: Infrastructure Configuration & Base Image
# Purpose: Defines the core operating system, environment variables, system-level dependencies, and the base python libraries needed for video processing and AI model inference.
# ==========================================

# Setup optimized development container image with required compilation tooling
base_image = modal.Image.from_registry(
    "nvidia/cuda:12.5.1-devel-ubuntu24.04", 
    add_python="3.12"
).apt_install(
    "git", "wget", "ffmpeg", "libgl1", "libglib2.0-0", 
    "build-essential", "ninja-build", "cmake", "clang", "llvm"
)

# Build image with explicit cache-locked dependencies
build_image = base_image.env({
    "CUDA_HOME": "/usr/local/cuda",
    "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", ""),
    "FORCE_CUDA": "1",
    "TORCH_CUDA_ARCH_LIST": "8.9", 
    "MAX_JOBS": "1",
    "CC": "gcc",
    "CXX": "g++"
}).pip_install(
    "fastapi", "aiohttp", "boto3", "triton>=3.1.0", 
    "ninja", "setuptools>=70.0.0", "wheel", "pip>=24.0"
).pip_install(
    "pandas", "numexpr", "pytz", "python-dateutil", 
    "scipy", "matplotlib", "colorama", "librosa", "soundfile", 
    "decord", "imageio", "scikit-image", "numba", "einops", 
    "transformers", "diffusers", "accelerate", "bitsandbytes"
)

# ==========================================
# PART 2: Advanced Optimization Patches & Custom Node Installation
# Purpose: Clean compilation of required nodes, dependency locks, and SageAttention binaries.
# ==========================================

# Clone ComfyUI and install required custom nodes (VFI Purged)
custom_nodes_image = build_image.run_commands(
    "git clone https://github.com/comfyanonymous/ComfyUI /workspace/ComfyUI",
    "pip install -r /workspace/ComfyUI/requirements.txt"
).run_commands(
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
).run_commands(
    "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/requirements.txt",
    r"find /workspace/ComfyUI/custom_nodes -name 'requirements.txt' -exec pip install -r {} \;"
).run_commands(
    # 🔥 THE SLEDGEHAMMER: Force overwrite any rogue PyTorch upgrades from the custom nodes.
    "pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124",
    # Re-enforce kornia and numpy limits to prevent array mismatches
    "pip install --force-reinstall numpy==1.26.4 \"kornia<=0.7.3\""
)

# SageAttention source compilation (Placed post-Sledgehammer to compile against the correct PyTorch binary)
final_image = custom_nodes_image.run_commands(
    "git clone https://github.com/thu-ml/SageAttention.git /workspace/SageAttention",
    "cd /workspace/SageAttention && pip install --no-build-isolation ."
)


# ==========================================
# PART 3: Production Class Definition & Engine Initialization
# Purpose: Defines the Modal application resources, handles the pre-execution model symlinking, and launches the ComfyUI subprocess.
# ==========================================

app = modal.App("ltx-2-19b-v20-api")
weights_volume = modal.Volume.from_name("ltx-20-19b-weights")

@app.cls(
    gpu="L4", 
    image=final_image, 
    volumes={"/mnt/weights": weights_volume},
    secrets=[modal.Secret.from_name("video-generator-workflow")], 
    memory=8192,
    scaledown_window=60,
    timeout=3600 
)
class LTXEngine:
    def _log_reader(self):
        for line in iter(self.process.stdout.readline, ""):
            if line: print(f"[ComfyUI] {line.strip()}")

    async def _ram_squeezer(self):
        print("🛡️ RAM Watchdog Active. Forcing Linux to drop page cache...")
        while True:
            try:
                with open('/proc/sys/vm/drop_caches', 'w') as f:
                    f.write('1\n')
            except Exception:
                try: ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception: pass
            await asyncio.sleep(2)

    @modal.enter()
    def start_comfy(self):
        import boto3
        print("🔗 Running Atomic Model Folder Linker...")
        base_models_dir = "/workspace/ComfyUI/models"
        
        # Removed "vfi" from the directory list
        dirs = ["unet", "vae", "clip", "text_encoders", "text_encoder", "checkpoints", "diffusion_models", "gguf", "loras"]
        for d in dirs:
            os.makedirs(os.path.join(base_models_dir, d), exist_ok=True)

        if os.path.exists("/mnt/weights"):
            for root_dir, _, files in os.walk("/mnt/weights"):
                for filename in files:
                    if not filename.endswith((".safetensors", ".gguf", ".pth", ".pt", ".bin")):
                        continue
                        
                    src_path = os.path.join(root_dir, filename)
                    for target_dir in ["unet", "vae", "clip", "text_encoders", "text_encoder", "checkpoints", "diffusion_models", "loras"]:
                        dest = os.path.join(base_models_dir, target_dir, filename)
                        if not os.path.exists(dest):
                            try: os.symlink(src_path, dest)
                            except FileExistsError: pass

        self.s3 = boto3.client(
            service_name='s3', 
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com", 
            aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'], 
            aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], 
            region_name="auto"
        )

        # Lightweight attribute patch to prevent CFGGuider raw_conds AttributeErrors during looping sampling
        init_file_path = "/workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/__init__.py"
        if os.path.exists(init_file_path):
            print("🔗 Injecting CFGGuider attribute patch into ComfyUI-LTXVideo...")
            patch_code = """
# Monkeypatch comfy.samplers.CFGGuider to support raw_conds
try:
    import comfy.samplers
    _orig_set_conds = comfy.samplers.CFGGuider.set_conds
    def _patched_set_conds(self, positive, negative):
        self.raw_conds = (positive, negative)
        return _orig_set_conds(self, positive, negative)
    comfy.samplers.CFGGuider.set_conds = _patched_set_conds
    print("Successfully monkeypatched comfy.samplers.CFGGuider to support raw_conds attribute!")
except Exception as e:
    print("Failed to monkeypatch comfy.samplers.CFGGuider:", e)
"""
            with open(init_file_path, "a") as f:
                f.write(patch_code)

        print("🚀 Launching Clean LTX Server Engine...")
        os.makedirs("/tmp/comfy_swap", exist_ok=True)
        os.makedirs("/tmp/hf_offload", exist_ok=True)

        env_vars = os.environ.copy()
        env_vars["TORCH_NUM_THREADS"] = "1"
        env_vars["OMP_NUM_THREADS"] = "1"
        env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
        env_vars["CUDA_MODULE_LOADING"] = "LAZY" 
        env_vars["MALLOC_TRIM_THRESHOLD_"] = "65536" 
        env_vars["HF_HUB_OFFLOAD_DIR"] = "/tmp/hf_offload"
        
        # Configured with lowvram and disable-smart-memory to protect the 24 GB VRAM ceiling on L4
        self.process = subprocess.Popen([
            "python", "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--mmap", "--cache-none", "--temp-directory", "/tmp/comfy_swap", 
            "--bf16-vae", "--disable-xformers", "--fp8_e4m3fn-text-enc",
            "--lowvram", "--disable-smart-memory"
        ], cwd="/workspace/ComfyUI", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env_vars)
        
        self.t = threading.Thread(target=self._log_reader, daemon=True)
        self.t.start()

        start_time = time.time()
        while time.time() - start_time < 300:
            if self.process.poll() is not None:
                print("❌ Startup Crash!")
                os._exit(1)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8188/", timeout=1) as response:
                    if response.status == 200:
                        print("⚡ LTX-2 API ONLINE!")
                        return
            except Exception:
                time.sleep(2)
        os._exit(1)

# ==========================================
# PART 4: Hybrid Endpoint Handler & Execution Process
# Purpose: Exposes a FastAPI endpoint to process incoming requests, handle dynamic JSON injection, download assets, and render output.
# ==========================================

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        if x_api_key != os.environ.get("API_KEY"): 
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        body = await request.json()
        
        if isinstance(body, dict):
            if "json" in body:
                body = body["json"]
            elif "body" in body:
                body = body["body"]

        image_url = body.get("image_url") if isinstance(body, dict) else None
        workflow = body.get("workflow") if isinstance(body, dict) else body
        requested_length = body.get("length") if isinstance(body, dict) else None
        prompts_override = body.get("prompts") if isinstance(body, dict) else None
        prompt_override = body.get("prompt") if isinstance(body, dict) else None

        if isinstance(workflow, str):
            try: workflow = json.loads(workflow)
            except Exception: pass

        if isinstance(workflow, dict) and "workflow" in workflow:
            if not any("class_type" in v for v in workflow.values() if isinstance(v, dict)):
                workflow = workflow["workflow"]

        if isinstance(workflow, dict):
            if "nodes" in workflow and "last_node_id" in workflow:
                raise HTTPException(status_code=400, detail="Format Mismatch: Passed Canvas UI instead of API layout.")

        target_unet = "ltx-2-19b-distilled-fp8.safetensors"
        target_gemma = "gemma-3-12b-it-FP8.safetensors"
        target_connector = "ltx-2-19b-embeddings_connector_dev_bf16.safetensors"
        target_video_vae = "ltx-2-19b-dev_video_vae.safetensors"
        target_audio_vae = "ltx-2-19b-dev_audio_vae.safetensors"
        target_distilled_lora = "ltx-2-19b-distilled-lora-384.safetensors"
        target_detailer_lora = "ltx-2-19b-ic-lora-detailer.safetensors"

        def find_node(cls_name):
            return next((k for k, v in workflow.items() if v.get("class_type") == cls_name), None)

        dynamic_guides_dir = "/workspace/ComfyUI/input/dynamic_guides"
        if os.path.exists(dynamic_guides_dir):
            shutil.rmtree(dynamic_guides_dir)
        os.makedirs(dynamic_guides_dir, exist_ok=True)

        urls_to_download = []
        if image_url:
            if isinstance(image_url, list):
                urls_to_download = [str(u).strip() for u in image_url if str(u).strip()]
            elif isinstance(image_url, str) and image_url.strip():
                if "," in image_url:
                    urls_to_download = [u.strip() for u in image_url.split(",") if u.strip()]
                else:
                    urls_to_download = [image_url.strip()]

        if not urls_to_download:
            from PIL import Image
            img = Image.new('RGB', (1024, 1024), color='black')
            img.save(os.path.join(dynamic_guides_dir, "guide_0.png"))
            print("Creating black canvas fallback guide.")
        else:
            async def download_one(session, url_str, target_dest):
                from urllib.parse import urlparse
                parsed = urlparse(url_str)
                is_r2_storage = ("r2.cloudflarestorage.com" in url_str or "pub-" in url_str or parsed.netloc == "" or not parsed.scheme)
                
                if is_r2_storage:
                    file_key = parsed.path.lstrip('/')
                    while "//" in file_key:
                        file_key = file_key.replace("//", "/")
                    print(f"📥 Fetching R2 file: {file_key}")
                    await asyncio.get_event_loop().run_in_executor(
                        None, 
                        self.s3.download_file, 
                        "video-asset-files-storage-workflow", 
                        file_key, 
                        target_dest
                    )
                else:
                    print(f"📥 Downloading HTTP file: {url_str}")
                    try:
                        async with session.get(url_str, timeout=120) as r:
                            if r.status == 200:
                                f_content = await r.read()
                                with open(target_dest, "wb") as f:
                                    f.write(f_content)
                            else:
                                raise Exception(f"HTTP code {r.status}")
                    except Exception as err:
                        print(f"HTTP direct download failed, falling back to urllib: {err}")
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            urllib.request.urlretrieve,
                            url_str,
                            target_dest
                        )

            async with aiohttp.ClientSession() as session:
                tasks = []
                for idx, url in enumerate(urls_to_download):
                    dest = os.path.join(dynamic_guides_dir, f"guide_{idx}.png")
                    tasks.append(download_one(session, url, dest))
                await asyncio.gather(*tasks)

        # Re-resolve dynamic absolute file references to avoid custom image loader directory errors
        downloaded_paths = []
        if urls_to_download:
            for idx in range(len(urls_to_download)):
                downloaded_paths.append(os.path.join(dynamic_guides_dir, f"guide_{idx}.png"))
        else:
            downloaded_paths.append(os.path.join(dynamic_guides_dir, "guide_0.png"))
        
        image_paths_str = "\n".join(downloaded_paths)

        if isinstance(workflow, dict):
            if requested_length is not None:
                try:
                    tgt_len = int(requested_length)
                    if (tgt_len - 1) % 8 != 0:
                        tgt_len = ((tgt_len - 1) // 8) * 8 + 1
                        if tgt_len < 9: tgt_len = 9
                    
                    video_latent_node = find_node("EmptyLTXVLatentVideo")
                    audio_latent_node = find_node("LTXVEmptyLatentAudio")
                    
                    if video_latent_node:
                        workflow[video_latent_node]["inputs"]["length"] = tgt_len
                    if audio_latent_node:
                        workflow[audio_latent_node]["inputs"]["frames_number"] = tgt_len

                    batch_prompt_nodes = [k for k, v in workflow.items() if v.get("class_type") == "BatchPromptSchedule"]
                    for node in batch_prompt_nodes:
                        workflow[node]["inputs"]["max_frames"] = tgt_len
                except Exception as e:
                    print(f"⚠️ Dynamic framing error: {e}")

            if prompts_override or prompt_override:
                multi_string_node = find_node("MultiStringPrompts")
                if multi_string_node:
                    if prompts_override:
                        if isinstance(prompts_override, list):
                            for idx, p_text in enumerate(prompts_override[:5]):
                                workflow[multi_string_node]["inputs"][f"multi_prompt_{idx+1}"] = p_text
                        elif isinstance(prompts_override, dict):
                            for k, p_text in prompts_override.items():
                                if k in workflow[multi_string_node]["inputs"]:
                                    workflow[multi_string_node]["inputs"][k] = p_text
                    elif prompt_override and isinstance(prompt_override, str):
                        workflow[multi_string_node]["inputs"]["multi_prompt_1"] = prompt_override

            sanitized_workflow = {}
            for node_id, node_data in workflow.items():
                if isinstance(node_data, dict) and "class_type" in node_data:
                    if "inputs" not in node_data or node_data["inputs"] is None:
                        node_data["inputs"] = {}

                    class_type = node_data.get("class_type")

                    if class_type in ["UNETLoader", "UnetLoaderGGUFAdvanced"]:
                        if "unet_name" in node_data["inputs"]:
                            node_data["inputs"]["unet_name"] = target_unet
                        if "ckpt_name" in node_data["inputs"]:
                            node_data["inputs"]["ckpt_name"] = target_unet
                        if "widgets_values" in node_data and isinstance(node_data["widgets_values"], list):
                            if len(node_data["widgets_values"]) > 0:
                                node_data["widgets_values"][0] = target_unet

                    if class_type == "LTXAVTextEncoderLoader":
                        node_data["inputs"]["text_encoder"] = target_gemma
                        node_data["inputs"]["ckpt_name"] = target_connector

                    if class_type in ["VAELoaderKJ", "VAELoader"]:
                        if "vae_name" in node_data["inputs"]:
                            node_data["inputs"]["vae_name"] = target_video_vae
                        if "ckpt_name" in node_data["inputs"]:
                            node_data["inputs"]["ckpt_name"] = target_video_vae

                    if class_type == "LTXVAudioVAELoader":
                        if "ckpt_name" in node_data["inputs"]:
                            node_data["inputs"]["ckpt_name"] = target_audio_vae
                        if "vae_name" in node_data["inputs"]:
                            node_data["inputs"]["vae_name"] = target_audio_vae

                    # Safe path formatting bypass for multiple image loaders
                    if class_type in ["DenoMultiImageLoader", "MultiImageLoader"]:
                        node_data["inputs"]["image_paths"] = image_paths_str

                    if class_type == "LTXVEmptyLatentAudio":
                        node_data["inputs"]["frame_rate"] = 12

                    if class_type == "LoraLoader":
                        lora_input_name = node_data["inputs"].get("lora_name") or ""
                        if not lora_input_name and "widgets_values" in node_data and isinstance(node_data["widgets_values"], list):
                            if len(node_data["widgets_values"]) > 0:
                                lora_input_name = node_data["widgets_values"][0] or ""
                        
                        resolved_lora = None
                        if "distilled" in str(lora_input_name).lower() or not lora_input_name:
                            resolved_lora = target_distilled_lora
                        elif "detail" in str(lora_input_name).lower() or "ic-lora" in str(lora_input_name).lower():
                            resolved_lora = target_detailer_lora
                        
                        if resolved_lora:
                            node_data["inputs"]["lora_name"] = resolved_lora
                            if "widgets_values" in node_data and isinstance(node_data["widgets_values"], list):
                                if len(node_data["widgets_values"]) > 0:
                                    node_data["widgets_values"][0] = resolved_lora

                    sanitized_workflow[str(node_id)] = node_data
            
            sage_node_id = find_node("LTX2MemoryEfficientSageAttentionPatch")
            if sage_node_id:
                sage_input = sanitized_workflow[sage_node_id]["inputs"].get("model")
                if sage_input:
                    for node_id, node_data in sanitized_workflow.items():
                        if isinstance(node_data, dict) and "inputs" in node_data:
                            for k, v in node_data["inputs"].items():
                                if isinstance(v, list) and len(v) > 0 and v[0] == sage_node_id:
                                    node_data["inputs"][k] = sage_input
                del sanitized_workflow[sage_node_id]

            workflow = sanitized_workflow

        out_dir = "/workspace/ComfyUI/output"
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        ram_task = asyncio.create_task(self._ram_squeezer())

        try:
            async with aiohttp.ClientSession() as session:
                print("🎨 Running Pipeline...")
                async with session.post("http://127.0.0.1:8188/prompt", json={"prompt": workflow}) as resp:
                    res_json = await resp.json()
                    if "error" in res_json or "prompt_id" not in res_json:
                        error_msg = res_json.get("error", res_json)
                        raise HTTPException(status_code=400, detail=f"Invalid JSON: {error_msg}")
                    prompt_id = res_json['prompt_id']

                start_time = time.time()
                while True:
                    if self.process.poll() is not None:
                        raise HTTPException(status_code=500, detail="Backend server execution failure.")
                    async with session.get("http://127.0.0.1:8188/history") as hist_resp:
                        if hist_resp.status == 200:
                            history = await hist_resp.json()
                            if prompt_id in history:
                                break
                    if time.time() - start_time > 2400:
                        raise HTTPException(status_code=540, detail="Execution timeout reached.")
                    await asyncio.sleep(5)

            videos = [v for v in os.listdir(out_dir) if v.endswith((".mp4", ".mkv", ".webm"))]
            if not videos:
                raise HTTPException(status_code=500, detail="Output generation target missing.")
                
            videos.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
            with open(os.path.join(out_dir, videos[0]), "rb") as f:
                return Response(content=f.read(), media_type="video/mp4")
        finally:
            ram_task.cancel()
