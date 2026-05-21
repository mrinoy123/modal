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

# Clone ComfyUI and install required custom nodes (VFI Purged)
final_image = build_image.run_commands(
    "git clone https://github.com/comfyanonymous/ComfyUI /workspace/ComfyUI",
    "cd /workspace/ComfyUI && git checkout $(git rev-list -n 1 --before=\"2026-03-01\" HEAD)",
    "pip install -r /workspace/ComfyUI/requirements.txt"
).run_commands(
    "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
    "git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo",
    "cd /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo && git checkout $(git rev-list -n 1 --before=\"2026-03-01\" HEAD)",
    "git clone https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes",
    "git clone https://github.com/yolain/ComfyUI-Easy-Use.git /workspace/ComfyUI/custom_nodes/ComfyUI-Easy-Use",
    "git clone https://github.com/Deno2026/comfyui-deno-custom-nodes.git /workspace/ComfyUI/custom_nodes/comfyui-deno-custom-nodes",
    "git clone https://github.com/cubiq/ComfyUI_essentials.git /workspace/ComfyUI/custom_nodes/ComfyUI_essentials",
    "git clone https://github.com/FizzleDorf/ComfyUI_FizzNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI_FizzNodes",
    "git clone https://github.com/SquirrelRat/MultiString-Prompts.git /workspace/ComfyUI/custom_nodes/MultiString-Prompts",
    "git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git /workspace/ComfyUI/custom_nodes/ComfyUI-Custom-Scripts",
    "git clone https://github.com/IvanRybakov/comfyui-node-int-to-string-convertor.git /workspace/ComfyUI/custom_nodes/comfyui-node-int-to-string-convertor"
).run_commands(
    "pip install diffusers accelerate transformers",
    "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/requirements.txt",
    "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt"
).run_commands(
    # 🔥 PATCH 1: Fix FizzNodes NoneType crash
    "sed -i 's/final_pooled_output = torch.cat(pooled_out, dim=0)/final_pooled_output = torch.cat([p for p in pooled_out if p is not None], dim=0) if any(p is not None for p in pooled_out) else None/g' /workspace/ComfyUI/custom_nodes/ComfyUI_FizzNodes/BatchFuncs.py",
    # 🔥 PATCH 2: Robust fallback patch for guider.raw_conds to avoid AttributeError exceptions
    "python3 -c \"filepath = '/workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/looping_sampler.py'; code = open(filepath).read(); code = code.replace('positive, negative = guider.raw_conds', 'positive, negative = getattr(guider, \\'raw_conds\\', None) or (getattr(guider, \\'original_conds\\', {}).get(\\'positive\\'), getattr(guider, \\'original_conds\\', {}).get(\\'negative\\'))'); open(filepath, 'w').write(code)\""
).run_commands(
    "pip uninstall -y torch torchvision torchaudio numpy",
    "pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124",
    "pip install --no-cache-dir numpy==1.26.4"
)

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
        
        dirs = ["unet", "vae", "clip", "text_encoders", "text_encoder", "checkpoints", "diffusion_models", "gguf", "loras"]
        for d in dirs:
            os.makedirs(os.path.join(base_models_dir, d), exist_ok=True)

        def get_target_dirs(filename: str):
            fn = filename.lower()
            # 1. Distilled base model & Quantized GGUF models
            if "distilled-fp8" in fn or "dev-fp8" in fn or "dev-q3" in fn or fn.endswith(".gguf"):
                return ["unet", "diffusion_models"]
            # 2. Gemma Text Encoders
            if "gemma" in fn:
                return ["text_encoders", "text_encoder"]
            # 3. Embedding Connectors
            if "embeddings_connector" in fn:
                return ["clip"]
            # 4. Audio and Video VAEs
            if "video_vae" in fn or "audio_vae" in fn:
                return ["vae"]
            # 5. IC-LoRAs and Standard LoRAs
            if "lora" in fn:
                return ["loras"]
            # 6. Fallback CLIPs
            if "clip_l" in fn:
                return ["clip"]
            # 7. Safe Fallbacks based on standard file extension mappings
            if fn.endswith((".pth", ".pt")):
                return ["unet"]
            return ["checkpoints"]

        if os.path.exists("/mnt/weights"):
            for root_dir, _, files in os.walk("/mnt/weights"):
                for filename in files:
                    if not filename.endswith((".safetensors", ".gguf", ".pth", ".pt", ".bin")):
                        continue
                        
                    src_path = os.path.join(root_dir, filename)
                    target_dirs = get_target_dirs(filename)
                    
                    for target_dir in target_dirs:
                        dest = os.path.join(base_models_dir, target_dir, filename)
                        if not os.path.exists(dest):
                            try: 
                                os.symlink(src_path, dest)
                                print(f"🔗 Linked weight: {filename} -> models/{target_dir}")
                            except FileExistsError: pass

        self.s3 = boto3.client(
            service_name='s3', 
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com", 
            aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'], 
            aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], 
            region_name="auto"
        )

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
        
        # Stream model files dynamically with memory mapping and instant unloading
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

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        if x_api_key != os.environ.get("API_KEY"): 
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        body = await request.json()
        
        if isinstance(body, dict):
            if "json" in body: body = body["json"]
            elif "body" in body: body = body["body"]

        image_url = body.get("image_url") if isinstance(body, dict) else None
        workflow = body.get("workflow") if isinstance(body, dict) else body
        requested_length = body.get("length") if isinstance(body, dict) else None

        if isinstance(workflow, str):
            try: workflow = json.loads(workflow)
            except Exception: pass

        if isinstance(workflow, dict) and "workflow" in workflow:
            if not any("class_type" in v for v in workflow.values() if isinstance(v, dict)):
                workflow = workflow["workflow"]

        if isinstance(workflow, dict):
            if "nodes" in workflow and "last_node_id" in workflow:
                raise HTTPException(status_code=400, detail="Format Mismatch: Passed Canvas UI instead of API layout.")

        # =========================================================================
        # 🔗 AGGRESSIVE FUZZYLINKER: FORCED MODEL INJECTION
        # =========================================================================
        target_unet = "ltx-2-19b-distilled-fp8.safetensors"
        target_gemma = "gemma-3-12b-it-FP8.safetensors"
        target_connector = "ltx-2-19b-embeddings_connector_dev_bf16.safetensors"
        target_video_vae = "ltx-2-19b-dev_video_vae.safetensors"
        target_audio_vae = "ltx-2-19b-dev_audio_vae.safetensors"
        target_distilled_lora = "ltx-2-19b-distilled-lora-384.safetensors"
        target_detailer_lora = "ltx-2-19b-ic-lora-detailer.safetensors"

        def fuzzy_linker(wf_data):
            for node_id, node in wf_data.items():
                if not isinstance(node, dict) or "inputs" not in node:
                    continue
                
                cls = node.get("class_type", "")
                inputs = node["inputs"]
                
                # 1. UNET & Checkpoint Forcing
                if cls in ["UNETLoader", "UnetLoaderGGUFAdvanced", "CheckpointLoaderSimple"]:
                    inputs["unet_name"] = target_unet
                    inputs["ckpt_name"] = target_unet
                    if "widgets_values" in node and isinstance(node["widgets_values"], list) and len(node["widgets_values"]) > 0:
                        node["widgets_values"][0] = target_unet

                # 2. Text Encoder (Gemma) Forcing
                elif cls == "LTXAVTextEncoderLoader":
                    inputs["text_encoder"] = target_gemma
                    inputs["ckpt_name"] = target_connector

                # 3. Video VAE Forcing
                elif cls in ["VAELoaderKJ", "VAELoader"]:
                    inputs["vae_name"] = target_video_vae
                    inputs["ckpt_name"] = target_video_vae

                # 4. Audio VAE Forcing
                elif cls == "LTXVAudioVAELoader":
                    inputs["ckpt_name"] = target_audio_vae
                    inputs["vae_name"] = target_audio_vae

                # 5. LoRA Assignment Logic
                elif cls == "LoraLoader":
                    current_lora = str(inputs.get("lora_name", "")).lower()
                    resolved = target_distilled_lora
                    if "detail" in current_lora or "ic-lora" in current_lora:
                        resolved = target_detailer_lora
                    
                    inputs["lora_name"] = resolved
                    if "widgets_values" in node and isinstance(node["widgets_values"], list) and len(node["widgets_values"]) > 0:
                        node["widgets_values"][0] = resolved

                # 6. Guide Image Input
                elif cls == "DenoMultiImageLoader":
                    inputs["image_paths"] = "input/dynamic_guides"

            return wf_data

        if isinstance(workflow, dict):
            workflow = fuzzy_linker(workflow)

            # Global Sequence & Frame Matching
            if requested_length is not None:
                try:
                    tgt_len = int(requested_length)
                    if (tgt_len - 1) % 8 != 0:
                        tgt_len = ((tgt_len - 1) // 8) * 8 + 1
                        if tgt_len < 9: tgt_len = 9
                    
                    for node_id, node in workflow.items():
                        if not isinstance(node, dict) or "inputs" not in node: continue
                        cls = node.get("class_type", "")
                        if "EmptyLTXVLatentVideo" in cls or "LTXVEmptyLatentVideo" in cls: 
                            node["inputs"]["length"] = tgt_len
                        if "LTXVEmptyLatentAudio" in cls: 
                            node["inputs"]["frames_number"] = tgt_len
                            node["inputs"]["frame_rate"] = 12
                        if "BatchPromptSchedule" in cls: 
                            node["inputs"]["max_frames"] = tgt_len
                except Exception as e:
                    print(f"⚠️ Dynamic framing error: {e}")

            # Sage Attention Patch Optimizer
            sage_node_id = next((k for k, v in workflow.items() if v.get("class_type") == "LTX2MemoryEfficientSageAttentionPatch"), None)
            if sage_node_id:
                sage_input = workflow[sage_node_id]["inputs"].get("model")
                if sage_input:
                    for node_id, node_data in workflow.items():
                        if isinstance(node_data, dict) and "inputs" in node_data:
                            for k, v in node_data["inputs"].items():
                                if isinstance(v, list) and len(v) > 0 and v[0] == sage_node_id:
                                    node_data["inputs"][k] = sage_input
                del workflow[sage_node_id]

        # =========================================================================

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
                
                print("🧹 Generation Complete. Purging VRAM & Unloading Models...")
                try:
                    await session.post("http://127.0.0.1:8188/free", json={"unload_models": True, "free_memory": True})
                    await session.post("http://127.0.0.1:8188/api/free", json={"unload_models": True, "free_memory": True})
                except Exception as e:
                    print(f"VRAM Purge Notice: {e}")

                with open(os.path.join(out_dir, videos[0]), "rb") as f:
                    return Response(content=f.read(), media_type="video/mp4")
        finally:
            ram_task.cancel()
