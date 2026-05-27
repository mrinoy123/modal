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

# Switched from ubuntu24.04 to the official ubuntu22.04 CUDA tag
base_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-devel-ubuntu22.04", 
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

final_image = build_image.run_commands(
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
    "pip install sageattention",
    "pip install --force-reinstall numpy==1.26.4 \"kornia<=0.7.3\"",
    "pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124"
)

app = modal.App("ltx-2-19b-v20-api")
weights_volume = modal.Volume.from_name("ltx-20-19b-weights")

@app.cls(
    gpu="L4", 
    image=final_image, 
    volumes={"/mnt/weights": weights_volume},
    secrets=[modal.Secret.from_name("video-generator-workflow")], 
    memory=8192, 
    scaledown_window=30,
    timeout=3600 
)
class LTXEngine:
    def _log_reader(self):
        for line in iter(self.process.stdout.readline, ""):
            if line: print(f"[ComfyUI] {line.strip()}")

    async def _ram_squeezer(self):
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
        for d in dirs: os.makedirs(os.path.join(base_models_dir, d), exist_ok=True)

        if os.path.exists("/mnt/weights"):
            for root_dir, _, files in os.walk("/mnt/weights"):
                for filename in files:
                    if not filename.endswith((".safetensors", ".gguf", ".pth", ".pt", ".bin")): continue
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

        # =====================================================================
        # 🔥 THE ULTIMATE HOT-PATCH: NATIVE TORCH SAVING
        # completely replace safetensors with native PyTorch .pt formats
        # to support the complex nested arrays from LTXVMultiPromptProvider
        # =====================================================================
        saver_path = "/workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/conditioning_saver.py"
        if os.path.exists(saver_path):
            with open(saver_path, "w") as f:
                f.write('''import torch\nimport os\nimport folder_paths\n\nclass LTXVSaveConditioning:\n    @classmethod\n    def INPUT_TYPES(s):\n        return {"required": {"conditioning": ("CONDITIONING",), "file_name": ("STRING", {"default": "conditioning.pt"}), "device": (["default", "float16", "bfloat16", "float32"],)}}\n    RETURN_TYPES = ()\n    FUNCTION = "execute"\n    CATEGORY = "Lightricks/LTXVideo"\n    OUTPUT_NODE = True\n\n    def execute(self, conditioning, file_name, device):\n        output_dir = folder_paths.get_output_directory()\n        file_path = os.path.join(output_dir, file_name)\n        def recursive_cast(obj, dtype):\n            if isinstance(obj, torch.Tensor):\n                if obj.is_floating_point(): return obj.to(dtype).contiguous()\n                return obj.contiguous()\n            elif isinstance(obj, dict): return {k: recursive_cast(v, dtype) for k, v in obj.items()}\n            elif isinstance(obj, list): return [recursive_cast(v, dtype) for v in obj]\n            elif isinstance(obj, tuple): return tuple(recursive_cast(v, dtype) for v in obj)\n            return obj\n        target_dtype = torch.bfloat16 if device == "bfloat16" else torch.float16 if device == "float16" else torch.float32\n        converted = conditioning if device == "default" else recursive_cast(conditioning, target_dtype)\n        torch.save(converted, file_path)\n        return ()\n''')
                
        loader_path = "/workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/conditioning_loader.py"
        if os.path.exists(loader_path):
            with open(loader_path, "w") as f:
                f.write('''import torch\nimport os\nimport folder_paths\n\nclass LTXVLoadConditioning:\n    @classmethod\n    def INPUT_TYPES(s):\n        input_dir = folder_paths.get_output_directory()\n        files = [f for f in os.listdir(input_dir) if f.endswith(".pt") or f.endswith(".safetensors")] if os.path.exists(input_dir) else []\n        return {"required": {"file_name": (files,), "device": (["default", "float16", "bfloat16", "float32"],)}}\n    RETURN_TYPES = ("CONDITIONING",)\n    FUNCTION = "execute"\n    CATEGORY = "Lightricks/LTXVideo"\n\n    def execute(self, file_name, device):\n        input_dir = folder_paths.get_output_directory()\n        file_path = os.path.join(input_dir, file_name)\n        conditioning = torch.load(file_path, weights_only=False)\n        def recursive_cast(obj, dtype):\n            if isinstance(obj, torch.Tensor):\n                if obj.is_floating_point(): return obj.to(dtype).contiguous()\n                return obj.contiguous()\n            elif isinstance(obj, dict): return {k: recursive_cast(v, dtype) for k, v in obj.items()}\n            elif isinstance(obj, list): return [recursive_cast(v, dtype) for v in obj]\n            elif isinstance(obj, tuple): return tuple(recursive_cast(v, dtype) for v in obj)\n            return obj\n        target_dtype = torch.bfloat16 if device == "bfloat16" else torch.float16 if device == "float16" else torch.float32\n        if device != "default": conditioning = recursive_cast(conditioning, target_dtype)\n        return (conditioning,)\n''')
        # =====================================================================

        print("🚀 Launching High-Speed Unthrottled LTX Server Engine...")
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
            "--mmap", "--cache-none", "--temp-directory", "/tmp/comfy_swap", 
            "--bf16-vae", "--disable-xformers", "--fp8_e4m3fn-text-enc",
            "--gpu-only"
        ], cwd="/workspace/ComfyUI", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env_vars)
        
        self.t = threading.Thread(target=self._log_reader, daemon=True)
        self.t.start()

        start_time = time.time()
        while time.time() - start_time < 300:
            if self.process.poll() is not None: os._exit(1)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8188/", timeout=1) as response:
                    if response.status == 200: return
            except Exception: time.sleep(2)
        os._exit(1)

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        if x_api_key != os.environ.get("API_KEY"): 
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        body = await request.json()
        if isinstance(body, dict):
            if "json" in body: body = body["json"]
            elif "body" in body: body = body["body"]

        incoming_image_urls = body.get("image_url")
        requested_length = body.get("length", 73)
        prompts_dict = body.get("prompts", {})
        negative_prompt = body.get("negative", "worst quality, blurry, distorted")

        if isinstance(prompts_dict, dict):
            try:
                sorted_keys = sorted(prompts_dict.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
                prompts_list = [str(prompts_dict[k]).strip() for k in sorted_keys if str(prompts_dict[k]).strip()]
                prompts_timeline_str = "|".join(prompts_list)
            except Exception as e:
                print(f"[Warning] Failed to parse prompts dict numerically: {e}")
                prompts_timeline_str = "|".join([str(v).strip() for v in prompts_dict.values()])
        else:
            prompts_timeline_str = str(prompts_dict)

        target_unet = "ltx-2-19b-distilled-fp8.safetensors"
        target_gemma = "gemma-3-12b-it-FP8.safetensors"
        target_connector = "ltx-2-19b-embeddings_connector_dev_bf16.safetensors"
        target_video_vae = "ltx-2-19b-dev_video_vae.safetensors"
        target_audio_vae = "ltx-2-19b-dev_audio_vae.safetensors"
        target_detailer_lora = "ltx-2-19b-ic-lora-detailer.safetensors"

        dynamic_guides_dir = "/workspace/ComfyUI/input/dynamic_guides"
        if os.path.exists(dynamic_guides_dir): shutil.rmtree(dynamic_guides_dir)
        os.makedirs(dynamic_guides_dir, exist_ok=True)

        urls_to_download = []
        if incoming_image_urls:
            if isinstance(incoming_image_urls, list): urls_to_download = [str(u).strip() for u in incoming_image_urls if str(u).strip()]
            elif isinstance(incoming_image_urls, str) and incoming_image_urls.strip():
                urls_to_download = [u.strip() for u in incoming_image_urls.split(",") if u.strip()]

        if not urls_to_download:
            from PIL import Image
            img = Image.new('RGB', (384, 480), color='black')
            img.save(os.path.join(dynamic_guides_dir, "guide_0.png"))
        else:
            async def download_one(session, url_str, target_dest):
                from urllib.parse import urlparse
                import botocore.exceptions
                try:
                    parsed = urlparse(url_str)
                    if "r2.cloudflarestorage.com" in url_str or "pub-" in url_str or parsed.netloc == "" or not parsed.scheme:
                        file_key = parsed.path.lstrip('/')
                        while "//" in file_key: file_key = file_key.replace("//", "/")
                        print(f"📥 Downloading dynamic key from R2: {file_key}")
                        await asyncio.get_event_loop().run_in_executor(
                            None, 
                            self.s3.download_file, 
                            "video-asset-files-storage-workflow", 
                            file_key, 
                            target_dest
                        )
                    else:
                        print(f"📥 Downloading dynamic public URL: {url_str}")
                        async with session.get(url_str, timeout=120) as r:
                            if r.status == 200:
                                with open(target_dest, "wb") as f: f.write(await r.read())
                            else:
                                print(f"[Warning] HTTP status {r.status} for URL: {url_str}")
                except botocore.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == "404":
                        print(f"[Warning] Key not found in R2: {url_str}")
                    else:
                        print(f"[Warning] R2 ClientError downloading {url_str}: {e}")
                except Exception as e:
                    print(f"[Warning] Unexpected error downloading {url_str}: {e}")
                
                if not os.path.exists(target_dest):
                    from PIL import Image
                    img = Image.new('RGB', (384, 480), color='black')
                    img.save(target_dest)
                    print(f"[Fallback] Generated blank placeholder guide image at: {target_dest}")

            async with aiohttp.ClientSession() as download_session:
                tasks = [download_one(download_session, url, os.path.join(dynamic_guides_dir, f"guide_{i}.png")) for i, url in enumerate(urls_to_download)]
                await asyncio.gather(*tasks)

        out_dir = "/workspace/ComfyUI/output"
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        ram_task = asyncio.create_task(self._ram_squeezer())

        try:
            async with aiohttp.ClientSession() as session:
                
                # ====================================================
                # PRE-FLIGHT: Update Subgraphs to use native .pt format
                # ====================================================
                sg1 = body.get("subgraph_1", {})
                if isinstance(sg1, str): sg1 = json.loads(sg1)
                for node_id, node_data in sg1.items():
                    if node_data.get("class_type") == "LTXVSaveConditioning":
                        if "POSITIVE" in str(node_data.get("inputs", {}).get("file_name", "")).upper():
                            node_data["inputs"]["file_name"] = "(POSITIVE)conditioning.pt"
                        else:
                            node_data["inputs"]["file_name"] = "(NEGATIVE)conditioning.pt"

                sg2 = body.get("subgraph_2", {})
                if isinstance(sg2, str): sg2 = json.loads(sg2)
                for node_id, node_data in sg2.items():
                    if node_data.get("class_type") == "LTXVLoadConditioning":
                        if "POSITIVE" in str(node_data.get("inputs", {}).get("file_name", "")).upper():
                            node_data["inputs"]["file_name"] = "(POSITIVE)conditioning.pt"
                        else:
                            node_data["inputs"]["file_name"] = "(NEGATIVE)conditioning.pt"

                sg3 = body.get("subgraph_3", {})
                if isinstance(sg3, str): sg3 = json.loads(sg3)
                for node_id, node_data in sg3.items():
                    if node_data.get("class_type") == "LTXVLoadConditioning":
                        if "POSITIVE" in str(node_data.get("inputs", {}).get("file_name", "")).upper():
                            node_data["inputs"]["file_name"] = "(POSITIVE)conditioning.pt"
                        else:
                            node_data["inputs"]["file_name"] = "(NEGATIVE)conditioning.pt"

                # ====================================================
                # PHASE 1: SUBGRAPH 1 - Language Execution Pass
                # ====================================================
                print("🧠 Phase 1 Active: Initializing Subgraph 1...")
                
                if "243" in sg1:
                    sg1["243"]["inputs"]["text_encoder"] = target_gemma
                    sg1["243"]["inputs"]["ckpt_name"] = target_connector
                    sg1["243"]["inputs"]["device"] = "default"  
                if "246" in sg1: 
                    if prompts_timeline_str:
                        sg1["246"]["inputs"]["prompts"] = prompts_timeline_str
                if "112" in sg1:
                    if negative_prompt:
                        sg1["112"]["inputs"]["text"] = negative_prompt

                async with session.post("http://127.0.0.1:8188/prompt", json={"prompt": sg1}) as resp:
                    r1 = await resp.json()
                    if "error" in r1: raise HTTPException(status_code=400, detail=f"Subgraph 1 Error: {r1['error']}")
                    prompt_id1 = r1['prompt_id']

                while True:
                    async with session.get("http://127.0.0.1:8188/history") as h_resp:
                        history = await h_resp.json()
                        if prompt_id1 in history: break
                    await asyncio.sleep(1)

                print("🔀 Phase 1 Complete. Evicting Gemma completely from GPU memory blocks...")
                async with session.post("http://127.0.0.1:8188/free", json={"unload_models": True, "free_memory": True}) as free_resp:
                    await free_resp.read()
                
                import torch
                torch.cuda.empty_cache()
                ctypes.CDLL("libc.so.6").malloc_trim(0)

                # ====================================================
                # PHASE 2: SUBGRAPH 2 - Pure Video Sampling Pass
                # ====================================================
                print("🎬 Phase 2 Active: Initializing Subgraph 2...")

                if "238" in sg2:
                    sg2["238"]["inputs"]["unet_name"] = target_unet
                if "248" in sg2: 
                    sg2["248"]["inputs"]["lora_name"] = target_detailer_lora
                if "194" in sg2:
                    sg2["194"]["inputs"]["length"] = int(requested_length)
                if "237" in sg2:
                    sg2["237"]["inputs"]["image_paths"] = dynamic_guides_dir
                if "235" in sg2:
                    sg2["235"]["inputs"]["num_images"] = len(urls_to_download) if urls_to_download else 1

                async with session.post("http://127.0.0.1:8188/prompt", json={"prompt": sg2}) as resp:
                    r2 = await resp.json()
                    if "error" in r2: raise HTTPException(status_code=400, detail=f"Subgraph 2 Error: {r2['error']}")
                    prompt_id2 = r2['prompt_id']

                while True:
                    async with session.get("http://127.0.0.1:8188/history") as h_resp:
                        history = await h_resp.json()
                        if prompt_id2 in history: break
                    await asyncio.sleep(4)

                print("💾 Phase 2 Complete. Latents cached to disk storage layers.")

                latent_files = [f for f in os.listdir(out_dir) if f.endswith(".latent")]
                if latent_files:
                    latent_files.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
                    src_latent = os.path.join(out_dir, latent_files[0])
                    dest_latent = os.path.join(out_dir, "video_latent_output.latent")
                    if src_latent != dest_latent:
                        shutil.copy(src_latent, dest_latent)
                        print(f"Copied latent from {src_latent} to {dest_latent} for Subgraph 3 compatibility.")

                # ====================================================
                # PHASE 3: SUBGRAPH 3 - Audio Synthesis & Final Pack
                # ====================================================
                print("🎵 Phase 3 Active: Initializing Subgraph 3...")

                if "278" in sg3: 
                    sg3["278"]["inputs"]["unet_name"] = target_unet
                if "279" in sg3: 
                    sg3["279"]["inputs"]["lora_name"] = target_detailer_lora
                if "295" in sg3:
                    sg3["295"]["inputs"]["ckpt_name"] = target_audio_vae
                if "296" in sg3:
                    sg3["296"]["inputs"]["vae_name"] = target_video_vae
                if "290" in sg3: 
                    sg3["290"]["inputs"]["frames_number"] = int(requested_length)

                async with session.post("http://127.0.0.1:8188/prompt", json={"prompt": sg3}) as resp:
                    r3 = await resp.json()
                    if "error" in r3: raise HTTPException(status_code=400, detail=f"Subgraph 3 Error: {r3['error']}")
                    prompt_id3 = r3['prompt_id']

                while True:
                    async with session.get("http://127.0.0.1:8188/history") as h_resp:
                        history = await h_resp.json()
                        if prompt_id3 in history: break
                    await asyncio.sleep(4)

            videos = [v for v in os.listdir(out_dir) if v.endswith((".mp4", ".mkv", ".webm"))]
            if not videos: raise HTTPException(status_code=500, detail="Output tracking buffers are empty.")
                
            videos.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
            with open(os.path.join(out_dir, videos[0]), "rb") as f:
                return Response(content=f.read(), media_type="video/mp4")
        finally:
            ram_task.cancel()
