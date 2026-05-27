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
    # 1. Install SageAttention and generic requirements first
    "pip install sageattention",
    # 2. Re-enforce stable numpy and kornia packages
    "pip install --force-reinstall numpy==1.26.4 \"kornia<=0.7.3\"",
    # 3. THE SLEDGEHAMMER (MUST BE LAST): Overwrite any rogue PyTorch/Torchvision installations with clean CUDA 12.4 packages
    "pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124"
)

app = modal.App("ltx-2-19b-v20-api")
weights_volume = modal.Volume.from_name("ltx-20-19b-weights")

@app.cls(
    gpu="L4", 
    image=final_image, 
    volumes={"/mnt/weights": weights_volume},
    secrets=[modal.Secret.from_name("video-generator-workflow")], 
    memory=8192, # Memory successfully set back to 8GB
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

        init_file_path = "/workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/__init__.py"
        if os.path.exists(init_file_path):
            patch_code = """\ntry:\n    import comfy.samplers\n    _orig_set_conds = comfy.samplers.CFGGuider.set_conds\n    def _patched_set_conds(self, positive, negative):\n        self.raw_conds = (positive, negative)\n        return _orig_set_conds(self, positive, negative)\n    comfy.samplers.CFGGuider.set_conds = _patched_set_conds\nexcept Exception: pass\n"""
            with open(init_file_path, "a") as f: f.write(patch_code)

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
        
        # Aggressive '--gpu-only' flag ensures all active inference is executed strictly on L4 GPU VRAM
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

        # Format prompt payload map into a clean pipe-separated string timeline for LTXVMultiPromptProvider
        if isinstance(prompts_dict, dict):
            try:
                # Sort numerically by frame index to keep correct temporal progression
                sorted_keys = sorted(prompts_dict.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
                prompts_list = [str(prompts_dict[k]).strip() for k in sorted_keys if str(prompts_dict[k]).strip()]
                prompts_timeline_str = "|".join(prompts_list)
            except Exception as e:
                print(f"[Warning] Failed to parse prompts dict numerically: {e}")
                prompts_timeline_str = "|".join([str(v).strip() for v in prompts_dict.values()])
        else:
            prompts_timeline_str = str(prompts_dict)

        # Baseline Target Weights
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
                
                # Resiliency fallback: Always write placeholder image if download failed
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
                # PHASE 1: SUBGRAPH 1 - Language Execution Pass
                # ====================================================
                print("🧠 Phase 1 Active: Initializing Subgraph 1...")
                sg1_raw = body.get("subgraph_1")
                if sg1_raw:
                    sg1 = json.loads(sg1_raw) if isinstance(sg1_raw, str) else sg1_raw
                else:
                    with open("comfyui-ltx-20-subgraph-1(api).json", "r") as f:
                        sg1 = json.load(f)

                # Overwrite internal node parameter targets dynamically matching API files
                if "243" in sg1:
                    sg1["243"]["inputs"]["text_encoder"] = target_gemma
                    sg1["243"]["inputs"]["ckpt_name"] = target_connector
                    sg1["243"]["inputs"]["device"] = "default"  # Validation list only contains ['default', 'cpu']
                if "246" in sg1:  # FIXED: dynamically overwrites inputs of LTXVMultiPromptProvider (Node 246) with our pipe-separated timeline
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

                # WIPE GEMMA FROM VRAM: Clear space for unthrottled UNET sampling loops
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
                sg2_raw = body.get("subgraph_2")
                if sg2_raw:
                    sg2 = json.loads(sg2_raw) if isinstance(sg2_raw, str) else sg2_raw
                else:
                    with open("comfyui-ltx-20-subgraph-2(api).json", "r") as f:
                        sg2 = json.load(f)

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
                
                if "245" in sg2:
                    sg2["245"]["inputs"]["file_name"] = "POSITIVEconditioning.safetensors"
                if "246" in sg2:
                    sg2["246"]["inputs"]["file_name"] = "NEGATIVEconditioning.safetensors"

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
                sg3_raw = body.get("subgraph_3")
                if sg3_raw:
                    sg3 = json.loads(sg3_raw) if isinstance(sg3_raw, str) else sg3_raw
                else:
                    with open("comfyui-ltx-20-Subgraph-3(api).json", "r") as f:
                        sg3 = json.load(f)

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
                
                if "282" in sg3:
                    sg3["282"]["inputs"]["file_name"] = "POSITIVEconditioning.safetensors"
                if "283" in sg3:
                    sg3["283"]["inputs"]["file_name"] = "NEGATIVEconditioning.safetensors"

                async with session.post("http://127.0.0.1:8188/prompt", json={"prompt": sg3}) as resp:
                    r3 = await resp.json()
                    if "error" in r3: raise HTTPException(status_code=400, detail=f"Subgraph 3 Error: {r3['error']}")
                    prompt_id3 = r3['prompt_id']

                while True:
                    async with session.get("http://127.0.0.1:8188/history") as h_resp:
                        history = await h_resp.json()
                        if prompt_id3 in history: break
                    await asyncio.sleep(4)

            # Look for video output and respond with binary content streams
            videos = [v for v in os.listdir(out_dir) if v.endswith((".mp4", ".mkv", ".webm"))]
            if not videos: raise HTTPException(status_code=500, detail="Output tracking buffers are empty.")
                
            videos.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
            with open(os.path.join(out_dir, videos[0]), "rb") as f:
                return Response(content=f.read(), media_type="video/mp4")
        finally:
            ram_task.cancel()
