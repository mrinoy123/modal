import modal
import subprocess
import time
import os
import json
import shutil
import sys
import asyncio
import threading
import aiohttp
import urllib.request
from fastapi import Request, Response, HTTPException, Header
from typing import Optional

# ==========================================
# 1. IMAGE DEFINITION (PRESERVED)
# ==========================================
base_image = modal.Image.from_registry("nvidia/cuda:12.5.1-devel-ubuntu24.04", add_python="3.12").apt_install("git", "wget", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential", "ninja-build", "cmake", "clang", "llvm")
deps_image = base_image.env({"CUDA_HOME": "/usr/local/cuda", "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", ""), "FORCE_CUDA": "1", "TORCH_CUDA_ARCH_LIST": "8.9", "MAX_JOBS": "1", "CC": "gcc", "CXX": "g++"}).pip_install("torch==2.5.1", "torchvision", "torchaudio", index_url="https://download.pytorch.org/whl/cu124").pip_install("fastapi", "aiohttp", "boto3", "triton>=3.1.0", "ninja", "setuptools>=70.0.0", "wheel", "pip>=24.0")
compiled_image = deps_image.run_commands("git clone https://github.com/thu-ml/SageAttention.git /workspace/SageAttention", "cd /workspace/SageAttention && pip install --no-build-isolation .")
final_image = compiled_image.run_commands("git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI", "pip install -r /workspace/ComfyUI/requirements.txt").run_commands("git clone https://github.com/city96/ComfyUI-GGUF.git /workspace/ComfyUI/custom_nodes/ComfyUI-GGUF", "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite", "git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git /workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation", "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/requirements-no-cupy.txt", "git clone https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes", "git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git /workspace/ComfyUI/custom_nodes/ComfyUI-Impact-Pack").run_commands(r"find /workspace/ComfyUI/custom_nodes -name 'requirements.txt' -exec pip install -r {} \;")

app = modal.App("ltx-v2-3-api")
weights_volume = modal.Volume.from_name("ltx-video-weights")

@app.cls(
    gpu="L4", 
    image=final_image, 
    volumes={"/mnt/weights": weights_volume},
    secrets=[modal.Secret.from_name("video-generator-workflow")], 
    scaledown_window=60,
    timeout=3600 
)
class LTXEngine:
    def _log_reader(self):
        for line in iter(self.process.stdout.readline, ""):
            if line: print(f"[ComfyUI] {line.strip()}")

    @modal.enter()
    def start_comfy(self):
        import boto3
        # Linker Logic
        weights_root = None
        for mount in ["/mnt/weights", "/mnt"]:
            if os.path.exists(mount):
                for root, dirs, _ in os.walk(mount):
                    if any(x in dirs for x in ["unet", "vae", "clip"]):
                        weights_root = root
                        break
                if weights_root: break

        if weights_root:
            for m_type in ["unet", "vae", "clip", "text_encoders", "upscale_models"]:
                src, dest = os.path.join(weights_root, m_type), f"/workspace/ComfyUI/models/{m_type}"
                if os.path.exists(src):
                    if os.path.exists(dest): shutil.rmtree(dest) if not os.path.islink(dest) else os.unlink(dest)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    os.symlink(src, dest)

        self.s3 = boto3.client(service_name='s3', endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com", aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'], aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], region_name="auto")

        print("🚀 Launching Sequential Engine...")
        
        # 1. FORCING SEQUENTIAL: Environment variables to stop memory fragmentation
        env_vars = os.environ.copy()
        env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        env_vars["CUDA_MODULE_LOADING"] = "LAZY" # Saves startup VRAM
        
        # 2. THE CLI FLAGS:
        # --cache-none: This is the 'Nuclear Option'. It kills the model from VRAM the second it's done.
        # --lowvram: Ensures only one block is active.
        # --disable-smart-memory: Stops ComfyUI from trying to keep things 'ready' for the next run.
        self.process = subprocess.Popen([
            "python", "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--lowvram",
            "--cache-none", 
            "--use-sage-attention",
            "--bf16-vae",
            "--disable-smart-memory",
            "--disable-xformers" 
        ], cwd="/workspace/ComfyUI", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env_vars)
        
        self.t = threading.Thread(target=self._log_reader, daemon=True)
        self.t.start()

        print("⏳ Waiting for Sequential initialization...")
        start_time = time.time()
        while time.time() - start_time < 300:
            if self.process.poll() is not None:
                print("❌ Startup Crash!")
                os._exit(1)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8188/", timeout=1) as response:
                    if response.status == 200:
                        print("⚡ Sequential Engine ONLINE!")
                        return
            except Exception:
                time.sleep(2)
        os._exit(1)

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        if x_api_key != os.environ.get("API_KEY"): 
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        body = await request.json()
        image_url, workflow = body.get("image_url"), body.get("workflow")
        if isinstance(workflow, str): workflow = json.loads(workflow)

        local_input = "/workspace/ComfyUI/input/master_plane.png"
        os.makedirs(os.path.dirname(local_input), exist_ok=True)
        file_key = image_url.split(".dev/")[-1]
        self.s3.download_file("video-asset-files-storage-workflow", file_key, local_input)

        out_dir = "/workspace/ComfyUI/output"
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        async with aiohttp.ClientSession() as session:
            print(f"🎨 Queueing Sequential Tasks...")
            async with session.post("http://127.0.0.1:8188/prompt", json={"prompt": workflow}) as resp:
                res_json = await resp.json()
                prompt_id = res_json['prompt_id']

            start_time = time.time()
            while True:
                if self.process.poll() is not None:
                    print("❌ GPU CRASHED (Xid 43 / Out of Memory). Check logs.")
                    os._exit(1) 

                async with session.get(f"http://127.0.0.1:8188/history/{prompt_id}") as resp:
                    if resp.status == 200:
                        history = await resp.json()
                        if prompt_id in history: break
                
                if time.time() - start_time > 1500: raise HTTPException(status_code=504, detail="Timeout")
                await asyncio.sleep(5)

        videos = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]
        videos.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
        with open(os.path.join(out_dir, videos[0]), "rb") as f:
            return Response(content=f.read(), media_type="video/mp4")
