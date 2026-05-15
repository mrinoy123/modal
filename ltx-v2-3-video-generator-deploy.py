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
from fastapi import Request, Response, HTTPException, Header
from typing import Optional

# =========================================================================
# 1. IMAGE DEFINITION (KEEP IDENTICAL TO PRESERVE SAGE ATTENTION CACHE)
# =========================================================================
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
        """Drains the pipe so ComfyUI doesn't hang when the 64KB buffer is full."""
        for line in iter(self.process.stdout.readline, ""):
            if line: print(f"[ComfyUI] {line.strip()}")

    @modal.enter()
    def start_comfy(self):
        import boto3
        # Resilient Linker
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

        print("🚀 Starting ComfyUI (Normal VRAM)...")
        self.process = subprocess.Popen([
            "python", "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--normalvram", "--use-sage-attention", "--bf16-vae"
        ], cwd="/workspace/ComfyUI", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        self.t = threading.Thread(target=self._log_reader, daemon=True)
        self.t.start()

        # Simple blocking wait for startup (allowed in @modal.enter)
        time.sleep(10)

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        if x_api_key != os.environ.get("API_KEY"): raise HTTPException(status_code=403, detail="Unauthorized")
        
        body = await request.json()
        image_url, workflow = body.get("image_url"), body.get("workflow")
        if isinstance(workflow, str): workflow = json.loads(workflow)

        # Secure Fetch
        local_input = "/workspace/ComfyUI/input/master_plane.png"
        os.makedirs(os.path.dirname(local_input), exist_ok=True)
        file_key = image_url.split(".dev/")[-1]
        self.s3.download_file("video-asset-files-storage-workflow", file_key, local_input)

        out_dir = "/workspace/ComfyUI/output"
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        # ASYNC API Interaction
        async with aiohttp.ClientSession() as session:
            # 1. Queue Prompt
            async with session.post("http://127.0.0.1:8188/prompt", json={"prompt": workflow}) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=500, detail="ComfyUI rejected prompt")
                res_json = await resp.json()
                prompt_id = res_json['prompt_id']

            print(f"🎬 Processing: {prompt_id}")

            # 2. Polling Loop
            start_time = time.time()
            while True:
                # Watchdog: os._exit(1) is a hard kill to stop billing immediately 
                # without confusing the async event loop with a SystemExit exception
                if self.process.poll() is not None:
                    print("❌ Server CRASHED. Hard-killing container to stop billing.")
                    os._exit(1) 

                try:
                    async with session.get(f"http://127.0.0.1:8188/history/{prompt_id}") as resp:
                        history = await resp.json()
                        if prompt_id in history:
                            break
                except Exception:
                    pass # Server busy

                if time.time() - start_time > 1500:
                    raise HTTPException(status_code=504, detail="Generation Timeout")
                
                await asyncio.sleep(5) # Keeps the event loop heartbeat alive

        # 3. Return Result
        videos = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]
        if not videos: raise HTTPException(status_code=500, detail="No output file")
        videos.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
        
        with open(os.path.join(out_dir, videos[0]), "rb") as f:
            return Response(content=f.read(), media_type="video/mp4")
