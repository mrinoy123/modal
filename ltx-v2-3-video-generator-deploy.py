import modal
import subprocess
import time
import os
import json
import urllib.request
import shutil
import sys
import asyncio
import threading
from fastapi import Request, Response, HTTPException, Header
from typing import Optional

# ==========================================
# 1. IMAGE DEFINITION (STAYING IDENTICAL TO PRESERVE CACHE)
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
    timeout=1800 
)
class LTXEngine:
    def _log_reader(self):
        """Drains the subprocess pipe to prevent ComfyUI from hanging."""
        for line in iter(self.process.stdout.readline, ""):
            if line: print(f"[ComfyUI] {line.strip()}")

    @modal.enter()
    def start_comfy(self):
        import boto3
        
        # ==========================================
        # 🕵️ THE RESILIENT LINKER
        # ==========================================
        def resolve_and_link_models():
            print("🕵️ Resilient Linker: Searching for model weights...")
            candidate_mounts = ["/mnt/weights", "/mnt", "/weights"]
            weights_root = None

            # 1. Find the folder that contains 'unet', 'vae', or 'clip'
            for mount in candidate_mounts:
                if not os.path.exists(mount): continue
                for root, dirs, _ in os.walk(mount, topdown=True):
                    # If we find a folder with standard Comfy subdirs, that's our root
                    if any(x in dirs for x in ["unet", "vae", "clip", "text_encoders"]):
                        weights_root = root
                        break
                if weights_root: break

            if not weights_root:
                print("❌ Resilient Linker Failed: Could not find model folders on Volume.")
                return

            print(f"✅ Found weights root at: {weights_root}")

            # 2. Map discovered subfolders to ComfyUI
            model_types = ["unet", "vae", "clip", "text_encoders", "upscale_models"]
            for m_type in model_types:
                src = os.path.join(weights_root, m_type)
                dest = f"/workspace/ComfyUI/models/{m_type}"
                
                if os.path.exists(src):
                    # Force delete existing empty ComfyUI folder
                    if os.path.exists(dest):
                        if os.path.islink(dest): os.unlink(dest)
                        else: shutil.rmtree(dest)
                    
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    os.symlink(src, dest)
                    print(f"🔗 Linked: {m_type} -> {src}")
                    # Verify first file for logs
                    try:
                        content = os.listdir(dest)
                        print(f"   ∟ Found: {content[:1]}")
                    except: pass
        
        resolve_and_link_models()

        # Cloudflare R2 Setup
        self.s3 = boto3.client(
            service_name='s3', 
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com", 
            aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'], 
            aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], 
            region_name="auto"
        )

        print("🚀 Starting ComfyUI (Low VRAM Stability Mode)...")
        self.process = subprocess.Popen([
            "python", "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--lowvram", "--bf16-vae", "--use-sage-attention"
        ], cwd="/workspace/ComfyUI", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        self.t = threading.Thread(target=self._log_reader, daemon=True)
        self.t.start()

        # Health check
        for i in range(60):
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/", timeout=1)
                print("⚡ Server Online!")
                return
            except: time.sleep(2)

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        if x_api_key != os.environ.get("API_KEY"): raise HTTPException(status_code=403, detail="Unauthorized")
        
        try:
            data = await request.json()
        except:
            raise HTTPException(status_code=400, detail="Client Disconnect or Invalid JSON")
            
        image_url, workflow = data.get("image_url"), data.get("workflow")
        if isinstance(workflow, str): workflow = json.loads(workflow)

        local_input = "/workspace/ComfyUI/input/master_plane.png"
        os.makedirs(os.path.dirname(local_input), exist_ok=True)
        file_key = image_url.split(".dev/")[-1]
        self.s3.download_file("video-asset-files-storage-workflow", file_key, local_input)

        out_dir = "/workspace/ComfyUI/output"
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=json.dumps({"prompt": workflow}).encode('utf-8'))
        prompt_id = json.loads(urllib.request.urlopen(req).read())['prompt_id']
        
        start_time = time.time()
        while True:
            if self.process.poll() is not None: sys.exit(1) # Kill container to stop billing if server crashes
            try:
                res = urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}", timeout=2)
                if prompt_id in json.loads(res.read()): break
            except: pass
            if time.time() - start_time > 1500: raise HTTPException(status_code=504, detail="Timeout")
            await asyncio.sleep(5)
            
        videos = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]
        if not videos: raise HTTPException(status_code=500, detail="Render failed")
        
        videos.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
        with open(os.path.join(out_dir, videos[0]), "rb") as f:
            return Response(content=f.read(), media_type="video/mp4")
