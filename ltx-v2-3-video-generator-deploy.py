import modal
import subprocess
import time
import os
import json
import urllib.request
import shutil
import sys
from fastapi import Request, Response, HTTPException, Header
from typing import Optional

# ==========================================
# 1. IMAGE DEFINITION (LAYERED FOR CACHING)
# ==========================================

# Layer 1: OS Packages
base_image = (
    modal.Image.from_registry("nvidia/cuda:12.5.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install(
        "git", "wget", "ffmpeg", "libgl1", "libglib2.0-0", 
        "build-essential", "ninja-build", "cmake", "clang", "llvm"
    )
)

# Layer 2: Core ML Dependencies
deps_image = (
    base_image
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", ""),
        "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": "8.9", 
        "MAX_JOBS": "1", 
        "CC": "gcc",
        "CXX": "g++",
    })
    .pip_install(
        "torch==2.5.1", "torchvision", "torchaudio", 
        index_url="https://download.pytorch.org/whl/cu124"
    )
    .pip_install("fastapi", "aiohttp", "boto3", "triton>=3.1.0", "ninja", "setuptools>=70.0.0", "wheel", "pip>=24.0")
)

# Layer 3: SageAttention Compilation (Slowest part - cached once done)
compiled_image = (
    deps_image
    .run_commands(
        "git clone https://github.com/thu-ml/SageAttention.git /workspace/SageAttention",
        "cd /workspace/SageAttention && pip install --no-build-isolation ."
    )
)

# Layer 4: ComfyUI & Custom Nodes
final_image = (
    compiled_image
    .run_commands(
        "git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI",
        "pip install -r /workspace/ComfyUI/requirements.txt"
    )
    .run_commands(
        "git clone https://github.com/city96/ComfyUI-GGUF.git /workspace/ComfyUI/custom_nodes/ComfyUI-GGUF",
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        "git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git /workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation",
        "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/requirements-no-cupy.txt",
        "git clone https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes",
        "git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git /workspace/ComfyUI/custom_nodes/ComfyUI-Impact-Pack"
    )
    .run_commands(
        # Fixed syntax warning with raw string 'r'
        r"find /workspace/ComfyUI/custom_nodes -name 'requirements.txt' -exec pip install -r {} \;"
    )
)

app = modal.App("ltx-v2-3-api")
weights_volume = modal.Volume.from_name("ltx-video-weights")

@app.cls(
    gpu="L4", 
    image=final_image, 
    volumes={"/mnt/weights": weights_volume},
    secrets=[modal.Secret.from_name("video-generator-workflow")], 
    scaledown_window=30, # Kill GPU quickly when idle
    timeout=1200 
)
class LTXEngine:
    @modal.enter()
    def start_comfy(self):
        import boto3
        
        print("🔗 Mapping Models from Volume...")
        for d in ["unet", "vae", "clip", "text_encoders", "upscale_models"]:
            src, dest = f"/mnt/weights/comfyui_models/{d}", f"/workspace/ComfyUI/models/{d}"
            if os.path.exists(src):
                if os.path.exists(dest) and not os.path.islink(dest): shutil.rmtree(dest)
                if not os.path.exists(dest):
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    os.symlink(src, dest)

        self.s3 = boto3.client(
            service_name='s3',
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
            region_name="auto"
        )

        print("🚀 Starting ComfyUI (Normal VRAM Mode)...")
        # IMPORTANT: Using --normalvram to allow the 22B model to swap during VAE/RIFE steps
        self.process = subprocess.Popen([
            "python", "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--use-sage-attention", 
            "--normalvram", 
            "--bf16-vae"
        ], cwd="/workspace/ComfyUI", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Health Check
        start_wait = time.time()
        while time.time() - start_wait < 180:
            if self.process.poll() is not None:
                print(f"❌ ComfyUI died:\n{self.process.stdout.read()}")
                sys.exit(1)
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/", timeout=1)
                print("⚡ Online")
                return
            except:
                time.sleep(2)
        sys.exit(1)

    @modal.exit()
    def stop_comfy(self):
        if hasattr(self, 'process'):
            self.process.terminate()

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        if x_api_key != os.environ.get("API_KEY"):
            raise HTTPException(status_code=403, detail="Unauthorized")

        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON or Client Disconnect")

        image_url, workflow = data.get("image_url"), data.get("workflow")
        if isinstance(workflow, str): workflow = json.loads(workflow)

        # R2 Fetch
        local_input = "/workspace/ComfyUI/input/master_plane.png"
        os.makedirs("/workspace/ComfyUI/input", exist_ok=True)
        file_key = image_url.split(".dev/")[-1]
        
        try:
            self.s3.download_file("video-asset-files-storage-workflow", file_key, local_input)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"R2 Download Error: {e}")

        # Clean Output Directory
        out_dir = "/workspace/ComfyUI/output"
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        # Queue Prompt
        try:
            prompt_payload = json.dumps({"prompt": workflow}).encode('utf-8')
            req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=prompt_payload)
            prompt_id = json.loads(urllib.request.urlopen(req).read())['prompt_id']
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to reach ComfyUI")

        # Monitor with Billing Protection (sys.exit(1) if server crashes)
        start_time = time.time()
        while True:
            if self.process.poll() is not None:
                print("❌ Server crashed during render!")
                sys.exit(1) 

            try:
                check = urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}")
                history = json.loads(check.read())
                if prompt_id in history: break
            except:
                pass 

            if time.time() - start_time > 1100: # 18 minute timeout
                raise HTTPException(status_code=504, detail="Generation Timeout")
            time.sleep(5)

        # Find and Return Video
        videos = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]
        if not videos:
            raise HTTPException(status_code=500, detail="Render finished but no file found")
        
        videos.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
        target_video = os.path.join(out_dir, videos[0])

        with open(target_video, "rb") as f:
            return Response(content=f.read(), media_type="video/mp4")
