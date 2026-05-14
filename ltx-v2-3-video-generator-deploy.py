import modal
import subprocess
import time
import os
import json
import urllib.request
import shutil
from fastapi import Request, Response, HTTPException, Header
from typing import Optional

weights_volume = modal.Volume.from_name("ltx-video-weights")

# CHANGED: Using Ubuntu 24.04 to fix GLIBCXX version mismatch
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "wget", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "build-essential")
    .pip_install("torch==2.5.1", "torchvision", "torchaudio", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("fastapi", "aiohttp", "boto3", "triton>=3.1.0") 
    .run_commands(
        # SageAttention 2.2.0 fits perfectly on Ubuntu 24.04
        "pip install https://huggingface.co/Kijai/PrecompiledWheels/resolve/main/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl",
        "git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI",
        "pip install -r /workspace/ComfyUI/requirements.txt",
        "git clone https://github.com/city96/ComfyUI-GGUF.git /workspace/ComfyUI/custom_nodes/ComfyUI-GGUF",
        "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt",
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        "git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git /workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation",
        "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/requirements-no-cupy.txt",
        "git clone https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes",
        "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes/requirements.txt",
        "git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git /workspace/ComfyUI/custom_nodes/ComfyUI-Impact-Pack",
        "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-Impact-Pack/requirements.txt"
    )
)

app = modal.App("ltx-v2-3-api")

@app.cls(
    gpu="L4", 
    image=image, 
    volumes={"/mnt/weights": weights_volume},
    secrets=[modal.Secret.from_name("video-generator-workflow")], 
    scaledown_window=60,
    timeout=1200 
)
class LTXEngine:
    @modal.enter()
    def start_comfy(self):
        import boto3
        
        # Ensure directories exist and link models
        print("🔗 Linking Models from Volume...")
        model_paths = {
            "unet": "/workspace/ComfyUI/models/unet",
            "vae": "/workspace/ComfyUI/models/vae",
            "clip": "/workspace/ComfyUI/models/clip",
            "text_encoders": "/workspace/ComfyUI/models/text_encoders"
        }
        
        for folder, dest in model_paths.items():
            src = f"/mnt/weights/comfyui_models/{folder}"
            if os.path.exists(dest) and not os.path.islink(dest): shutil.rmtree(dest)
            if not os.path.exists(dest) and os.path.exists(src):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                os.symlink(src, dest)

        self.s3 = boto3.client(
            service_name='s3',
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
            region_name="auto"
        )

        print("🚀 Booting LTX ComfyUI Server...")
        # Added --disable-cuda-malloc to prevent some L4 fragmentation issues
        self.process = subprocess.Popen([
            "python", "main.py", 
            "--listen", "127.0.0.1", 
            "--use-sage-attention",
            "--highvram",
            "--bf16-vae",
            "--disable-smart-memory"
        ], cwd="/workspace/ComfyUI")
        
        # Health check loop
        for i in range(60):
            if self.process.poll() is not None:
                raise RuntimeError("ComfyUI process died during startup. Check logs.")
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/")
                print("⚡ Server Ready!")
                break
            except:
                time.sleep(2)

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        if x_api_key != os.environ.get("API_KEY"):
            raise HTTPException(status_code=403, detail="Unauthorized")

        data = await request.json()
        image_url = data.get("image_url") 
        workflow = data.get("workflow")
        if isinstance(workflow, str):
            workflow = json.loads(workflow)

        # Better path extraction for R2
        file_key = image_url.split(".dev/")[-1] 
        local_input_path = "/workspace/ComfyUI/input/master_plane.png"
        os.makedirs("/workspace/ComfyUI/input", exist_ok=True)
        
        print(f"🎬 Private Fetch: {file_key}")
        try:
            self.s3.download_file("video-asset-files-storage-workflow", file_key, local_input_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"R2 Download Error: {str(e)}")

        # Clear output dir
        out_dir = "/workspace/ComfyUI/output"
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        print("🌀 Starting GPU Render...")
        prompt_payload = json.dumps({"prompt": workflow}).encode('utf-8')
        req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=prompt_payload)
        
        try:
            resp = urllib.request.urlopen(req)
            prompt_id = json.loads(resp.read())['prompt_id']
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ComfyUI API Error: {str(e)}")
        
        start_time = time.time()
        while True:
            # Check if process is still alive while waiting
            if self.process.poll() is not None:
                raise HTTPException(status_code=500, detail="ComfyUI crashed during render")
                
            res = urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}")
            history = json.loads(res.read())
            if prompt_id in history: break
            if time.time() - start_time > 1100: raise HTTPException(status_code=504, detail="Timeout")
            time.sleep(5)
            
        videos = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]
        if not videos: raise HTTPException(status_code=500, detail="Render failed - no video found")
        
        videos.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
        with open(os.path.join(out_dir, videos[0]), "rb") as f:
            return Response(content=f.read(), media_type="video/mp4")
