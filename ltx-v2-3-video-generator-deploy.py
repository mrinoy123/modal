import modal
import subprocess
import time
import os
import json
import urllib.request
import shutil
import boto3 # Required for private R2 access
from fastapi import Request, Response, HTTPException, Header
from typing import Optional

# 1. MOUNT THE CORRECT VOLUME
weights_volume = modal.Volume.from_name("ltx-video-weights")

# 2. Build the Python 3.11 Image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch", "torchvision", "torchaudio", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("fastapi", "aiohttp", "boto3") # Added boto3
    .run_commands(
        "git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI",
        "pip install -r /workspace/ComfyUI/requirements.txt",
        
        "git clone https://github.com/city96/ComfyUI-GGUF.git /workspace/ComfyUI/custom_nodes/ComfyUI-GGUF",
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        "git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git /workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation",
        
        "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt",
        "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/requirements-no-cupy.txt"
    )
)

app = modal.App("ltx-v2-3-api")

@app.cls(
    gpu="L4", 
    image=image, 
    volumes={"/mnt/weights": weights_volume},
    secrets=[modal.Secret.from_name("video-generator-workflow")], 
    scaledown_window=60,
    timeout=900 
)
class LTXEngine:
    @modal.enter()
    def start_comfy(self):
        # Symlink weights
        print("🔗 Linking Models from Volume...")
        folders = ["unet", "vae", "clip", "loras", "text_encoders", "upscale_models"]
        for f in folders:
            src, dest = f"/mnt/weights/comfyui_models/{f}", f"/workspace/ComfyUI/models/{f}"
            if os.path.exists(dest) and not os.path.islink(dest): shutil.rmtree(dest)
            if not os.path.exists(dest) and os.path.exists(src): os.symlink(src, dest)

        # Initialize Boto3 Client for Private R2
        self.s3 = boto3.client(
            service_name='s3',
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
            region_name="auto"
        )

        print("🚀 Booting LTX ComfyUI Server...")
        self.process = subprocess.Popen(["python", "main.py", "--listen", "127.0.0.1"], cwd="/workspace/ComfyUI")
        for i in range(30):
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/")
                print("⚡ ComfyUI is Ready!")
                break
            except:
                time.sleep(2)

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        # 1. Security check
        if x_api_key != os.environ.get("API_KEY"):
            raise HTTPException(status_code=403, detail="Unauthorized")

        # 2. Parse Request
        data = await request.json()
        image_url = data.get("image_url") # e.g. https://...dev/2026-05-14/plane.png
        workflow_raw = data.get("workflow")
        workflow = json.loads(workflow_raw) if isinstance(workflow_raw, str) else workflow_raw

        # 3. Secure Download via Boto3 (Private Storage)
        # Extract the Key (path) from the URL: everything after the domain
        file_key = image_url.split(".dev/")[-1] 
        local_input_path = "/workspace/ComfyUI/input/master_plane.png"
        
        print(f"🎬 Private Fetch: {file_key}")
        try:
            self.s3.download_file("video-asset-files-storage-workflow", file_key, local_input_path)
            print("✅ Image downloaded securely via Boto3")
        except Exception as e:
            print(f"❌ R2 Download Error: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch private asset")

        # 4. Cleanup Output
        out_dir = "/workspace/ComfyUI/output"
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        # 5. Execute Render
        print("🌀 Starting GPU Render...")
        prompt_data = json.dumps({"prompt": workflow}).encode('utf-8')
        req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=prompt_data)
        prompt_id = json.loads(urllib.request.urlopen(req).read())['prompt_id']
        
        start_time = time.time()
        while True:
            res = urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}")
            history = json.loads(res.read())
            if prompt_id in history: break
            if time.time() - start_time > 600: raise HTTPException(status_code=504, detail="Timeout")
            time.sleep(4)
            
        # 6. Return Result
        videos = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]
        if not videos: raise HTTPException(status_code=500, detail="Render failed")
        
        videos.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
        with open(os.path.join(out_dir, videos[0]), "rb") as f:
            print(f"🔥 Sending video: {videos[0]}")
            return Response(content=f.read(), media_type="video/mp4")
