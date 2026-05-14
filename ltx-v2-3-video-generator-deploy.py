import modal
import subprocess
import time
import os
import json
import urllib.request
import shutil
from fastapi import Request, Response

# 1. MOUNT THE CORRECT VOLUME
weights_volume = modal.Volume.from_name("ltx-video-weights")

# 2. Build the Python 3.11 Image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch", "torchvision", "torchaudio", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("fastapi", "aiohttp")
    .run_commands(
        "git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI",
        "pip install -r /workspace/ComfyUI/requirements.txt",
        
        # EXACT NODES FOR YOUR LTX WORKFLOW A:
        "git clone https://github.com/city96/ComfyUI-GGUF.git /workspace/ComfyUI/custom_nodes/ComfyUI-GGUF",
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        "git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git /workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation",
        
        "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt",
        "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/requirements.txt"
    )
)

app = modal.App("ltx-v2-3-api")

# 3. The Serverless Engine
@app.cls(
    gpu="L4", 
    image=image, 
    volumes={"/mnt/weights": weights_volume},
    scaledown_window=60 # SHUT DOWN EXACTLY 1 MINUTE LATER
)
class LTXEngine:
    @modal.enter()
    def start_comfy(self):
        print("🔗 Linking Models from Volume...")
        folders_to_link = ["unet", "vae", "clip", "loras", "text_encoders", "upscale_models"]
        
        for folder in folders_to_link:
            src = f"/mnt/weights/comfyui_models/{folder}"
            dest = f"/workspace/ComfyUI/models/{folder}"
            if os.path.exists(dest) and not os.path.islink(dest):
                shutil.rmtree(dest)
            if not os.path.exists(dest):
                try:
                    os.symlink(src, dest)
                    print(f"✅ Linked: {folder}")
                except Exception as e:
                    pass

        print("🚀 Booting LTX ComfyUI Server...")
        self.process = subprocess.Popen(["python", "main.py"], cwd="/workspace/ComfyUI")
        while True:
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/")
                break
            except:
                time.sleep(1)

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request):
        data = await request.json()
        image_url = data.get("image_url")
        workflow = data.get("workflow")
        
        urllib.request.urlretrieve(image_url, "/workspace/ComfyUI/input/master_plane.png")
        
        prompt_data = json.dumps({"prompt": workflow}).encode('utf-8')
        req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=prompt_data)
        prompt_id = json.loads(urllib.request.urlopen(req).read())['prompt_id']
        
        while True:
            res = urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}")
            history = json.loads(res.read())
            if prompt_id in history:
                break
            time.sleep(2)
            
        out_dir = "/workspace/ComfyUI/output"
        videos = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]
        videos.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
        
        with open(os.path.join(out_dir, videos[0]), "rb") as f:
            video_bytes = f.read()
            
        return Response(content=video_bytes, media_type="video/mp4")
