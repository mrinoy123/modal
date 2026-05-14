import modal
import subprocess
import time
import os
import json
import urllib.request
from fastapi import Request, Response

# 1. Mount the exact storage volume you confirmed
weights_volume = modal.Volume.from_name("modal-comfyui-storage")

# 2. Build the Python 3.10 Image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch", "torchvision", "torchaudio", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("fastapi", "aiohttp")
    .run_commands(
        "git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI",
        "pip install -r /workspace/ComfyUI/requirements.txt",
        
        # Sonic Specific Nodes (NO RIFE)
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        "git clone https://github.com/comfyanonymous/ComfyUI_Sonic.git /workspace/ComfyUI/custom_nodes/ComfyUI_Sonic",
        
        "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI_Sonic/requirements.txt"
    )
)

app = modal.App("sonic-lypsync-api")

# 3. The Serverless Engine
@app.cls(
    gpu="L4", 
    image=image, 
    # Mount the entire models directory so Sonic can find SVD and its specific checkpoints
    volumes={"/workspace/ComfyUI/models": weights_volume},
    container_idle_timeout=60 # SHUT DOWN 1 MINUTE AFTER n8n FINISHES
)
class SonicEngine:
    @modal.enter()
    def start_comfy(self):
        print("🗣️ Booting Sonic Lip-Sync Server...")
        self.process = subprocess.Popen(["python", "main.py"], cwd="/workspace/ComfyUI")
        while True:
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/")
                break
            except:
                time.sleep(1)

    @modal.web_endpoint(method="POST")
    async def generate(self, request: Request):
        data = await request.json()
        image_url = data.get("image_url")
        audio_url = data.get("audio_url") # Sonic needs an audio file URL from n8n
        workflow = data.get("workflow")
        
        # Ensure the ComfyUI LoadImage and LoadAudio nodes expect these exact filenames
        urllib.request.urlretrieve(image_url, "/workspace/ComfyUI/input/master_face.png")
        urllib.request.urlretrieve(audio_url, "/workspace/ComfyUI/input/narration.wav")
        
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
