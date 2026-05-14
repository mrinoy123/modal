import modal
import subprocess
import time
import os
import json
import urllib.request
import shutil
from fastapi import Request, Response, HTTPException, Header
from typing import Optional

# 1. MOUNT THE CORRECT VOLUME FOR SONIC
weights_volume = modal.Volume.from_name("sonic-video-weights")

# 2. Build the Python 3.10 Image (Crucial for Sonic stability)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch", "torchvision", "torchaudio", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("fastapi", "aiohttp")
    .run_commands(
        "git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI",
        "pip install -r /workspace/ComfyUI/requirements.txt",
        
        # Sonic Specific Nodes
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        "git clone https://github.com/smthemex/ComfyUI_Sonic.git /workspace/ComfyUI/custom_nodes/ComfyUI_Sonic",
        
        "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI_Sonic/requirements.txt"
    )
)

app = modal.App("sonic-lypsync-api")

# 3. The Serverless Engine with Security Gate
@app.cls(
    gpu="L4", 
    image=image, 
    volumes={"/mnt/weights": weights_volume},
    # Link the secret vault you created in the Modal dashboard
    secrets=[modal.Secret.from_name("video-generator-workflow")],
    scaledown_window=60 
)
class SonicEngine:
    @modal.enter()
    def start_comfy(self):
        print("🔗 Linking Sonic Models from Volume...")
        folders_to_link = ["checkpoints", "clip_vision", "face_detection", "sonic", "vae", "unet", "clip"]
        
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

        print("🗣️ Booting Sonic Lip-Sync Server...")
        self.process = subprocess.Popen(["python", "main.py"], cwd="/workspace/ComfyUI")
        while True:
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/")
                break
            except:
                time.sleep(1)

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        # 🛡️ SECURITY CHECK
        # Matches the 'API_KEY' inside your Modal Secret 'video-generator-workflow'
        expected_key = os.environ.get("API_KEY")
        
        if x_api_key != expected_key:
            print("🚫 Unauthorized request attempt.")
            raise HTTPException(
                status_code=403, 
                detail="Unauthorized: GPU Access Denied."
            )

        # ⚙️ AUTH PASSED: Process Lip-Sync
        data = await request.json()
        image_url = data.get("image_url")
        audio_url = data.get("audio_url") 
        workflow = data.get("workflow")
        
        print(f"🎤 Starting Lip-Sync for audio: {audio_url}")
        
        # Download assets into ComfyUI input folder
        urllib.request.urlretrieve(image_url, "/workspace/ComfyUI/input/master_face.png")
        urllib.request.urlretrieve(audio_url, "/workspace/ComfyUI/input/narration.wav")
        
        # Trigger ComfyUI workflow
        prompt_data = json.dumps({"prompt": workflow}).encode('utf-8')
        req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=prompt_data)
        prompt_id = json.loads(urllib.request.urlopen(req).read())['prompt_id']
        
        # Wait for the render to finish
        while True:
            res = urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}")
            history = json.loads(res.read())
            if prompt_id in history:
                break
            time.sleep(2)
            
        # Locate the latest rendered video
        out_dir = "/workspace/ComfyUI/output"
        videos = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]
        videos.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
        
        if not videos:
            raise HTTPException(status_code=500, detail="Lip-sync generation failed.")

        with open(os.path.join(out_dir, videos[0]), "rb") as f:
            video_bytes = f.read()
            
        print("✅ Sonic render complete. Sending file.")
        return Response(content=video_bytes, media_type="video/mp4")
