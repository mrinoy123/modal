import modal
import subprocess
import time
import os
import json
import urllib.request
import shutil
from fastapi import Request, Response, HTTPException, Header
from typing import Optional

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
    timeout=600 # 10 minute timeout for long renders
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
            if not os.path.exists(dest) and os.path.exists(src):
                try:
                    os.symlink(src, dest)
                    print(f"✅ Linked: {folder}")
                except Exception as e:
                    print(f"❌ Failed to link {folder}: {e}")

        print("🚀 Booting LTX ComfyUI Server...")
        # Run main.py with --listen to ensure accessibility within the container
        self.process = subprocess.Popen(["python", "main.py", "--listen", "127.0.0.1"], cwd="/workspace/ComfyUI")
        
        # Wait for server to be ready
        for i in range(30):
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/")
                print("⚡ ComfyUI is Ready!")
                break
            except:
                time.sleep(2)

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        # 1. Security Gate
        expected_key = os.environ.get("API_KEY")
        if not x_api_key or x_api_key != expected_key:
            print(f"🚫 Unauthorized. Key provided: {x_api_key}")
            raise HTTPException(status_code=403, detail="Unauthorized: GPU Access Denied.")

        # 2. Parse Incoming Data
        body = await request.json()
        image_url = body.get("image_url")
        workflow_raw = body.get("workflow")
        
        # Handle n8n sending workflow as a string
        if isinstance(workflow_raw, str):
            workflow = json.loads(workflow_raw)
        else:
            workflow = workflow_raw

        # 3. Clean Output Folder (Important: Don't return old videos)
        out_dir = "/workspace/ComfyUI/output"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        
        # 4. Download Input Image
        print(f"🎬 Downloading start frame: {image_url}")
        # Some CDNs block default python user-agents
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(image_url, "/workspace/ComfyUI/input/master_plane.png")
        
        # 5. Send Prompt to ComfyUI
        print("🌀 Injecting Workflow into Engine...")
        prompt_data = json.dumps({"prompt": workflow}).encode('utf-8')
        req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=prompt_data)
        try:
            response = urllib.request.urlopen(req)
            prompt_id = json.loads(response.read())['prompt_id']
            print(f"✅ Render Started. Prompt ID: {prompt_id}")
        except Exception as e:
            print(f"❌ ComfyUI API Error: {e}")
            raise HTTPException(status_code=500, detail="ComfyUI refused the workflow.")
        
        # 6. Wait for Completion
        start_time = time.time()
        while True:
            res = urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}")
            history = json.loads(res.read())
            if prompt_id in history:
                break
            
            # Timeout after 8 minutes
            if time.time() - start_time > 480:
                raise HTTPException(status_code=504, detail="Render Timeout.")
            
            time.sleep(3)
            
        # 7. Find and Return Video
        videos = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]
        if not videos:
            print("❌ No video found in output folder.")
            raise HTTPException(status_code=500, detail="Generation complete but video file missing.")

        # Sort by newest
        videos.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
        video_path = os.path.join(out_dir, videos[0])
        
        with open(video_path, "rb") as f:
            video_bytes = f.read()
            
        print(f"🔥 Successfully rendered {videos[0]}. Sending back to n8n.")
        return Response(content=video_bytes, media_type="video/mp4")
