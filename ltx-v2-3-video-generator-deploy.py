import modal
import subprocess
import time
import os
import json
import urllib.request
import shutil
from fastapi import Request, Response, HTTPException, Header
from typing import Optional

# 1. Define the Volume for Persistent Models
weights_volume = modal.Volume.from_name("ltx-video-weights")

# 2. Build the Image
# Using CUDA 12.5.1 + Ubuntu 24.04 solves the GLIBCXX version mismatch
image = (
    modal.Image.from_registry("nvidia/cuda:12.5.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "wget", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "build-essential")
    .pip_install(
        "torch==2.5.1", 
        "torchvision", 
        "torchaudio", 
        index_url="https://download.pytorch.org/whl/cu124"
    )
    .pip_install("fastapi", "aiohttp", "boto3", "triton>=3.1.0") 
    .run_commands(
        # Precompiled SageAttention 2.2.0 for Python 3.12 / Ubuntu 24.04
        "pip install https://huggingface.co/Kijai/PrecompiledWheels/resolve/main/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl",
        
        # Clone ComfyUI
        "git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI",
        "pip install -r /workspace/ComfyUI/requirements.txt",
        
        # Install Custom Nodes
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
    scaledown_window=120, # Keep alive for 2 mins to avoid cold start on batching
    timeout=1200 
)
class LTXEngine:
    @modal.enter()
    def start_comfy(self):
        import boto3
        
        # Robust Symlinking
        print("🔗 Mapping Models from Volume...")
        model_dirs = ["unet", "vae", "clip", "text_encoders", "upscale_models"]
        for d in model_dirs:
            src = f"/mnt/weights/comfyui_models/{d}"
            dest = f"/workspace/ComfyUI/models/{d}"
            
            if os.path.exists(src):
                if os.path.exists(dest) and not os.path.islink(dest):
                    shutil.rmtree(dest)
                if not os.path.exists(dest):
                    # Ensure parent exists
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    os.symlink(src, dest)
                    print(f"✅ Linked {d}")

        # Initialize Cloudflare R2
        self.s3 = boto3.client(
            service_name='s3',
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
            region_name="auto"
        )

        print("🚀 Launching ComfyUI Server...")
        # Redirect stderr to stdout to see errors in Modal logs
        self.process = subprocess.Popen([
            "python", "main.py", 
            "--listen", "127.0.0.1", 
            "--port", "8188",
            "--use-sage-attention",
            "--highvram",
            "--bf16-vae",
            "--disable-smart-memory"
        ], cwd="/workspace/ComfyUI", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Wait for port to open
        ready = False
        for i in range(60):
            if self.process.poll() is not None:
                # Get the error message from the failed process
                err_output = self.process.stdout.read()
                print(f"❌ ComfyUI failed to start:\n{err_output}")
                raise RuntimeError("ComfyUI crashed on startup.")
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/", timeout=2)
                print("⚡ ComfyUI is Online!")
                ready = True
                break
            except:
                time.sleep(2)
        
        if not ready:
            raise TimeoutError("ComfyUI health check timed out.")

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        # 1. Auth Check
        if x_api_key != os.environ.get("API_KEY"):
            raise HTTPException(status_code=403, detail="Unauthorized")

        # 2. Data Parsing
        data = await request.json()
        image_url = data.get("image_url") 
        workflow_raw = data.get("workflow")
        workflow = json.loads(workflow_raw) if isinstance(workflow_raw, str) else workflow_raw

        # 3. R2 Image Download
        # The n8n configure-workflow sets Node 29's image to "master_plane.png"
        local_input_path = "/workspace/ComfyUI/input/master_plane.png"
        os.makedirs("/workspace/ComfyUI/input", exist_ok=True)
        
        file_key = image_url.split(".dev/")[-1] 
        print(f"📥 Fetching: {file_key}")
        try:
            self.s3.download_file("video-asset-files-storage-workflow", file_key, local_input_path)
            print("✅ Image Ready")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"R2 Error: {str(e)}")

        # 4. Clean Output
        out_dir = "/workspace/ComfyUI/output"
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        # 5. Execute Render
        print("🎨 Sending Prompt to ComfyUI...")
        try:
            prompt_data = json.dumps({"prompt": workflow}).encode('utf-8')
            req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=prompt_data)
            res_data = json.loads(urllib.request.urlopen(req).read())
            prompt_id = res_data['prompt_id']
        except Exception as e:
            print(f"❌ ComfyUI API Error: {e}")
            raise HTTPException(status_code=500, detail="Failed to queue prompt")
        
        # 6. Poll for completion
        start_time = time.time()
        while True:
            # Check if server is still alive
            if self.process.poll() is not None:
                raise HTTPException(status_code=500, detail="ComfyUI server died during render")

            try:
                res = urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}")
                history = json.loads(res.read())
                if prompt_id in history: 
                    print("✅ Render Finished!")
                    break
            except:
                pass # Server busy

            if time.time() - start_time > 1100: 
                raise HTTPException(status_code=504, detail="Generation Timeout")
            time.sleep(5)
            
        # 7. Collect and Return File
        videos = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]
        if not videos:
            raise HTTPException(status_code=500, detail="Render completed but no video file found")
        
        # Sort by modification time to get the latest
        videos.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
        target_video = os.path.join(out_dir, videos[0])

        with open(target_video, "rb") as f:
            return Response(content=f.read(), media_type="video/mp4")
