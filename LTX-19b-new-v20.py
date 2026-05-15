import modal
import subprocess
import time
import os
import json
import shutil
import threading
import aiohttp
import urllib.request
import asyncio
import ctypes
from fastapi import Request, Response, HTTPException, Header
from typing import Optional

# ==========================================
# 1. IMAGE DEFINITION
# ==========================================
base_image = modal.Image.from_registry(
    "nvidia/cuda:12.5.1-devel-ubuntu24.04", 
    add_python="3.12"
).apt_install(
    "git", "wget", "ffmpeg", "libgl1", "libglib2.0-0", 
    "build-essential", "ninja-build", "cmake", "clang", "llvm"
)

build_image = base_image.env({
    "CUDA_HOME": "/usr/local/cuda",
    "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", ""),
    "FORCE_CUDA": "1",
    "TORCH_CUDA_ARCH_LIST": "8.9", 
    "MAX_JOBS": "1",
    "CC": "gcc",
    "CXX": "g++"
}).pip_install(
    "torch==2.5.1", "torchvision", "torchaudio", 
    index_url="https://download.pytorch.org/whl/cu124"
).pip_install(
    "fastapi", "aiohttp", "boto3", "triton>=3.1.0", 
    "ninja", "setuptools>=70.0.0", "wheel", "pip>=24.0"
)

compiled_image = build_image.run_commands(
    "git clone https://github.com/thu-ml/SageAttention.git /workspace/SageAttention",
    "cd /workspace/SageAttention && pip install --no-build-isolation ."
)

final_image = compiled_image.run_commands(
    "git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI",
    "pip install -r /workspace/ComfyUI/requirements.txt"
).run_commands(
    "git clone https://github.com/smthemex/ComfyUI_LTX2_SM.git /workspace/ComfyUI/custom_nodes/ComfyUI_LTX2_SM",
    "git clone https://github.com/city96/ComfyUI-GGUF.git /workspace/ComfyUI/custom_nodes/ComfyUI-GGUF",
    "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
    "git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git /workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation",
    "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/requirements-no-cupy.txt",
    "git clone https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes",
    "git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo",
    "git clone https://github.com/yolain/ComfyUI-Easy-Use.git /workspace/ComfyUI/custom_nodes/ComfyUI-Easy-Use",
    "git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git /workspace/ComfyUI/custom_nodes/ComfyUI-Impact-Pack"
).run_commands(r"find /workspace/ComfyUI/custom_nodes -name 'requirements.txt' -exec pip install -r {} \;").run_commands(
    # 🔥 SURGICAL HARD DRIVE PATCH v2: Stitches on GPU, then pushes to CPU for MP4 saving
    "python -c \"import re; file='/workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/vfi_models/rife/__init__.py'; data=open(file).read(); data=re.sub(r'torch\\.cat\\(output_frames, dim=0\\)', 'torch.cat([f.to(output_frames[0].device) for f in output_frames], dim=0).cpu()', data); open(file, 'w').write(data)\""
)

app = modal.App("ltx-2-19b-v20-api")
weights_volume = modal.Volume.from_name("ltx-20-19b-weights")

@app.cls(
    gpu="L4", 
    image=final_image, 
    volumes={"/mnt/weights": weights_volume},
    secrets=[modal.Secret.from_name("video-generator-workflow")], 
    memory=8192,
    scaledown_window=60,
    timeout=3600 
)
class LTXEngine:
    def _log_reader(self):
        for line in iter(self.process.stdout.readline, ""):
            if line: print(f"[ComfyUI] {line.strip()}")

    async def _ram_squeezer(self):
        print("🛡️ RAM Watchdog Active. Forcing Linux to drop page cache...")
        while True:
            try:
                with open('/proc/sys/vm/drop_caches', 'w') as f:
                    f.write('1\n')
            except Exception:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            await asyncio.sleep(2)

    @modal.enter()
    def start_comfy(self):
        import boto3
        
print("🔗 Running Aggressive Fuzzy Linker...")
        base_models_dir = "/workspace/ComfyUI/models"
        
        # 🔥 Added "gguf" to the allowed directories list
        dirs = ["unet", "vae", "clip", "text_encoders", "vfi", "checkpoints", "diffusion_models", "gguf"]
        for d in dirs:
            os.makedirs(os.path.join(base_models_dir, d), exist_ok=True)

        def safe_link(source, destination):
            if not os.path.exists(destination):
                os.symlink(source, destination)

        if os.path.exists("/mnt/weights"):
            for root_dir, _, files in os.walk("/mnt/weights"):
                for filename in files:
                    if not filename.endswith((".safetensors", ".gguf", ".pth", ".pt", ".bin")):
                        continue
                        
                    src_path = os.path.join(root_dir, filename)
                    fn = filename.lower()
                    linked_to = []

                    def link_it(target_dir):
                        dest = os.path.join(base_models_dir, target_dir, filename)
                        if not os.path.exists(dest):
                            os.symlink(src_path, dest)
                            linked_to.append(target_dir)

                    # --- AGGRESSIVE FUZZY ROUTING LOGIC ---
                    
                    if "unet" in fn or "ltx-2-19b-dev-q3" in fn:
                        link_it("unet")
                        link_it("diffusion_models")
                        
                    if "gemma" in fn or "clip" in fn or "t5" in fn:
                        link_it("clip")
                        link_it("text_encoders")
                        link_it("checkpoints")
                        # 🔥 THE FIX: SM Node natively looks in gguf and unet for its clip inputs!
                        link_it("gguf") 
                        link_it("unet") 
                        link_it("diffusion_models")
                        
                    if "connector" in fn:
                        link_it("checkpoints") 
                        link_it("clip")
                        link_it("unet")
                        
                    if "audio_vae" in fn:
                        link_it("checkpoints")
                        link_it("vae")
                        
                    if "vae" in fn and "audio" not in fn:
                        link_it("vae")
                        
                    if "rife" in fn or "vfi" in fn:
                        link_it("vfi")
                        
                    if linked_to:
                        print(f"✅ Linked [{filename}] -> {linked_to}")

        self.s3 = boto3.client(
            service_name='s3', 
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com", 
            aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'], 
            aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], 
            region_name="auto"
        )

        print("🚀 Launching LTX-2 19B Engine (Direct GPU Safetensors Streaming)...")
        
        os.makedirs("/tmp/comfy_swap", exist_ok=True)
        os.makedirs("/tmp/hf_offload", exist_ok=True)

        env_vars = os.environ.copy()
        env_vars["TORCH_NUM_THREADS"] = "1"
        env_vars["OMP_NUM_THREADS"] = "1"
        env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
        env_vars["CUDA_MODULE_LOADING"] = "LAZY" 
        env_vars["MALLOC_TRIM_THRESHOLD_"] = "65536" 
        env_vars["HF_HUB_OFFLOAD_DIR"] = "/tmp/hf_offload"
        
        self.process = subprocess.Popen([
            "python", "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--gpu-only",             
            "--cache-none",           
            "--mmap",                 
            "--temp-directory", "/tmp/comfy_swap", 
            "--disable-smart-memory", 
            "--bf16-vae",
            "--disable-xformers",
            "--fp8_e4m3fn-text-enc"   
        ], cwd="/workspace/ComfyUI", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env_vars)
        
        self.t = threading.Thread(target=self._log_reader, daemon=True)
        self.t.start()

        start_time = time.time()
        while time.time() - start_time < 300:
            if self.process.poll() is not None:
                print("❌ Startup Crash!")
                os._exit(1)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8188/", timeout=1) as response:
                    if response.status == 200:
                        print("⚡ LTX-2 19B API ONLINE!")
                        return
            except Exception:
                time.sleep(2)
        os._exit(1)

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        if x_api_key != os.environ.get("API_KEY"): 
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        body = await request.json()
        image_url, workflow = body.get("image_url"), body.get("workflow")
        if isinstance(workflow, str): workflow = json.loads(workflow)

        local_input = "/workspace/ComfyUI/input/master_plane.png"
        os.makedirs(os.path.dirname(local_input), exist_ok=True)
        file_key = image_url.split(".dev/")[-1]
        self.s3.download_file("video-asset-files-storage-workflow", file_key, local_input)

        out_dir = "/workspace/ComfyUI/output"
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        ram_task = asyncio.create_task(self._ram_squeezer())

        try:
            async with aiohttp.ClientSession() as session:
                print(f"🎨 Processing 19B Joint Audio-Video Workflow...")
                async with session.post("http://127.0.0.1:8188/prompt", json={"prompt": workflow}) as resp:
                    res_json = await resp.json()
                    
                    if "error" in res_json or "prompt_id" not in res_json:
                        error_msg = res_json.get("error", res_json)
                        print(f"❌ ComfyUI Rejected Prompt: {error_msg}")
                        raise HTTPException(status_code=400, detail=f"Invalid Workflow JSON: {error_msg}")
                    
                    prompt_id = res_json['prompt_id']

                start_time = time.time()
                while True:
                    if self.process.poll() is not None:
                        print("❌ GPU CRASHED. Check Modal Logs.")
                        os._exit(1) 

                    async with session.get(f"http://127.0.0.1:8188/history/{prompt_id}") as resp:
                        if resp.status == 200:
                            history = await resp.json()
                            if prompt_id in history: break
                    
                    if time.time() - start_time > 2400: raise HTTPException(status_code=504, detail="Timeout")
                    await asyncio.sleep(5)

            videos = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]
            if not videos:
                raise HTTPException(status_code=500, detail="Generation finished but no MP4 was found in output folder.")
                
            videos.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
            
            with open(os.path.join(out_dir, videos[0]), "rb") as f:
                return Response(content=f.read(), media_type="video/mp4")
        finally:
            ram_task.cancel()
