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

# Setup optimized development container image with required compilation tooling
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
    "git clone https://github.com/comfyanonymous/ComfyUI /workspace/ComfyUI",
    "pip install -r /workspace/ComfyUI/requirements.txt"
).run_commands(
    "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
    "git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git /workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation",
    "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/requirements-no-cupy.txt",
    "git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo",
    "git clone https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes",
    "git clone https://github.com/yolain/ComfyUI-Easy-Use.git /workspace/ComfyUI/custom_nodes/ComfyUI-Easy-Use"
).run_commands(
    r"find /workspace/ComfyUI/custom_nodes -name 'requirements.txt' -exec pip install -r {} \;"
).run_commands(
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
                try: ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception: pass
            await asyncio.sleep(2)

    @modal.enter()
    def start_comfy(self):
        import boto3
        print("🔗 Running Atomic Model Folder Linker...")
        base_models_dir = "/workspace/ComfyUI/models"
        
        dirs = ["unet", "vae", "clip", "text_encoders", "text_encoder", "vfi", "checkpoints", "diffusion_models", "gguf", "loras"]
        for d in dirs:
            os.makedirs(os.path.join(base_models_dir, d), exist_ok=True)

        if os.path.exists("/mnt/weights"):
            for root_dir, _, files in os.walk("/mnt/weights"):
                for filename in files:
                    if not filename.endswith((".safetensors", ".gguf", ".pth", ".pt", ".bin")):
                        continue
                        
                    src_path = os.path.join(root_dir, filename)
                    # Cross-link everything cleanly to prevent loader mismatch issues
                    for target_dir in ["unet", "vae", "clip", "text_encoders", "text_encoder", "checkpoints", "diffusion_models", "vfi", "loras"]:
                        dest = os.path.join(base_models_dir, target_dir, filename)
                        if not os.path.exists(dest):
                            try: os.symlink(src_path, dest)
                            except FileExistsError: pass

        self.s3 = boto3.client(
            service_name='s3', 
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com", 
            aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'], 
            aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], 
            region_name="auto"
        )

        print("🚀 Launching Clean LTX Server Engine...")
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
            "--mmap", "--cache-none", "--temp-directory", "/tmp/comfy_swap", 
            "--bf16-vae", "--disable-xformers", "--fp8_e4m3fn-text-enc"        
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
                        print("⚡ LTX-2 API ONLINE!")
                        return
            except Exception:
                time.sleep(2)
        os._exit(1)

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        if x_api_key != os.environ.get("API_KEY"): 
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        body = await request.json()
        if isinstance(body, dict) and "json" in body:
            body = body["json"]

        image_url = body.get("image_url")
        workflow = body.get("workflow")
        requested_length = body.get("length")

        if isinstance(workflow, str):
            try: workflow = json.loads(workflow)
            except Exception: pass

        if isinstance(workflow, dict) and "workflow" in workflow:
            if not any("class_type" in v for v in workflow.values() if isinstance(v, dict)):
                workflow = workflow["workflow"]

        if isinstance(workflow, dict):
            if "nodes" in workflow and "last_node_id" in workflow:
                raise HTTPException(status_code=400, detail="Format Mismatch: Passed Canvas UI instead of API layout.")

        # Real model file names from your ltx-20-19b-weights volume
        target_unet = "ltx-2-19b-dev-fp8.safetensors"
        target_gemma = "gemma-3-12b-it-FP8.safetensors"
        target_connector = "ltx-2-19b-embeddings_connector_dev_bf16.safetensors"
        target_video_vae = "ltx-2-19b-dev_video_vae.safetensors"
        target_audio_vae = "ltx-2-19b-dev_audio_vae.safetensors"

        # LoRA targets mapping from storage volume
        target_distilled_lora = "ltx-2-19b-distilled-lora-384.safetensors"
        target_detailer_lora = "ltx-2-19b-ic-lora-detailer.safetensors"

        def find_node(cls_name):
            return next((k for k, v in workflow.items() if v.get("class_type") == cls_name), None)

        if isinstance(workflow, dict):
            # Dynamic frame lengths logic handling boundary equations (8N + 1)
            if requested_length is not None:
                try:
                    tgt_len = int(requested_length)
                    if (tgt_len - 1) % 8 != 0:
                        tgt_len = ((tgt_len - 1) // 8) * 8 + 1
                        if tgt_len < 9: tgt_len = 9
                    
                    video_latent_node = find_node("EmptyLTXVLatentVideo")
                    audio_latent_node = find_node("LTXVEmptyLatentAudio")
                    
                    if video_latent_node:
                        workflow[video_latent_node]["inputs"]["length"] = tgt_len
                        print(f"📏 DYNAMIC TIMEFRAME: Set Video Length to {tgt_len}")
                    if audio_latent_node:
                        workflow[audio_latent_node]["inputs"]["frames_number"] = tgt_len
                        print(f"📏 DYNAMIC TIMEFRAME: Set Audio Length to {tgt_len}")
                except Exception as e:
                    print(f"⚠️ Dynamic framing error: {e}")

            # Safe Standalone Loader Synchronizer Engine
            sanitized_workflow = {}
            for node_id, node_data in workflow.items():
                if isinstance(node_data, dict) and "class_type" in node_data:
                    if "inputs" not in node_data or node_data["inputs"] is None:
                        node_data["inputs"] = {}

                    class_type = node_data.get("class_type")

                    # 1. Standalone UNET Loader Routing (Handles both core inputs and widget lists)
                    if class_type in ["UNETLoader", "UnetLoaderGGUFAdvanced"]:
                        if "unet_name" in node_data["inputs"]:
                            node_data["inputs"]["unet_name"] = target_unet
                        if "ckpt_name" in node_data["inputs"]:
                            node_data["inputs"]["ckpt_name"] = target_unet
                        if "widgets_values" in node_data and isinstance(node_data["widgets_values"], list):
                            if len(node_data["widgets_values"]) > 0:
                                node_data["widgets_values"][0] = target_unet

                    # 2. Isolated Text Model Loader Injection
                    if class_type == "LTXAVTextEncoderLoader":
                        node_data["inputs"]["text_encoder"] = target_gemma
                        node_data["inputs"]["ckpt_name"] = target_connector

                    # 3. Clean Standalone Video VAE Loader Routing
                    if class_type in ["VAELoaderKJ", "VAELoader"]:
                        if "vae_name" in node_data["inputs"]:
                            node_data["inputs"]["vae_name"] = target_video_vae
                        if "ckpt_name" in node_data["inputs"]:
                            node_data["inputs"]["ckpt_name"] = target_video_vae

                    # 4. Clean Standalone Audio VAE Loader Routing
                    if class_type == "LTXVAudioVAELoader":
                        if "ckpt_name" in node_data["inputs"]:
                            node_data["inputs"]["ckpt_name"] = target_audio_vae
                        if "vae_name" in node_data["inputs"]:
                            node_data["inputs"]["vae_name"] = target_audio_vae

                    # 5. Failsafe Frame Rate Enforcement
                    if class_type == "LTXVEmptyLatentAudio":
                        node_data["inputs"]["frame_rate"] = 12

                    # 6. Asset Input Synchronization
                    if class_type == "LoadImage":
                        node_data["inputs"]["image"] = "master_plane.png"

                    # 7. Safe Standalone Lora Loader Routing (Matches inputs or widgets by filename keywords)
                    if class_type == "LoraLoader":
                        lora_input_name = node_data["inputs"].get("lora_name", "")
                        if "widgets_values" in node_data and isinstance(node_data["widgets_values"], list):
                            if len(node_data["widgets_values"]) > 0:
                                lora_input_name = node_data["widgets_values"][0]
                        
                        resolved_lora = None
                        if "distilled" in str(lora_input_name).lower():
                            resolved_lora = target_distilled_lora
                        elif "detail" in str(lora_input_name).lower() or "ic-lora" in str(lora_input_name).lower():
                            resolved_lora = target_detailer_lora
                        
                        if resolved_lora:
                            if "lora_name" in node_data["inputs"]:
                                node_data["inputs"]["lora_name"] = resolved_lora
                            if "widgets_values" in node_data and isinstance(node_data["widgets_values"], list):
                                if len(node_data["widgets_values"]) > 0:
                                    node_data["widgets_values"][0] = resolved_lora
                            print(f"🔗 LORA SYNC: Set Lora Loader to {resolved_lora}")

                    sanitized_workflow[str(node_id)] = node_data
            
            workflow = sanitized_workflow

        local_input = "/workspace/ComfyUI/input/master_plane.png"
        os.makedirs(os.path.dirname(local_input), exist_ok=True)
        
        if image_url and str(image_url).strip():
            from urllib.parse import urlparse
            file_key = urlparse(image_url).path.lstrip('/')
            while "//" in file_key:
                file_key = file_key.replace("//", "/")
            
            try:
                print(f"📥 Downloading input asset key from R2: {file_key}")
                self.s3.download_file("video-asset-files-storage-workflow", file_key, local_input)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"R2 asset sync failed: {str(e)}")
        else:
            from PIL import Image
            img = Image.new('RGB', (1024, 1024), color='black')
            img.save(local_input)

        out_dir = "/workspace/ComfyUI/output"
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        ram_task = asyncio.create_task(self._ram_squeezer())

        try:
            async with aiohttp.ClientSession() as session:
                print("🎨 Running Separation of Powers Generation Pipeline...")
                async with session.post("http://127.0.0.1:8188/prompt", json={"prompt": workflow}) as resp:
                    res_json = await resp.json()
                    if "error" in res_json or "prompt_id" not in res_json:
                        error_msg = res_json.get("error", res_json)
                        raise HTTPException(status_code=400, detail=f"Invalid JSON: {error_msg}")
                    prompt_id = res_json['prompt_id']

                start_time = time.time()
                while True:
                    if self.process.poll() is not None:
                        raise HTTPException(status_code=500, detail="Backend server execution failure.")
                    async with session.get("http://127.0.0.1:8188/history") as hist_resp:
                        if hist_resp.status == 200:
                            history = await hist_resp.json()
                            if prompt_id in history:
                                break
                    if time.time() - start_time > 2400:
                        raise HTTPException(status_code=504, detail="Execution timeout.")
                    await asyncio.sleep(5)

            videos = [v for v in os.listdir(out_dir) if v.endswith(".mp4")]
            if not videos:
                raise HTTPException(status_code=500, detail="No output video files were found.")
                
            videos.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
            with open(os.path.join(out_dir, videos[0]), "rb") as f:
                return Response(content=f.read(), media_type="video/mp4")
        finally:
            ram_task.cancel()
