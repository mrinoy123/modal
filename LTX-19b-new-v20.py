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
# PART 1: Infrastructure Configuration & Base Dependencies
# ==========================================

# Setup optimized development container image with required compilation tooling
base_image = modal.Image.from_registry(
    "nvidia/cuda:12.5.1-devel-ubuntu24.04", 
    add_python="3.12"
).apt_install(
    "git", "wget", "ffmpeg", "libgl1", "libglib2.0-0", 
    "build-essential", "ninja-build", "cmake", "clang", "llvm"
)

# Build image with explicit cache-locked dependencies
build_image = base_image.env({
    "CUDA_HOME": "/usr/local/cuda",
    "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", ""),
    "FORCE_CUDA": "1",
    "TORCH_CUDA_ARCH_LIST": "8.9", 
    "MAX_JOBS": "1",
    "CC": "gcc",
    "CXX": "g++"
}).pip_install(
    "fastapi", "aiohttp", "boto3", "triton>=3.1.0", 
    "ninja", "setuptools>=70.0.0", "wheel", "pip>=24.0"
).pip_install(
    "pandas", "numexpr", "pytz", "python-dateutil", 
    "scipy", "matplotlib", "colorama", "librosa", "soundfile", 
    "decord", "imageio", "scikit-image", "numba", "einops", 
    "transformers", "diffusers", "accelerate", "bitsandbytes"
)

# Global variables defining immutable model file mappings
TARGET_UNET = "ltx-2-19b-distilled-fp8.safetensors"
TARGET_GEMMA = "gemma-3-12b-it-FP8.safetensors"
TARGET_CONNECTOR = "ltx-2-19b-embeddings_connector_dev_bf16.safetensors"
TARGET_VIDEO_VAE = "ltx-2-19b-dev_video_vae.safetensors"
TARGET_AUDIO_VAE = "ltx-2-19b-dev_audio_vae.safetensors"
TARGET_DISTILLED_LORA = "ltx-2-19b-distilled-lora-384.safetensors"
TARGET_DETAILER_LORA = "ltx-2-19b-ic-lora-detailer.safetensors"

# ==========================================
# PART 2: Secure Build-Time Appliance Baker
# ==========================================

def bake_private_workflow_into_image():
    """
    Executes strictly during the 'modal deploy' image assembly layer.
    Securely accesses the private R2 bucket, downloads the JSON, wires the target
    model weights permanently to the nodes, and freezes it into the container disk
    to serve as the hyper-fast appliance fallback.
    """
    import boto3
    import json
    import os

    print("🏗️ Build Phase: Securely fetching master workflow from private Cloudflare R2...")
    
    # Authenticate using the temporary build-time secrets
    s3 = boto3.client(
        service_name='s3', 
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com", 
        aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'], 
        aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], 
        region_name="auto"
    )

    # Download the private blueprint
    raw_path = "/tmp/raw_workflow.json"
    try:
        s3.download_file(
            "video-asset-files-storage-workflow", 
            "Comfyui-workflows-json/new-workflow-modified-vastly-change(api)new.json", 
            raw_path
        )

        with open(raw_path, "r") as f:
            wf_data = json.load(f)

        if "workflow" in wf_data and not any("class_type" in v for v in wf_data.values() if isinstance(v, dict)):
            wf_data = wf_data["workflow"]

        print("🏗️ Build Phase: Executing Hardcoded Model Injection Mapping...")
        
        for node_id, node in wf_data.items():
            if not isinstance(node, dict) or "inputs" not in node:
                continue
                
            cls = node.get("class_type", "")
            inputs = node["inputs"]
            
            if cls in ["UNETLoader", "UnetLoaderGGUFAdvanced", "CheckpointLoaderSimple"]:
                inputs["unet_name"] = TARGET_UNET
                inputs["ckpt_name"] = TARGET_UNET
                if "widgets_values" in node and isinstance(node["widgets_values"], list) and len(node["widgets_values"]) > 0:
                    node["widgets_values"][0] = TARGET_UNET
            elif cls == "LTXAVTextEncoderLoader":
                inputs["text_encoder"] = TARGET_GEMMA
                inputs["ckpt_name"] = TARGET_CONNECTOR
            elif cls in ["VAELoaderKJ", "VAELoader"]:
                inputs["vae_name"] = TARGET_VIDEO_VAE
                inputs["ckpt_name"] = TARGET_VIDEO_VAE
            elif cls == "LTXVAudioVAELoader":
                inputs["ckpt_name"] = TARGET_AUDIO_VAE
                inputs["vae_name"] = TARGET_AUDIO_VAE
            elif cls == "LoraLoader":
                current_lora = str(inputs.get("lora_name", "")).lower()
                resolved = TARGET_DISTILLED_LORA
                if "detail" in current_lora or "ic-lora" in current_lora:
                    resolved = TARGET_DETAILER_LORA
                inputs["lora_name"] = resolved
                if "widgets_values" in node and isinstance(node["widgets_values"], list) and len(node["widgets_values"]) > 0:
                    node["widgets_values"][0] = resolved
            elif cls == "DenoMultiImageLoader":
                inputs["image_paths"] = "input/dynamic_guides"

        os.makedirs("/workspace", exist_ok=True)
        with open("/workspace/prebaked_workflow.json", "w") as f:
            json.dump(wf_data, f, indent=2)
            
        print("🏗️ Build Phase: Successfully sealed prebaked_workflow.json to container disk!")
    except Exception as e:
        print(f"⚠️ Build Phase Issue (Fallback Skipped): {e}")


final_image = (
    build_image.pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124"
    )
    .pip_install("numpy==1.26.4", "diffusers", "accelerate", "transformers")
    
    # 1. Clone core ComfyUI and pin version
    .run_commands(
        "git clone https://github.com/comfyanonymous/ComfyUI /workspace/ComfyUI",
        "cd /workspace/ComfyUI && git checkout $(git rev-list -n 1 --before='2026-03-01' HEAD)",
        "pip install -r /workspace/ComfyUI/requirements.txt"
    )
    
    # 2. Clone custom nodes sequentially
    .run_commands(
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        "git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo",
        "git clone https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes",
        "git clone https://github.com/yolain/ComfyUI-Easy-Use.git /workspace/ComfyUI/custom_nodes/ComfyUI-Easy-Use",
        "git clone https://github.com/Deno2026/comfyui-deno-custom-nodes.git /workspace/ComfyUI/custom_nodes/comfyui-deno-custom-nodes",
        "git clone https://github.com/cubiq/ComfyUI_essentials.git /workspace/ComfyUI/custom_nodes/ComfyUI_essentials",
        "git clone https://github.com/FizzleDorf/ComfyUI_FizzNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI_FizzNodes",
        "git clone https://github.com/SquirrelRat/MultiString-Prompts.git /workspace/ComfyUI/custom_nodes/MultiString-Prompts",
        "git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git /workspace/ComfyUI/custom_nodes/ComfyUI-Custom-Scripts",
        "git clone https://github.com/IvanRybakov/comfyui-node-int-to-string-convertor.git /workspace/ComfyUI/custom_nodes/comfyui-node-int-to-string-convertor"
    )
    
    # 3. Handle checkout constraints
    .run_commands(
        "cd /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo && git checkout $(git rev-list -n 1 --before='2026-03-01' HEAD)",
        "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/requirements.txt",
        "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt"
    )
    
    # 4. Apply critical engine patches
    .run_commands(
        "sed -i 's/final_pooled_output = torch.cat(pooled_out, dim=0)/final_pooled_output = torch.cat([p for p in pooled_out if p is not None], dim=0) if any(p is not None for p in pooled_out) else None/g' /workspace/ComfyUI/custom_nodes/ComfyUI_FizzNodes/BatchFuncs.py",
        "sed -i 's/guider.raw_conds/guider.inner_set_conds/g' /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/looping_sampler.py"
    )

    # 5. Execute Private R2 Ingestion
    .run_function(bake_private_workflow_into_image, secrets=[modal.Secret.from_name("video-generator-workflow")])
)

# ==========================================
# PART 3: Reclamation Loops & Storage Linker
# ==========================================

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
                with open('/proc/sys/vm/drop_caches', 'w') as f: f.write('1\n')
            except Exception:
                try: ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception: pass
            await asyncio.sleep(2)

    @modal.enter()
    def start_comfy(self):
        import boto3
        print("🔗 Running Atomic Model Folder Linker...")
        base_models_dir = "/workspace/ComfyUI/models"
        
        dirs = ["unet", "vae", "clip", "text_encoders", "text_encoder", "checkpoints", "diffusion_models", "gguf", "loras"]
        for d in dirs: os.makedirs(os.path.join(base_models_dir, d), exist_ok=True)

        exact_mapping = {
            "gemma-3-12b-it-FP8.safetensors": ["text_encoders", "text_encoder"],
            "ltx-2-19b-embeddings_connector_dev_bf16.safetensors": ["checkpoints"],
            "ltx-2-19b-distilled-fp8.safetensors": ["unet", "diffusion_models"],
            "ltx-2-19b-ic-lora-detailer.safetensors": ["loras"],
            "ltx-2-19b-dev_audio_vae.safetensors": ["checkpoints"],
            "ltx-2-19b-dev_video_vae.safetensors": ["vae"]
        }

        if os.path.exists("/mnt/weights"):
            for root_dir, _, files in os.walk("/mnt/weights"):
                for filename in files:
                    if filename in exact_mapping:
                        src_path = os.path.join(root_dir, filename)
                        for target_dir in exact_mapping[filename]:
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
            "--mmap-torch-files", "--cache-none", "--temp-directory", "/tmp/comfy_swap", 
            "--bf16-vae", "--disable-xformers", "--fp8_e4m3fn-text-enc"        
        ], cwd="/workspace/ComfyUI", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env_vars)
        
        self.t = threading.Thread(target=self._log_reader, daemon=True)
        self.t.start()

        start_time = time.time()
        while time.time() - start_time < 300:
            if self.process.poll() is not None: os._exit(1)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8188/", timeout=1) as response:
                    if response.status == 200:
                        print("⚡ LTX-2 API ONLINE!")
                        return
            except Exception: time.sleep(2)
        os._exit(1)

# ==========================================
# PART 4: Hybrid Gateway Interceptor & Runtime Execution
# ==========================================

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        if x_api_key != os.environ.get("API_KEY"): 
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        body = await request.json()
        if isinstance(body, dict):
            if "json" in body: body = body["json"]
            elif "body" in body: body = body["body"]

        image_url = body.get("image_url")
        requested_length = body.get("length", 73)
        width = body.get("width", 384)
        height = body.get("height", 480)
        prompt_timeline = body.get("prompts", {})
        negative_prompt = body.get("negative", "worst quality, blurry")
        filename_prefix = body.get("filename", "output_scene")
        
        # Capture the dynamic workflow sent from n8n (if it exists)
        inbound_wf = body.get("workflow")

        # ⚡ INTELLIGENT HYBRID OVERRIDE HANDSHAKE
        workflow = None
        if inbound_wf:
            if isinstance(inbound_wf, str):
                try: workflow = json.loads(inbound_wf)
                except Exception: pass
            elif isinstance(inbound_wf, dict):
                workflow = inbound_wf

        # Fallback to local image copy if n8n passes no workflow parameters
        if not workflow or not isinstance(workflow, dict):
            print("💡 Appliance Fallback: Processing script via internal prebaked template layout...")
            with open("/workspace/prebaked_workflow.json", "r") as f:
                workflow = json.load(f)
        else:
            print("⚡ Dynamic Ingress Override: Processing live mutated template directly from n8n network...")

        if "workflow" in workflow and not any("class_type" in v for v in workflow.values() if isinstance(v, dict)):
            workflow = workflow["workflow"]

        # ⚡ AGGRESSIVE FUZZY LINKER RUNTIME FORCING 
        # (Guarantees no filename conflicts, even if n8n modified the graph)
        def fuzzy_linker(wf_data):
            for node_id, node in wf_data.items():
                if not isinstance(node, dict) or "inputs" not in node: continue
                cls = node.get("class_type", "")
                inputs = node["inputs"]
                
                if cls in ["UNETLoader", "UnetLoaderGGUFAdvanced", "CheckpointLoaderSimple"]:
                    inputs["unet_name"] = TARGET_UNET
                    inputs["ckpt_name"] = TARGET_UNET
                    if "widgets_values" in node and isinstance(node["widgets_values"], list) and len(node["widgets_values"]) > 0:
                        node["widgets_values"][0] = TARGET_UNET
                elif cls == "LTXAVTextEncoderLoader":
                    inputs["text_encoder"] = TARGET_GEMMA
                    inputs["ckpt_name"] = TARGET_CONNECTOR
                elif cls in ["VAELoaderKJ", "VAELoader"]:
                    inputs["vae_name"] = TARGET_VIDEO_VAE
                    inputs["ckpt_name"] = TARGET_VIDEO_VAE
                elif cls == "LTXVAudioVAELoader":
                    inputs["ckpt_name"] = TARGET_AUDIO_VAE
                    inputs["vae_name"] = TARGET_AUDIO_VAE
                elif cls == "LoraLoader":
                    current_lora = str(inputs.get("lora_name", "")).lower()
                    resolved = TARGET_DISTILLED_LORA
                    if "detail" in current_lora or "ic-lora" in current_lora:
                        resolved = TARGET_DETAILER_LORA
                    inputs["lora_name"] = resolved
                    if "widgets_values" in node and isinstance(node["widgets_values"], list) and len(node["widgets_values"]) > 0:
                        node["widgets_values"][0] = resolved
                elif cls == "DenoMultiImageLoader":
                    inputs["image_paths"] = "input/dynamic_guides"
            return wf_data

        workflow = fuzzy_linker(workflow)

        # 3. DYNAMICALLY APPLY N8N CONFIGURATIONS TO THE ACTIVE NODES
        try:
            tgt_len = int(requested_length)
            if (tgt_len - 1) % 8 != 0:
                tgt_len = ((tgt_len - 1) // 8) * 8 + 1
                if tgt_len < 9: tgt_len = 9

            schedule_lines = []
            for frame, p_text in prompt_timeline.items():
                escaped = str(p_text).replace("\\", "\\\\").replace('"', '\\"')
                schedule_lines.append(f'"{frame}": "{escaped}"')
            schedule_string = ",\n".join(schedule_lines)

            for node_id, node in workflow.items():
                cls = node.get("class_type", "")
                inputs = node.get("inputs", {})
                
                if "EmptyLTXVLatentVideo" in cls or "LTXVEmptyLatentVideo" in cls:
                    inputs["length"] = tgt_len
                    inputs["width"] = int(width)
                    inputs["height"] = int(height)
                elif "LTXVEmptyLatentAudio" in cls:
                    inputs["frames_number"] = tgt_len
                    inputs["frame_rate"] = 12
                elif "BatchPromptSchedule" in cls:
                    inputs["text"] = schedule_string
                    inputs["max_frames"] = tgt_len
                elif "CLIPTextEncode" in cls and "quality" in str(inputs.get("text", "")).lower():
                    inputs["text"] = negative_prompt
                elif cls == "DenoLTXSequencer" and image_url:
                    urls_list = [u.strip() for u in image_url.split(",")] if "," in str(image_url) else [str(image_url).strip()]
                    inputs["num_images"] = len(urls_list)
                elif cls in ["VAEDecodeTiled", "VAEDecode", "LTXVSpatioTemporalTiledVAEDecode"]:
                    inputs["tile_size"] = 512
                    inputs["overlap"] = 64
                    inputs["temporal_size"] = 64
                    inputs["temporal_overlap"] = 8
                elif cls == "VHS_VideoCombine":
                    inputs["frame_rate"] = 24
                    if "filename_prefix" in inputs: inputs["filename_prefix"] = filename_prefix

            sage_node_id = next((k for k, v in workflow.items() if v.get("class_type") == "LTX2MemoryEfficientSageAttentionPatch"), None)
            if sage_node_id:
                sage_input = workflow[sage_node_id]["inputs"].get("model")
                if sage_input:
                    for node_id, node_data in workflow.items():
                        if isinstance(node_data, dict) and "inputs" in node_data:
                            for k, v in node_data["inputs"].items():
                                if isinstance(v, list) and len(v) > 0 and v[0] == sage_node_id:
                                    node_data["inputs"][k] = sage_input
                del workflow[sage_node_id]

        except Exception as e:
            print(f"⚠️ Dynamic parameter binding notice: {e}")

        # 4. DOWNLOAD GUIDING ASSETS
        dynamic_guides_dir = "/workspace/ComfyUI/input/dynamic_guides"
        if os.path.exists(dynamic_guides_dir): shutil.rmtree(dynamic_guides_dir)
        os.makedirs(dynamic_guides_dir, exist_ok=True)

        urls_to_download = []
        if image_url:
            if isinstance(image_url, list): urls_to_download = [str(u).strip() for u in image_url if str(u).strip()]
            elif isinstance(image_url, str) and image_url.strip():
                urls_to_download = [u.strip() for u in image_url.split(",")] if "," in image_url else [image_url.strip()]

        if not urls_to_download:
            from PIL import Image
            img = Image.new('RGB', (1024, 1024), color='black')
            img.save(os.path.join(dynamic_guides_dir, "guide_0.png"))
        else:
            async def download_one(session, url_str, target_dest):
                from urllib.parse import urlparse
                parsed = urlparse(url_str)
                is_r2_storage = ("r2.cloudflarestorage.com" in url_str or "pub-" in url_str or parsed.netloc == "" or not parsed.scheme)
                
                if is_r2_storage:
                    file_key = parsed.path.lstrip('/')
                    while "//" in file_key: file_key = file_key.replace("//", "/")
                    await asyncio.get_event_loop().run_in_executor(None, self.s3.download_file, "video-asset-files-storage-workflow", file_key, target_dest)
                else:
                    try:
                        async with session.get(url_str, timeout=120) as r:
                            if r.status == 200:
                                with open(target_dest, "wb") as f: f.write(await r.read())
                            else: raise Exception(f"HTTP {r.status}")
                    except Exception:
                        await asyncio.get_event_loop().run_in_executor(None, urllib.request.urlretrieve, url_str, target_dest)

            async with aiohttp.ClientSession() as session:
                tasks = [download_one(session, url, os.path.join(dynamic_guides_dir, f"guide_{idx}.png")) for idx, url in enumerate(urls_to_download)]
                await asyncio.gather(*tasks)

        out_dir = "/workspace/ComfyUI/output"
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        ram_task = asyncio.create_task(self._ram_squeezer())

        # 5. EXECUTE GENERATION PIPELINE
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post("http://127.0.0.1:8188/prompt", json={"prompt": workflow}) as resp:
                    res_json = await resp.json()
                    if "error" in res_json or "prompt_id" not in res_json:
                        raise HTTPException(status_code=400, detail=f"Invalid Execution Pattern: {res_json.get('error', res_json)}")
                    prompt_id = res_json['prompt_id']

                start_time = time.time()
                while True:
                    if self.process.poll() is not None: raise HTTPException(status_code=500, detail="Backend failed.")
                    async with session.get("http://127.0.0.1:8188/history") as hist_resp:
                        if hist_resp.status == 200:
                            history = await hist_resp.json()
                            if prompt_id in history: break
                    if time.time() - start_time > 2400: raise HTTPException(status_code=540, detail="Timeout.")
                    await asyncio.sleep(5)

                videos = [v for v in os.listdir(out_dir) if v.endswith((".mp4", ".mkv", ".webm"))]
                if not videos: raise HTTPException(status_code=500, detail="Output file missing.")
                videos.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
                
                try:
                    await session.post("http://127.0.0.1:8188/free", json={"unload_models": True, "free_memory": True})
                except Exception: pass

                with open(os.path.join(out_dir, videos[0]), "rb") as f:
                    return Response(content=f.read(), media_type="video/mp4")
        finally:
            ram_task.cancel()
