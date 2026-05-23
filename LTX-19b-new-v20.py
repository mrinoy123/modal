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
# PART 1: Infrastructure Configuration & Base Image
# Purpose: Defines the core operating system, environment variables, system-level dependencies, and the base python libraries needed for video processing and AI model inference.
# ==========================================

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


# ==========================================
# PART 2: Advanced Optimization Patches & Custom Node Installation
# Purpose: Compiles high-performance kernels, handles Git rollbacks to stable pre-March states, downgrades Kornia, and injects critical code modifications.
# ==========================================

def patch_comfy_raw_conds():
    """
    HOTFIX: Restores the missing 'raw_conds' attribute to ComfyUI's samplers and guiders.
    This resolves the AttributeError caused when using newer LTXVideo nodes with older ComfyUI rollbacks.
    """
    import os
    main_path = "/workspace/ComfyUI/main.py"
    if not os.path.exists(main_path): return
    
    with open(main_path, "r") as f:
        content = f.read()
        
    patch_code = """
# --- HOTFIX: RESTORE RAW_CONDS FOR OLDER COMFYUI VERSIONS ---
try:
    import threading
    import time
    def background_patcher():
        try:
            patched = False
            while not patched:
                import sys
                modules_to_patch = []
                if "comfy.samplers" in sys.modules: modules_to_patch.append(sys.modules["comfy.samplers"])
                if "comfy.guiders" in sys.modules: modules_to_patch.append(sys.modules["comfy.guiders"])
                
                if not modules_to_patch:
                    time.sleep(0.1)
                    continue
                    
                patched_classes = set()
                for module in modules_to_patch:
                    for name in dir(module):
                        obj = getattr(module, name)
                        if isinstance(obj, type) and hasattr(obj, "set_conds"):
                            if obj in patched_classes: continue
                            orig_set_conds = obj.set_conds
                            if getattr(orig_set_conds, "__name__", "") == "patched_set_conds": continue
                            def make_patched_set_conds(original_method):
                                def patched_set_conds(self, *args, **kwargs):
                                    pos = args[0] if len(args) >= 1 else None
                                    neg = args[1] if len(args) >= 2 else None
                                    if "positive" in kwargs: pos = kwargs["positive"]
                                    if "negative" in kwargs: neg = kwargs["negative"]
                                    self.raw_conds = (pos, neg)
                                    return original_method(self, *args, **kwargs)
                                return patched_set_conds
                            obj.set_conds = make_patched_set_conds(orig_set_conds)
                            patched_classes.add(obj)
                patched = True
                time.sleep(0.1)
        except Exception: pass
    threading.Thread(target=background_patcher, daemon=True).start()
except Exception: pass
"""
    if "background_patcher" not in content:
        with open(main_path, "w") as f:
            f.write(patch_code + "\n" + content)
        print("✅ Successfully injected backward-compatible raw_conds patch into main.py!")

# ⚡ SageAttention source compilation matching your working config
compiled_image = build_image.run_commands(
    "git clone https://github.com/thu-ml/SageAttention.git /workspace/SageAttention",
    "cd /workspace/SageAttention && pip install --no-build-isolation ."
)

final_image = compiled_image.run_commands(
    "git clone https://github.com/comfyanonymous/ComfyUI /workspace/ComfyUI",
    # ⚡ ROLLBACK 1: Revert ComfyUI Core to pre-March 2026 to restore old cond formatting
    "cd /workspace/ComfyUI && git checkout $(git rev-list -n 1 --before=\"2026-03-01\" HEAD)",
    "pip install -r /workspace/ComfyUI/requirements.txt"
).run_commands(
    "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
    "git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git /workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation",
    "pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/requirements-no-cupy.txt",
    "git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo",
    # ⚡ ROLLBACK 2: Revert LTXVideo to match the pre-March ComfyUI API
    "cd /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo && git checkout $(git rev-list -n 1 --before=\"2026-03-01\" HEAD)",
    "git clone https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes",
    "git clone https://github.com/yolain/ComfyUI-Easy-Use.git /workspace/ComfyUI/custom_nodes/ComfyUI-Easy-Use",
    "git clone https://github.com/Deno2026/comfyui-deno-custom-nodes.git /workspace/ComfyUI/custom_nodes/comfyui-deno-custom-nodes",
    "git clone https://github.com/cubiq/ComfyUI_essentials.git /workspace/ComfyUI/custom_nodes/ComfyUI_essentials",
    "git clone https://github.com/siraxe/ComfyUI-LTX-FDG.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTX-FDG"
).run_commands(
    r"find /workspace/ComfyUI/custom_nodes -name 'requirements.txt' -exec pip install -r {} \;",
    # ⚡ RESTORED: Numpy/Torch version lock
    "pip uninstall -y torch torchvision torchaudio numpy",
    "pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124",
    "pip install --no-cache-dir numpy==1.26.4 kornia==0.6.12",
    "python -c \"import re; file='/workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/vfi_models/rife/__init__.py'; data=open(file).read(); data=re.sub(r'torch\\.cat\\(output_frames, dim=0\\)', 'torch.cat([f.to(output_frames[0].device) for f in output_frames], dim=0).cpu()', data); open(file, 'w').write(data)\"",
    # ⚡ THE DEFINITIVE PATCH: Bulletproofing ComfyUI's core tensor conversion
    "python3 -c \"filepath = '/workspace/ComfyUI/comfy/sampler_helpers.py'; code = open(filepath).read(); code = code.replace('def convert_cond(cond):', 'def convert_cond(cond):\\n    import torch\\n    if isinstance(cond, torch.Tensor): return [[cond, {}]]'); code = code.replace('for x in cond:', 'for x in cond:\\n        if isinstance(x, torch.Tensor):\\n            c.append([x, {}])\\n            continue'); code = code.replace('t = x[1].copy()', 't = x[1].copy() if len(x) > 1 and isinstance(x[1], dict) else {}'); code = code.replace('p = x[0]', 'p = x[0] if isinstance(x, (list, tuple)) else x'); open(filepath, 'w').write(code)\""
).run_function(patch_comfy_raw_conds) # ⚡ EXECUTING THE NEW COND FIX




# ==========================================
# PART 3: Production Class Definition & Engine Initialization
# Purpose: Defines the Modal application resources, handles the pre-execution model symlinking, and launches the ComfyUI subprocess.
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

# ==========================================
# PART 4: Hybrid Endpoint Handler & Execution Process
# Purpose: Exposes a FastAPI endpoint to process incoming requests, handle dynamic JSON injection, download assets, and render output.
# ==========================================

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

        # Real model file names from weight volume
        target_unet = "ltx-2-19b-distilled-fp8.safetensors"
        target_gemma = "gemma-3-12b-it-FP8.safetensors"
        target_connector = "ltx-2-19b-embeddings_connector_dev_bf16.safetensors"
        target_video_vae = "ltx-2-19b-dev_video_vae.safetensors"
        target_audio_vae = "ltx-2-19b-dev_audio_vae.safetensors"
        target_distilled_lora = "ltx-2-19b-distilled-lora-384.safetensors"
        target_detailer_lora = "ltx-2-19b-ic-lora-detailer.safetensors"

        def find_node(cls_name):
            return next((k for k, v in workflow.items() if v.get("class_type") == cls_name), None)

        if isinstance(workflow, dict):
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
                    if audio_latent_node:
                        workflow[audio_latent_node]["inputs"]["frames_number"] = tgt_len
                except Exception as e:
                    print(f"⚠️ Dynamic framing error: {e}")

            sanitized_workflow = {}
            for node_id, node_data in workflow.items():
                if isinstance(node_data, dict) and "class_type" in node_data:
                    class_type = node_data.get("class_type")

                    # 🧹 Safe Purge: Skip unused text compilation nodes
                    if class_type in ["MultiStringPrompts", "JoinStringMulti"]:
                        continue

                    if "inputs" not in node_data or node_data["inputs"] is None:
                        node_data["inputs"] = {}

                    if class_type in ["UNETLoader", "UnetLoaderGGUFAdvanced"]:
                        if "unet_name" in node_data["inputs"]:
                            node_data["inputs"]["unet_name"] = target_unet
                        if "ckpt_name" in node_data["inputs"]:
                            node_data["inputs"]["ckpt_name"] = target_unet
                        if "widgets_values" in node_data and isinstance(node_data["widgets_values"], list):
                            if len(node_data["widgets_values"]) > 0:
                                node_data["widgets_values"][0] = target_unet

                    if class_type == "LTXAVTextEncoderLoader":
                        node_data["inputs"]["text_encoder"] = target_gemma
                        node_data["inputs"]["ckpt_name"] = target_connector

                    if class_type in ["VAELoaderKJ", "VAELoader"]:
                        if "vae_name" in node_data["inputs"]:
                            node_data["inputs"]["vae_name"] = target_video_vae
                        if "ckpt_name" in node_data["inputs"]:
                            node_data["inputs"]["ckpt_name"] = target_video_vae

                    if class_type == "LTXVAudioVAELoader":
                        if "ckpt_name" in node_data["inputs"]:
                            node_data["inputs"]["ckpt_name"] = target_audio_vae
                        if "vae_name" in node_data["inputs"]:
                            node_data["inputs"]["vae_name"] = target_audio_vae

                    if class_type == "DenoMultiImageLoader":
                        node_data["inputs"]["image_paths"] = "input/dynamic_guides"

                    if class_type == "LTXVEmptyLatentAudio":
                        node_data["inputs"]["frame_rate"] = 12

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

                    sanitized_workflow[str(node_id)] = node_data
            
            workflow = sanitized_workflow

        dynamic_guides_dir = "/workspace/ComfyUI/input/dynamic_guides"
        if os.path.exists(dynamic_guides_dir):
            shutil.rmtree(dynamic_guides_dir)
        os.makedirs(dynamic_guides_dir, exist_ok=True)

        urls_to_download = []
        if image_url:
            if isinstance(image_url, list):
                urls_to_download = [str(u).strip() for u in image_url if str(u).strip()]
            elif isinstance(image_url, str) and image_url.strip():
                if "," in image_url:
                    urls_to_download = [u.strip() for u in image_url.split(",") if u.strip()]
                else:
                    urls_to_download = [image_url.strip()]

        if not urls_to_download:
            from PIL import Image
            img = Image.new('RGB', (1024, 1024), color='black')
            img.save(os.path.join(dynamic_guides_dir, "guide_0.png"))
            print("Creating black canvas fallback guide.")
        else:
            async def download_one(session, url_str, target_dest):
                from urllib.parse import urlparse
                parsed = urlparse(url_str)
                is_r2_storage = ("r2.cloudflarestorage.com" in url_str or "pub-" in url_str or parsed.netloc == "" or not parsed.scheme)
                
                if is_r2_storage:
                    file_key = parsed.path.lstrip('/')
                    while "//" in file_key:
                        file_key = file_key.replace("//", "/")
                    print(f"📥 Fetching R2 file: {file_key}")
                    await asyncio.get_event_loop().run_in_executor(
                        None, 
                        self.s3.download_file, 
                        "video-asset-files-storage-workflow", 
                        file_key, 
                        target_dest
                    )
                else:
                    print(f"📥 Downloading HTTP file: {url_str}")
                    try:
                        async with session.get(url_str, timeout=120) as r:
                            if r.status == 200:
                                f_content = await r.read()
                                with open(target_dest, "wb") as f:
                                    f.write(f_content)
                            else:
                                raise Exception(f"HTTP code {r.status}")
                    except Exception as err:
                        print(f"HTTP direct download failed, falling back to urllib: {err}")
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            urllib.request.urlretrieve,
                            url_str,
                            target_dest
                        )

            async with aiohttp.ClientSession() as session:
                tasks = []
                for idx, url in enumerate(urls_to_download):
                    dest = os.path.join(dynamic_guides_dir, f"guide_{idx}.png")
                    tasks.append(download_one(session, url, dest))
                await asyncio.gather(*tasks)

        out_dir = "/workspace/ComfyUI/output"
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        ram_task = asyncio.create_task(self._ram_squeezer())

        try:
            async with aiohttp.ClientSession() as session:
                print("🎨 Running Pipeline...")
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
                        raise HTTPException(status_code=540, detail="Execution timeout reached.")
                    await asyncio.sleep(5)

            videos = [v for v in os.listdir(out_dir) if v.endswith(".mp4")]
            if not videos:
                raise HTTPException(status_code=500, detail="Output generation target missing.")
                
            videos.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
            with open(os.path.join(out_dir, videos[0]), "rb") as f:
                return Response(content=f.read(), media_type="video/mp4")
        finally:
            ram_task.cancel()
