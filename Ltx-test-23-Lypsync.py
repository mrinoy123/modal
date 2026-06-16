# ==============================================================================
# PART 1: IMPORTS & ENVIRONMENT SETUP
# ==============================================================================
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
import math
import glob
from urllib.parse import urlparse
from fastapi import Request, Response, HTTPException, Header
from fastapi.responses import StreamingResponse
from typing import Optional
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="numba")

# ==============================================================================
# PART 2 & 3: BASE IMAGE & OS CONFIGURATION
# ==============================================================================
base_image = modal.Image.from_registry(
    "nvidia/cuda:12.5.1-devel-ubuntu24.04",
    add_python="3.12"
).apt_install(
    "git", "wget", "ffmpeg", "libgl1", "libglib2.0-0",
    "build-essential", "ninja-build", "cmake", "clang", "llvm",
    "libgoogle-perftools-dev", "sox", "libsox-fmt-all"
).env({
    "FORCE_REBUILD_INDEX": "523"  
})

build_image = base_image.env({
    "CUDA_HOME": "/usr/local/cuda",
    "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", ""),
    "FORCE_CUDA": "1",
    "TORCH_CUDA_ARCH_LIST": "8.9",
    "MAX_JOBS": "1",
    "CC": "gcc",
    "CXX": "g++"
}).run_commands(
    "python3.12 -m pip install --no-cache-dir fastapi aiohttp boto3 triton>=3.1.0 ninja setuptools>=70.0.0 wheel pip>=24.0 Pillow",
    "python3.12 -m pip install --no-cache-dir pandas numexpr pytz python-dateutil scipy matplotlib colorama torchvision librosa soundfile decord imageio scikit-image numba einops bitsandbytes rotary_embedding_torch"
)

# ==============================================================================
# PART 4: COMFYUI & CUSTOM NODES CLONING
# ==============================================================================
torch_image = build_image.run_commands(
    "python3.12 -m pip install --no-cache-dir torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124",
    "python3.12 -m pip install --no-cache-dir diffusers accelerate transformers>=4.49.0 torchsde numpy==1.26.4 kornia==0.7.3",
    "python3.12 -m pip install --no-cache-dir sageattention==1.0.6"
)

clone_image = torch_image.run_commands(
    "git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /workspace/ComfyUI",
    "GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
    "git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo",
    "git clone --depth 1 https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI.git /workspace/ComfyUI/custom_nodes/WhatDreamsCost-ComfyUI",
    "git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes",
    "git clone --depth 1 https://github.com/yolain/ComfyUI-Easy-Use.git /workspace/ComfyUI/custom_nodes/ComfyUI-Easy-Use",
    "git clone --depth 1 https://github.com/Deno2026/comfyui-deno-custom-nodes.git /workspace/ComfyUI/custom_nodes/comfyui-deno-custom-nodes",
    "git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git /workspace/ComfyUI/custom_nodes/rgthree-comfy",
    "git clone --depth 1 https://github.com/flybirdxx/ComfyUI-Qwen-TTS.git /workspace/ComfyUI/custom_nodes/ComfyUI-Qwen-TTS",
    "git clone --depth 1 https://github.com/kijai/ComfyUI-PromptRelay.git /workspace/ComfyUI/custom_nodes/ComfyUI-PromptRelay"
)

deps_image = clone_image.run_commands(
    "sed -i '/torch/d' /workspace/ComfyUI/requirements.txt",
    r"find /workspace/ComfyUI/custom_nodes -name 'requirements.txt' -exec sed -i '/torch/d' {} \;",
    "python3.12 -m pip install --no-cache-dir -r /workspace/ComfyUI/requirements.txt",
    r"find /workspace/ComfyUI/custom_nodes -name 'requirements.txt' -exec python3.12 -m pip install --no-cache-dir -r {} \;"
)

final_image = deps_image.run_commands(
    "python3.12 -m pip install --no-cache-dir transformers>=4.49.0 pydub",
    "echo '' >> /usr/local/lib/python3.12/site-packages/sageattention/__init__.py",
    "echo 'sageattn_qk_int8_pv_fp16_triton = sageattn' >> /usr/local/lib/python3.12/site-packages/sageattention/__init__.py",
    "echo 'import sys; sys.modules[\"torch\"].float8_e8m0fnu = getattr(sys.modules[\"torch\"], \"float8_e8m0fnu\", sys.modules[\"torch\"].float32)' >> /usr/local/lib/python3.12/site-packages/torch/__init__.py",
    env={"CUDA_HOME": "/usr/local/cuda", "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", ""), "FORCE_CUDA": "1", "TORCH_CUDA_ARCH_LIST": "8.9"}
)

# ==============================================================================
# PART 5: MODAL APP CONFIGURATION & CLOUD VOLUMES 
# ==============================================================================
app = modal.App("media-worker-ltx23-director-lypsync-v3")
weights_volume = modal.Volume.from_name("ltx-2-3-all-model-weights", create_if_missing=False)

@app.cls(
    gpu="L40S", 
    image=final_image,
    volumes={"/mnt/weights": weights_volume},
    secrets=[modal.Secret.from_name("video-generator-workflow")],
    memory=8192, 
    scaledown_window=8,
    timeout=3600
)
class LTX23DirectorLypsyncEngine:

    def _log_reader(self):
        for line in iter(self.process.stdout.readline, ""):
            if line: print(f"[ComfyUI] {line.strip()}")

    async def _ram_squeezer(self):
        while True:
            try:
                with open('/proc/sys/vm/drop_caches', 'w') as f:
                    f.write('1\n')
            except Exception: pass
            await asyncio.sleep(15)

    @modal.enter()
    def start_comfy(self):
        import boto3
        
        print("🎨 Building Disk-Backed Pointer-Pass Memory Nodes for Infinite Batching...")
        os.makedirs("/workspace/ComfyUI/custom_nodes/LTXCustomPipeline", exist_ok=True)
        custom_nodes_path = "/workspace/ComfyUI/custom_nodes/LTXCustomPipeline/__init__.py"
        with open(custom_nodes_path, "w") as f:
            f.write("""
import torch
import nodes
import os

LTX_CACHE = {}
CACHE_DIR = "/tmp/comfy_swap/ltx_tensors"
os.makedirs(CACHE_DIR, exist_ok=True)

class MemoryCacheWriter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),  
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "video_latent": ("LATENT",),
            "audio_latent": ("LATENT",),
            "guide_data": ("GUIDE_DATA",),
            "frame_rate": ("FLOAT",)
        }, "optional": {
            "scene_id": ("STRING", {"default": "0"})
        }}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "write_cache"
    CATEGORY = "LTXBatch"

    def write_cache(self, model, positive, negative, video_latent, audio_latent, guide_data, frame_rate, scene_id="0"):
        global LTX_CACHE
        
        # 1. Store ONLY the lightweight Model reference in RAM
        LTX_CACHE[str(scene_id)] = {
            "model": model,
            "frame_rate": frame_rate
        }
        
        # 2. Serialize the massive tensor dictionaries to DISK (Prevents OOM)
        tensor_data = {
            "positive": positive,
            "negative": negative,
            "video_latent": video_latent,
            "audio_latent": audio_latent,
            "guide_data": guide_data
        }
        
        target_path = os.path.join(CACHE_DIR, f"scene_{scene_id}.pt")
        torch.save(tensor_data, target_path)
        
        print(f"[LTX Cache] 💾 Saved Scene {scene_id}. Model in RAM, Tensors stored safely on Disk.", flush=True)
        return ()

class MemoryCacheReader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}, "optional": {"scene_id": ("STRING", {"default": "0"})}}
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "LATENT", "GUIDE_DATA", "FLOAT")
    RETURN_NAMES = ("model", "positive", "negative", "video_latent", "audio_latent", "guide_data", "frame_rate")
    FUNCTION = "read_cache"
    CATEGORY = "LTXBatch"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # Force ComfyUI to NEVER use an execution cache for this node
        return float("NaN")

    def read_cache(self, scene_id="0"):
        global LTX_CACHE
        ram_data = LTX_CACHE.get(str(scene_id))
        if ram_data is None: 
            raise ValueError(f"Model Cache for Scene {scene_id} not found in RAM!")
            
        target_path = os.path.join(CACHE_DIR, f"scene_{scene_id}.pt")
        if not os.path.exists(target_path):
            raise ValueError(f"Tensor Cache for Scene {scene_id} not found on Disk!")
            
        # 3. Load fresh tensors from disk. This creates pristine, deep-cloned copies 
        # so the Sampler does not mutate them, completely fixing the Static Noise.
        tensor_data = torch.load(target_path, map_location="cpu", weights_only=False)
        
        print(f"[LTX Cache] 📂 Loaded Scene {scene_id}. Model mapped from RAM, Tensors imported from Disk.", flush=True)
        
        return (
            ram_data["model"], 
            tensor_data["positive"], 
            tensor_data["negative"], 
            tensor_data["video_latent"], 
            tensor_data["audio_latent"],
            tensor_data["guide_data"],
            ram_data["frame_rate"]
        )

NODE_CLASS_MAPPINGS = {
    "MemoryCacheWriter": MemoryCacheWriter,
    "MemoryCacheReader": MemoryCacheReader
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MemoryCacheWriter": "Memory Cache Writer",
    "MemoryCacheReader": "Memory Cache Reader"
}
""")
        
        print("🔗 Configuring Native Extra Model Paths for Canonical Storage...")
        extra_paths_content = """
modal_weights:
    base_path: /mnt/weights/canonical_storage
    checkpoints: .
    unet: .
    diffusion_models: .
    clip: .
    text_encoders: .
    vae: .
    loras: .
    latent_upscale_models: .
    audio_vae: .
"""
        with open("/workspace/ComfyUI/extra_model_paths.yaml", "w") as f:
            f.write(extra_paths_content)

        print("🔗 Running Atomic Model Folder Linker...")
        base_models_dir = "/workspace/ComfyUI/models"
        for d in ["unet", "vae", "clip", "text_encoders", "checkpoints", "loras", "upscale_models", "latent_upscale_models", "diffusion_models", "audio_vae", "qwen-tts"]: 
            os.makedirs(os.path.join(base_models_dir, d), exist_ok=True)

        print("🔗 Mapping HuggingFace Qwen structure dynamically...")
        qwen_src_dir = "/mnt/weights/qwen3tts"
        if os.path.exists(qwen_src_dir):
            for d in os.listdir(qwen_src_dir):
                src = os.path.join(qwen_src_dir, d)
                if not os.path.isdir(src): continue
                
                for pb in ["qwen-tts", "qwen3_tts", "Qwen3-TTS", "qwen_tts"]:
                    dst1 = os.path.join(base_models_dir, pb, d)
                    os.makedirs(os.path.dirname(dst1), exist_ok=True)
                    if not os.path.exists(dst1):
                        try: os.symlink(src, dst1)
                        except Exception: pass
                        
                    dst2 = os.path.join(base_models_dir, pb, "Qwen", d)
                    os.makedirs(os.path.dirname(dst2), exist_ok=True)
                    if not os.path.exists(dst2):
                        try: os.symlink(src, dst2)
                        except Exception: pass

        if os.path.exists("/mnt/weights/canonical_storage"):
            for root_dir, _, files in os.walk("/mnt/weights/canonical_storage"):
                for filename in files:
                    if not filename.endswith((".safetensors", ".gguf", ".pth", ".pt", ".bin")): continue
                    src_path = os.path.join(root_dir, filename)
                    for dest in ["checkpoints", "unet", "diffusion_models", "clip", "vae", "loras", "latent_upscale_models", "audio_vae", "text encoders"]:
                        symlink_dest = os.path.join(base_models_dir, dest, filename)
                        if not os.path.exists(symlink_dest):
                            try: os.symlink(src_path, symlink_dest)
                            except Exception: pass

        self.s3 = boto3.client(
            service_name='s3', 
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com", 
            aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'], 
            aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], 
            region_name="auto"
        )

        print("🚀 Launching LTX-Director Server Engine on L40S GPU...")
        os.makedirs("/tmp/comfy_swap", exist_ok=True)

        env_vars = os.environ.copy()
        env_vars["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
        env_vars["TORCH_NUM_THREADS"] = "1"
        env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
        env_vars["CUDA_MODULE_LOADING"] = "LAZY" 
        
        self.process = subprocess.Popen([
            "python3.12", "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--mmap-torch-files", "--cache-none", "--temp-directory", "/tmp/comfy_swap", 
            "--fp8_e4m3fn-unet", "--fp8_e4m3fn-text-enc"
        ], cwd="/workspace/ComfyUI", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env_vars)
        
        self.t = threading.Thread(target=self._log_reader, daemon=True)
        self.t.start()

        start_time = time.time()
        comfy_ready = False
        while time.time() - start_time < 300:
            if self.process.poll() is not None: os._exit(1)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8188/", timeout=1) as response:
                    if response.status == 200: 
                        comfy_ready = True
                        break
            except Exception: time.sleep(2)
                
        if not comfy_ready: os._exit(1)
        print("✅ Base pipeline active. Awaiting 3-Subgraph Triggers.")

    async def clear_comfy_memory(self, session, unload_models=False):
        try:
            async with session.post("http://127.0.0.1:8188/free", json={"unload_models": unload_models, "free_memory": True}) as r: await r.read()
        except Exception: pass
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        try: ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception: pass
        await asyncio.sleep(1)

    async def execute_comfy_workflow(self, session, workflow_json):
        async with session.post("http://127.0.0.1:8188/prompt", json={"prompt": workflow_json}) as r:
            if r.status != 200:
                err_text = await r.text()
                raise HTTPException(status_code=500, detail=f"Failed to queue prompt: {r.status} - {err_text}")
            res = await r.json()
            
            if "error" in res: raise HTTPException(status_code=500, detail=f"Validation Error: {res['error']}")
            if "node_errors" in res and res["node_errors"]: raise HTTPException(status_code=500, detail=f"Node Errors: {res['node_errors']}")
                
            prompt_id = res["prompt_id"]

        while True:
            async with session.get(f"http://127.0.0.1:8188/history/{prompt_id}") as r:
                if r.status == 200:
                    history_data = await r.json()
                    if prompt_id in history_data:
                        step_data = history_data[prompt_id]
                        if "status" in step_data and "messages" in step_data["status"]:
                            for msg in step_data["status"]["messages"]:
                                if msg[0] == "execution_error": raise HTTPException(status_code=500, detail=f"ComfyUI execution error: {msg[1]}")
                        return step_data
            if self.process.poll() is not None: raise HTTPException(status_code=500, detail="ComfyUI server process crashed.")
            await asyncio.sleep(1)

    # ==============================================================================
    # PART 6: LYPSYNC 3-SUBGRAPH ORCHESTRATION & BATCHING
    # ==============================================================================
    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        if x_api_key != "testing-modal-workflow-2": 
            raise HTTPException(status_code=403, detail="Unauthorized Pipeline Request")
        
        body = await request.json()
        if isinstance(body, dict):
            if "json" in body: body = body["json"]
            elif "body" in body: body = body["body"]

        async def process_pipeline():
            date_folder = body.get("date_folder", time.strftime('%Y-%m-%d'))
            batch_scenes = body.get("batch_scenes", [])
            subgraph_1 = body.get("subgraph_1")
            subgraph_2 = body.get("subgraph_2")
            subgraph_3 = body.get("subgraph_3")
            
            # Create a globally unique session ID to map Phase 2 and Phase 3 perfectly
            session_id = int(time.time())

            if not batch_scenes: raise HTTPException(status_code=400, detail="Missing batch_scenes array.")
            if not subgraph_1 or not subgraph_2 or not subgraph_3: raise HTTPException(status_code=400, detail="Missing Subgraph definitions.")

            dynamic_guides_dir = "/workspace/ComfyUI/input/dynamic_guides"
            if os.path.exists(dynamic_guides_dir): shutil.rmtree(dynamic_guides_dir)
            os.makedirs(dynamic_guides_dir, exist_ok=True)

            ram_task = asyncio.create_task(self._ram_squeezer())
            generated_outputs = []

            def inject_node_overrides(sg, idx, custom_w, custom_h, exact_audio_duration, total_frames, scene_data, s_id):
                for node_id, node_data in list(sg.items()):
                    c_type = node_data.get("class_type", "")
                    if "inputs" not in node_data: node_data["inputs"] = {}
                    inputs = node_data["inputs"]
                    widgets = node_data.get("widgets_values", [])

                    def set_val(k, widx, v):
                        inputs[k] = v
                        if widgets is not None and widx is not None and len(widgets) > widx: widgets[widx] = v

                    if c_type == "DiffusionModelLoaderKJ":
                        set_val("model_name", 0, "ltx-2.3-22b-dev-fp8.safetensors")
                        set_val("weight_dtype", 1, "default") 
                        set_val("sage_attention", None, "auto")
                        set_val("compute_dtype", None, "bf16") 
                        
                    elif c_type == "LTXAVTextEncoderLoader":
                        set_val("text_encoder", 0, "gemma-3-12b-it-heretic-v2_fp8_e4m3fn.safetensors")
                        set_val("ckpt_name", 1, "ltx-2.3_text_projection_bf16.safetensors")
                        
                    elif c_type == "LTXVAudioVAELoader":
                        set_val("ckpt_name", 0, "LTX23_audio_vae_bf16.safetensors")
                          
                    elif c_type in ["VAELoader", "VAELoaderKJ"]:
                        set_val("vae_name", 0, "LTX23_video_vae_bf16.safetensors")
                        set_val("weight_dtype", None, "bf16") 
                         
                    elif c_type == "LowVRAMLatentUpscaleModelLoader":
                        set_val("model_name", 0, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")

                    elif c_type == "DenoLTXMultiLoraLoader":
                        set_val("lora_1", 2, "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
                        set_val("enabled_1", 1, True)
                        set_val("strength_1", None, 0.45) 
                        set_val("lora_2", 7, "LTX_2.3_ID_LoRA_TalkVid_3K.safetensors")
                        set_val("enabled_2", 6, True)  
                        set_val("strength_2", None, 0.45)

                    elif c_type in ["MemoryCacheWriter", "MemoryCacheReader"]:
                        set_val("scene_id", None, f"{s_id}_{idx}")

                    elif c_type in ["LTXDirector", "LTXDirectorGuide", "PromptRelayEncode"]:
                        set_val("custom_width", None, custom_w)
                        set_val("width", None, custom_w)
                        set_val("custom_height", None, custom_h)
                        set_val("height", None, custom_h)
                        
                        if c_type == "LTXDirector":
                            set_val("img_compression", None, 100)
                            set_val("divisible_by", None, 32)
                            set_val("use_custom_audio", None, True)
                        
                        if total_frames > 0:  
                            set_val("duration_frames", None, total_frames)
                            set_val("length", None, total_frames)
                            set_val("duration_seconds", None, exact_audio_duration)
                        
                        if c_type in ["LTXDirector", "LTXDirectorGuide"] and "timeline_payload" in scene_data:
                            payload_str = json.dumps(scene_data["timeline_payload"])
                            set_val("timeline_data", None, payload_str)
                            set_val("timeline_ui", None, payload_str)
                            set_val("timeline", None, payload_str)
                             
                            if "local_prompts_str" in scene_data:
                                set_val("local_prompts", None, scene_data["local_prompts_str"])
                                set_val("segment_lengths", None, scene_data["segment_lengths_str"])
                                set_val("guide_strength", None, scene_data["guide_strength_str"])
                                set_val("global_prompt", None, scene_data["global_prompt_str"])
                             
                            if widgets is not None and len(widgets) > 3:
                                widgets[1] = total_frames 
                                widgets[3] = payload_str
                            
                return sg

            try:
                async with aiohttp.ClientSession() as session:
                    custom_w = 448 
                    custom_h = 768

                    async def download_asset(url, target_path):
                        if not url: return False
                        if "r2.dev" in url or "cloudflarestorage" in url:
                            parsed = urlparse(url)
                            key = parsed.path.lstrip('/')
                            bucket_name = "video-asset-files-storage-workflow"
                            if key.startswith(bucket_name + "/"): key = key.replace(bucket_name + "/", "", 1)
                            try:
                                await asyncio.get_event_loop().run_in_executor(None, self.s3.download_file, bucket_name, key, target_path)
                                return True
                            except Exception: pass
                        try:
                            async with session.get(url, timeout=60) as r:
                                if r.status == 200:
                                    with open(target_path, "wb") as f: f.write(await r.read())
                                    return True
                        except Exception: pass
                        return False

                    print(f"\n[Lypsync API] 🎙️ STARTING PHASE 1: DYNAMIC QWEN3-TTS AUDIO CALCULATION", flush=True)
                    
                    import soundfile as sf
                    import numpy as np 
                    from PIL import Image
                    import librosa
                    import re
                    
                    for idx, scene in enumerate(batch_scenes):
                        speakers_conf = scene.get("speakers", {"A": {"mode": "design", "prompt": "A cinematic voice.", "ref_url": ""}})
                        script_array = scene.get("script", [{"speaker": "A", "text": "Testing audio.", "visual_action": "Stable camera."}])
                        
                        # --- PULL DYNAMIC FALLBACKS FROM N8N JSON ---
                        fallback_action = scene.get("default_visual_action", "A cinematic shot, stable camera, talking.")
                        silent_suffix = scene.get("silent_action_suffix", ", character listening, silent, mouth completely closed.")
                        
                        img1_path = os.path.join(dynamic_guides_dir, f"master_shot_1_{idx}.png")
                        img2_path = os.path.join(dynamic_guides_dir, f"master_shot_2_{idx}.png")
                        
                        await download_asset(scene.get("image1_url"), img1_path)
                        await download_asset(scene.get("image2_url"), img2_path)

                        if not os.path.exists(img1_path): Image.new('RGB', (custom_w, custom_h), color='black').save(img1_path)
                        else: Image.open(img1_path).convert("RGB").resize((custom_w, custom_h), Image.Resampling.LANCZOS).save(img1_path)
                        if os.path.exists(img2_path): Image.open(img2_path).convert("RGB").resize((custom_w, custom_h), Image.Resampling.LANCZOS).save(img2_path)

                        master_audio_arrays = []
                        total_frames_tracked = 0
                        segments_timeline = []
                        target_samplerate = 16000 # LTX-Video optimal native conditioning rate
                        
                        current_speaker = None
                        last_image_seg = None
                         
                        for line_idx, line in enumerate(script_array):
                            spk_id = line.get("speaker", "A")
                            spk_conf = speakers_conf.get(spk_id, {"mode": "design", "prompt": "A default voice."})
                            line_text = line.get("text", "")
                            
                            # Use n8n-provided string, or fallback if completely empty
                            visual_action = line.get("visual_action", "").replace('"', "'").replace("\n", " ").strip()
                            if not visual_action or len(visual_action) < 2:
                                visual_action = fallback_action
                                
                            sg1 = json.loads(json.dumps(subgraph_1))
                            
                            qwen_design_id = next((k for k, v in sg1.items() if v["class_type"] == "FB_Qwen3TTSVoiceDesign"), None)
                            qwen_clone_id = next((k for k, v in sg1.items() if v["class_type"] == "FB_Qwen3TTSVoiceClonePrompt"), None)
                            qwen_rolebank_id = next((k for k, v in sg1.items() if v["class_type"] == "FB_Qwen3TTSRoleBank"), None)
                            qwen_dialogue_id = next((k for k, v in sg1.items() if v["class_type"] == "FB_Qwen3TTSDialogueInference"), None)
                            save_audio_id = next((k for k, v in sg1.items() if v["class_type"] == "SaveAudio"), None)
                            load_audio_id = next((k for k, v in sg1.items() if v["class_type"] == "LoadAudio"), None)

                            if qwen_rolebank_id and "inputs" in sg1[qwen_rolebank_id]:
                                for i in range(2, 9):
                                    if f"prompt_{i}" in sg1[qwen_rolebank_id]["inputs"]:
                                        del sg1[qwen_rolebank_id]["inputs"][f"prompt_{i}"]

                            sg1[save_audio_id]["inputs"]["filename_prefix"] = f"raw_line_{idx}_{line_idx}"
                            
                            if "text" in sg1[qwen_dialogue_id]["inputs"]:
                                sg1[qwen_dialogue_id]["inputs"]["text"] = f"Speaker {spk_id}: {line_text}"
                            elif "script" in sg1[qwen_dialogue_id]["inputs"]:
                                sg1[qwen_dialogue_id]["inputs"]["script"] = f"Speaker {spk_id}: {line_text}"
                            
                            if spk_conf.get("mode") == "clone":
                                ref_path = os.path.join(dynamic_guides_dir, f"ref_spk_{spk_id}.wav")
                                if not os.path.exists(ref_path): await download_asset(spk_conf.get("ref_url"), ref_path)
                                
                                sg1[load_audio_id]["inputs"]["audio"] = ref_path
                                sg1[qwen_clone_id]["inputs"]["ref_audio"] = [str(load_audio_id), 0]
                                sg1[qwen_rolebank_id]["inputs"]["prompt_1"] = [str(qwen_clone_id), 0]
                            else:
                                sg1[qwen_design_id]["inputs"]["instruct"] = spk_conf.get("prompt", "")
                                sg1[qwen_clone_id]["inputs"]["ref_audio"] = [str(qwen_design_id), 0]
                                sg1[qwen_rolebank_id]["inputs"]["prompt_1"] = [str(qwen_clone_id), 0]

                            sg1[qwen_rolebank_id]["inputs"]["role_name_1"] = f"Speaker {spk_id}"
                            sg1[qwen_dialogue_id]["inputs"]["role_bank"] = [str(qwen_rolebank_id), 0]

                            print(f"🎤 Synthesizing Audio Slice: Speaker {spk_id} - Line {line_idx}...", flush=True)
                            await self.execute_comfy_workflow(session, sg1)

                            output_files = glob.glob(f"/workspace/ComfyUI/output/raw_line_{idx}_{line_idx}_*.*")
                            if not output_files: raise Exception(f"Failed to generate audio for Line {line_idx}.")
                            output_files.sort(key=os.path.getmtime)
                            raw_slice_path = output_files[-1]

                            data, samplerate = sf.read(raw_slice_path)
                             
                            data, _ = librosa.effects.trim(data, top_db=35) 
                            
                            if samplerate != target_samplerate:
                                data = librosa.resample(data, orig_sr=samplerate, target_sr=target_samplerate)
                            
                            duration_seconds = len(data) / target_samplerate
                            frames_for_line = math.ceil(duration_seconds * 25)
                            
                            exact_required_samples = int((frames_for_line / 25.0) * target_samplerate)
                            padding_needed = exact_required_samples - len(data)
                            
                            if padding_needed > 0:
                                silence_array = np.zeros(padding_needed, dtype=data.dtype)
                                data = np.concatenate((data, silence_array))

                            # Apply dynamic n8n suffix instead of hardcoded english string
                            silent_prompt = re.sub(r'(?i)\b(speaking|talking|lip sync)\b', '', visual_action)
                            silent_prompt = re.sub(r',\s*,', ',', silent_prompt).strip(', ')
                            silent_prompt += silent_suffix

                            silence_frames = 12 
                            silence_samples = int((silence_frames / 25.0) * target_samplerate)
                            line_total_frames = frames_for_line + silence_frames

                            img_relative_target = f"dynamic_guides/master_shot_2_{idx}.png" if (spk_id == "B" and os.path.exists(img2_path)) else f"dynamic_guides/master_shot_1_{idx}.png"

                            if spk_id != current_speaker:
                                last_image_seg = {
                                    "id": f"image_{line_idx}",
                                    "start": total_frames_tracked,
                                    "length": line_total_frames,
                                    "type": "image",
                                    "imageFile": img_relative_target,
                                    "guideStrength": 1.0
                                }
                                segments_timeline.append(last_image_seg)
                                current_speaker = spk_id
                            else:
                                if last_image_seg is not None:
                                    last_image_seg["length"] += line_total_frames

                            segments_timeline.append({
                                "id": f"text_{line_idx}",
                                "start": total_frames_tracked,
                                "length": frames_for_line,
                                "type": "text",
                                "text": visual_action,          
                                "prompt": visual_action,        
                                "prompts": [visual_action]
                            })
                            
                            segments_timeline.append({
                                "id": f"text_silence_{line_idx}",
                                "start": total_frames_tracked + frames_for_line,
                                "length": silence_frames,
                                "type": "text",
                                "text": silent_prompt,
                                "prompt": silent_prompt,
                                "prompts": [silent_prompt]
                            })
                            
                            total_frames_tracked += line_total_frames
                            
                            master_audio_arrays.append(data)
                            master_audio_arrays.append(np.zeros(silence_samples, dtype=data.dtype))
                             
                            os.remove(raw_slice_path)

                        padded_frames = (math.ceil(total_frames_tracked / 8) * 8) + 1
                         
                        last_text_seg = next((s for s in reversed(segments_timeline) if s["type"] == "text"), None)
                         
                        text_total_frames = sum(s["length"] for s in segments_timeline if s["type"] == "text")
                        
                        if text_total_frames < padded_frames:
                            diff = padded_frames - text_total_frames
                            if last_text_seg: last_text_seg["length"] += diff
                            if last_image_seg: last_image_seg["length"] += diff
                            
                            pad_samples = int((diff / 25.0) * target_samplerate)
                            master_audio_arrays.append(np.zeros(pad_samples, dtype=master_audio_arrays[0].dtype))
                            
                        elif text_total_frames > padded_frames:
                            diff = text_total_frames - padded_frames
                            for s_type in ["text", "image"]:
                                d_diff = diff
                                for s in reversed(segments_timeline):
                                    if s["type"] != s_type: continue
                                    if s["length"] > d_diff:
                                        s["length"] -= d_diff
                                        break
                                    else:
                                        d_diff -= s["length"]
                                        s["length"] = 0
                                        
                        segments_timeline = [s for s in segments_timeline if s["length"] > 0]
                        
                        master_audio = np.concatenate(master_audio_arrays)
                        perfect_audio_path = os.path.join(dynamic_guides_dir, f"perfect_dialog_{idx}.wav")
                        sf.write(perfect_audio_path, master_audio, target_samplerate)
                        
                        local_prompts_str = " | ".join([s["prompt"] for s in segments_timeline if s["type"] == "text"])
                        segment_lengths_str = ",".join([str(s["length"]) for s in segments_timeline if s["type"] == "text"])
                        guide_strength_str = ",".join(["1.0"] * len([s for s in segments_timeline if s["type"] == "image"]))
                        
                        text_segments = [s for s in segments_timeline if s["type"] == "text"]
                        
                        # Apply dynamic n8n fallback instead of hardcoded english string
                        global_prompt_str = text_segments[0]["prompt"] if text_segments else fallback_action
                        
                        audio_relative_path = f"dynamic_guides/perfect_dialog_{idx}.wav"

                        scene["exact_audio_duration"] = float(padded_frames - 1) / 25.0
                        scene["total_frames"] = padded_frames
                        scene["timeline_payload"] = {
                            "segments": segments_timeline,
                            "audioSegments": [{
                                "audioFile": audio_relative_path, 
                                "start": 0,
                                "length": padded_frames,
                                "trimStart": 0
                            }]
                        }
                        
                        scene["local_prompts_str"] = local_prompts_str
                        scene["segment_lengths_str"] = segment_lengths_str
                        scene["guide_strength_str"] = guide_strength_str
                        scene["global_prompt_str"] = global_prompt_str
                        
                        print(f"✅ Scene {idx} Calculated: {padded_frames} Frames. Timeline Accurately Padded.", flush=True)

                    print("\n🧹 Phase 1 Complete. Purging Audio Engines...", flush=True)
                    await self.clear_comfy_memory(session, unload_models=True)

                    print(f"\n[Lypsync API] 🎬 STARTING PHASE 2: BATCH DIRECTOR WORKFLOW CACHING", flush=True)
                    for idx, scene in enumerate(batch_scenes):
                        sg2 = json.loads(json.dumps(subgraph_2))
                        
                        sg2 = inject_node_overrides(sg2, idx, custom_w, custom_h, scene["exact_audio_duration"], scene["total_frames"], scene, session_id)
                        
                        print(f"🎥 Injecting Timeline & Caching Guide Data for Scene {idx} to Disk...", flush=True)
                        await self.execute_comfy_workflow(session, sg2)
                        await self.clear_comfy_memory(session, unload_models=False)

                    print(f"\n[Lypsync API] 🎞️ STARTING PHASE 3: BATCH FINAL COMBINE & RENDER", flush=True)
                    out_dir = "/workspace/ComfyUI/output"
                    if os.path.exists(out_dir): shutil.rmtree(out_dir)
                    os.makedirs(out_dir)

                    for idx, scene in enumerate(batch_scenes):
                        sg3 = json.loads(json.dumps(subgraph_3))

                        sg3 = inject_node_overrides(sg3, idx, custom_w, custom_h, scene["exact_audio_duration"], scene["total_frames"], scene, session_id)

                        print(f"🚀 Diffusing & Rendering Video for Scene {idx} from Disk Cache...", flush=True)
                        await self.execute_comfy_workflow(session, sg3)
                        await self.clear_comfy_memory(session, unload_models=False)

                        output_files = []
                        for root_p, _, filenames in os.walk(out_dir):
                            for name in filenames:
                                if name.endswith((".mp4", ".gif", ".webm")): output_files.append(os.path.join(root_p, name))

                        if not output_files: raise Exception(f"Phase 3 for Scene {idx} finished but no media files detected.")
                        
                        output_files.sort(key=os.path.getmtime)
                        target_video_file = output_files[-1]
                        saved_filename = os.path.basename(target_video_file)

                        target_key = f"{date_folder}/lypsync_v3_clips/{int(time.time())}_{scene.get('name', 'clip')}_{saved_filename}"
                        print(f"📤 Syncing Finished Asset {idx} to R2...", flush=True)
                        await asyncio.get_event_loop().run_in_executor(None, self.s3.upload_file, target_video_file, "video-asset-files-storage-workflow", target_key)
                        
                        generated_outputs.append({
                            "scene": scene.get("name", f"Clip_{idx+1}"),
                            "status": "success",
                            "file_key": target_key,
                            "public_url": f"https://pub-4d91f4d3d0366568a54ffa32ffcb7bf4.r2.dev/{target_key}",
                            "filename": saved_filename
                        })
                        os.remove(target_video_file)

                    print(f"\n[Lypsync API] 🎉 All Subgraphs Rendered. Pipeline Finished.", flush=True)
                    await self.clear_comfy_memory(session, unload_models=True)
                    
                    return generated_outputs

            finally:
                ram_task.cancel()

        async def stream_response():
            task = asyncio.create_task(process_pipeline())
            while not task.done():
                yield b'{"status": "processing", "message": "Heartbeat... Keeping connection active"}\n' + (b" " * 1024)
                done, pending = await asyncio.wait([task], timeout=10.0)
                if task in done: break
            try:
                result = task.result()
                if isinstance(result, (dict, list)): yield json.dumps(result).encode("utf-8")
                else: yield str(result).encode("utf-8")
            except HTTPException as e: 
                yield json.dumps({"status": "error", "detail": str(e.detail)}).encode("utf-8")
            except Exception as e: 
                yield json.dumps({"status": "error", "detail": str(e)}).encode("utf-8")

        return StreamingResponse(stream_response(), media_type="application/json")
