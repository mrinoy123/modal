import modal
import os

# =========================================================
# 1. IMAGE DEFINITION
# =========================================================

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libsm6", "libxext6", "wget")
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "diffusers>=0.36.1", 
        "transformers>=4.48.2",
        "accelerate>=1.3.0",
        "sentencepiece",
        "protobuf",
        "imageio[ffmpeg]",
        "pillow",
        "boto3",
        "hf_transfer",
        "xformers",
        "peft",
        "bitsandbytes>=0.45.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

app = modal.App("freedom-force-ltx-pro-optimized")

# =========================================================
# 2. MODEL CLASS
# =========================================================

@app.cls(
    gpu="L4", # 24GB VRAM
    image=image,
    timeout=7200,
)
class LTXVideoGenerator:

    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import LTXImageToVideoPipeline, LTXVideoTransformer3DModel, AutoencoderKLVideo
        from transformers import T5EncoderModel, T5Tokenizer, BitsAndBytesConfig

        print("🚀 Loading LTX-Video Components with manual quantization...")
        
        # 1. 8-bit Quantization Config (Saves ~10GB VRAM)
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
            bnb_8bit_quant_type="fp8" # Community optimization for L4/H100
        )

        # 2. Load Transformer separately (The heavy part)
        print("📦 Loading Quantized Transformer...")
        transformer = LTXVideoTransformer3DModel.from_pretrained(
            "Lightricks/LTX-Video",
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # 3. Load T5 Text Encoder separately (The second heavy part)
        print("📦 Loading Quantized Text Encoder...")
        text_encoder = T5EncoderModel.from_pretrained(
            "Lightricks/LTX-Video",
            subfolder="text_encoder",
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # 4. Assemble the Pipeline manually to bypass 'PipelineQuantizationConfig' validation
        print("🔧 Assembling Pipeline...")
        self.pipe = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=torch.bfloat16,
        )

        # 5. Peak Memory Optimization
        # enable_model_cpu_offload is the most stable way to run 11B models on 24GB
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("⚡ xformers active")
        except:
            pass

    @modal.method()
    def generate(self, endpoint: str, access_key: str, secret_key: str, bucket: str):
        import gc
        import torch
        import boto3
        from PIL import Image
        from diffusers.utils import export_to_video

        # Connect to R2
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )

        input_path = "/tmp/input.png"
        target_file = "queue/f22_raptor_raw.png"
        
        print(f"📥 Downloading: {target_file}")
        s3.download_file(bucket, target_file, input_path)

        # Dimensions: Must be multiples of 32
        image = Image.open(input_path).convert("RGB").resize((768, 512))

        prompt = (
            "cinematic aerial combat, F22 fighter jet flying through a 3D futuristic city, "
            "extreme motion blur, camera tracking the jet, explosions in background, 4k"
        )
        
        print("🎬 Generating High-Fidelity Video...")
        torch.cuda.empty_cache()
        gc.collect()

        try:
            with torch.inference_mode():
                # num_frames = (n * 8) + 1
                video_frames = self.pipe(
                    image=image,
                    prompt=prompt,
                    negative_prompt="low quality, blurry, static, watermark, distorted",
                    width=768,
                    height=512,
                    num_frames=73,
                    num_inference_steps=30,
                    guidance_scale=3.5,
                ).frames[0]

            output_path = "/tmp/output.mp4"
            export_to_video(video_frames, output_path, fps=24)
            
            output_key = "output/ltx_v2_3_final.mp4"
            s3.upload_file(output_path, bucket, output_key)
            return {"status": "success", "video_key": output_key}
            
        except Exception as e:
            return {"status": "error", "reason": str(e)}

# =========================================================
# 3. TERMINAL ENTRYPOINT
# =========================================================

@app.local_entrypoint()
def main():
    config = {
        "endpoint": "https://4d91f4d3d0366568a54ffa32ffcb7bf4.r2.cloudflarestorage.com",
        "access_key": "3c33425ba6e5abbd3e63afab14dc8866",
        "secret_key": "d65f107bb61093843c6dd980c764443fdf50924a7701078b99f007d3060e25a8",
        "bucket": "video-asset-files-storage-workflow"
    }

    print("🛰️ Initializing Modal Remote Task...")
    gen = LTXVideoGenerator()
    result = gen.generate.remote(**config)
    
    if result["status"] == "success":
        print(f"🏁 Video uploaded to: {result['video_key']}")
    else:
        print(f"❌ Failed: {result['reason']}")
