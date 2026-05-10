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
        "bitsandbytes>=0.45.0", # Required for stable 8-bit/FP8 logic
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

app = modal.App("freedom-force-ltx-v2-3-stable")

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
        from diffusers import LTXImageToVideoPipeline, BitsAndBytesConfig
        from transformers import T5EncoderModel

        print("🚀 Loading LTX-Video with 8-bit Quantized Transformer (FP8-equivalent memory)...")
        
        # 1. Configure Quantization for the 11B Transformer
        # This replaces the manual FP8 load and fixes the mat1/mat2 dtype error
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

        # 2. Load Pipeline
        # We load the whole pipeline in BF16, and use the quant_config 
        # specifically for the transformer to fit in VRAM.
        self.pipe = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )

        # 3. Aggressive Memory Management for 24GB L4
        # LTX-Video has a massive T5 Text Encoder (11GB) and a Transformer (11GB).
        # cpu_offload is mandatory to run both + VAE on a single 24GB card.
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("⚡ xformers enabled")
        except:
            pass

    @modal.method()
    def generate(self, endpoint: str, access_key: str, secret_key: str, bucket: str):
        import gc
        import torch
        import boto3
        from botocore.exceptions import ClientError
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

        # 1. Download Input
        input_path = "/tmp/input.png"
        target_file = "queue/f22_raptor_raw.png"
        
        print(f"📥 Downloading: {target_file}")
        try:
            s3.download_file(bucket, target_file, input_path)
        except ClientError as e:
            return {"status": "error", "reason": f"S3 Download Failed: {str(e)}"}

        # 2. Process Image
        image = Image.open(input_path).convert("RGB")
        # 768x512 (divisible by 64) is the optimal resolution for LTX
        input_image = image.resize((768, 512)) 

        # 3. Generate
        prompt = (
            "cinematic aerial combat, F22 fighter jet flying through a 3D futuristic city, "
            "extreme motion blur, camera tracking the jet, explosions in background, 4k, high quality"
        )
        
        print("🎬 Starting Inference (8-bit Optimized)...")
        torch.cuda.empty_cache()
        gc.collect()

        try:
            with torch.inference_mode():
                # LTX Parameters: num_frames must be (8n + 1)
                # 73 frames = ~3 seconds at 24fps
                video_frames = self.pipe(
                    image=input_image,
                    prompt=prompt,
                    negative_prompt="low quality, blurry, static, distorted, watermark, deformed",
                    width=768,
                    height=512,
                    num_frames=73,
                    num_inference_steps=30,
                    guidance_scale=3.0,
                ).frames[0]

            # 4. Export & Upload
            output_path = "/tmp/output.mp4"
            export_to_video(video_frames, output_path, fps=24)
            
            output_key = "output/ltx_final_result.mp4"
            print(f"☁️ Uploading to R2: {output_key}")
            s3.upload_file(output_path, bucket, output_key)
            
            return {"status": "success", "video_key": output_key}
            
        except Exception as e:
            # Catching locally to avoid serialization errors
            return {"status": "error", "reason": f"Generation failed: {str(e)}"}

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

    print("🛰️ Initializing Modal Remote Task (LTX-Video Optimized)...")
    gen = LTXVideoGenerator()
    result = gen.generate.remote(**config)
    
    if result["status"] == "success":
        print(f"🏁 Finished! Video available at R2 Key: {result['video_key']}")
    else:
        print(f"❌ Process Failed: {result['reason']}")
