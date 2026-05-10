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
        "optimum" # Required for some FP8 operations
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

app = modal.App("freedom-force-ltx-v2-3-fp8")

# =========================================================
# 2. MODEL CLASS
# =========================================================

@app.cls(
    gpu="L4", # 24GB is perfect for FP8 LTX
    image=image,
    timeout=7200,
)
class LTXVideoGenerator:

    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import LTXImageToVideoPipeline, LTXVideoTransformer3DModel

        print("🚀 Loading LTX-Video with FP8 Transformer...")
        
        # 1. Load the Transformer in FP8 (Native Quantization)
        # This is the 'v2.3' style optimization to make it fit and run fast
        transformer = LTXVideoTransformer3DModel.from_pretrained(
            "Lightricks/LTX-Video",
            subfolder="transformer",
            torch_dtype=torch.float8_e4m3fn, # Native FP8
        )

        # 2. Load the full pipeline using the FP8 transformer
        self.pipe = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            transformer=transformer,
            torch_dtype=torch.bfloat16, # Other components in BF16
        )

        # 3. High-Efficiency Memory Management
        # This allows the L4 to handle the T5-XXL text encoder (11GB) + Transformer
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
            return {"status": "error", "reason": f"S3 Error: {str(e)}"}

        # 2. Process Image (Resolution MUST be divisible by 32/64)
        image = Image.open(input_path).convert("RGB")
        # 768x512 is the 'sweet spot' for LTX high quality
        input_image = image.resize((768, 512)) 

        # 3. Generate (Using Latest STG Parameters)
        # Note: LTX v0.9/v2.3 logic uses stg_mode and specific guidance
        prompt = (
            "cinematic aerial combat, F22 fighter jet flying through a 3D futuristic city, "
            "extreme motion blur, camera tracking the jet, explosions in background, 4k, high detail"
        )
        
        print("🎬 Generating High-Quality Video (FP8 Mode)...")
        torch.cuda.empty_cache()
        gc.collect()

        try:
            with torch.inference_mode():
                # num_frames = (n * 8) + 1.   73 frames = ~3 seconds at 24fps
                video_frames = self.pipe(
                    image=input_image,
                    prompt=prompt,
                    negative_prompt="low quality, blurry, static, distorted, watermark",
                    width=768,
                    height=512,
                    num_frames=73,
                    num_inference_steps=30,
                    guidance_scale=3.0,
                ).frames[0]

            # 4. Export & Upload
            output_path = "/tmp/output.mp4"
            export_to_video(video_frames, output_path, fps=24)
            
            output_key = "output/ltx_v2_3_output.mp4"
            s3.upload_file(output_path, bucket, output_key)
            return {"status": "success", "video_key": output_key}
            
        except Exception as e:
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

    print("🛰️ Initializing Modal Task (LTX FP8)...")
    gen = LTXVideoGenerator()
    result = gen.generate.remote(**config)
    
    if result["status"] == "success":
        print(f"✅ Success! Key: {result['video_key']}")
    else:
        print(f"❌ Failed: {result['reason']}")
