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
        "diffusers>=0.36.1", # CRITICAL: LTX-Video needs 0.35.0+
        "transformers>=4.48.0",
        "accelerate>=1.2.0",
        "sentencepiece",
        "protobuf",
        "imageio[ffmpeg]",
        "pillow",
        "boto3",
        "hf_transfer",
        "xformers",
        "peft" 
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

app = modal.App("freedom-force-ltx-stable")

# =========================================================
# 2. MODEL CLASS
# =========================================================

@app.cls(
    gpu="L4", # 24GB VRAM is tight but sufficient with offloading
    image=image,
    timeout=7200,
    scaledown_window=300,
)
class LTXVideoGenerator:

    @modal.enter()
    def load_model(self):
        import torch
        # Now this import will succeed because diffusers is updated
        from diffusers import LTXImageToVideoPipeline

        print("🚀 Loading LTX Video Pipeline (Lightricks/LTX-Video)...")
        
        # Load in bfloat16 to save memory and maintain quality
        self.pipe = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16,
        )

        # Memory Optimizations for L4 GPU
        # enable_model_cpu_offload is better than .to("cuda") for 24GB cards
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("⚡ xformers enabled")
        except Exception as e:
            print(f"⚠️ xformers notice: {e}")

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

        # 1. Download Input
        print("📥 Downloading input image...")
        input_path = "/tmp/input.png"
        # Note: Using your hardcoded key from the original script
        s3.download_file(bucket, "queue/f22_raptor_raw.png", input_path)

        # 2. Process Image (LTX-Video likes multiples of 32 or 64)
        image = Image.open(input_path).convert("RGB")
        image = image.resize((768, 448)) # Adjusted to 448 (divisible by 64)

        # 3. Generate
        prompt = (
            "cinematic aerial combat, F22 fighter jet flying through a 3D futuristic city, "
            "extreme motion blur, camera tracking the jet, explosions in background, 4k"
        )
        
        print("🎬 Generating Action Video (this may take a few minutes)...")
        torch.cuda.empty_cache()
        gc.collect()

        with torch.inference_mode():
            # LTX-Video specific parameters
            video_frames = self.pipe(
                image=image,
                prompt=prompt,
                negative_prompt="low quality, static, 2d, blurry, watermark, deformed, ugly",
                width=768,
                height=448,
                num_frames=73,
                num_inference_steps=30,
                guidance_scale=3.0,
            ).frames[0]

        # 4. Export & Upload
        output_path = "/tmp/output.mp4"
        export_to_video(video_frames, output_path, fps=24)
        
        output_key = "output/ltx_freedom_force.mp4"
        print(f"☁️ Uploading to R2: {output_key}")
        s3.upload_file(output_path, bucket, output_key)

        return {"status": "success", "video_key": output_key}

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
    # Instantiate the class
    gen = LTXVideoGenerator()
    
    # Call the remote method
    result = gen.generate.remote(
        endpoint=config["endpoint"],
        access_key=config["access_key"],
        secret_key=config["secret_key"],
        bucket=config["bucket"]
    )
    
    print(f"🏁 Process Finished: {result}")
