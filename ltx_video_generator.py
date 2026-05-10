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
        "diffusers==0.31.0",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "sentencepiece",
        "protobuf",
        "imageio[ffmpeg]",
        "pillow",
        "boto3",
        "hf_transfer",
        "xformers==0.0.28.post3"
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
    gpu="L4",
    image=image,
    timeout=7200,
    scaledown_window=300,
)
class LTXVideoGenerator:

    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import LTXImageToVideoPipeline

        print("🚀 Loading LTX Video Pipeline...")
        self.pipe = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16,
        )

        # Memory Optimizations
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("⚡ xformers enabled")
        except:
            print("⚠️ xformers not available, using default attention")

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

        # 1. Download
        print("📥 Downloading input image from queue/input.png...")
        input_path = "/tmp/input.png"
        s3.download_file(bucket, "queue/f22_raptor_raw.png", input_path)

        # 2. Process Image
        image = Image.open(input_path).convert("RGB").resize((768, 432))

        # 3. Generate
        prompt = (
            "cinematic aerial combat, F22 fighter jet flying through a 3D futuristic city, "
            "extreme motion blur, camera tracking the jet, explosions in background, 4k"
        )
        
        print("🎬 Generating Action Video...")
        torch.cuda.empty_cache()
        gc.collect()

        with torch.inference_mode():
            result = self.pipe(
                image=image,
                prompt=prompt,
                negative_prompt="low quality, static, 2d, blurry, watermark",
                width=768,
                height=432,
                num_frames=73,
                num_inference_steps=28,
                guidance_scale=3.5,
                decode_timestep=0.03,
                decode_noise_scale=0.025,
            ).frames[0]

        # 4. Export & Upload
        output_path = "/tmp/output.mp4"
        export_to_video(result, output_path, fps=24)
        
        output_key = "output/ltx_freedom_force.mp4"
        print(f"☁️ Uploading to R2: {output_key}")
        s3.upload_file(output_path, bucket, output_key)

        return {"status": "success", "video_key": output_key}

# =========================================================
# 3. TERMINAL ENTRYPOINT (Fixes the GitHub Action Error)
# =========================================================

@app.local_entrypoint()
def main():
    # Hardcoded config for your automated workflow
    config = {
        "endpoint": "https://4d91f4d3d0366568a54ffa32ffcb7bf4.r2.cloudflarestorage.com",
        "access_key": "3c33425ba6e5abbd3e63afab14dc8866",
        "secret_key": "d65f107bb61093843c6dd980c764443fdf50924a7701078b99f007d3060e25a8",
        "bucket": "video-asset-files-storage-workflow"
    }

    print("🛰️ Initializing Modal Remote Task...")
    gen = LTXVideoGenerator()
    
    # Passing arguments individually fixes the "dict" parsing error
    result = gen.generate.remote(
        endpoint=config["endpoint"],
        access_key=config["access_key"],
        secret_key=config["secret_key"],
        bucket=config["bucket"]
    )
    
    print(f"🏁 Process Finished: {result}")
