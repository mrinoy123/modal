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
    gpu="L4",
    image=image,
    timeout=7200,
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

        # Optimization for L4 (24GB)
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
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

        # 1. Robust Download with Error Handling
        input_path = "/tmp/input.png"
        target_file = "queue/f22_raptor_raw.png"
        
        print(f"📥 Attempting to download: {target_file}")
        try:
            s3.download_file(bucket, target_file, input_path)
        except ClientError as e:
            # Catching locally to prevent "Deserialization Error" on the user's terminal
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "404":
                print(f"❌ Error: {target_file} not found in bucket '{bucket}'")
                # Debug: List what IS in the bucket
                objs = s3.list_objects_v2(Bucket=bucket, Prefix="queue/", MaxKeys=5)
                files = [obj['Key'] for obj in objs.get('Contents', [])]
                return {
                    "status": "error", 
                    "reason": f"File Not Found: {target_file}",
                    "found_instead": files
                }
            return {"status": "error", "reason": str(e)}

        # 2. Process Image
        image = Image.open(input_path).convert("RGB")
        # Dimensions MUST be divisible by 64 for LTX-Video
        image = image.resize((768, 448)) 

        # 3. Generate
        prompt = (
            "cinematic aerial combat, F22 fighter jet flying through a 3D futuristic city, "
            "extreme motion blur, camera tracking the jet, explosions in background, 4k"
        )
        
        print("🎬 Generating Video...")
        torch.cuda.empty_cache()
        gc.collect()

        try:
            with torch.inference_mode():
                video_frames = self.pipe(
                    image=image,
                    prompt=prompt,
                    negative_prompt="low quality, static, blurry, watermark",
                    width=768,
                    height=448,
                    num_frames=73, # ~3 seconds at 24fps
                    num_inference_steps=30,
                    guidance_scale=3.0,
                ).frames[0]

            # 4. Export & Upload
            output_path = "/tmp/output.mp4"
            export_to_video(video_frames, output_path, fps=24)
            
            output_key = "output/ltx_freedom_force.mp4"
            s3.upload_file(output_path, bucket, output_key)
            return {"status": "success", "video_key": output_key}
            
        except Exception as e:
            return {"status": "error", "reason": f"Inference failed: {str(e)}"}

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
    
    # We use a standard dictionary return to avoid local environment dependency issues
    result = gen.generate.remote(**config)
    
    if result["status"] == "success":
        print(f"✅ Finished! Video uploaded to: {result['video_key']}")
    else:
        print(f"❌ Task Failed: {result['reason']}")
        if "found_instead" in result:
            print(f"📂 Files found in /queue: {result['found_instead']}")
