import modal
import os

# =========================================================
# 1. IMAGE DEFINITION
# =========================================================

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libsm6", "libxext6", "wget")
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        # We need the cutting-edge diffusers branch that supports LTX-2.3
        "git+https://github.com/huggingface/diffusers.git",
        "transformers>=4.48.2",
        "accelerate>=1.3.0",
        "sentencepiece",
        "imageio[ffmpeg]",
        "pillow",
        "boto3",
        "hf_transfer",
        "gguf",      # Required for loading the GGUF transformer
        "xformers",
        "peft"
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

app = modal.App("freedom-force-ltx-2-3-gguf")

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
        # LTX2 classes are for the 2.3 version
        from diffusers import LTX2Pipeline, LTX2VideoTransformer3DModel, GGUFQuantizationConfig
        from huggingface_hub import hf_hub_download

        print("🚀 Loading LTX-2.3 (22B) GGUF Optimized Model...")
        
        # 1. Download the LTX-2.3 GGUF file
        # Using Unsloth's Q4_K_M quant which fits perfectly on L4 (approx 14GB)
        model_path = hf_hub_download(
            repo_id="unsloth/LTX-2.3-GGUF",
            filename="ltx-2.3-22b-dev-Q4_K_M.gguf"
        )

        # 2. Load the GGUF Transformer specifically
        # Note the class name: LTX2VideoTransformer3DModel
        transformer = LTX2VideoTransformer3DModel.from_single_file(
            model_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )

        # 3. Load the full LTX2 Pipeline (Text-to-Audio-Video)
        # We use the official Lightricks repo for components (VAE, Scheduler) but our GGUF transformer
        self.pipe = LTX2Pipeline.from_pretrained(
            "Lightricks/LTX-2.3",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )

        # 4. Memory Optimization for 24GB
        self.pipe.enable_model_cpu_offload() # Offloads Gemma 3 (12B) text encoder
        self.pipe.vae.enable_tiling()
        
        print("✅ LTX-2.3 GGUF Pipeline Ready!")

    @modal.method()
    def generate(self, endpoint: str, access_key: str, secret_key: str, bucket: str):
        import torch
        import boto3
        from PIL import Image
        from diffusers.pipelines.ltx2.export_utils import encode_video

        # Connect to R2
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )

        input_path = "/tmp/input.png"
        s3.download_file(bucket, "queue/f22_raptor_raw.png", input_path)

        # Process Image (multiples of 32/64)
        image = Image.open(input_path).convert("RGB").resize((768, 512))

        prompt = (
            "Cinematic aerial combat, F22 fighter jet flying through a 3D futuristic city, "
            "extreme motion blur, camera tracking the jet, explosions in background, 4k"
        )
        
        print("🎬 Generating LTX-2.3 Synchronized Video + Audio...")
        
        with torch.inference_mode():
            # LTX-2.3 generates video AND audio in one pass
            # frames formula: divisible by 8 + 1 (e.g., 97, 121)
            video, audio = self.pipe(
                image=image,
                prompt=prompt,
                negative_prompt="low quality, blurry, static, watermark",
                width=768,
                height=512,
                num_frames=121, 
                num_inference_steps=30,
                guidance_scale=3.5,
                return_dict=False
            )

        # 4. Combine Video and Audio using LTX2 utilities
        output_path = "/tmp/output.mp4"
        video_tensor = torch.from_numpy(video[0] * 255).round().to(torch.uint8)
        
        encode_video(
            video_tensor,
            fps=24.0,
            audio=audio[0].float().cpu(),
            audio_sample_rate=self.pipe.vocoder.config.output_sampling_rate,
            output_path=output_path,
        )
        
        output_key = "output/ltx_2_3_final.mp4"
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

    print("🛰️ Initializing Modal Task...")
    gen = LTXVideoGenerator()
    result = gen.generate.remote(**config)
    print(f"🏁 Finished: {result}")
