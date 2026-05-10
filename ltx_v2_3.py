import modal
import os

# =========================================================
# 1. IMAGE DEFINITION
# =========================================================

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libsm6", "libxext6", "wget")
    .pip_install(
        "torch",
        "torchvision",
        # Install diffusers from main to get latest LTX-2.3 support
        "git+https://github.com/huggingface/diffusers.git",
        "transformers>=4.51.0",
        "accelerate>=1.6.0",
        "sentencepiece",
        "imageio[ffmpeg]",
        "pillow",
        "boto3",
        "hf_transfer",
        "gguf",      # Required for GGUF loading
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
        # LTX-2.3 specific imports from latest diffusers
        from diffusers import LTX2Pipeline, LTX2VideoTransformer3DModel, GGUFQuantizationConfig
        from huggingface_hub import hf_hub_download

        print("🚀 Loading LTX-2.3 (19B) GGUF Optimized Model...")
        
        # 1. Download the GGUF Transformer (using Q5_0 for best quality/size balance)
        # Using Unsloth or city96 community quants to fit in 24GB VRAM
        model_path = hf_hub_download(
            repo_id="unsloth/LTX-2-GGUF",
            filename="ltx-2-19b-dev-Q5_0.gguf"
        )

        # 2. Load the GGUF Transformer
        transformer = LTXVideoTransformer3DModel.from_single_file(
            model_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )

        # 3. Load full pipeline (handles VAE and Text Encoder internally)
        # We point to the base Lightricks/LTX-2 repo for components, but swap the transformer
        self.pipe = LTX2Pipeline.from_pretrained(
            "Lightricks/LTX-2",
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )

        # 4. Memory Optimization
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()
        
        print("✅ LTX-2.3 Pipeline Ready!")

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

        # LTX-2.3 Resolution: Must be divisible by 32
        image = Image.open(input_path).convert("RGB").resize((768, 512))

        prompt = (
            "Cinematic aerial combat, F22 fighter jet flying through a 3D futuristic city, "
            "extreme motion blur, camera tracking the jet, explosions in background, 4k. "
            "Sound of jet engines roaring and distant explosions."
        )
        
        print("🎬 Generating LTX-2.3 Synchronized Video + Audio...")
        
        with torch.inference_mode():
            # LTX-2.3 returns both video and audio latents
            # num_frames must be divisible by 8 + 1 (e.g., 121)
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

        # Export & Upload
        output_path = "/tmp/output.mp4"
        # LTX-2 has a special utility to combine the generated video and audio
        video_tensor = torch.from_numpy(video[0] * 255).round().to(torch.uint8)
        encode_video(
            video_tensor,
            fps=24.0,
            audio=audio[0].float().cpu(),
            audio_sample_rate=self.pipe.vocoder.config.output_sampling_rate,
            output_path=output_path,
        )
        
        output_key = "output/ltx_2_3_gguf_final.mp4"
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

    print("🛰️ Initializing LTX-2.3 Task...")
    gen = LTXVideoGenerator()
    result = gen.generate.remote(**config)
    print(f"🏁 Result: {result}")
