import modal

# =========================================================
# IMAGE
# =========================================================

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )

    # -----------------------------------------------------
    # SYSTEM
    # -----------------------------------------------------

    .apt_install(
        "git",
        "ffmpeg",
        "wget",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6"
    )

    # -----------------------------------------------------
    # INSTALLS
    # -----------------------------------------------------

    .run_commands(

        "pip install --upgrade pip",

        # CUDA TORCH
        "pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",

        # STABLE LTX STACK
        "pip install diffusers==0.31.0",
        "pip install transformers==4.46.3",
        "pip install accelerate==1.1.1",

        # VIDEO
        "pip install imageio imageio[ffmpeg]",
        "pip install opencv-python pillow",

        # UTILS
        "pip install sentencepiece protobuf",
        "pip install boto3",
        "pip install hf_transfer",

        # XFORMERS
        "pip install xformers==0.0.28.post3",
    )

    # -----------------------------------------------------
    # ENVIRONMENT
    # -----------------------------------------------------

    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",

        # IMPORTANT
        "TORCHINDUCTOR_DISABLE": "1",

        # Prevent CUDA fragmentation
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

# =========================================================
# APP
# =========================================================

app = modal.App("ltx-video-stable")


# =========================================================
# MODEL CLASS
# =========================================================

@app.cls(
    gpu="L4",
    image=image,
    timeout=7200,
    scaledown_window=300,
)
class LTXVideoGenerator:

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    @modal.enter()
    def load_model(self):

        import torch

        from diffusers import LTXImageToVideoPipeline

        print("🚀 Loading LTX Video Pipeline...")

        self.pipe = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16,
        )

        # -------------------------------------------------
        # MEMORY OPTIMIZATION
        # -------------------------------------------------

        self.pipe.enable_model_cpu_offload()

        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("⚡ xformers enabled")
        except Exception as e:
            print("xformers failed:", e)

        print("✅ LTX pipeline ready")


    # =====================================================
    # GENERATE VIDEO
    # =====================================================

    @modal.method()
    def generate_video(self, config: dict):

        import gc
        import torch
        import boto3

        from PIL import Image
        from diffusers.utils import export_to_video

        # -------------------------------------------------
        # S3 / R2
        # -------------------------------------------------

        s3 = boto3.client(
            "s3",
            endpoint_url=config["endpoint"],
            aws_access_key_id=config["access_key"],
            aws_secret_access_key=config["secret_key"],
            region_name="auto",
        )

        # -------------------------------------------------
        # DOWNLOAD IMAGE
        # -------------------------------------------------

        print("📥 Downloading input image...")

        input_path = "/tmp/input.png"

        s3.download_file(
            config["bucket"],
            "queue/input.png",
            input_path
        )

        image = (
            Image.open(input_path)
            .convert("RGB")
            .resize((768, 432))
        )

        # -------------------------------------------------
        # PROMPTS
        # -------------------------------------------------

        prompt = (
            "cinematic aerial combat scene, "
            "an F22 fighter jet flies aggressively through "
            "a futuristic cyberpunk city skyline at sunset, "
            "dynamic camera movement, realistic lighting, "
            "smoke trails, cinematic action movie atmosphere, "
            "high detail, realistic motion"
        )

        negative_prompt = (
            "low quality, blurry, distorted, "
            "watermark, text, flickering, duplicate objects"
        )

        # -------------------------------------------------
        # GENERATE
        # -------------------------------------------------

        print("🎬 Generating video...")

        torch.cuda.empty_cache()
        gc.collect()

        with torch.inference_mode():

            result = self.pipe(

                image=image,

                prompt=prompt,
                negative_prompt=negative_prompt,

                width=768,
                height=432,

                num_frames=73,

                num_inference_steps=28,

                guidance_scale=3.5,

                # improves temporal consistency
                decode_timestep=0.03,
                decode_noise_scale=0.025,
            )

            frames = result.frames[0]

        # -------------------------------------------------
        # EXPORT
        # -------------------------------------------------

        output_path = "/tmp/ltx_video.mp4"

        print("💾 Exporting MP4...")

        export_to_video(
            frames,
            output_path,
            fps=24
        )

        # -------------------------------------------------
        # UPLOAD
        # -------------------------------------------------

        output_key = "output/ltx_video.mp4"

        print("☁️ Uploading to R2...")

        s3.upload_file(
            output_path,
            config["bucket"],
            output_key
        )

        print("✅ Upload complete")

        return {
            "status": "success",
            "video": output_key
        }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    cloudflare_config = {
        "endpoint": "https://4d91f4d3d0366568a54ffa32ffcb7bf4.r2.cloudflarestorage.com",
        "access_key": "3c33425ba6e5abbd3e63afab14dc8866",
        "secret_key": "d65f107bb61093843c6dd980c764443fdf50924a7701078b99f007d3060e25a8",
        "bucket": "video-asset-files-storage-workflow"
    }

    with app.run():

        generator = LTXVideoGenerator()

        result = generator.generate_video.remote(
            cloudflare_config
        )

        print(result)
