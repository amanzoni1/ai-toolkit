import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import os
import json
import logging
from pathlib import Path
import sys
import time

# Configure PyTorch inductor settings for improved diffusion model performance
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_images(
    prompts: list,
    output_dir: str,
    model_path: str,
    num_steps: int = 20,
    guidance_scale: float = 4.5,
):
    """Generate images using the trained model with inference optimizations.

    Args:
        prompts (list): List of prompts to generate images from
        output_dir (str): Directory to save generated images
        model_path (str): Path to the trained model weights
        num_steps (int): Number of inference steps (default 20)
        guidance_scale (float): Guidance scale for generation
    """
    try:
        start_time = time.time()
        logger.info(f"Loading base model and LoRA weights from {model_path}")

        device = "cuda"
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")

        pipeline = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16,
        )
        pipeline.load_lora_weights(".", weight_name=model_path)
        pipeline = pipeline.to(device)

        # Create output directory if it doesn't exist.
        os.makedirs(output_dir, exist_ok=True)

        # Replace default scheduler with an optimized one.
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )

        # Enable attention slicing to save memory
        pipeline.enable_attention_slicing(slice_size="auto")

        # Memory and speed optimizations for GPU:
        pipeline.unet.to(memory_format=torch.channels_last)
        pipeline.vae.to(memory_format=torch.channels_last)

        # Compile UNet and VAE decoder for lower latency.
        # Try and consider also mode="max-autotune" for more aggressive optimizations.
        # This may require more memory and is not always faster, so check.
        try:
            pipeline.unet = torch.compile(
                pipeline.unet, mode="reduce-overhead", fullgraph=True
            )
            pipeline.vae.decode = torch.compile(
                pipeline.vae.decode, mode="reduce-overhead", fullgraph=True
            )
            logger.info("Successfully compiled UNet and VAE for faster inference")
        except Exception as e:
            logger.warning(f"Model compilation failed, continuing without it: {e}")

        # Process prompts
        for idx, prompt in enumerate(prompts):
            logger.info(
                f"Generating image {idx+1}/{len(prompts)} with prompt: {prompt}"
            )
            start_single = time.time()
            image = pipeline(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=1024,
                width=1024,
            ).images[0]
            gen_time = time.time() - start_single

            output_path = os.path.join(output_dir, f"gen_{idx:03d}.png")
            image.save(output_path)
            logger.info(f"Saved image to: {output_path} (generated in {gen_time:.2f}s)")

        total_time = time.time() - start_time
        logger.info(f"Generation completed successfully in {total_time:.2f} seconds")
        logger.info(f"Average time per image: {total_time/len(prompts):.2f} seconds")

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise


def main():
    try:
        # Get environment variables
        required_vars = ["PROMPTS", "OUTPUT_DIR", "MODEL_PATH"]
        missing_vars = [var for var in required_vars if var not in os.environ]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

        prompts = json.loads(os.environ["PROMPTS"])
        output_dir = os.environ["OUTPUT_DIR"]
        model_path = os.environ["MODEL_PATH"]

        # Optional parameters (using defaults: num_steps=20, guidance_scale=4.5)
        num_steps = int(os.environ.get("NUM_STEPS", "20"))
        guidance_scale = float(os.environ.get("GUIDANCE_SCALE", "4.5"))

        # Run generation
        generate_images(
            prompts=prompts,
            output_dir=output_dir,
            model_path=model_path,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
