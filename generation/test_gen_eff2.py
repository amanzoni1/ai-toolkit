import torch
from diffusers import FluxPipeline
from diffusers.hooks import apply_group_offloading
import os
import json
import logging
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_images(
    prompts: list,
    output_dir: str,
    model_path: str = None,
    num_steps: int = 40,
    guidance_scale: float = 4.5,
    height: int = 1024,
    width: int = 1024,
    use_fp16: bool = True,
    optimize_vram: bool = True,
):
    """Generate images using the Flux model with optimized settings.

    Args:
        prompts (list): List of prompts to generate images from
        output_dir (str): Directory to save generated images
        model_path (str, optional): Path to LoRA weights if using fine-tuned model
        num_steps (int): Number of inference steps
        guidance_scale (float): Guidance scale for generation
        height (int): Height of generated images
        width (int): Width of generated images
        use_fp16 (bool): Whether to use FP16 precision
        optimize_vram (bool): Whether to apply VRAM optimizations
    """
    try:
        # Determine model ID (using dev variant which provides better quality at 50 steps)
        model_id = "black-forest-labs/FLUX.1-dev"

        # Set appropriate dtype based on hardware availability and user preference
        if use_fp16:
            dtype = torch.float16
        else:
            dtype = torch.bfloat16  # Better numerical stability than float16

        logger.info(f"Loading Flux model from {model_id} with {dtype} precision")

        # Load the pipeline with proper dtype
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

        # Apply optimizations for VRAM usage
        if optimize_vram:
            logger.info("Applying VRAM optimizations")

            # Enable VAE optimizations
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()

            # Apply group offloading to reduce VRAM usage
            apply_group_offloading(
                pipe.transformer,
                offload_type="leaf_level",
                offload_device=torch.device("cpu"),
                onload_device=torch.device("cuda"),
                use_stream=True,  # For async data streaming on supported devices
            )
            apply_group_offloading(
                pipe.text_encoder,
                offload_device=torch.device("cpu"),
                onload_device=torch.device("cuda"),
                offload_type="leaf_level",
                use_stream=True,
            )
            apply_group_offloading(
                pipe.text_encoder_2,
                offload_device=torch.device("cpu"),
                onload_device=torch.device("cuda"),
                offload_type="leaf_level",
                use_stream=True,
            )
            apply_group_offloading(
                pipe.vae,
                offload_device=torch.device("cpu"),
                onload_device=torch.device("cuda"),
                offload_type="leaf_level",
                use_stream=True,
            )
        else:
            # If not using group offloading, still enable model CPU offload
            # as a lighter optimization
            pipe.enable_model_cpu_offload()

        # If LoRA weights are provided, load them
        if model_path:
            logger.info(f"Loading LoRA weights from {model_path}")
            pipe.load_lora_weights(".", weight_name=model_path)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Recommended steps for Flux.1-dev is ~50 for good quality
        if num_steps < 30 and guidance_scale > 0:
            logger.warning(
                f"Flux.1-dev typically needs ~50 steps for good quality with guidance. Currently using {num_steps}"
            )

        # Process each prompt
        for idx, prompt in enumerate(prompts):
            logger.info(f"Generating image {idx+1}/{len(prompts)}")
            logger.info(f"Prompt: {prompt}")

            # Generate the image with recommended settings for Flux
            # Using fixed seed for reproducibility
            generator = torch.Generator("cuda").manual_seed(idx)

            image = pipe(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
                output_type="pil",  # Ensure PIL output for saving
            ).images[0]

            # Save the generated image
            output_path = os.path.join(output_dir, f"gen_{idx:03d}.png")
            image.save(output_path)
            logger.info(f"Saved image to: {output_path}")

        logger.info("Generation completed successfully")

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise


def main():
    try:
        # Get environment variables
        required_vars = ["PROMPTS", "OUTPUT_DIR"]
        missing_vars = [var for var in required_vars if var not in os.environ]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

        prompts = json.loads(os.environ["PROMPTS"])
        output_dir = os.environ["OUTPUT_DIR"]

        # Optional parameters
        model_path = os.environ.get("MODEL_PATH", None)
        num_steps = int(os.environ.get("NUM_STEPS", "50"))  # Default to 50 for Flux-dev
        guidance_scale = float(
            os.environ.get("GUIDANCE_SCALE", "3.5")
        )  # Flux default is 3.5
        height = int(
            os.environ.get("HEIGHT", "768")
        )  # Changed default to more memory-efficient size
        width = int(os.environ.get("WIDTH", "1360"))  # Flux default size
        use_fp16 = os.environ.get("USE_FP16", "True").lower() == "true"
        optimize_vram = os.environ.get("OPTIMIZE_VRAM", "True").lower() == "true"

        # Run generation
        generate_images(
            prompts=prompts,
            output_dir=output_dir,
            model_path=model_path,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            use_fp16=use_fp16,
            optimize_vram=optimize_vram,
        )
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
