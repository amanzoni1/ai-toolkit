import torch
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionUpscalePipeline,
)
import os
import json
import logging
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_image(image, output_path, metadata=None):
    """Save image and optional metadata."""
    image.save(output_path)
    if metadata:
        metadata_path = output_path.replace(".png", ".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    return output_path


def high_res_tiled_generation(
    pipeline,
    prompt,
    negative_prompt="",
    num_inference_steps=20,
    guidance_scale=4.5,
    width=2048,
    height=2048,
    tile_size=512,
    tile_overlap=64,
):
    """Generate high-resolution image using tiling technique.

    Args:
        pipeline: Diffusion pipeline
        prompt: Text prompt
        negative_prompt: Negative prompt text
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale
        width: Target image width
        height: Target image height
        tile_size: Size of each tile
        tile_overlap: Overlap between tiles

    Returns:
        PIL Image of the generated high-resolution image
    """
    # Calculate number of tiles in each dimension
    x_tiles = (width - tile_overlap) // (tile_size - tile_overlap)
    y_tiles = (height - tile_overlap) // (tile_size - tile_overlap)

    # Adjust final size based on tiles
    final_width = (tile_size - tile_overlap) * x_tiles + tile_overlap
    final_height = (tile_size - tile_overlap) * y_tiles + tile_overlap

    logger.info(
        f"Generating {x_tiles}x{y_tiles} tiles for {final_width}x{final_height} image"
    )

    # Create empty canvas
    canvas = Image.new("RGB", (final_width, final_height))

    # Generate each tile
    for y in range(y_tiles):
        for x in range(x_tiles):
            # Calculate tile position
            pos_x = x * (tile_size - tile_overlap)
            pos_y = y * (tile_size - tile_overlap)

            logger.info(f"Generating tile ({x+1},{y+1}) at position ({pos_x},{pos_y})")

            # Adjust prompt to focus on this tile section
            # Position encoding helps the model understand what part to generate
            position_text = (
                f", {x/x_tiles*100:.0f}% from left, {y/y_tiles*100:.0f}% from top"
            )
            tile_prompt = prompt + position_text

            # Generate the tile
            tile = pipeline(
                prompt=tile_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=tile_size,
                width=tile_size,
            ).images[0]

            # Create blending mask for smooth transitions
            mask = Image.new("L", (tile_size, tile_size), 255)

            # Feather edges for tiles that are not on the borders
            if x > 0:  # Left edge
                for i in range(tile_overlap):
                    mask.putpixel((i, 0), int(255 * (i / tile_overlap)))
            if y > 0:  # Top edge
                for i in range(tile_overlap):
                    for j in range(tile_size):
                        mask.putpixel(
                            (j, i),
                            min(mask.getpixel((j, i)), int(255 * (i / tile_overlap))),
                        )
            if x < x_tiles - 1:  # Right edge
                for i in range(tile_overlap):
                    for j in range(tile_size):
                        mask.putpixel(
                            (tile_size - i - 1, j),
                            min(
                                mask.getpixel((tile_size - i - 1, j)),
                                int(255 * (i / tile_overlap)),
                            ),
                        )
            if y < y_tiles - 1:  # Bottom edge
                for i in range(tile_overlap):
                    for j in range(tile_size):
                        mask.putpixel(
                            (j, tile_size - i - 1),
                            min(
                                mask.getpixel((j, tile_size - i - 1)),
                                int(255 * (i / tile_overlap)),
                            ),
                        )

            # Paste tile onto canvas with blending mask
            canvas.paste(tile, (pos_x, pos_y), mask)

    return canvas


def multi_step_generation(
    text_to_image_pipeline,
    img2img_pipeline,
    upscale_pipeline,
    prompt,
    negative_prompt="",
    initial_steps=20,
    refinement_steps=15,
    upscale_steps=10,
    initial_size=1024,
    final_size=2048,
    guidance_scale=4.5,
    refinement_strength=0.4,
):
    """Generate image using a multi-step pipeline with refinement.

    Args:
        text_to_image_pipeline: Text-to-image pipeline
        img2img_pipeline: Image-to-image pipeline
        upscale_pipeline: Upscale pipeline (optional)
        prompt: Text prompt
        negative_prompt: Negative prompt text
        initial_steps: Steps for initial generation
        refinement_steps: Steps for refinement
        upscale_steps: Steps for upscaling
        initial_size: Initial image size
        final_size: Final image size after upscaling
        guidance_scale: Guidance scale
        refinement_strength: Strength for img2img refinement

    Returns:
        PIL Image of the final generated image
    """
    # Step 1: Initial image generation
    logger.info("Step 1: Generating initial image")
    initial_image = text_to_image_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=initial_steps,
        guidance_scale=guidance_scale,
        height=initial_size,
        width=initial_size,
    ).images[0]

    # Step 2: Refinement using img2img
    logger.info("Step 2: Refining image")
    refined_image = img2img_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=initial_image,
        strength=refinement_strength,
        num_inference_steps=refinement_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    # Step 3: Upscaling
    if upscale_pipeline is not None and final_size > initial_size:
        logger.info("Step 3: Upscaling image")
        final_image = upscale_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=refined_image,
            num_inference_steps=upscale_steps,
            guidance_scale=guidance_scale,
        ).images[0]
    else:
        final_image = refined_image

    return final_image


def generate_images(
    prompts: list,
    output_dir: str,
    model_path: str,
    num_steps: int = 20,
    guidance_scale: float = 4.5,
    batch_size: int = 1,
    save_metadata: bool = True,
    generation_mode: str = "standard",  # standard, high_res, or multi_step
    width: int = 1024,
    height: int = 1024,
    tile_size: int = 512,
    tile_overlap: int = 64,
    refinement_strength: float = 0.4,
    negative_prompt: str = "",
):
    """Generate images using the trained model with improved inference optimizations.

    Args:
        prompts (list): List of prompts to generate images from
        output_dir (str): Directory to save generated images
        model_path (str): Path to the trained model weights
        num_steps (int): Number of inference steps
        guidance_scale (float): Guidance scale for generation
        batch_size (int): Number of images to generate in parallel
        save_metadata (bool): Whether to save generation metadata
        generation_mode (str): "standard", "high_res", or "multi_step"
        width (int): Width of the generated image
        height (int): Height of the generated image
        tile_size (int): Size of tiles for high-res generation
        tile_overlap (int): Overlap between tiles
        refinement_strength (float): Strength for img2img refinement
        negative_prompt (str): Negative prompt for generation
    """
    try:
        start_time = time.time()
        logger.info(f"Loading base model and LoRA weights from {model_path}")

        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        else:
            device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")

        # Load pipeline with appropriate dtype based on device
        dtype = torch.float16 if device == "cuda" else torch.float32

        # Load different pipelines based on generation mode
        if generation_mode == "standard" or generation_mode == "high_res":
            pipeline = DiffusionPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", torch_dtype=dtype, safety_checker=None
            )
            img2img_pipeline = None
            upscale_pipeline = None
        else:  # multi_step
            # Load text-to-image pipeline
            pipeline = DiffusionPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", torch_dtype=dtype, safety_checker=None
            )

            # Load img2img pipeline
            img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", torch_dtype=dtype, safety_checker=None
            )

            # Load upscale pipeline if needed
            if width > 1024 or height > 1024:
                upscale_pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                    "stabilityai/stable-diffusion-x4-upscaler",
                    torch_dtype=dtype,
                    safety_checker=None,
                )
            else:
                upscale_pipeline = None

        # Load LoRA weights if model_path is provided
        if model_path:
            pipeline.load_lora_weights(".", weight_name=model_path)
            if img2img_pipeline:
                img2img_pipeline.load_lora_weights(".", weight_name=model_path)

        # Move models to device
        pipeline = pipeline.to(device)
        if img2img_pipeline:
            img2img_pipeline = img2img_pipeline.to(device)
        if upscale_pipeline:
            upscale_pipeline = upscale_pipeline.to(device)

        # Apply optimizations to all pipelines
        for pipe in [
            p for p in [pipeline, img2img_pipeline, upscale_pipeline] if p is not None
        ]:
            # Change scheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="sde-dpmsolver++",
                timestep_spacing="trailing",
                use_karras_sigmas=True,
            )

            if device == "cuda":
                # Enable attention slicing
                pipe.enable_attention_slicing(slice_size="auto")

                # Set memory format
                if hasattr(pipe, "unet"):
                    pipe.unet.to(memory_format=torch.channels_last)
                if hasattr(pipe, "vae"):
                    pipe.vae.to(memory_format=torch.channels_last)

                # Enable xformers
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    logger.warning(f"Could not enable xformers for one pipeline: {e}")

                # Try to compile models
                try:
                    if hasattr(pipe, "unet"):
                        pipe.unet = torch.compile(
                            pipe.unet, mode="reduce-overhead", fullgraph=True
                        )
                    if hasattr(pipe, "vae") and hasattr(pipe.vae, "decode"):
                        pipe.vae.decode = torch.compile(
                            pipe.vae.decode, mode="reduce-overhead", fullgraph=True
                        )
                except Exception as e:
                    logger.warning(f"Model compilation failed for one pipeline: {e}")

        logger.info(
            f"Model loading completed in {time.time() - start_time:.2f} seconds"
        )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Process each prompt
        for idx, prompt in enumerate(prompts):
            logger.info(f"Generating image {idx+1}/{len(prompts)}")
            logger.info(f"Prompt: {prompt}")

            start_single = time.time()

            # Different generation modes
            if generation_mode == "standard":
                image = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                ).images[0]

            elif generation_mode == "high_res":
                image = high_res_tiled_generation(
                    pipeline=pipeline,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                )

            elif generation_mode == "multi_step":
                image = multi_step_generation(
                    text_to_image_pipeline=pipeline,
                    img2img_pipeline=img2img_pipeline,
                    upscale_pipeline=upscale_pipeline,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    initial_steps=num_steps,
                    refinement_steps=max(10, num_steps // 2),
                    upscale_steps=max(10, num_steps // 2),
                    initial_size=min(1024, width),
                    final_size=max(width, height),
                    guidance_scale=guidance_scale,
                    refinement_strength=refinement_strength,
                )

            gen_time = time.time() - start_single

            output_path = os.path.join(output_dir, f"gen_{idx:03d}.png")

            if save_metadata:
                metadata = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "steps": num_steps,
                    "guidance_scale": guidance_scale,
                    "generation_time": gen_time,
                    "generation_mode": generation_mode,
                    "width": width,
                    "height": height,
                    "model": "black-forest-labs/FLUX.1-dev",
                    "lora": model_path,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_image(image, output_path, metadata)
            else:
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

        # Optional parameters
        num_steps = int(os.environ.get("NUM_STEPS", "20"))
        guidance_scale = float(os.environ.get("GUIDANCE_SCALE", "4.5"))
        batch_size = int(os.environ.get("BATCH_SIZE", "1"))
        save_metadata = os.environ.get("SAVE_METADATA", "true").lower() == "true"
        generation_mode = os.environ.get("GENERATION_MODE", "standard")
        width = int(os.environ.get("WIDTH", "1024"))
        height = int(os.environ.get("HEIGHT", "1024"))
        tile_size = int(os.environ.get("TILE_SIZE", "512"))
        tile_overlap = int(os.environ.get("TILE_OVERLAP", "64"))
        refinement_strength = float(os.environ.get("REFINEMENT_STRENGTH", "0.4"))
        negative_prompt = os.environ.get("NEGATIVE_PROMPT", "")

        # Run generation
        generate_images(
            prompts=prompts,
            output_dir=output_dir,
            model_path=model_path,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            batch_size=batch_size,
            save_metadata=save_metadata,
            generation_mode=generation_mode,
            width=width,
            height=height,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            refinement_strength=refinement_strength,
            negative_prompt=negative_prompt,
        )

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
