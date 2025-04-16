import torch
from diffusers import DiffusionPipeline
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

# Enable TF32 for faster matrix multiplications on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True


def get_optimized_pipeline(model_path: str):
    """
    Load and optimize the Flux pipeline for inference speed.

    Args:
        model_path (str): Path to the trained model weights

    Returns:
        An optimized pipeline for generation
    """
    try:
        logger.info(f"Loading base model from 'black-forest-labs/FLUX.1-dev'")
        pipeline = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16
        )
        logger.info(f"Loading LoRA weights from {model_path}")
        pipeline.load_lora_weights(".", weight_name=model_path)

        # Disable dynamic shifting in the scheduler to avoid the mu parameter issue
        if hasattr(pipeline.scheduler, "config") and hasattr(
            pipeline.scheduler.config, "use_dynamic_shifting"
        ):
            pipeline.scheduler.config.use_dynamic_shifting = False
            logger.info(
                "Disabled dynamic shifting in scheduler for better compatibility"
            )

        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
            logger.info("Moved pipeline to GPU ('cuda')")

            # Apply memory optimizations
            offload_device = torch.device("cpu")
            onload_device = torch.device("cuda")
            use_stream = True

            try:
                if hasattr(pipeline, "transformer"):
                    apply_group_offloading(
                        pipeline.transformer,
                        offload_type="leaf_level",
                        offload_device=offload_device,
                        onload_device=onload_device,
                        use_stream=use_stream,
                    )
                if hasattr(pipeline, "text_encoder"):
                    apply_group_offloading(
                        pipeline.text_encoder,
                        offload_type="leaf_level",
                        offload_device=offload_device,
                        onload_device=onload_device,
                        use_stream=use_stream,
                    )
                if hasattr(pipeline, "text_encoder_2"):
                    apply_group_offloading(
                        pipeline.text_encoder_2,
                        offload_type="leaf_level",
                        offload_device=offload_device,
                        onload_device=onload_device,
                        use_stream=use_stream,
                    )
                if hasattr(pipeline, "vae"):
                    apply_group_offloading(
                        pipeline.vae,
                        offload_type="leaf_level",
                        offload_device=offload_device,
                        onload_device=onload_device,
                        use_stream=use_stream,
                    )
                logger.info("Applied group offloading for model components")
            except Exception as e:
                logger.warning(f"Group offloading could not be applied: {e}")
        else:
            logger.info("CUDA not available. Running on CPU; performance may be slow.")
            pipeline = pipeline.to("cpu")

        # Enable memory optimizations for the VAE
        if hasattr(pipeline, "vae"):
            try:
                pipeline.vae.enable_slicing()
                pipeline.vae.enable_tiling()
                logger.info(
                    "Enabled VAE slicing and tiling for additional memory efficiency"
                )
            except Exception as e:
                logger.warning(f"VAE slicing/tiling could not be enabled: {e}")

        # Try to enable xFormers for faster attention
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xFormers memory efficient attention")
        except Exception as e:
            logger.warning(f"xFormers could not be enabled: {e}")

        return pipeline
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def generate_images(
    pipeline,
    prompts: list,
    output_dir: str,
    num_steps: int = 15,
    guidance_scale: float = 4.5,
    height: int = 1024,
    width: int = 1024,
    num_images_per_prompt: int = 1,
    start_index: int = 0,
):
    """
    Generate images using the provided pipeline.

    Args:
        pipeline: The DiffusionPipeline to use
        prompts (list): List of prompts to generate images from
        output_dir (str): Directory to save generated images
        num_steps (int): Number of inference steps
        guidance_scale (float): Guidance scale for generation
        height (int): Height of generated images
        width (int): Width of generated images
        num_images_per_prompt (int): Number of images to generate per prompt
        start_index (int): Starting index for naming generated files

    Returns:
        int: The next available index after generation
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a generator for reproducibility
    generator = None
    if torch.cuda.is_available():
        generator = torch.Generator(device="cuda").manual_seed(42)

    # Patch the retrieve_timesteps function to avoid custom sigmas issues
    from diffusers.pipelines.flux import pipeline_flux

    original_retrieve_timesteps = getattr(pipeline_flux, "retrieve_timesteps", None)

    if original_retrieve_timesteps:

        def patched_retrieve_timesteps(
            scheduler, num_inference_steps=None, device=None, timesteps=None, **kwargs
        ):
            """A patched version that doesn't use custom sigmas"""
            if not hasattr(scheduler, "timesteps") or scheduler.timesteps is None:
                if num_inference_steps is not None:
                    scheduler.set_timesteps(
                        num_inference_steps=num_inference_steps, device=device
                    )
                else:
                    raise ValueError(
                        f"Number of inference steps is {num_inference_steps}, but `set_timesteps` was not called for the scheduler"
                    )

            timesteps = scheduler.timesteps
            return timesteps, len(timesteps)

        # Apply the patch
        pipeline_flux.retrieve_timesteps = patched_retrieve_timesteps
        logger.info("Applied patch to retrieve_timesteps function")

    current_index = start_index

    try:
        # Generate images for each prompt
        for idx, prompt in enumerate(prompts):
            logger.info(
                f"Generating {num_images_per_prompt} image(s) for prompt {idx+1}/{len(prompts)}"
            )
            logger.info(f"Prompt: {prompt}")

            try:
                # Set timesteps before calling the pipeline
                pipeline.scheduler.set_timesteps(
                    num_inference_steps=num_steps, device=pipeline.device
                )

                # Generate the image(s)
                result = pipeline(
                    prompt=prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                    num_images_per_prompt=num_images_per_prompt,
                )

                # Save the images with sequential naming
                for img_idx, image in enumerate(result.images):
                    output_path = os.path.join(
                        output_dir, f"gen_{current_index:03d}.png"
                    )
                    image.save(output_path)
                    logger.info(f"Saved image to: {output_path}")
                    current_index += 1

            except Exception as e:
                logger.error(f"Error generating image for prompt {idx+1}: {e}")
                # Continue with next prompt instead of failing completely
                continue
    finally:
        # Restore the original function if we patched it
        if original_retrieve_timesteps:
            pipeline_flux.retrieve_timesteps = original_retrieve_timesteps

    return current_index


def process_prompt_data(
    prompts_data, pipeline, output_dir, num_steps, guidance_scale, height, width
):
    """
    Process a list of prompts with their counts.

    Args:
        prompts_data (list): List of dictionaries with 'prompt' and 'count' keys
        pipeline: The DiffusionPipeline to use
        output_dir (str): Directory to save generated images
        num_steps (int): Number of inference steps
        guidance_scale (float): Guidance scale for generation
        height (int): Height of generated images
        width (int): Width of generated images

    Returns:
        int: Total number of images generated
    """
    current_index = 0

    for prompt_data in prompts_data:
        prompt = prompt_data.get("prompt", "")
        count = prompt_data.get("count", 1)

        if not prompt:
            continue

        logger.info(f"Processing prompt with count {count}: {prompt[:50]}...")

        current_index = generate_images(
            pipeline=pipeline,
            prompts=[prompt],  # Single prompt
            output_dir=output_dir,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_images_per_prompt=count,
            start_index=current_index,
        )

    return current_index


def main():
    try:
        # Check if we're in prompt_data mode or regular mode
        prompts_data_str = os.environ.get("PROMPTS_DATA")

        # Get common parameters
        output_dir = os.environ["OUTPUT_DIR"]
        model_path = os.environ["MODEL_PATH"]
        num_steps = int(os.environ.get("NUM_STEPS", "15"))
        guidance_scale = float(os.environ.get("GUIDANCE_SCALE", "4.5"))
        width = int(os.environ.get("WIDTH", "1024"))
        height = int(os.environ.get("HEIGHT", "1024"))

        # Load the model once
        pipeline = get_optimized_pipeline(model_path)

        if prompts_data_str:
            # Process in prompt_data mode (multiple prompts with counts)
            prompts_data = json.loads(prompts_data_str)
            logger.info(
                f"Running in prompt_data mode with {len(prompts_data)} prompt items"
            )

            total_images = process_prompt_data(
                prompts_data=prompts_data,
                pipeline=pipeline,
                output_dir=output_dir,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            )

            logger.info(
                f"Generation completed successfully. Total images: {total_images}"
            )

        else:
            # Process in regular mode (simple list of prompts)
            prompts = json.loads(os.environ["PROMPTS"])
            num_images_per_prompt = int(os.environ.get("NUM_IMAGES_PER_PROMPT", "1"))

            logger.info(f"Running in standard mode with {len(prompts)} prompts")

            generate_images(
                pipeline=pipeline,
                prompts=prompts,
                output_dir=output_dir,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_images_per_prompt=num_images_per_prompt,
            )

            logger.info("Generation completed successfully")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
