import torch
from diffusers import FluxPipeline
from diffusers.hooks import apply_group_offloading
import os
import json
import logging
import sys
import time


# Configure logging normally
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Enable TF32 for faster matrix multiplications on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True


def get_optimized_pipeline(model_path: str) -> FluxPipeline:
    """
    Load and optimize the Flux pipeline for inference speed.

    This function loads the Flux model with FP16 precision, applies group offloading for the
    transformer and text encoders, and keeps the original scheduler that comes with Flux.

    Args:
        model_path (str): Path or identifier for the LoRA weight file.

    Returns:
        FluxPipeline: An optimized Flux pipeline for generation.
    """
    try:
        logger.info(
            "Loading Flux model with base weights from 'black-forest-labs/FLUX.1-dev'"
        )
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16
        )
        logger.info(f"Loading LoRA weights from {model_path}")
        pipe.load_lora_weights(".", weight_name=model_path)

        # Keep the original scheduler but disable dynamic shifting which requires mu
        logger.info(f"Using original scheduler: {pipe.scheduler.__class__.__name__}")

        # Modify scheduler config to disable dynamic shifting
        if hasattr(pipe.scheduler, "config"):
            # Store original config values for reference
            original_use_dynamic_shifting = pipe.scheduler.config.use_dynamic_shifting
            original_shift = pipe.scheduler.config.shift

            # Disable dynamic shifting to avoid requiring mu parameter
            pipe.scheduler.config.use_dynamic_shifting = False

            logger.info(
                f"Original scheduler config: use_dynamic_shifting={original_use_dynamic_shifting}, shift={original_shift}"
            )
            logger.info(
                f"Modified scheduler config: use_dynamic_shifting={pipe.scheduler.config.use_dynamic_shifting}, shift={pipe.scheduler.config.shift}"
            )
    except Exception as e:
        logger.error(f"Failed to load Flux model or LoRA weights: {e}")
        raise

    if torch.cuda.is_available():
        pipe.to("cuda")
        logger.info("Moved pipeline to GPU ('cuda')")
        offload_device = torch.device("cpu")
        onload_device = torch.device("cuda")
        use_stream = True
        try:
            if hasattr(pipe, "transformer"):
                apply_group_offloading(
                    pipe.transformer,
                    offload_type="leaf_level",
                    offload_device=offload_device,
                    onload_device=onload_device,
                    use_stream=use_stream,
                )
            if hasattr(pipe, "text_encoder"):
                apply_group_offloading(
                    pipe.text_encoder,
                    offload_type="leaf_level",
                    offload_device=offload_device,
                    onload_device=onload_device,
                    use_stream=use_stream,
                )
            if hasattr(pipe, "text_encoder_2"):
                apply_group_offloading(
                    pipe.text_encoder_2,
                    offload_type="leaf_level",
                    offload_device=offload_device,
                    onload_device=onload_device,
                    use_stream=use_stream,
                )
            if hasattr(pipe, "vae"):
                apply_group_offloading(
                    pipe.vae,
                    offload_type="leaf_level",
                    offload_device=offload_device,
                    onload_device=onload_device,
                    use_stream=use_stream,
                )
            logger.info("Applied group offloading for model components.")
        except Exception as e:
            logger.warning(
                f"Group offloading could not be applied to some components: {e}"
            )
    else:
        logger.info("CUDA not available. Running on CPU; performance may be slow.")
        pipe.to("cpu")

    if hasattr(pipe, "vae"):
        try:
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            logger.info(
                "Enabled VAE slicing and tiling for additional memory efficiency."
            )
        except Exception as e:
            logger.warning(f"VAE slicing/tiling could not be enabled: {e}")

    try:
        pipe.enable_xformers_memory_efficient_attention()
        logger.info("Enabled xFormers memory efficient attention.")
    except Exception as e:
        logger.warning(f"xFormers could not be enabled: {e}")

    return pipe


def generate_images(
    prompts: list,
    output_dir: str,
    model_path: str,
    num_steps: int = 20,
    guidance_scale: float = 4.5,
    height: int = 1024,
    width: int = 1024,
    num_images_per_prompt: int = 4,
):
    """
    Generate images using the Flux pipeline and log inference time.

    Args:
        prompts (list): List of prompts to generate images from.
        output_dir (str): Directory to save generated images.
        model_path (str): Path or identifier for the LoRA weight file.
        num_steps (int): Number of inference steps (e.g., 20 for faster inference).
        guidance_scale (float): Guidance scale for image generation.
        height (int): Height (in pixels) of the generated image.
        width (int): Width (in pixels) of the generated image.
        num_images_per_prompt (int): Number of images to generate per prompt.
    """
    try:
        pipe = get_optimized_pipeline(model_path)
        os.makedirs(output_dir, exist_ok=True)

        # Create a generator for reproducibility
        generator = None
        if torch.cuda.is_available():
            generator = torch.Generator(device="cuda").manual_seed(42)

        # Try to monkeypatch the retrieve_timesteps function in the pipeline_flux module
        from diffusers.pipelines.flux import pipeline_flux

        # Store original function
        original_retrieve_timesteps = pipeline_flux.retrieve_timesteps

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

        # Now generate images
        for idx, prompt in enumerate(prompts):
            logger.info(
                f"Generating {num_images_per_prompt} image(s) for prompt {idx+1}/{len(prompts)}: {prompt}"
            )
            start_time = time.time()

            try:
                # Set timesteps before calling the pipe
                pipe.scheduler.set_timesteps(
                    num_inference_steps=num_steps, device=pipe.device
                )

                # Now call the pipeline
                result = pipe(
                    prompt=prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                )

            except Exception as e:
                logger.error(f"Error in generation: {e}")

                # Try with pure manual implementation
                logger.info("Falling back to manual implementation...")

                try:
                    # Manual implementation of key steps
                    # 1. Encode prompt
                    text_inputs = pipe.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=pipe.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_inputs = text_inputs.to(pipe.device)
                    text_embeddings = pipe.text_encoder(text_inputs.input_ids)[0]

                    # 2. Create unconditional embedding
                    max_length = text_inputs.input_ids.shape[-1]
                    uncond_input = pipe.tokenizer(
                        [""] * num_images_per_prompt,
                        padding="max_length",
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    uncond_input = uncond_input.to(pipe.device)
                    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids)[0]

                    # 3. Concatenate for classifier-free guidance
                    prompt_embeds = torch.cat([uncond_embeddings, text_embeddings])

                    # 4. Generate initial latents
                    latents_shape = (
                        num_images_per_prompt,
                        pipe.vae.config.latent_channels,
                        height // 8,
                        width // 8,
                    )
                    latents = torch.randn(
                        latents_shape,
                        generator=generator,
                        device=pipe.device,
                        dtype=prompt_embeds.dtype,
                    )

                    # 5. Set timesteps
                    pipe.scheduler.set_timesteps(
                        num_inference_steps=num_steps, device=pipe.device
                    )
                    timesteps = pipe.scheduler.timesteps

                    # 6. Denoising loop
                    for i, t in enumerate(timesteps):
                        # Duplicate latents for guidance
                        latent_model_input = (
                            torch.cat([latents] * 2)
                            if guidance_scale > 1.0
                            else latents
                        )

                        # Get transformer prediction
                        noise_pred = pipe.transformer(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                        ).sample

                        # Apply guidance
                        if guidance_scale > 1.0:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            )

                        # Step
                        latents = pipe.scheduler.step(
                            noise_pred, t, latents
                        ).prev_sample

                    # 7. Decode latents
                    latents = 1 / pipe.vae.config.scaling_factor * latents
                    images = pipe.vae.decode(latents).sample
                    images = (images / 2 + 0.5).clamp(0, 1)
                    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
                    images = (images * 255).round().astype("uint8")

                    # 8. Convert to PIL
                    from PIL import Image

                    pil_images = [Image.fromarray(img) for img in images]

                    # Create result object
                    class DummyPipelineOutput:
                        def __init__(self, images):
                            self.images = images

                    result = DummyPipelineOutput(pil_images)

                except Exception as nested_e:
                    logger.error(f"Manual implementation also failed: {nested_e}")
                    raise

            # Restore the original function
            pipeline_flux.retrieve_timesteps = original_retrieve_timesteps

            end_time = time.time()
            elapsed = end_time - start_time
            logger.info(f"Inference time for prompt {idx+1}: {elapsed:.2f} seconds")

            # Save the generated images
            for img_idx, image in enumerate(result.images):
                output_path = os.path.join(
                    output_dir, f"gen_{idx:03d}_{img_idx:02d}.png"
                )
                image.save(output_path)
                logger.info(f"Saved image to: {output_path}")

        logger.info("Image generation completed successfully.")
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        # Print the full traceback for better debugging
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


def main():
    try:
        # Check required environment variables
        required_vars = ["PROMPTS", "OUTPUT_DIR", "MODEL_PATH"]
        missing_vars = [var for var in required_vars if var not in os.environ]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        prompts = json.loads(os.environ["PROMPTS"])
        output_dir = os.environ["OUTPUT_DIR"]
        model_path = os.environ["MODEL_PATH"]
        num_steps = int(os.environ.get("NUM_STEPS", "20"))
        guidance_scale = float(os.environ.get("GUIDANCE_SCALE", "4.5"))
        height = int(os.environ.get("HEIGHT", "1024"))
        width = int(os.environ.get("WIDTH", "1024"))
        num_images_per_prompt = int(os.environ.get("NUM_IMAGES_PER_PROMPT", "4"))

        generate_images(
            prompts,
            output_dir,
            model_path,
            num_steps,
            guidance_scale,
            height,
            width,
            num_images_per_prompt,
        )
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
