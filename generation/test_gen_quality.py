"""
Current Optimizations include:
  - Using DPMSolverMultistepScheduler with trailing timestep spacing and Karras sigmas enabled to optimize the noise schedule, enhancing both speed and image quality.
  - Enabling attention slicing to reduce GPU memory usage.
  - Setting the UNet and VAE memory format to torch.channels_last for improved GPU efficiency.
  - Compiling the UNet and VAE decoder with torch.compile (mode="reduce-overhead") to reduce latency.
  - Enabling xformers memory efficient attention for additional speed improvements.

Potential Additional Optimizations for Further Quality and Efficiency Enhancements:
  - Switching to torch.compile(mode="max-autotune") for more aggressive optimization (at the expense of longer compilation time).
  - Applying dynamic quantization (e.g., via the torchao library) to accelerate linear operations while preserving output fidelity.
  - Distributed inference using Accelerate or PyTorch Distributed to process multiple prompts in parallel across GPUs.
  - Model sharding for very large diffusion models that do not fit on a single GPU.
  - Experimenting with custom noise schedules (e.g., AYS schedules) and rescaling the noise schedule to mitigate signal leakage and further boost output quality.
  - Combining attention block projection matrices (QKV fusion) for marginal speed improvements.
"""

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import os
import json
import logging
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Configure PyTorch inductor settings for optimal diffusion model performance
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

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


def generate_images(
    prompts: list,
    output_dir: str,
    model_path: str,
    num_steps: int = 20,
    guidance_scale: float = 4.5,
    batch_size: int = 1,
    save_metadata: bool = True,
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
    """
    try:
        start_time = time.time()
        logger.info(f"Loading base model and LoRA weights from {model_path}")

        # Assume CUDA is available
        device = "cuda"
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")

        # Set dtype to float16 for GPU for faster computation
        dtype = torch.float16
        pipeline = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=dtype,
            safety_checker=None,  # Disable safety checker for faster loading
        )
        if model_path:
            pipeline.load_lora_weights(".", weight_name=model_path)
        pipeline = pipeline.to(device)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Replace default scheduler with an optimized DPMSolverMultistepScheduler
        # Using trailing timestep spacing and enabling Karras sigmas to improve quality with fewer steps.
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            timestep_spacing="trailing",
            use_karras_sigmas=True,
        )

        # GPU-specific optimizations
        if device == "cuda":
            # Enable attention slicing to reduce memory usage
            pipeline.enable_attention_slicing(slice_size="auto")

            # Set memory format for faster inference
            pipeline.unet.to(memory_format=torch.channels_last)
            pipeline.vae.to(memory_format=torch.channels_last)

            # Enable xformers memory efficient attention if available
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")

            # Try compiling UNet and VAE decoder for lower latency
            # Try and consider also mode="max-autotune" for more aggressive optimizations.
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

        logger.info(
            f"Model loading completed in {time.time() - start_time:.2f} seconds"
        )

        # Process prompts in batches if batch_size > 1 and more than one prompt is provided
        if batch_size > 1 and len(prompts) > 1:
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]
                logger.info(
                    f"Processing batch {i // batch_size + 1}/{(len(prompts) - 1) // batch_size + 1}"
                )

                start_batch = time.time()
                batch_output = pipeline(
                    prompt=batch_prompts,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    height=1024,
                    width=1024,
                )
                batch_time = time.time() - start_batch

                # Save batch results concurrently
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for j, image in enumerate(batch_output.images):
                        prompt_idx = i + j
                        output_path = os.path.join(
                            output_dir, f"gen_{prompt_idx:03d}.png"
                        )
                        metadata = (
                            {
                                "prompt": batch_prompts[j],
                                "steps": num_steps,
                                "guidance_scale": guidance_scale,
                                "generation_time": batch_time / len(batch_prompts),
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                            if save_metadata
                            else None
                        )
                        futures.append(
                            executor.submit(save_image, image, output_path, metadata)
                        )
                    for future in futures:
                        saved_path = future.result()
                        logger.info(f"Saved image to: {saved_path}")
                logger.info(f"Batch processed in {batch_time:.2f} seconds")
        else:
            # Process prompts one by one
            for idx, prompt in enumerate(prompts):
                logger.info(
                    f"Generating image {idx + 1}/{len(prompts)} with prompt: {prompt}"
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
                if save_metadata:
                    metadata = {
                        "prompt": prompt,
                        "steps": num_steps,
                        "guidance_scale": guidance_scale,
                        "generation_time": gen_time,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    save_image(image, output_path, metadata)
                else:
                    image.save(output_path)
                logger.info(
                    f"Saved image to: {output_path} (generated in {gen_time:.2f}s)"
                )

        total_time = time.time() - start_time
        logger.info(f"Generation completed successfully in {total_time:.2f} seconds")
        logger.info(f"Average time per image: {total_time / len(prompts):.2f} seconds")

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

        # Run generation
        generate_images(
            prompts=prompts,
            output_dir=output_dir,
            model_path=model_path,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            batch_size=batch_size,
            save_metadata=save_metadata,
        )

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
