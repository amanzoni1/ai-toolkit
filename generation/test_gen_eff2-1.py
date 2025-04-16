import torch
from diffusers import FluxPipeline
from diffusers.hooks import apply_group_offloading
import os
import json
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_optimized_pipeline(model_path: str) -> FluxPipeline:
    """
    Load and optimize the Flux pipeline.

    This function loads the Flux model and any corresponding LoRA weights,
    applies GPU and memory optimizations such as group offloading and VAE slicing/tiling,
    and returns an optimized pipeline instance.

    Args:
        model_path (str): Path or identifier for the LoRA weight file.

    Returns:
        FluxPipeline: An optimized pipeline for generation.
    """
    try:
        logger.info(
            f"Loading Flux model with base weights from 'black-forest-labs/FLUX.1-dev'"
        )
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16
        )
        # Load LoRA weights (assumes weights are available at model_path)
        logger.info(f"Loading LoRA weights from {model_path}")
        pipe.load_lora_weights(".", weight_name=model_path)
    except Exception as e:
        logger.error(f"Failed to load Flux model or LoRA weights: {e}")
        raise

    # Check CUDA availability and move the model accordingly
    if torch.cuda.is_available():
        pipe.to("cuda")
        logger.info("Moved pipeline to GPU ('cuda')")
        # Apply group offloading to reduce GPU VRAM consumption
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

    # Enable VAE slicing and tiling to reduce memory footprint during VAE decoding/encoding.
    if hasattr(pipe, "vae"):
        try:
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            logger.info(
                "Enabled VAE slicing and tiling for additional memory efficiency."
            )
        except Exception as e:
            logger.warning(f"VAE slicing/tiling could not be enabled: {e}")

    return pipe


def generate_images(
    prompts: list,
    output_dir: str,
    model_path: str,
    num_steps: int = 40,
    guidance_scale: float = 4.5,
    height: int = 1024,
    width: int = 1024,
    num_images_per_prompt: int = 4,
):
    """
    Generate images using the Flux model with optimizations.

    Args:
        prompts (list): List of prompts to generate images from.
        output_dir (str): Directory to save generated images.
        model_path (str): Path or identifier for the LoRA weights.
        num_steps (int): Number of inference steps.
        guidance_scale (float): Guidance scale for image generation.
        height (int): Height in pixels for the generated image.
        width (int): Width in pixels for the generated image.
        num_images_per_prompt (int): Number of images to generate per prompt.
    """
    try:
        # Initialize the optimized Flux pipeline
        pipe = get_optimized_pipeline(model_path)
        os.makedirs(output_dir, exist_ok=True)
        for idx, prompt in enumerate(prompts):
            logger.info(
                f"Generating {num_images_per_prompt} image(s) for prompt {idx + 1}/{len(prompts)}: {prompt}"
            )
            result = pipe(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_images_per_prompt=num_images_per_prompt,
            )
            for img_idx, image in enumerate(result.images):
                output_path = os.path.join(
                    output_dir, f"gen_{idx:03d}_{img_idx:02d}.png"
                )
                image.save(output_path)
                logger.info(f"Saved image to: {output_path}")

        logger.info("Image generation completed successfully.")
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise


def main():
    try:
        # Check for required environment variables
        required_vars = ["PROMPTS", "OUTPUT_DIR", "MODEL_PATH"]
        missing_vars = [var for var in required_vars if var not in os.environ]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

        # Parse configuration from environment
        prompts = json.loads(os.environ["PROMPTS"])
        output_dir = os.environ["OUTPUT_DIR"]
        model_path = os.environ["MODEL_PATH"]
        num_steps = int(os.environ.get("NUM_STEPS", "40"))
        guidance_scale = float(os.environ.get("GUIDANCE_SCALE", "4.5"))
        height = int(os.environ.get("HEIGHT", "1024"))
        width = int(os.environ.get("WIDTH", "1024"))
        num_images_per_prompt = int(os.environ.get("NUM_IMAGES_PER_PROMPT", "4"))

        # Run image generation
        generate_images(
            prompts=prompts,
            output_dir=output_dir,
            model_path=model_path,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_images_per_prompt=num_images_per_prompt,
        )
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


# update diffureser
###############
# import torch
# from diffusers import FluxPipeline
# from diffusers.hooks import apply_group_offloading
# import os
# import json
# import logging
# import sys
# import time

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)


# def get_optimized_pipeline(model_path: str) -> FluxPipeline:
#     """
#     Load and optimize the Flux pipeline.

#     This function loads the Flux model and any corresponding LoRA weights,
#     applies GPU and memory optimizations such as group offloading and VAE slicing/tiling,
#     and returns an optimized pipeline instance.

#     Args:
#         model_path (str): Path or identifier for the LoRA weight file.

#     Returns:
#         FluxPipeline: An optimized pipeline for generation.
#     """
#     try:
#         logger.info(
#             "Loading Flux model with base weights from 'black-forest-labs/FLUX.1-dev'"
#         )
#         pipe = FluxPipeline.from_pretrained(
#             "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16
#         )
#         # Load LoRA weights (assumes weights are available at model_path)
#         logger.info(f"Loading LoRA weights from {model_path}")
#         pipe.load_lora_weights(".", weight_name=model_path)
#     except Exception as e:
#         logger.error(f"Failed to load Flux model or LoRA weights: {e}")
#         raise

#     if torch.cuda.is_available():
#         pipe.to("cuda")
#         logger.info("Moved pipeline to GPU ('cuda')")
#         offload_device = torch.device("cpu")
#         onload_device = torch.device("cuda")
#         use_stream = True
#         try:
#             if hasattr(pipe, "transformer"):
#                 apply_group_offloading(
#                     pipe.transformer,
#                     offload_type="leaf_level",
#                     offload_device=offload_device,
#                     onload_device=onload_device,
#                     use_stream=use_stream,
#                 )
#             if hasattr(pipe, "text_encoder"):
#                 apply_group_offloading(
#                     pipe.text_encoder,
#                     offload_type="leaf_level",
#                     offload_device=offload_device,
#                     onload_device=onload_device,
#                     use_stream=use_stream,
#                 )
#             if hasattr(pipe, "text_encoder_2"):
#                 apply_group_offloading(
#                     pipe.text_encoder_2,
#                     offload_type="leaf_level",
#                     offload_device=offload_device,
#                     onload_device=onload_device,
#                     use_stream=use_stream,
#                 )
#             if hasattr(pipe, "vae"):
#                 apply_group_offloading(
#                     pipe.vae,
#                     offload_type="leaf_level",
#                     offload_device=offload_device,
#                     onload_device=onload_device,
#                     use_stream=use_stream,
#                 )
#             logger.info("Applied group offloading for model components.")
#         except Exception as e:
#             logger.warning(
#                 f"Group offloading could not be applied to some components: {e}"
#             )
#     else:
#         logger.info("CUDA not available. Running on CPU; performance may be slow.")
#         pipe.to("cpu")

#     if hasattr(pipe, "vae"):
#         try:
#             pipe.vae.enable_slicing()
#             pipe.vae.enable_tiling()
#             logger.info(
#                 "Enabled VAE slicing and tiling for additional memory efficiency."
#             )
#         except Exception as e:
#             logger.warning(f"VAE slicing/tiling could not be enabled: {e}")

#     return pipe


# def generate_images(
#     prompts: list,
#     output_dir: str,
#     model_path: str,
#     num_steps: int = 40,
#     guidance_scale: float = 4.5,
#     height: int = 1024,
#     width: int = 1024,
#     num_images_per_prompt: int = 4,
# ):
#     """
#     Generate images using the Flux model with optimizations and logs inference time.

#     Args:
#         prompts (list): List of prompts to generate images from.
#         output_dir (str): Directory to save generated images.
#         model_path (str): Path or identifier for the LoRA weights.
#         num_steps (int): Number of inference steps.
#         guidance_scale (float): Guidance scale for image generation.
#         height (int): Height in pixels for the generated image.
#         width (int): Width in pixels for the generated image.
#         num_images_per_prompt (int): Number of images to generate per prompt.
#     """
#     try:
#         pipe = get_optimized_pipeline(model_path)
#         os.makedirs(output_dir, exist_ok=True)

#         for idx, prompt in enumerate(prompts):
#             logger.info(
#                 f"Generating {num_images_per_prompt} image(s) for prompt {idx + 1}/{len(prompts)}: {prompt}"
#             )
#             start_time = time.time()  # start timing
#             result = pipe(
#                 prompt=prompt,
#                 num_inference_steps=num_steps,
#                 guidance_scale=guidance_scale,
#                 height=height,
#                 width=width,
#                 num_images_per_prompt=num_images_per_prompt,
#             )
#             end_time = time.time()  # end timing
#             elapsed = end_time - start_time
#             logger.info(f"Inference time for prompt {idx + 1}: {elapsed:.2f} seconds")

#             for img_idx, image in enumerate(result.images):
#                 output_path = os.path.join(
#                     output_dir, f"gen_{idx:03d}_{img_idx:02d}.png"
#                 )
#                 image.save(output_path)
#                 logger.info(f"Saved image to: {output_path}")

#         logger.info("Image generation completed successfully.")
#     except Exception as e:
#         logger.error(f"Image generation failed: {e}")
#         raise


# def main():
#     try:
#         # Check for required environment variables
#         required_vars = ["PROMPTS", "OUTPUT_DIR", "MODEL_PATH"]
#         missing_vars = [var for var in required_vars if var not in os.environ]
#         if missing_vars:
#             raise ValueError(f"Missing required environment variables: {missing_vars}")

#         # Parse configuration from environment
#         prompts = json.loads(os.environ["PROMPTS"])
#         output_dir = os.environ["OUTPUT_DIR"]
#         model_path = os.environ["MODEL_PATH"]
#         num_steps = int(os.environ.get("NUM_STEPS", "40"))
#         guidance_scale = float(os.environ.get("GUIDANCE_SCALE", "4.5"))
#         height = int(os.environ.get("HEIGHT", "1024"))
#         width = int(os.environ.get("WIDTH", "1024"))
#         num_images_per_prompt = int(os.environ.get("NUM_IMAGES_PER_PROMPT", "4"))

#         generate_images(
#             prompts=prompts,
#             output_dir=output_dir,
#             model_path=model_path,
#             num_steps=num_steps,
#             guidance_scale=guidance_scale,
#             height=height,
#             width=width,
#             num_images_per_prompt=num_images_per_prompt,
#         )
#     except Exception as e:
#         logger.error(f"Error in main: {e}")
#         sys.exit(1)


# if __name__ == "__main__":
#     main()


##################################
# SECOND
##################################


# import torch._dynamo

# torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.ignore_logger_methods = {"debug", "info", "warning", "error"}

# import torch
# from diffusers import FluxPipeline
# from diffusers.hooks import apply_group_offloading
# import os
# import json
# import logging
# import sys
# import time

# # Enable TF32 for faster matrix multiplications on Ampere+ GPUs
# torch.backends.cuda.matmul.allow_tf32 = True

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

# # Disable logger methods to avoid issues in compiled graphs (if any)
# logger.debug = lambda *args, **kwargs: None
# logger.info = lambda *args, **kwargs: None
# logger.warning = lambda *args, **kwargs: None
# logger.error = lambda *args, **kwargs: None


# def get_optimized_pipeline(model_path: str) -> FluxPipeline:
#     """
#     Load and optimize the Flux pipeline for inference speed.

#     This function loads the Flux model with FP16 precision, applies group offloading for the
#     transformer and text encoders, and enables xFormers memory-efficient attention (if installed).
#     The torch.compile step for the transformer has been commented out.

#     Args:
#         model_path (str): Path or identifier for the LoRA weight file.

#     Returns:
#         FluxPipeline: An optimized Flux pipeline for generation.
#     """
#     try:
#         logger.info(
#             "Loading Flux model with base weights from 'black-forest-labs/FLUX.1-dev'"
#         )
#         pipe = FluxPipeline.from_pretrained(
#             "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16
#         )
#         logger.info(f"Loading LoRA weights from {model_path}")
#         pipe.load_lora_weights(".", weight_name=model_path)
#     except Exception as e:
#         logger.error(f"Failed to load Flux model or LoRA weights: {e}")
#         raise

#     if torch.cuda.is_available():
#         pipe.to("cuda")
#         logger.info("Moved pipeline to GPU ('cuda')")
#         offload_device = torch.device("cpu")
#         onload_device = torch.device("cuda")
#         use_stream = True
#         try:
#             if hasattr(pipe, "transformer"):
#                 apply_group_offloading(
#                     pipe.transformer,
#                     offload_type="leaf_level",
#                     offload_device=offload_device,
#                     onload_device=onload_device,
#                     use_stream=use_stream,
#                 )
#                 # Commenting out torch.compile for transformer:
#                 # try:
#                 #     pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
#                 #     logger.info("Compiled transformer with torch.compile")
#                 # except Exception as e:
#                 #     logger.warning(f"torch.compile on transformer failed: {e}")
#             if hasattr(pipe, "text_encoder"):
#                 apply_group_offloading(
#                     pipe.text_encoder,
#                     offload_type="leaf_level",
#                     offload_device=offload_device,
#                     onload_device=onload_device,
#                     use_stream=use_stream,
#                 )
#             if hasattr(pipe, "text_encoder_2"):
#                 apply_group_offloading(
#                     pipe.text_encoder_2,
#                     offload_type="leaf_level",
#                     offload_device=offload_device,
#                     onload_device=onload_device,
#                     use_stream=use_stream,
#                 )
#             if hasattr(pipe, "vae"):
#                 apply_group_offloading(
#                     pipe.vae,
#                     offload_type="leaf_level",
#                     offload_device=offload_device,
#                     onload_device=onload_device,
#                     use_stream=use_stream,
#                 )
#             logger.info("Applied group offloading for model components.")
#         except Exception as e:
#             logger.warning(
#                 f"Group offloading could not be applied to some components: {e}"
#             )
#     else:
#         logger.info("CUDA not available. Running on CPU; performance may be slow.")
#         pipe.to("cpu")

#     try:
#         pipe.enable_xformers_memory_efficient_attention()
#         logger.info("Enabled xFormers memory efficient attention.")
#     except Exception as e:
#         logger.warning(f"xFormers could not be enabled: {e}")

#     return pipe


# def generate_images(
#     prompts: list,
#     output_dir: str,
#     model_path: str,
#     num_steps: int = 40,
#     guidance_scale: float = 4.5,
#     height: int = 1024,
#     width: int = 1024,
#     num_images_per_prompt: int = 4,
# ):
#     """
#     Generate images using the optimized Flux pipeline and log the inference time.

#     Args:
#         prompts (list): Prompts to generate images from.
#         output_dir (str): Directory to save generated images.
#         model_path (str): Path or identifier for the LoRA weight file.
#         num_steps (int): Number of inference steps.
#         guidance_scale (float): Guidance scale for image generation.
#         height (int): Height of the generated image.
#         width (int): Width of the generated image.
#         num_images_per_prompt (int): Number of images per prompt.
#     """
#     try:
#         pipe = get_optimized_pipeline(model_path)
#         os.makedirs(output_dir, exist_ok=True)
#         for idx, prompt in enumerate(prompts):
#             logger.info(
#                 f"Generating {num_images_per_prompt} image(s) for prompt {idx + 1}/{len(prompts)}: {prompt}"
#             )
#             start_time = time.time()
#             result = pipe(
#                 prompt=prompt,
#                 num_inference_steps=num_steps,
#                 guidance_scale=guidance_scale,
#                 height=height,
#                 width=width,
#                 num_images_per_prompt=num_images_per_prompt,
#             )
#             end_time = time.time()
#             elapsed = end_time - start_time
#             logger.info(f"Inference time for prompt {idx + 1}: {elapsed:.2f} seconds")
#             for img_idx, image in enumerate(result.images):
#                 output_path = os.path.join(
#                     output_dir, f"gen_{idx:03d}_{img_idx:02d}.png"
#                 )
#                 image.save(output_path)
#                 logger.info(f"Saved image to: {output_path}")
#         logger.info("Image generation completed successfully.")
#     except Exception as e:
#         logger.error(f"Image generation failed: {e}")
#         raise


# def main():
#     try:
#         required_vars = ["PROMPTS", "OUTPUT_DIR", "MODEL_PATH"]
#         missing_vars = [var for var in required_vars if var not in os.environ]
#         if missing_vars:
#             raise ValueError(f"Missing required environment variables: {missing_vars}")
#         prompts = json.loads(os.environ["PROMPTS"])
#         output_dir = os.environ["OUTPUT_DIR"]
#         model_path = os.environ["MODEL_PATH"]
#         num_steps = int(os.environ.get("NUM_STEPS", "40"))
#         guidance_scale = float(os.environ.get("GUIDANCE_SCALE", "4.5"))
#         height = int(os.environ.get("HEIGHT", "1024"))
#         width = int(os.environ.get("WIDTH", "1024"))
#         num_images_per_prompt = int(os.environ.get("NUM_IMAGES_PER_PROMPT", "4"))
#         generate_images(
#             prompts=prompts,
#             output_dir=output_dir,
#             model_path=model_path,
#             num_steps=num_steps,
#             guidance_scale=guidance_scale,
#             height=height,
#             width=width,
#             num_images_per_prompt=num_images_per_prompt,
#         )
#     except Exception as e:
#         logger.error(f"Error in main: {e}")
#         sys.exit(1)


# if __name__ == "__main__":
#     main()
