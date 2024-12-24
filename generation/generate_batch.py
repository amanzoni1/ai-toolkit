# ai-toolkit/generation/generate_batch.py
import torch
from diffusers import DiffusionPipeline
import os
import json
import logging
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_images(prompts: list, output_dir: str, model_path: str, 
                   num_steps: int = 20, guidance_scale: float = 4.0):
    """Generate images using the trained model.
    
    Args:
        prompts (list): List of prompts to generate images from
        output_dir (str): Directory to save generated images
        model_path (str): Path to the trained model weights
        num_steps (int): Number of inference steps
        guidance_scale (float): Guidance scale for generation
    """
    try:
        logger.info(f"Loading base model and LoRA weights from {model_path}")
        pipeline = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16
        )
        pipeline.load_lora_weights(".", weight_name=model_path)
        pipeline = pipeline.to("cuda")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, prompt in enumerate(prompts):
            logger.info(f"Generating image {idx+1}/{len(prompts)}")
            logger.info(f"Prompt: {prompt}")
            
            image = pipeline(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512
            ).images[0]
            
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
        required_vars = ['PROMPTS', 'OUTPUT_DIR', 'MODEL_PATH']
        missing_vars = [var for var in required_vars if var not in os.environ]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
            
        prompts = json.loads(os.environ['PROMPTS'])
        output_dir = os.environ['OUTPUT_DIR']
        model_path = os.environ['MODEL_PATH']
        
        # Optional parameters
        num_steps = int(os.environ.get('NUM_STEPS', '20'))
        guidance_scale = float(os.environ.get('GUIDANCE_SCALE', '4.0'))
        
        # Run generation
        generate_images(
            prompts=prompts,
            output_dir=output_dir,
            model_path=model_path,
            num_steps=num_steps,
            guidance_scale=guidance_scale
        )
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

