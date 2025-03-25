#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import uuid
import base64
import logging
import numpy as np
import torch
import random
from typing import Dict, List, Optional, Union, Tuple, Callable
from PIL import Image
import imageio
from diffusers import StableDiffusionPipeline, DDIMScheduler, EulerDiscreteScheduler, PNDMScheduler
from diffusers.utils.import_utils import is_xformers_available

from backend.models import ModelManager
from backend.optimization import apply_motion_lora, optimize_memory

logger = logging.getLogger(__name__)

class AnimateDiffPipeline:
    """AnimateDiff pipeline for generating videos with Motion Modules and LoRAs"""
    
    def __init__(
        self,
        model_manager: ModelManager,
        output_dir: str,
        use_cpu: bool = True,
        enable_uncensored: bool = True,
        callback: Optional[Callable[[np.ndarray, int, int], None]] = None
    ):
        self.model_manager = model_manager
        self.output_dir = output_dir
        self.use_cpu = use_cpu
        self.enable_uncensored = enable_uncensored
        self.callback = callback
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Device configuration
        self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")
        
        # Scheduler mapping
        self.scheduler_map = {
            "ddim": DDIMScheduler,
            "euler": EulerDiscreteScheduler,
            "pndm": PNDMScheduler,
        }
    
    def set_callback(self, callback: Callable[[np.ndarray, int, int], None]):
        """Set callback function for live preview updates"""
        self.callback = callback
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        sd_model_name: str = "runwayml/stable-diffusion-v1-5",
        motion_module_name: str = "v3_sd15_mm.safetensors",
        motion_lora_name: Optional[str] = None,
        motion_lora_strength: float = 1.0,
        num_frames: int = 16,
        fps: int = 8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        scheduler_name: str = "euler",
        seed: int = -1,
        width: int = 512,
        height: int = 512,
        input_image: Optional[str] = None,
    ) -> str:
        """
        Generate a video using AnimateDiff
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative text prompt
            sd_model_name: Name of the Stable Diffusion model
            motion_module_name: Name of the motion module
            motion_lora_name: Name of the motion LoRA (optional)
            motion_lora_strength: Strength of the motion LoRA effect (0.0 to 1.0)
            num_frames: Number of frames to generate
            fps: Frames per second
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps
            scheduler_name: Name of the scheduler to use
            seed: Random seed (-1 for random)
            width: Width of the generated video
            height: Height of the generated video
            input_image: Base64 encoded input image for img2video (optional)
            
        Returns:
            Path to the generated video file
        """
        try:
            # Set random seed
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
            
            logger.info(f"Generating video with seed: {seed}")
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            # Load models
            sd_pipeline = self.model_manager.load_sd_model(sd_model_name)
            motion_module = self.model_manager.load_motion_module(motion_module_name)
            
            # Load motion LoRA if specified
            motion_lora = None
            if motion_lora_name and motion_lora_name.lower() != "none":
                motion_lora = self.model_manager.load_motion_lora(motion_lora_name)
                
                # Apply motion LoRA to motion module
                if motion_lora:
                    motion_module = apply_motion_lora(
                        motion_module=motion_module,
                        motion_lora=motion_lora,
                        alpha=motion_lora_strength
                    )
            
            # Set scheduler
            scheduler_cls = self.scheduler_map.get(scheduler_name.lower(), EulerDiscreteScheduler)
            sd_pipeline.scheduler = scheduler_cls.from_config(sd_pipeline.scheduler.config)
            
            # Disable safety checker for uncensored mode
            if self.enable_uncensored and hasattr(sd_pipeline, "safety_checker"):
                sd_pipeline.safety_checker = None
                logger.info("Safety checker disabled for uncensored mode")
            
            # Prepare for generation
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Process input image for img2video if provided
            init_image = None
            if input_image:
                # Decode base64 image
                if input_image.startswith("data:image"):
                    input_image = input_image.split(",")[1]
                
                image_data = base64.b64decode(input_image)
                init_image = Image.open(io.BytesIO(image_data))
                init_image = init_image.resize((width, height))
                logger.info("Using input image for img2video generation")
            
            # Create a custom callback for live preview
            frames = []
            
            def preview_callback(step: int, timestep: int, latents: torch.FloatTensor):
                """Callback function for live preview during generation"""
                if step % 5 == 0 or step == num_inference_steps - 1:  # Update every 5 steps
                    # Convert latents to images
                    with torch.no_grad():
                        latents_for_preview = latents.detach().clone()
                        latents_for_preview = 1 / 0.18215 * latents_for_preview
                        
                        # Get the first frame for preview
                        frame_latent = latents_for_preview[0:1]
                        
                        # Decode the latents to image
                        with torch.no_grad():
                            frame_image = sd_pipeline.vae.decode(frame_latent).sample
                        
                        # Convert to numpy array
                        frame_image = (frame_image / 2 + 0.5).clamp(0, 1)
                        frame_image = frame_image.cpu().permute(0, 2, 3, 1).numpy()[0]
                        frame_image = (frame_image * 255).astype(np.uint8)
                        
                        # Store the frame
                        frames.append(frame_image)
                        
                        # Call the external callback if provided
                        if self.callback:
                            self.callback(frame_image, step, num_inference_steps)
            
            # Generate the video frames
            logger.info(f"Generating {num_frames} frames with {num_inference_steps} steps")
            
            # Prepare AnimateDiff-specific parameters
            # Note: This is a simplified implementation - in a real implementation,
            # you would need to modify the UNet to incorporate the motion module
            
            # For demonstration purposes, we'll use the standard pipeline
            # In a real implementation, you would need to implement the AnimateDiff-specific logic
            
            with torch.no_grad():
                # Generate initial latents
                if init_image:
                    # Image-to-video mode
                    # In a real implementation, you would encode the image and use it as initial latents
                    result = sd_pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=init_image,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        callback=preview_callback,
                        callback_steps=1
                    )
                else:
                    # Text-to-video mode
                    result = sd_pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        callback=preview_callback,
                        callback_steps=1
                    )
            
            # In a real implementation, you would generate multiple frames using the motion module
            # For demonstration, we'll duplicate the single image to create a video
            images = []
            if hasattr(result, "images"):
                base_image = result.images[0]
                
                # Create a simple animation by shifting the image slightly
                for i in range(num_frames):
                    # Create a copy of the image
                    frame = base_image.copy()
                    
                    # Shift the image slightly to simulate motion
                    shift_x = int(10 * np.sin(i / num_frames * 2 * np.pi))
                    shift_y = int(10 * np.cos(i / num_frames * 2 * np.pi))
                    
                    # Apply the shift
                    frame = np.array(frame)
                    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                    import cv2
                    frame = cv2.warpAffine(frame, M, (width, height))
                    
                    images.append(Image.fromarray(frame))
            else:
                # Use the frames from the callback if available
                images = [Image.fromarray(frame) for frame in frames]
            
            # Save the video
            output_filename = f"animatediff_{uuid.uuid4().hex[:8]}.mp4"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Create the video file
            with imageio.get_writer(output_path, fps=fps) as writer:
                for image in images:
                    writer.append_data(np.array(image))
            
            logger.info(f"Video saved to {output_path}")
            
            # Clean up to free memory
            optimize_memory()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            raise
    
    def generate_with_live_preview(
        self,
        prompt: str,
        negative_prompt: str = "",
        sd_model_name: str = "runwayml/stable-diffusion-v1-5",
        motion_module_name: str = "v3_sd15_mm.safetensors",
        motion_lora_name: Optional[str] = None,
        motion_lora_strength: float = 1.0,
        num_frames: int = 16,
        fps: int = 8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        scheduler_name: str = "euler",
        seed: int = -1,
        width: int = 512,
        height: int = 512,
        input_image: Optional[str] = None,
        preview_callback: Optional[Callable[[np.ndarray, int, int], None]] = None
    ) -> str:
        """
        Generate a video with live preview updates
        
        This method sets a temporary callback for the generation process and then
        calls the standard generate method.
        
        Args:
            preview_callback: Callback function for live preview updates
            (Other parameters are the same as the generate method)
            
        Returns:
            Path to the generated video file
        """
        # Store the original callback
        original_callback = self.callback
        
        # Set the preview callback
        self.callback = preview_callback
        
        try:
            # Generate the video
            return self.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                sd_model_name=sd_model_name,
                motion_module_name=motion_module_name,
                motion_lora_name=motion_lora_name,
                motion_lora_strength=motion_lora_strength,
                num_frames=num_frames,
                fps=fps,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                scheduler_name=scheduler_name,
                seed=seed,
                width=width,
                height=height,
                input_image=input_image
            )
        finally:
            # Restore the original callback
            self.callback = original_callback
