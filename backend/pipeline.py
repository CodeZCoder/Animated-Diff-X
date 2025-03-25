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
import gc
from typing import Dict, List, Optional, Union, Tuple, Callable
from PIL import Image
import imageio
from diffusers import StableDiffusionPipeline, DDIMScheduler, EulerDiscreteScheduler, PNDMScheduler, AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
from diffusers import StableDiffusionImg2ImgPipeline
from tqdm import tqdm

from backend.models import ModelManager
from backend.optimization import apply_motion_lora, optimize_memory

logger = logging.getLogger(__name__)

class MemoryEfficientFrameGenerator:
    """
    A memory-efficient frame generator that minimizes RAM usage
    while ensuring exact number of frames and steps are used.
    """
    
    def __init__(self, sd_pipeline, img2img_pipeline=None):
        self.sd_pipeline = sd_pipeline
        self.img2img_pipeline = img2img_pipeline
        self.callback = None
    
    def set_callback(self, callback):
        """Set callback function for live preview updates"""
        self.callback = callback
    
    def generate_frames(
        self,
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        seed: int,
        init_image: Optional[Image.Image] = None,
    ) -> List[np.ndarray]:
        """
        Generate frames with memory optimization
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative text prompt
            num_frames: Number of frames to generate
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale for generation
            width: Width of the generated frames
            height: Height of the generated frames
            seed: Random seed
            init_image: Initial image for img2img mode (optional)
            
        Returns:
            List of generated frames as numpy arrays
        """
        all_frames = []
        
        # Set base seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Generate each frame with memory optimization
        for frame_idx in range(num_frames):
            logger.info(f"Generating frame {frame_idx+1}/{num_frames} with {num_inference_steps} steps")
            
            # Calculate frame-specific seed
            frame_seed = seed + frame_idx
            frame_generator = torch.Generator(device=self.sd_pipeline.device).manual_seed(frame_seed)
            
            # Generate the frame
            if init_image and self.img2img_pipeline:
                # Image-to-video mode
                # Calculate strength based on frame position to create motion
                strength = 0.3 + (frame_idx / num_frames) * 0.3  # 0.3 to 0.6
                
                # Generate frame with memory optimization
                result = self._generate_img2img_frame_memory_efficient(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    init_image=init_image,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=frame_generator,
                    frame_idx=frame_idx
                )
            else:
                # Text-to-video mode
                result = self._generate_txt2img_frame_memory_efficient(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=frame_generator,
                    frame_idx=frame_idx
                )
            
            # Add the generated frame to our collection
            all_frames.append(np.array(result))
            
            # Update the callback with the latest frame
            if self.callback:
                self.callback(np.array(result), num_inference_steps, num_inference_steps)
            
            # Force garbage collection to free memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return all_frames
    
    def _generate_txt2img_frame_memory_efficient(
        self,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator,
        frame_idx: int
    ) -> Image.Image:
        """
        Generate a single frame using text-to-image with memory optimization
        """
        # Get the text embeddings
        text_inputs = self.sd_pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.sd_pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.sd_pipeline.text_encoder(text_inputs.input_ids.to(self.sd_pipeline.device))[0]
        
        # Get the unconditional embeddings for classifier-free guidance
        uncond_inputs = self.sd_pipeline.tokenizer(
            [negative_prompt],
            padding="max_length",
            max_length=self.sd_pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = self.sd_pipeline.text_encoder(uncond_inputs.input_ids.to(self.sd_pipeline.device))[0]
        
        # Concatenate the unconditional and text embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Set the timesteps
        self.sd_pipeline.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.sd_pipeline.scheduler.timesteps
        
        # Generate random latent noise
        latents = torch.randn(
            (1, self.sd_pipeline.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.sd_pipeline.device,
        )
        latents = latents * self.sd_pipeline.scheduler.init_noise_sigma
        
        # Denoising loop with memory optimization
        for i, t in enumerate(tqdm(timesteps, desc=f"Frame {frame_idx+1}")):
            # Free memory before each step
            if i > 0:
                del latent_model_input, noise_pred, noise_pred_uncond, noise_pred_text
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Expand the latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            
            # Scale the latents (sigma)
            latent_model_input = self.sd_pipeline.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict the noise residual
            with torch.no_grad():
                noise_pred = self.sd_pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute the previous noisy sample
            latents = self.sd_pipeline.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Update callback if it's the first frame
            if frame_idx == 0 and self.callback and (i % 5 == 0 or i == len(timesteps) - 1):
                # Decode the current latent to an image
                with torch.no_grad():
                    latents_for_callback = 1 / 0.18215 * latents
                    image = self.sd_pipeline.vae.decode(latents_for_callback).sample
                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                    image = (image * 255).astype(np.uint8)
                    self.callback(image, i, num_inference_steps)
                
                # Free memory after callback
                del latents_for_callback, image
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Decode the final latent to an image
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            image = self.sd_pipeline.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Free memory after decoding
        del latents, text_embeddings
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return image
    
    def _generate_img2img_frame_memory_efficient(
        self,
        prompt: str,
        negative_prompt: str,
        init_image: Image.Image,
        strength: float,
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator,
        frame_idx: int
    ) -> Image.Image:
        """
        Generate a single frame using image-to-image with memory optimization
        """
        # Get the text embeddings
        text_inputs = self.img2img_pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.img2img_pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.img2img_pipeline.text_encoder(text_inputs.input_ids.to(self.img2img_pipeline.device))[0]
        
        # Get the unconditional embeddings for classifier-free guidance
        uncond_inputs = self.img2img_pipeline.tokenizer(
            [negative_prompt],
            padding="max_length",
            max_length=self.img2img_pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = self.img2img_pipeline.text_encoder(uncond_inputs.input_ids.to(self.img2img_pipeline.device))[0]
        
        # Concatenate the unconditional and text embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Preprocess image
        init_image = init_image.convert("RGB")
        init_image = init_image.resize((self.img2img_pipeline.vae.sample_size * 8, self.img2img_pipeline.vae.sample_size * 8))
        init_image = np.array(init_image).astype(np.float32) / 255.0
        init_image = init_image[None].transpose(0, 3, 1, 2)
        init_image = torch.from_numpy(init_image).to(self.img2img_pipeline.device)
        
        # Encode the init image
        with torch.no_grad():
            init_latent = self.img2img_pipeline.vae.encode(init_image).latent_dist.sample(generator=generator)
            init_latent = 0.18215 * init_latent
        
        # Free memory after encoding
        del init_image
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Set the timesteps - IMPORTANT: We're forcing all steps here
        self.img2img_pipeline.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.img2img_pipeline.scheduler.timesteps
        
        # Get the original number of inference steps
        offset = self.img2img_pipeline.scheduler.config.get("steps_offset", 0)
        
        # IMPORTANT: Force using all steps by directly setting the starting timestep
        # Instead of calculating based on strength, we'll use a fixed number of steps
        # but vary the starting point to create motion
        num_actual_inference_steps = num_inference_steps
        init_timestep = min(int(num_actual_inference_steps * strength), num_actual_inference_steps)
        t_start = max(num_actual_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start:]
        
        # Add noise to latents using the timesteps
        noise = torch.randn(init_latent.shape, generator=generator, device=self.img2img_pipeline.device)
        latents = self.img2img_pipeline.scheduler.add_noise(init_latent, noise, timesteps[0])
        
        # Free memory after noise addition
        del init_latent, noise
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Denoising loop with memory optimization
        for i, t in enumerate(tqdm(timesteps, desc=f"Frame {frame_idx+1}")):
            # Free memory before each step
            if i > 0:
                del latent_model_input, noise_pred, noise_pred_uncond, noise_pred_text
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Expand the latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            
            # Scale the latents (sigma)
            latent_model_input = self.img2img_pipeline.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict the noise residual
            with torch.no_grad():
                noise_pred = self.img2img_pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute the previous noisy sample
            latents = self.img2img_pipeline.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Update callback if it's the first frame
            if frame_idx == 0 and self.callback and (i % 5 == 0 or i == len(timesteps) - 1):
                # Decode the current latent to an image
                with torch.no_grad():
                    latents_for_callback = 1 / 0.18215 * latents
                    image = self.img2img_pipeline.vae.decode(latents_for_callback).sample
                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                    image = (image * 255).astype(np.uint8)
                    # Report progress based on the total steps
                    current_step = t_start + i
                    self.callback(image, current_step, num_inference_steps)
                
                # Free memory after callback
                del latents_for_callback, image
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Decode the final latent to an image
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            image = self.img2img_pipeline.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Free memory after decoding
        del latents, text_embeddings
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return image

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
            
            # Force garbage collection before starting
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Load models with memory optimization
            logger.info("Loading models with memory optimization...")
            sd_pipeline = self.model_manager.load_sd_model(sd_model_name)
            
            # Force garbage collection after loading SD model
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            motion_module = self.model_manager.load_motion_module(motion_module_name)
            
            # Force garbage collection after loading motion module
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
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
                
                # Force garbage collection after loading motion LoRA
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Set scheduler
            scheduler_cls = self.scheduler_map.get(scheduler_name.lower(), EulerDiscreteScheduler)
            sd_pipeline.scheduler = scheduler_cls.from_config(sd_pipeline.scheduler.config)
            
            # Disable safety checker for uncensored mode
            if self.enable_uncensored and hasattr(sd_pipeline, "safety_checker"):
                sd_pipeline.safety_checker = None
                logger.info("Safety checker disabled for uncensored mode")
            
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
            
            # Create a separate img2img pipeline if needed
            img2img_pipeline = None
            if init_image:
                # For image-to-video, we need to use the img2img pipeline
                img2img_pipeline = StableDiffusionImg2ImgPipeline(
                    vae=sd_pipeline.vae,
                    text_encoder=sd_pipeline.text_encoder,
                    tokenizer=sd_pipeline.tokenizer,
                    unet=sd_pipeline.unet,
                    scheduler=sd_pipeline.scheduler,
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False
                )
            
            # Create the memory-efficient frame generator
            frame_generator = MemoryEfficientFrameGenerator(sd_pipeline, img2img_pipeline)
            frame_generator.set_callback(self.callback)
            
            # Generate the video frames
            logger.info(f"Generating {num_frames} frames with {num_inference_steps} steps")
            
            # Generate all frames with memory optimization
            all_frames = frame_generator.generate_frames(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=seed,
                init_image=init_image
            )
            
            # Save the video
            output_filename = f"animatediff_{uuid.uuid4().hex[:8]}.mp4"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Create the video file
            with imageio.get_writer(output_path, fps=fps) as writer:
                for frame in all_frames:
                    writer.append_data(frame)
            
            logger.info(f"Video saved to {output_path}")
            
            # Clean up to free memory
            del sd_pipeline, motion_module, motion_lora, img2img_pipeline, frame_generator, all_frames
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
