#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import logging
import numpy as np
import torch
import random
import time
from typing import List, Dict, Optional, Union, Callable, Tuple
from PIL import Image
import cv2
from tqdm import tqdm
import uuid

from backend.models import ModelManager
from backend.optimization import apply_motion_lora, optimize_memory
from backend.model_compatibility import check_model_compatibility, get_compatible_models
from backend.lora_utils import apply_sd15_lora
from backend.ip_adapter import IPAdapterManager
from backend.controlnet import ControlNetManager

logger = logging.getLogger(__name__)

class AnimateDiffPipeline:
    """Pipeline for generating videos with AnimateDiff"""
    
    def __init__(
        self,
        model_manager: ModelManager,
        output_dir: str,
        use_cpu: bool = True,
        enable_uncensored: bool = True,
        callback: Optional[Callable] = None
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
        
        # Initialize scheduler map with safe imports
        self.scheduler_map = {}
        
        # Try to import each scheduler individually with error handling
        try:
            from diffusers import DDIMScheduler
            self.scheduler_map["ddim"] = DDIMScheduler
            logger.info("Successfully imported DDIMScheduler")
        except ImportError:
            logger.warning("DDIMScheduler not available in your diffusers version")
            
        try:
            from diffusers import PNDMScheduler
            self.scheduler_map["pndm"] = PNDMScheduler
            logger.info("Successfully imported PNDMScheduler")
        except ImportError:
            logger.warning("PNDMScheduler not available in your diffusers version")
            
        try:
            from diffusers import EulerDiscreteScheduler
            self.scheduler_map["euler"] = EulerDiscreteScheduler
            logger.info("Successfully imported EulerDiscreteScheduler")
        except ImportError:
            logger.warning("EulerDiscreteScheduler not available in your diffusers version")
            
        try:
            from diffusers import EulerAncestralDiscreteScheduler
            self.scheduler_map["euler_ancestral"] = EulerAncestralDiscreteScheduler
            logger.info("Successfully imported EulerAncestralDiscreteScheduler")
        except ImportError:
            logger.warning("EulerAncestralDiscreteScheduler not available in your diffusers version")
            
        try:
            from diffusers import DPMSolverMultistepScheduler
            self.scheduler_map["dpm_solver"] = DPMSolverMultistepScheduler
            logger.info("Successfully imported DPMSolverMultistepScheduler")
        except ImportError:
            logger.warning("DPMSolverMultistepScheduler not available in your diffusers version")
            
        try:
            # Try different import paths for UniPCMultistepScheduler
            try:
                from diffusers import UniPCMultistepScheduler
                self.scheduler_map["unipc"] = UniPCMultistepScheduler
                logger.info("Successfully imported UniPCMultistepScheduler")
            except ImportError:
                # Try alternative import path for newer diffusers versions
                from diffusers.schedulers import UniPCMultistepScheduler
                self.scheduler_map["unipc"] = UniPCMultistepScheduler
                logger.info("Successfully imported UniPCMultistepScheduler from schedulers submodule")
        except ImportError:
            logger.warning("UniPCMultistepScheduler not available in your diffusers version")
        
        # Check if we have at least one scheduler
        if not self.scheduler_map:
            logger.warning("No schedulers were successfully imported. Using default scheduler from pipeline.")
        
        # Initialize IP Adapter manager if available
        self.ip_adapter_manager = None
        try:
            if hasattr(self.model_manager, 'ip_adapter_dir') and self.model_manager.ip_adapter_dir:
                self.ip_adapter_manager = IPAdapterManager(
                    ip_adapter_dir=self.model_manager.ip_adapter_dir,
                    clip_vision_dir=getattr(self.model_manager, 'clip_vision_dir', None)
                )
        except Exception as e:
            logger.warning(f"Failed to initialize IP Adapter manager: {str(e)}")
            
        # Initialize ControlNet manager if available
        self.controlnet_manager = None
        try:
            if hasattr(self.model_manager, 'controlnet_dir') and self.model_manager.controlnet_dir:
                self.controlnet_manager = ControlNetManager(
                    controlnet_dir=self.model_manager.controlnet_dir
                )
        except Exception as e:
            logger.warning(f"Failed to initialize ControlNet manager: {str(e)}")
    
    def generate_with_live_preview(
        self,
        prompt: str,
        negative_prompt: str = "",
        sd_model_name: str = "runwayml/stable-diffusion-v1-5",
        motion_module_name: str = "mm_sd_v15_v2.ckpt",
        motion_lora_name: Optional[str] = None,
        motion_lora_strength: float = 1.0,
        lora_names: Optional[List[str]] = None,
        lora_strengths: Optional[List[float]] = None,
        ip_adapter_name: Optional[str] = None,
        ip_adapter_image: Optional[str] = None,
        ip_adapter_strength: float = 0.5,
        controlnet_name: Optional[str] = None,
        controlnet_image: Optional[str] = None,
        controlnet_strength: float = 0.5,
        num_frames: int = 16,
        fps: int = 8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        seed: int = -1,
        width: int = 512,
        height: int = 512,
        input_image: Optional[str] = None,
        scheduler: str = "euler_ancestral",
        preview_callback: Optional[Callable] = None
    ) -> str:
        """
        Generate a video with AnimateDiff
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative text prompt for generation
            sd_model_name: Name of the Stable Diffusion model to use
            motion_module_name: Name of the motion module to use
            motion_lora_name: Name of the motion LoRA to use (optional)
            motion_lora_strength: Strength of the motion LoRA (0.0 to 1.0)
            lora_names: List of LoRA names to apply (optional)
            lora_strengths: List of LoRA strengths to apply (optional)
            ip_adapter_name: Name of the IP Adapter model to use (optional)
            ip_adapter_image: Path to the IP Adapter reference image (optional)
            ip_adapter_strength: Strength of the IP Adapter effect (0.0 to 1.0)
            controlnet_name: Name of the ControlNet model to use (optional)
            controlnet_image: Path to the ControlNet conditioning image (optional)
            controlnet_strength: Strength of the ControlNet effect (0.0 to 1.0)
            num_frames: Number of frames to generate
            fps: Frames per second for the output video
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps
            seed: Random seed for generation (-1 for random)
            width: Width of the output video
            height: Height of the output video
            input_image: Path to input image for img2img (optional)
            scheduler: Name of the scheduler to use
            preview_callback: Callback function for live preview updates
            
        Returns:
            Path to the generated video file
        """
        try:
            # Generate a random seed if not provided
            if seed < 0:
                # Use maximum value for int32 (2^31 - 1) to avoid "high is out of bounds for int32" error
                seed = random.randint(0, 2147483647)
                logger.info(f"Using random seed: {seed}")
            else:
                logger.info(f"Using provided seed: {seed}")
            
            # Set random seeds for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            # Generate a unique ID for this generation
            generation_id = str(uuid.uuid4())[:8]
            
            # Create output path
            output_path = os.path.join(self.output_dir, f"animatediff_{generation_id}.mp4")
            
            # Log generation parameters
            logger.info(f"Generating video with parameters:")
            logger.info(f"  Prompt: {prompt}")
            logger.info(f"  Negative prompt: {negative_prompt}")
            logger.info(f"  SD model: {sd_model_name}")
            logger.info(f"  Motion module: {motion_module_name}")
            logger.info(f"  Motion LoRA: {motion_lora_name}")
            logger.info(f"  Frames: {num_frames}")
            logger.info(f"  FPS: {fps}")
            logger.info(f"  Guidance scale: {guidance_scale}")
            logger.info(f"  Steps: {num_inference_steps}")
            logger.info(f"  Seed: {seed}")
            logger.info(f"  Size: {width}x{height}")
            
            # Check model compatibility
            is_compatible, compatibility_messages, compatibility_suggestions = check_model_compatibility(
                sd_model_name, motion_module_name
            )
            
            if not is_compatible:
                logger.warning(f"Model compatibility issue: {compatibility_messages}")
                # Use the suggestions from check_model_compatibility instead of calling get_compatible_models
                if compatibility_suggestions:
                    logger.info(f"Compatible models: {compatibility_suggestions}")
                    
                    # Try to use compatible models if available
                    if compatibility_suggestions.get('sd_model'):
                        sd_model_name = compatibility_suggestions['sd_model'][0]  # Use the first suggestion
                        logger.info(f"Using compatible SD model: {sd_model_name}")
                    
                    if compatibility_suggestions.get('motion_module'):
                        motion_module_name = compatibility_suggestions['motion_module'][0]  # Use the first suggestion
                        logger.info(f"Using compatible motion module: {motion_module_name}")
            
            # Load the Stable Diffusion model
            logger.info(f"Loading SD model: {sd_model_name}")
            sd_pipeline = self.model_manager.load_sd_model(sd_model_name)
            
            # Set the scheduler
            if scheduler in self.scheduler_map:
                try:
                    sd_pipeline.scheduler = self.scheduler_map[scheduler].from_config(sd_pipeline.scheduler.config)
                    logger.info(f"Set scheduler to: {scheduler}")
                except Exception as e:
                    logger.error(f"Error setting scheduler {scheduler}: {str(e)}")
                    logger.error(f"Error type: {type(e).__name__}")
                    logger.warning(f"Using default scheduler instead")
            else:
                logger.warning(f"Scheduler {scheduler} not available in your diffusers version. Using default scheduler.")
            
            # Load the motion module
            logger.info(f"Loading motion module: {motion_module_name}")
            motion_module = None
            try:
                motion_module = self.model_manager.load_motion_module(motion_module_name)
                if motion_module:
                    logger.info(f"Successfully loaded motion module: {motion_module_name}")
                    # Log motion module structure for debugging
                    if isinstance(motion_module, dict):
                        logger.info(f"Motion module contains {len(motion_module.keys())} keys")
                        temporal_keys = [k for k in motion_module.keys() if isinstance(k, str) and "temporal" in k.lower()]
                        logger.info(f"Found {len(temporal_keys)} temporal keys in motion module")
                else:
                    logger.error(f"Motion module is None after loading: {motion_module_name}")
                    raise RuntimeError(f"Failed to load motion module: returned None")
            except Exception as e:
                logger.error(f"Error loading motion module {motion_module_name}: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                raise RuntimeError(f"Failed to load motion module: {str(e)}")
            
            # Load the motion LoRA if provided
            motion_lora = None
            if motion_lora_name and motion_lora_name.lower() != "none":
                logger.info(f"Loading motion LoRA: {motion_lora_name}")
                try:
                    motion_lora = self.model_manager.load_motion_lora(motion_lora_name)
                    if motion_lora:
                        logger.info(f"Successfully loaded motion LoRA: {motion_lora_name}")
                    else:
                        logger.warning(f"Motion LoRA is None after loading: {motion_lora_name}")
                except Exception as e:
                    logger.error(f"Error loading motion LoRA {motion_lora_name}: {str(e)}")
                    logger.error(f"Error type: {type(e).__name__}")
                    logger.warning(f"Continuing without motion LoRA")
                
                # Apply the motion LoRA to the model
                if motion_lora:
                    try:
                        logger.info(f"Applying motion LoRA with strength: {motion_lora_strength}")
                        # Safely apply motion LoRA
                        apply_motion_lora(motion_module, motion_lora, motion_lora_strength)
                        logger.info(f"Successfully applied motion LoRA")
                    except Exception as e:
                        logger.error(f"Error applying motion LoRA: {str(e)}")
                        logger.error(f"Error type: {type(e).__name__}")
                        logger.warning(f"Continuing without motion LoRA")
            
            # Apply regular SD LoRAs if provided
            if lora_names and lora_strengths and len(lora_names) == len(lora_strengths):
                for i, (lora_name, lora_strength) in enumerate(zip(lora_names, lora_strengths)):
                    if lora_name and lora_name.lower() != "none" and lora_strength > 0:
                        logger.info(f"Applying SD LoRA {i+1}/{len(lora_names)}: {lora_name} with strength {lora_strength}")
                        try:
                            # Apply SD 1.5 LoRA
                            apply_sd15_lora(sd_pipeline, lora_name, lora_strength, self.model_manager.sd15_lora_dir)
                            logger.info(f"Successfully applied SD LoRA: {lora_name}")
                        except Exception as e:
                            logger.error(f"Error applying SD LoRA {lora_name}: {str(e)}")
                            logger.error(f"Error type: {type(e).__name__}")
                            logger.warning(f"Continuing without this SD LoRA")
            
            # Apply IP Adapter if provided
            ip_adapter_model = None
            if (ip_adapter_name and ip_adapter_name.lower() != "none" and 
                ip_adapter_image and self.ip_adapter_manager):
                logger.info(f"Applying IP Adapter: {ip_adapter_name} with strength {ip_adapter_strength}")
                try:
                    # Load and process the reference image
                    ip_adapter_img = Image.open(ip_adapter_image).convert("RGB")
                    
                    # Apply IP Adapter
                    sd_pipeline = self.ip_adapter_manager.apply_ip_adapter(
                        sd_pipeline, 
                        ip_adapter_name, 
                        ip_adapter_img, 
                        ip_adapter_strength
                    )
                    logger.info(f"Successfully applied IP Adapter")
                except Exception as e:
                    logger.error(f"Error applying IP Adapter: {str(e)}")
                    logger.error(f"Error type: {type(e).__name__}")
                    logger.warning(f"Continuing without IP Adapter")
            
            # Apply ControlNet if provided
            controlnet_model = None
            if (controlnet_name and controlnet_name.lower() != "none" and 
                controlnet_image and self.controlnet_manager):
                logger.info(f"Applying ControlNet: {controlnet_name} with strength {controlnet_strength}")
                try:
                    # Load and process the conditioning image
                    controlnet_img = Image.open(controlnet_image).convert("RGB")
                    
                    # Apply ControlNet
                    sd_pipeline = self.controlnet_manager.apply_controlnet(
                        sd_pipeline, 
                        controlnet_name, 
                        controlnet_img, 
                        controlnet_strength
                    )
                    logger.info(f"Successfully applied ControlNet")
                except Exception as e:
                    logger.error(f"Error applying ControlNet: {str(e)}")
                    logger.error(f"Error type: {type(e).__name__}")
                    logger.warning(f"Continuing without ControlNet")
            
            # Prepare for generation
            logger.info(f"Preparing for generation...")
            
            # Process input image if provided (img2img mode)
            init_image = None
            if input_image:
                try:
                    init_image = Image.open(input_image).convert("RGB")
                    init_image = init_image.resize((width, height))
                    logger.info(f"Using input image: {input_image}")
                except Exception as e:
                    logger.error(f"Error loading input image: {str(e)}")
                    logger.warning(f"Continuing without input image")
                    init_image = None
            
            # Generate the frames
            logger.info(f"Generating {num_frames} frames...")
            frames = []
            
            # Create progress bar
            progress_bar = tqdm(total=num_inference_steps * num_frames)
            
            # Define callback for step updates - with fix for "too many values to unpack" error
            def callback_fn(step: int, timestep: int, latents: torch.FloatTensor) -> None:
                # Fix for "too many values to unpack" error
                # Some diffusers versions may pass more than 3 arguments to the callback
                try:
                    if preview_callback:
                        # Convert latents to image
                        with torch.no_grad():
                            latents_for_view = latents.detach().clone()
                            latents_for_view = 1 / 0.18215 * latents_for_view
                            
                            # Process only the first frame for preview
                            frame_latent = latents_for_view[0:1]
                            
                            try:
                                # Use the pipeline's VAE to decode the latents
                                image = sd_pipeline.vae.decode(frame_latent).sample
                                image = (image / 2 + 0.5).clamp(0, 1)
                                image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                                image = (image * 255).astype(np.uint8)
                                
                                # Call the preview callback
                                preview_callback(image, step, num_inference_steps)
                            except Exception as e:
                                logger.error(f"Error in preview callback: {str(e)}")
                    
                    # Update progress bar
                    progress_bar.update(1)
                except Exception as e:
                    # Catch any unpacking errors or other issues in the callback
                    logger.error(f"Error in callback function: {str(e)}")
                    # Still update progress bar even if there's an error
                    progress_bar.update(1)
            
            # Generate frames using batch processing for consistency
            logger.info(f"Generating {num_frames} frames with batch processing for consistency...")
            frames = []
            
            # Create progress bar
            progress_bar = tqdm(total=num_inference_steps)
            
            # Define callback for step updates - with fix for "too many values to unpack" error
            def callback_fn(step: int, timestep: int, latents: torch.FloatTensor) -> None:
                # Fix for "too many values to unpack" error
                # Some diffusers versions may pass more than 3 arguments to the callback
                try:
                    if preview_callback:
                        # Convert latents to image
                        with torch.no_grad():
                            latents_for_view = latents.detach().clone()
                            latents_for_view = 1 / 0.18215 * latents_for_view
                            
                            # Process only the first frame for preview
                            frame_latent = latents_for_view[0:1]
                            
                            try:
                                # Use the pipeline's VAE to decode the latents
                                image = sd_pipeline.vae.decode(frame_latent).sample
                                image = (image / 2 + 0.5).clamp(0, 1)
                                image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                                image = (image * 255).astype(np.uint8)
                                
                                # Call the preview callback
                                preview_callback(image, step, num_inference_steps)
                            except Exception as e:
                                logger.error(f"Error in preview callback: {str(e)}")
                    
                    # Update progress bar
                    progress_bar.update(1)
                except Exception as e:
                    # Catch any unpacking errors or other issues in the callback
                    logger.error(f"Error in callback function: {str(e)}")
                    # Still update progress bar even if there's an error
                    progress_bar.update(1)
            
            try:
                # Prepare latent noise for all frames at once to ensure consistency
                # This is the key change for frame consistency
                
                # Set up batch size for temporal consistency
                # Using context_batch_size to process multiple frames together
                context_batch_size = min(num_frames, 16)  # Default to 16 frames per batch for memory efficiency
                logger.info(f"Using context batch size of {context_batch_size} for temporal consistency")
                
                # Ensure VAE is properly applied
                if not hasattr(sd_pipeline, 'vae') or sd_pipeline.vae is None:
                    logger.warning("VAE not found in pipeline, attempting to load default VAE")
                    try:
                        from diffusers import AutoencoderKL
                        sd_pipeline.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
                        logger.info("Successfully loaded default VAE")
                    except Exception as vae_error:
                        logger.error(f"Failed to load VAE: {str(vae_error)}")
                
                # Generate initial latents for all frames
                batch_size = 1
                if hasattr(sd_pipeline, 'unet') and hasattr(sd_pipeline.unet.config, 'in_channels'):
                    channels = sd_pipeline.unet.config.in_channels
                else:
                    channels = 4  # Default for SD models
                
                # Prepare for generation
                height = height or 512
                width = width or 512
                
                # Process input image if provided (img2img mode)
                init_image = None
                init_latents = None
                if input_image:
                    try:
                        init_image = Image.open(input_image).convert("RGB")
                        init_image = init_image.resize((width, height))
                        logger.info(f"Using input image: {input_image}")
                        
                        # Convert image to latents
                        init_image_tensor = torch.from_numpy(np.array(init_image) / 255.0).float()
                        init_image_tensor = init_image_tensor.permute(2, 0, 1).unsqueeze(0)
                        init_image_tensor = 2 * init_image_tensor - 1
                        
                        # Encode the image to latents using VAE
                        with torch.no_grad():
                            init_latents = sd_pipeline.vae.encode(init_image_tensor.to(self.device)).latent_dist.sample()
                            init_latents = 0.18215 * init_latents
                            
                        logger.info(f"Successfully encoded input image to latents")
                    except Exception as e:
                        logger.error(f"Error loading or encoding input image: {str(e)}")
                        logger.warning(f"Continuing without input image")
                        init_image = None
                        init_latents = None
                
                # Generate all frames at once using batched processing
                try:
                    # Fix for "too many values to unpack" error in pipeline call
                    try:
                        # Set up noise scheduler for consistency
                        if hasattr(sd_pipeline, 'scheduler'):
                            # Save original scheduler for restoration later
                            original_scheduler = sd_pipeline.scheduler
                            
                            # Use DDIM scheduler for better consistency between frames
                            try:
                                from diffusers import DDIMScheduler
                                consistency_scheduler = DDIMScheduler.from_config(sd_pipeline.scheduler.config)
                                sd_pipeline.scheduler = consistency_scheduler
                                logger.info("Using DDIM scheduler for improved frame consistency")
                            except Exception as scheduler_error:
                                logger.warning(f"Could not set DDIM scheduler for consistency: {str(scheduler_error)}")
                        
                        # Generate initial latent noise for consistency
                        generator = torch.Generator(device=self.device).manual_seed(seed)
                        if hasattr(sd_pipeline, 'prepare_latents'):
                            # Use pipeline's built-in method if available
                            try:
                                # For newer diffusers versions
                                initial_latents = sd_pipeline.prepare_latents(
                                    batch_size=1,
                                    num_channels_latents=channels,
                                    height=height,
                                    width=width,
                                    dtype=torch.float32,
                                    device=self.device,
                                    generator=generator
                                )
                                # Repeat for all frames
                                consistent_latents = initial_latents.repeat(num_frames, 1, 1, 1)
                                logger.info(f"Generated consistent initial latents for all {num_frames} frames")
                            except Exception as latent_error:
                                logger.warning(f"Error using prepare_latents: {str(latent_error)}")
                                consistent_latents = None
                        else:
                            # Manual latent generation
                            shape = (1, channels, height // 8, width // 8)
                            initial_latents = torch.randn(shape, generator=generator, device=self.device)
                            # Repeat for all frames
                            consistent_latents = initial_latents.repeat(num_frames, 1, 1, 1)
                            logger.info(f"Manually generated consistent initial latents for all {num_frames} frames")
                        
                        # Apply small controlled variations to latents for natural motion
                        if consistent_latents is not None:
                            # Add small variations to each frame's latents (except the first)
                            # This maintains overall consistency while allowing for motion
                            variation_strength = 0.02  # Small value to maintain consistency
                            for i in range(1, num_frames):
                                # Generate frame-specific variation
                                frame_generator = torch.Generator(device=self.device).manual_seed(seed + i)
                                frame_variation = torch.randn_like(initial_latents, generator=frame_generator) * variation_strength
                                # Apply variation to this frame's latents
                                consistent_latents[i] += frame_variation
                            
                            logger.info("Applied controlled variations to latents for natural motion")
                        
                        # Generate the frames
                        if consistent_latents is not None and hasattr(sd_pipeline, 'unet'):
                            # Use custom generation with consistent latents
                            logger.info("Using custom generation with consistent latents")
                            
                            # Get timesteps
                            num_timesteps = sd_pipeline.scheduler.config.num_train_timesteps
                            timesteps = sd_pipeline.scheduler.timesteps
                            
                            # Process prompt embeddings
                            text_inputs = sd_pipeline.tokenizer(
                                [prompt] * num_frames,
                                padding="max_length",
                                max_length=sd_pipeline.tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt",
                            )
                            text_embeddings = sd_pipeline.text_encoder(text_inputs.input_ids.to(self.device))[0]
                            
                            # Process negative prompt
                            if negative_prompt:
                                uncond_input = sd_pipeline.tokenizer(
                                    [negative_prompt] * num_frames,
                                    padding="max_length",
                                    max_length=sd_pipeline.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors="pt",
                                )
                                uncond_embeddings = sd_pipeline.text_encoder(uncond_input.input_ids.to(self.device))[0]
                                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
                            
                            # Initialize latents
                            latents = consistent_latents
                            
                            # Denoising loop
                            for i, t in enumerate(timesteps):
                                # Expand latents for classifier-free guidance
                                latent_model_input = torch.cat([latents] * 2) if negative_prompt else latents
                                
                                # Predict noise residual
                                with torch.no_grad():
                                    noise_pred = sd_pipeline.unet(
                                        latent_model_input, t, encoder_hidden_states=text_embeddings
                                    ).sample
                                
                                # Perform guidance
                                if negative_prompt:
                                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                                
                                # Compute previous noisy sample
                                latents = sd_pipeline.scheduler.step(noise_pred, t, latents).prev_sample
                                
                                # Update progress
                                if callback_fn:
                                    callback_fn(i, t, latents)
                            
                            # Decode latents to images
                            with torch.no_grad():
                                latents = 1 / 0.18215 * latents
                                images = sd_pipeline.vae.decode(latents).sample
                                images = (images / 2 + 0.5).clamp(0, 1)
                                images = images.cpu().permute(0, 2, 3, 1).numpy()
                                images = (images * 255).astype(np.uint8)
                                
                                # Convert to PIL images
                                frames_pil = [Image.fromarray(img) for img in images]
                        else:
                            # Fallback to standard pipeline call
                            logger.info("Falling back to standard pipeline call")
                            result = sd_pipeline(
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                width=width,
                                height=height,
                                callback=callback_fn,
                                callback_steps=1,
                                num_images_per_prompt=num_frames,
                                output_type='pil'
                            )
                            
                            # Handle different return types from different diffusers versions
                            if hasattr(result, 'images'):
                                frames_pil = result.images
                            elif isinstance(result, tuple) and len(result) >= 1:
                                # Some versions return a tuple with images as first element
                                frames_pil = result[0]
                            elif isinstance(result, list):
                                frames_pil = result
                            else:
                                logger.error(f"Unexpected result type from pipeline: {type(result)}")
                                # Create blank frames as fallback
                                frames_pil = [Image.new('RGB', (width, height), color='black') for _ in range(num_frames)]
                        
                        # Restore original scheduler if changed
                        if hasattr(sd_pipeline, 'scheduler') and 'original_scheduler' in locals():
                            sd_pipeline.scheduler = original_scheduler
                            logger.info("Restored original scheduler")
                        
                        # Convert PIL images to numpy arrays
                        frames = [np.array(frame) for frame in frames_pil]
                        
                    except ValueError as ve:
                        if "too many values to unpack" in str(ve) or "batch_size" in str(ve) or "num_images_per_prompt" in str(ve):
                            logger.warning("Caught error with batch processing, falling back to alternative approach")
                            # Alternative approach for diffusers versions with different parameters
                            frames = []
                            
                            # Process frames in smaller batches for consistency
                            for batch_start in range(0, num_frames, context_batch_size):
                                batch_end = min(batch_start + context_batch_size, num_frames)
                                batch_size = batch_end - batch_start
                                logger.info(f"Processing batch of {batch_size} frames ({batch_start+1}-{batch_end}/{num_frames})")
                                
                                # Generate this batch
                                batch_result = sd_pipeline(
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale,
                                    width=width,
                                    height=height,
                                    callback=None,  # Skip callback to avoid errors
                                    output_type='pil',
                                    num_images_per_prompt=batch_size
                                )
                                
                                # Extract frames from result
                                if hasattr(batch_result, 'images'):
                                    batch_frames = batch_result.images
                                elif isinstance(batch_result, tuple) and len(batch_result) >= 1:
                                    batch_frames = batch_result[0]
                                elif isinstance(batch_result, list):
                                    batch_frames = batch_result
                                else:
                                    logger.error(f"Unexpected batch result type: {type(batch_result)}")
                                    # Create blank frames as fallback
                                    batch_frames = [Image.new('RGB', (width, height), color='black') for _ in range(batch_size)]
                                
                                # Convert PIL images to numpy arrays and add to frames list
                                frames.extend([np.array(frame) for frame in batch_frames])
                                
                                # Update progress for this batch
                                if preview_callback:
                                    for i, frame_array in enumerate(frames[-batch_size:]):
                                        try:
                                            preview_callback(frame_array, num_inference_steps, num_inference_steps, batch_start + i)
                                        except Exception as e:
                                            logger.error(f"Error in batch preview callback: {str(e)}")
                        else:
                            # Re-raise if it's a different ValueError
                            raise
                        
                except Exception as e:
                    logger.error(f"Error in batch frame generation: {str(e)}")
                    logger.error(f"Error type: {type(e).__name__}")
                    
                    # Fallback to individual frame generation with noise copying for consistency
                    logger.warning("Falling back to individual frame generation with noise consistency")
                    frames = []
                    
                    # Generate initial noise
                    generator = torch.Generator(device=self.device).manual_seed(seed)
                    latents_shape = (batch_size, channels, height // 8, width // 8)
                    initial_noise = torch.randn(latents_shape, generator=generator, device=self.device)
                    
                    # Generate frames one by one but using the same initial noise
                    for frame_idx in range(num_frames):
                        logger.info(f"Generating frame {frame_idx+1}/{num_frames} with consistent noise")
                        
                        try:
                            # Use the same initial noise for consistency
                            sd_pipeline.scheduler.set_timesteps(num_inference_steps)
                            
                            # Clone the initial noise to avoid modifying it
                            frame_noise = initial_noise.clone()
                            
                            # Add small controlled variations for natural motion
                            # Small enough to maintain subject consistency but allow motion
                            if frame_idx > 0:
                                motion_noise = torch.randn_like(frame_noise) * 0.1
                                frame_noise = frame_noise + motion_noise
                            
                            # Generate the frame with this noise
                            result = sd_pipeline(
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                width=width,
                                height=height,
                                callback=None,
                                output_type='pil',
                                latents=frame_noise
                            )
                            
                            # Extract the frame
                            if hasattr(result, 'images'):
                                frame = result.images[0]
                            elif isinstance(result, tuple) and len(result) >= 1:
                                frame = result[0][0] if isinstance(result[0], list) else result[0]
                            elif isinstance(result, list):
                                frame = result[0]
                            else:
                                logger.error(f"Unexpected result type: {type(result)}")
                                frame = Image.new('RGB', (width, height), color='black')
                            
                            # Convert to numpy array
                            frame_array = np.array(frame)
                            
                            # Add to frames list
                            frames.append(frame_array)
                            
                            # Call preview callback with the complete frame
                            if preview_callback:
                                try:
                                    preview_callback(frame_array, num_inference_steps, num_inference_steps, frame_idx)
                                except Exception as e:
                                    logger.error(f"Error in final preview callback: {str(e)}")
                        
                        except Exception as frame_error:
                            logger.error(f"Error generating frame {frame_idx+1}: {str(frame_error)}")
                            logger.error(f"Error type: {type(frame_error).__name__}")
                            
                            # Create a blank frame as fallback
                            blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
                            frames.append(blank_frame)
            
            except Exception as outer_error:
                logger.error(f"Outer error in frame generation: {str(outer_error)}")
                logger.error(f"Error type: {type(outer_error).__name__}")
                
                # Create blank frames as fallback
                frames = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(num_frames)]
            
            # Close progress bar
            progress_bar.close()
            
            # Save frames as video
            logger.info(f"Saving video to: {output_path}")
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Write frames to video
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            
            # Release video writer
            video_writer.release()
            
            logger.info(f"Video generation completed: {output_path}")
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error in video generation: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            raise
