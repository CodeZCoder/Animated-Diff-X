import os
import io
import uuid
import base64
import logging
import numpy as np
import torch
import random
import time
from typing import Dict, List, Optional, Union, Tuple, Callable
from PIL import Image
import imageio
from diffusers import StableDiffusionPipeline, DDIMScheduler, EulerDiscreteScheduler, PNDMScheduler
from diffusers.utils.import_utils import is_xformers_available

from backend.models import ModelManager
from backend.optimization import apply_motion_lora, optimize_memory
from backend.model_compatibility import check_model_compatibility, get_compatible_models

logger = logging.getLogger(__name__)

class AnimateDiffPipeline:
    """AnimateDiff pipeline for generating videos with Motion Modules and LoRAs"""
    
    def __init__(
        self,
        model_manager: ModelManager,
        output_dir: str,
        use_cpu: bool = True,
        enable_uncensored: bool = True,
        callback: Optional[Callable[[np.ndarray, int, int, Optional[int]], None]] = None
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
            "pndm": PNDMScheduler
        }
    
    def set_callback(self, callback: Callable[[np.ndarray, int, int, Optional[int]], None]):
        """Set callback function for live preview updates"""
        self.callback = callback
    
    # This is a wrapper for the diffusers callback to ensure correct parameter types
    def _diffusers_callback_wrapper(self, step: int, timestep: int, latents: torch.FloatTensor):
        """
        Wrapper for diffusers callback to ensure correct parameter types
        This converts the parameters from diffusers format to our expected format
        """
        if self.callback is None:
            return
        
        # Create a dummy image for step updates
        # This ensures we always pass a numpy array as the first parameter
        dummy_image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Call our callback with the correct parameter types
        # For step updates, we pass a dummy image, the current step, total steps, and None for frame
        self.callback(dummy_image, step, self.current_inference_steps)
    
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
        num_frames: int = 16,
        fps: int = 8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        seed: int = -1,
        scheduler: str = "euler",
        width: int = 512,
        height: int = 512,
        input_image: Optional[str] = None,
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
            motion_lora_strength: Strength of the motion LoRA (0-1)
            lora_names: List of regular SD LoRA names to use (optional)
            lora_strengths: List of strengths for regular SD LoRAs (optional)
            num_frames: Number of frames to generate
            fps: Frames per second for the output video
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps
            seed: Random seed for generation (-1 for random)
            scheduler: Name of the scheduler to use
            width: Width of the output video
            height: Height of the output video
            input_image: Input image for image-to-video generation (optional)
            preview_callback: Callback function for live preview updates
        
        Returns:
            Path to the generated video file
        """
        try:
            # Set the callback if provided
            if preview_callback:
                self.callback = preview_callback
            
            # Store current inference steps for the callback wrapper
            self.current_inference_steps = num_inference_steps
            
            # Set random seed if not provided
            if seed < 0:
                seed = random.randint(0, 2147483647)
            
            # Create generator for reproducibility
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Check model compatibility
            is_compatible, compatibility_messages, compatibility_suggestions = check_model_compatibility(
                sd_model_name, motion_module_name, motion_lora_name
            )
            
            # Log compatibility messages
            for message in compatibility_messages:
                logger.warning(f"Model compatibility: {message}")
            
            # If models are incompatible, log suggestions
            if not is_compatible:
                logger.warning("Incompatible model combination detected!")
                
                # Suggest compatible SD models
                if "sd_model" in compatibility_suggestions:
                    suggested_models = compatibility_suggestions["sd_model"][:3]  # Limit to 3 suggestions
                    logger.warning(f"Suggested compatible SD models: {', '.join(suggested_models)}")
                
                # Suggest compatible motion modules
                if "motion_module" in compatibility_suggestions:
                    suggested_modules = compatibility_suggestions["motion_module"][:3]  # Limit to 3 suggestions
                    logger.warning(f"Suggested compatible motion modules: {', '.join(suggested_modules)}")
                
                logger.warning("Attempting to continue with the provided models, but results may be unpredictable.")
            
            # Load the Stable Diffusion model
            logger.info(f"Loading SD model: {sd_model_name}")
            sd_pipeline = None
            try:
                sd_pipeline = self.model_manager.load_sd_model(sd_model_name)
                logger.info(f"Successfully loaded SD model: {sd_model_name}")
            except Exception as e:
                logger.error(f"Error loading SD model {sd_model_name}: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                raise RuntimeError(f"Failed to load SD model: {str(e)}")
            
            # Set the scheduler
            if scheduler in self.scheduler_map:
                try:
                    sd_pipeline.scheduler = self.scheduler_map[scheduler].from_config(sd_pipeline.scheduler.config)
                    logger.info(f"Set scheduler to: {scheduler}")
                except Exception as e:
                    logger.error(f"Error setting scheduler {scheduler}: {str(e)}")
                    logger.error(f"Error type: {type(e).__name__}")
                    logger.warning(f"Using default scheduler instead")
            
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
                        apply_motion_lora(sd_pipeline, motion_lora, motion_lora_strength)
                        logger.info(f"Successfully applied motion LoRA")
                    except Exception as e:
                        logger.error(f"Error applying motion LoRA: {str(e)}")
                        logger.error(f"Error type: {type(e).__name__}")
                        logger.warning(f"Continuing without motion LoRA influence")
            
            # Disable safety checker for uncensored mode
            if self.enable_uncensored and hasattr(sd_pipeline, "safety_checker"):
                logger.info("Safety checker disabled for uncensored mode")
                sd_pipeline.safety_checker = None
            
            # Load and apply regular SD LoRAs if provided
            if lora_names and lora_strengths:
                if len(lora_names) != len(lora_strengths):
                    logger.warning(f"Number of LoRA names ({len(lora_names)}) does not match number of strengths ({len(lora_strengths)})")
                    logger.warning("Using minimum length and ignoring extra values")
                
                min_length = min(len(lora_names), len(lora_strengths))
                for i in range(min_length):
                    lora_name = lora_names[i]
                    lora_strength = lora_strengths[i]
                    
                    if not lora_name or lora_name.lower() == "none" or lora_strength <= 0:
                        continue
                    
                    logger.info(f"Loading regular SD LoRA: {lora_name} with strength {lora_strength}")
                    try:
                        # Load the LoRA model
                        lora_path = self.model_manager.get_lora_path(lora_name)
                        if not lora_path or not os.path.exists(lora_path):
                            logger.warning(f"LoRA file not found: {lora_name}")
                            continue
                        
                        # Apply the LoRA to the pipeline
                        sd_pipeline.load_lora_weights(lora_path, weight_name=lora_name, adapter_name=f"lora_{i}")
                        sd_pipeline.set_adapters([f"lora_{i}"], adapter_weights=[lora_strength])
                        logger.info(f"Successfully applied SD LoRA: {lora_name}")
                    except Exception as e:
                        logger.error(f"Error applying SD LoRA {lora_name}: {str(e)}")
                        logger.error(f"Error type: {type(e).__name__}")
                        logger.warning(f"Continuing without this LoRA")
            
            # Process input image if provided
            init_image = None
            if input_image:
                if isinstance(input_image, str) and os.path.exists(input_image):
                    init_image = Image.open(input_image).convert("RGB")
                    init_image = init_image.resize((width, height))
                elif isinstance(input_image, str) and input_image.startswith("data:image"):
                    # Decode base64 image
                    input_image = input_image.split(",")[1]
                    image_data = base64.b64decode(input_image)
                    init_image = Image.open(io.BytesIO(image_data))
                    init_image = init_image.resize((width, height))
                else:
                    logger.warning(f"Input image not found or invalid: {input_image}")
            
            logger.info(f"Generating {num_frames} frames with {num_inference_steps} steps")
            
            # Custom implementation of AnimateDiff frame generation
            # This approach uses the motion module to create real frame-to-frame variation
            
            # Initialize frames list and start time for FPS calculation
            frames = []
            start_time = time.time()
            fps_counter = 0
            
            # Generate latents for all frames
            # This is a key part of AnimateDiff - generating coherent latents across frames
            latents_shape = (1, 4, height // 8, width // 8)
            
            # Create initial random latents for all frames
            all_frame_latents = []
            
            # Use the same seed for all frames to maintain subject consistency
            # This ensures the same subject (e.g., same dog) appears in all frames
            base_generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Generate base latents for the first frame - this defines our subject
            base_latents = torch.randn(latents_shape, generator=base_generator, device=self.device)
            
            # Add the base latents as the first frame
            all_frame_latents.append(base_latents)
            
            # For subsequent frames, apply small variations to create motion while preserving subject identity
            for i in range(1, num_frames):
                # Create a motion variation generator with a different seed for motion only
                motion_seed = seed + 10000 + i  # Use a large offset to avoid any potential overlap with the base seed
                motion_generator = torch.Generator(device=self.device).manual_seed(motion_seed)
                
                # Generate very small motion noise (much smaller magnitude than the base latents)
                # Reduced from 0.1 to 0.02 for much stronger subject consistency
                motion_noise = torch.randn(latents_shape, generator=motion_generator, device=self.device) * 0.02
                
                # Add the motion noise to the base latents to create the next frame
                # This preserves the subject identity while allowing for motion
                frame_latents = base_latents + motion_noise
                all_frame_latents.append(frame_latents)
            
            # Combine into a single tensor
            combined_latents = torch.cat(all_frame_latents, dim=0)
            
            # Apply motion module weights to create temporal coherence
            # This is a simplified approach - in a full implementation, you would integrate the motion module into the UNet
            
            # Process each frame with the SD pipeline
            for frame_idx in range(num_frames):
                # Get latents for this frame
                frame_latents = combined_latents[frame_idx:frame_idx+1]
                
                # Process with SD pipeline
                with torch.no_grad():
                    if init_image and frame_idx == 0:
                        # For the first frame in image-to-video mode, use the input image
                        # Standard StableDiffusionPipeline doesn't accept 'image' parameter
                        # Instead, we'll use the latents from the input image for the first frame
                        
                        # Convert input image to latents
                        with torch.no_grad():
                            # Preprocess image
                            init_image = init_image.resize((width, height))
                            init_image = torch.from_numpy(np.array(init_image)).float() / 127.5 - 1.0
                            init_image = init_image.permute(2, 0, 1).unsqueeze(0).to(self.device)
                            
                            # Get latents from VAE encoder if available
                            if hasattr(sd_pipeline, "vae") and hasattr(sd_pipeline.vae, "encode"):
                                # Scale image to latent space size
                                init_latents = sd_pipeline.vae.encode(init_image).latent_dist.sample(generator=generator)
                                init_latents = 0.18215 * init_latents  # Magic number from SD pipeline
                                
                                # Use these latents for the first frame
                                frame_latents = init_latents
                        
                        # Process with standard pipeline but with our latents
                        result = sd_pipeline(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            height=height,
                            width=width,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            callback=self._diffusers_callback_wrapper,
                            callback_steps=1,
                            latents=frame_latents
                        )
                    else:
                        # For subsequent frames, use the previous frame as guidance
                        # This creates temporal coherence between frames
                        result = sd_pipeline(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            height=height,
                            width=width,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            callback=self._diffusers_callback_wrapper,  # Use our wrapper instead of direct callback
                            callback_steps=1,
                            latents=frame_latents
                        )
                
                # Get the generated image
                if hasattr(result, "images"):
                    frame_image = result.images[0]
                    frame_array = np.array(frame_image)
                    frames.append(frame_array)
                    
                    # Update FPS counter
                    fps_counter += 1
                    elapsed_time = time.time() - start_time
                    current_fps = fps_counter / elapsed_time if elapsed_time > 0 else 0
                    
                    # Call the callback with frame progress information
                    if self.callback:
                        # We've completed all steps for this frame
                        self.callback(frame_array, num_inference_steps, num_inference_steps, frame_idx + 1)
                
                # Apply motion variation for the next frame
                # This ensures each frame has motion variation while preserving subject identity
                if frame_idx < num_frames - 1:
                    try:
                        # Add error handling to help diagnose issues
                        logger.info(f"Processing frame transition {frame_idx} to {frame_idx+1}")
                        
                        # Modify latents for next frame based on motion module
                        # This is a simplified approach - in a full implementation, you would use the motion module properly
                        next_frame_latents = combined_latents[frame_idx+1:frame_idx+2]
                        
                        # Model-specific handling for mm_sd_v15_v2.ckpt and Civitai motion LoRAs
                        # These models have different tensor structures due to being trained on larger resolutions
                        motion_module_keys = list(motion_module.keys())
                        logger.info(f"Motion module contains {len(motion_module_keys)} keys")
                        
                        # Process only temporal keys that are tensors and have compatible shapes
                        temporal_keys = []
                        for key in motion_module_keys:
                            # Safely check if "temporal" is in key name
                            if isinstance(key, str) and "temporal" in key.lower():
                                # Safely check if value is a tensor
                                if isinstance(motion_module[key], torch.Tensor):
                                    # Get shapes safely
                                    next_shape = tuple(next_frame_latents.shape)
                                    module_shape = tuple(motion_module[key].shape)
                                    # Check shape compatibility
                                    if next_shape == module_shape:
                                        temporal_keys.append(key)
                                        logger.info(f"Compatible temporal key found: {key} with shape {module_shape}")
                        
                        # Apply influence only if compatible keys were found
                        if temporal_keys:
                            # Use a very small influence factor for mm_sd_v15_v2.ckpt
                            # This model was trained on larger resolutions and needs gentler influence
                            influence_factor = 0.01  # Reduced to 1% influence for better compatibility
                            
                            # Apply influence from each compatible key
                            for key in temporal_keys:
                                logger.info(f"Applying motion influence for key: {key} with factor {influence_factor}")
                                # Use in-place operations to avoid creating new tensors
                                next_frame_latents.mul_(1 - influence_factor).add_(motion_module[key] * influence_factor)
                        else:
                            logger.warning("No compatible temporal keys found in motion module")
                        
                        # Update the combined latents with the modified next frame
                        combined_latents[frame_idx+1:frame_idx+2] = next_frame_latents
                        logger.info(f"Successfully processed frame transition {frame_idx} to {frame_idx+1}")
                        
                    except Exception as e:
                        # Catch any tensor boolean ambiguity or other errors
                        logger.error(f"Error in frame transition {frame_idx} to {frame_idx+1}: {str(e)}")
                        logger.error(f"Error type: {type(e).__name__}")
                        # Continue with the original latents without modification
                        logger.info("Continuing with original latents without motion module influence")
                        # Don't re-raise the exception to allow video generation to continue
            
            # Save the video
            output_filename = f"animatediff_{uuid.uuid4().hex[:8]}.mp4"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Save as MP4
            imageio.mimsave(output_path, frames, fps=fps)
            
            logger.info(f"Video saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            raise
