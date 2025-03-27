"""
ControlNet integration for AnimateDiff.

This module provides functions for loading and applying ControlNet models
to control image generation based on conditioning images.
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from PIL import Image
from diffusers import StableDiffusionPipeline
try:
    from diffusers import ControlNetModel
except ImportError:
    # Create a placeholder class if ControlNetModel is not available
    class ControlNetModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("ControlNetModel is not available in your version of diffusers. Please upgrade diffusers.")

logger = logging.getLogger(__name__)

class ControlNetManager:
    """Manages loading and applying ControlNet models"""
    
    def __init__(
        self,
        controlnet_dir: str,
        use_cpu: bool = True,
        optimize_memory: bool = True
    ):
        self.controlnet_dir = controlnet_dir
        self.use_cpu = use_cpu
        self.optimize_memory = optimize_memory
        
        # Create directories if they don't exist
        os.makedirs(self.controlnet_dir, exist_ok=True)
        
        # Cache for loaded models
        self.controlnet_cache = {}
        
        # Device configuration
        self.device = torch.device("cpu")
        logger.info(f"ControlNet using device: {self.device}")
    
    def get_controlnet_models(self) -> List[str]:
        """Get list of available ControlNet models"""
        models = []
        for ext in ["*.safetensors", "*.bin", "*.pt"]:
            try:
                models.extend([os.path.basename(f) for f in os.listdir(self.controlnet_dir) 
                              if f.endswith(ext.replace("*", ""))])
            except FileNotFoundError:
                pass
        
        # Add "none" option
        models = ["none"] + models
        return models
    
    def load_controlnet(
        self,
        pipeline: StableDiffusionPipeline,
        controlnet_name: str,
        conditioning_image: Optional[Union[str, Image.Image]] = None,
        conditioning_scale: float = 0.5
    ) -> StableDiffusionPipeline:
        """
        Load and apply a ControlNet model to a pipeline
        
        Args:
            pipeline: StableDiffusionPipeline to apply ControlNet to
            controlnet_name: Name of the ControlNet model
            conditioning_image: Conditioning image path or PIL Image
            conditioning_scale: Scale of the conditioning (0-1)
            
        Returns:
            Modified pipeline with ControlNet applied
        """
        if not controlnet_name or controlnet_name.lower() == "none" or not conditioning_image:
            return pipeline
        
        try:
            logger.info(f"Loading ControlNet: {controlnet_name} with scale {conditioning_scale}")
            
            # Get the path to the ControlNet model
            controlnet_path = os.path.join(self.controlnet_dir, controlnet_name)
            if not os.path.exists(controlnet_path):
                logger.warning(f"ControlNet model not found: {controlnet_path}")
                return pipeline
            
            # Load the conditioning image if it's a path
            if isinstance(conditioning_image, str):
                if os.path.exists(conditioning_image):
                    conditioning_image = Image.open(conditioning_image).convert("RGB")
                else:
                    logger.warning(f"Conditioning image not found: {conditioning_image}")
                    return pipeline
            
            # Load the ControlNet model
            if controlnet_name in self.controlnet_cache:
                controlnet = self.controlnet_cache[controlnet_name]
            else:
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_path,
                    torch_dtype=torch.float32
                )
                self.controlnet_cache[controlnet_name] = controlnet
            
            # Apply the ControlNet to the pipeline
            pipeline.controlnet = controlnet
            
            # Store the conditioning image and scale for later use during generation
            pipeline.controlnet_image = conditioning_image
            pipeline.controlnet_scale = conditioning_scale
            
            logger.info(f"Successfully applied ControlNet: {controlnet_name}")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error applying ControlNet {controlnet_name}: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            return pipeline
    
    def apply_controlnet_to_pipeline(
        self,
        pipeline: StableDiffusionPipeline,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        num_frames: int,
        **kwargs
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Apply ControlNet during pipeline generation
        
        Args:
            pipeline: StableDiffusionPipeline with ControlNet
            prompt: Text prompt for generation
            negative_prompt: Negative text prompt for generation
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale for generation
            num_frames: Number of frames to generate
            
        Returns:
            Tuple containing:
            - List of generated frames as numpy arrays
            - Dictionary of generation parameters
        """
        if not hasattr(pipeline, "controlnet") or not hasattr(pipeline, "controlnet_image"):
            # No ControlNet applied, use regular generation
            return pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_frames,
                **kwargs
            )
        
        # Use ControlNet for generation
        return pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_frames,
            image=pipeline.controlnet_image,
            controlnet_conditioning_scale=pipeline.controlnet_scale,
            **kwargs
        )
