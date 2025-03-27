"""
IP Adapter integration for AnimateDiff.

This module provides functions for loading and applying IP Adapter models
to control image generation based on reference images.
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from PIL import Image
from diffusers import StableDiffusionPipeline
from diffusers.utils import load_image

logger = logging.getLogger(__name__)

class IPAdapterManager:
    """Manages loading and applying IP Adapter models"""
    
    def __init__(
        self,
        ip_adapter_dir: str,
        clip_vision_dir: str,
        use_cpu: bool = True,
        optimize_memory: bool = True
    ):
        self.ip_adapter_dir = ip_adapter_dir
        self.clip_vision_dir = clip_vision_dir
        self.use_cpu = use_cpu
        self.optimize_memory = optimize_memory
        
        # Create directories if they don't exist
        os.makedirs(self.ip_adapter_dir, exist_ok=True)
        os.makedirs(self.clip_vision_dir, exist_ok=True)
        
        # Cache for loaded models
        self.ip_adapter_cache = {}
        self.clip_vision_cache = {}
        
        # Device configuration
        self.device = torch.device("cpu")
        logger.info(f"IP Adapter using device: {self.device}")
    
    def get_ip_adapter_models(self) -> List[str]:
        """Get list of available IP Adapter models"""
        models = []
        for ext in ["*.safetensors", "*.bin", "*.pt"]:
            models.extend([os.path.basename(f) for f in os.listdir(self.ip_adapter_dir) if f.endswith(ext.replace("*", ""))])
        
        # Add "none" option
        models = ["none"] + models
        return models
    
    def get_clip_vision_models(self) -> List[str]:
        """Get list of available CLIP Vision models"""
        models = []
        for ext in ["*.safetensors", "*.bin", "*.pt"]:
            models.extend([os.path.basename(f) for f in os.listdir(self.clip_vision_dir) if f.endswith(ext.replace("*", ""))])
        return models
    
    def load_ip_adapter(
        self,
        pipeline: StableDiffusionPipeline,
        ip_adapter_name: str,
        reference_image: Optional[Union[str, Image.Image]] = None,
        weight: float = 0.5
    ) -> StableDiffusionPipeline:
        """
        Load and apply an IP Adapter model to a pipeline
        
        Args:
            pipeline: StableDiffusionPipeline to apply IP Adapter to
            ip_adapter_name: Name of the IP Adapter model
            reference_image: Reference image path or PIL Image
            weight: Weight of the IP Adapter (0-1)
            
        Returns:
            Modified pipeline with IP Adapter applied
        """
        if not ip_adapter_name or ip_adapter_name.lower() == "none" or not reference_image:
            return pipeline
        
        try:
            # Import here to avoid dependency issues if IP Adapter is not installed
            from diffusers import IPAdapterProcessor, IPAdapterModel
            
            logger.info(f"Loading IP Adapter: {ip_adapter_name} with weight {weight}")
            
            # Get the path to the IP Adapter model
            ip_adapter_path = os.path.join(self.ip_adapter_dir, ip_adapter_name)
            if not os.path.exists(ip_adapter_path):
                logger.warning(f"IP Adapter model not found: {ip_adapter_path}")
                return pipeline
            
            # Load the reference image if it's a path
            if isinstance(reference_image, str):
                if os.path.exists(reference_image):
                    reference_image = load_image(reference_image)
                else:
                    logger.warning(f"Reference image not found: {reference_image}")
                    return pipeline
            
            # Load the IP Adapter model
            ip_adapter = IPAdapterModel.from_pretrained(
                ip_adapter_path,
                torch_dtype=torch.float32,
                device_map=self.device
            )
            
            # Process the reference image
            processor = IPAdapterProcessor()
            image_embeds = processor(reference_image)
            
            # Apply the IP Adapter to the pipeline
            pipeline.set_ip_adapter(ip_adapter, image_embeds, weight=weight)
            
            logger.info(f"Successfully applied IP Adapter: {ip_adapter_name}")
            return pipeline
            
        except ImportError:
            logger.error("IP Adapter not installed. Please install it with: pip install diffusers[ip_adapter]")
            return pipeline
        except Exception as e:
            logger.error(f"Error applying IP Adapter {ip_adapter_name}: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            return pipeline
