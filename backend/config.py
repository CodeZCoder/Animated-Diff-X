#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration for AnimateDiff GUI"""
    
    # Base directories
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir: str = os.path.join(base_dir, "models")
    output_dir: str = os.path.join(base_dir, "outputs")
    
    # Model directories
    sd_dir: str = os.path.join(models_dir, "stable_diffusion")
    motion_module_dir: str = os.path.join(models_dir, "motion_module")
    motion_lora_dir: str = os.path.join(models_dir, "motion_lora")
    
    # Default models
    default_sd_model: str = "runwayml/stable-diffusion-v1-5"
    default_motion_module: str = "v3_sd15_mm.safetensors"
    
    # Generation settings
    default_width: int = 512
    default_height: int = 512
    default_num_frames: int = 16
    default_fps: int = 8
    default_guidance_scale: float = 7.5
    default_num_inference_steps: int = 25
    
    # CPU optimization settings
    cpu_threads: int = os.cpu_count() or 4
    quantization_type: str = "int8"  # Options: "fp16", "int8"
    optimize_memory: bool = True
    batch_size: int = 1
    
    # Uncensored settings
    enable_uncensored: bool = True
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.sd_dir, exist_ok=True)
        os.makedirs(self.motion_module_dir, exist_ok=True)
        os.makedirs(self.motion_lora_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Config initialized with base_dir: {self.base_dir}")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"CPU optimization: quantization={self.quantization_type}, threads={self.cpu_threads}")
        logger.info(f"Uncensored mode: {self.enable_uncensored}")
