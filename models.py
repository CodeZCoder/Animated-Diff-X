#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import logging
import torch
from typing import List, Dict, Optional, Union
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from huggingface_hub import snapshot_download

from backend.optimization import optimize_model

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages loading and caching of models for AnimateDiff"""
    
    def __init__(
        self,
        models_dir: str,
        motion_module_dir: str,
        motion_lora_dir: str,
        sd_dir: str,
        use_cpu: bool = True,
        optimize_memory: bool = True,
        quantization_type: str = "int8"
    ):
        self.models_dir = models_dir
        self.motion_module_dir = motion_module_dir
        self.motion_lora_dir = motion_lora_dir
        self.sd_dir = sd_dir
        self.use_cpu = use_cpu
        self.optimize_memory = optimize_memory
        self.quantization_type = quantization_type
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.motion_module_dir, exist_ok=True)
        os.makedirs(self.motion_lora_dir, exist_ok=True)
        os.makedirs(self.sd_dir, exist_ok=True)
        
        # Cache for loaded models
        self.sd_models_cache = {}
        self.motion_modules_cache = {}
        self.motion_loras_cache = {}
        
        # Device configuration
        self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")
        
        # Set torch threads for CPU optimization
        if self.use_cpu:
            torch.set_num_threads(os.cpu_count() or 4)
            logger.info(f"Set torch threads to {torch.get_num_threads()}")
    
    def get_sd_models(self) -> List[str]:
        """Get list of available Stable Diffusion models"""
        # Check local models
        local_models = []
        for ext in ["*.safetensors", "*.ckpt", "*.pt"]:
            local_models.extend(glob.glob(os.path.join(self.sd_dir, ext)))
        
        # Get model names without path and extension
        model_names = [os.path.basename(model) for model in local_models]
        
        # Add default HuggingFace models
        hf_models = ["runwayml/stable-diffusion-v1-5", "CompVis/stable-diffusion-v1-4"]
        
        return hf_models + model_names
    
    def get_motion_modules(self) -> List[str]:
        """Get list of available motion modules"""
        modules = []
        for ext in ["*.safetensors", "*.ckpt", "*.pt"]:
            modules.extend(glob.glob(os.path.join(self.motion_module_dir, ext)))
        
        return [os.path.basename(module) for module in modules]
    
    def get_motion_loras(self) -> List[str]:
        """Get list of available motion LoRAs"""
        loras = []
        for ext in ["*.safetensors", "*.ckpt", "*.pt"]:
            loras.extend(glob.glob(os.path.join(self.motion_lora_dir, ext)))
        
        return [os.path.basename(lora) for lora in loras]
    
    def load_sd_model(self, model_name: str) -> StableDiffusionPipeline:
        """Load a Stable Diffusion model"""
        if model_name in self.sd_models_cache:
            logger.info(f"Using cached SD model: {model_name}")
            return self.sd_models_cache[model_name]
        
        logger.info(f"Loading SD model: {model_name}")
        
        # Check if it's a HuggingFace model or local file
        if "/" in model_name and not os.path.exists(model_name):
            # HuggingFace model
            try:
                # First load as standard StableDiffusionPipeline
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    safety_checker=None,  # Disable safety checker for uncensored mode
                    requires_safety_checker=False,  # Disable safety checker for uncensored mode
                    cache_dir=self.sd_dir,  # Use application's models directory
                    local_files_only=False  # Allow downloading if not found locally
                )
                
                # Optimize the model for CPU if needed
                if self.optimize_memory:
                    pipeline = optimize_model(pipeline, self.quantization_type)
                
                self.sd_models_cache[model_name] = pipeline
                return pipeline
            
            except Exception as e:
                logger.error(f"Error loading HuggingFace model {model_name}: {str(e)}")
                raise
        else:
            # Local model file
            model_path = model_name if os.path.exists(model_name) else os.path.join(self.sd_dir, model_name)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            try:
                # Load with CPU optimization
                pipeline = StableDiffusionPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    safety_checker=None,  # Disable safety checker for uncensored mode
                    requires_safety_checker=False  # Disable safety checker for uncensored mode
                )
                
                # Optimize the model for CPU if needed
                if self.optimize_memory:
                    pipeline = optimize_model(pipeline, self.quantization_type)
                
                self.sd_models_cache[model_name] = pipeline
                return pipeline
            
            except Exception as e:
                logger.error(f"Error loading local model {model_path}: {str(e)}")
                raise
    
    def load_motion_module(self, module_name: str) -> Dict[str, torch.Tensor]:
        """Load a motion module"""
        if module_name in self.motion_modules_cache:
            logger.info(f"Using cached motion module: {module_name}")
            return self.motion_modules_cache[module_name]
        
        logger.info(f"Loading motion module: {module_name}")
        
        module_path = os.path.join(self.motion_module_dir, module_name)
        
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Motion module not found: {module_path}")
        
        try:
            # Load the motion module
            if module_path.endswith('.safetensors'):
                state_dict = load_file(module_path)
            else:
                state_dict = torch.load(module_path, map_location=self.device)
            
            # Convert to CPU and optimize if needed
            if self.optimize_memory:
                for key in state_dict:
                    if isinstance(state_dict[key], torch.Tensor):
                        # Use lower precision for CPU to save memory
                        if self.quantization_type == "int8":
                            # This is a simplified approach - in production you'd use proper quantization
                            state_dict[key] = state_dict[key].to(torch.int8).to(self.device)
                        elif self.quantization_type == "fp16":
                            state_dict[key] = state_dict[key].to(torch.float16).to(self.device)
                        else:
                            state_dict[key] = state_dict[key].to(self.device)
            
            self.motion_modules_cache[module_name] = state_dict
            return state_dict
        
        except Exception as e:
            logger.error(f"Error loading motion module {module_path}: {str(e)}")
            raise
    
    def load_motion_lora(self, lora_name: str) -> Dict[str, torch.Tensor]:
        """Load a motion LoRA"""
        if not lora_name or lora_name.lower() == "none":
            return None
            
        if lora_name in self.motion_loras_cache:
            logger.info(f"Using cached motion LoRA: {lora_name}")
            return self.motion_loras_cache[lora_name]
        
        logger.info(f"Loading motion LoRA: {lora_name}")
        
        lora_path = os.path.join(self.motion_lora_dir, lora_name)
        
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"Motion LoRA not found: {lora_path}")
        
        try:
            # Load the motion LoRA
            if lora_path.endswith('.safetensors'):
                state_dict = load_file(lora_path)
            else:
                state_dict = torch.load(lora_path, map_location=self.device)
            
            # Convert to CPU and optimize if needed
            if self.optimize_memory:
                for key in state_dict:
                    if isinstance(state_dict[key], torch.Tensor):
                        # Use lower precision for CPU to save memory
                        if self.quantization_type == "int8":
                            # This is a simplified approach - in production you'd use proper quantization
                            state_dict[key] = state_dict[key].to(torch.int8).to(self.device)
                        elif self.quantization_type == "fp16":
                            state_dict[key] = state_dict[key].to(torch.float16).to(self.device)
                        else:
                            state_dict[key] = state_dict[key].to(self.device)
            
            self.motion_loras_cache[lora_name] = state_dict
            return state_dict
        
        except Exception as e:
            logger.error(f"Error loading motion LoRA {lora_path}: {str(e)}")
            raise
    
    def clear_cache(self):
        """Clear model caches to free memory"""
        logger.info("Clearing model caches")
        self.sd_models_cache.clear()
        self.motion_modules_cache.clear()
        self.motion_loras_cache.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
