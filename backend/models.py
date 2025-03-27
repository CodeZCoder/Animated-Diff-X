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
        ip_adapter_dir: str = None,
        clip_vision_dir: str = None,
        controlnet_dir: str = None,
        sd15_lora_dir: str = None,
        use_cpu: bool = True,
        optimize_memory: bool = True,
        quantization_type: str = "int8"
    ):
        self.models_dir = models_dir
        self.motion_module_dir = motion_module_dir
        self.motion_lora_dir = motion_lora_dir
        self.sd_dir = sd_dir
        self.ip_adapter_dir = ip_adapter_dir
        self.clip_vision_dir = clip_vision_dir
        self.controlnet_dir = controlnet_dir
        self.sd15_lora_dir = sd15_lora_dir
        self.use_cpu = use_cpu
        self.optimize_memory = optimize_memory
        self.quantization_type = quantization_type
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.motion_module_dir, exist_ok=True)
        os.makedirs(self.motion_lora_dir, exist_ok=True)
        os.makedirs(self.sd_dir, exist_ok=True)
        if self.ip_adapter_dir:
            os.makedirs(self.ip_adapter_dir, exist_ok=True)
        if self.clip_vision_dir:
            os.makedirs(self.clip_vision_dir, exist_ok=True)
        if self.controlnet_dir:
            os.makedirs(self.controlnet_dir, exist_ok=True)
        if self.sd15_lora_dir:
            os.makedirs(self.sd15_lora_dir, exist_ok=True)
        
        # Cache for loaded models
        self.sd_models_cache = {}
        self.motion_modules_cache = {}
        self.motion_loras_cache = {}
        self.ip_adapters_cache = {}
        self.controlnets_cache = {}
        
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
        sd15_models = ["runwayml/stable-diffusion-v1-5", "CompVis/stable-diffusion-v1-4"]
        sdxl_models = ["stabilityai/stable-diffusion-xl-base-1.0", "none"]
        
        return sd15_models + sdxl_models + model_names
    
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
    
    def get_sd15_loras(self) -> List[str]:
        """Get list of available SD 1.5 LoRAs"""
        if not self.sd15_lora_dir:
            return ["none"]
            
        loras = ["none"]  # Always include "none" option
        for ext in ["*.safetensors", "*.ckpt", "*.pt"]:
            try:
                loras.extend([os.path.basename(f) for f in glob.glob(os.path.join(self.sd15_lora_dir, ext))])
            except:
                pass
        
        return loras
    
    def get_ip_adapters(self) -> List[str]:
        """Get list of available IP Adapter models"""
        if not self.ip_adapter_dir:
            return ["none"]
            
        adapters = ["none"]  # Always include "none" option
        for ext in ["*.safetensors", "*.bin", "*.pt"]:
            try:
                adapters.extend([os.path.basename(f) for f in glob.glob(os.path.join(self.ip_adapter_dir, ext))])
            except:
                pass
        
        return adapters
    
    def get_controlnets(self) -> List[str]:
        """Get list of available ControlNet models"""
        if not self.controlnet_dir:
            return ["none"]
            
        controlnets = ["none"]  # Always include "none" option
        # Added .pth extension to the list of supported extensions
        for ext in ["*.safetensors", "*.bin", "*.pt", "*.pth"]:
            try:
                # Get all files with this extension
                files = glob.glob(os.path.join(self.controlnet_dir, ext))
                # Filter out .yaml files and any other non-model files
                model_files = [f for f in files if not f.endswith('.yaml')]
                controlnets.extend([os.path.basename(f) for f in model_files])
            except Exception as e:
                logger.warning(f"Error scanning for ControlNet models with extension {ext}: {str(e)}")
                pass
        
        # Log the found models for debugging
        logger.info(f"Found ControlNet models: {controlnets}")
        return controlnets
    
    def get_lora_path(self, lora_name: str) -> Optional[str]:
        """Get the path to a LoRA file"""
        if not lora_name or lora_name.lower() == "none":
            return None
            
        # Check if it's a full path
        if os.path.exists(lora_name):
            return lora_name
            
        # Check in the LoRA directory
        lora_path = os.path.join(self.motion_lora_dir, lora_name)
        if os.path.exists(lora_path):
            return lora_path
            
        # Try common extensions if not found
        for ext in [".safetensors", ".pt", ".ckpt"]:
            if not lora_name.endswith(ext):
                test_path = os.path.join(self.motion_lora_dir, lora_name + ext)
                if os.path.exists(test_path):
                    return test_path
                    
        return None
    
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
                
    def load_motion_module(self, module_name: str) -> torch.nn.Module:
        """Load a motion module
        
        Args:
            module_name: Name of the motion module to load
            
        Returns:
            Loaded motion module as torch.nn.Module
        """
        if module_name in self.motion_modules_cache:
            logger.info(f"Using cached motion module: {module_name}")
            return self.motion_modules_cache[module_name]
        
        logger.info(f"Loading motion module: {module_name}")
        
        # Determine the path to the motion module
        module_path = module_name if os.path.exists(module_name) else os.path.join(self.motion_module_dir, module_name)
        
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Motion module file not found: {module_path}")
        
        try:
            # Load the motion module based on file extension
            if module_path.endswith('.safetensors'):
                # Load safetensors file
                state_dict = load_file(module_path)
            else:
                # Load regular checkpoint file
                state_dict = torch.load(module_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
            
            # Create a simple container module to hold the state dict
            motion_module = torch.nn.Module()
            motion_module.load_state_dict = lambda _: None  # Placeholder
            motion_module.state_dict = lambda: state_dict
            
            # Cache the loaded module
            self.motion_modules_cache[module_name] = motion_module
            return motion_module
            
        except Exception as e:
            logger.error(f"Error loading motion module {module_path}: {str(e)}")
            raise
            
    def load_motion_lora(self, lora_name: str) -> torch.nn.Module:
        """Load a motion LoRA
        
        Args:
            lora_name: Name of the motion LoRA to load
            
        Returns:
            Loaded motion LoRA as torch.nn.Module
        """
        if lora_name in self.motion_loras_cache:
            logger.info(f"Using cached motion LoRA: {lora_name}")
            return self.motion_loras_cache[lora_name]
        
        logger.info(f"Loading motion LoRA: {lora_name}")
        
        # Get the path to the LoRA file
        lora_path = self.get_lora_path(lora_name)
        
        if not lora_path:
            raise FileNotFoundError(f"Motion LoRA file not found: {lora_name}")
        
        try:
            # Load the LoRA based on file extension
            if lora_path.endswith('.safetensors'):
                # Load safetensors file
                state_dict = load_file(lora_path)
            else:
                # Load regular checkpoint file
                state_dict = torch.load(lora_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
            
            # Create a simple container module to hold the state dict
            motion_lora = torch.nn.Module()
            motion_lora.load_state_dict = lambda _: None  # Placeholder
            motion_lora.state_dict = lambda: state_dict
            
            # Cache the loaded LoRA
            self.motion_loras_cache[lora_name] = motion_lora
            return motion_lora
            
        except Exception as e:
            logger.error(f"Error loading motion LoRA {lora_path}: {str(e)}")
            raise
