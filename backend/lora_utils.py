#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import torch
import importlib
from typing import Optional, Union, List
from diffusers import StableDiffusionPipeline

logger = logging.getLogger(__name__)

def apply_sd15_lora(
    pipeline: StableDiffusionPipeline,
    lora_name: str,
    lora_strength: float = 1.0,
    lora_dir: Optional[str] = None
) -> StableDiffusionPipeline:
    """
    Apply a SD 1.5 LoRA to a Stable Diffusion pipeline
    
    Args:
        pipeline: The StableDiffusionPipeline to apply the LoRA to
        lora_name: Name or path of the LoRA file
        lora_strength: Strength of the LoRA effect (0.0 to 1.0)
        lora_dir: Directory containing LoRA files (optional)
        
    Returns:
        The modified StableDiffusionPipeline
    """
    try:
        # Skip if lora_name is None or "none"
        if not lora_name or lora_name.lower() == "none":
            logger.info("No LoRA specified, skipping")
            return pipeline
            
        # Validate strength
        if lora_strength <= 0:
            logger.info(f"LoRA strength is {lora_strength}, skipping")
            return pipeline
            
        # Find the LoRA file
        lora_path = lora_name
        
        # If it's not a full path and lora_dir is provided, look in lora_dir
        if not os.path.exists(lora_path) and lora_dir:
            # Try direct path
            test_path = os.path.join(lora_dir, lora_name)
            if os.path.exists(test_path):
                lora_path = test_path
            else:
                # Try with common extensions
                for ext in [".safetensors", ".pt", ".ckpt", ".bin"]:
                    if not lora_name.endswith(ext):
                        test_path = os.path.join(lora_dir, lora_name + ext)
                        if os.path.exists(test_path):
                            lora_path = test_path
                            break
        
        # Check if file exists
        if not os.path.exists(lora_path):
            logger.warning(f"LoRA file not found: {lora_path}")
            return pipeline
            
        logger.info(f"Loading LoRA from: {lora_path}")
        
        # Load the LoRA weights
        try:
            # More robust LoRA loading approach
            # First try the direct load_lora_weights method (newest diffusers versions)
            if hasattr(pipeline, "load_lora_weights") and hasattr(pipeline, "set_adapters"):
                pipeline.load_lora_weights(lora_path, adapter_name="default")
                if lora_strength < 1.0:
                    pipeline.set_adapters(["default"], adapter_weights=[lora_strength])
                logger.info(f"Successfully loaded LoRA with strength {lora_strength}")
            # Second try the direct load_lora_weights method without set_adapters
            elif hasattr(pipeline, "load_lora_weights"):
                pipeline.load_lora_weights(lora_path)
                logger.info(f"Successfully loaded LoRA with strength {lora_strength} (fixed adapter strength)")
            # Third try using PEFT if available
            elif importlib.util.find_spec("peft") is not None:
                # Fallback for older diffusers versions with PEFT
                from peft import PeftModel
                
                try:
                    # Load PEFT model
                    text_encoder = PeftModel.from_pretrained(
                        pipeline.text_encoder,
                        lora_path,
                        adapter_name="default"
                    )
                    unet = PeftModel.from_pretrained(
                        pipeline.unet,
                        lora_path,
                        adapter_name="default"
                    )
                    
                    # Set adapter weights if strength < 1.0
                    if lora_strength < 1.0:
                        text_encoder.set_adapter_weight("default", lora_strength)
                        unet.set_adapter_weight("default", lora_strength)
                    
                    # Update pipeline components
                    pipeline.text_encoder = text_encoder
                    pipeline.unet = unet
                    
                    logger.info(f"Successfully loaded LoRA with strength {lora_strength} using PEFT")
                except Exception as peft_error:
                    logger.warning(f"Error loading LoRA with PEFT: {str(peft_error)}")
                    raise
            else:
                logger.warning("No compatible LoRA loading method available")
                raise ValueError("No compatible LoRA loading method available")
        except Exception as e:
            # Try alternative method for older models
            logger.warning(f"Error loading LoRA with primary method: {str(e)}")
            logger.info("Trying alternative LoRA loading method...")
            
            try:
                # Try loading with custom diffusers code
                state_dict = load_lora_state_dict(lora_path)
                pipeline = apply_lora_state_dict(pipeline, state_dict, lora_strength)
                logger.info(f"Successfully loaded LoRA with strength {lora_strength} using alternative method")
            except Exception as e2:
                logger.error(f"Error loading LoRA with alternative method: {str(e2)}")
                logger.warning("Continuing without LoRA")
                return pipeline
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Error applying LoRA: {str(e)}")
        logger.warning("Continuing without LoRA")
        return pipeline

def load_lora_state_dict(lora_path: str) -> dict:
    """
    Load a LoRA state dict from a file
    
    Args:
        lora_path: Path to the LoRA file
        
    Returns:
        The LoRA state dict
    """
    if lora_path.endswith(".safetensors"):
        # Load safetensors
        from safetensors.torch import load_file
        state_dict = load_file(lora_path)
    else:
        # Load torch format
        state_dict = torch.load(lora_path, map_location="cpu")
    
    # Handle different LoRA formats
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    return state_dict

def apply_lora_state_dict(
    pipeline: StableDiffusionPipeline,
    state_dict: dict,
    alpha: float = 1.0
) -> StableDiffusionPipeline:
    """
    Apply a LoRA state dict to a Stable Diffusion pipeline
    
    Args:
        pipeline: The StableDiffusionPipeline to apply the LoRA to
        state_dict: The LoRA state dict
        alpha: Strength of the LoRA effect (0.0 to 1.0)
        
    Returns:
        The modified StableDiffusionPipeline
    """
    # Get visited keys to avoid duplicates
    visited = []
    
    # Apply to text encoder
    for key in state_dict:
        if "text_encoder" in key:
            # Get the original layer name by removing prefixes
            if "lora_te_" in key:
                base_key = key.replace("lora_te_", "")
            else:
                base_key = key.replace("text_encoder.", "")
            
            # Skip if already visited
            if base_key in visited:
                continue
                
            visited.append(base_key)
            
            # Get up and down weights
            up_key = key
            down_key = key.replace("lora_up", "lora_down")
            
            # Check if keys exist
            if up_key not in state_dict or down_key not in state_dict:
                continue
                
            # Get weights
            up_weight = state_dict[up_key]
            down_weight = state_dict[down_key]
            
            # Calculate LoRA weight
            lora_weight = alpha * torch.mm(up_weight, down_weight)
            
            # Get original weight
            original_weight = pipeline.text_encoder.get_parameter(base_key)
            
            # Apply LoRA weight
            pipeline.text_encoder.get_parameter(base_key).data += lora_weight
    
    # Apply to unet
    visited = []
    for key in state_dict:
        if "unet" in key:
            # Get the original layer name by removing prefixes
            if "lora_unet_" in key:
                base_key = key.replace("lora_unet_", "")
            else:
                base_key = key.replace("unet.", "")
            
            # Skip if already visited
            if base_key in visited:
                continue
                
            visited.append(base_key)
            
            # Get up and down weights
            up_key = key
            down_key = key.replace("lora_up", "lora_down")
            
            # Check if keys exist
            if up_key not in state_dict or down_key not in state_dict:
                continue
                
            # Get weights
            up_weight = state_dict[up_key]
            down_weight = state_dict[down_key]
            
            # Calculate LoRA weight
            lora_weight = alpha * torch.mm(up_weight, down_weight)
            
            # Get original weight
            original_weight = pipeline.unet.get_parameter(base_key)
            
            # Apply LoRA weight
            pipeline.unet.get_parameter(base_key).data += lora_weight
    
    return pipeline
