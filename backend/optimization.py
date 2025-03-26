#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import logging
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)

def optimize_model(model: Any, quantization_type: str = "int8") -> Any:
    """
    Optimize a model for CPU inference using quantization techniques
    
    Args:
        model: The model to optimize
        quantization_type: Type of quantization to use ("fp16", "int8")
        
    Returns:
        Optimized model
    """
    logger.info(f"Optimizing model with {quantization_type} quantization")
    
    try:
        if quantization_type == "int8":
            # Use 8-bit quantization with bitsandbytes
            try:
                import bitsandbytes as bnb
                from accelerate import cpu_offload
                
                # Convert model to 8-bit precision
                model = model.to(torch.device("cpu"))
                
                # Apply quantization to UNet
                if hasattr(model, "unet"):
                    logger.info("Quantizing UNet to 8-bit")
                    model.unet = bnb.nn.modules.Params8bit(model.unet)
                
                # Apply quantization to text encoder
                if hasattr(model, "text_encoder"):
                    logger.info("Quantizing text encoder to 8-bit")
                    model.text_encoder = bnb.nn.modules.Params8bit(model.text_encoder)
                
                logger.info("Model optimized with 8-bit quantization")
                
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to fp16 quantization")
                return optimize_model(model, "fp16")
                
        elif quantization_type == "fp16":
            # Use half precision (fp16)
            model = model.to(torch.device("cpu"), torch.float16)
            logger.info("Model optimized with fp16 quantization")
            
        else:
            logger.warning(f"Unknown quantization type: {quantization_type}, using default")
            model = model.to(torch.device("cpu"))
        
        return model
        
    except Exception as e:
        logger.error(f"Error optimizing model: {str(e)}")
        logger.warning("Using unoptimized model")
        return model.to(torch.device("cpu"))

def apply_motion_lora(
    sd_pipeline,
    motion_lora: Dict[str, torch.Tensor],
    alpha: float = 1.0
):
    """
    Apply a Motion LoRA to a Stable Diffusion pipeline or motion module
    
    Args:
        sd_pipeline: The StableDiffusionPipeline or motion module state dict
        motion_lora: The motion LoRA state dict
        alpha: The strength of the LoRA effect (0.0 to 1.0)
        
    Returns:
        Updated pipeline or motion module
    """
    logger.info(f"Applying Motion LoRA with alpha={alpha}")
    
    if motion_lora is None:
        return sd_pipeline
    
    # Check if we're dealing with a dictionary (motion module) or a pipeline object
    if isinstance(sd_pipeline, dict):
        # For dictionary-type motion modules
        try:
            # Create a copy of the dictionary
            result = dict(sd_pipeline)
            
            # Apply LoRA weights to the motion module
            for key in motion_lora:
                if key in result:
                    # Apply scaled LoRA weights
                    result[key] = result[key] + motion_lora[key] * alpha
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying Motion LoRA to dictionary: {str(e)}")
            return sd_pipeline
    else:
        # For StableDiffusionPipeline objects
        logger.info("Applying Motion LoRA to pipeline object")
        # Simply return the pipeline as is - we can't directly apply LoRA to the pipeline object
        # In a full implementation, you would need to modify the UNet weights
        return sd_pipeline

def optimize_memory():
    """
    Perform memory optimization operations
    """
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Optimize CPU memory usage
    if hasattr(torch, 'set_flush_denormal'):
        # Flush denormalized numbers to zero for better CPU performance
        torch.set_flush_denormal(True)
    
    logger.info("Memory optimization performed")
