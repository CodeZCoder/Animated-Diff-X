"""
Model compatibility checking for AnimateDiff.

This module provides functions to check compatibility between different types of models:
- Motion modules (mm_sd_v15_v2.ckpt, v3_sd15_mm.ckpt, etc.)
- Base SD models (SD1.5, SDXL, etc.)
- Motion LoRAs

It also provides suggestions for compatible models when incompatible combinations are detected.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Define model compatibility information
SD_MODEL_TYPES = {
    "SD1.5": [
        "runwayml/stable-diffusion-v1-5",
        "dreamlike-art/dreamlike-photoreal-2.0",
        "CompVis/stable-diffusion-v1-4",
        "stabilityai/stable-diffusion-v1-5",
        "SG161222/Realistic_Vision_V5.1_noVAE",
        "Lykon/DreamShaper",
        "Lykon/AbsoluteReality",
        "Lykon/NeverEnding_Dream",
        "Yntec/epiCRealism",
        "digiplay/majicMIX_realistic_v6",
        "Linaqruf/anything-v3.0",
        "hassanblend/hassanblend1.5.1.2",
    ],
    "SDXL": [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "Lykon/dreamshaper-xl-1.0",
        "Lykon/absolute-reality-xl-1.0",
        "cagliostrolab/animagine-xl-3.0",
        "digiplay/majicmix_realistic_xl_v1",
        "segmind/SSD-1B",
        "playgroundai/playground-v2.5-1024px-aesthetic",
    ],
}

MOTION_MODULE_TYPES = {
    "SD1.5_V1": [
        "animatediff_v1_5.safetensors",
        "mm_sd_v15.ckpt",
    ],
    "SD1.5_V2": [
        "mm_sd_v15_v2.ckpt",
    ],
    "SD1.5_V3": [
        "v3_sd15_mm.ckpt",
    ],
    "SDXL": [
        "mm_sdxl_v10_beta.ckpt",
        "mm_sdxl_v1.0.safetensors",
        "mm_sdxl_v1.0_beta.ckpt",
    ],
}

MOTION_LORA_COMPATIBILITY = {
    "PanLeft": ["SD1.5_V2"],
    "PanRight": ["SD1.5_V2"],
    "RollingAnticlockwise": ["SD1.5_V2"],
    "RollingClockwise": ["SD1.5_V2"],
    "TiltDown": ["SD1.5_V2"],
    "TiltUp": ["SD1.5_V2"],
    "ZoomIn": ["SD1.5_V2"],
    "ZoomOut": ["SD1.5_V2"],
    "Wiggle": ["SD1.5_V2", "SD1.5_V3"],
    "Dance": ["SD1.5_V2", "SD1.5_V3"],
    "Bouncing": ["SD1.5_V2", "SD1.5_V3"],
    "Shatter": ["SD1.5_V2", "SD1.5_V3"],
}

def get_sd_model_type(sd_model_name: str) -> str:
    """
    Determine the type of SD model (SD1.5, SDXL, etc.) based on the model name.
    
    Args:
        sd_model_name: Name of the SD model
        
    Returns:
        Type of SD model ("SD1.5", "SDXL", or "UNKNOWN")
    """
    for model_type, model_list in SD_MODEL_TYPES.items():
        for model in model_list:
            if model.lower() in sd_model_name.lower():
                return model_type
    
    # Try to determine type from name patterns
    if any(x in sd_model_name.lower() for x in ["xl", "sdxl"]):
        return "SDXL"
    if any(x in sd_model_name.lower() for x in ["sd15", "sd-15", "sd1.5", "sd-1.5", "v1.5", "v1-5"]):
        return "SD1.5"
    
    return "UNKNOWN"

def get_motion_module_type(motion_module_name: str) -> str:
    """
    Determine the type of motion module based on the module name.
    
    Args:
        motion_module_name: Name of the motion module
        
    Returns:
        Type of motion module ("SD1.5_V1", "SD1.5_V2", "SD1.5_V3", "SDXL", or "UNKNOWN")
    """
    for module_type, module_list in MOTION_MODULE_TYPES.items():
        for module in module_list:
            if module.lower() in motion_module_name.lower():
                return module_type
    
    # Try to determine type from name patterns
    if any(x in motion_module_name.lower() for x in ["sdxl", "xl"]):
        return "SDXL"
    if "v3" in motion_module_name.lower():
        return "SD1.5_V3"
    if "v2" in motion_module_name.lower():
        return "SD1.5_V2"
    if "v1" in motion_module_name.lower():
        return "SD1.5_V1"
    
    return "UNKNOWN"

def get_motion_lora_type(motion_lora_name: str) -> str:
    """
    Determine the type of motion LoRA based on the LoRA name.
    
    Args:
        motion_lora_name: Name of the motion LoRA
        
    Returns:
        Type of motion LoRA or "UNKNOWN"
    """
    if not motion_lora_name or motion_lora_name.lower() == "none":
        return "NONE"
    
    for lora_type in MOTION_LORA_COMPATIBILITY.keys():
        if lora_type.lower() in motion_lora_name.lower():
            return lora_type
    
    # Check for common patterns in LoRA names
    if any(x in motion_lora_name.lower() for x in ["zoom", "in"]):
        return "ZoomIn"
    if any(x in motion_lora_name.lower() for x in ["zoom", "out"]):
        return "ZoomOut"
    if any(x in motion_lora_name.lower() for x in ["pan", "left"]):
        return "PanLeft"
    if any(x in motion_lora_name.lower() for x in ["pan", "right"]):
        return "PanRight"
    if any(x in motion_lora_name.lower() for x in ["tilt", "up"]):
        return "TiltUp"
    if any(x in motion_lora_name.lower() for x in ["tilt", "down"]):
        return "TiltDown"
    if any(x in motion_lora_name.lower() for x in ["roll", "clock"]):
        return "RollingClockwise"
    if any(x in motion_lora_name.lower() for x in ["roll", "anti"]):
        return "RollingAnticlockwise"
    if "wiggle" in motion_lora_name.lower():
        return "Wiggle"
    if "dance" in motion_lora_name.lower():
        return "Dance"
    if "bounce" in motion_lora_name.lower():
        return "Bouncing"
    if "shatter" in motion_lora_name.lower():
        return "Shatter"
    
    return "UNKNOWN"

def check_model_compatibility(
    sd_model_name: str,
    motion_module_name: str,
    motion_lora_name: Optional[str] = None
) -> Tuple[bool, List[str], Dict[str, List[str]]]:
    """
    Check compatibility between SD model, motion module, and motion LoRA.
    
    Args:
        sd_model_name: Name of the SD model
        motion_module_name: Name of the motion module
        motion_lora_name: Name of the motion LoRA (optional)
        
    Returns:
        Tuple containing:
        - Boolean indicating if models are compatible
        - List of warning/error messages
        - Dictionary of suggested compatible alternatives
    """
    messages = []
    suggestions = {}
    is_compatible = True
    
    # Determine model types
    sd_type = get_sd_model_type(sd_model_name)
    motion_type = get_motion_module_type(motion_module_name)
    lora_type = get_motion_lora_type(motion_lora_name) if motion_lora_name else "NONE"
    
    logger.info(f"SD model type: {sd_type}")
    logger.info(f"Motion module type: {motion_type}")
    logger.info(f"Motion LoRA type: {lora_type}")
    
    # Check SD model and motion module compatibility
    if sd_type == "UNKNOWN":
        messages.append(f"Unknown SD model type: {sd_model_name}")
        suggestions["sd_model"] = SD_MODEL_TYPES["SD1.5"][:3]  # Suggest a few common SD1.5 models
    
    if motion_type == "UNKNOWN":
        messages.append(f"Unknown motion module type: {motion_module_name}")
        suggestions["motion_module"] = MOTION_MODULE_TYPES["SD1.5_V2"] + MOTION_MODULE_TYPES["SD1.5_V3"]
    
    # Check SD model and motion module compatibility
    if sd_type == "SD1.5" and "SDXL" in motion_type:
        is_compatible = False
        messages.append(f"Motion module '{motion_module_name}' is intended for SDXL models, but '{sd_model_name}' is a SD1.5 model.")
        suggestions["motion_module"] = MOTION_MODULE_TYPES["SD1.5_V2"] + MOTION_MODULE_TYPES["SD1.5_V3"]
    
    if sd_type == "SDXL" and "SD1.5" in motion_type:
        is_compatible = False
        messages.append(f"Motion module '{motion_module_name}' is intended for SD1.5 models, but '{sd_model_name}' is an SDXL model.")
        suggestions["motion_module"] = MOTION_MODULE_TYPES["SDXL"]
    
    # Check motion LoRA compatibility
    if lora_type != "NONE" and lora_type != "UNKNOWN":
        compatible_modules = MOTION_LORA_COMPATIBILITY.get(lora_type, [])
        if motion_type not in compatible_modules:
            is_compatible = False
            messages.append(f"Motion LoRA '{motion_lora_name}' is not compatible with motion module '{motion_module_name}'.")
            
            # Suggest compatible motion modules for this LoRA
            compatible_module_names = []
            for module_type in compatible_modules:
                compatible_module_names.extend(MOTION_MODULE_TYPES.get(module_type, []))
            
            if compatible_module_names:
                suggestions["motion_module"] = compatible_module_names
    
    # If LoRA type is unknown, provide a general warning
    if lora_type == "UNKNOWN" and motion_lora_name and motion_lora_name.lower() != "none":
        messages.append(f"Unknown motion LoRA type: {motion_lora_name}. Most motion LoRAs work best with V2 motion modules like mm_sd_v15_v2.ckpt.")
    
    return is_compatible, messages, suggestions

def get_compatible_models(model_type: str, model_name: str) -> List[str]:
    """
    Get a list of compatible models for a given model.
    
    Args:
        model_type: Type of model ("sd_model", "motion_module", or "motion_lora")
        model_name: Name of the model
        
    Returns:
        List of compatible model names
    """
    if model_type == "sd_model":
        sd_type = get_sd_model_type(model_name)
        if sd_type == "SD1.5":
            return MOTION_MODULE_TYPES["SD1.5_V2"] + MOTION_MODULE_TYPES["SD1.5_V3"]
        elif sd_type == "SDXL":
            return MOTION_MODULE_TYPES["SDXL"]
    
    elif model_type == "motion_module":
        motion_type = get_motion_module_type(model_name)
        if "SD1.5" in motion_type:
            return SD_MODEL_TYPES["SD1.5"]
        elif motion_type == "SDXL":
            return SD_MODEL_TYPES["SDXL"]
        
        # Also return compatible LoRAs
        compatible_loras = []
        for lora_type, compatible_modules in MOTION_LORA_COMPATIBILITY.items():
            if motion_type in compatible_modules:
                compatible_loras.append(lora_type)
        return compatible_loras
    
    elif model_type == "motion_lora":
        lora_type = get_motion_lora_type(model_name)
        compatible_modules = []
        for module_type in MOTION_LORA_COMPATIBILITY.get(lora_type, []):
            compatible_modules.extend(MOTION_MODULE_TYPES.get(module_type, []))
        return compatible_modules
    
    return []
