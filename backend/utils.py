#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Callable
import base64
import io
from PIL import Image

logger = logging.getLogger(__name__)

def save_image_base64(image_data: str, output_path: str) -> str:
    """
    Save a base64 encoded image to a file
    
    Args:
        image_data: Base64 encoded image data
        output_path: Path to save the image
        
    Returns:
        Path to the saved image
    """
    try:
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64 data
        image_bytes = base64.b64decode(image_data)
        
        # Open as PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save the image
        image.save(output_path)
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error saving base64 image: {str(e)}")
        raise

def image_to_base64(image_path: str) -> str:
    """
    Convert an image file to base64 encoded data URL
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded data URL
    """
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
            
        # Encode as base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Determine MIME type
        mime_type = "image/jpeg"
        if image_path.lower().endswith('.png'):
            mime_type = "image/png"
        elif image_path.lower().endswith('.gif'):
            mime_type = "image/gif"
        
        # Create data URL
        data_url = f"data:{mime_type};base64,{image_base64}"
        
        return data_url
    
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        raise

def numpy_to_base64(image: np.ndarray, format: str = 'jpeg') -> str:
    """
    Convert a numpy array image to base64 encoded data URL
    
    Args:
        image: Numpy array image
        format: Image format (jpeg, png)
        
    Returns:
        Base64 encoded data URL
    """
    try:
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format.upper())
        buffer.seek(0)
        
        # Encode as base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create data URL
        mime_type = f"image/{format.lower()}"
        data_url = f"data:{mime_type};base64,{image_base64}"
        
        return data_url
    
    except Exception as e:
        logger.error(f"Error converting numpy array to base64: {str(e)}")
        raise

def create_directory_if_not_exists(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist
    
    Args:
        directory_path: Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Created directory: {directory_path}")
