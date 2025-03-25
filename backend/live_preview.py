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

class LivePreviewManager:
    """Manages live preview updates during video generation"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.current_frames = []
        self.current_step = 0
        self.total_steps = 0
        self.generation_id = None
        self.callbacks = []
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def register_callback(self, callback: Callable[[str, int, int, List[str]], None]):
        """Register a callback function for live preview updates"""
        self.callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable):
        """Unregister a callback function"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def start_generation(self, generation_id: str, total_steps: int):
        """Start a new generation process"""
        self.generation_id = generation_id
        self.current_frames = []
        self.current_step = 0
        self.total_steps = total_steps
        logger.info(f"Starting generation {generation_id} with {total_steps} steps")
    
    def update_preview(self, frame: np.ndarray, step: int, total_steps: int):
        """
        Update the preview with a new frame
        
        Args:
            frame: The frame image as a numpy array
            step: Current generation step
            total_steps: Total number of generation steps
        """
        if self.generation_id is None:
            logger.warning("No active generation process")
            return
        
        self.current_step = step
        self.total_steps = total_steps
        
        # Save the frame as an image
        frame_path = os.path.join(self.output_dir, f"{self.generation_id}_step_{step}.jpg")
        Image.fromarray(frame).save(frame_path)
        
        # Add to current frames
        self.current_frames.append(frame_path)
        
        # Convert frame to base64 for web preview
        buffered = io.BytesIO()
        Image.fromarray(frame).save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        data_url = f"data:image/jpeg;base64,{img_str}"
        
        # Notify all callbacks
        for callback in self.callbacks:
            try:
                callback(self.generation_id, step, total_steps, data_url)
            except Exception as e:
                logger.error(f"Error in preview callback: {str(e)}")
    
    def get_current_preview(self) -> Optional[str]:
        """Get the current preview image as a data URL"""
        if not self.current_frames:
            return None
        
        # Get the latest frame
        latest_frame_path = self.current_frames[-1]
        
        # Convert to data URL
        with open(latest_frame_path, "rb") as f:
            img_data = f.read()
            img_str = base64.b64encode(img_data).decode()
            return f"data:image/jpeg;base64,{img_str}"
    
    def get_progress(self) -> Dict[str, Union[int, float]]:
        """Get the current generation progress"""
        if self.total_steps == 0:
            progress = 0
        else:
            progress = (self.current_step / self.total_steps) * 100
            
        return {
            "generation_id": self.generation_id,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress": progress
        }
    
    def end_generation(self):
        """End the current generation process"""
        self.generation_id = None
        self.current_frames = []
        self.current_step = 0
        self.total_steps = 0
        logger.info("Generation process ended")
