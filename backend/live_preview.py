#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
import time
from typing import Dict, List, Optional, Union, Callable
import base64
import io
from PIL import Image
import uuid
import threading
from flask import Flask, request
from flask_socketio import SocketIO, emit

logger = logging.getLogger(__name__)

class LivePreviewManager:
    """Manages live preview updates during video generation"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.current_frames = []
        self.current_step = 0
        self.total_steps = 0
        self.current_frame = 0
        self.total_frames = 0
        self.start_time = None
        self.fps = 0
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
    
    def start_generation(self, generation_id: str, total_steps: int, total_frames: int = 16):
        """Start a new generation process"""
        self.generation_id = generation_id
        self.current_frames = []
        self.current_step = 0
        self.total_steps = total_steps
        self.current_frame = 0
        self.total_frames = total_frames
        self.start_time = time.time()
        self.fps = 0
        logger.info(f"Starting generation {generation_id} with {total_steps} steps and {total_frames} frames")
    
    def update_preview(self, frame: np.ndarray, step: int, total_steps: int, current_frame: int = None):
        """
        Update the preview with a new frame
        
        Args:
            frame: The frame image as a numpy array
            step: Current generation step
            total_steps: Total number of generation steps
            current_frame: Current frame number (optional)
        """
        if self.generation_id is None:
            logger.warning("No active generation process")
            return
        
        self.current_step = step
        self.total_steps = total_steps
        
        # Update frame tracking if provided
        if current_frame is not None:
            self.current_frame = current_frame
        else:
            # If not explicitly provided, increment frame counter
            self.current_frame += 1
        
        # Calculate FPS
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            self.fps = len(self.current_frames) / elapsed_time
        
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
        # Calculate steps progress
        if self.total_steps == 0:
            steps_progress = 0
        else:
            steps_progress = (self.current_step / self.total_steps) * 100
        
        # Calculate frames progress
        if self.total_frames == 0:
            frames_progress = 0
        else:
            frames_progress = (self.current_frame / self.total_frames) * 100
        
        # Calculate FPS progress (assuming target FPS is 30)
        target_fps = 30
        fps_progress = min(100, (self.fps / target_fps) * 100)
        
        # Calculate total progress (weighted average)
        total_progress = (steps_progress * 0.5) + (frames_progress * 0.3) + (fps_progress * 0.2)
            
        return {
            "generation_id": self.generation_id,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "steps_progress": steps_progress,
            "frames_progress": frames_progress,
            "fps_progress": fps_progress,
            "total_progress": total_progress,
            "progress": steps_progress  # Keep for backward compatibility
        }
    
    def end_generation(self):
        """End the current generation process"""
        self.generation_id = None
        self.current_frames = []
        self.current_step = 0
        self.total_steps = 0
        logger.info("Generation process ended")


class LivePreviewServer:
    """Server for handling live preview updates via WebSockets"""
    
    def __init__(self, socketio: SocketIO):
        """
        Initialize the LivePreviewServer
        
        Args:
            socketio: The SocketIO instance to use for WebSocket communication
        """
        self.socketio = socketio
        self.preview_manager = None
        self.pipeline = None
        self.active_generations = {}
        
    def update_preview(self, image: np.ndarray, step: int, total_steps: int, frame: int = None):
        """
        Update the preview with a new frame
        
        Args:
            image: The frame image as a numpy array
            step: Current generation step
            total_steps: Total number of generation steps
            frame: Current frame number (optional)
        """
        try:
            if image is None:
                return
                
            # Convert image to base64
            img = Image.fromarray(image)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Calculate progress
            progress = (step / total_steps) * 100
            frames_progress = (frame / 16 * 100) if frame is not None else 0
            
            # Emit preview update
            self.socketio.emit('preview_update', {
                'preview_image': f"data:image/jpeg;base64,{img_str}",
                'step': step,
                'total_steps': total_steps,
                'progress': progress,
                'frames_progress': frames_progress,
                'fps_progress': 0,  # Placeholder
                'total_progress': progress
            })
        except Exception as e:
            logger.error(f"Error in update_preview: {str(e)}")
