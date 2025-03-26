#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import uuid
import json
import threading
import time
from typing import Dict, List, Optional, Union, Callable
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

from backend.live_preview import LivePreviewManager
from backend.pipeline import AnimateDiffPipeline

logger = logging.getLogger(__name__)

class LivePreviewServer:
    """Server for handling live preview updates via WebSockets"""
    
    def __init__(
        self, 
        app: Flask, 
        preview_manager: LivePreviewManager,
        pipeline: AnimateDiffPipeline
    ):
        self.app = app
        self.preview_manager = preview_manager
        self.pipeline = pipeline
        self.socketio = SocketIO(app, cors_allowed_origins="*")
        self.active_generations = {}
        
        # Register socket events
        self.register_socket_events()
        
        # Register preview callback
        self.preview_manager.register_callback(self.send_preview_update)
    
    def register_socket_events(self):
        """Register WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {request.sid}")
            emit('connection_status', {'status': 'connected'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('start_generation')
        def handle_start_generation(data):
            """Handle start generation request from client"""
            try:
                # Extract parameters from request
                prompt = data.get('prompt', '')
                negative_prompt = data.get('negative_prompt', '')
                sd_model = data.get('sd_model', '')
                motion_module = data.get('motion_module', '')
                motion_lora = data.get('motion_lora', None)
                motion_lora_strength = float(data.get('motion_lora_strength', 1.0))
                num_frames = int(data.get('num_frames', 16))
                fps = int(data.get('fps', 8))
                guidance_scale = float(data.get('guidance_scale', 7.5))
                num_inference_steps = int(data.get('num_inference_steps', 25))
                seed = int(data.get('seed', -1))
                width = int(data.get('width', 512))
                height = int(data.get('height', 512))
                input_image = data.get('input_image', None)
                
                # Generate a unique ID for this generation
                generation_id = str(uuid.uuid4())
                
                # Start the preview manager
                self.preview_manager.start_generation(generation_id, num_inference_steps, num_frames)
                
                # Send initial status
                emit('generation_started', {
                    'generation_id': generation_id,
                    'total_steps': num_inference_steps
                })
                
                # Start generation in a separate thread
                thread = threading.Thread(
                    target=self._run_generation,
                    args=(
                        generation_id,
                        prompt,
                        negative_prompt,
                        sd_model,
                        motion_module,
                        motion_lora,
                        motion_lora_strength,
                        num_frames,
                        fps,
                        guidance_scale,
                        num_inference_steps,
                        seed,
                        width,
                        height,
                        input_image
                    )
                )
                thread.daemon = True
                thread.start()
                
                # Store the thread
                self.active_generations[generation_id] = thread
                
                return {'status': 'started', 'generation_id': generation_id}
                
            except Exception as e:
                logger.error(f"Error starting generation: {str(e)}")
                emit('generation_error', {'error': str(e)})
                return {'status': 'error', 'error': str(e)}
        
        @self.socketio.on('cancel_generation')
        def handle_cancel_generation(data):
            """Handle cancel generation request from client"""
            generation_id = data.get('generation_id')
            
            if generation_id in self.active_generations:
                # We can't actually stop the thread, but we can mark it as cancelled
                # and the preview manager will stop sending updates
                self.preview_manager.end_generation()
                
                emit('generation_cancelled', {'generation_id': generation_id})
                return {'status': 'cancelled'}
            else:
                emit('generation_error', {'error': 'Generation not found'})
                return {'status': 'error', 'error': 'Generation not found'}
        
        @self.socketio.on('get_progress')
        def handle_get_progress():
            """Handle get progress request from client"""
            progress = self.preview_manager.get_progress()
            emit('generation_progress', progress)
            return progress
    
    def _run_generation(
        self,
        generation_id: str,
        prompt: str,
        negative_prompt: str,
        sd_model: str,
        motion_module: str,
        motion_lora: Optional[str],
        motion_lora_strength: float,
        num_frames: int,
        fps: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
        width: int,
        height: int,
        input_image: Optional[str]
    ):
        """Run the generation process in a separate thread"""
        try:
            # Create a callback for the pipeline
            def preview_callback(frame, step, total_steps, current_frame=None):
                self.preview_manager.update_preview(frame, step, total_steps, current_frame)
            
            # Generate the video
            output_path = self.pipeline.generate_with_live_preview(
                prompt=prompt,
                negative_prompt=negative_prompt,
                sd_model_name=sd_model,
                motion_module_name=motion_module,
                motion_lora_name=motion_lora,
                motion_lora_strength=motion_lora_strength,
                num_frames=num_frames,
                fps=fps,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                width=width,
                height=height,
                input_image=input_image,
                preview_callback=preview_callback
            )
            
            # Send completion notification
            self.socketio.emit('generation_completed', {
                'generation_id': generation_id,
                'output_path': output_path,
                'output_url': f'/outputs/{os.path.basename(output_path)}'
            })
            
            # End the generation in the preview manager
            self.preview_manager.end_generation()
            
            # Remove from active generations
            if generation_id in self.active_generations:
                del self.active_generations[generation_id]
                
        except Exception as e:
            logger.error(f"Error in generation thread: {str(e)}")
            
            # Send error notification
            self.socketio.emit('generation_error', {
                'generation_id': generation_id,
                'error': str(e)
            })
            
            # End the generation in the preview manager
            self.preview_manager.end_generation()
            
            # Remove from active generations
            if generation_id in self.active_generations:
                del self.active_generations[generation_id]
    
    def send_preview_update(self, generation_id: str, step: int, total_steps: int, preview_image: str):
        """Send a preview update to connected clients"""
        # Get complete progress information
        progress_info = self.preview_manager.get_progress()
        
        # Create update data with all progress metrics
        update_data = {
            'generation_id': generation_id,
            'step': step,
            'total_steps': total_steps,
            'progress': progress_info['progress'],
            'steps_progress': progress_info['steps_progress'],
            'frames_progress': progress_info['frames_progress'],
            'fps_progress': progress_info['fps_progress'],
            'total_progress': progress_info['total_progress'],
            'current_frame': progress_info['current_frame'],
            'total_frames': progress_info['total_frames'],
            'fps': progress_info['fps'],
            'preview_image': preview_image
        }
        
        self.socketio.emit('preview_update', update_data)
    
    def run(self, host: str = '0.0.0.0', port: int = 7860, debug: bool = False):
        """Run the WebSocket server"""
        logger.info(f"Starting LivePreviewServer on {host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug)
