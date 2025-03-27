#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import argparse
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
import threading
import uuid
import base64
from PIL import Image
import io
import time

from backend.config import Config
from backend.models import ModelManager
from backend.pipeline import AnimateDiffPipeline
from backend.live_preview import LivePreviewServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
    template_folder=os.path.join(os.path.dirname(__file__), 'frontend', 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), 'frontend', 'static')
)
app.config['SECRET_KEY'] = 'animatediff-gui'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
config = None
model_manager = None
pipeline = None
preview_server = None
active_generations = {}

def initialize_components():
    """Initialize all components"""
    global config, model_manager, pipeline, preview_server
    
    try:
        # Initialize config
        config = Config()
        
        # Initialize model manager
        logger.info("Initializing model manager...")
        model_manager = ModelManager(
            models_dir=config.models_dir,
            motion_module_dir=config.motion_module_dir,
            motion_lora_dir=config.motion_lora_dir,
            sd_dir=config.sd_dir,
            ip_adapter_dir=getattr(config, 'ip_adapter_dir', None),
            clip_vision_dir=getattr(config, 'clip_vision_dir', None),
            controlnet_dir=getattr(config, 'controlnet_dir', None),
            sd15_lora_dir=getattr(config, 'sd15_lora_dir', None),
            use_cpu=config.use_cpu,
            optimize_memory=config.optimize_memory,
            quantization_type=config.quantization_type
        )
        
        # Initialize preview server
        logger.info("Initializing preview manager...")
        try:
            preview_server = LivePreviewServer(socketio)
        except Exception as e:
            logger.error(f"Error initializing preview server: {str(e)}")
            preview_server = None
        
        # Initialize pipeline
        logger.info("Initializing AnimateDiff pipeline...")
        pipeline = AnimateDiffPipeline(
            model_manager=model_manager,
            output_dir=config.output_dir,
            use_cpu=config.use_cpu,
            enable_uncensored=config.enable_uncensored,
            callback=preview_server.update_preview if preview_server else None
        )
        
        return True
    
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        return False

# Routes
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/outputs/<path:filename>')
def output_file(filename):
    """Serve output files"""
    return send_from_directory(config.output_dir, filename)

# API endpoints
@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    try:
        # Get models from model manager
        sd_models = model_manager.get_sd_models()
        motion_modules = model_manager.get_motion_modules()
        motion_loras = model_manager.get_motion_loras()
        
        # Get additional models if available
        ip_adapters = []
        controlnets = []
        sd15_loras = []
        
        try:
            if hasattr(model_manager, 'get_ip_adapters'):
                ip_adapters = model_manager.get_ip_adapters()
        except Exception as e:
            logger.warning(f"Error getting IP Adapters: {str(e)}")
            
        try:
            if hasattr(model_manager, 'get_controlnets'):
                controlnets = model_manager.get_controlnets()
        except Exception as e:
            logger.warning(f"Error getting ControlNets: {str(e)}")
            
        try:
            if hasattr(model_manager, 'get_sd15_loras'):
                sd15_loras = model_manager.get_sd15_loras()
        except Exception as e:
            logger.warning(f"Error getting SD 1.5 LoRAs: {str(e)}")
        
        # Return models as JSON
        return jsonify({
            'stable_diffusion': sd_models,
            'motion_module': motion_modules,
            'motion_lora': motion_loras,
            'ip_adapter': ip_adapters,
            'controlnet': controlnets,
            'sd15_lora': sd15_loras
        })
    
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('start_generation')
def handle_start_generation(data):
    """Handle generation request"""
    try:
        # Generate a unique ID for this generation
        generation_id = str(uuid.uuid4())
        
        # Store generation data
        active_generations[generation_id] = {
            'status': 'starting',
            'data': data,
            'client_id': request.sid
        }
        
        # Notify client that generation has started
        emit('generation_started', {
            'generation_id': generation_id,
            'total_steps': data.get('num_inference_steps', 25)
        })
        
        # Validate image data before starting generation
        validation_errors = []
        
        # Validate IP Adapter image if provided
        if data.get('ip_adapter_image') and data.get('ip_adapter_name'):
            try:
                # Check if it's a base64 image
                if not data['ip_adapter_image'].startswith('data:image'):
                    validation_errors.append("IP Adapter image is not in valid base64 format")
                else:
                    # Try to decode a small part to validate
                    image_data = data['ip_adapter_image'].split(',')[1][:100]  # Just check the beginning
                    base64.b64decode(image_data)
            except Exception as e:
                validation_errors.append(f"Invalid IP Adapter image: {str(e)}")
        
        # Validate ControlNet image if provided
        if data.get('controlnet_image') and data.get('controlnet_name'):
            try:
                # Check if it's a base64 image
                if not data['controlnet_image'].startswith('data:image'):
                    validation_errors.append("ControlNet image is not in valid base64 format")
                else:
                    # Try to decode a small part to validate
                    image_data = data['controlnet_image'].split(',')[1][:100]  # Just check the beginning
                    base64.b64decode(image_data)
            except Exception as e:
                validation_errors.append(f"Invalid ControlNet image: {str(e)}")
        
        # Validate input image if provided
        if data.get('input_image'):
            try:
                # Check if it's a base64 image
                if not data['input_image'].startswith('data:image'):
                    validation_errors.append("Input image is not in valid base64 format")
                else:
                    # Try to decode a small part to validate
                    image_data = data['input_image'].split(',')[1][:100]  # Just check the beginning
                    base64.b64decode(image_data)
            except Exception as e:
                validation_errors.append(f"Invalid input image: {str(e)}")
        
        # If there are validation errors, report them and don't start generation
        if validation_errors:
            error_message = "Image validation failed: " + "; ".join(validation_errors)
            logger.error(error_message)
            
            # Update generation status
            active_generations[generation_id]['status'] = 'error'
            active_generations[generation_id]['error'] = error_message
            
            # Notify client of the error
            emit('generation_error', {
                'generation_id': generation_id,
                'error': error_message
            })
            return
        
        # Start generation in a separate thread
        threading.Thread(
            target=generate_video,
            args=(generation_id, data),
            daemon=True
        ).start()
    except Exception as e:
        error_message = f"Error starting generation: {str(e)}"
        logger.error(error_message)
        
        # Try to notify client of the error
        try:
            emit('generation_error', {
                'error': error_message
            })
        except Exception as emit_error:
            logger.error(f"Failed to emit error to client: {str(emit_error)}")

@socketio.on('cancel_generation')
def handle_cancel_generation(data):
    """Handle cancellation request"""
    generation_id = data.get('generation_id')
    
    if generation_id in active_generations:
        active_generations[generation_id]['status'] = 'cancelled'
        
        # Notify client that generation has been cancelled
        emit('generation_cancelled', {
            'generation_id': generation_id
        })
        
        # TODO: Implement actual cancellation in the pipeline
        logger.info(f"Generation cancelled: {generation_id}")

def generate_video(generation_id, data):
    """Generate video with AnimateDiff"""
    try:
        # Update generation status
        active_generations[generation_id]['status'] = 'generating'
        
        # Extract parameters from data
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', '')
        sd_model = data.get('sd_model', 'runwayml/stable-diffusion-v1-5')
        motion_module = data.get('motion_module', 'mm_sd_v15_v2.ckpt')
        motion_lora = data.get('motion_lora')
        motion_lora_strength = float(data.get('motion_lora_strength', 1.0))
        num_frames = int(data.get('num_frames', 16))
        fps = int(data.get('fps', 8))
        guidance_scale = float(data.get('guidance_scale', 7.5))
        num_inference_steps = int(data.get('num_inference_steps', 25))
        seed = int(data.get('seed', -1))
        width = int(data.get('width', 512))
        height = int(data.get('height', 512))
        
        # Extract optional parameters
        lora_names = data.get('lora_names')
        lora_strengths = data.get('lora_strengths')
        ip_adapter_name = data.get('ip_adapter_name')
        ip_adapter_image = data.get('ip_adapter_image')
        ip_adapter_strength = float(data.get('ip_adapter_strength', 0.5)) if data.get('ip_adapter_strength') else 0.5
        controlnet_name = data.get('controlnet_name')
        controlnet_image = data.get('controlnet_image')
        controlnet_strength = float(data.get('controlnet_strength', 0.5)) if data.get('controlnet_strength') else 0.5
        input_image = data.get('input_image')
        
        # Log all parameters for debugging
        logger.info(f"Generation parameters for {generation_id}:")
        logger.info(f"  Prompt: {prompt}")
        logger.info(f"  Negative prompt: {negative_prompt}")
        logger.info(f"  SD model: {sd_model}")
        logger.info(f"  Motion module: {motion_module}")
        logger.info(f"  Motion LoRA: {motion_lora}")
        logger.info(f"  Frames: {num_frames}")
        logger.info(f"  FPS: {fps}")
        logger.info(f"  Guidance scale: {guidance_scale}")
        logger.info(f"  Steps: {num_inference_steps}")
        logger.info(f"  Seed: {seed}")
        logger.info(f"  Size: {width}x{height}")
        
        if ip_adapter_name and ip_adapter_image:
            logger.info(f"  Using IP Adapter: {ip_adapter_name} with strength {ip_adapter_strength}")
        
        if controlnet_name and controlnet_image:
            logger.info(f"  Using ControlNet: {controlnet_name} with strength {controlnet_strength}")
        
        if input_image:
            logger.info(f"  Using input image for image-to-video generation")
        
        # Process base64 images and save to temporary files
        temp_image_paths = {}
        
        # Process IP Adapter image if provided
        if ip_adapter_image and ip_adapter_image.startswith('data:image'):
            try:
                # Extract base64 data
                image_data = ip_adapter_image.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                
                # Save to temporary file
                temp_file = os.path.join(config.output_dir, f"temp_ip_adapter_{generation_id}.jpg")
                with open(temp_file, 'wb') as f:
                    f.write(image_bytes)
                
                # Replace base64 data with file path
                temp_image_paths['ip_adapter'] = temp_file
                ip_adapter_image = temp_file
                logger.info(f"Saved IP Adapter image to temporary file: {temp_file}")
                
                # Verify the file exists and is readable
                if not os.path.exists(temp_file):
                    raise FileNotFoundError(f"Temporary file {temp_file} was not created")
                
                # Try to open the image to verify it's valid
                try:
                    from PIL import Image
                    test_img = Image.open(temp_file)
                    logger.info(f"IP Adapter image verified: {test_img.size}")
                except Exception as img_error:
                    logger.error(f"IP Adapter image is not a valid image: {str(img_error)}")
                    raise ValueError(f"Invalid IP Adapter image: {str(img_error)}")
                
            except Exception as e:
                logger.error(f"Error processing IP Adapter image: {str(e)}")
                # Notify client of the error
                socketio.emit('generation_error', {
                    'generation_id': generation_id,
                    'error': f"Error processing IP Adapter image: {str(e)}"
                }, room=active_generations[generation_id]['client_id'])
                ip_adapter_image = None
                # Update generation status
                active_generations[generation_id]['status'] = 'error'
                active_generations[generation_id]['error'] = f"Error processing IP Adapter image: {str(e)}"
                return
        
        # Process ControlNet image if provided
        if controlnet_image and controlnet_image.startswith('data:image'):
            try:
                # Extract base64 data
                image_data = controlnet_image.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                
                # Save to temporary file
                temp_file = os.path.join(config.output_dir, f"temp_controlnet_{generation_id}.jpg")
                with open(temp_file, 'wb') as f:
                    f.write(image_bytes)
                
                # Replace base64 data with file path
                temp_image_paths['controlnet'] = temp_file
                controlnet_image = temp_file
                logger.info(f"Saved ControlNet image to temporary file: {temp_file}")
                
                # Verify the file exists and is readable
                if not os.path.exists(temp_file):
                    raise FileNotFoundError(f"Temporary file {temp_file} was not created")
                
                # Try to open the image to verify it's valid
                try:
                    from PIL import Image
                    test_img = Image.open(temp_file)
                    logger.info(f"ControlNet image verified: {test_img.size}")
                except Exception as img_error:
                    logger.error(f"ControlNet image is not a valid image: {str(img_error)}")
                    raise ValueError(f"Invalid ControlNet image: {str(img_error)}")
                
            except Exception as e:
                logger.error(f"Error processing ControlNet image: {str(e)}")
                # Notify client of the error
                socketio.emit('generation_error', {
                    'generation_id': generation_id,
                    'error': f"Error processing ControlNet image: {str(e)}"
                }, room=active_generations[generation_id]['client_id'])
                controlnet_image = None
                # Update generation status
                active_generations[generation_id]['status'] = 'error'
                active_generations[generation_id]['error'] = f"Error processing ControlNet image: {str(e)}"
                return
        
        # Process input image if provided
        if input_image and input_image.startswith('data:image'):
            try:
                # Extract base64 data
                image_data = input_image.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                
                # Save to temporary file
                temp_file = os.path.join(config.output_dir, f"temp_input_{generation_id}.jpg")
                with open(temp_file, 'wb') as f:
                    f.write(image_bytes)
                
                # Replace base64 data with file path
                temp_image_paths['input'] = temp_file
                input_image = temp_file
                logger.info(f"Saved input image to temporary file: {temp_file}")
                
                # Verify the file exists and is readable
                if not os.path.exists(temp_file):
                    raise FileNotFoundError(f"Temporary file {temp_file} was not created")
                
                # Try to open the image to verify it's valid
                try:
                    from PIL import Image
                    test_img = Image.open(temp_file)
                    logger.info(f"Input image verified: {test_img.size}")
                except Exception as img_error:
                    logger.error(f"Input image is not a valid image: {str(img_error)}")
                    raise ValueError(f"Invalid input image: {str(img_error)}")
                
            except Exception as e:
                logger.error(f"Error processing input image: {str(e)}")
                # Notify client of the error
                socketio.emit('generation_error', {
                    'generation_id': generation_id,
                    'error': f"Error processing input image: {str(e)}"
                }, room=active_generations[generation_id]['client_id'])
                input_image = None
                # Update generation status
                active_generations[generation_id]['status'] = 'error'
                active_generations[generation_id]['error'] = f"Error processing input image: {str(e)}"
                return
        
        # Define preview callback
        def preview_callback(image, step, total_steps, frame=None):
            # Convert image to base64
            if image is not None:
                try:
                    img = Image.fromarray(image)
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    # Calculate progress
                    progress = (step / total_steps) * 100
                    frames_progress = (frame / num_frames * 100) if frame is not None else 0
                    
                    # Emit preview update
                    socketio.emit('preview_update', {
                        'generation_id': generation_id,
                        'preview_image': f"data:image/jpeg;base64,{img_str}",
                        'step': step,
                        'total_steps': total_steps,
                        'progress': progress,
                        'frames_progress': frames_progress,
                        'fps_progress': 0,  # TODO: Implement FPS progress
                        'total_progress': progress
                    }, room=active_generations[generation_id]['client_id'])
                except Exception as e:
                    logger.error(f"Error in preview callback: {str(e)}")
        
        # Generate the video
        logger.info(f"Starting pipeline for generation {generation_id}")
        try:
            output_file = pipeline.generate_with_live_preview(
                prompt=prompt,
                negative_prompt=negative_prompt,
                sd_model_name=sd_model,
                motion_module_name=motion_module,
                motion_lora_name=motion_lora,
                motion_lora_strength=motion_lora_strength,
                lora_names=lora_names,
                lora_strengths=lora_strengths,
                ip_adapter_name=ip_adapter_name,
                ip_adapter_image=ip_adapter_image,
                ip_adapter_strength=ip_adapter_strength,
                controlnet_name=controlnet_name,
                controlnet_image=controlnet_image,
                controlnet_strength=controlnet_strength,
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
            logger.info(f"Pipeline completed successfully for generation {generation_id}")
        except Exception as pipeline_error:
            error_message = f"Error in pipeline execution: {str(pipeline_error)}"
            logger.error(error_message)
            # Include traceback for debugging
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Notify client of the error
            socketio.emit('generation_error', {
                'generation_id': generation_id,
                'error': error_message
            }, room=active_generations[generation_id]['client_id'])
            
            # Update generation status
            active_generations[generation_id]['status'] = 'error'
            active_generations[generation_id]['error'] = error_message
            
            # Clean up temporary files
            for temp_file in temp_image_paths.values():
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        logger.info(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Error removing temporary file {temp_file}: {str(e)}")
            
            return
        
        # Clean up temporary files
        for temp_file in temp_image_paths.values():
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.info(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Error removing temporary file {temp_file}: {str(e)}")
        
        # Update generation status
        active_generations[generation_id]['status'] = 'completed'
        active_generations[generation_id]['output_file'] = output_file
        
        # Get relative path for URL
        output_url = f"/outputs/{os.path.basename(output_file)}"
        
        # Notify client that generation has completed
        socketio.emit('generation_completed', {
            'generation_id': generation_id,
            'output_url': output_url
        }, room=active_generations[generation_id]['client_id'])
        
        logger.info(f"Generation completed: {generation_id}")
    
    except Exception as e:
        error_message = f"Error generating video: {str(e)}"
        logger.error(error_message)
        # Include traceback for debugging
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Update generation status
        if generation_id in active_generations:
            active_generations[generation_id]['status'] = 'error'
            active_generations[generation_id]['error'] = error_message
            
            # Notify client of the error
            socketio.emit('generation_error', {
                'generation_id': generation_id,
                'error': error_message
            }, room=active_generations[generation_id]['client_id'])

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AnimateDiff GUI')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    # Initialize components
    if initialize_components():
        # Start the server
        logger.info(f"Starting AnimateDiff GUI on {args.host}:{args.port}")
        if preview_server is None:
            logger.warning("Preview server not initialized, falling back to basic Socket.IO")
        
        socketio.run(app, host=args.host, port=args.port, debug=args.debug)
    else:
        logger.error("Failed to initialize components, exiting")
