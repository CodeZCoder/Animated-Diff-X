#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import base64
import logging
import argparse
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO
import threading
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set cache directories with defaults if not in .env
os.environ["HF_HOME"] = os.getenv("HF_HOME", "D:/animate diff x/animatediff-gui/models/hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "D:/animate diff x/animatediff-gui/models/transformers_cache")
os.environ["DIFFUSERS_CACHE"] = os.getenv("DIFFUSERS_CACHE", "D:/animate diff x/animatediff-gui/models/diffusers_cache")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, 
            static_folder='frontend/static',
            template_folder='frontend/templates')

# Create SocketIO instance
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize components
model_manager = None
pipeline = None
preview_server = None

def parse_args():
    parser = argparse.ArgumentParser(description='AnimateDiff GUI')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    return parser.parse_args()

def initialize_components():
    global model_manager, pipeline, preview_server
    
    try:
        # Import backend modules
        from backend.models import ModelManager
        from backend.pipeline import AnimateDiffPipeline
        from backend.live_preview_enhanced import LivePreviewServer
        
        # Initialize model manager
        model_manager = ModelManager(
            sd_models_path="models/stable-diffusion",
            motion_module_path="models/motion_module",
            motion_lora_path="models/motion_lora"
        )
        
        # Initialize pipeline
        pipeline = AnimateDiffPipeline(
            model_manager=model_manager,
            output_dir="outputs",
            use_cpu=True,
            enable_uncensored=True
        )
        
        # Initialize preview server
        preview_server = LivePreviewServer(app, socketio)
        
        logger.info("All components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        return False

# Routes
@app.route('/')
def index():
    return render_template('index_enhanced.html')

@app.route('/outputs/<path:filename>')
def outputs(filename):
    return send_from_directory('outputs', filename)

@app.route('/models')
def get_models():
    try:
        sd_models = model_manager.get_sd_models()
        motion_modules = model_manager.get_motion_modules()
        motion_loras = model_manager.get_motion_loras()
        
        return jsonify({
            'sd_models': sd_models,
            'motion_modules': motion_modules,
            'motion_loras': motion_loras
        })
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify({
            'sd_models': [],
            'motion_modules': [],
            'motion_loras': []
        })

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get request data
        data = request.json
        
        # Start generation in a separate thread
        success = preview_server.start_generation(
            pipeline.generate_with_live_preview,
            **data
        )
        
        return jsonify({'success': success})
    except Exception as e:
        logger.error(f"Error starting generation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/cancel', methods=['POST'])
def cancel():
    try:
        # TODO: Implement cancellation
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error cancelling generation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    args = parse_args()
    
    # Initialize components
    if not initialize_components():
        logger.error("Failed to initialize components. Exiting.")
        sys.exit(1)
    
    # Run the server with Socket.IO
    logger.info(f"Starting AnimateDiff GUI on {args.host}:{args.port}")
    try:
        if preview_server is not None:
            preview_server.run(host=args.host, port=args.port, debug=args.debug)
        else:
            # Fallback if preview_server is None
            socketio.run(app, host=args.host, port=args.port, debug=args.debug, allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error(f"Error running application: {str(e)}")
        logger.error("Failed to run the application.")
