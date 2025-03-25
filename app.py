#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import uuid
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from flask_socketio import SocketIO

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.models import ModelManager
from backend.pipeline import AnimateDiffPipeline
from backend.config import Config
from backend.live_preview import LivePreviewManager
from backend.live_preview_server import LivePreviewServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'animatediff.log'))
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend/static'),
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend/templates'))
CORS(app)

# Initialize configuration
config = Config()

# Initialize model manager, pipeline, and preview manager
model_manager = None
pipeline = None
preview_manager = None
preview_server = None

def initialize_components():
    """Initialize all components"""
    global model_manager, pipeline, preview_manager, preview_server
    
    logger.info("Initializing model manager...")
    model_manager = ModelManager(
        models_dir=config.models_dir,
        motion_module_dir=config.motion_module_dir,
        motion_lora_dir=config.motion_lora_dir,
        sd_dir=config.sd_dir,
        use_cpu=True,  # Force CPU-only mode
        optimize_memory=True,
        quantization_type=config.quantization_type
    )
    
    logger.info("Initializing preview manager...")
    preview_manager = LivePreviewManager(
        output_dir=os.path.join(config.output_dir, "previews")
    )
    
    logger.info("Initializing AnimateDiff pipeline...")
    pipeline = AnimateDiffPipeline(
        model_manager=model_manager,
        output_dir=config.output_dir,
        use_cpu=True,  # Force CPU-only mode
        enable_uncensored=config.enable_uncensored  # Enable uncensored generation
    )
    
    logger.info("Initializing preview server...")
    preview_server = LivePreviewServer(
        app=app,
        preview_manager=preview_manager,
        pipeline=pipeline
    )
    
    logger.info("All components initialized successfully")

# Routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    try:
        return jsonify({
            'stable_diffusion': model_manager.get_sd_models(),
            'motion_module': model_manager.get_motion_modules(),
            'motion_lora': model_manager.get_motion_loras()
        })
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    """Serve generated output files"""
    return send_from_directory(config.output_dir, filename)

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get the status of the server"""
    return jsonify({
        'status': 'running',
        'cpu_only': True,
        'uncensored': config.enable_uncensored,
        'version': '1.0.0'
    })

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AnimateDiff GUI')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Initialize components
    initialize_components()
    
    # Run the server with Socket.IO
    logger.info(f"Starting AnimateDiff GUI on {args.host}:{args.port}")
    preview_server.run(host=args.host, port=args.port, debug=args.debug)
