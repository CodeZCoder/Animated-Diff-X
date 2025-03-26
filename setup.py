#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Create the necessary directory structure for the application"""
    logger.info("Creating directory structure...")
    
    # Base directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    output_dir = os.path.join(base_dir, "outputs")
    
    # Model subdirectories
    sd_dir = os.path.join(models_dir, "stable_diffusion")
    motion_module_dir = os.path.join(models_dir, "motion_module")
    motion_lora_dir = os.path.join(models_dir, "motion_lora")
    
    # Create directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sd_dir, exist_ok=True)
    os.makedirs(motion_module_dir, exist_ok=True)
    os.makedirs(motion_lora_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "previews"), exist_ok=True)
    
    logger.info("Directory structure created successfully")

def check_dependencies():
    """Check if all required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    try:
        import torch
        import diffusers
        import transformers
        import flask
        import flask_socketio
        import imageio
        import numpy
        import PIL
        import safetensors
        import omegaconf
        
        # Check huggingface_hub version
        import huggingface_hub
        logger.info(f"huggingface_hub version: {huggingface_hub.__version__}")
        
        # Check if cached_download exists
        if not hasattr(huggingface_hub, 'cached_download'):
            logger.warning("huggingface_hub.cached_download not found. You may need to install version 0.16.4")
            return False
        
        logger.info("All dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        return False

def install_dependencies(fixed=False):
    """Install required dependencies"""
    logger.info("Installing dependencies...")
    
    try:
        import subprocess
        
        # Get the requirements file path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        requirements_file = os.path.join(base_dir, "requirements_fixed.txt" if fixed else "requirements.txt")
        
        if fixed and not os.path.exists(requirements_file):
            logger.warning("Fixed requirements file not found, using standard requirements")
            requirements_file = os.path.join(base_dir, "requirements.txt")
        
        # Install huggingface_hub first to ensure correct version
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub==0.16.4"])
        
        # Install diffusers next
        subprocess.check_call([sys.executable, "-m", "pip", "install", "diffusers==0.11.1"])
        
        # Install remaining dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        
        logger.info("Dependencies installed successfully")
        return True
    except Exception as e:
        logger.error(f"Error installing dependencies: {str(e)}")
        return False

def download_default_models():
    """Download default models if they don't exist"""
    logger.info("Checking for default models...")
    
    try:
        from huggingface_hub import snapshot_download
        import torch
        from diffusers import StableDiffusionPipeline
        
        # Get model directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        motion_module_dir = os.path.join(models_dir, "motion_module")
        
        # Check if default motion module exists
        default_motion_module = os.path.join(motion_module_dir, "v3_sd15_mm.safetensors")
        if not os.path.exists(default_motion_module):
            logger.info("Default motion module not found, downloading...")
            
            # Create a placeholder file to indicate download is in progress
            with open(os.path.join(motion_module_dir, "DOWNLOADING.txt"), "w") as f:
                f.write("Download in progress, please wait...")
            
            # In a real implementation, you would download from HuggingFace or another source
            # For this example, we'll just create a placeholder file
            with open(default_motion_module, "w") as f:
                f.write("This is a placeholder for the motion module. In a real implementation, you would download the actual model.")
            
            # Remove the placeholder file
            os.remove(os.path.join(motion_module_dir, "DOWNLOADING.txt"))
            
            logger.info("Default motion module downloaded successfully")
        else:
            logger.info("Default motion module already exists")
        
        # Check if default SD model exists (we'll use the HuggingFace model)
        logger.info("Checking for default Stable Diffusion model...")
        try:
            # Try to load the model to see if it's cached
            StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
            logger.info("Default Stable Diffusion model is available")
        except Exception:
            logger.info("Default Stable Diffusion model not found, downloading...")
            # This will download and cache the model
            StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
            logger.info("Default Stable Diffusion model downloaded successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error downloading default models: {str(e)}")
        return False

def run_application(host="0.0.0.0", port=7860, debug=False):
    """Run the AnimateDiff GUI application"""
    logger.info(f"Starting AnimateDiff GUI on {host}:{port}")
    
    try:
        # Import the app module
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from app import app, initialize_components, preview_server
        
        # Initialize components
        initialize_components()
        
        # Run the server
        preview_server.run(host=host, port=port, debug=debug)
        
        return True
    except Exception as e:
        logger.error(f"Error running application: {str(e)}")
        return False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AnimateDiff GUI Setup')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--skip-models', action='store_true', help='Skip model downloads')
    parser.add_argument('--fixed-deps', action='store_true', help='Use fixed dependencies installation order')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create directory structure
    create_directory_structure()
    
    # Check and install dependencies if needed
    if not args.skip_deps:
        if not check_dependencies():
            if not install_dependencies(fixed=args.fixed_deps):
                logger.error("Failed to install dependencies. Please install them manually.")
                sys.exit(1)
    
    # Download default models if needed
    if not args.skip_models:
        if not download_default_models():
            logger.warning("Failed to download default models. You may need to download them manually.")
    
    # Run the application
    if not run_application(host=args.host, port=args.port, debug=args.debug):
        logger.error("Failed to run the application.")
        sys.exit(1)
