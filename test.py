#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import unittest
import tempfile
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TestAnimateDiffGUI(unittest.TestCase):
    """Test cases for AnimateDiff GUI"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        
        # Set up test paths
        self.models_dir = os.path.join(self.test_dir, "models")
        self.output_dir = os.path.join(self.test_dir, "outputs")
        self.sd_dir = os.path.join(self.models_dir, "stable_diffusion")
        self.motion_module_dir = os.path.join(self.models_dir, "motion_module")
        self.motion_lora_dir = os.path.join(self.models_dir, "motion_lora")
        
        # Create test directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.sd_dir, exist_ok=True)
        os.makedirs(self.motion_module_dir, exist_ok=True)
        os.makedirs(self.motion_lora_dir, exist_ok=True)
        
        # Create test files
        with open(os.path.join(self.motion_module_dir, "test_module.safetensors"), "w") as f:
            f.write("Test motion module")
        
        with open(os.path.join(self.motion_lora_dir, "test_lora.safetensors"), "w") as f:
            f.write("Test motion LoRA")
    
    def tearDown(self):
        """Clean up test environment"""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_config(self):
        """Test configuration loading"""
        from backend.config import Config
        
        # Create a config with test directories
        config = Config()
        config.base_dir = self.test_dir
        config.models_dir = self.models_dir
        config.output_dir = self.output_dir
        config.sd_dir = self.sd_dir
        config.motion_module_dir = self.motion_module_dir
        config.motion_lora_dir = self.motion_lora_dir
        
        # Check if directories are set correctly
        self.assertEqual(config.base_dir, self.test_dir)
        self.assertEqual(config.models_dir, self.models_dir)
        self.assertEqual(config.output_dir, self.output_dir)
        self.assertEqual(config.sd_dir, self.sd_dir)
        self.assertEqual(config.motion_module_dir, self.motion_module_dir)
        self.assertEqual(config.motion_lora_dir, self.motion_lora_dir)
        
        # Check default settings
        self.assertTrue(config.enable_uncensored)
        self.assertEqual(config.quantization_type, "int8")
    
    def test_model_manager(self):
        """Test model manager functionality"""
        from backend.models import ModelManager
        
        # Create a model manager with test directories
        model_manager = ModelManager(
            models_dir=self.models_dir,
            motion_module_dir=self.motion_module_dir,
            motion_lora_dir=self.motion_lora_dir,
            sd_dir=self.sd_dir,
            use_cpu=True,
            optimize_memory=True
        )
        
        # Check if model lists are correct
        motion_modules = model_manager.get_motion_modules()
        self.assertIn("test_module.safetensors", motion_modules)
        
        motion_loras = model_manager.get_motion_loras()
        self.assertIn("test_lora.safetensors", motion_loras)
    
    def test_utils(self):
        """Test utility functions"""
        from backend.utils import create_directory_if_not_exists
        
        # Test directory creation
        test_dir = os.path.join(self.test_dir, "test_utils")
        create_directory_if_not_exists(test_dir)
        self.assertTrue(os.path.exists(test_dir))
    
    def test_live_preview(self):
        """Test live preview functionality"""
        from backend.live_preview import LivePreviewManager
        import numpy as np
        
        # Create a preview manager with test directory
        preview_manager = LivePreviewManager(
            output_dir=self.output_dir
        )
        
        # Test preview generation
        preview_manager.start_generation("test_id", 10)
        
        # Create a dummy frame
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Update preview with the frame
        preview_manager.update_preview(frame, 1, 10)
        
        # Check progress
        progress = preview_manager.get_progress()
        self.assertEqual(progress["generation_id"], "test_id")
        self.assertEqual(progress["current_step"], 1)
        self.assertEqual(progress["total_steps"], 10)
        self.assertEqual(progress["progress"], 10.0)
        
        # End generation
        preview_manager.end_generation()
        
        # Check if generation is ended
        progress = preview_manager.get_progress()
        self.assertIsNone(progress["generation_id"])
        self.assertEqual(progress["current_step"], 0)
        self.assertEqual(progress["total_steps"], 0)
        self.assertEqual(progress["progress"], 0)

def run_tests():
    """Run all tests"""
    logger.info("Running tests...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    logger.info("Tests completed")

if __name__ == "__main__":
    run_tests()
