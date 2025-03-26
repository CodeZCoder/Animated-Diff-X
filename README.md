# AnimateDiff GUI

An advanced AnimateDiff GUI interface with Python backend and HTML/JS frontend. This implementation is CPU-only optimized and supports AnimateDiff Motion LoRAs, image-to-video, and text-to-video functionality with uncensored content generation.

## Features

- CPU-only optimized implementation
- Support for AnimateDiff Motion LoRAs
- Image-to-video and text-to-video functionality
- Live generation preview
- Uncensored content generation
- User-friendly interface

## Installation

1. Unzip the package to a directory of your choice
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Alternative Manual Installation If you still encounter issues, you can try installing the dependencies manually in this specific order: pip install huggingface_hub==0.16.4 pip install diffusers==0.11.1 pip install -r requirements_fixed.txt

3. Run the setup script to initialize the application:

```bash
python setup.py
```

## Usage

1. Start the application:

```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:7860`

3. Configure the generation settings:
   - Enter a prompt and negative prompt
   - Select a Stable Diffusion model
   - Select a motion module
   - Optionally select a Motion LoRA and adjust its strength
   - Configure other parameters like number of frames, FPS, dimensions, etc.
   - For image-to-video, toggle the switch and upload an image

4. Click "Generate" to start the generation process
   - Watch the live preview as the generation progresses
   - The final video will appear in the Results section when complete

5. Download the generated video using the "Download Video" button

## Models

The application will automatically download the default Stable Diffusion model (SD 1.5) and a placeholder for the motion module. For better results, you should download the actual models:

1. Stable Diffusion models: Place in `models/stable_diffusion/`
2. Motion modules: Place in `models/motion_module/`
3. Motion LoRAs: Place in `models/motion_lora/`

Recommended models:
- Motion Module: `v3_sd15_mm.safetensors`
- Motion LoRAs: Various motion effect LoRAs like `v2_lora_ZoomIn.ckpt`, `v2_lora_PanLeft.ckpt`, etc.

## Advanced Configuration

You can modify the configuration in `backend/config.py` to adjust:
- Default models
- Generation settings
- CPU optimization settings
- Uncensored mode

## Troubleshooting

- If you encounter memory issues, try reducing the image dimensions or number of frames
- For CPU optimization, the application uses quantization which may affect quality
- Check the log file `animatediff.log` for detailed error information

## License

This software is provided as-is without any warranty. Use at your own risk.
