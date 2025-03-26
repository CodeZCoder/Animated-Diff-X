# AnimateDiff Model Compatibility Guide

This document provides information about model compatibility in AnimateDiff and how to use the enhanced features.

## Model Compatibility

AnimateDiff requires specific combinations of models to work correctly:

### SD Model and Motion Module Compatibility

| SD Model Type | Compatible Motion Modules |
|---------------|---------------------------|
| SD 1.5        | mm_sd_v15_v2.ckpt, v3_sd15_mm.ckpt |
| SDXL          | mm_sdxl_v10_beta.ckpt, mm_sdxl_v1.0.safetensors |

Using incompatible combinations (like an SDXL motion module with an SD 1.5 base model) will result in errors or poor quality outputs.

### Motion LoRA Compatibility

Motion LoRAs are special LoRAs that control camera movement and animation effects. They have specific compatibility requirements:

IMPORTANT
(((NEED THE ABILITY TO TURN OFF SDXL MODEL AKA RUNWAYML FOR THE SD1.5 LORA AND THE MOTION LORA TO WORK))))))

| Motion LoRA Type | Compatible Motion Modules |
|------------------|---------------------------|
| PanLeft, PanRight, ZoomIn, ZoomOut, etc. | mm_sd_v15_v2.ckpt (V2 modules) |
| Wiggle, Dance, Bouncing, Shatter | mm_sd_v15_v2.ckpt, v3_sd15_mm.ckpt (V2 and V3 modules) |

Most motion LoRAs from Civitai work best with V2 motion modules like mm_sd_v15_v2.ckpt.

## New Features

### Regular SD 1.5 LoRA Support

You can now use regular SD 1.5 LoRAs (character LoRAs, style LoRAs, etc.) alongside motion modules and motion LoRAs. This allows you to customize the appearance of your animations while maintaining motion consistency.

To use regular SD LoRAs:
1. Provide a list of LoRA names in the `lora_names` parameter
2. Provide a corresponding list of strengths in the `lora_strengths` parameter

Example:
```python
pipeline.generate_with_live_preview(
    prompt="a beautiful portrait of a woman",
    sd_model_name="runwayml/stable-diffusion-v1-5",
    motion_module_name="mm_sd_v15_v2.ckpt",
    motion_lora_name="ZoomIn",
    motion_lora_strength=0.6,
    lora_names=["koreanDollLikeness", "epiNoiseoffset"],
    lora_strengths=[0.7, 0.4],
    num_frames=16,
    fps=8
)
```

### Model Compatibility Checking

The system now automatically checks for compatibility between:
- SD base models (SD 1.5, SDXL)
- Motion modules (V1, V2, V3, SDXL)
- Motion LoRAs

When incompatible combinations are detected:
1. Warning messages are logged explaining the compatibility issue
2. Suggestions for compatible alternatives are provided
3. The system attempts to continue with the provided models, but results may be unpredictable

This helps you identify and resolve compatibility issues quickly.

## Recommended Model Combinations

For best results, use these tested combinations:

### SD 1.5 Animation
- SD Model: "runwayml/stable-diffusion-v1-5"
- Motion Module: "mm_sd_v15_v2.ckpt"
- Motion LoRA: Any from Civitai (ZoomIn, PanLeft, etc.)
- Regular LoRAs: Any SD 1.5 compatible LoRAs

### SDXL Animation
- SD Model: "stabilityai/stable-diffusion-xl-base-1.0"
- Motion Module: "mm_sdxl_v1.0.safetensors"
- Motion LoRA: SDXL-specific motion LoRAs only
- Regular LoRAs: Any SDXL compatible LoRAs

## Troubleshooting

If you encounter errors:
1. Check the logs for compatibility warnings
2. Try the suggested compatible models
3. Ensure your motion LoRAs are compatible with your motion module
4. For tensor-related errors, try reducing the number of LoRAs or their strengths
