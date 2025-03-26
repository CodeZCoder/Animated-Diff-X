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

(((ADVANCE MODEL COMPATABILTY FOR ANIMATE DIFF BELOW)))

The Ultimate AnimateDiff Compatibility Guide (Everything You Need to Know)

Introduction: The Golden Rule of AnimateDiff

AnimateDiff is incredibly powerful for creating animations with Stable Diffusion, but getting different models and tools to work together can be confusing. There's ONE RULE that solves 90% of compatibility problems:

!!! ALL COMPONENTS MUST MATCH THE BASE STABLE DIFFUSION VERSION (SD 1.5 or SDXL) !!!

Mixing components designed for SD 1.5 with components designed for SDXL (or vice-versa) is the #1 cause of errors, crashes, and broken animations. Choose your path (SD 1.5 or SDXL) and stick to it for everything.

Part 1: Detailed Component Breakdown

Here’s what each piece does and its compatibility requirement:

Base Stable Diffusion Checkpoint:

What it is: The main model defining the core look, style, and knowledge (e.g., Realistic Vision, DreamShaper, SDXL Base).

Compatibility: This defines your pathway (SD 1.5 or SDXL). All other components must align with this choice.

Examples:

SD 1.5: Realistic Vision v6, DreamShaper 8, Anything v5, ToonYou Beta6, custom SD 1.5 merges.

SDXL: SDXL 1.0 Base, DreamShaper XL, Juggernaut XL, RealVisXL, custom SDXL merges.

AnimateDiff Motion Module:

What it is: The core AnimateDiff model that injects motion understanding into the generation process.

Compatibility: MUST MATCH the Base Checkpoint version (SD 1.5 or SDXL).

Examples:

SD 1.5: mm_sd_v14, mm_sd_v15, mm_sd_v15_v2, v3_sd15_mm, specific fine-tunes.

SDXL: mm_sdxl_v10_beta, HotshotXL Temporal Layers, specific fine-tunes.

IP-Adapter Model:

What it is: Uses a reference image to guide the style, composition, or character appearance in the output (Image Prompting).

Compatibility: MUST MATCH the Base Checkpoint version (SD 1.5 or SDXL). Requires a compatible CLIP Vision Encoder.

Examples:

SD 1.5: ip-adapter_sd15, ip-adapter-plus_sd15, ip-adapter-face_sd15. (Often uses ViT-H CLIP Vision).

SDXL: ip-adapter_sdxl, ip-adapter-plus_sdxl_vit-h. (Often uses ViT-H or ViT-G CLIP Vision - check model specifics).

Note: Adjust weight carefully to balance reference image influence with motion/prompt.

Standard LoRA (Style, Character, Concept):

What it is: Modifies the Base Checkpoint for specific looks, characters, or objects.

Compatibility: MUST MATCH the Base Checkpoint version (SD 1.5 or SDXL).

Note: High weights can sometimes interfere with motion consistency or cause flickering. Reduce weight (e.g., 0.5-0.8) if issues occur.

Motion LoRA (Camera Motion):

What it is: Special LoRAs designed to add specific camera movements (pan, tilt, zoom, rotate) on top of the Motion Module.

Compatibility: MUST MATCH the Base Checkpoint version. CRITICAL: Most common SD 1.5 Motion LoRAs only work correctly with the mm_sd_v15_v2 Motion Module. Always check the Motion LoRA's documentation. SDXL Motion LoRAs are less common or part of specific systems (e.g., MotionDirector).

Note: Weights often work best around 0.7-0.8.

LCM LoRA (Latent Consistency Model):

What it is: Used to dramatically speed up inference time.

Compatibility: MUST MATCH the Base Checkpoint version (SD 1.5 or SDXL).

Note: Requires using the LCM sampler, very low step counts (e.g., 4-8), and low CFG Scale (e.g., 1.0-2.0).

ControlNet Model:

What it is: Provides structural control using conditioning images (pose, depth, canny edge, lineart, etc.).

Compatibility: MUST MATCH the Base Checkpoint version (SD 1.5 or SDXL).

Note: Ensure your ControlNet preprocessor also matches the SD version. Essential for guiding character poses or scene structure consistently across frames.

VAE (Variational Autoencoder):

What it is: Handles the final decoding from latent space to pixel space, impacting colors and fine details.

Compatibility: MUST MATCH the Base Checkpoint version (SD 1.5 or SDXL).

Examples:

SD 1.5: vae-ft-mse-840000-ema-pruned, or often baked into the checkpoint.

SDXL: Specific SDXL VAE, often baked into the checkpoint.

Note: Using the wrong VAE leads to washed-out colors or artifacts. Best to use the one recommended for/baked into your Base Checkpoint.

Part 2: Advanced Compatibility Chart

Use this table as a quick reference. Pick a column (SD 1.5 or SDXL) and ensure ALL your chosen components fall within that column.

Component Type	✅ SD 1.5 Generation Pathway	✅ SDXL Generation Pathway	⚠️ Compatibility Rule / Key Notes
Base Checkpoint	Any SD 1.5 Model<br/>(e.g., Realistic Vision, DreamShaper 8, Anything V5)	Any SDXL 1.0 Model<br/>(e.g., SDXL Base, DreamShaper XL, Juggernaut XL)	Defines the pathway. All others MUST align.
Motion Module	SD 1.5 Motion Module<br/>(e.g., mm_sd_v15_v2, v3_sd15_mm)	SDXL Motion Module<br/>(e.g., mm_sdxl_v10_beta, HotshotXL)	MUST match Base. Core AnimateDiff part. NO MIXING!
IP-Adapter Model	SD 1.5 IP-Adapter<br/>(e.g., ip-adapter-plus_sd15, requires SD1.5 CLIP Vision)	SDXL IP-Adapter<br/>(e.g., ip-adapter-plus_sdxl_vit-h, requires SDXL CLIP Vision)	MUST match Base. Needs correct CLIP Vision model too. Weight tuning needed.
Standard LoRA	Any SD 1.5 LoRA<br/>(Style, Character, Concept)	Any SDXL LoRA<br/>(Style, Character, Concept)	MUST match Base. Adjust weight if motion/style breaks.
Motion LoRA	SD 1.5 Motion LoRA<br/>(Often ONLY works with mm_sd_v15_v2)	SDXL Motion LoRA<br/>(Less common, check specific system docs)	MUST match Base. Check specific Motion Module dependency!
LCM LoRA	SD 1.5 LCM LoRA	SDXL LCM LoRA	MUST match Base. Requires LCM sampler, low steps/CFG.
ControlNet Model	Any SD 1.5 ControlNet<br/>(e.g., control_v11p_sd15_openpose)	Any SDXL ControlNet<br/>(e.g., thibaud_xl_openpose)	MUST match Base. Use matching preprocessor.
VAE	SD 1.5 VAE<br/>(e.g., vae-ft-mse-840000-ema-pruned or baked-in)	SDXL VAE<br/>(Often baked-in or specific SDXL version)	MUST match Base. Use recommended/baked-in VAE for best colors/details.
Part 3: The "AnimateDiff For Dummies" Compatibility Flowchart

Think of it like choosing items from ONLY ONE MENU: Menu A (SD 1.5) or Menu B (SDXL). You CANNOT order appetisers from Menu A and a main course from Menu B!

graph TD
    A[Start: Make Animation!] --> B{Choose Style Base};

    subgraph SD 1.5 Pathway (Menu A)
    B --> C[Pick SD 1.5 Base Model];
    C --> D[Pick SD 1.5 Motion Module];
    D --> E[Pick SD 1.5 LoRAs (Optional)];
    E --> F[Pick SD 1.5 ControlNet (Optional)];
    F --> G[Pick SD 1.5 IP-Adapter (Optional)];
    G --> H[Pick SD 1.5 VAE];
    H --> Z[✅ Generate SD 1.5 Animation];
    end

    subgraph SDXL Pathway (Menu B)
    B --> I[Pick SDXL Base Model];
    I --> J[Pick SDXL Motion Module];
    J --> K[Pick SDXL LoRAs (Optional)];
    K --> L[Pick SDXL ControlNet (Optional)];
    L --> M[Pick SDXL IP-Adapter (Optional)];
    M --> N[Pick SDXL VAE];
    N --> Z2[✅ Generate SDXL Animation];
    end

    style Z fill:#ccffcc,stroke:#333,stroke-width:2px;
    style Z2 fill:#ccffcc,stroke:#333,stroke-width:2px;

    %% Emphasize NO MIXING
    O[❌ DO NOT MIX ❌] -- Absolutely NO --> P{Components from Menu A + Menu B};
    style O fill:#ffcccc,stroke:#f00,stroke-width:4px,color:#f00;
    style P fill:#ffcccc,stroke:#f00,stroke-width:2px;
Use code with caution.
Mermaid
Simple Steps:

Decide: Do you want the SD 1.5 look or the SDXL look? This is your ONLY menu choice.

Select Base Model: Pick a Checkpoint from your chosen menu (SD 1.5 or SDXL).

Select Motion Module: Pick the Motion Module from the SAME menu.

Select Optional Extras (LoRAs, ControlNet, IP-Adapter): If you use these, pick them ONLY from the SAME menu.

Select VAE: Pick the VAE from the SAME menu (or use the one built-in).

Generate! Everything matches, so it should work!

Part 4: Valid vs. Invalid Examples

✅ Valid SD 1.5 Example:

Base: DreamShaper 8 (SD 1.5)

Motion: mm_sd_v15_v2 (SD 1.5)

LoRA: epiNoiseoffset (SD 1.5)

ControlNet: control_v11p_sd15_openpose (SD 1.5)

VAE: vae-ft-mse-840000 (SD 1.5)

✅ Valid SDXL Example:

Base: Juggernaut XL (SDXL)

Motion: mm_sdxl_v10_beta (SDXL)

IP-Adapter: ip-adapter-plus_sdxl_vit-h (SDXL)

ControlNet: thibaud_xl_openpose (SDXL)

VAE: Baked-in SDXL VAE

❌ Invalid Example (WILL FAIL!):

Base: DreamShaper 8 (SD 1.5)

Motion: mm_sdxl_v10_beta (SDXL - MISMATCH!)

LoRA: Some SDXL Style LoRA (SDXL - MISMATCH!)

ControlNet: control_v11p_sd15_openpose (SD 1.5)

VAE: Baked-in SDXL VAE (SDXL - MISMATCH!)

Part 5: Common Troubleshooting (Compatibility Focus)

Error Message on Load/Generate: 99% chance you mixed SD 1.5 and SDXL components. Double-check everything against the chart/flowchart. Check model file paths.

No Motion / Static Image: Is the AnimateDiff extension/node enabled? Is the Motion Module correctly selected and loaded?

Garbled Output / Weird Colors: Check VAE compatibility (SD 1.5 VAE for SD 1.5, SDXL VAE for SDXL). Reduce LoRA weights if they are too high.

Flickering / Inconsistency: Reduce LoRA weights (especially Style/Character LoRAs). Simplify prompt. Try a different Motion Module (some are smoother). Check IP-Adapter weight.

Motion LoRA Not Working: Are you using an SD 1.5 Motion LoRA? It probably needs the mm_sd_v15_v2 motion module.

Final Disclaimer: The world of AI image/video generation moves fast! New models and techniques appear constantly. Always check the specific documentation for any new models, nodes, or extensions you download. When in doubt, start simple (Base + Motion Module) and add complexity one step at a time.

This comprehensive guide should cover nearly all compatibility scenarios you'll encounter with AnimateDiff! Good luck animating!
