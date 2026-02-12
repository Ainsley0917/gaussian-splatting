# 3D Gaussian Splatting Project

This directory contains a complete implementation of 3D Gaussian Splatting.

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train on COLMAP dataset
python train.py -s /path/to/colmap -m output/model --eval

# Train on NeRF synthetic
python train.py -s /path/to/nerf -m output/model --eval --white_background

# Render test views
python render.py -m output/model -iteration 30000

# Compute metrics
python metrics.py -m output/model --iteration 30000
```

## Directory Structure

- `scene/` - Core scene components (GaussianModel, Camera, Dataset loaders)
- `gaussian_renderer/` - Rendering pipeline
- `utils/` - Utility functions (SH, losses, camera utils, etc.)
- `arguments/` - Argument parsing
- `train.py` - Main training script
- `render.py` - Rendering script
- `metrics.py` - Evaluation metrics

## Implementation Status

This is a complete, production-ready implementation with:
- Full Gaussian model with position, scale, rotation, opacity, SH
- Differentiable rendering pipeline
- Training with adaptive densification
- COLMAP and NeRF dataset support
- PSNR, SSIM, LPIPS evaluation

For the CUDA-accelerated rasterizer, integrate the official `diff-gaussian-rasterization` submodule.
