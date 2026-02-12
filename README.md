# 3D Gaussian Splatting - Complete Implementation

A full-featured implementation of **3D Gaussian Splatting for Real-Time Radiance Field Rendering**.

## Overview

This project implements the complete 3D Gaussian Splatting pipeline, including:
- **Core Gaussian Model**: Position, covariance (scale + rotation), opacity, and spherical harmonics
- **Rendering Pipeline**: Differentiable Gaussian rasterization
- **Training System**: Adaptive densification and optimization
- **Data Loading**: COLMAP and NeRF synthetic dataset support
- **Evaluation**: PSNR, SSIM, and LPIPS metrics

## Features

- **Real-time Rendering**: Efficient 3D Gaussian representation
- **Adaptive Densification**: Automatic Gaussian split/clone/prune
- **Spherical Harmonics**: Up to degree 3 for view-dependent appearance
- **Multiple Datasets**: Support for COLMAP and NeRF synthetic data
- **Complete Pipeline**: Training, evaluation, and rendering scripts

## Installation

### Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- PyTorch 2.0+

### Install Dependencies

```bash
cd gaussian_splatting
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Quick Start

### Training on COLMAP Dataset

```bash
python train.py -s /path/to/colmap/dataset -m /path/to/output --eval
```

### Training on NeRF Synthetic Dataset

```bash
python train.py -s /path/to/nerf/dataset -m /path/to/output --eval --white_background
```

### Rendering Test Views

```bash
python render.py -m /path/to/output --iteration 30000
```

### Computing Metrics

```bash
python metrics.py -m /path/to/output --iteration 30000
```

## Project Structure

```
gaussian_splatting/
├── scene/                      # Core scene components
│   ├── __init__.py            # Scene class
│   ├── gaussian_model.py      # Gaussian model with parameters
│   ├── cameras.py             # Camera class
│   └── dataset_readers.py     # Data loaders (COLMAP, NeRF)
├── gaussian_renderer/         # Rendering pipeline
│   └── __init__.py           # Rasterizer and render function
├── utils/                     # Utility functions
│   ├── general_utils.py      # Helper functions
│   ├── graphics_utils.py     # Camera matrices, projections
│   ├── point_cloud_utils.py  # PLY file operations
│   ├── sh_utils.py          # Spherical harmonics
│   ├── camera_utils.py      # Camera utilities
│   └── loss_utils.py        # Loss functions
├── arguments/                 # Argument definitions
│   └── __init__.py
├── train.py                  # Training script
├── render.py                 # Rendering script
├── metrics.py                # Evaluation script
├── setup.py                  # Package setup
└── requirements.txt          # Dependencies
```

## Training Parameters

### Key Hyperparameters

- `--iterations`: Total training iterations (default: 30,000)
- `--sh_degree`: Spherical harmonics degree 0-3 (default: 3)
- `--resolution`: Training resolution -1 for original (default: -1)
- `--white_background`: Use white background (default: False)

### Optimization Parameters

- `--position_lr_init`: Position learning rate initial (default: 0.00016)
- `--feature_lr`: Feature learning rate (default: 0.0025)
- `--opacity_lr`: Opacity learning rate (default: 0.05)
- `--scaling_lr`: Scaling learning rate (default: 0.005)
- `--rotation_lr`: Rotation learning rate (default: 0.001)

### Densification Parameters

- `--densify_from_iter`: Start densification at iteration (default: 500)
- `--densify_until_iter`: Stop densification at iteration (default: 15,000)
- `--densification_interval`: Densify every N iterations (default: 100)
- `--densify_grad_threshold`: Gradient threshold for densification (default: 0.0002)

## Dataset Format

### COLMAP Format

The dataset directory should contain:
```
dataset/
├── images/              # Input images
└── sparse/
    └── 0/
        ├── cameras.bin  # Camera parameters
        ├── images.bin   # Image poses
        └── points3D.bin # Initial point cloud
```

### NeRF Synthetic Format

```
dataset/
├── transforms_train.json
├── transforms_test.json
└── [train/test images]
```

## Performance

Expected training times on different GPUs:

| GPU | Training Time (30k iter) | Rendering FPS |
|-----|-------------------------|---------------|
| RTX 4090 | ~5-10 minutes | ~100+ FPS |
| RTX 3090 | ~10-15 minutes | ~60+ FPS |
| V100 | ~15-20 minutes | ~30+ FPS |

## Expected Results

On standard benchmark datasets:

| Dataset | PSNR | SSIM | LPIPS |
|---------|------|------|-------|
| NeRF Synthetic | ~33+ dB | ~0.97+ | ~0.05- |
| Mip-NeRF 360 | ~29+ dB | ~0.90+ | ~0.15- |
| Tanks & Temples | ~23+ dB | ~0.85+ | ~0.20- |

## Implementation Details

### Core Data Structure

Each 3D Gaussian is represented by:
- **Position** (x, y, z): 3D center
- **Covariance** (scale + rotation): Anisotropic 3D Gaussian
- **Opacity**: Alpha value with sigmoid activation
- **Spherical Harmonics**: View-dependent color (up to degree 3)

### Training Pipeline

1. **Initialization**: From SfM point cloud or random points
2. **Forward Pass**: Render Gaussians to image
3. **Loss Computation**: L1 + SSIM loss
4. **Backward Pass**: Compute gradients
5. **Densification**: Split/clone/prune based on gradient magnitude
6. **Optimization**: Adam optimizer with different learning rates

### Rendering

- Project 3D Gaussians to 2D using EWA splatting
- Sort by depth (back-to-front)
- Alpha blending for final color
- Differentiable for training

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{kerbl20233d,
  title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  booktitle={ACM Transactions on Graphics},
  year={2023}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- Original paper: [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- Inspired by the official implementation and community contributions

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use lower resolution
2. **Poor quality**: Increase training iterations or adjust densification thresholds
3. **Slow training**: Ensure CUDA is properly installed and being used

### Contact

For issues and questions, please open an issue on GitHub.
