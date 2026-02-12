# 3D GAUSSIAN SPLATTING KNOWLEDGE BASE

**Generated:** 2026-02-09  
**Last updated:** 2026-02-12

## OVERVIEW

Complete 3D Gaussian Splatting implementation: differentiable rendering of anisotropic 3D Gaussians for novel view synthesis. PyTorch 2.0 / CUDA 11.8+ / Python 3.10.

## PROJECT STATUS

- **Git**: 已初始化并完成首个 commit（`527d0e1`），已配置远端 `origin`（`git@github.com:Ainsley0917/gaussian-splatting.git`），当前仓库为 GitHub Public
- **虚拟环境**: `uv venv --python 3.10 .venv`（Python 3.10.19, macOS aarch64）
- **依赖管理**: `requirements.txt` / `setup.py` / `pyproject.toml` 三处已同步为精确固定版本（`==`）
- **依赖安装**: 开发机/实验机尚未完成全量安装；Kaggle 已按最小依赖跑通 smoke test
- **内网 PyPI 源**: `http://artifactory.intra.xiaojukeji.com/artifactory/api/pypi/pypi/simple`
- **已有训练产出**: `output/lego_smoke/`, `output/lego_smoke_fast/`（NeRF synthetic lego 数据集）
- **Kaggle 跑通记录**: `runs/lego_tiny_100`（100 iter，PSNR 9.6484 / SSIM 0.6200 / LPIPS 0.6304）
- **课题文档**: `THESIS_INFO.md`（课题/任务书/开题信息）
- **周记归档**: `weekly_reports/week05_2026-02-12.md`（第5周周记，当前最新）

## DEPLOYMENT STRATEGY

**当前环境角色：**
- **开发机**（当前 macOS）：代码开发、依赖调试、本地测试
- **实验机**（内网 Linux）：模型训练、大规模实验

**代码同步策略：**
1. 开发机维护代码仓库，推送到 GitHub（私有仓库）
2. 实验机通过 GitHub 拉取代码（支持换端口/代理访问）
3. 实验机连 GitHub 不通时，可改用公司 GitLab 或开发机 `scp` 传输

**实验机访问 GitHub 方案：**
```bash
# 方案1：SSH 默认端口 22
git@github.com:username/repo.git

# 方案2：SSH 走 443 端口（常用绕过）
# ~/.ssh/config
Host github.com
    Hostname ssh.github.com
    Port 443
    User git

# 方案3：走公司代理
git config --global http.proxy http://proxy.company.com:8080
```

## STRUCTURE

```
gaussian_splatting/
├── train.py              # Training entry point
├── render.py             # Inference/rendering entry
├── metrics.py            # PSNR/SSIM/LPIPS evaluation
├── requirements.txt      # 精确固定版本依赖（==）
├── setup.py              # setuptools 包元数据（依赖与 requirements.txt 同步）
├── pyproject.toml        # PEP 621 项目配置（依赖与 requirements.txt 同步）
├── THESIS_INFO.md        # 课题/任务书/开题信息
├── IMPLEMENTATION.md     # 实现说明文档
├── weekly_reports/       # 周记归档
│   ├── week04_2025-12-16.md  # 第4周周记
│   └── week05_2026-02-12.md  # 第5周周记（环境搭建 + Kaggle pipeline 跑通）
├── scene/                # Core 3DGS domain (see scene/AGENTS.md)
│   ├── __init__.py       # Scene class
│   ├── gaussian_model.py # GaussianModel - THE central class
│   ├── cameras.py        # Camera with precomputed matrices
│   └── dataset_readers.py# COLMAP/NeRF data loaders
├── gaussian_renderer/    # Rendering pipeline
│   └── __init__.py       # PyTorch fallback rasterizer
├── submodules/           # CUDA 子模块
│   └── diff_gaussian_rasterization/  # CUDA 高性能光栅化器
├── utils/                # Shared utilities
│   ├── sh_utils.py       # Spherical harmonics
│   ├── graphics_utils.py # Projection matrices
│   ├── loss_utils.py     # L1 + SSIM loss
│   ├── point_cloud_utils.py # PLY I/O
│   ├── camera_utils.py   # Camera helpers
│   ├── general_utils.py  # General utilities
│   └── image_utils.py    # Image utilities
├── arguments/            # CLI argument groups
├── data/                 # 训练数据
│   └── nerf_synthetic/lego/  # NeRF synthetic lego 数据集
├── output/               # 训练产出
│   ├── lego_smoke/       # lego 训练结果
│   └── lego_smoke_fast/  # lego 快速训练结果
├── tests/                # 测试目录（空，仅 .keep）
└── .venv/                # uv 虚拟环境（Python 3.10.19, 未安装依赖）
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Modify Gaussian representation | `scene/gaussian_model.py` | Parameters: `_xyz`, `_scaling`, `_rotation`, `_opacity`, `_features_*` |
| Change training loop | `train.py` | Densification in `training()` after line 138 |
| Add dataset format | `scene/dataset_readers.py` | Register in `sceneLoadTypeCallbacks` dict |
| Modify rendering math | `gaussian_renderer/__init__.py` | `GaussianRasterizer.forward()` |
| Add loss function | `utils/loss_utils.py` | Combine in `train.py` loss computation |
| Change camera model | `scene/cameras.py` | Projection in `getProjectionMatrix()` |
| Adjust SH degree | `arguments/__init__.py` | `ModelParams.sh_degree` (0-3) |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `GaussianModel` | class | `scene/gaussian_model.py:30` | Core - stores all trainable parameters |
| `Scene` | class | `scene/__init__.py:17` | Container for cameras + GaussianModel |
| `Camera` | class | `scene/cameras.py:11` | Precomputes world_view/projection matrices |
| `render()` | function | `gaussian_renderer/__init__.py:429` | Main render entry - camera + model → image |
| `GaussianRasterizer` | class | `gaussian_renderer/__init__.py:48` | EWA splatting, alpha blending |
| `training()` | function | `train.py:28` | Main training loop with densification |
| `densify()` | method | `scene/gaussian_model.py:596` | Adaptive density control |

## CONVENTIONS

- **Line length**: 100 chars (Black + isort)
- **Short variables allowed**: `i, j, k, v, x, y, z, R, T, K` (graphics conventions)
- **Quaternion format**: `[w, x, y, z]` (scalar-first)
- **Image format**: `(C, H, W)` PyTorch tensors, `[0, 1]` normalized
- **Coordinate system**: COLMAP convention (Y-down in camera space)

## ANTI-PATTERNS

- **NEVER** use `as any`/`@ts-ignore` equivalents - full typing required
- **NEVER** modify optimizer state directly - use `GaussianModel.replace_tensor_to_optimizer()`
- **NEVER** skip sigmoid/exp activations - parameters stored in unconstrained space
- **AVOID** CPU tensors in render path - all rendering on CUDA
- **AVOID** per-Gaussian Python loops - use vectorized ops (PyTorch fallback is slow)

## UNIQUE PATTERNS

### Parameter Activation
Parameters stored unconstrained, activated on read:
```python
# Stored: _scaling (log-space), _opacity (logit-space)
# Read: get_scaling → exp(), get_opacity → sigmoid()
```

### Dynamic Topology
Gaussians added/removed during training:
```python
gaussians.densify_and_split()  # Large Gaussians → 2 smaller
gaussians.densify_and_clone()  # Small Gaussians → duplicate
gaussians.prune_points()       # Remove low-opacity
```

### Per-Group Learning Rates
```python
# xyz: position_lr_init (decays exponentially)
# f_dc: feature_lr
# f_rest: feature_lr / 20
# opacity/scaling/rotation: separate fixed LRs
```

## COMMANDS

```bash
# Install
pip install -e .

# Train on COLMAP
python train.py -s /path/to/colmap -m output/model --eval

# Train on NeRF synthetic (white bg)
python train.py -s /path/to/nerf -m output/model --eval --white_background

# Render test views
python render.py -m output/model --iteration 30000

# Compute metrics
python metrics.py -m output/model --iteration 30000

# Format code
black . && isort .
```

## NOTES

- **PyTorch fallback**: `gaussian_renderer/__init__.py` contains pure-PyTorch rasterizer - works without CUDA but 100x slower. For production, compile `submodules/diff_gaussian_rasterization`.
- **tests/ empty**: Pytest configured but no unit tests. "Testing" = model quality evaluation via `metrics.py`.
- **SH coefficients**: DC term (`_features_dc`) separate from higher orders (`_features_rest`) for different LRs.
- **Densification stats**: `xyz_gradient_accum` tracks view-space gradients to decide where to add Gaussians.
