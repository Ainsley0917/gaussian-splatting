# SCENE MODULE

Core 3DGS domain: Gaussian representation, cameras, and data loading.

## OVERVIEW

This module owns the **data model** for 3D Gaussian Splatting. Everything that represents "what we're rendering" lives here.

## STRUCTURE

```
scene/
├── __init__.py         # Scene class - orchestrates loading
├── gaussian_model.py   # GaussianModel - THE central class
├── cameras.py          # Camera with precomputed GPU matrices
└── dataset_readers.py  # COLMAP/NeRF parsers
```

## WHERE TO LOOK

| Task | File | Function/Class |
|------|------|----------------|
| Add Gaussian parameter | `gaussian_model.py` | Add to `__init__`, `create_from_pcd`, `training_setup`, `save_ply`, `load_ply` |
| Change densification logic | `gaussian_model.py` | `densify()`, `densify_and_split()`, `densify_and_clone()` |
| Add dataset format | `dataset_readers.py` | Create reader function, register in `sceneLoadTypeCallbacks` |
| Modify camera intrinsics | `cameras.py` | `__init__()`, `get_intrinsic_matrix()` |
| Change train/test split | `dataset_readers.py` | `llffhold` parameter in `readColmapSceneInfo()` |

## KEY ABSTRACTIONS

### GaussianModel Parameters
```python
_xyz            # (N, 3) positions - unconstrained
_scaling        # (N, 3) log-scales → exp() on read
_rotation       # (N, 4) quaternions [w,x,y,z] → normalize() on read
_opacity        # (N, 1) logits → sigmoid() on read
_features_dc    # (N, 1, 3) SH DC term (color)
_features_rest  # (N, 15, 3) SH higher orders (view-dependent)
```

### Data Flow
```
SceneInfo → Scene.__init__() → GaussianModel.create_from_pcd()
                            → cameraList_from_camInfos() → Camera objects
```

## ANTI-PATTERNS

- **NEVER** access `_xyz`, `_scaling` etc. directly for rendering - use `get_xyz`, `get_scaling` properties
- **NEVER** call `optimizer.add_param_group()` - use `cat_tensors_to_optimizer()` for dynamic topology
- **NEVER** forget to update `xyz_gradient_accum`, `denom`, `max_radii2D` when adding/removing points

## UNIQUE PATTERNS

### Optimizer Surgery
When Gaussians are added/removed, optimizer state must be updated:
```python
# Adding: cat_tensors_to_optimizer() extends exp_avg, exp_avg_sq
# Removing: _prune_optimizer() slices optimizer state with mask
# Replacing: replace_tensor_to_optimizer() resets momentum
```

### Camera Matrix Precomputation
All matrices computed once in `Camera.__init__()` and stored on GPU:
- `world_view_transform` - extrinsics (world → camera)
- `projection_matrix` - intrinsics + clip (camera → NDC)
- `full_proj_transform` - combined (world → NDC)
- `camera_center` - position in world space

### Dataset Detection
```python
# Auto-detect in Scene.__init__:
if exists("sparse/"):     → COLMAP
elif exists("transforms_train.json"): → Blender/NeRF
```
