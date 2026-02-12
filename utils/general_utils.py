"""
General utility functions for 3DGS
"""

import torch
import numpy as np
import random
import os


def safe_state(silent):
    """
    Set random seeds for reproducibility and configure logging

    Args:
        silent: If True, suppress verbose output
    """
    old_f = os.dup(1)
    old_e = os.dup(2)

    # Set random seeds
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.device("cuda", 0))

    return old_f, old_e


def inverse_sigmoid(x):
    """
    Inverse sigmoid (logit) function
    Converts probabilities in (0, 1) to real numbers

    Args:
        x: Tensor with values in (0, 1)
    Returns:
        Tensor with log(x / (1 - x))
    """
    return torch.log(x / (1 - x))


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Get exponential learning rate decay function

    Args:
        lr_init: Initial learning rate
        lr_final: Final learning rate
        lr_delay_steps: Steps to delay before decay starts
        lr_delay_mult: Multiplier during delay period
        max_steps: Maximum number of steps

    Returns:
        Function that computes LR for a given step
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0

        if lr_delay_steps > 0:
            # Delay period
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        # Exponential decay
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def build_rotation(r):
    """
    Build 3x3 rotation matrix from quaternion(s)

    Args:
        r: Quaternion tensor (N, 4) or (4,) in [w, x, y, z] format
    Returns:
        Rotation matrix (N, 3, 3) or (3, 3)
    """
    if r.dim() == 1:
        r = r.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )
    q = r / norm[:, None]

    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.stack(
        [
            torch.stack(
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - r * z),
                    2.0 * (x * z + r * y),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2.0 * (x * y + r * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - r * x),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2.0 * (x * z - r * y),
                    2.0 * (y * z + r * x),
                    1.0 - 2.0 * (x * x + y * y),
                ],
                dim=-1,
            ),
        ],
        dim=1,
    )

    if squeeze:
        R = R.squeeze(0)

    return R


def build_scaling_rotation(s, r):
    """
    Build 3x3 scaling-rotation matrix S @ R

    Args:
        s: Scaling factors (N, 3)
        r: Quaternion rotations (N, 4)
    Returns:
        Scaling-rotation matrices (N, 3, 3)
    """
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def strip_lowerdiag(L):
    """
    Extract lower diagonal elements from 3x3 matrix
    Used for symmetric matrix representation

    Args:
        L: Lower triangular matrices (N, 3, 3)
    Returns:
        Lower diagonal elements (N, 6)
    """
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]

    return uncertainty


def strip_symmetric(sym):
    """
    Extract upper triangular part of symmetric matrix

    Args:
        sym: Symmetric matrices (N, 3, 3)
    Returns:
        Upper triangular elements (N, 6)
    """
    return sym[:, :, [0, 1, 2, 1, 2, 2]][:, [0, 1, 2, 3, 4, 5]]


def build_symmetric(uncertainty):
    """
    Build symmetric 3x3 matrix from 6 elements

    Args:
        uncertainty: (N, 6) - [x0, x1, x2, x3, x4, x5]
    Returns:
        Symmetric matrices (N, 3, 3)
    """
    idx = torch.tensor([0, 1, 2, 1, 3, 4, 2, 4, 5], device=uncertainty.device)

    def idx2_1(i):
        return i // 3, i % 3

    def idx2_2(i):
        return i // 3, i % 3

    sym = torch.zeros(
        (uncertainty.shape[0], 9), dtype=torch.float, device=uncertainty.device
    )
    for i in range(9):
        if i < 6:
            sym[:, i] = uncertainty[:, i]
        else:
            sym[:, i] = uncertainty[:, idx[i]]

    return sym.reshape(-1, 3, 3)


def PILtoTorch(pil_image, resolution):
    """
    Convert PIL image to torch tensor

    Args:
        pil_image: PIL Image
        resolution: Target resolution (W, H)
    Returns:
        Image tensor (3, H, W) normalized to [0, 1]
    """
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0

    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_spherical(cameras, center=torch.tensor([0, 0, 0], dtype=torch.float32)):
    """
    Compute spherical coordinates of cameras relative to center

    Args:
        cameras: List of Camera objects
        center: Center point for spherical coordinates
    Returns:
        theta, phi angles for each camera
    """
    poses = []
    for camera in cameras:
        pose = np.array(camera.get_extrinsic_matrix())
        poses.append(pose)

    poses = np.stack(poses, axis=0)
    cam_poses = poses[:, :3, 3]  # Extract camera positions

    # Compute relative positions
    relative_pos = cam_poses - center.numpy()

    # Convert to spherical coordinates
    r = np.linalg.norm(relative_pos, axis=1)
    theta = np.arccos(relative_pos[:, 1] / r)  # Polar angle
    phi = np.arctan2(relative_pos[:, 0], relative_pos[:, 2])  # Azimuthal angle

    return theta, phi


def render_net_image(render_pkg, render_mode):
    """
    Extract specific image from render package based on mode

    Args:
        render_pkg: Dictionary with render results
        render_mode: One of ['rgb', 'depth', 'alpha', 'normal']
    Returns:
        Image tensor
    """
    if render_mode == "rgb":
        return render_pkg["render"]
    elif render_mode == "depth":
        return render_pkg["depth"]
    elif render_mode == "alpha":
        return render_pkg["alpha"]
    elif render_mode == "normal":
        return render_pkg["normal"]
    else:
        raise ValueError(f"Unknown render mode: {render_mode}")


def colmap_camera_to_OpenCV_camera(camera):
    """
    Convert COLMAP camera parameters to OpenCV format

    Args:
        camera: COLMAP camera dict with 'model', 'width', 'height', 'params'
    Returns:
        K: 3x3 intrinsic matrix
        dist_coeff: distortion coefficients
    """
    width = camera["width"]
    height = camera["height"]
    params = camera["params"]

    if camera["model"] == "PINHOLE":
        fx, fy, cx, cy = params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist_coeff = np.zeros(4)
    elif camera["model"] == "SIMPLE_PINHOLE":
        f, cx, cy = params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        dist_coeff = np.zeros(4)
    else:
        raise ValueError(f"Unsupported camera model: {camera['model']}")

    return K, dist_coeff


def depth2normal(depth, K, windowsize=5):
    """
    Compute surface normals from depth map

    Args:
        depth: Depth map tensor (H, W)
        K: Intrinsic matrix
        windowsize: Window size for normal estimation
    Returns:
        Normal map (3, H, W)
    """
    h, w = depth.shape

    # Create pixel coordinate grids
    u, v = torch.meshgrid(
        torch.arange(w, device=depth.device),
        torch.arange(h, device=depth.device),
        indexing="xy",
    )

    # Backproject to 3D
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = torch.stack([x, y, z], dim=-1)  # (H, W, 3)

    # Compute normals using finite differences
    half_w = windowsize // 2
    normals = torch.zeros((h, w, 3), device=depth.device)

    for i in range(half_w, h - half_w):
        for j in range(half_w, w - half_w):
            p = points[i, j]
            pu = points[i + half_w, j]
            pv = points[i, j + half_w]

            du = pu - p
            dv = pv - p

            n = torch.cross(dv, du)
            n = n / (torch.norm(n) + 1e-8)
            normals[i, j] = n

    return normals.permute(2, 0, 1)  # (3, H, W)
