"""
Image utilities - PSNR, image operations
"""

import torch
import torch.nn.functional as F
import numpy as np


def psnr(img1, img2, max_val=1.0):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio)

    Args:
        img1: First image tensor
        img2: Second image tensor
        max_val: Maximum pixel value

    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return torch.tensor(100.0)
    return 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))


def mse2psnr(mse, max_val=1.0):
    """
    Convert MSE to PSNR

    Args:
        mse: Mean squared error
        max_val: Maximum pixel value

    Returns:
        PSNR value
    """
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(max_val / np.sqrt(mse))


def img2mse(img_src, img_tgt):
    """
    Calculate MSE between two images

    Args:
        img_src: Source image
        img_tgt: Target image

    Returns:
        MSE value
    """
    return torch.mean((img_src - img_tgt) ** 2)


def normalize_image(img):
    """
    Normalize image to [0, 1] range

    Args:
        img: Image tensor

    Returns:
        Normalized image
    """
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min + 1e-8)


def resize_image(img, size):
    """
    Resize image using bilinear interpolation

    Args:
        img: Image tensor (C, H, W) or (N, C, H, W)
        size: Target size (H, W)

    Returns:
        Resized image
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    resized = F.interpolate(img, size=size, mode="bilinear", align_corners=False)

    if squeeze:
        resized = resized.squeeze(0)

    return resized


def save_image(img, path):
    """
    Save image to file

    Args:
        img: Image tensor (C, H, W) in [0, 1]
        path: Output path
    """
    import torchvision

    torchvision.utils.save_image(img, path)


def load_image(path):
    """
    Load image from file

    Args:
        path: Image path

    Returns:
        Image tensor (C, H, W) in [0, 1]
    """
    from PIL import Image

    img = Image.open(path).convert("RGB")
    img = np.array(img) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def apply_colormap(depth, cmap="jet"):
    """
    Apply colormap to depth map

    Args:
        depth: Depth map tensor (H, W) or (1, H, W)
        cmap: Colormap name

    Returns:
        Colored depth map (3, H, W)
    """
    import matplotlib.pyplot as plt

    if depth.dim() == 3 and depth.shape[0] == 1:
        depth = depth.squeeze(0)

    depth_np = depth.cpu().numpy()

    # Normalize
    depth_normalized = (depth_np - depth_np.min()) / (
        depth_np.max() - depth_np.min() + 1e-8
    )

    # Apply colormap
    cmap_func = plt.get_cmap(cmap)
    colored = cmap_func(depth_normalized)[:, :, :3]  # Remove alpha

    # Convert to tensor
    colored_tensor = torch.from_numpy(colored).permute(2, 0, 1).float()

    return colored_tensor


def create_video_from_frames(frame_dir, output_path, fps=30):
    """
    Create video from frame images

    Args:
        frame_dir: Directory containing frame images
        output_path: Output video path
        fps: Frames per second
    """
    import imageio
    import os

    frames = []
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])

    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = imageio.imread(frame_path)
        frames.append(frame)

    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Video saved to {output_path}")
