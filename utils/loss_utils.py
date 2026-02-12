"""
Loss functions for 3D Gaussian Splatting training
"""

import torch
import torch.nn.functional as F
from math import exp


def l1_loss(network_output, gt):
    """
    L1 loss (mean absolute error)

    Args:
        network_output: Predicted values
        gt: Ground truth values

    Returns:
        L1 loss value
    """
    return torch.abs(network_output - gt).mean()


def l2_loss(network_output, gt):
    """
    L2 loss (mean squared error)

    Args:
        network_output: Predicted values
        gt: Ground truth values

    Returns:
        L2 loss value
    """
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    """
    Create Gaussian kernel

    Args:
        window_size: Size of the window
        sigma: Standard deviation

    Returns:
        1D Gaussian kernel
    """
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """
    Create 2D Gaussian window for SSIM

    Args:
        window_size: Size of the window
        channel: Number of channels

    Returns:
        2D Gaussian kernel (channel, 1, window_size, window_size)
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    Compute SSIM between two images

    Args:
        img1: First image (N, C, H, W)
        img2: Second image (N, C, H, W)
        window: Gaussian window
        window_size: Window size
        channel: Number of channels
        size_average: Whether to average over batch

    Returns:
        SSIM value
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """
    Calculate SSIM (Structural Similarity Index)

    Args:
        img1: First image (N, C, H, W)
        img2: Second image (N, C, H, W)
        window_size: SSIM window size
        size_average: Whether to average

    Returns:
        SSIM value
    """
    (_, channel, _, _) = img1.size()

    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return ssim(img1, img2, window, window_size, channel, size_average)


def psnr(img1, img2, max_val=1.0):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio)

    Args:
        img1: Predicted image
        img2: Ground truth image
        max_val: Maximum pixel value

    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(max_val / torch.sqrt(mse))


class SSIMLoss(torch.nn.Module):
    """SSIM loss module"""

    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1.0 - ssim(
            img1, img2, window, self.window_size, channel, self.size_average
        )


def ssim_loss(img1, img2, window_size=11):
    """
    SSIM loss (1 - SSIM)

    Args:
        img1: First image
        img2: Second image
        window_size: Window size

    Returns:
        SSIM loss
    """
    return 1.0 - calculate_ssim(img1, img2, window_size)


def combined_loss(pred, gt, lambda_ssim=0.2):
    """
    Combined L1 + SSIM loss used in 3DGS

    Args:
        pred: Predicted image
        gt: Ground truth image
        lambda_ssim: Weight for SSIM loss

    Returns:
        Combined loss
    """
    l1 = l1_loss(pred, gt)
    ssim = ssim_loss(pred, gt)
    return (1.0 - lambda_ssim) * l1 + lambda_ssim * ssim


def depth_loss(pred_depth, gt_depth, mask=None):
    """
    Depth loss (optional for depth-supervised training)

    Args:
        pred_depth: Predicted depth
        gt_depth: Ground truth depth
        mask: Optional mask for valid depth

    Returns:
        Depth loss
    """
    if mask is not None:
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

    return l1_loss(pred_depth, gt_depth)


def normal_loss(pred_normal, gt_normal, mask=None):
    """
    Normal loss using cosine similarity

    Args:
        pred_normal: Predicted normals
        gt_normal: Ground truth normals
        mask: Optional mask

    Returns:
        Normal loss
    """
    # Normalize
    pred_normal = F.normalize(pred_normal, dim=0)
    gt_normal = F.normalize(gt_normal, dim=0)

    if mask is not None:
        pred_normal = pred_normal[:, mask]
        gt_normal = gt_normal[:, mask]

    # Cosine similarity
    similarity = (pred_normal * gt_normal).sum(dim=0)
    loss = 1.0 - similarity.mean()

    return loss


def opacity_loss(opacity, target=0.01):
    """
    Opacity regularization loss

    Args:
        opacity: Opacity values
        target: Target opacity value

    Returns:
        Opacity loss
    """
    return torch.abs(opacity - target).mean()


def scale_loss(scales, target_ratio=10.0):
    """
    Scale regularization to prevent extreme anisotropic Gaussians

    Args:
        scales: Scale values
        target_ratio: Target max/min scale ratio

    Returns:
        Scale regularization loss
    """
    max_scale = scales.max(dim=-1)[0]
    min_scale = scales.min(dim=-1)[0]
    ratio = max_scale / (min_scale + 1e-7)
    violation = torch.relu(ratio - target_ratio)
    return violation.mean()
