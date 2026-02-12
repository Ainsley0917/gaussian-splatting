"""
Spherical Harmonics utilities - RGB to SH conversion and evaluation
"""

import torch
import numpy as np


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.3153915652525205,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate spherical harmonics basis functions given an active degree

    Args:
        deg: Active SH degree (0-3)
        sh: SH coefficients (..., (deg+1)^2, 3)
        dirs: Normalized direction vectors (..., 3)

    Returns:
        RGB values (..., 3)
    """
    assert deg <= 3 and deg >= 0

    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    yz = y * z
    xz = x * z

    result = C0 * sh[..., 0, :]

    if deg > 0:
        result = (
            result
            - C1 * y * sh[..., 1, :]
            + C1 * z * sh[..., 2, :]
            - C1 * x * sh[..., 3, :]
        )

    if deg > 1:
        result = (
            result
            + C2[0] * xy * sh[..., 4, :]
            + C2[1] * yz * sh[..., 5, :]
            + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6, :]
            + C2[3] * xz * sh[..., 7, :]
            + C2[4] * (xx - yy) * sh[..., 8, :]
        )

    if deg > 2:
        result = (
            result
            + C3[0] * y * (3 * xx - yy) * sh[..., 9, :]
            + C3[1] * xy * z * sh[..., 10, :]
            + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11, :]
            + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12, :]
            + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13, :]
            + C3[5] * z * (xx - yy) * sh[..., 14, :]
            + C3[6] * x * (xx - 3 * yy) * sh[..., 15, :]
        )

    return result


def eval_sh_bases(deg: int, dirs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate spherical harmonics basis functions (without coefficients)

    Args:
        deg: SH degree (0-3)
        dirs: Normalized directions (..., 3)

    Returns:
        SH bases (..., (deg+1)^2)
    """
    assert deg <= 3 and deg >= 0

    result = torch.zeros((*dirs.shape[:-1], (deg + 1) ** 2), device=dirs.device)

    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    yz = y * z
    xz = x * z

    result[..., 0] = C0

    if deg > 0:
        result[..., 1] = -C1 * y
        result[..., 2] = C1 * z
        result[..., 3] = -C1 * x

    if deg > 1:
        result[..., 4] = C2[0] * xy
        result[..., 5] = C2[1] * yz
        result[..., 6] = C2[2] * (2.0 * zz - xx - yy)
        result[..., 7] = C2[3] * xz
        result[..., 8] = C2[4] * (xx - yy)

    if deg > 2:
        result[..., 9] = C3[0] * y * (3 * xx - yy)
        result[..., 10] = C3[1] * xy * z
        result[..., 11] = C3[2] * y * (4 * zz - xx - yy)
        result[..., 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
        result[..., 13] = C3[4] * x * (4 * zz - xx - yy)
        result[..., 14] = C3[5] * z * (xx - yy)
        result[..., 15] = C3[6] * x * (xx - 3 * yy)

    return result


def RGB2SH(rgb):
    """
    Convert RGB values to SH coefficients (DC term only)

    Args:
        rgb: RGB colors (..., 3) in [0, 1]

    Returns:
        SH coefficients (..., 3)
    """
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Convert SH coefficients to RGB (using DC term only)

    Args:
        sh: SH coefficients (..., 3)

    Returns:
        RGB colors (..., 3) in [0, 1]
    """
    return sh * C0 + 0.5
