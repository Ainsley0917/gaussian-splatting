"""
Point cloud utilities - Loading, saving, and manipulating PLY files
"""

import os
import numpy as np
from plyfile import PlyData, PlyElement


def storePly(path, xyz, rgb):
    """
    Save point cloud to PLY file format

    Args:
        path: Output file path
        xyz: Point coordinates (N, 3) float32
        rgb: RGB colors (N, 3) uint8 [0, 255]
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Define dtype for PLY elements
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    # Prepare data array
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create PLY element
    el = PlyElement.describe(elements, "vertex")

    # Write to file
    PlyData([el]).write(path)


def fetchPly(path):
    """
    Load point cloud from PLY file

    Args:
        path: Path to PLY file

    Returns:
        PlyData object containing the point cloud
    """
    plydata = PlyData.read(path)
    return plydata


def load_ply(path):
    """
    Load point cloud from PLY file and return arrays

    Args:
        path: Path to PLY file

    Returns:
        xyz: (N, 3) coordinates
        rgb: (N, 3) colors in [0, 1]
        normals: (N, 3) normals
    """
    plydata = PlyData.read(path)

    xyz = np.stack(
        [
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ],
        axis=1,
    ).astype(np.float32)

    # Try to get colors
    try:
        rgb = (
            np.stack(
                [
                    np.asarray(plydata.elements[0]["red"]),
                    np.asarray(plydata.elements[0]["green"]),
                    np.asarray(plydata.elements[0]["blue"]),
                ],
                axis=1,
            ).astype(np.float32)
            / 255.0
        )
    except ValueError:
        # No colors in file, use default gray
        rgb = np.ones_like(xyz) * 0.5

    # Try to get normals
    try:
        normals = np.stack(
            [
                np.asarray(plydata.elements[0]["nx"]),
                np.asarray(plydata.elements[0]["ny"]),
                np.asarray(plydata.elements[0]["nz"]),
            ],
            axis=1,
        ).astype(np.float32)
    except ValueError:
        # No normals in file, use zeros
        normals = np.zeros_like(xyz)

    return xyz, rgb, normals


def save_ply(path, xyz, rgb, normals=None):
    """
    Save point cloud arrays to PLY file

    Args:
        path: Output file path
        xyz: (N, 3) coordinates float32
        rgb: (N, 3) colors in [0, 1] or [0, 255]
        normals: (N, 3) normals float32 (optional)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert rgb to uint8 if needed
    if rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)

    if normals is None:
        normals = np.zeros_like(xyz)

    # Define dtype
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)


def transform_point_cloud(xyz, transform):
    """
    Apply 4x4 transformation matrix to point cloud

    Args:
        xyz: (N, 3) coordinates
        transform: (4, 4) transformation matrix

    Returns:
        Transformed coordinates (N, 3)
    """
    # Convert to homogeneous coordinates
    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
    xyz_homo = np.concatenate([xyz, ones], axis=1)

    # Apply transformation
    xyz_transformed = (transform @ xyz_homo.T).T

    # Return to 3D
    return xyz_transformed[:, :3]


def merge_point_clouds(pcd_list):
    """
    Merge multiple point clouds into one

    Args:
        pcd_list: List of (xyz, rgb, normals) tuples

    Returns:
        Merged (xyz, rgb, normals)
    """
    all_xyz = []
    all_rgb = []
    all_normals = []

    for xyz, rgb, normals in pcd_list:
        all_xyz.append(xyz)
        all_rgb.append(rgb)
        all_normals.append(normals)

    merged_xyz = np.concatenate(all_xyz, axis=0)
    merged_rgb = np.concatenate(all_rgb, axis=0)
    merged_normals = np.concatenate(all_normals, axis=0)

    return merged_xyz, merged_rgb, merged_normals


def filter_point_cloud(xyz, rgb, normals, mask):
    """
    Filter point cloud by boolean mask

    Args:
        xyz: (N, 3) coordinates
        rgb: (N, 3) colors
        normals: (N, 3) normals
        mask: (N,) boolean mask

    Returns:
        Filtered (xyz, rgb, normals)
    """
    return xyz[mask], rgb[mask], normals[mask]


def downsample_point_cloud(xyz, rgb, normals, target_count):
    """
    Randomly downsample point cloud to target count

    Args:
        xyz: (N, 3) coordinates
        rgb: (N, 3) colors
        normals: (N, 3) normals
        target_count: Target number of points

    Returns:
        Downsampled (xyz, rgb, normals)
    """
    if xyz.shape[0] <= target_count:
        return xyz, rgb, normals

    indices = np.random.choice(xyz.shape[0], target_count, replace=False)
    return xyz[indices], rgb[indices], normals[indices]


def compute_bounding_box(xyz):
    """
    Compute axis-aligned bounding box

    Args:
        xyz: (N, 3) coordinates

    Returns:
        min_bound, max_bound: (3,) arrays
    """
    min_bound = xyz.min(axis=0)
    max_bound = xyz.max(axis=0)
    return min_bound, max_bound


def normalize_point_cloud(xyz):
    """
    Normalize point cloud to unit sphere centered at origin

    Args:
        xyz: (N, 3) coordinates

    Returns:
        Normalized xyz, center, scale
    """
    center = xyz.mean(axis=0)
    xyz_centered = xyz - center
    scale = np.abs(xyz_centered).max()
    xyz_normalized = xyz_centered / scale

    return xyz_normalized, center, scale


def estimate_normals(xyz, k=10):
    """
    Estimate normals using PCA on k-nearest neighbors

    Args:
        xyz: (N, 3) coordinates
        k: Number of neighbors

    Returns:
        normals: (N, 3) estimated normals
    """
    from sklearn.neighbors import NearestNeighbors

    # Find k nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(xyz)
    _, indices = nbrs.kneighbors(xyz)

    normals = np.zeros_like(xyz)

    for i in range(xyz.shape[0]):
        neighbors = xyz[indices[i]]
        # Compute covariance matrix
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered / k
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Normal is eigenvector with smallest eigenvalue
        normals[i] = eigenvectors[:, 0]

    return normals
