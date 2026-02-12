"""
Graphics utilities - Camera matrices, projections, and transformations
"""

import torch
import numpy as np
import math


def getWorld2View(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    """
    Compute world-to-view (world-to-camera) transformation matrix

    Args:
        R: Rotation matrix (3, 3) - world to camera rotation
        t: Translation vector (3,) - camera position in world coords
        translate: Additional translation
        scale: Scale factor

    Returns:
        4x4 transformation matrix (world to view space)
    """
    # Camera pose in world: C = -R^T @ t
    # World to view: view = R @ world + t

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center

    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    """
    Extended version of getWorld2View with numpy inputs

    Args:
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
        translate: Additional translation
        scale: Scale factor

    Returns:
        4x4 transformation matrix
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center

    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY, K=None, w=None, h=None):
    """
    Create OpenGL-style projection matrix from camera parameters

    Args:
        znear: Near clipping plane
        zfar: Far clipping plane
        fovX: Horizontal field of view in radians
        fovY: Vertical field of view in radians
        K: Optional intrinsic matrix (3, 3)
        w: Image width
        h: Image height

    Returns:
        4x4 projection matrix (torch tensor)
    """
    if K is not None and w is not None and h is not None:
        # Use provided intrinsics
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        P = torch.zeros(4, 4)

        # Standard OpenGL projection with custom intrinsics
        P[0, 0] = 2.0 * fx / w
        P[1, 1] = 2.0 * fy / h
        P[0, 2] = 1.0 - 2.0 * cx / w
        P[1, 2] = 2.0 * cy / h - 1.0
        P[2, 2] = -(zfar + znear) / (zfar - znear)
        P[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
        P[3, 2] = -1.0
    else:
        # Compute from FOV
        tanHalfFovX = math.tan(fovX / 2)
        tanHalfFovY = math.tan(fovY / 2)

        P = torch.zeros(4, 4)

        P[0, 0] = 1.0 / tanHalfFovX
        P[1, 1] = 1.0 / tanHalfFovY
        P[2, 2] = -(zfar + znear) / (zfar - znear)
        P[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
        P[3, 2] = -1.0

    return P


def focal2fov(focal, pixels):
    """
    Convert focal length to field of view

    Args:
        focal: Focal length in pixels
        pixels: Image dimension (width or height)

    Returns:
        Field of view in radians
    """
    return 2 * math.atan(pixels / (2 * focal))


def fov2focal(fov, pixels):
    """
    Convert field of view to focal length

    Args:
        fov: Field of view in radians
        pixels: Image dimension (width or height)

    Returns:
        Focal length in pixels
    """
    return pixels / (2 * math.tan(fov / 2))


def get_intrinsic_matrix(K, device="cuda"):
    """
    Convert numpy intrinsic matrix to torch tensor

    Args:
        K: 3x3 numpy array
        device: Target device

    Returns:
        3x3 torch tensor
    """
    return torch.from_numpy(K).float().to(device)


def homogeneous_transform(points, transform):
    """
    Apply homogeneous transformation to 3D points

    Args:
        points: (N, 3) or (H, W, 3) array of points
        transform: (4, 4) transformation matrix

    Returns:
        Transformed points
    """
    if isinstance(points, np.ndarray):
        # Convert to homogeneous coordinates
        if points.ndim == 2:
            ones = np.ones((points.shape[0], 1))
            homo = np.concatenate([points, ones], axis=1)
            transformed = (transform @ homo.T).T
            return transformed[:, :3]
        else:
            original_shape = points.shape
            points_flat = points.reshape(-1, 3)
            ones = np.ones((points_flat.shape[0], 1))
            homo = np.concatenate([points_flat, ones], axis=1)
            transformed = (transform @ homo.T).T
            return transformed[:, :3].reshape(original_shape)
    else:
        # Torch tensor version
        if points.ndim == 2:
            ones = torch.ones((points.shape[0], 1), device=points.device)
            homo = torch.cat([points, ones], dim=1)
            transformed = (transform @ homo.T).T
            return transformed[:, :3]
        else:
            original_shape = points.shape
            points_flat = points.reshape(-1, 3)
            ones = torch.ones((points_flat.shape[0], 1), device=points.device)
            homo = torch.cat([points_flat, ones], dim=1)
            transformed = (transform @ homo.T).T
            return transformed[:, :3].reshape(original_shape)


def look_at(eye, center, up):
    """
    Create look-at view matrix

    Args:
        eye: Camera position (3,)
        center: Look-at point (3,)
        up: Up vector (3,)

    Returns:
        4x4 view matrix
    """
    if isinstance(eye, torch.Tensor):
        eye = eye.cpu().numpy()
    if isinstance(center, torch.Tensor):
        center = center.cpu().numpy()
    if isinstance(up, torch.Tensor):
        up = up.cpu().numpy()

    forward = center - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)

    new_up = np.cross(forward, right)

    view_matrix = np.eye(4)
    view_matrix[0, :3] = right
    view_matrix[1, :3] = new_up
    view_matrix[2, :3] = -forward
    view_matrix[:3, 3] = -view_matrix[:3, :3] @ eye

    return view_matrix


def perspective_projection(fov, aspect, near, far):
    """
    Create perspective projection matrix

    Args:
        fov: Vertical field of view in radians
        aspect: Aspect ratio (width/height)
        near: Near clipping plane
        far: Far clipping plane

    Returns:
        4x4 projection matrix
    """
    f = 1.0 / math.tan(fov / 2)

    proj = np.zeros((4, 4))
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1

    return proj


def rotation_matrix_x(theta):
    """Rotation matrix around X axis"""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_matrix_y(theta):
    """Rotation matrix around Y axis"""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_matrix_z(theta):
    """Rotation matrix around Z axis"""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to 3x3 rotation matrix

    Args:
        q: Quaternion [w, x, y, z] or (x, y, z, w)

    Returns:
        3x3 rotation matrix
    """
    if len(q) == 4:
        w, x, y, z = q
    else:
        raise ValueError("Quaternion must have 4 elements")

    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )

    return R


def rotation_matrix_to_quaternion(R):
    """
    Convert 3x3 rotation matrix to quaternion

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion [w, x, y, z]
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def transform_points_matrix(points, M):
    """
    Transform points by 4x4 matrix

    Args:
        points: (N, 3) array
        M: (4, 4) transformation matrix

    Returns:
        Transformed points (N, 3)
    """
    if isinstance(points, np.ndarray):
        ones = np.ones((points.shape[0], 1))
        homogeneous = np.concatenate([points, ones], axis=1)
        transformed = (M @ homogeneous.T).T
        return transformed[:, :3]
    else:
        ones = torch.ones((points.shape[0], 1), device=points.device)
        homogeneous = torch.cat([points, ones], dim=1)
        transformed = (M @ homogeneous.T).T
        return transformed[:, :3]
