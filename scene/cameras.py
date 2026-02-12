"""
Camera class - Represents camera with intrinsics and extrinsics
"""

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    """
    Camera representation with intrinsics, extrinsics, and image data
    """

    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        image_name,
        uid,
        data_device="cuda",
    ):
        """
        Initialize camera

        Args:
            colmap_id: Camera ID from COLMAP
            R: Rotation matrix (3, 3) - world to camera
            T: Translation vector (3,) - world to camera
            FoVx: Field of view in x direction (radians)
            FoVy: Field of view in y direction (radians)
            image: Image tensor (3, H, W) in [0, 1]
            gt_alpha_mask: Alpha mask if available
            image_name: Name of the image
            uid: Unique identifier
            data_device: Device to store image data
        """
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(f"[Warning] {data_device} not available, using cpu")
            self.data_device = torch.device("cpu")

        if self.data_device.type == "cuda" and not torch.cuda.is_available():
            print("[Warning] CUDA not available, using cpu")
            self.data_device = torch.device("cpu")
        if self.data_device.type == "mps":
            mps_available = (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            )
            if not mps_available:
                print("[Warning] MPS not available, using cpu")
                self.data_device = torch.device("cpu")

        # Store original image
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # Apply alpha mask if provided
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            # Create white background mask
            self.original_image *= torch.ones(
                (1, self.image_height, self.image_width), device=self.data_device
            )

        # Precompute transforms
        self.zfar = 100.0
        self.znear = 0.01

        # World to view transform (world to camera)
        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, np.array([0, 0, 0]), 1.0))
            .transpose(0, 1)
            .to(self.data_device)
        )

        # Projection matrix (camera to clip space)
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .to(self.data_device)
        )

        # Full transform (world to clip)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

        # Camera center in world space (inverse of translation in camera frame)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def get_world_view_transform(self):
        """Return world to view transform"""
        return self.world_view_transform

    def get_full_proj_transform(self):
        """Return full projection transform"""
        return self.full_proj_transform

    def get_camera_center(self):
        """Return camera center in world space"""
        return self.camera_center

    def get_camera_position(self):
        """Return camera position as numpy array"""
        return self.camera_center.cpu().numpy()

    def get_intrinsic_matrix(self):
        """
        Get intrinsic matrix K from field of view
        Returns 3x3 matrix
        """
        fx = self.image_width / (2.0 * np.tan(self.FoVx / 2.0))
        fy = self.image_height / (2.0 * np.tan(self.FoVy / 2.0))
        cx = self.image_width / 2.0
        cy = self.image_height / 2.0

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        return K

    def get_extrinsic_matrix(self):
        """
        Get extrinsic matrix [R|t] (world to camera)
        Returns 3x4 matrix
        """
        Rt = np.eye(4, dtype=np.float32)
        Rt[:3, :3] = self.R
        Rt[:3, 3] = self.T
        return Rt
