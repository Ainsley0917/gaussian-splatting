"""
Gaussian rasterizer wrapper - Interface to CUDA rasterization
This is a PyTorch implementation of the rasterization (no CUDA required)
For production use, the CUDA version from submodules is recommended
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional


class GaussianRasterizationSettings:
    """
    Settings for Gaussian rasterization
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        tanfovx: float,
        tanfovy: float,
        bg: torch.Tensor,
        scale_modifier: float,
        viewmatrix: torch.Tensor,
        projmatrix: torch.Tensor,
        sh_degree: int,
        campos: torch.Tensor,
        prefiltered: bool,
        debug: bool,
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.tanfovx = tanfovx
        self.tanfovy = tanfovy
        self.bg = bg
        self.scale_modifier = scale_modifier
        self.viewmatrix = viewmatrix
        self.projmatrix = projmatrix
        self.sh_degree = sh_degree
        self.campos = campos
        self.prefiltered = prefiltered
        self.debug = debug


class GaussianRasterizer(nn.Module):
    """
    Gaussian rasterizer - renders 3D Gaussians to 2D image

    This is a PyTorch implementation. For production use with CUDA acceleration,
    use the version from the diff_gaussian_rasterization submodule.
    """

    def __init__(self, raster_settings: GaussianRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        shs: Optional[torch.Tensor],
        colors_precomp: Optional[torch.Tensor],
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        cov3Ds_precomp: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Rasterize 3D Gaussians to 2D image

        Args:
            means3D: 3D positions (N, 3)
            means2D: 2D projected positions for gradient (N, 3)
            shs: Spherical harmonics coefficients (N, (deg+1)^2, 3)
            colors_precomp: Precomputed colors if SH not used (N, 3)
            opacities: Opacity values (N, 1)
            scales: Scale factors (N, 3)
            rotations: Quaternion rotations (N, 4)
            cov3Ds_precomp: Precomputed covariance matrices (N, 6)

        Returns:
            Dictionary with:
                - render: Rendered image (3, H, W)
                - depth: Depth map (1, H, W)
                - alpha: Alpha mask (1, H, W)
        """
        rs = self.raster_settings

        # Compute colors from SH if needed
        if colors_precomp is None:
            assert shs is not None
            colors = self.compute_colors_from_sh(means3D, shs, rs.sh_degree, rs.campos)
        else:
            colors = colors_precomp

        # Project 3D Gaussians to 2D
        proj_matrix = rs.projmatrix  # (4, 4)

        # Transform to camera space
        means_homo = torch.cat(
            [means3D, torch.ones(means3D.shape[0], 1, device=means3D.device)], dim=1
        )
        means_cam = (rs.viewmatrix @ means_homo.T).T

        # Project to clip space
        means_clip = (proj_matrix @ means_homo.T).T

        # Perspective divide to get NDC
        means_ndc = means_clip[:, :3] / (means_clip[:, 3:4] + 1e-7)

        # Convert to screen space
        means_screen = torch.zeros_like(means_ndc[:, :2])
        means_screen[:, 0] = ((means_ndc[:, 0] + 1.0) * rs.image_width - 1.0) * 0.5
        means_screen[:, 1] = ((means_ndc[:, 1] + 1.0) * rs.image_height - 1.0) * 0.5

        # Compute 2D covariances
        if cov3Ds_precomp is None:
            cov3D = self.compute_cov3D(scales, rotations)  # (N, 3, 3)
        else:
            cov3D = self.cov_from_upper_triangular(cov3Ds_precomp)

        # Project covariance to 2D
        cov2D = self.project_cov3D_to_2D(
            cov3D,
            means_cam,
            rs.viewmatrix,
            rs.projmatrix,
            rs.image_height,
            rs.image_width,
        )

        # Sort by depth
        depths = means_cam[:, 2]
        sorted_indices = torch.argsort(depths, descending=True)

        means_screen = means_screen[sorted_indices]
        cov2D = cov2D[sorted_indices]
        colors = colors[sorted_indices]
        opacities = opacities[sorted_indices]
        depths = depths[sorted_indices]

        # Render by splatting
        rendered_image, depth_map, alpha_map = self.splat_gaussians(
            means_screen,
            cov2D,
            colors,
            opacities,
            depths,
            rs.image_height,
            rs.image_width,
            rs.bg,
        )

        return {
            "render": rendered_image,
            "depth": depth_map,
            "alpha": alpha_map,
            "viewspace_points": means_screen,
            "visibility_filter": torch.ones(
                means3D.shape[0], dtype=torch.bool, device=means3D.device
            ),
            "radii": torch.ones(means3D.shape[0], device=means3D.device)
            * max(rs.image_width, rs.image_height),
        }

    def compute_colors_from_sh(self, means3D, shs, sh_degree, campos):
        """
        Compute RGB colors from spherical harmonics

        Args:
            means3D: 3D positions (N, 3)
            shs: SH coefficients (N, (deg+1)^2, 3)
            sh_degree: SH degree to use
            campos: Camera position (3,)

        Returns:
            colors: RGB colors (N, 3)
        """
        # Compute view directions
        directions = means3D - campos.unsqueeze(0)
        directions = directions / (torch.norm(directions, dim=1, keepdim=True) + 1e-7)

        # Import SH utils
        from utils.sh_utils import eval_sh

        colors = eval_sh(sh_degree, shs, directions)
        colors = torch.clamp(colors + 0.5, 0.0, 1.0)

        return colors

    def compute_cov3D(self, scales, rotations):
        """
        Compute 3D covariance matrix from scales and rotations

        Args:
            scales: (N, 3) scale factors
            rotations: (N, 4) quaternions

        Returns:
            cov3D: (N, 3, 3) covariance matrices
        """
        # Build rotation matrix from quaternion
        r, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]

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

        # Build scale matrix
        S = torch.zeros((scales.shape[0], 3, 3), device=scales.device)
        S[:, 0, 0] = scales[:, 0]
        S[:, 1, 1] = scales[:, 1]
        S[:, 2, 2] = scales[:, 2]

        # Covariance = R @ S @ S^T @ R^T
        RS = R @ S
        cov3D = RS @ RS.transpose(1, 2)

        return cov3D

    def cov_from_upper_triangular(self, cov_params):
        """
        Reconstruct covariance matrix from upper triangular parameters

        Args:
            cov_params: (N, 6) [c11, c12, c13, c22, c23, c33]

        Returns:
            cov3D: (N, 3, 3) symmetric matrices
        """
        N = cov_params.shape[0]
        cov3D = torch.zeros((N, 3, 3), device=cov_params.device)

        cov3D[:, 0, 0] = cov_params[:, 0]
        cov3D[:, 0, 1] = cov_params[:, 1]
        cov3D[:, 0, 2] = cov_params[:, 2]
        cov3D[:, 1, 0] = cov_params[:, 1]
        cov3D[:, 1, 1] = cov_params[:, 3]
        cov3D[:, 1, 2] = cov_params[:, 4]
        cov3D[:, 2, 0] = cov_params[:, 2]
        cov3D[:, 2, 1] = cov_params[:, 4]
        cov3D[:, 2, 2] = cov_params[:, 5]

        return cov3D

    def project_cov3D_to_2D(
        self, cov3D, means_cam, viewmatrix, projmatrix, image_height, image_width
    ):
        """
        Project 3D covariance to 2D using EWA splatting

        Args:
            cov3D: (N, 3, 3) 3D covariance
            means_cam: (N, 4) camera space positions
            viewmatrix: (4, 4) view matrix
            projmatrix: (4, 4) projection matrix
            image_height: Image height
            image_width: Image width

        Returns:
            cov2D: (N, 2, 2) 2D covariance matrices
        """
        # Extract view Jacobian
        N = cov3D.shape[0]

        # Compute Jacobian of perspective projection
        # J = [[focal_x / z, 0, -x * focal_x / z^2],
        #      [0, focal_y / z, -y * focal_y / z^2]]

        x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]

        # Approximate focal length from projection matrix
        focal_x = projmatrix[0, 0] * image_width / 2.0
        focal_y = projmatrix[1, 1] * image_height / 2.0

        J = torch.zeros((N, 2, 3), device=cov3D.device)
        J[:, 0, 0] = focal_x / (z + 1e-7)
        J[:, 0, 2] = -x * focal_x / (z * z + 1e-7)
        J[:, 1, 1] = focal_y / (z + 1e-7)
        J[:, 1, 2] = -y * focal_y / (z * z + 1e-7)

        # Project covariance: cov2D = J @ cov3D @ J^T
        cov2D = J @ cov3D @ J.transpose(1, 2)

        # Add low-pass filter to prevent numerical issues
        cov2D[:, 0, 0] += 0.3
        cov2D[:, 1, 1] += 0.3

        return cov2D

    def splat_gaussians(
        self, means2D, cov2D, colors, opacities, depths, height, width, bg
    ):
        """
        Splat 2D Gaussians onto image plane

        Args:
            means2D: (N, 2) 2D centers
            cov2D: (N, 2, 2) 2D covariances
            colors: (N, 3) colors
            opacities: (N, 1) opacities
            depths: (N,) depth values
            height: Image height
            width: Image width
            bg: (3,) background color

        Returns:
            image: (3, H, W) rendered image
            depth: (1, H, W) depth map
            alpha: (1, H, W) alpha mask
        """
        device = means2D.device
        N = means2D.shape[0]

        # Create output buffers
        image = bg.clone().reshape(3, 1, 1).expand(3, height, width).contiguous()
        depth_map = torch.zeros(1, height, width, device=device)
        alpha_acc = torch.zeros(1, height, width, device=device)

        # Render each Gaussian
        for i in range(N):
            # Skip if outside frustum
            mean = means2D[i]
            if (
                mean[0] < -50
                or mean[0] > width + 50
                or mean[1] < -50
                or mean[1] > height + 50
            ):
                continue

            # Compute Gaussian extent
            cov = cov2D[i]
            det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
            if det < 1e-7:
                continue

            # Inverse covariance for Gaussian evaluation
            inv_det = 1.0 / det
            inv_cov = torch.zeros(2, 2, device=device)
            inv_cov[0, 0] = cov[1, 1] * inv_det
            inv_cov[0, 1] = -cov[0, 1] * inv_det
            inv_cov[1, 0] = -cov[1, 0] * inv_det
            inv_cov[1, 1] = cov[0, 0] * inv_det

            # Compute bounding box
            # 3-sigma bounding box
            std_x = torch.sqrt(cov[0, 0]) * 3.0
            std_y = torch.sqrt(cov[1, 1]) * 3.0

            x_min = int(max(0, mean[0] - std_x))
            x_max = int(min(width, mean[0] + std_x + 1))
            y_min = int(max(0, mean[1] - std_y))
            y_max = int(min(height, mean[1] + std_y + 1))

            if x_min >= x_max or y_min >= y_max:
                continue

            # Evaluate 2D Gaussian over bounding box
            y_coords, x_coords = torch.meshgrid(
                torch.arange(y_min, y_max, device=device, dtype=torch.float32),
                torch.arange(x_min, x_max, device=device, dtype=torch.float32),
                indexing="ij",
            )

            # Compute offset from mean
            offset_x = x_coords - mean[0]
            offset_y = y_coords - mean[1]

            # Compute Gaussian weight
            # w = exp(-0.5 * x^T * inv_cov * x)
            power = (
                -(
                    inv_cov[0, 0] * offset_x * offset_x
                    + (inv_cov[0, 1] + inv_cov[1, 0]) * offset_x * offset_y
                    + inv_cov[1, 1] * offset_y * offset_y
                )
                * 0.5
            )

            weight = torch.exp(power) * opacities[i, 0]

            # Alpha blending
            alpha = weight * (1.0 - alpha_acc[0, y_min:y_max, x_min:x_max])

            # Blend color
            image[:, y_min:y_max, x_min:x_max] += alpha.unsqueeze(0) * colors[
                i
            ].reshape(3, 1, 1)

            # Accumulate alpha
            alpha_acc[0, y_min:y_max, x_min:x_max] += alpha

            # Update depth (weighted by alpha)
            depth_map[0, y_min:y_max, x_min:x_max] += alpha * depths[i]

        return image, depth_map, alpha_acc


def render(
    viewpoint_camera,
    pc,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    computer_pseudo_normal=False,
):
    """
    Main render function - renders Gaussian model from a camera viewpoint

    Args:
        viewpoint_camera: Camera object
        pc: GaussianModel object
        pipe: Pipeline parameters
        bg_color: Background color (3,) tensor
        scaling_modifier: Scale modifier for debugging
        override_color: Override color (for visualization)
        computer_pseudo_normal: Whether to compute normals

    Returns:
        Dictionary with rendered outputs
    """
    # Create rasterization settings
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Get Gaussian parameters
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # Compute covariance or use precomputed
    cov3D_precomp = None

    # Get colors from SH or use override
    if override_color is None:
        shs = pc.get_features
        colors_precomp = None
    else:
        shs = None
        colors_precomp = override_color

    # Project to 2D for gradient computation
    means2D = torch.zeros_like(means3D, requires_grad=True, device=means3D.device) + 0

    # Rasterize
    raster_output = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=cov3D_precomp,
    )

    rendered_image = raster_output["render"]
    rendered_depth = raster_output["depth"]
    rendered_alpha = raster_output["alpha"]
    radii = raster_output["radii"]

    return {
        "render": rendered_image,
        "depth": rendered_depth,
        "alpha": rendered_alpha,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
