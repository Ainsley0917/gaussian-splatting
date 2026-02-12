"""
GaussianModel - Core data structure for 3D Gaussian Splatting
Implements the parametric 3D Gaussian representation with trainable parameters
"""

import torch
import torch.nn as nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    build_scaling_rotation,
)
from utils.point_cloud_utils import fetchPly
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BasicPointCloud:
    """Point cloud data structure"""

    points: np.ndarray  # (N, 3)
    colors: np.ndarray  # (N, 3) in [0, 1]
    normals: np.ndarray  # (N, 3)


class GaussianModel(nn.Module):
    """
    3D Gaussian model representing the scene as a set of 3D Gaussians
    Each Gaussian has position, covariance (scale + rotation), opacity, and spherical harmonics
    """

    def setup_functions(self):
        """Setup activation functions for different parameters"""

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            """Build covariance matrix from scale and rotation"""
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    @staticmethod
    def _resolve_device(data_device: str) -> torch.device:
        try:
            device = torch.device(data_device)
        except (TypeError, RuntimeError):
            return torch.device("cpu")

        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")

        if device.type == "mps":
            mps_available = (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            )
            if not mps_available:
                return torch.device("cpu")

        return device

    def __init__(self, sh_degree: int, data_device: str = "cuda"):
        """
        Args:
            sh_degree: Spherical harmonics degree (0-3)
        """
        super(GaussianModel, self).__init__()

        self.device = self._resolve_device(data_device)
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0  # Start at 0, increase during training

        # Initialize parameters as None
        self._xyz = None  # Positions (N, 3)
        self._features_dc = None  # SH DC term (N, 1, 3)
        self._features_rest = None  # SH higher orders (N, 15, 3) for degree 3
        self._opacity = None  # Opacity (N, 1)
        self._scaling = None  # Scale factors (N, 3)
        self._rotation = None  # Quaternion rotation (N, 4)

        self.max_radii2D = None  # Track max 2D radii for densification
        self.xyz_gradient_accum = None  # Accumulate gradients for densification
        self.denom = None  # Denominator for gradient accumulation

        self.optimizer = None  # Optimizer reference
        self.percent_dense = 0.01  # Percentage for densification threshold
        self.spatial_lr_scale = 0.0  # Spatial learning rate scale

        self.setup_functions()

    @property
    def get_scaling(self):
        """Get scales with activation applied"""
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        """Get normalized quaternions"""
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        """Get positions directly"""
        return self._xyz

    @property
    def get_features(self):
        """Get all SH features (concatenated DC and rest)"""
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        """Get opacities with sigmoid activation"""
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1.0):
        """
        Get covariance matrices from scales and rotations

        Args:
            scaling_modifier: Factor to modify scales
        Returns:
            Covariance matrices in upper-triangular form (N, 6)
        """
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        """Increment active SH degree up to max"""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict() if self.optimizer is not None else None,
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args

        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if opt_dict is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(opt_dict)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        """
        Initialize Gaussian parameters from point cloud

        Args:
            pcd: Point cloud with points, colors, normals
            spatial_lr_scale: Scale factor for spatial learning rate
        """
        self.spatial_lr_scale = spatial_lr_scale

        # Convert to torch tensors
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to(self.device)
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().to(self.device)

        # Convert RGB to SH coefficients
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .to(self.device)
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation:", fused_point_cloud.shape[0])

        # Initialize scales based on nearest neighbor distances
        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().to(self.device)),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1  # Identity quaternion [1, 0, 0, 0]

        # Initialize opacities to low values
        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1),
                dtype=torch.float,
                device=self.device,
            )
        )

        # Set parameters
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        # Initialize tracking tensors
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device=self.device
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

    def training_setup(self, training_args):
        """
        Setup optimizer with different learning rates for different parameters

        Args:
            training_args: Configuration with learning rates
        """
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device=self.device
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        # Create parameter groups with different learning rates
        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # Learning rate scheduler for positions
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Update learning rate for position parameters"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr
        return None

    def construct_list_of_attributes(self):
        """Build list of attribute names for PLY file export"""
        l = ["x", "y", "z", "nx", "ny", "nz"]

        # All SH features
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))

        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        """Save Gaussian parameters to PLY file"""
        # Ensure directory exists
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))

        from plyfile import PlyData, PlyElement

        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        """Load Gaussian parameters from PLY file"""
        plydata = fetchPly(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device=self.device).requires_grad_(
                True
            )
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device=self.device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device=self.device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(
                opacities, dtype=torch.float, device=self.device
            ).requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device=self.device).requires_grad_(
                True
            )
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device=self.device).requires_grad_(
                True
            )
        )

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        """Replace a parameter tensor in optimizer"""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"] = [tensor]
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """Remove parameters marked by mask from optimizer"""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"] = [group["params"][0][mask]]
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"] = [group["params"][0][mask]]
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """Prune Gaussians based on boolean mask"""
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """Concatenate new tensors to existing optimizer parameters"""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"] = [
                    torch.cat((group["params"][0], extension_tensor), dim=0)
                ]
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"] = [
                    torch.cat((group["params"][0], extension_tensor), dim=0)
                ]
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
    ):
        """Add new Gaussians after densification (split/clone)"""
        self.denom = torch.cat((self.denom, torch.zeros_like(new_opacities)), dim=0)
        self.xyz_gradient_accum = torch.cat(
            (self.xyz_gradient_accum, torch.zeros_like(new_opacities)), dim=0
        )
        self.max_radii2D = torch.cat(
            (self.max_radii2D, torch.zeros_like(new_opacities[:, 0])), dim=0
        )

        tensors_dict = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(tensors_dict)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        Split Gaussians with high gradient and large view-space extent

        Args:
            grads: Gradient magnitudes
            grad_threshold: Threshold for high gradient
            scene_extent: Scene size for determining split scale
            N: Number of splits per Gaussian (default 2)
        """
        n_init_points = self.get_xyz.shape[0]
        device = self.get_xyz.device

        # Get Gaussians with high gradients
        padded_grad = torch.zeros(n_init_points, device=device)
        padded_grad[: grads.shape[0]] = grads.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        # Sample new locations around parent Gaussian
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=device)
        samples = torch.normal(mean=means, std=stds)

        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        )

        # Remove original Gaussians that were split
        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device=device, dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        Clone Gaussians with high gradient but small view-space extent

        Args:
            grads: Gradient magnitudes
            grad_threshold: Threshold for high gradient
            scene_extent: Scene size for determining clone eligibility
        """
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def densify(self, max_grad, min_opacity, extent, max_screen_size):
        """
        Main densification routine - clone and split based on gradient

        Args:
            max_grad: Maximum gradient threshold
            min_opacity: Minimum opacity for pruning
            extent: Scene extent
            max_screen_size: Maximum screen size for pruning
        """
        # Get gradients averaged over accumulation
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Clone small Gaussians with high gradients
        self.densify_and_clone(grads, max_grad, extent)

        # Split large Gaussians with high gradients
        self.densify_and_split(grads, max_grad, extent)

        # Prune low opacity Gaussians
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

        # Reset gradient accumulators
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """Accumulate view-space gradients for densification"""
        if self.xyz_gradient_accum is None or self.denom is None:
            return

        grad = viewspace_point_tensor.grad
        if grad is None:
            return

        self.xyz_gradient_accum[update_filter] += torch.norm(
            grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def reset_opacity(self):
        """Reset all opacities to a low value (for opacity reset strategy)"""
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


def strip_symmetric(sym):
    """Extract upper triangular part of symmetric matrix (N, 3, 3) -> (N, 6)"""
    return sym[:, :, [0, 1, 2, 1, 2, 2]][:, [0, 1, 2, 3, 4, 5]]


def distCUDA2(points):
    """
    Compute distance to nearest neighbor for each point
    Uses PyTorch for GPU acceleration
    """
    points = points.contiguous()
    n = points.shape[0]

    # Compute pairwise distances
    distances = torch.cdist(points, points, p=2)

    # Set diagonal to infinity to exclude self
    distances.fill_diagonal_(float("inf"))

    # Get minimum distance for each point
    min_distances = torch.min(distances, dim=1)[0]

    return min_distances**2
