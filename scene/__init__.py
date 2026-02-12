"""
Scene class - Manages cameras and Gaussian model
"""

import os
import json
import torch
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from typing import List


class Scene:
    """
    Scene container managing cameras and Gaussian model
    Handles loading/saving and coordinate transformations
    """

    gaussians: GaussianModel

    def __init__(
        self,
        args,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
    ):
        """
        Initialize scene from dataset

        Args:
            args: Arguments containing source path, etc.
            gaussians: GaussianModel instance
            load_iteration: If specified, load checkpoint from this iteration
            shuffle: Whether to shuffle camera order
            resolution_scales: List of scales for multi-resolution training
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # Load from checkpoint if specified
        if load_iteration:
            if load_iteration == -1:
                # Load latest
                loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                loaded_iter = load_iteration

            print(f"Loading trained model at iteration {loaded_iter}")
            self.loaded_iter = loaded_iter
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    f"iteration_{loaded_iter}",
                    "point_cloud.ply",
                )
            )

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
            )
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            # Save scene info
            with (
                open(scene_info.ply_path, "rb") as src_file,
                open(os.path.join(self.model_path, "input.ply"), "wb") as dest_file,
            ):
                dest_file.write(src_file.read())

            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)

            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))

            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        # Load cameras at different scales
        if shuffle:
            import random

            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print(f"Loading Training Cameras at scale {resolution_scale}...")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print(f"Loading Test Cameras at scale {resolution_scale}...")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )

        if not self.loaded_iter:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        """Save checkpoint"""
        point_cloud_path = os.path.join(
            self.model_path, f"point_cloud/iteration_{iteration}"
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        """Get training cameras at specified scale"""
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        """Get test cameras at specified scale"""
        return self.test_cameras[scale]


def searchForMaxIteration(folder):
    """Find the highest iteration number in checkpoints"""
    saved_iters = [
        int(fname.split("_")[-1])
        for fname in os.listdir(folder)
        if fname.startswith("iteration")
    ]
    return max(saved_iters)
