"""
Camera utilities - Camera list creation and JSON serialization
"""

import json
import numpy as np
from scene.cameras import Camera
from PIL import Image
import torch


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    """
    Create list of Camera objects from CameraInfo objects

    Args:
        cam_infos: List of CameraInfo objects
        resolution_scale: Scale factor for resolution
        args: Arguments with data_device

    Returns:
        List of Camera objects
    """
    camera_list = []

    for id, c in enumerate(cam_infos):
        # Resize image according to scale
        orig_w, orig_h = c.width, c.height
        resolution = (int(orig_w / resolution_scale), int(orig_h / resolution_scale))

        # Load and resize image
        image = Image.fromarray((c.image * 255).astype(np.uint8))
        resized_image = image.resize(resolution)
        resized_image = np.array(resized_image) / 255.0

        # Convert to torch tensor (CHW format)
        resized_image = torch.from_numpy(resized_image).permute(2, 0, 1).float()

        camera_list.append(
            Camera(
                colmap_id=c.uid,
                R=c.R,
                T=c.T,
                FoVx=c.FovX,
                FoVy=c.FovY,
                image=resized_image,
                gt_alpha_mask=None,
                image_name=c.image_name,
                uid=id,
                data_device=args.data_device,
            )
        )

    return camera_list


def camera_to_JSON(id, camera: Camera):
    """
    Convert Camera object to JSON serializable dictionary

    Args:
        id: Camera ID
        camera: Camera object

    Returns:
        Dictionary with camera parameters
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]

    # Convert rotation matrix to quaternion
    rot = W2C[:3, :3]
    from scene.dataset_readers import rotmat2qvec

    qvec = rotmat2qvec(rot.transpose())

    width = getattr(camera, "image_width", getattr(camera, "width"))
    height = getattr(camera, "image_height", getattr(camera, "height"))
    fovx = getattr(camera, "FoVx", getattr(camera, "FovX"))
    fovy = getattr(camera, "FoVy", getattr(camera, "FovY"))

    return {
        "id": id,
        "img_name": camera.image_name,
        "width": width,
        "height": height,
        "position": pos.tolist(),
        "rotation": qvec.tolist(),
        "fovx": fovx,
        "fovy": fovy,
    }


def loadCamerasFromJSON(json_path, args):
    """
    Load cameras from JSON file

    Args:
        json_path: Path to cameras.json
        args: Arguments

    Returns:
        List of Camera objects
    """
    with open(json_path, "r") as f:
        camera_jsons = json.load(f)

    cameras = []
    for cam_data in camera_jsons:
        # Convert quaternion to rotation matrix
        qvec = np.array(cam_data["rotation"])
        from scene.dataset_readers import qvec2rotmat

        R = np.transpose(qvec2rotmat(qvec))

        # Translation
        # Reconstruct translation from position
        # C2W = [R^T | -R^T @ t], so t = -R @ C
        pos = np.array(cam_data["position"])
        T = -R @ pos

        # Load image
        image_path = cam_data.get("img_path", "")
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path)
            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            # Create placeholder
            image = torch.zeros(3, cam_data["height"], cam_data["width"])

        cam = Camera(
            colmap_id=cam_data["id"],
            R=R,
            T=T,
            FoVx=cam_data["fovx"],
            FoVy=cam_data["fovy"],
            image=image,
            gt_alpha_mask=None,
            image_name=cam_data["img_name"],
            uid=cam_data["id"],
            data_device=args.data_device,
        )
        cameras.append(cam)

    return cameras


import os
