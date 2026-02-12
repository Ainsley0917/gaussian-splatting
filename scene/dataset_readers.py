"""Dataset readers - Load COLMAP format data and create camera/scene structures."""

from __future__ import annotations

import os
import json
import numpy as np

from PIL import Image

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from utils.point_cloud_utils import storePly, fetchPly, load_ply
from scene.cameras import Camera
import torch


@dataclass
class CameraInfo:
    """Camera information data structure"""

    uid: int
    R: np.ndarray  # (3, 3) rotation matrix
    T: np.ndarray  # (3,) translation vector
    FovY: float
    FovX: float
    image: np.ndarray  # (H, W, 3) RGB image in [0, 1]
    image_path: str
    image_name: str
    width: int
    height: int


@dataclass
class SceneInfo:
    """Scene information data structure"""

    point_cloud: "BasicPointCloud"
    train_cameras: List["CameraInfo"]
    test_cameras: List["CameraInfo"]
    nerf_normalization: Dict[str, Any]
    ply_path: str


@dataclass
class BasicPointCloud:
    """Point cloud data structure"""

    points: np.ndarray  # (N, 3)
    colors: np.ndarray  # (N, 3) in [0, 1]
    normals: np.ndarray  # (N, 3)


def getNerfppNorm(cam_info):
    """
    Calculate normalization parameters for NeRF++ style normalization

    Args:
        cam_info: List of CameraInfo objects

    Returns:
        Dictionary with 'translate' and 'radius'
    """

    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        # Compute camera center from R and T
        # C = -R^T @ T
        W2C = getWorld2View(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def getWorld2View(R, t):
    """Compute world-to-view transformation matrix"""
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    """
    Read cameras from NeRF-style transforms.json file

    Args:
        path: Base path to scene directory
        transformsfile: Name of transforms file
        white_background: Whether to use white background
        extension: Image file extension

    Returns:
        List of CameraInfo objects
    """
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

    fovx = contents["camera_angle_x"]

    frames = contents["frames"]
    missing_images = 0
    for idx, frame in enumerate(frames):
        cam_name = frame["file_path"] + extension

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # Change from OpenGL/Blender (Y up) to COLMAP (Y down) conventions.
        # This is not required if using synthetic data from nerf_synthetic dataset
        c2w[:3, 1:3] *= -1

        # Get world-to-camera transform
        w2c = np.linalg.inv(c2w)

        # R is stored transposed due to 'glm' in CUDA code
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        image_path = os.path.join(path, cam_name)
        if not os.path.exists(image_path):
            missing_images += 1
            continue
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
            1 - norm_data[:, :, 3:4]
        )
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy
        FovX = fovx

        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=np.array(image) / 255.0,
                image_path=image_path,
                image_name=image_name,
                width=image.size[0],
                height=image.size[1],
            )
        )

    if missing_images > 0:
        print(f"[Warning] Skipped {missing_images} missing images in {transformsfile}")

    if len(cam_infos) == 0:
        raise RuntimeError(f"No valid images found for {transformsfile} at {path}")

    return cam_infos


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    """
    Read cameras from COLMAP format

    Args:
        cam_extrinsics: Dictionary of COLMAP extrinsics (Image objects)
        cam_intrinsics: Dictionary of COLMAP intrinsics (Camera objects)
        images_folder: Path to images directory

    Returns:
        List of CameraInfo objects
    """
    cam_infos = []

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        sys.stdout.write(f"Reading camera {idx + 1}/{len(cam_extrinsics)}")
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, (
                "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            )

        image_path = os.path.join(images_folder, extr.name)
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=np.array(image) / 255.0,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )
        cam_infos.append(cam_info)

    sys.stdout.write("\n")
    return cam_infos


def readColmapSceneInfo(path, images, eval, llffhold=8):
    """
    Read complete scene information from COLMAP output

    Args:
        path: Path to COLMAP directory
        images: Images subdirectory name
        eval: Whether to create train/test split
        llffhold: Holdout every Nth image for test

    Returns:
        SceneInfo object
    """
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")

    if not os.path.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)

        storePly(ply_path, xyz, rgb)

    try:
        xyz, rgb, normals = load_ply(ply_path)
    except:
        # If loading fails, create a small point cloud
        print("Could not load point cloud, creating minimal one")
        num_pts = 1000
        xyz = (
            np.random.random((num_pts, 3)) * nerf_normalization["radius"]
            + nerf_normalization["translate"]
        )
        rgb = np.ones((num_pts, 3)) * 0.5
        normals = np.zeros((num_pts, 3))
        storePly(ply_path, xyz, (rgb * 255).astype(np.uint8))

    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )

    return scene_info


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    """
    Read NeRF synthetic dataset format

    Args:
        path: Path to scene directory
        white_background: Use white background
        eval: Whether to split train/test
        extension: Image extension

    Returns:
        SceneInfo object
    """
    print("Reading Nerf Synthetic Info")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension
    )
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension
    )

    if len(test_cam_infos) == 0:
        val_transforms_path = os.path.join(path, "transforms_val.json")
        if os.path.exists(val_transforms_path):
            print("[Warning] No valid test cameras found, using validation split")
            test_cam_infos = readCamerasFromTransforms(
                path, "transforms_val.json", white_background, extension
            )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    target_num_pts = 100_000
    if not torch.cuda.is_available():
        target_num_pts = 10_000
        print(
            f"[Warning] CUDA unavailable, using reduced init point count ({target_num_pts})"
        )

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = target_num_pts
        print(f"Generating random point cloud ({num_pts})...")

        # We need to be more careful with random point initialization
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        rgb = np.random.random((num_pts, 3))
        normals = np.zeros((num_pts, 3))

        storePly(ply_path, xyz, (rgb * 255).astype(np.uint8))

    try:
        xyz, rgb, normals = load_ply(ply_path)
    except:
        # If loading fails, recreate
        num_pts = target_num_pts
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        rgb = np.random.random((num_pts, 3))
        normals = np.zeros((num_pts, 3))
        storePly(ply_path, xyz, (rgb * 255).astype(np.uint8))

    if xyz.shape[0] > target_num_pts:
        sample_idx = np.random.choice(xyz.shape[0], target_num_pts, replace=False)
        xyz = xyz[sample_idx]
        rgb = rgb[sample_idx]
        normals = normals[sample_idx]

    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )

    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
}


# Helper functions for COLMAP format reading
import struct
import sys


def read_next_bytes(
    fid, num_bytes: int, format_char_sequence: str, endian_character: str = "<"
) -> Tuple[Any, ...]:
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError(f"Unexpected EOF while reading {num_bytes} bytes")
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = ColmapCameraIntrinsics(
                    id=camera_id, model=model, width=width, height=height, params=params
                )
    return cameras


class ColmapCameraIntrinsics:
    """COLMAP camera intrinsics (id, model, width, height, params)"""

    def __init__(self, id, model, width, height, params):
        self.id = id
        self.model = model
        self.width = width
        self.height = height
        self.params = params


class ImageInfo:
    """Simple image info class for COLMAP parsing"""

    def __init__(self, id, qvec, tvec, camera_id, name, xys, point3D_ids):
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name
        self.xys = xys
        self.point3D_ids = point3D_ids


class Point3DInfo:
    """Simple point3D info class for COLMAP parsing"""

    def __init__(self, id, xyz, rgb, error, image_ids, point2D_idxs):
        self.id = id
        self.xyz = xyz
        self.rgb = rgb
        self.error = error
        self.image_ids = image_ids
        self.point2D_idxs = point2D_idxs


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params
            )
            cameras[camera_id] = ColmapCameraIntrinsics(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteImagesText(const std::string& path)
        void Reconstruction::ReadImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = ImageInfo(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteImagesBinary(const std::string& path)
        void Reconstruction::ReadImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = ImageInfo(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WritePoints3DText(const std::string& path)
        void Reconstruction::ReadPoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3DInfo(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs,
                )

    # Convert to arrays
    xyzs = np.array([points3D[k].xyz for k in points3D])
    rgbs = np.array([points3D[k].rgb for k in points3D])
    normals = np.zeros_like(xyzs)

    return xyzs, rgbs, normals


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WritePoints3DBinary(const std::string& path)
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3DInfo(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )

    xyzs = np.array([points3D[k].xyz for k in points3D])
    rgbs = np.array([points3D[k].rgb for k in points3D])
    normals = np.zeros_like(xyzs)

    return xyzs, rgbs, normals


CAMERA_MODELS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
    "FULL_OPENCV": 6,
    "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8,
    "RADIAL_FISHEYE": 9,
    "THIN_PRISM_FISHEYE": 10,
}


@dataclass(frozen=True)
class CameraModel:
    model_name: str
    num_params: int


CAMERA_MODEL_IDS: Dict[int, CameraModel] = {
    0: CameraModel(model_name="SIMPLE_PINHOLE", num_params=3),
    1: CameraModel(model_name="PINHOLE", num_params=4),
}


def qvec2rotmat(qvec):
    """
    Convert quaternion to rotation matrix

    Args:
        qvec: Quaternion [w, x, y, z]

    Returns:
        3x3 rotation matrix
    """
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    """
    Convert rotation matrix to quaternion

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion [w, x, y, z]
    """
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def focal2fov(focal, pixels):
    """Convert focal length to field of view"""
    import math

    return 2 * math.atan(pixels / (2 * focal))


def fov2focal(fov, pixels):
    """Convert field of view to focal length"""
    import math

    return pixels / (2 * math.tan(fov / 2))


read_extrinsics_text = read_images_text
read_extrinsics_binary = read_images_binary
read_intrinsics_text = read_cameras_text
read_intrinsics_binary = read_cameras_binary
