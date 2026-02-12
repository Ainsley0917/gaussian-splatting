"""
Render script - Generate novel views from trained model
"""

import torch
import torchvision
import os
import sys
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
import numpy as np
from PIL import Image


def _resolve_runtime_device(device_name: str) -> torch.device:
    try:
        device = torch.device(device_name)
    except (TypeError, RuntimeError):
        return torch.device("cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable, falling back to CPU.")
        return torch.device("cpu")

    if device.type == "mps":
        mps_available = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if not mps_available:
            print("[Warning] MPS unavailable, falling back to CPU.")
            return torch.device("cpu")

    return device


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    """
    Render a set of views and save images

    Args:
        model_path: Path to output model
        name: Name of the set (train/test)
        iteration: Iteration number
        views: List of views to render
        gaussians: Gaussian model
        pipeline: Pipeline parameters
        background: Background color
    """
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # Render
        render_result = render(view, gaussians, pipeline, background)
        rendering = render_result["render"]

        # Save rendered image
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, f"{view.image_name}.png")
        )

        # Save ground truth
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, f"{view.image_name}.png")
        )


def render_video(model_path, iteration, views, gaussians, pipeline, background, fps=30):
    """
    Render a video from a camera path

    Args:
        model_path: Path to output model
        iteration: Iteration number
        views: List of views to render
        gaussians: Gaussian model
        pipeline: Pipeline parameters
        background: Background color
        fps: Frames per second
    """
    render_path = os.path.join(model_path, "video", f"ours_{iteration}")
    os.makedirs(render_path, exist_ok=True)

    frames = []
    for idx, view in enumerate(tqdm(views, desc="Rendering video frames")):
        # Render
        render_result = render(view, gaussians, pipeline, background)
        rendering = render_result["render"]

        # Convert to numpy for video
        frame = rendering.permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        frames.append(frame)

        # Save individual frame
        Image.fromarray(frame).save(os.path.join(render_path, f"frame_{idx:04d}.png"))

    # Try to save as video
    try:
        import imageio

        video_path = os.path.join(model_path, f"video_{iteration}.mp4")
        imageio.mimsave(video_path, frames, fps=fps, quality=8)
        print(f"Video saved to {video_path}")
    except ImportError:
        print("imageio not available, frames saved as images")


def render_sets(
    dataset,
    iteration,
    pipeline,
    skip_train,
    skip_test,
    render_path,
):
    """
    Render train and test sets

    Args:
        dataset: Dataset parameters
        iteration: Iteration to load
        pipeline: Pipeline parameters
        skip_train: Skip rendering train set
        skip_test: Skip rendering test set
        render_path: Render camera path as video
    """
    with torch.no_grad():
        runtime_device = _resolve_runtime_device(dataset.data_device)
        dataset.data_device = str(runtime_device)

        # Load model
        gaussians = GaussianModel(dataset.sh_degree, data_device=dataset.data_device)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # Background color
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(
            bg_color, dtype=torch.float32, device=gaussians.device
        )

        if not skip_train:
            print("\nRendering training set...")
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
            )

        if not skip_test:
            print("\nRendering test set...")
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
            )

        if render_path:
            print("\nRendering camera path...")
            # Generate a circular camera path
            from utils.camera_utils import cameraList_from_camInfos
            from utils.graphics_utils import look_at

            # Get test cameras as base
            test_cams = scene.getTestCameras()
            if len(test_cams) == 0:
                test_cams = scene.getTrainCameras()

            # Create circular path around center
            num_frames = 120
            radius = 4.0
            height = 1.0

            path_cameras = []
            for i in range(num_frames):
                angle = 2 * np.pi * i / num_frames
                cam_pos = np.array(
                    [radius * np.cos(angle), height, radius * np.sin(angle)]
                )
                target = np.array([0, 0, 0])
                up = np.array([0, 1, 0])

                view_matrix = look_at(cam_pos, target, up)

                # Create camera with same intrinsics as first test camera
                base_cam = test_cams[0]

                # Note: This is a simplified version
                # In practice, you'd create proper Camera objects

            print(f"Camera path rendering would generate {num_frames} frames")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--render_path", action="store_true", help="Render a camera path"
    )
    args = parser.parse_args(sys.argv[1:])

    print(f"Rendering {args.model_path}")

    # Initialize system state
    safe_state(args.quiet)

    # Render sets
    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.render_path,
    )
