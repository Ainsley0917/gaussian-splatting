"""
Main training script for 3D Gaussian Splatting
"""

import os
import torch
import torchvision
import sys
from datetime import datetime
from typing import Any, Callable, Dict, cast
import argparse
from random import randint

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.loss_utils import l1_loss as compute_l1_loss, ssim_loss as compute_ssim_loss
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


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


def _safe_elapsed_ms(iter_start, iter_end) -> float:
    if iter_start is None or iter_end is None:
        return 0.0
    try:
        return float(iter_start.elapsed_time(iter_end))
    except RuntimeError:
        return 0.0


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
):
    """
    Main training loop

    Args:
        dataset: Dataset parameters
        opt: Optimization parameters
        pipe: Pipeline parameters
        testing_iterations: Iterations to run testing
        saving_iterations: Iterations to save checkpoints
        checkpoint_iterations: Iterations to save checkpoints for resuming
        checkpoint: Path to checkpoint to resume from
        debug_from: Iteration to start debugging
    """
    # Initialize
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    runtime_device = _resolve_runtime_device(dataset.data_device)
    dataset.data_device = str(runtime_device)

    # Create Gaussian model
    gaussians = GaussianModel(dataset.sh_degree, data_device=dataset.data_device)

    # Create scene
    scene = Scene(dataset, gaussians)

    # Setup training
    gaussians.training_setup(opt)
    if gaussians.optimizer is None:
        raise RuntimeError("Optimizer was not initialized")

    # Load checkpoint if provided
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, map_location=runtime_device)
        restore_fn = cast(Callable[[Any, Any], None], gaussians.restore)
        restore_fn(model_params, opt)

    # Get background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=runtime_device)

    # Training loop
    iter_start = None
    iter_end = None
    if runtime_device.type == "cuda":
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # Increase SH degree every 1000 iterations
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick random training camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]

        # Get ground truth
        gt_image = viewpoint_cam.original_image.to(runtime_device)

        # Compute loss
        Ll1 = compute_l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - compute_ssim_loss(image.unsqueeze(0), gt_image.unsqueeze(0))
        )
        loss.backward()

        # Log
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
            progress_bar.update(10)

        if iteration == opt.iterations:
            progress_bar.close()

        # Training report
        training_report(
            tb_writer,
            iteration,
            Ll1,
            loss,
            _safe_elapsed_ms(iter_start, iter_end),
            testing_iterations,
            scene,
            render,
            (pipe, background),
        )

        # Optimizer step
        with torch.no_grad():
            # Progress bar
            if iter_start is not None:
                iter_start.record()

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                visibility_filter = render_pkg["visibility_filter"]
                radii = render_pkg["radii"]
                if radii is None:
                    continue
                if gaussians.max_radii2D is None:
                    raise RuntimeError("max_radii2D is not initialized")
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )

                # Add densification stats
                gaussians.add_densification_stats(
                    render_pkg["viewspace_points"], visibility_filter
                )

                # Densify and clone/split
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify(
                        opt.densify_grad_threshold,
                        opt.opacity_cull,
                        scene.cameras_extent,
                        size_threshold,
                    )

                # Reset opacity periodically
                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            if iter_end is not None:
                iter_end.record()

        # Save checkpoint
        if iteration in saving_iterations:
            print(f"\n[ITER {iteration}] Saving Checkpoint")
            scene.save(iteration)

        if iteration in checkpoint_iterations:
            print(f"\n[ITER {iteration}] Saving Checkpoint for resuming")
            capture_fn = cast(Callable[[], Any], gaussians.capture)
            torch.save(
                (capture_fn(), iteration),
                scene.model_path + f"/chkpnt{iteration}.pth",
            )


def prepare_output_and_logger(args):
    """Setup output directory and logger"""
    if not args.model_path:
        job_id = os.getenv("OAR_JOB_ID")
        if job_id:
            unique_str = job_id
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)

    # Save args
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create TensorBoard writer
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter(args.model_path)
    except ImportError:
        print("TensorBoard not available, logging to console only")

    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    """Log training progress"""
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)


def test_report(tb_writer, iteration, scene, renderFunc, renderArgs):
    """Run testing and report metrics"""
    # Render test views
    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        return

    psnr_test = 0.0
    for idx, viewpoint in enumerate(test_cameras):
        render_result = renderFunc(viewpoint, scene.gaussians, *renderArgs)
        image = render_result["render"]
        gt_image = viewpoint.original_image.to(image.device)
        psnr_test += psnr(image, gt_image).mean().double()

    psnr_test /= len(test_cameras)

    if tb_writer:
        tb_writer.add_scalar("test/psnr", psnr_test, iteration)

    print(f"\n[ITER {iteration}] Evaluating test: PSNR = {psnr_test:.4f}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    print("Optimizing " + args.model_path)

    # Initialize system state
    safe_state(args.quiet)

    # Enable anomaly detection if requested
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # Start training
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )

    # All done
    print("\nTraining complete.")
