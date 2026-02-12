"""
Metrics computation script - PSNR, SSIM, LPIPS evaluation
"""

import torch
import torch.nn.functional as F
import os
import sys
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.loss_utils import calculate_ssim
from argparse import ArgumentParser
from arguments import ModelParams

lpips_lib = None
try:
    import lpips as lpips_lib

    lpips_lib_available = True
except ImportError:
    lpips_lib_available = False
    print("lpips library not available, LPIPS will not be computed")


def read_images(renders_dir, gts_dir):
    """
    Read rendered and ground truth images

    Args:
        renders_dir: Directory with rendered images
        gts_dir: Directory with ground truth images

    Returns:
        List of (render, gt) tuples as tensors
    """
    renders = []
    gts = []

    render_files = sorted([f for f in os.listdir(renders_dir) if f.endswith(".png")])

    for fname in render_files:
        # Load render
        render_path = os.path.join(renders_dir, fname)
        render = Image.open(render_path)
        render = np.array(render) / 255.0
        render = torch.from_numpy(render).permute(2, 0, 1).float()

        # Load ground truth
        gt_path = os.path.join(gts_dir, fname)
        gt = Image.open(gt_path)
        gt = np.array(gt) / 255.0
        gt = torch.from_numpy(gt).permute(2, 0, 1).float()

        renders.append(render)
        gts.append(gt)

    return list(zip(renders, gts))


def evaluate_model(model_path, iteration):
    """
    Evaluate model on test set

    Args:
        model_path: Path to model
        iteration: Iteration number

    Returns:
        Dictionary with metrics
    """
    test_renders_dir = os.path.join(model_path, "test", f"ours_{iteration}", "renders")
    test_gts_dir = os.path.join(model_path, "test", f"ours_{iteration}", "gt")

    if not os.path.exists(test_renders_dir):
        print(f"Test renders not found at {test_renders_dir}")
        return None

    # Read images
    image_pairs = read_images(test_renders_dir, test_gts_dir)

    if len(image_pairs) == 0:
        print("No images found")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize LPIPS
    if lpips_lib_available and lpips_lib is not None:
        loss_fn = lpips_lib.LPIPS(net="alex").to(device)
    else:
        loss_fn = None

    # Compute metrics
    psnrs = []
    ssims = []
    lpipss = []

    print(f"Evaluating {len(image_pairs)} images...")

    for render, gt in tqdm(image_pairs):
        render = render.to(device)
        gt = gt.to(device)

        # PSNR
        mse = torch.mean((render - gt) ** 2)
        if mse > 0:
            psnr = 10 * torch.log10(1.0 / mse)
        else:
            psnr = torch.tensor(float("inf"))
        psnrs.append(psnr.item())

        # SSIM
        ssim = calculate_ssim(render.unsqueeze(0), gt.unsqueeze(0))
        ssims.append(ssim.item())

        # LPIPS
        if loss_fn is not None:
            with torch.no_grad():
                lpips_val = loss_fn(render.unsqueeze(0), gt.unsqueeze(0))
            lpipss.append(lpips_val.item())

    # Compute averages
    metrics = {
        "PSNR": np.mean(psnrs),
        "SSIM": np.mean(ssims),
        "count": len(image_pairs),
    }

    if len(lpipss) > 0:
        metrics["LPIPS"] = np.mean(lpipss)

    return metrics


def evaluate(model_path, iteration):
    """
    Run evaluation and print results

    Args:
        model_path: Path to model output
        iteration: Iteration to evaluate (-1 for latest)
    """
    if iteration == -1:
        # Find latest iteration
        test_dir = os.path.join(model_path, "test")
        if os.path.exists(test_dir):
            subdirs = [d for d in os.listdir(test_dir) if d.startswith("ours_")]
            if subdirs:
                iteration = int(subdirs[-1].split("_")[1])
            else:
                print("No rendered images found")
                return
        else:
            print(f"Test directory not found: {test_dir}")
            return

    print(f"Evaluating iteration {iteration}...")

    metrics = evaluate_model(model_path, iteration)

    if metrics is None:
        return

    # Print results
    print("\n" + "=" * 50)
    print(f"Evaluation Results ({metrics['count']} images)")
    print("=" * 50)
    print(f"PSNR:  {metrics['PSNR']:.4f} dB")
    print(f"SSIM:  {metrics['SSIM']:.4f}")
    if "LPIPS" in metrics:
        print(f"LPIPS: {metrics['LPIPS']:.4f}")
    print("=" * 50)

    # Save to JSON
    results_file = os.path.join(model_path, f"results_{iteration}.json")
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Metrics evaluation script")
    model = ModelParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = parser.parse_args(sys.argv[1:])

    # Run evaluation
    evaluate(model.extract(args).model_path, args.iteration)
