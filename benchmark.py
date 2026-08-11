"""Performance benchmark for the non-rigid dense lucky-imaging pipeline.

Runs the full synthetic-data pipeline, measures per-frame PSNR/SSIM/sharpness
before and after correction, and prints a summary table.  All images are 512×512
by default (matches ``config.py``).

Usage::

    python benchmark.py

Optional environment variable overrides:

    NUM_IMAGES=80 python benchmark.py
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple

import cv2
import numpy as np

from config import ELASTIC_ALPHA, ELASTIC_SIGMA, HEIGHT, NUM_IMAGES, WIDTH
from optical_flow import compute_dense_flow
from quality_estimator import calculate_sharpness, find_reference_frame
from synthetic_data import elastic_deformation, generate_base_planet, generate_dataset
from warping import revert_deformation

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB) between two uint8 images."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse == 0 else 20.0 * np.log10(255.0 / np.sqrt(mse))


def _ssim_single_channel(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    mu1, mu2 = a.mean(), b.mean()
    sigma1_sq, sigma2_sq = a.var(), b.var()
    sigma12 = np.mean((a - mu1) * (b - mu2))
    c1, c2 = 6.5025, 58.5225  # (0.01*255)^2, (0.03*255)^2
    return ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    )


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM across all channels."""
    if a.ndim == 2:
        return _ssim_single_channel(a, b)
    return float(np.mean([_ssim_single_channel(a[:, :, c], b[:, :, c]) for c in range(a.shape[2])]))


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def run_benchmark(num_images: int = NUM_IMAGES) -> None:
    """Run the pipeline on ``num_images`` synthetic frames and print metrics."""

    print(f"\n{'='*64}")
    print(f"  Non-Rigid Dense Lucky Imaging — Performance Benchmark")
    print(f"{'='*64}")
    print(f"  Resolution : {WIDTH}×{HEIGHT} px")
    print(f"  Frames     : {num_images}")
    print(f"  Turbulence : alpha={ELASTIC_ALPHA}, sigma={ELASTIC_SIGMA}")
    print(f"{'='*64}\n")

    # --- generate synthetic data ---
    print("Generating synthetic planetary data…")
    t0 = time.perf_counter()
    base = generate_base_planet(WIDTH, HEIGHT)
    images = generate_dataset(num_images)
    gen_time = time.perf_counter() - t0
    print(f"  Generated {num_images} frames in {gen_time:.2f} s\n")

    # --- select reference frame ---
    ref_idx, ref_img = find_reference_frame(images)
    print(f"  Reference frame selected: #{ref_idx}  "
          f"(sharpness={calculate_sharpness(ref_img):.1f})\n")

    # --- correct each frame ---
    print("Running correction pipeline…")
    psnr_before: List[float] = []
    psnr_after: List[float] = []
    ssim_before: List[float] = []
    ssim_after: List[float] = []
    sharp_before: List[float] = []
    sharp_after: List[float] = []
    corrected_stack: List[np.ndarray] = [ref_img.astype(np.float64)]

    t1 = time.perf_counter()
    for i, img in enumerate(images):
        if i == ref_idx:
            continue
        flow = compute_dense_flow(ref_img, img)
        corrected = revert_deformation(img, flow)
        corrected_stack.append(corrected.astype(np.float64))

        psnr_before.append(_psnr(base, img))
        psnr_after.append(_psnr(base, corrected))
        ssim_before.append(_ssim(base, img))
        ssim_after.append(_ssim(base, corrected))
        sharp_before.append(calculate_sharpness(img))
        sharp_after.append(calculate_sharpness(corrected))

    correction_time = time.perf_counter() - t1
    n = len(psnr_before)
    fps = n / correction_time

    # --- stacked average ---
    avg_stack = np.clip(np.mean(corrected_stack, axis=0), 0, 255).astype(np.uint8)
    psnr_stack = _psnr(base, avg_stack)
    ssim_stack = _ssim(base, avg_stack)

    # --- report ---
    pb, pa = float(np.mean(psnr_before)), float(np.mean(psnr_after))
    sb, sa = float(np.mean(ssim_before)), float(np.mean(ssim_after))
    shb, sha = float(np.mean(sharp_before)), float(np.mean(sharp_after))

    col = 24
    print(f"\n{'─'*64}")
    print(f"  {'Metric':<{col}} {'Distorted':>12}  {'Corrected':>12}  {'Stack avg':>12}")
    print(f"{'─'*64}")
    print(f"  {'PSNR (dB)':<{col}} {pb:>12.2f}  {pa:>12.2f}  {psnr_stack:>12.2f}")
    print(f"  {'SSIM':<{col}} {sb:>12.4f}  {sa:>12.4f}  {ssim_stack:>12.4f}")
    print(f"  {'Sharpness (Laplacian)':<{col}} {shb:>12.1f}  {sha:>12.1f}  {'—':>12}")
    print(f"{'─'*64}")
    print(f"\n  Throughput  : {n} frames in {correction_time:.2f} s  →  {fps:.1f} fps")
    print(f"  PSNR gain   : +{pa - pb:.2f} dB per frame  |  stack: +{psnr_stack - pb:.2f} dB")
    print(f"  SSIM gain   : +{sa - sb:.4f} per frame  |  stack: +{ssim_stack - sb:.4f}")
    print()

    # --- save assets ---
    out_dir = "output/benchmark"
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(f"{out_dir}/ground_truth.png", base)
    cv2.imwrite(f"{out_dir}/reference_frame.png", ref_img)
    sample_idx = next(i for i in range(len(images)) if i != ref_idx)
    cv2.imwrite(f"{out_dir}/distorted_sample.png", images[sample_idx])
    flow_sample = compute_dense_flow(ref_img, images[sample_idx])
    cv2.imwrite(f"{out_dir}/corrected_sample.png", revert_deformation(images[sample_idx], flow_sample))
    cv2.imwrite(f"{out_dir}/stacked_average.png", avg_stack)
    print(f"  Output images written to {out_dir}/")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    n = int(os.environ.get("NUM_IMAGES", NUM_IMAGES))
    run_benchmark(n)
