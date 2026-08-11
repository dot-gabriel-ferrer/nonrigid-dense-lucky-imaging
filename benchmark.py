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

    # --- select reference frame (lucky frame) ---
    ref_idx, ref_img = find_reference_frame(images)
    print(f"  Lucky reference frame : #{ref_idx}  "
          f"(sharpness={calculate_sharpness(ref_img):.1f}, "
          f"PSNR vs GT={_psnr(base, ref_img):.2f} dB)\n")

    # --- correct each frame ---
    print("Running correction pipeline…")
    psnr_before: List[float] = []
    psnr_after: List[float] = []
    ssim_before: List[float] = []
    ssim_after: List[float] = []
    sharp_before: List[float] = []
    sharp_after: List[float] = []

    # The running accumulator starts with just the lucky frame itself
    running_sum: np.ndarray = ref_img.astype(np.float64)
    running_count: int = 1
    # Snapshots: (n_frames_stacked, running_average_uint8)
    stacking_snapshots: List[Tuple[int, np.ndarray]] = [
        (1, ref_img.copy())
    ]
    snapshot_steps = _snapshot_schedule(num_images)

    t1 = time.perf_counter()
    for i, img in enumerate(images):
        if i == ref_idx:
            continue
        flow = compute_dense_flow(ref_img, img)
        corrected = revert_deformation(img, flow)

        running_sum += corrected.astype(np.float64)
        running_count += 1

        if running_count in snapshot_steps:
            snap = np.clip(running_sum / running_count, 0, 255).astype(np.uint8)
            stacking_snapshots.append((running_count, snap))

        psnr_before.append(_psnr(base, img))
        psnr_after.append(_psnr(base, corrected))
        ssim_before.append(_ssim(base, img))
        ssim_after.append(_ssim(base, corrected))
        sharp_before.append(calculate_sharpness(img))
        sharp_after.append(calculate_sharpness(corrected))

    correction_time = time.perf_counter() - t1
    n = len(psnr_before)
    fps = n / correction_time

    # --- final stacked average ---
    avg_stack = np.clip(running_sum / running_count, 0, 255).astype(np.uint8)
    # Ensure final stack is in snapshots
    if stacking_snapshots[-1][0] != running_count:
        stacking_snapshots.append((running_count, avg_stack))

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

    # --- save individual assets ---
    out_dir = "output/benchmark"
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(f"{out_dir}/ground_truth.png", base)
    cv2.imwrite(f"{out_dir}/lucky_frame.png", ref_img)
    # Pick a distorted sample that is NOT the lucky frame
    sample_idx = next(i for i in range(len(images)) if i != ref_idx)
    distorted_sample = images[sample_idx]
    cv2.imwrite(f"{out_dir}/distorted_sample.png", distorted_sample)
    flow_sample = compute_dense_flow(ref_img, distorted_sample)
    corrected_sample = revert_deformation(distorted_sample, flow_sample)
    cv2.imwrite(f"{out_dir}/corrected_sample.png", corrected_sample)
    cv2.imwrite(f"{out_dir}/stacked_average.png", avg_stack)

    # --- comparison strip (README asset) ---
    strip_path = _build_comparison_strip(
        base, ref_img, distorted_sample, corrected_sample, avg_stack,
        running_count,
    )

    # --- stacking GIF (README asset) ---
    gif_path = _build_stacking_gif(base, stacking_snapshots)

    print(f"  Output images written to {out_dir}/")
    print(f"  Comparison strip      : {strip_path}")
    print(f"  Stacking GIF          : {gif_path}")
    print(f"{'='*64}\n")


def _snapshot_schedule(num_images: int) -> List[int]:
    """Return a list of running-count values at which to capture GIF frames.

    Logarithmically spaced so early frames show fast improvement and later
    frames show convergence.

    Args:
        num_images: Total number of frames in the sequence.

    Returns:
        Sorted list of integer counts (including ``num_images``).
    """
    import math
    steps: list[int] = set()
    # Logarithmic spacing: 2, 3, 4, 6, 8, 12, 16, ... up to num_images
    k = 2
    while k <= num_images:
        steps.add(k)
        k = max(k + 1, int(k * 1.35))
    steps.add(num_images)
    return sorted(steps)


def _build_comparison_strip(
    ground_truth: np.ndarray,
    lucky_frame: np.ndarray,
    distorted: np.ndarray,
    corrected: np.ndarray,
    stacked: np.ndarray,
    num_stacked: int,
) -> str:
    """Build and save a 5-panel labeled comparison strip for the README.

    Panels (left to right):
      1. Ground truth — the ideal undeformed planet.
      2. Lucky frame  — the sharpest raw frame selected as reference.
      3. Typical distorted frame — shows severity of turbulence.
      4. Single corrected frame — that same distorted frame after flow correction.
      5. Stacked average — mean of all ``num_stacked`` corrected frames.

    Args:
        ground_truth: Clean base planet image (uint8, HxWx3).
        lucky_frame: Selected reference (sharpest distorted) frame.
        distorted: Representative raw distorted frame (uint8, HxWx3).
        corrected: Non-rigidly corrected version of the distorted frame.
        stacked: Mean stack of all corrected frames (uint8, HxWx3).
        num_stacked: Total number of frames that were stacked.

    Returns:
        Path to the saved comparison strip PNG.
    """
    panels = [ground_truth, lucky_frame, distorted, corrected, stacked]
    labels = [
        "Ground truth",
        "Lucky frame (ref)",
        "Distorted frame",
        "Single corrected",
        f"Stacked ({num_stacked} frames)",
    ]

    h, w = ground_truth.shape[:2]
    label_height = 40
    border = 4
    panel_w = w + 2 * border
    panel_h = h + 2 * border + label_height

    strip = np.zeros((panel_h, panel_w * len(panels), 3), dtype=np.uint8)
    strip[:] = (30, 30, 30)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    font_thickness = 1
    text_colour = (220, 220, 220)

    for i, (panel, label) in enumerate(zip(panels, labels)):
        x0 = i * panel_w + border
        y0 = border
        strip[y0: y0 + h, x0: x0 + w] = panel

        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        tx = i * panel_w + (panel_w - tw) // 2
        ty = border + h + label_height // 2 + th // 2
        cv2.putText(strip, label, (tx, ty), font, font_scale, text_colour,
                    font_thickness, cv2.LINE_AA)

    path = "docs/assets/comparison_strip_labeled.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, strip)
    return path


def _build_stacking_gif(
    ground_truth: np.ndarray,
    snapshots: List[Tuple[int, np.ndarray]],
) -> str:
    """Build and save an animated GIF showing iterative stacking convergence.

    Each frame of the GIF shows a side-by-side panel:
      - Left:  Ground truth (static reference).
      - Right: Running stacked average after ``n`` corrected frames.

    A text overlay on the right panel reports the frame count and live PSNR
    vs ground truth, letting viewers watch the image quality converge.

    Args:
        ground_truth: Clean base planet image (uint8, HxWx3).
        snapshots: List of ``(n_frames, avg_image)`` tuples in increasing order
            of ``n_frames``.

    Returns:
        Path to the saved GIF file.
    """
    h, w = ground_truth.shape[:2]
    gap = 6
    label_h = 44
    border = 4
    frame_w = 2 * (w + 2 * border) + gap
    frame_h = h + 2 * border + label_h

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thick = 1
    col_label = (220, 220, 220)
    col_psnr_good = (80, 200, 80)

    gif_frames: list[np.ndarray] = []

    for n_frames, avg_img in snapshots:
        canvas = np.full((frame_h, frame_w, 3), 30, dtype=np.uint8)

        # Left panel: ground truth
        x0 = border
        canvas[border: border + h, x0: x0 + w] = ground_truth
        lbl = "Ground truth"
        (tw, th), _ = cv2.getTextSize(lbl, font, font_scale, font_thick)
        cv2.putText(canvas, lbl,
                    (x0 + (w - tw) // 2, border + h + label_h // 2 + th // 2),
                    font, font_scale, col_label, font_thick, cv2.LINE_AA)

        # Right panel: running stack
        x1 = border + w + 2 * border + gap
        canvas[border: border + h, x1: x1 + w] = avg_img
        live_psnr = _psnr(ground_truth, avg_img)
        rbl = f"Stack  n={n_frames:3d}   PSNR {live_psnr:.2f} dB"
        (tw2, th2), _ = cv2.getTextSize(rbl, font, font_scale, font_thick)
        cv2.putText(canvas, rbl,
                    (x1 + (w - tw2) // 2, border + h + label_h // 2 + th2 // 2),
                    font, font_scale, col_psnr_good, font_thick, cv2.LINE_AA)

        # Convert BGR→RGB for Pillow
        gif_frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    path = "docs/assets/stacking_convergence.gif"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        from PIL import Image as PilImage
    except ImportError:
        print("  [warn] Pillow not installed; skipping GIF generation.")
        return path

    pil_frames = [PilImage.fromarray(f) for f in gif_frames]
    # Hold last frame longer so it's readable
    durations = [200] * len(pil_frames)
    durations[-1] = 1500

    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    return path


if __name__ == "__main__":
    n = int(os.environ.get("NUM_IMAGES", NUM_IMAGES))
    run_benchmark(n)
