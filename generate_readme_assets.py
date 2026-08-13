"""Generate README images from the local Saturn capture and reconstruction outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).parent
ASSETS = ROOT / "docs" / "assets"
VIDEO = ROOT / "input" / "saturn.mp4"
RESULTS = (
    ("50% of capture", 799, ROOT / "output" / "evaluation" / "first_50_percent" / "saturn_final.png"),
    ("75% of capture", 1198, ROOT / "output" / "evaluation" / "first_75_percent" / "saturn_final.png"),
    ("100% of capture", 1597, ROOT / "output" / "saturn" / "saturn_final.png"),
)
MESH_RESULT = ROOT / "output" / "documentation_mesh"


def _read_frames() -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(VIDEO))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, frame_count - 1, 42, dtype=int)
    sample_set = set(sample_indices.tolist())
    frames: list[np.ndarray] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in sample_set:
            frames.append(frame[0:360, 110:470])
        index += 1
    capture.release()
    return frames


def _label(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    height, width = image.shape[:2]
    banner = np.full((70, width, 3), 18, dtype=np.uint8)
    cv2.putText(banner, title, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(banner, subtitle, (14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (160, 196, 220), 1, cv2.LINE_AA)
    return np.vstack((banner, image))


def make_source_gif() -> Path:
    frames = _read_frames()
    gif_frames = []
    for number, frame in enumerate(frames, start=1):
        labelled = _label(frame, "Saturn source capture", f"Sampled frame {number}/{len(frames)} | 640 x 360 at 30 fps")
        gif_frames.append(Image.fromarray(cv2.cvtColor(labelled, cv2.COLOR_BGR2RGB)))
    path = ASSETS / "saturn_source_capture.gif"
    gif_frames[0].save(path, save_all=True, append_images=gif_frames[1:], duration=90, loop=0, optimize=True)
    return path


def make_comparison() -> Path:
    panels: list[np.ndarray] = []
    for title, frames, path in RESULTS:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Missing reconstruction result: {path}")
        image = cv2.resize(image, (360, 360), interpolation=cv2.INTER_LANCZOS4)
        panels.append(_label(image, title, f"All {frames:,} aligned frames stacked"))
    gap = np.full((430, 8, 3), 245, dtype=np.uint8)
    comparison = np.hstack([item for panel in panels for item in (panel, gap)][:-1])
    path = ASSETS / "saturn_all_frames_comparison.png"
    cv2.imwrite(str(path), comparison)
    return path


def _radial_power(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray))) ** 2
    height, width = gray.shape
    y, x = np.ogrid[:height, :width]
    radius = np.sqrt((x - width / 2) ** 2 + (y - height / 2) ** 2).astype(np.int32)
    counts = np.bincount(radius.ravel())
    power = np.bincount(radius.ravel(), spectrum.ravel()) / np.maximum(counts, 1)
    frequency = np.arange(len(power)) / max(height, width)
    return frequency[1:], power[1:]


def make_processing_comparison() -> Path:
    stages = (
        ("Drizzle 2x stack", "stack_linear.png", "Original sub-pixel samples on dense grid"),
        ("FFT restoration", "stack_restored.png", "Regularised Wiener deconvolution"),
        ("EDSR 4x", "stack_superres.png", "One learned enlargement after restoration"),
        ("Dense mesh 2x", "saturn_final.png", "Five back-projection iterations"),
    )
    panels: list[np.ndarray] = []
    for title, filename, subtitle in stages:
        image = cv2.imread(str(MESH_RESULT / filename))
        if image is None:
            raise FileNotFoundError(f"Missing processing stage: {filename}")
        image = cv2.resize(image, (520, 520), interpolation=cv2.INTER_AREA)
        panels.append(_label(image, title, subtitle))
    gap = np.full((590, 8, 3), 245, dtype=np.uint8)
    comparison = np.hstack([item for panel in panels for item in (panel, gap)][:-1])
    path = ASSETS / "saturn_processing_chain.png"
    cv2.imwrite(str(path), comparison)
    return path


def make_diagnostics() -> Path:
    registration = np.genfromtxt(MESH_RESULT / "registration.csv", delimiter=",", names=True)
    stages = (
        ("Drizzle 2x stack", "stack_linear.png", "#4c78a8"),
        ("FFT restoration", "stack_restored.png", "#f58518"),
        ("EDSR 4x", "stack_superres.png", "#54a24b"),
        ("Dense mesh 2x", "saturn_final.png", "#e45756"),
    )
    figure, (registration_axis, spectrum_axis) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    registration_axis.scatter(registration["shift_x_px"], registration["shift_y_px"],
                              c=registration["phase_response"], cmap="viridis", s=24, alpha=0.8)
    registration_axis.axhline(0, color="#777777", linewidth=0.8)
    registration_axis.axvline(0, color="#777777", linewidth=0.8)
    registration_axis.set_title("Sub-pixel registration coverage")
    registration_axis.set_xlabel("Horizontal translation (px)")
    registration_axis.set_ylabel("Vertical translation (px)")
    registration_axis.set_aspect("equal", adjustable="box")
    for title, filename, colour in stages:
        image = cv2.imread(str(MESH_RESULT / filename))
        frequency, power = _radial_power(image)
        spectrum_axis.semilogy(frequency, power, label=title, color=colour, linewidth=2)
    spectrum_axis.set_title("Radial Fourier power")
    spectrum_axis.set_xlabel("Spatial frequency (cycles/pixel)")
    spectrum_axis.set_ylabel("Power (log scale)")
    spectrum_axis.set_xlim(0, 0.5)
    spectrum_axis.grid(alpha=0.25)
    spectrum_axis.legend(frameon=False)
    path = ASSETS / "saturn_processing_diagnostics.png"
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate README assets from local Saturn processing outputs")
    parser.add_argument("--source", action="store_true", help="Generate the sampled source-capture GIF")
    parser.add_argument("--coverage", action="store_true", help="Generate the 50/75/100 percent comparison")
    parser.add_argument("--mesh", action="store_true", help="Generate dense-mesh comparison and diagnostics")
    args = parser.parse_args()
    selected = (args.source, args.coverage, args.mesh)
    ASSETS.mkdir(parents=True, exist_ok=True)
    if not any(selected) or args.source:
        print(make_source_gif())
    if not any(selected) or args.coverage:
        print(make_comparison())
    if not any(selected) or args.mesh:
        print(make_processing_comparison())
        print(make_diagnostics())


if __name__ == "__main__":
    main()