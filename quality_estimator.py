"""Frame quality estimation based on sharpness and geometric stability metrics."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def calculate_sharpness(img: np.ndarray) -> float:
    """Estimate image sharpness as the variance of the Laplacian response.

    Args:
        img: Input image in grayscale or RGB/BGR format.

    Returns:
        Scalar sharpness score, where higher values indicate sharper images.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def calculate_geometric_stability(img: np.ndarray, mean_img: np.ndarray) -> float:
    """Estimate how close a frame is to the mean of the sequence.

    A frame with low mean absolute deviation from the temporal average is
    geometrically stable: its distortion field is close to the average
    distortion rather than an outlier.

    Args:
        img: Single frame as a float32 or uint8 array.
        mean_img: Per-pixel temporal mean of the sequence, same shape.

    Returns:
        Scalar stability score; higher values indicate less deviation from the
        mean (i.e. a more representative / less distorted frame).
    """
    diff = cv2.absdiff(img.astype(np.float32), mean_img)
    mad = float(diff.mean())
    # Invert so that higher is better, matching the sharpness convention
    return 1.0 / (mad + 1e-6)


def find_reference_frame(images: List[np.ndarray]) -> Tuple[int, np.ndarray]:
    """Find the best reference frame combining sharpness and geometric stability.

    The score is the product of the normalised Laplacian-sharpness score and
    the normalised geometric-stability score.  This avoids choosing a frame
    that is merely sharp due to a random deformation that happens to create
    strong local gradients, while also avoiding frames that are extremely
    blurry despite being geometrically close to the mean.

    Args:
        images: Non-empty list of image arrays.

    Returns:
        Tuple ``(index, image)`` where ``index`` is the position of the best
        frame in ``images`` and ``image`` is the corresponding array.

    Raises:
        ValueError: If ``images`` is empty.
    """
    if not images:
        raise ValueError("The image sequence is empty; cannot choose a reference frame.")

    # Temporal mean (float32 to avoid overflow)
    stack = np.stack([img.astype(np.float32) for img in images], axis=0)
    mean_img = stack.mean(axis=0)

    sharpness_scores = np.array([calculate_sharpness(img) for img in images])
    stability_scores = np.array([calculate_geometric_stability(img, mean_img) for img in images])

    # Normalise each metric to [0, 1] to give them equal weight
    def _normalise(arr: np.ndarray) -> np.ndarray:
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / rng if rng > 0 else np.ones_like(arr)

    combined = _normalise(sharpness_scores) * _normalise(stability_scores)
    best_index = int(np.argmax(combined))

    return best_index, images[best_index]
