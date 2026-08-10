"""Frame quality estimation based on Laplacian sharpness metrics."""

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


def find_reference_frame(images: List[np.ndarray]) -> Tuple[int, np.ndarray]:
    """Find the sharpest frame in a sequence of images.

    Args:
        images: Non-empty list of image arrays.

    Returns:
        Tuple ``(index, image)`` where ``index`` is the position of the sharpest
        frame in ``images`` and ``image`` is the corresponding array.

    Raises:
        ValueError: If ``images`` is empty.
    """
    if not images:
        raise ValueError("The image sequence is empty; cannot choose a reference frame.")

    best_index = 0
    best_sharpness = calculate_sharpness(images[0])

    for index, image in enumerate(images[1:], start=1):
        sharpness = calculate_sharpness(image)
        if sharpness > best_sharpness:
            best_sharpness = sharpness
            best_index = index

    return best_index, images[best_index]
