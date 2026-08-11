"""Dense optical flow computation module."""

from __future__ import annotations

import cv2
import numpy as np

from config import OPTICAL_FLOW_PRESET


def compute_dense_flow(ref_img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
    """Compute dense DIS optical flow from reference image to target image.

    Args:
        ref_img: Reference image array.
        target_img: Target image array.

    Returns:
        Dense optical flow tensor of shape ``(H, W, 2)`` with ``float32`` values,
        where the last dimension stores horizontal and vertical displacements.
    """
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if ref_img.ndim == 3 else ref_img
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY) if target_img.ndim == 3 else target_img

    dis = cv2.DISOpticalFlow_create(OPTICAL_FLOW_PRESET)
    flow = dis.calc(ref_gray, target_gray, None)
    return flow
