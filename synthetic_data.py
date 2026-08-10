"""Synthetic dataset generation utilities for turbulent lucky imaging experiments."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from config import ELASTIC_ALPHA, ELASTIC_SIGMA, HEIGHT, WIDTH


def generate_base_planet(w: int, h: int) -> np.ndarray:
    """Create a synthetic RGB planetary disk with internal high-frequency texture.

    Args:
        w: Output image width in pixels.
        h: Output image height in pixels.

    Returns:
        RGB image of shape ``(h, w, 3)`` and dtype ``uint8`` containing a
        gray filled planetary disk and high-frequency horizontal textures that
        remain confined to the interior of the disk.
    """
    planet = np.zeros((h, w, 3), dtype=np.uint8)

    center_x = w // 2
    center_y = h // 2
    radius = 150

    cv2.circle(planet, (center_x, center_y), radius, (150, 150, 150), thickness=-1)

    y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    disk_mask = (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2 <= radius**2

    rng = np.random.default_rng(seed=12345)
    base_pattern = rng.normal(loc=0.0, scale=18.0, size=(h, w)).astype(np.float32)

    stripe_frequency = 0.42
    stripe_signal = 20.0 * np.sin(y_indices.astype(np.float32) * stripe_frequency)
    high_frequency_texture = base_pattern + stripe_signal

    textured = planet.astype(np.float32)
    for channel in range(3):
        channel_data = textured[:, :, channel]
        channel_data[disk_mask] = np.clip(
            channel_data[disk_mask] + high_frequency_texture[disk_mask],
            0.0,
            255.0,
        )
        textured[:, :, channel] = channel_data

    return textured.astype(np.uint8)


def elastic_deformation(img: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    """Apply smooth elastic displacement to an image to mimic atmospheric turbulence.

    Args:
        img: Input image of shape ``(H, W, C)`` or ``(H, W)``.
        alpha: Displacement amplitude scale factor.
        sigma: Gaussian smoothing parameter controlling displacement smoothness.

    Returns:
        Distorted image produced by backward remapping with linear interpolation
        and reflective border handling.
    """
    h, w = img.shape[:2]

    dx = np.random.uniform(low=-1.0, high=1.0, size=(h, w)).astype(np.float32)
    dy = np.random.uniform(low=-1.0, high=1.0, size=(h, w)).astype(np.float32)

    dx = gaussian_filter(dx, sigma=sigma).astype(np.float32) * float(alpha)
    dy = gaussian_filter(dy, sigma=sigma).astype(np.float32) * float(alpha)

    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x_coords.astype(np.float32) + dx).astype(np.float32)
    map_y = (y_coords.astype(np.float32) + dy).astype(np.float32)

    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def generate_dataset(num: int) -> List[np.ndarray]:
    """Generate a list of elastically distorted synthetic planetary images.

    Args:
        num: Number of distorted frames to produce.

    Returns:
        List of ``num`` RGB images, each generated from the same base planet and
        independently deformed by stochastic turbulence fields.
    """
    base = generate_base_planet(WIDTH, HEIGHT)
    images: List[np.ndarray] = []

    for seed in range(num):
        np.random.seed(seed)
        distorted = elastic_deformation(base, alpha=ELASTIC_ALPHA, sigma=ELASTIC_SIGMA)
        images.append(distorted)

    return images
