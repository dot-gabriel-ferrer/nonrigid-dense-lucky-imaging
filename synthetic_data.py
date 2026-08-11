"""Synthetic dataset generation utilities for turbulent lucky imaging experiments."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from config import ELASTIC_ALPHA, ELASTIC_SIGMA, HEIGHT, WIDTH


def generate_base_planet(w: int, h: int) -> np.ndarray:
    """Create a synthetic RGB planetary disk modelled after a gas giant.

    The disk features:
    * A smooth limb-darkening gradient for physical realism.
    * Horizontal cloud bands alternating between warm tan and cool cream tones.
    * Fine high-frequency turbulence texture within each band.
    * Polar darkening caps at both poles.
    * A Great-Red-Spot-like oval storm in the southern equatorial belt.

    Args:
        w: Output image width in pixels.
        h: Output image height in pixels.

    Returns:
        RGB image of shape ``(h, w, 3)`` and dtype ``uint8``.
    """
    rng = np.random.default_rng(seed=42)

    planet = np.zeros((h, w, 3), dtype=np.float32)

    cx, cy = w // 2, h // 2
    radius = min(w, h) // 2 - 20

    y_idx, x_idx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r_sq = (x_idx - cx) ** 2 + (y_idx - cy) ** 2
    disk_mask = r_sq <= radius ** 2

    # Normalised latitude/longitude within the disk (-1 … +1)
    lat = np.where(disk_mask, (y_idx - cy) / radius, 0.0).astype(np.float32)
    lon_raw = np.where(disk_mask, (x_idx - cx) / radius, 0.0).astype(np.float32)
    # Convert to proper [-1,1] range accounting for the spherical projection
    r_norm = np.sqrt(np.clip(r_sq / radius ** 2, 0.0, 1.0)).astype(np.float32)

    # ------------------------------------------------------------------
    # Band colour palette  (BGR order)
    # ------------------------------------------------------------------
    # Each band defined as (latitude_centre, half_width, BGR_colour)
    bands = [
        # polar regions
        (0.85,  0.20, np.array([90,  100, 110], dtype=np.float32)),
        (-0.85, 0.20, np.array([90,  100, 110], dtype=np.float32)),
        # north equatorial belt
        (0.40,  0.14, np.array([80,  130, 180], dtype=np.float32)),
        # north temperate belt
        (0.65,  0.09, np.array([100, 140, 185], dtype=np.float32)),
        # equatorial zone (bright cream)
        (0.00,  0.18, np.array([160, 195, 220], dtype=np.float32)),
        # south equatorial belt
        (-0.30, 0.16, np.array([90,  135, 190], dtype=np.float32)),
        # south temperate belt
        (-0.58, 0.10, np.array([110, 148, 195], dtype=np.float32)),
        # south polar region
        (-0.82, 0.12, np.array([85,  105, 115], dtype=np.float32)),
    ]

    base_colour = np.array([130, 170, 210], dtype=np.float32)  # overall tint
    colour_map = np.ones((h, w, 3), dtype=np.float32) * base_colour

    for lat_c, hw, col in bands:
        weight = np.exp(-0.5 * ((lat - lat_c) / hw) ** 2)
        weight = weight[:, :, np.newaxis]
        colour_map = colour_map * (1.0 - weight) + col * weight

    # ------------------------------------------------------------------
    # Fine-grain texture (simulate cloud turbulence within bands)
    # ------------------------------------------------------------------
    noise = rng.standard_normal((h, w)).astype(np.float32)
    # Anisotropic blur: spread horizontally to mimic zonal flow
    noise_h = gaussian_filter(noise, sigma=(2.5, 7.0)) * 18.0
    noise_v = gaussian_filter(noise, sigma=(7.0, 2.5)) * 10.0
    texture = (noise_h + noise_v).astype(np.float32)

    # Stripe ripple (higher frequency within bands)
    ripple = (12.0 * np.sin(y_idx.astype(np.float32) * 0.55)
              + 6.0 * np.sin(y_idx.astype(np.float32) * 1.1 + 0.3)).astype(np.float32)
    fine_detail = texture + ripple

    for c in range(3):
        colour_map[:, :, c] += fine_detail * 0.35

    # ------------------------------------------------------------------
    # Great-Red-Spot-like oval (prominent southern storm)
    # ------------------------------------------------------------------
    grs_lat = -0.32        # latitude (in normalised coords)
    grs_lon = 0.18         # longitude offset
    grs_a   = 0.145        # semi-axis along longitude
    grs_b   = 0.065        # semi-axis along latitude
    grs_colour = np.array([60, 80, 190], dtype=np.float32)  # deep reddish-orange

    grs_dist = ((lat - grs_lat) / grs_b) ** 2 + ((lon_raw - grs_lon) / grs_a) ** 2
    grs_weight = np.exp(-2.0 * grs_dist).astype(np.float32)
    grs_weight[~disk_mask] = 0.0
    grs_weight = grs_weight[:, :, np.newaxis]
    colour_map = colour_map * (1.0 - grs_weight) + grs_colour * grs_weight

    # ------------------------------------------------------------------
    # Limb darkening  (cosine law, exponent 0.5 gives gentle falloff)
    # ------------------------------------------------------------------
    cos_theta = np.sqrt(np.clip(1.0 - r_norm ** 2, 0.0, 1.0)).astype(np.float32)
    limb = (cos_theta ** 0.5)[:, :, np.newaxis]
    colour_map = colour_map * limb

    # ------------------------------------------------------------------
    # Apply mask and clip
    # ------------------------------------------------------------------
    planet = np.where(disk_mask[:, :, np.newaxis], colour_map, 0.0)
    planet = np.clip(planet, 0.0, 255.0).astype(np.uint8)

    return planet


def elastic_deformation(
    img: np.ndarray,
    alpha: float,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply smooth elastic displacement to an image to mimic atmospheric turbulence.

    Args:
        img: Input image of shape ``(H, W, C)`` or ``(H, W)``.
        alpha: Displacement amplitude scale factor.
        sigma: Gaussian smoothing parameter controlling displacement smoothness.
        rng: Optional local random number generator used for displacement fields.

    Returns:
        Distorted image produced by backward remapping with linear interpolation
        and reflective border handling.
    """
    h, w = img.shape[:2]
    if rng is None:
        rng = np.random.default_rng()

    dx = rng.uniform(low=-1.0, high=1.0, size=(h, w)).astype(np.float32)
    dy = rng.uniform(low=-1.0, high=1.0, size=(h, w)).astype(np.float32)

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
        rng = np.random.default_rng(seed)
        distorted = elastic_deformation(
            base, alpha=ELASTIC_ALPHA, sigma=ELASTIC_SIGMA, rng=rng
        )
        images.append(distorted)

    return images



def elastic_deformation(
    img: np.ndarray,
    alpha: float,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply smooth elastic displacement to an image to mimic atmospheric turbulence.

    Args:
        img: Input image of shape ``(H, W, C)`` or ``(H, W)``.
        alpha: Displacement amplitude scale factor.
        sigma: Gaussian smoothing parameter controlling displacement smoothness.
        rng: Optional local random number generator used for displacement fields.

    Returns:
        Distorted image produced by backward remapping with linear interpolation
        and reflective border handling.
    """
    h, w = img.shape[:2]
    if rng is None:
        rng = np.random.default_rng()

    dx = rng.uniform(low=-1.0, high=1.0, size=(h, w)).astype(np.float32)
    dy = rng.uniform(low=-1.0, high=1.0, size=(h, w)).astype(np.float32)

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
        rng = np.random.default_rng(seed)
        distorted = elastic_deformation(
            base, alpha=ELASTIC_ALPHA, sigma=ELASTIC_SIGMA, rng=rng
        )
        images.append(distorted)

    return images
