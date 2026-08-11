"""Global immutable configuration values for the lucky imaging pipeline."""

from __future__ import annotations

import cv2

WIDTH: int = 512
HEIGHT: int = 512
NUM_IMAGES: int = 120
ELASTIC_ALPHA: float = 400.0
ELASTIC_SIGMA: float = 20.0
OPTICAL_FLOW_PRESET: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
