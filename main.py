"""Main execution entrypoint for non-rigid dense lucky imaging."""

from __future__ import annotations

import os

import cv2

from config import NUM_IMAGES
from optical_flow import compute_dense_flow
from quality_estimator import find_reference_frame
from synthetic_data import generate_dataset
from warping import revert_deformation


def main() -> None:
    """Run the full synthetic turbulence generation and correction pipeline."""
    output_distorted = "output/1_distorted"
    output_reference = "output/2_reference"
    output_corrected = "output/3_corrected"

    os.makedirs(output_distorted, exist_ok=True)
    os.makedirs(output_reference, exist_ok=True)
    os.makedirs(output_corrected, exist_ok=True)

    images = generate_dataset(NUM_IMAGES)

    for i, image in enumerate(images):
        cv2.imwrite(f"{output_distorted}/distorted_{i:03d}.png", image)

    reference_index, reference_image = find_reference_frame(images)
    print(f"Reference frame index: {reference_index}")
    cv2.imwrite(f"{output_reference}/reference_{reference_index:03d}.png", reference_image)

    for i, image in enumerate(images):
        if i == reference_index:
            continue
        flow = compute_dense_flow(reference_image, image)
        corrected = revert_deformation(image, flow)
        cv2.imwrite(f"{output_corrected}/corrected_{i:03d}.png", corrected)


if __name__ == "__main__":
    main()
