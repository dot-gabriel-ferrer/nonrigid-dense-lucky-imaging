"""Main execution entrypoint for non-rigid dense lucky imaging."""

from __future__ import annotations

import argparse
import os

import cv2

from config import NUM_IMAGES
from optical_flow import compute_dense_flow
from quality_estimator import find_reference_frame
from synthetic_data import generate_dataset
from video_pipeline import VideoProcessingOptions, process_image, process_video
from warping import revert_deformation


def run_synthetic_demo() -> None:
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


def main() -> None:
    """Run the synthetic demo or reconstruct an image from a local video."""
    parser = argparse.ArgumentParser(description="Planetary image and video reconstruction")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--video", help="Path to a planetary video")
    source_group.add_argument("--image", help="Path to a planetary still image")
    parser.add_argument("--output", default="output/video", help="Output directory for video processing")
    parser.add_argument("--keep", type=float, default=1.0, help="Fraction of aligned frames to stack")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to decode")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth decoded frame")
    parser.add_argument("--superres-scale", type=int, choices=(1, 2, 3, 4), default=4,
                        help="EDSR enlargement of the restored stack (default: 4)")
    parser.add_argument("--drizzle-scale", type=int, choices=(1, 2, 3), default=2,
                        help="Sub-pixel drizzle reconstruction scale before FFT restoration (default: 2)")
    parser.add_argument("--mesh-scale", type=int, choices=(1, 2, 3), default=1,
                        help="Optional dense mesh reconstruction scale after stacking (default: 1)")
    parser.add_argument("--mesh-iterations", type=int, default=5,
                        help="Dense mesh back-projection iterations (default: 5)")
    parser.add_argument("--no-nonrigid", action="store_true",
                        help="Disable native-resolution dense optical-flow refinement")
    args = parser.parse_args()
    if not args.video and not args.image:
        run_synthetic_demo()
        return
    options = VideoProcessingOptions(
        keep_fraction=args.keep, max_frames=args.max_frames, frame_step=args.frame_step,
        nonrigid=not args.no_nonrigid, superres_scale=args.superres_scale,
        drizzle_scale=args.drizzle_scale,
        mesh_scale=args.mesh_scale, mesh_iterations=args.mesh_iterations,
    )
    result = process_video(args.video, args.output, options) if args.video else process_image(args.image, args.output, options)
    print(f"Decoded {result['raw_frames']} frames; stacked {result['stacked_frames']}.")
    print(f"Crop: {result['crop']}")
    print(f"Final image: {result['final']}")


if __name__ == "__main__":
    main()
