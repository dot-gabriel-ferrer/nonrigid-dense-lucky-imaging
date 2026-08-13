"""End-to-end lucky-imaging pipeline for locally downloaded planetary videos."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.request import urlretrieve

import cv2
import numpy as np

from optical_flow import compute_dense_flow
from warping import revert_deformation


@dataclass(frozen=True)
class VideoProcessingOptions:
    """Controls for a planetary video reconstruction."""

    keep_fraction: float = 1.0
    max_frames: int | None = None
    frame_step: int = 1
    max_dimension: int = 900
    crop_scale: float = 2.4
    nonrigid: bool = True
    superres_scale: int = 4
    superres_model_path: Path | None = None
    drizzle_scale: int = 2
    mesh_scale: int = 1
    mesh_iterations: int = 5
    sharpen_amount: float = 0.55


@dataclass(frozen=True)
class PlanetDetection:
    """Planetary mask and centroid detected in one frame."""

    mask: np.ndarray
    center: tuple[float, float]


@dataclass(frozen=True)
class RegistrationMeasurement:
    """Translation diagnostics for one frame registered to the reference."""

    shift_x: float
    shift_y: float
    phase_response: float
    ecc_correlation: float | None


@dataclass(frozen=True)
class DrizzleDiagnostics:
    """Coverage characteristics of a dense-grid drizzle reconstruction."""

    scale: int
    covered_fraction: float
    mean_samples_per_covered_pixel: float


EDSR_X4_URL = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
DEFAULT_EDSR_X4_PATH = Path(__file__).parent / "models" / "EDSR_x4.pb"


class SuperResolutionModel:
    """OpenCV EDSR super-resolution model with reproducible local weights."""

    def __init__(self, scale: int, model_path: Path | None = None) -> None:
        if scale not in (2, 3, 4):
            raise ValueError("superres_scale must be 1, 2, 3, or 4.")
        path = model_path or DEFAULT_EDSR_X4_PATH
        if scale != 4:
            raise ValueError("Only EDSR 4x weights are currently bundled for automatic download.")
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading EDSR {scale}x weights to {path}...")
            urlretrieve(EDSR_X4_URL, path)
        if not hasattr(cv2, "dnn_superres"):
            raise RuntimeError("OpenCV dnn_superres is unavailable; install opencv-contrib-python.")
        self._engine = cv2.dnn_superres.DnnSuperResImpl_create()
        self._engine.readModel(str(path))
        self._engine.setModel("edsr", scale)

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """Return the learned EDSR enlargement of one BGR image."""
        return self._engine.upsample(image)


def _read_video(path: Path, options: VideoProcessingOptions) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    frames: list[np.ndarray] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % options.frame_step == 0:
            largest = max(frame.shape[:2])
            if largest > options.max_dimension:
                scale = options.max_dimension / largest
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(frame)
            if options.max_frames is not None and len(frames) >= options.max_frames:
                break
        index += 1
    capture.release()
    if len(frames) < 3:
        raise ValueError("The video must contain at least three readable frames.")
    return frames


def _planet_mask_and_centroid(frame: np.ndarray) -> PlanetDetection | None:
    """Return the largest bright object mask and its centroid."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3.0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates = [index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] > 20]
    if not candidates:
        return None
    largest = max(candidates, key=lambda index: stats[index, cv2.CC_STAT_AREA])
    object_mask = np.where(cv2.connectedComponentsWithStats(mask)[1] == largest, 255, 0).astype(np.uint8)
    object_mask = cv2.dilate(object_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    return PlanetDetection(object_mask, (float(centroids[largest][0]), float(centroids[largest][1])))


def _planet_crop_rect(frame: np.ndarray, crop_scale: float) -> tuple[int, int, int, int]:
    """Find a generous square ROI around the bright planetary disk/rings."""
    detected = _planet_mask_and_centroid(frame)
    h, w = frame.shape[:2]
    if detected is None:
        return 0, 0, w, h
    mask, (cx, cy) = detected.mask, detected.center
    x, y, component_w, component_h = cv2.boundingRect(mask)
    component_size = max(component_w, component_h)
    side = int(np.clip(component_size * crop_scale, 48, min(h, w)))
    x = int(np.clip(round(cx - side / 2), 0, w - side))
    y = int(np.clip(round(cy - side / 2), 0, h - side))
    return x, y, side, side


def _crop(frames: Sequence[np.ndarray], rect: tuple[int, int, int, int]) -> list[np.ndarray]:
    x, y, w, h = rect
    return [frame[y:y + h, x:x + w].copy() for frame in frames]


def _quality_scores(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Rank frames by detail while penalising clipped or nearly blank frames."""
    scores = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = _planet_mask_and_centroid(frame)
        if detected is None:
            scores.append(0.0)
            continue
        mask = detected.mask
        valid = gray[mask > 0]
        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F)[mask > 0].var())
        contrast = float(np.std(valid)) if valid.size else 0.0
        saturation = float(np.mean(valid >= 250)) if valid.size else 1.0
        scores.append(sharpness * (1.0 + contrast / 255.0) * (1.0 - saturation))
    return np.asarray(scores, dtype=np.float64)


def _centering_shift(reference_center: tuple[float, float], frame: np.ndarray) -> tuple[float, float]:
    """Return the coarse translation that centres a detected planet."""
    detected = _planet_mask_and_centroid(frame)
    if detected is None:
        return 0.0, 0.0
    center = detected.center
    return reference_center[0] - center[0], reference_center[1] - center[1]


def _translate(frame: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
    """Apply one sub-pixel translation with a high-quality interpolator."""
    matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REFLECT_101)


def _phase_shift(reference_gray: np.ndarray, reference_mask: np.ndarray, frame: np.ndarray) -> tuple[tuple[float, float], float]:
    """Measure the residual sub-pixel shift by masked phase correlation."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = reference_mask.astype(np.float32) / 255.0
    window = cv2.createHanningWindow((gray.shape[1], gray.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(
        reference_gray.astype(np.float32) * mask,
        gray.astype(np.float32) * mask,
        window,
    )
    if response < 0.03 or not np.all(np.isfinite(shift)):
        return (0.0, 0.0), float(response)
    # phaseCorrelate reports target displacement relative to reference. The
    # target therefore needs the inverse translation to land on the reference.
    return (-float(shift[0]), -float(shift[1])), float(response)


def _ecc_shift(reference_gray: np.ndarray, reference_mask: np.ndarray, frame: np.ndarray) -> tuple[tuple[float, float], float | None]:
    """Measure the residual masked ECC translation after phase alignment."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1e-5)
    try:
        correlation, warp = cv2.findTransformECC(reference_gray, gray, warp, cv2.MOTION_TRANSLATION, criteria, reference_mask)
    except cv2.error:
        return (0.0, 0.0), None
    # The legacy WARP_INVERSE_MAP application is equivalent to this inverse
    # translation, which can be combined with the other shifts before resampling.
    return (-float(warp[0, 2]), -float(warp[1, 2])), float(correlation)


def _register_to_reference(
    reference_gray: np.ndarray,
    reference_mask: np.ndarray,
    reference_center: tuple[float, float],
    frame: np.ndarray,
) -> tuple[np.ndarray, RegistrationMeasurement]:
    """Measure coarse and sub-pixel shifts, then resample the source once."""
    coarse = _centering_shift(reference_center, frame)
    centred = _translate(frame, coarse)
    phase, phase_response = _phase_shift(reference_gray, reference_mask, centred)
    phase_aligned = _translate(centred, phase)
    ecc, ecc_correlation = _ecc_shift(reference_gray, reference_mask, phase_aligned)
    total_shift = (coarse[0] + phase[0] + ecc[0], coarse[1] + phase[1] + ecc[1])
    measurement = RegistrationMeasurement(*total_shift, phase_response, ecc_correlation)
    return _translate(frame, total_shift), measurement


def _masked_alignment_error(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    """Return robust photometric mismatch on the planetary region only."""
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = np.abs(reference_gray - candidate_gray)[mask > 0]
    return float(np.median(residual)) if residual.size else float("inf")


def _validated_nonrigid_correction(
    reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None]:
    """Use DIS flow only when it improves masked agreement with the reference."""
    flow = compute_dense_flow(reference, candidate)
    corrected = revert_deformation(candidate, flow)
    if _masked_alignment_error(reference, corrected, mask) < _masked_alignment_error(reference, candidate, mask):
        return corrected, flow
    return candidate, None


def _sigma_clipped_mean(frames: Sequence[np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    """Stack frames after rejecting transient sensor noise and bad registrations."""
    stack = np.stack([frame.astype(np.float32) for frame in frames], axis=0)
    if weights is None:
        weights = np.ones(len(frames), dtype=np.float32)
    if weights.shape != (len(frames),):
        raise ValueError("weights must contain one value per frame.")
    median = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - median), axis=0)
    sigma = 1.4826 * mad + 1.0
    valid = np.abs(stack - median) <= 3.0 * sigma
    weighted_valid = valid * weights[:, np.newaxis, np.newaxis, np.newaxis]
    weight_sum = weighted_valid.sum(axis=0).clip(min=1e-6)
    return (stack * weighted_valid).sum(axis=0) / weight_sum


def _quality_weights(scores: np.ndarray) -> np.ndarray:
    """Return strictly positive detail weights without excluding any frame."""
    score_range = float(scores.max() - scores.min())
    normalized = (scores - scores.min()) / score_range if score_range > 0 else np.ones_like(scores)
    return (0.25 + 0.75 * normalized ** 2).astype(np.float32)


def _drizzle_stack(
    source_frames: Sequence[np.ndarray],
    registrations: Sequence[RegistrationMeasurement],
    flows: Sequence[np.ndarray | None],
    weights: np.ndarray,
    scale: int,
) -> tuple[np.ndarray, DrizzleDiagnostics]:
    """Drizzle original samples onto a dense reference grid.

    Unlike resizing an already stacked image, this method splats each observed
    source pixel according to its measured global translation and optional
    native-resolution flow. The bilinear footprint preserves sub-pixel dither
    information accumulated across frames.
    """
    if scale not in (1, 2, 3):
        raise ValueError("drizzle_scale must be 1, 2, or 3.")
    if not (len(source_frames) == len(registrations) == len(flows) == len(weights)):
        raise ValueError("Drizzle inputs must contain one item per selected frame.")
    height, width = source_frames[0].shape[:2]
    dense_height, dense_width = height * scale, width * scale
    accumulations = [np.zeros((dense_height, dense_width), dtype=np.float64) for _ in range(3)]
    coverage = np.zeros((dense_height, dense_width), dtype=np.float64)
    source_y, source_x = np.indices((height, width), dtype=np.float32)

    for frame, registration, flow, frame_weight in zip(source_frames, registrations, flows, weights):
        mapped_x = source_x + registration.shift_x
        mapped_y = source_y + registration.shift_y
        if flow is not None:
            # flow is defined reference -> globally aligned frame. A source
            # sample at p maps approximately to p + t - flow(p + t).
            sampled_flow_x = cv2.remap(flow[:, :, 0], mapped_x, mapped_y, cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            sampled_flow_y = cv2.remap(flow[:, :, 1], mapped_x, mapped_y, cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mapped_x -= sampled_flow_x
            mapped_y -= sampled_flow_y

        dense_x = mapped_x * scale
        dense_y = mapped_y * scale
        base_x = np.floor(dense_x).astype(np.int32)
        base_y = np.floor(dense_y).astype(np.int32)
        fraction_x = dense_x - base_x
        fraction_y = dense_y - base_y
        samples = frame.astype(np.float64)
        for offset_x, offset_y, footprint in (
            (0, 0, (1.0 - fraction_x) * (1.0 - fraction_y)),
            (1, 0, fraction_x * (1.0 - fraction_y)),
            (0, 1, (1.0 - fraction_x) * fraction_y),
            (1, 1, fraction_x * fraction_y),
        ):
            target_x = base_x + offset_x
            target_y = base_y + offset_y
            valid = (target_x >= 0) & (target_x < dense_width) & (target_y >= 0) & (target_y < dense_height)
            flat_index = (target_y[valid] * dense_width + target_x[valid]).ravel()
            contribution = (footprint[valid] * frame_weight).ravel()
            np.add.at(coverage.ravel(), flat_index, contribution)
            for channel in range(3):
                np.add.at(accumulations[channel].ravel(), flat_index,
                          contribution * samples[:, :, channel][valid].ravel())

    covered = coverage > 1e-6
    reconstructed = np.stack(accumulations, axis=2) / np.maximum(coverage[:, :, np.newaxis], 1e-6)
    # Fill unsupported border holes conservatively from neighbouring samples.
    reconstructed = cv2.inpaint(np.clip(reconstructed, 0, 255).astype(np.uint8),
                                (~covered).astype(np.uint8), 3, cv2.INPAINT_NS).astype(np.float32)
    diagnostics = DrizzleDiagnostics(
        scale=scale,
        covered_fraction=float(covered.mean()),
        mean_samples_per_covered_pixel=float(coverage[covered].mean()) if np.any(covered) else 0.0,
    )
    return reconstructed, diagnostics


class StreamingRobustStack:
    """Memory-safe weighted stack with online per-pixel outlier suppression."""

    def __init__(self, frame: np.ndarray, weight: float) -> None:
        self.mean = frame.astype(np.float32)
        self.variance = np.zeros_like(self.mean, dtype=np.float32)
        self.weight_sum = np.full_like(self.mean, weight, dtype=np.float32)
        self.frame_count = 1

    def add(self, frame: np.ndarray, weight: float) -> None:
        """Add one frame without retaining it after its weighted update."""
        sample = frame.astype(np.float32)
        residual = sample - self.mean
        if self.frame_count >= 8:
            sigma = np.sqrt(np.maximum(self.variance, 4.0))
            sample = self.mean + np.clip(residual, -3.0 * sigma, 3.0 * sigma)
        new_weight_sum = self.weight_sum + weight
        delta = sample - self.mean
        self.mean += (weight / new_weight_sum) * delta
        self.variance += (weight / new_weight_sum) * (delta * delta - self.variance)
        self.weight_sum = new_weight_sum
        self.frame_count += 1

    def result(self) -> np.ndarray:
        """Return the current robust weighted stack."""
        return self.mean


def _streaming_robust_stack(frames: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Build a robust stack from an iterable without retaining source frames."""
    if len(frames) != len(weights):
        raise ValueError("frames and weights must have the same length.")
    accumulator = StreamingRobustStack(frames[0], float(weights[0]))
    for frame, weight in zip(frames[1:], weights[1:]):
        accumulator.add(frame, float(weight))
    return accumulator.result()


def _write_registration_diagnostics(path: Path, measurements: Sequence[RegistrationMeasurement]) -> None:
    """Persist the measured translation and registration confidence per frame."""
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("frame_index", "shift_x_px", "shift_y_px", "phase_response", "ecc_correlation"))
        for index, measurement in enumerate(measurements):
            writer.writerow((
                index,
                f"{measurement.shift_x:.5f}",
                f"{measurement.shift_y:.5f}",
                f"{measurement.phase_response:.6f}",
                "" if measurement.ecc_correlation is None else f"{measurement.ecc_correlation:.6f}",
            ))


def _write_drizzle_diagnostics(path: Path, diagnostics: DrizzleDiagnostics) -> None:
    """Persist dense-grid coverage so the drizzle result is auditable."""
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("drizzle_scale", "covered_fraction", "mean_samples_per_covered_pixel"))
        writer.writerow((diagnostics.scale, f"{diagnostics.covered_fraction:.6f}",
                         f"{diagnostics.mean_samples_per_covered_pixel:.6f}"))


def _wiener_deconvolve(image: np.ndarray, blur_sigma: float = 0.9, balance: float = 0.025) -> np.ndarray:
    """Recover attenuated detail with a regularised Gaussian-PSF FFT inverse."""
    height, width = image.shape[:2]
    frequency_y = np.fft.fftfreq(height)[:, np.newaxis]
    frequency_x = np.fft.fftfreq(width)[np.newaxis, :]
    transfer = np.exp(-2.0 * np.pi ** 2 * blur_sigma ** 2 * (frequency_x ** 2 + frequency_y ** 2))
    inverse_filter = transfer / (transfer ** 2 + balance)
    restored_channels = []
    for channel in cv2.split(image):
        spectrum = np.fft.fft2(channel.astype(np.float32))
        restored_channels.append(np.fft.ifft2(spectrum * inverse_filter).real)
    return cv2.merge(restored_channels).astype(np.float32)


def _presentation_render(image: np.ndarray, amount: float) -> np.ndarray:
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoisingColored(image_u8, None, 3, 3, 7, 21)
    deconvolved = _wiener_deconvolve(denoised)
    detail = deconvolved - cv2.GaussianBlur(deconvolved, (0, 0), 1.1)
    return np.clip(deconvolved + amount * detail, 0, 255).astype(np.uint8)


def _mesh_superresolve(image: np.ndarray, scale: int, iterations: int) -> np.ndarray:
    """Densify the stacked pixel lattice using iterative mesh back-projection.

    A Lanczos seed defines a dense output mesh. Each iteration projects that
    mesh through the measured low-resolution sampling process, then distributes
    the residual back to the mesh vertices. This enforces consistency with the
    stack instead of using unconstrained sharpening to create apparent detail.
    """
    if scale == 1:
        return image.astype(np.float32)
    if scale not in (2, 3):
        raise ValueError("mesh_scale must be 1, 2, or 3.")
    height, width = image.shape[:2]
    dense_size = (width * scale, height * scale)
    mesh = cv2.resize(image.astype(np.float32), dense_size, interpolation=cv2.INTER_LANCZOS4)
    observation = image.astype(np.float32)
    for _ in range(iterations):
        projected = cv2.GaussianBlur(mesh, (0, 0), 0.6 * scale)
        projected = cv2.resize(projected, (width, height), interpolation=cv2.INTER_AREA)
        residual = observation - projected
        correction = cv2.resize(residual, dense_size, interpolation=cv2.INTER_CUBIC)
        mesh = np.clip(mesh + 0.35 * correction, 0, 255)
    return mesh


def process_image(image_path: str | Path, output_dir: str | Path, options: VideoProcessingOptions) -> dict[str, Path | int | tuple[int, int, int, int]]:
    """Denoise and restore a single planetary still image."""
    source = Path(image_path)
    if not source.is_file():
        raise FileNotFoundError(f"Image not found: {source}")
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {source}")
    largest = max(image.shape[:2])
    if largest > options.max_dimension:
        scale = options.max_dimension / largest
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    crop_rect = _planet_crop_rect(image, options.crop_scale)
    cropped = _crop([image], crop_rect)[0]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    linear_path = destination / "stack_linear.png"
    final_path = destination / f"{source.stem}_final.png"
    restored = _presentation_render(cropped, options.sharpen_amount)
    superresolved = SuperResolutionModel(options.superres_scale, options.superres_model_path).upscale(restored) if options.superres_scale > 1 else restored
    final = _mesh_superresolve(superresolved, options.mesh_scale, options.mesh_iterations)
    cv2.imwrite(str(linear_path), cropped)
    cv2.imwrite(str(destination / "stack_restored.png"), restored)
    cv2.imwrite(str(destination / "stack_superres.png"), superresolved)
    cv2.imwrite(str(final_path), np.clip(final, 0, 255).astype(np.uint8))
    return {"reference": linear_path, "linear": linear_path, "final": final_path,
            "raw_frames": 1, "stacked_frames": 1, "crop": crop_rect}


def process_video(video_path: str | Path, output_dir: str | Path, options: VideoProcessingOptions) -> dict[str, Path | int | tuple[int, int, int, int]]:
    """Reconstruct a planetary still image from a local video file."""
    if not 0 < options.keep_fraction <= 1:
        raise ValueError("keep_fraction must be in the interval (0, 1].")
    if options.frame_step < 1:
        raise ValueError("frame_step must be at least 1.")
    if options.superres_scale not in (1, 2, 3, 4):
        raise ValueError("superres_scale must be 1, 2, 3, or 4.")
    if options.drizzle_scale not in (1, 2, 3):
        raise ValueError("drizzle_scale must be 1, 2, or 3.")
    if options.mesh_scale not in (1, 2, 3):
        raise ValueError("mesh_scale must be 1, 2, or 3.")
    if options.mesh_iterations < 1:
        raise ValueError("mesh_iterations must be at least 1.")

    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    raw_frames = _read_video(source, options)
    crop_rect = _planet_crop_rect(raw_frames[len(raw_frames) // 2], options.crop_scale)
    frames = _crop(raw_frames, crop_rect)
    initial_scores = _quality_scores(frames)
    reference_index = int(np.argmax(initial_scores))
    reference = frames[reference_index]
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    detected_reference = _planet_mask_and_centroid(reference)
    if detected_reference is None:
        raise ValueError("Could not detect the planet in the selected reference frame.")
    reference_mask, reference_center = detected_reference.mask, detected_reference.center

    aligned: list[np.ndarray] = []
    registrations: list[RegistrationMeasurement] = []
    flows: list[np.ndarray | None] = []
    for index, frame in enumerate(frames):
        if index == reference_index:
            aligned.append(reference)
            registrations.append(RegistrationMeasurement(0.0, 0.0, 1.0, 1.0))
            flows.append(None)
            continue
        global_aligned, measurement = _register_to_reference(
            reference_gray, reference_mask, reference_center, frame
        )
        if options.nonrigid:
            global_aligned, flow = _validated_nonrigid_correction(
                reference, global_aligned, reference_mask
            )
            flows.append(flow)
        else:
            flows.append(None)
        aligned.append(global_aligned)
        registrations.append(measurement)

    aligned_scores = _quality_scores(aligned)
    selected_count = max(3, int(np.ceil(len(aligned) * options.keep_fraction)))
    selected_indices = np.argsort(aligned_scores)[-selected_count:]
    lucky_frames = [aligned[int(index)] for index in selected_indices]
    lucky_scores = aligned_scores[selected_indices]
    weights = _quality_weights(lucky_scores)
    selected_sources = [frames[int(index)] for index in selected_indices]
    selected_registrations = [registrations[int(index)] for index in selected_indices]
    selected_flows = [flows[int(index)] for index in selected_indices]
    linear_stack, drizzle_diagnostics = _drizzle_stack(
        selected_sources, selected_registrations, selected_flows, weights, options.drizzle_scale
    )
    restored_image = _presentation_render(linear_stack, options.sharpen_amount)
    superresolved_image = SuperResolutionModel(options.superres_scale, options.superres_model_path).upscale(restored_image) if options.superres_scale > 1 else restored_image
    final_image = _mesh_superresolve(superresolved_image, options.mesh_scale, options.mesh_iterations)
    reference_path = destination / "reference.png"
    linear_path = destination / "stack_linear.png"
    final_path = destination / f"{source.stem}_final.png"
    diagnostics_path = destination / "registration.csv"
    drizzle_path = destination / "drizzle.csv"
    cv2.imwrite(str(reference_path), reference)
    cv2.imwrite(str(linear_path), np.clip(linear_stack, 0, 255).astype(np.uint8))
    cv2.imwrite(str(destination / "stack_restored.png"), restored_image)
    cv2.imwrite(str(destination / "stack_superres.png"), superresolved_image)
    cv2.imwrite(str(final_path), np.clip(final_image, 0, 255).astype(np.uint8))
    _write_registration_diagnostics(diagnostics_path, registrations)
    _write_drizzle_diagnostics(drizzle_path, drizzle_diagnostics)

    return {"reference": reference_path, "linear": linear_path, "final": final_path, "diagnostics": diagnostics_path,
            "drizzle_diagnostics": drizzle_path,
            "raw_frames": len(frames), "stacked_frames": len(lucky_frames), "crop": crop_rect}