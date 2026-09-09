"""Detection and rigid tracking of the five physical LUMO LED anchors."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .fingertip_segmentation import FingertipRegion
from .canonical import CanonicalFingerConfig, build_canonical_map


LED_POSITIONS_MM = np.asarray((0.0, 11.0, 22.0, 33.0, 44.0), dtype=np.float64)
LED_COUNT = len(LED_POSITIONS_MM)
_TOTAL_FINGERTIP_LENGTH_MM = 60.0
_LED1_OFFSET_FROM_DISTAL_MM = 10.5
_EXPECTED_LED_PITCH_FRACTION = 11.0 / _TOTAL_FINGERTIP_LENGTH_MM
_LED1_OFFSET_IN_PITCHES = _LED1_OFFSET_FROM_DISTAL_MM / 11.0
_MINIMUM_TRACK_CORRESPONDENCES = 4
_MINIMUM_SCALE = 0.80
_MAXIMUM_SCALE = 1.25
_MAXIMUM_RESIDUAL_SPACING_FRACTION = 0.20


@dataclass(frozen=True)
class LedGeometry:
    """Five distal-to-proximal image landmarks with known physical positions.

    The current side-view mounting defines smaller image ``y`` as distal. The
    physical coordinates are millimetres from distal LED1 toward proximal LED5.
    """

    landmarks_xy_px: np.ndarray
    positions_mm: np.ndarray
    median_spacing_px: float

    def __post_init__(self) -> None:
        landmarks = np.asarray(self.landmarks_xy_px, dtype=np.float64)
        positions = np.asarray(self.positions_mm, dtype=np.float64)
        if landmarks.shape != (LED_COUNT, 2) or not np.all(np.isfinite(landmarks)):
            raise ValueError("landmarks_xy_px must be a finite 5 x 2 array")
        if positions.shape != (LED_COUNT,) or not np.all(np.isfinite(positions)):
            raise ValueError("positions_mm must be a finite length-five vector")
        if not np.all(np.diff(positions) > 0.0):
            raise ValueError("positions_mm must increase from distal to proximal")
        if not np.isfinite(self.median_spacing_px) or self.median_spacing_px <= 0.0:
            raise ValueError("median_spacing_px must be finite and positive")
        landmarks = landmarks.copy()
        positions = positions.copy()
        landmarks.setflags(write=False)
        positions.setflags(write=False)
        object.__setattr__(self, "landmarks_xy_px", landmarks)
        object.__setattr__(self, "positions_mm", positions)


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be an H x W x 3 uint8 array")
    return image


def _geometry(landmarks_xy_px: np.ndarray) -> LedGeometry:
    landmarks = np.asarray(landmarks_xy_px, dtype=np.float64)
    if landmarks.shape != (LED_COUNT, 2) or not np.all(np.isfinite(landmarks)):
        raise RuntimeError("LED landmarks are not a finite 5 x 2 array")
    spacing = np.linalg.norm(np.diff(landmarks, axis=0), axis=1)
    if np.any(spacing <= np.finfo(np.float64).eps):
        raise RuntimeError("LED landmarks contain coincident points")
    median_spacing = float(np.median(spacing))
    if float(np.std(spacing) / np.mean(spacing)) > 0.30:
        raise RuntimeError("LED landmark spacing is not sufficiently regular")
    return LedGeometry(
        landmarks_xy_px=landmarks,
        positions_mm=LED_POSITIONS_MM,
        median_spacing_px=median_spacing,
    )


def _led_response(rgb: np.ndarray) -> np.ndarray:
    scale = max(rgb.shape[0] / 480.0, 0.5)
    red = rgb[:, :, 0].astype(np.float32)
    red_local = cv2.GaussianBlur(red, (0, 0), 1.2 * scale)
    red_background = cv2.GaussianBlur(red, (0, 0), 14.0 * scale)
    return np.maximum(red_local - red_background, 0.0)


def detect_leds(rgb: np.ndarray, fingertip: FingertipRegion) -> LedGeometry:
    """Register the known five-anchor lattice in silhouette coordinates.

    The longitudinal locations come from the physical 60-mm fingertip layout:
    LED1 is 10.5 mm from the distal silhouette end and the five anchors have an
    exact 11-mm pitch.  Photometric evidence selects only the common transverse
    LED line and a small shared longitudinal offset; it cannot move individual
    anchors onto unrelated peaks or terminal escape light.
    """

    image = _validate_rgb(rgb)
    if fingertip.mask.shape != image.shape[:2]:
        raise ValueError("fingertip mask must match the RGB image")
    canonical_map = build_canonical_map(
        fingertip,
        CanonicalFingerConfig(
            longitudinal_samples=256,
            transverse_samples=128,
            transverse_inset_fraction=0.04,
        ),
    )
    response = _led_response(image)
    canonical_response = cv2.remap(
        response,
        canonical_map.map_x,
        canonical_map.map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    canonical_response = cv2.GaussianBlur(canonical_response, (0, 0), 2.0)
    longitudinal_count, transverse_count = canonical_response.shape
    expected_pitch_rows = _EXPECTED_LED_PITCH_FRACTION * (longitudinal_count - 1)
    column_start = max(1, round(0.04 * transverse_count))
    column_stop = min(transverse_count - 1, round(0.96 * transverse_count))
    best: tuple[float, float, float, int] | None = None
    for pitch_rows in np.linspace(
        0.65 * expected_pitch_rows,
        1.15 * expected_pitch_rows,
        81,
    ):
        for offset_rows in np.linspace(
            -0.30 * pitch_rows,
            0.30 * pitch_rows,
            25,
        ):
            rows = (
                _LED1_OFFSET_IN_PITCHES * pitch_rows
                + offset_rows
                + np.arange(LED_COUNT) * pitch_rows
            )
            if rows[0] < 0.5 or rows[-1] >= longitudinal_count - 0.5:
                continue
            row_indices = np.rint(rows).astype(np.int32)
            midpoint_indices = np.rint(rows[:-1] + 0.5 * pitch_rows).astype(
                np.int32
            )
            # A five-tooth matched comb rewards simultaneous local support and
            # rejects one broad escape-light band that also illuminates the
            # spaces between nominal LED positions.
            comb_score = np.sum(canonical_response[row_indices], axis=0)
            comb_score -= 0.35 * np.sum(
                canonical_response[midpoint_indices],
                axis=0,
            )
            local_column = int(np.argmax(comb_score[column_start:column_stop]))
            column = column_start + local_column
            score = float(comb_score[column])
            if best is None or score > best[0]:
                best = (score, pitch_rows, offset_rows, column)
    if best is None or best[0] <= np.finfo(np.float32).eps:
        raise RuntimeError("five-LED silhouette lattice has no optical support")

    rows = (
        _LED1_OFFSET_IN_PITCHES * best[1]
        + best[2]
        + np.arange(LED_COUNT) * best[1]
    )
    columns = np.full(LED_COUNT, float(best[3]), dtype=np.float32)
    sample_x = cv2.remap(
        canonical_map.map_x,
        columns.reshape(-1, 1),
        rows.astype(np.float32).reshape(-1, 1),
        cv2.INTER_LINEAR,
    ).reshape(-1)
    sample_y = cv2.remap(
        canonical_map.map_y,
        columns.reshape(-1, 1),
        rows.astype(np.float32).reshape(-1, 1),
        cv2.INTER_LINEAR,
    ).reshape(-1)
    return _geometry(np.column_stack((sample_x, sample_y)))


def _fit_similarity(
    reference_xy_px: np.ndarray,
    current_xy_px: np.ndarray,
    valid: np.ndarray,
    reference_spacing_px: float,
) -> np.ndarray:
    reference = np.asarray(reference_xy_px, dtype=np.float64)
    current = np.asarray(current_xy_px, dtype=np.float64)
    accepted = np.asarray(valid, dtype=bool)
    if reference.shape != (LED_COUNT, 2) or current.shape != reference.shape:
        raise ValueError("reference and current landmarks must be 5 x 2 arrays")
    if accepted.shape != (LED_COUNT,):
        raise ValueError("valid must be a length-five mask")
    accepted &= np.all(np.isfinite(current), axis=1)
    if np.count_nonzero(accepted) < _MINIMUM_TRACK_CORRESPONDENCES:
        raise RuntimeError("LED similarity fit requires at least four correspondences")
    transform, inliers = cv2.estimateAffinePartial2D(
        reference[accepted],
        current[accepted],
        method=cv2.RANSAC,
        ransacReprojThreshold=0.15 * reference_spacing_px,
        maxIters=1000,
        confidence=0.99,
        refineIters=10,
    )
    if transform is None or inliers is None:
        raise RuntimeError("LED correspondences do not define a similarity transform")
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (2, 3) or not np.all(np.isfinite(transform)):
        raise RuntimeError("LED similarity transform is invalid")
    scale = float(np.hypot(transform[0, 0], transform[1, 0]))
    if not _MINIMUM_SCALE <= scale <= _MAXIMUM_SCALE:
        raise RuntimeError(f"LED similarity scale is implausible: {scale:.3f}")
    inlier_mask = np.asarray(inliers, dtype=bool).reshape(-1)
    if np.count_nonzero(inlier_mask) < _MINIMUM_TRACK_CORRESPONDENCES:
        raise RuntimeError("LED similarity fit retained fewer than four inliers")
    moved = np.column_stack((reference, np.ones(LED_COUNT))) @ transform.T
    residual = np.linalg.norm(
        moved[accepted][inlier_mask] - current[accepted][inlier_mask],
        axis=1,
    )
    if float(np.max(residual)) > _MAXIMUM_RESIDUAL_SPACING_FRACTION * reference_spacing_px:
        raise RuntimeError("LED similarity residual exceeds the physical-array limit")
    array_axis = moved[-1] - moved[0]
    if float(np.linalg.norm(array_axis)) <= np.finfo(np.float64).eps:
        raise RuntimeError("tracked LED array has no longitudinal extent")
    if np.any(np.diff(moved, axis=0) @ array_axis <= 0.0):
        raise RuntimeError("tracked LED ordering is not distal to proximal")
    return transform


def estimate_led_similarity_transform(
    reference_geometry: LedGeometry,
    current_geometry: LedGeometry,
) -> np.ndarray:
    """Estimate a robust reference-image to current-image similarity transform."""

    if not isinstance(reference_geometry, LedGeometry) or not isinstance(
        current_geometry,
        LedGeometry,
    ):
        raise TypeError("reference_geometry and current_geometry must be LedGeometry")
    return _fit_similarity(
        reference_geometry.landmarks_xy_px,
        current_geometry.landmarks_xy_px,
        np.ones(LED_COUNT, dtype=bool),
        reference_geometry.median_spacing_px,
    )


def track_leds(
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
    previous_geometry: LedGeometry,
) -> LedGeometry:
    """Track correspondences and apply one rigid similarity to all five LEDs."""

    previous = _validate_rgb(previous_rgb)
    current = _validate_rgb(current_rgb)
    if previous.shape != current.shape:
        raise ValueError("previous_rgb and current_rgb must have equal shape")
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    previous_points = previous_geometry.landmarks_xy_px.astype(np.float32).reshape(-1, 1, 2)
    window = max(21, round(1.4 * previous_geometry.median_spacing_px))
    if window % 2 == 0:
        window += 1
    parameters = {
        "winSize": (window, window),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_points,
        None,
        **parameters,
    )
    if current_points is None or forward_status is None:
        raise RuntimeError("forward LED optical flow returned no correspondences")
    candidates = current_points.reshape(-1, 2)
    valid = np.asarray(forward_status, dtype=bool).reshape(-1)
    valid &= np.all(np.isfinite(candidates), axis=1)
    if np.count_nonzero(valid) < _MINIMUM_TRACK_CORRESPONDENCES:
        raise RuntimeError("forward LED optical flow retained fewer than four points")

    indices = np.flatnonzero(valid)
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        current_points[indices],
        None,
        **parameters,
    )
    if backward is None or backward_status is None:
        raise RuntimeError("backward LED optical flow returned no correspondences")
    returned = np.full((LED_COUNT, 2), np.nan, dtype=np.float64)
    returned[indices] = backward.reshape(-1, 2)
    backward_valid = np.zeros(LED_COUNT, dtype=bool)
    backward_valid[indices] = np.asarray(backward_status, dtype=bool).reshape(-1)
    error = np.linalg.norm(returned - previous_points.reshape(-1, 2), axis=1)
    valid &= backward_valid & np.all(np.isfinite(returned), axis=1)
    valid &= error <= max(1.5, 0.12 * previous_geometry.median_spacing_px)
    transform = _fit_similarity(
        previous_geometry.landmarks_xy_px,
        candidates,
        valid,
        previous_geometry.median_spacing_px,
    )
    moved = np.column_stack(
        (previous_geometry.landmarks_xy_px, np.ones(LED_COUNT))
    ) @ transform.T
    return _geometry(moved)


def reanchor_leds(
    rgb: np.ndarray,
    fingertip: FingertipRegion,
    previous_geometry: LedGeometry,
    *,
    maximum_correction_spacing_fraction: float = 0.5,
) -> LedGeometry:
    """Accept an absolute redetection only when it remains near tracked geometry."""

    if not 0.0 < maximum_correction_spacing_fraction <= 1.0:
        raise ValueError("maximum_correction_spacing_fraction must be in (0, 1]")
    detected = detect_leds(rgb, fingertip)
    transform = estimate_led_similarity_transform(previous_geometry, detected)
    moved = np.column_stack(
        (previous_geometry.landmarks_xy_px, np.ones(LED_COUNT))
    ) @ transform.T
    correction = np.linalg.norm(moved - previous_geometry.landmarks_xy_px, axis=1)
    if float(np.max(correction)) > (
        maximum_correction_spacing_fraction * previous_geometry.median_spacing_px
    ):
        raise RuntimeError("absolute LED re-anchor correction is too large")
    return _geometry(moved)


__all__ = [
    "LED_POSITIONS_MM",
    "LedGeometry",
    "detect_leds",
    "estimate_led_similarity_transform",
    "reanchor_leds",
    "track_leds",
]
