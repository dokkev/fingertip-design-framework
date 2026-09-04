"""Simple fixed-setup Solaris LED localization."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .fingertip_segmentation import segment_fingertip
from .led_localization_common import LedLocalizationResult


LED_COUNT = 5

# Empirical choices of the verified fixed-camera method.
OUTER_SIDE_RED_PERCENTILE = 90.0
SMALL_SMOOTHING_SIGMA_SPAN_FRACTION = 0.008
BROAD_BACKGROUND_SIGMA_SPAN_FRACTION = 0.060
MINIMUM_PEAK_DISTANCE_SPAN_FRACTION = 0.025
MINIMUM_PROMINENCE_RANGE_FRACTION = 0.035
LOCAL_CENTROID_HALF_HEIGHT_PITCH_FRACTION = 0.28
BRIGHTEST_CENTROID_FRACTION = 0.05

# Broad image-normalized sequence sanity bounds.
MINIMUM_SEQUENCE_START_FRACTION = 0.10
MAXIMUM_SEQUENCE_STOP_FRACTION = 0.92
MINIMUM_SPACING_SPAN_FRACTION = 0.035
MAXIMUM_SPACING_SPAN_FRACTION = 0.22
MAXIMUM_SPACING_RATIO = 1.90
MAXIMUM_QUADRATIC_RMSE_PITCH_FRACTION = 0.15


def temporal_median_rgb(rgb_frames: np.ndarray) -> np.ndarray:
    """Return one uint8 calibration image from one frame or a temporal stack."""

    frames = np.asarray(rgb_frames)
    if frames.ndim == 3:
        if frames.shape[2] != 3 or frames.dtype != np.uint8:
            raise ValueError("rgb_frames must be H x W x 3 or N x H x W x 3 uint8")
        return frames.copy()
    if (
        frames.ndim != 4
        or frames.shape[-1] != 3
        or frames.dtype != np.uint8
        or len(frames) == 0
    ):
        raise ValueError("rgb_frames must be H x W x 3 or N x H x W x 3 uint8")
    return np.median(frames, axis=0).astype(np.uint8)


def _outer_side_profile(
    red: np.ndarray,
    mask: np.ndarray,
    rows: np.ndarray,
    side: str,
) -> np.ndarray:
    """Return the row-wise high-red statistic in one outer silhouette band."""

    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    values = np.full(len(rows), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        columns = np.flatnonzero(mask[row])
        if len(columns) < 4:
            continue
        midpoint = int(round(0.5 * (columns[0] + columns[-1])))
        if side == "left":
            start, stop = int(columns[0]), midpoint
        else:
            start, stop = midpoint, int(columns[-1])
        values[index] = np.percentile(
            red[row, start : stop + 1],
            OUTER_SIDE_RED_PERCENTILE,
        )

    valid = np.isfinite(values)
    if np.count_nonzero(valid) < 16:
        raise RuntimeError(f"{side} side has too little valid silhouette support")
    return np.interp(
        np.arange(len(values), dtype=np.float64),
        np.flatnonzero(valid),
        values[valid],
    )


def _regular_five_peak_sequence(
    rows: np.ndarray,
    profile: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Find the highest-quality valid five-lobe sequence from distal."""

    span = len(rows)
    small_sigma = max(1.0, SMALL_SMOOTHING_SIGMA_SPAN_FRACTION * span)
    broad_sigma = max(6.0, BROAD_BACKGROUND_SIGMA_SPAN_FRACTION * span)
    smooth = gaussian_filter1d(profile, small_sigma)
    contrast = smooth - gaussian_filter1d(smooth, broad_sigma)
    peak_indices, properties = find_peaks(
        contrast,
        distance=max(4, round(MINIMUM_PEAK_DISTANCE_SPAN_FRACTION * span)),
        prominence=max(
            2.0,
            MINIMUM_PROMINENCE_RANGE_FRACTION * float(np.ptp(contrast)),
        ),
    )
    prominences = properties["prominences"]

    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for start_index in range(max(0, len(peak_indices) - LED_COUNT + 1)):
        selected_indices = peak_indices[start_index : start_index + LED_COUNT]
        if len(selected_indices) != LED_COUNT:
            continue

        peak_rows = rows[selected_indices].astype(np.float64)
        row_span = max(float(rows[-1] - rows[0]), 1.0)
        normalized_start = float((peak_rows[0] - rows[0]) / row_span)
        normalized_stop = float((peak_rows[-1] - rows[0]) / row_span)
        if (
            normalized_start < MINIMUM_SEQUENCE_START_FRACTION
            or normalized_stop > MAXIMUM_SEQUENCE_STOP_FRACTION
        ):
            continue

        spacings = np.diff(peak_rows)
        median_spacing = float(np.median(spacings))
        if not (
            MINIMUM_SPACING_SPAN_FRACTION * span
            <= median_spacing
            <= MAXIMUM_SPACING_SPAN_FRACTION * span
        ):
            continue
        if float(np.max(spacings) / np.min(spacings)) > MAXIMUM_SPACING_RATIO:
            continue

        indices = np.arange(LED_COUNT, dtype=np.float64)
        fitted = np.polyval(np.polyfit(indices, peak_rows, 2), indices)
        fit_rmse = float(np.sqrt(np.mean((peak_rows - fitted) ** 2)))
        if fit_rmse > MAXIMUM_QUADRATIC_RMSE_PITCH_FRACTION * median_spacing:
            continue

        selected_prominences = prominences[
            start_index : start_index + LED_COUNT
        ].astype(np.float64)
        spacing_cv = float(np.std(spacings) / np.mean(spacings))
        geometry_factor = np.exp(-3.0 * fit_rmse / median_spacing - 1.2 * spacing_cv)
        sequence_score = float(
            (np.median(selected_prominences) + 0.8 * np.min(selected_prominences))
            * geometry_factor
            * (1.0 - 0.25 * normalized_start)
        )
        candidates.append((sequence_score, selected_indices, selected_prominences))

    if not candidates:
        raise RuntimeError("no regular distal five-lobe Solaris sequence was found")
    score, selected_indices, selected_prominences = max(
        candidates,
        key=lambda candidate: candidate[0],
    )
    return (
        score,
        rows[selected_indices].astype(np.float64),
        contrast,
        selected_prominences,
    )


def _brightest_fraction_centroid(
    red: np.ndarray,
    mask: np.ndarray,
    side: str,
    center_row: float,
    half_height: int,
) -> np.ndarray:
    """Return the centroid of the brightest local silhouette pixels."""

    active_rows = np.flatnonzero(mask.any(axis=1))
    row_start = max(int(active_rows[0]), int(round(center_row)) - half_height)
    row_stop = min(int(active_rows[-1]), int(round(center_row)) + half_height)
    coordinates: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for row in range(row_start, row_stop + 1):
        columns = np.flatnonzero(mask[row])
        if len(columns) < 4:
            continue
        midpoint = int(round(0.5 * (columns[0] + columns[-1])))
        if side == "left":
            start, stop = int(columns[0]), midpoint
        else:
            start, stop = midpoint, int(columns[-1])
        selected = np.arange(start, stop + 1)
        coordinates.append(
            np.column_stack((selected, np.full(len(selected), row, dtype=np.int32)))
        )
        values.append(red[row, selected])
    if not values:
        raise RuntimeError("empty local LED lobe window")

    coordinate_array = np.concatenate(coordinates).astype(np.float64)
    value_array = np.concatenate(values).astype(np.float64)
    count = max(1, int(np.ceil(BRIGHTEST_CENTROID_FRACTION * len(value_array))))
    selected = np.argpartition(value_array, len(value_array) - count)[-count:]
    return np.mean(coordinate_array[selected], axis=0)


def _fit_homogeneous_line(points_xy: np.ndarray) -> np.ndarray:
    vx, vy, x0, y0 = np.asarray(
        cv2.fitLine(
            np.asarray(points_xy, dtype=np.float32),
            cv2.DIST_HUBER,
            0.0,
            0.01,
            0.01,
        )
    ).reshape(-1)
    line = np.asarray((vy, -vx, vx * y0 - vy * x0), dtype=np.float64)
    return line / np.linalg.norm(line[:2])


def localize_solaris_leds(
    rgb_frames: np.ndarray,
    *,
    reference_mask: np.ndarray | None = None,
) -> LedLocalizationResult:
    """Localize five fixed Solaris LED lobes from one frame or a frame stack.

    A supplied mask reuses a silhouette calibrated for the same fixed setup;
    otherwise the temporal-median image is segmented once.
    """

    calibration_rgb = temporal_median_rgb(rgb_frames)
    if reference_mask is None:
        # GrabCut initializes internal models from OpenCV's process-wide RNG.
        # Fixed-image calibration should reproduce exactly on every replay.
        cv2.setRNGSeed(0)
        mask = segment_fingertip(calibration_rgb).final_mask
    else:
        mask = np.asarray(reference_mask, dtype=bool)
        if mask.shape != calibration_rgb.shape[:2] or not np.any(mask):
            raise ValueError("reference_mask must be nonempty and match the image")
    profile_rows = np.flatnonzero(mask.any(axis=1))
    if len(profile_rows) < 32:
        raise RuntimeError("reference silhouette is too short")
    red = calibration_rgb[:, :, 0].astype(np.float64)

    side_candidates = []
    for side in ("left", "right"):
        raw_profile = _outer_side_profile(red, mask, profile_rows, side)
        try:
            score, peak_rows, contrast, prominences = _regular_five_peak_sequence(
                profile_rows,
                raw_profile,
            )
        except RuntimeError:
            continue
        side_candidates.append(
            (score, side, peak_rows, raw_profile, contrast, prominences)
        )
    if not side_candidates:
        raise RuntimeError("neither silhouette side contained a five-lobe sequence")

    score, side, peak_rows, raw_profile, contrast, prominences = max(
        side_candidates,
        key=lambda candidate: candidate[0],
    )
    median_pitch = float(np.median(np.diff(peak_rows)))
    half_height = max(
        3,
        int(LOCAL_CENTROID_HALF_HEIGHT_PITCH_FRACTION * median_pitch),
    )
    centers = np.vstack(
        [
            _brightest_fraction_centroid(red, mask, side, row, half_height)
            for row in peak_rows
        ]
    )
    order = np.argsort(centers[:, 1])
    centers = centers[order]
    peak_rows = peak_rows[order]
    prominences = prominences[order]
    return LedLocalizationResult(
        image_shape=calibration_rgb.shape[:2],
        led_centers_xy_px=centers,
        led_line=_fit_homogeneous_line(centers),
        selected_side=side,
        peak_rows_px=peak_rows,
        profile_rows_px=profile_rows.astype(np.float64),
        red_profile_dn=raw_profile,
        red_contrast_dn=contrast,
        peak_prominences_dn=prominences,
        sequence_score=score,
        reference_mask=mask,
    )


__all__ = [
    "LED_COUNT",
    "LedLocalizationResult",
    "localize_solaris_leds",
    "temporal_median_rgb",
]
