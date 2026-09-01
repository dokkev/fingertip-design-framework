"""Learning-free LED-array detection and contact-position estimation."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np


_LED_COUNT = 5
_TOP_FRACTION = 0.10
_ROI_WIDTH_IN_LED_SPACINGS = 1.70
_ROI_HEIGHT_IN_LED_SPACINGS = 0.76
_ROI_INWARD_SHIFT_IN_LED_SPACINGS = 0.35
_SMALL_GAUSSIAN_SIGMA_PX = 1.2
_BROAD_GAUSSIAN_SIGMA_PX = 14.0
_SEARCH_X_FRACTION = (0.50, 0.70)
_SEARCH_Y_FRACTION = (0.28, 0.62)
_COMPONENT_SEARCH_Y_FRACTION = (0.28, 0.70)
_COMPONENT_PERCENTILE = 85.0


@dataclass(frozen=True)
class LedArrayGeometry:
    """Five ordered image landmarks and spacing-scaled response regions."""

    landmarks_xy_px: np.ndarray
    roi_polygons_xy_px: np.ndarray
    median_spacing_px: float

    def __post_init__(self) -> None:
        landmarks = np.asarray(self.landmarks_xy_px, dtype=np.float64)
        polygons = np.asarray(self.roi_polygons_xy_px, dtype=np.float64)
        if landmarks.shape != (_LED_COUNT, 2) or not np.all(np.isfinite(landmarks)):
            raise ValueError("landmarks_xy_px must be a finite 5 x 2 array")
        if polygons.shape != (_LED_COUNT, 4, 2) or not np.all(np.isfinite(polygons)):
            raise ValueError("roi_polygons_xy_px must be a finite 5 x 4 x 2 array")
        if not np.isfinite(self.median_spacing_px) or self.median_spacing_px <= 0.0:
            raise ValueError("median_spacing_px must be finite and positive")
        landmarks = landmarks.copy()
        polygons = polygons.copy()
        landmarks.setflags(write=False)
        polygons.setflags(write=False)
        object.__setattr__(self, "landmarks_xy_px", landmarks)
        object.__setattr__(self, "roi_polygons_xy_px", polygons)


@dataclass(frozen=True)
class ContactEstimate:
    """Baseline-relative LED response and its response-weighted position."""

    response: np.ndarray
    predicted_led_index: int
    position_mm: float | None
    top_two_margin: float

    def __post_init__(self) -> None:
        response = np.asarray(self.response, dtype=np.float64)
        if response.shape != (_LED_COUNT,) or not np.all(np.isfinite(response)):
            raise ValueError("response must be a finite length-five vector")
        if not 0 <= self.predicted_led_index < _LED_COUNT:
            raise ValueError("predicted_led_index is outside the LED array")
        if self.position_mm is not None and not np.isfinite(self.position_mm):
            raise ValueError("position_mm must be finite when available")
        if not np.isfinite(self.top_two_margin):
            raise ValueError("top_two_margin must be finite")
        response = response.copy()
        response.setflags(write=False)
        object.__setattr__(self, "response", response)


def _rgb_frames(rgb_frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(rgb_frames)
    if frames.ndim == 3:
        frames = frames[None, ...]
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("rgb_frames must be N x H x W x 3 uint8")
    if not len(frames):
        raise ValueError("rgb_frames must be nonempty")
    return frames


def _red_high_pass(rgb: np.ndarray) -> np.ndarray:
    red = rgb[:, :, 0].astype(np.float32)
    small = cv2.GaussianBlur(red, (0, 0), _SMALL_GAUSSIAN_SIGMA_PX)
    background = cv2.GaussianBlur(red, (0, 0), _BROAD_GAUSSIAN_SIGMA_PX)
    return np.maximum(small - background, 0.0)


def _smooth_profile(values: np.ndarray, sigma_px: float) -> np.ndarray:
    return cv2.GaussianBlur(
        np.asarray(values, dtype=np.float32)[:, None],
        (1, 0),
        sigma_px,
    ).ravel()


def _candidate_peaks(
    profile: np.ndarray,
    start: int,
    stop: int,
    minimum_distance: int,
) -> np.ndarray:
    local_span = float(np.ptp(profile[start:stop]))
    minimum_prominence = 0.018 * local_span
    candidates = []
    for index in range(start + 1, stop - 1):
        if profile[index] < profile[index - 1] or profile[index] <= profile[index + 1]:
            continue
        left = max(start, index - minimum_distance)
        right = min(stop, index + minimum_distance + 1)
        local_floor = max(
            float(np.min(profile[left : index + 1])),
            float(np.min(profile[index:right])),
        )
        if profile[index] - local_floor >= minimum_prominence:
            candidates.append(index)

    kept: list[int] = []
    for index in sorted(candidates, key=lambda item: float(profile[item]), reverse=True):
        if all(abs(index - selected) >= minimum_distance for selected in kept):
            kept.append(index)
    return np.asarray(sorted(kept), dtype=np.int32)


def _landmarks_from_median(median_rgb: np.ndarray) -> np.ndarray:
    high_pass = _red_high_pass(median_rgb)
    height, width = high_pass.shape
    x_start = round(_SEARCH_X_FRACTION[0] * width)
    x_stop = round(_SEARCH_X_FRACTION[1] * width)
    y_start = round(_SEARCH_Y_FRACTION[0] * height)
    y_stop = round(_SEARCH_Y_FRACTION[1] * height)
    profile = _smooth_profile(high_pass[:, x_start:x_stop].mean(axis=1), 1.4)
    peak_y = _candidate_peaks(
        profile,
        y_start,
        y_stop,
        minimum_distance=round(0.018 * height),
    )
    if peak_y.size < _LED_COUNT:
        raise RuntimeError(f"red detector found only {peak_y.size} candidate peaks")

    peak_x = []
    strengths = []
    row_half_width = round(0.012 * height)
    for y_coordinate in peak_y:
        column_profile = _smooth_profile(
            high_pass[
                max(0, y_coordinate - row_half_width) : min(
                    height, y_coordinate + row_half_width + 1
                ),
                x_start:x_stop,
            ].sum(axis=0),
            1.5,
        )
        peak_x.append(x_start + int(np.argmax(column_profile)))
        strengths.append(float(profile[y_coordinate]))

    peak_x_array = np.asarray(peak_x, dtype=np.float64)
    strength_array = np.asarray(strengths, dtype=np.float64)
    best: tuple[float, tuple[int, ...]] | None = None
    for selection in itertools.combinations(range(peak_y.size), _LED_COUNT):
        selected_y = peak_y[list(selection)].astype(np.float64)
        selected_x = peak_x_array[list(selection)]
        spacing = np.diff(selected_y)
        mean_spacing = float(np.mean(spacing))
        if not 0.025 * height <= mean_spacing <= 0.065 * height:
            continue
        if float(np.min(spacing)) < 0.014 * height:
            continue
        spacing_cv = float(np.std(spacing) / mean_spacing)
        line = np.polyfit(selected_y, selected_x, 1)
        line_rmse = float(
            np.sqrt(np.mean((selected_x - np.polyval(line, selected_y)) ** 2))
        )
        score = (
            float(np.mean(strength_array[list(selection)]))
            / (float(np.max(strength_array)) + 1.0e-6)
            - 2.0 * spacing_cv
            - line_rmse / (0.5 * mean_spacing)
        )
        if best is None or score > best[0]:
            best = (score, selection)
    if best is None:
        raise RuntimeError("red detector could not select five regular ordered peaks")

    selection = list(best[1])
    landmarks = np.column_stack((peak_x_array[selection], peak_y[selection])).astype(
        np.float64
    )
    spacings = np.linalg.norm(np.diff(landmarks, axis=0), axis=1)
    spacing_cv = float(np.std(spacings) / np.mean(spacings))
    if not np.all(np.diff(landmarks[:, 1]) > 0):
        raise RuntimeError("LED landmarks are not ordered along the image array")
    if spacing_cv > 0.30:
        raise RuntimeError(f"detected LED spacing is not regular: CV={spacing_cv:.3f}")
    return landmarks


def _component_landmarks(median_rgb: np.ndarray) -> np.ndarray:
    """Fallback for oblique views whose projected LED spacing is nonlinear."""

    high_pass = _red_high_pass(median_rgb)
    height, width = high_pass.shape
    x_start = round(_SEARCH_X_FRACTION[0] * width)
    x_stop = round(_SEARCH_X_FRACTION[1] * width)
    y_start = round(_COMPONENT_SEARCH_Y_FRACTION[0] * height)
    y_stop = round(_COMPONENT_SEARCH_Y_FRACTION[1] * height)
    crop = high_pass[y_start:y_stop, x_start:x_stop]
    positive = crop[crop > 0.0]
    if not positive.size:
        raise RuntimeError("red component detector found no positive response")
    threshold = float(np.percentile(positive, _COMPONENT_PERCENTILE))
    mask = np.asarray(crop >= threshold, dtype=np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)

    minimum_area = max(8, round(2.0e-5 * height * width))
    candidates = []
    areas = []
    for component in range(1, component_count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        centroid = centroids[component] + np.asarray((x_start, y_start))
        candidates.append(centroid)
        areas.append(area)
    if len(candidates) < _LED_COUNT:
        raise RuntimeError(
            "red component detector found only "
            f"{len(candidates)} candidate regions"
        )

    points = np.asarray(candidates, dtype=np.float64)
    component_areas = np.asarray(areas, dtype=np.float64)
    best: tuple[float, np.ndarray] | None = None
    for selection in itertools.combinations(range(len(points)), _LED_COUNT):
        selected = points[list(selection)]
        order = np.argsort(selected[:, 1])
        selected = selected[order]
        spacings = np.diff(selected[:, 1])
        mean_spacing = float(np.mean(spacings))
        if not 0.020 * height <= mean_spacing <= 0.10 * height:
            continue
        if float(np.min(spacings)) < 0.012 * height:
            continue
        line = np.polyfit(selected[:, 1], selected[:, 0], 1)
        line_rmse = float(
            np.sqrt(
                np.mean(
                    (selected[:, 0] - np.polyval(line, selected[:, 1])) ** 2
                )
            )
        )
        if line_rmse > 0.35 * mean_spacing:
            continue
        selected_areas = component_areas[np.asarray(selection)][order]
        score = (
            float(np.mean(np.sqrt(selected_areas)))
            / (float(np.max(np.sqrt(component_areas))) + 1.0e-6)
            - line_rmse / (0.35 * mean_spacing)
        )
        if best is None or score > best[0]:
            best = (score, selected)
    if best is None:
        raise RuntimeError("red component detector could not select five collinear regions")
    return best[1]


def _roi_polygons(landmarks: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    spacing_px = float(np.median(np.linalg.norm(np.diff(landmarks, axis=0), axis=1)))
    array_axis = landmarks[-1] - landmarks[0]
    array_axis /= np.linalg.norm(array_axis)
    outward_axis = np.asarray((array_axis[1], -array_axis[0]))
    if outward_axis[0] < 0.0:
        outward_axis *= -1.0

    half_width = 0.5 * _ROI_WIDTH_IN_LED_SPACINGS * spacing_px
    half_height = 0.5 * _ROI_HEIGHT_IN_LED_SPACINGS * spacing_px
    centers = landmarks - _ROI_INWARD_SHIFT_IN_LED_SPACINGS * spacing_px * outward_axis
    polygons = np.asarray(
        [
            (
                center - half_width * outward_axis - half_height * array_axis,
                center + half_width * outward_axis - half_height * array_axis,
                center + half_width * outward_axis + half_height * array_axis,
                center - half_width * outward_axis + half_height * array_axis,
            )
            for center in centers
        ],
        dtype=np.float64,
    )
    height, width = image_shape[:2]
    if (
        np.min(polygons[:, :, 0]) < 0.0
        or np.max(polygons[:, :, 0]) >= width
        or np.min(polygons[:, :, 1]) < 0.0
        or np.max(polygons[:, :, 1]) >= height
    ):
        raise RuntimeError("a detected LED ROI extends outside the image")
    return polygons


def detect_led_array(rgb_frames: np.ndarray) -> LedArrayGeometry:
    """Detect the common five-LED array from one or more fixed-camera frames."""

    frames = _rgb_frames(rgb_frames)
    median_rgb = np.median(frames, axis=0).astype(np.uint8)
    try:
        landmarks = _landmarks_from_median(median_rgb)
    except RuntimeError as profile_error:
        try:
            landmarks = _component_landmarks(median_rgb)
        except RuntimeError as component_error:
            raise RuntimeError(
                f"profile detector failed ({profile_error}); "
                f"component detector failed ({component_error})"
            ) from component_error
    polygons = _roi_polygons(landmarks, median_rgb.shape)
    spacing_px = float(np.median(np.linalg.norm(np.diff(landmarks, axis=0), axis=1)))
    return LedArrayGeometry(
        landmarks_xy_px=landmarks,
        roi_polygons_xy_px=polygons,
        median_spacing_px=spacing_px,
    )


def brightest_red_features(
    rgb: np.ndarray,
    geometry: LedArrayGeometry,
) -> np.ndarray:
    """Measure the brightest 10% red-channel mean in each LED-centered ROI."""

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be an H x W x 3 uint8 array")
    red = image[:, :, 0].astype(np.float64)
    values = []
    for polygon in geometry.roi_polygons_xy_px:
        mask = np.zeros(red.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
        pixels = red[mask > 0]
        if pixels.size == 0:
            raise RuntimeError("an LED ROI contains no pixels")
        count = max(1, int(np.ceil(_TOP_FRACTION * pixels.size)))
        values.append(float(np.mean(np.partition(pixels, pixels.size - count)[-count:])))
    return np.asarray(values, dtype=np.float64)


def estimate_contact_position(
    features: np.ndarray,
    unloaded_baseline: np.ndarray,
    led_positions_mm: np.ndarray,
) -> ContactEstimate:
    """Estimate contact from positive baseline-relative red-response changes.

    The discrete estimate is the maximum response channel. The continuous
    estimate is the positive-response-weighted centroid of the physical LED
    positions; it is unavailable when no channel increased from baseline.
    """

    current = np.asarray(features, dtype=np.float64)
    baseline = np.asarray(unloaded_baseline, dtype=np.float64)
    positions = np.asarray(led_positions_mm, dtype=np.float64)
    for name, values in (
        ("features", current),
        ("unloaded_baseline", baseline),
        ("led_positions_mm", positions),
    ):
        if values.shape != (_LED_COUNT,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a finite length-five vector")
    position_steps = np.diff(positions)
    if not (np.all(position_steps > 0.0) or np.all(position_steps < 0.0)):
        raise ValueError("led_positions_mm must be strictly ordered")

    response = current - baseline
    predicted = int(np.argmax(response))
    sorted_response = np.sort(response)
    margin = float(sorted_response[-1] - sorted_response[-2])
    positive = np.maximum(response, 0.0)
    total_positive = float(np.sum(positive))
    position = (
        None
        if total_positive <= np.finfo(np.float64).eps
        else float(np.dot(positive, positions) / total_positive)
    )
    return ContactEstimate(
        response=response,
        predicted_led_index=predicted,
        position_mm=position,
        top_two_margin=margin,
    )


__all__ = [
    "ContactEstimate",
    "LedArrayGeometry",
    "brightest_red_features",
    "detect_led_array",
    "estimate_contact_position",
]
