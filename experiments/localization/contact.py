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
_REFERENCE_IMAGE_WIDTH_PX = 640
_REFERENCE_IMAGE_HEIGHT_PX = 480
FEATURE_NOISE_FLOOR_DN = 0.75
CONTACT_Z_THRESHOLD = 4.0
_MINIMUM_RIGID_TRACK_CORRESPONDENCES = 4
_RANSAC_REPROJECTION_IN_LED_SPACINGS = 0.15
_MAXIMUM_INLIER_RESIDUAL_IN_LED_SPACINGS = 0.20
_MINIMUM_FRAME_SCALE = 0.80
_MAXIMUM_FRAME_SCALE = 1.25


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
    contact_detected: bool
    predicted_led_index: int
    position_mm: float | None
    top_two_margin: float

    def __post_init__(self) -> None:
        response = np.asarray(self.response, dtype=np.float64)
        if response.shape != (_LED_COUNT,) or not np.all(np.isfinite(response)):
            raise ValueError("response must be a finite length-five vector")
        if not isinstance(self.contact_detected, bool):
            raise ValueError("contact_detected must be a bool")
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
    scale = rgb.shape[0] / _REFERENCE_IMAGE_HEIGHT_PX
    small = cv2.GaussianBlur(red, (0, 0), scale * _SMALL_GAUSSIAN_SIGMA_PX)
    background = cv2.GaussianBlur(red, (0, 0), scale * _BROAD_GAUSSIAN_SIGMA_PX)
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
    profile = _smooth_profile(
        high_pass[:, x_start:x_stop].mean(axis=1),
        1.4 * height / _REFERENCE_IMAGE_HEIGHT_PX,
    )
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
            1.5 * width / _REFERENCE_IMAGE_WIDTH_PX,
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
    kernel_size = max(
        3,
        round(
            3
            * min(
                width / _REFERENCE_IMAGE_WIDTH_PX,
                height / _REFERENCE_IMAGE_HEIGHT_PX,
            )
        ),
    )
    if kernel_size % 2 == 0:
        kernel_size += 1
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((kernel_size, kernel_size), np.uint8),
    )
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


def _geometry_from_landmarks(
    landmarks: np.ndarray,
    image_shape: tuple[int, ...],
) -> LedArrayGeometry:
    landmarks = np.asarray(landmarks, dtype=np.float64)
    if landmarks.shape != (_LED_COUNT, 2) or not np.all(np.isfinite(landmarks)):
        raise RuntimeError("tracked LED landmarks are not a finite 5 x 2 array")
    spacings = np.linalg.norm(np.diff(landmarks, axis=0), axis=1)
    if np.any(spacings <= np.finfo(np.float64).eps):
        raise RuntimeError("tracked LED landmarks contain coincident points")
    spacing_px = float(np.median(spacings))
    return LedArrayGeometry(
        landmarks_xy_px=landmarks,
        roi_polygons_xy_px=_roi_polygons(landmarks, image_shape),
        median_spacing_px=spacing_px,
    )


def detect_led_array(rgb_frames: np.ndarray) -> LedArrayGeometry:
    """Detect the common five-LED array from one or more fixed-camera frames."""

    frames = _rgb_frames(rgb_frames)
    median_rgb = (
        frames[0] if len(frames) == 1 else np.median(frames, axis=0).astype(np.uint8)
    )
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
    return _geometry_from_landmarks(landmarks, median_rgb.shape)


def constrain_led_array_motion(
    previous_landmarks_xy_px: np.ndarray,
    candidate_landmarks_xy_px: np.ndarray,
    valid_correspondences: np.ndarray,
    median_spacing_px: float,
) -> np.ndarray:
    """Fit one robust similarity motion and transform the full LED array."""

    previous = np.asarray(previous_landmarks_xy_px, dtype=np.float64)
    candidates = np.asarray(candidate_landmarks_xy_px, dtype=np.float64)
    valid = np.asarray(valid_correspondences, dtype=bool)
    if previous.shape != (_LED_COUNT, 2) or not np.all(np.isfinite(previous)):
        raise ValueError("previous_landmarks_xy_px must be a finite 5 x 2 array")
    if candidates.shape != (_LED_COUNT, 2):
        raise ValueError("candidate_landmarks_xy_px must be a 5 x 2 array")
    if valid.shape != (_LED_COUNT,):
        raise ValueError("valid_correspondences must be a length-five vector")
    if not np.isfinite(median_spacing_px) or median_spacing_px <= 0.0:
        raise ValueError("median_spacing_px must be finite and positive")

    valid &= np.all(np.isfinite(candidates), axis=1)
    if np.count_nonzero(valid) < _MINIMUM_RIGID_TRACK_CORRESPONDENCES:
        raise RuntimeError(
            "rigid LED tracking requires at least four valid correspondences"
        )

    transform, ransac_inliers = cv2.estimateAffinePartial2D(
        previous[valid],
        candidates[valid],
        method=cv2.RANSAC,
        ransacReprojThreshold=(
            _RANSAC_REPROJECTION_IN_LED_SPACINGS * median_spacing_px
        ),
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if transform is None or ransac_inliers is None:
        raise RuntimeError("could not fit one similarity motion to the LED array")
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (2, 3) or not np.all(np.isfinite(transform)):
        raise RuntimeError("LED similarity transform is not finite")

    scale = float(np.hypot(transform[0, 0], transform[1, 0]))
    if not _MINIMUM_FRAME_SCALE <= scale <= _MAXIMUM_FRAME_SCALE:
        raise RuntimeError(f"LED similarity scale change is unreasonable: {scale:.3f}")

    inliers = np.asarray(ransac_inliers, dtype=bool).reshape(-1)
    if np.count_nonzero(inliers) < _MINIMUM_RIGID_TRACK_CORRESPONDENCES:
        raise RuntimeError("LED similarity fit retained fewer than four inliers")

    homogeneous = np.column_stack((previous, np.ones(_LED_COUNT)))
    constrained = homogeneous @ transform.T
    if not np.all(np.isfinite(constrained)):
        raise RuntimeError("constrained LED landmarks are not finite")

    residuals = np.linalg.norm(
        constrained[valid][inliers] - candidates[valid][inliers],
        axis=1,
    )
    maximum_residual_px = (
        _MAXIMUM_INLIER_RESIDUAL_IN_LED_SPACINGS * median_spacing_px
    )
    if float(np.max(residuals)) > maximum_residual_px:
        raise RuntimeError(
            "LED similarity inlier residual is too large: "
            f"{np.max(residuals):.2f} px"
        )
    return constrained


def track_led_array(
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
    previous_geometry: LedArrayGeometry,
) -> LedArrayGeometry:
    """Update the five image landmarks after a camera/fingertip pose change.

    Pyramidal Lucas-Kanade flow proposes point-wise correspondences. Valid
    forward-backward tracks robustly fit one partial-affine similarity motion,
    which is then applied to the previous rigid five-landmark array.
    """

    previous = np.asarray(previous_rgb)
    current = np.asarray(current_rgb)
    if (
        previous.shape != current.shape
        or previous.ndim != 3
        or previous.shape[2] != 3
        or previous.dtype != np.uint8
        or current.dtype != np.uint8
    ):
        raise ValueError("previous_rgb and current_rgb must be equal-shape RGB uint8")

    previous_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    previous_points = np.asarray(
        previous_geometry.landmarks_xy_px,
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    window_size = max(31, round(1.4 * previous_geometry.median_spacing_px))
    if window_size % 2 == 0:
        window_size += 1
    flow_parameters = {
        "winSize": (window_size, window_size),
        "maxLevel": 3,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }
    current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_points,
        None,
        **flow_parameters,
    )
    if current_points is None or forward_status is None:
        raise RuntimeError("forward LED optical flow returned no landmarks")
    candidates = current_points.reshape(-1, 2)
    forward_valid = np.asarray(forward_status, dtype=bool).reshape(-1)
    forward_valid &= np.all(np.isfinite(candidates), axis=1)
    if np.count_nonzero(forward_valid) < _MINIMUM_RIGID_TRACK_CORRESPONDENCES:
        raise RuntimeError("forward LED optical flow retained fewer than four points")

    forward_indices = np.flatnonzero(forward_valid)
    backward_subset, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        current_points[forward_indices],
        None,
        **flow_parameters,
    )
    if backward_subset is None or backward_status is None:
        raise RuntimeError("backward LED optical flow returned no landmarks")

    backward_points = np.full((_LED_COUNT, 2), np.nan, dtype=np.float64)
    backward_points[forward_indices] = backward_subset.reshape(-1, 2)
    backward_valid = np.zeros(_LED_COUNT, dtype=bool)
    backward_valid[forward_indices] = np.asarray(
        backward_status,
        dtype=bool,
    ).reshape(-1)
    backward_valid &= np.all(np.isfinite(backward_points), axis=1)
    forward_backward_error = np.linalg.norm(
        backward_points - previous_points.reshape(-1, 2),
        axis=1,
    )
    maximum_error_px = max(1.5, 0.12 * previous_geometry.median_spacing_px)
    valid = (
        forward_valid
        & backward_valid
        & (forward_backward_error <= maximum_error_px)
    )
    constrained = constrain_led_array_motion(
        previous_geometry.landmarks_xy_px,
        candidates,
        valid,
        previous_geometry.median_spacing_px,
    )
    return _geometry_from_landmarks(constrained, current.shape)


def reanchor_led_array(
    rgb: np.ndarray,
    previous_geometry: LedArrayGeometry,
) -> LedArrayGeometry:
    """Re-anchor a rigid tracked array to absolute red-image landmarks."""

    detected = detect_led_array(rgb)
    constrained = constrain_led_array_motion(
        previous_geometry.landmarks_xy_px,
        detected.landmarks_xy_px,
        np.ones(_LED_COUNT, dtype=bool),
        previous_geometry.median_spacing_px,
    )
    return _geometry_from_landmarks(constrained, np.asarray(rgb).shape)


def unloaded_baseline_statistics(
    feature_samples: np.ndarray,
    *,
    noise_floor_dn: float = FEATURE_NOISE_FLOOR_DN,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-LED temporal medians and robust unloaded noise scales."""

    samples = np.asarray(feature_samples, dtype=np.float64)
    if (
        samples.ndim != 2
        or samples.shape[1:] != (_LED_COUNT,)
        or not len(samples)
        or not np.all(np.isfinite(samples))
    ):
        raise ValueError("feature_samples must be a finite nonempty N x 5 array")
    if not np.isfinite(noise_floor_dn) or noise_floor_dn <= 0.0:
        raise ValueError("noise_floor_dn must be finite and positive")

    baseline = np.median(samples, axis=0)
    median_absolute_deviation = np.median(np.abs(samples - baseline), axis=0)
    noise_sigma = np.maximum(1.4826 * median_absolute_deviation, noise_floor_dn)
    return baseline, noise_sigma


def brightest_red_features(
    rgb: np.ndarray,
    geometry: LedArrayGeometry,
) -> np.ndarray:
    """Measure the brightest 10% red-channel mean in each LED-centered ROI."""

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be an H x W x 3 uint8 array")
    values = []
    for polygon in geometry.roi_polygons_xy_px:
        rounded_polygon = np.rint(polygon).astype(np.int32)
        x, y, width, height = cv2.boundingRect(rounded_polygon)
        local_polygon = rounded_polygon - np.asarray((x, y), dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, local_polygon, 255)
        red_crop = image[y : y + height, x : x + width, 0]
        pixels = red_crop[mask > 0]
        if pixels.size == 0:
            raise RuntimeError("an LED ROI contains no pixels")
        count = max(1, int(np.ceil(_TOP_FRACTION * pixels.size)))
        values.append(float(np.mean(np.partition(pixels, pixels.size - count)[-count:])))
    return np.asarray(values, dtype=np.float64)


def estimate_contact_position(
    features: np.ndarray,
    unloaded_baseline: np.ndarray,
    unloaded_noise_sigma: np.ndarray,
    led_positions_mm: np.ndarray,
) -> ContactEstimate:
    """Estimate noise-gated contact from baseline-relative red response.

    Contact is active when at least one positive response reaches four robust
    unloaded-noise standard deviations. The continuous estimate is the
    positive-response-weighted centroid of the physical LED positions and is
    unavailable while contact is inactive.
    """

    current = np.asarray(features, dtype=np.float64)
    baseline = np.asarray(unloaded_baseline, dtype=np.float64)
    noise_sigma = np.asarray(unloaded_noise_sigma, dtype=np.float64)
    positions = np.asarray(led_positions_mm, dtype=np.float64)
    for name, values in (
        ("features", current),
        ("unloaded_baseline", baseline),
        ("unloaded_noise_sigma", noise_sigma),
        ("led_positions_mm", positions),
    ):
        if values.shape != (_LED_COUNT,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a finite length-five vector")
    position_steps = np.diff(positions)
    if not (np.all(position_steps > 0.0) or np.all(position_steps < 0.0)):
        raise ValueError("led_positions_mm must be strictly ordered")
    if np.any(noise_sigma < 0.0):
        raise ValueError("unloaded_noise_sigma must be nonnegative")

    response = current - baseline
    standardized_positive = np.maximum(response, 0.0) / np.maximum(
        noise_sigma,
        FEATURE_NOISE_FLOOR_DN,
    )
    contact_detected = bool(np.max(standardized_positive) >= CONTACT_Z_THRESHOLD)
    predicted = int(np.argmax(response))
    sorted_response = np.sort(response)
    margin = float(sorted_response[-1] - sorted_response[-2])
    positive = np.maximum(response, 0.0)
    total_positive = float(np.sum(positive))
    position = None
    if contact_detected and total_positive > np.finfo(np.float64).eps:
        position = float(np.dot(positive, positions) / total_positive)
    return ContactEstimate(
        response=response,
        contact_detected=contact_detected,
        predicted_led_index=predicted,
        position_mm=position,
        top_two_margin=margin,
    )


def contact_image_point(
    estimate: ContactEstimate,
    geometry: LedArrayGeometry,
) -> np.ndarray | None:
    """Return the response-weighted contact marker in image pixel coordinates."""

    if not estimate.contact_detected:
        return None
    positive = np.maximum(estimate.response, 0.0)
    total_positive = float(np.sum(positive))
    if total_positive <= np.finfo(np.float64).eps:
        return None
    roi_centers = np.mean(geometry.roi_polygons_xy_px, axis=1)
    return np.sum(positive[:, None] * roi_centers, axis=0) / total_positive


__all__ = [
    "CONTACT_Z_THRESHOLD",
    "FEATURE_NOISE_FLOOR_DN",
    "ContactEstimate",
    "LedArrayGeometry",
    "brightest_red_features",
    "constrain_led_array_motion",
    "contact_image_point",
    "detect_led_array",
    "estimate_contact_position",
    "reanchor_led_array",
    "track_led_array",
    "unloaded_baseline_statistics",
]
