"""Offline Solaris LED localization from its rigid periodic optical pattern."""

from __future__ import annotations

import cv2
import numpy as np

from lumo.fingertip.layout import LED_CENTERS_Y_MM, TOTAL_Y_BOUNDS_MM

from .fingertip_segmentation import segment_fingertip
from .led_localization_common import LedLocalizationResult


# Empirical calibration choices.
SIDE_FIT_END_EXCLUSION_FRACTION = 0.10
RED_BACKGROUND_SIGMA_IN_LED_PITCHES = 0.75
PERIODIC_SAMPLE_HALF_WIDTH_IN_LED_PITCHES = 0.25

# Numerical discretization only.
LED_LINE_CANDIDATE_COUNT = 101
LONGITUDINAL_SCALE_CANDIDATE_COUNT = 257
PERIODIC_PROFILE_SAMPLE_COUNT = 257
_VANISHING_POINT_INFINITY_EPSILON = 1.0e-10
_DISTAL_ORIENTATIONS = ("minimum_longitudinal", "maximum_longitudinal")


def solaris_physical_led_layout() -> np.ndarray:
    """Return the five distal-to-proximal LED distances in millimetres."""

    distal_y_mm = max(TOTAL_Y_BOUNDS_MM)
    positions_mm = distal_y_mm - np.sort(np.asarray(LED_CENTERS_Y_MM))[::-1]
    if not np.allclose(np.diff(positions_mm), 11.0):
        raise RuntimeError("the fixed Solaris LED layout must have 11 mm pitch")
    return positions_mm


def _normalized_line(values: np.ndarray) -> np.ndarray:
    line = np.asarray(values, dtype=np.float64)
    if line.shape != (3,) or not np.all(np.isfinite(line)):
        raise RuntimeError("invalid image line")
    scale = float(np.linalg.norm(line[:2]))
    if scale <= 0.0:
        raise RuntimeError("invalid image line")
    return line / scale


def _line_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    point_h = np.cross(first, second)
    if abs(point_h[2]) <= np.finfo(np.float64).eps * np.linalg.norm(point_h[:2]):
        raise RuntimeError("finite image lines do not intersect")
    return point_h[:2] / point_h[2]


def _fit_line(points_xy: np.ndarray) -> np.ndarray:
    vx, vy, x0, y0 = np.asarray(
        cv2.fitLine(
            np.asarray(points_xy, dtype=np.float32),
            cv2.DIST_HUBER,
            0.0,
            0.01,
            0.01,
        )
    ).reshape(-1)
    return _normalized_line(np.asarray((vy, -vx, vx * y0 - vy * x0)))


def _cross_section_line(
    centroid_xy: np.ndarray,
    longitudinal_axis_xy: np.ndarray,
    longitudinal_coordinate: float,
) -> np.ndarray:
    offset = float(np.dot(longitudinal_axis_xy, centroid_xy) + longitudinal_coordinate)
    return _normalized_line(
        np.asarray(
            (
                longitudinal_axis_xy[0],
                longitudinal_axis_xy[1],
                -offset,
            )
        )
    )


def _side_geometry(
    mask: np.ndarray,
    distal_orientation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_coordinates, x_coordinates = np.nonzero(mask)
    pixels = np.column_stack((x_coordinates, y_coordinates)).astype(np.float64)
    centroid = np.mean(pixels, axis=0)
    centered = pixels - centroid
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
    longitudinal_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    dominant = int(np.argmax(np.abs(longitudinal_axis)))
    if longitudinal_axis[dominant] < 0.0:
        longitudinal_axis *= -1.0
    transverse_axis = np.asarray((-longitudinal_axis[1], longitudinal_axis[0]))
    longitudinal = centered @ longitudinal_axis
    transverse = centered @ transverse_axis
    minimum_s = float(np.min(longitudinal))
    maximum_s = float(np.max(longitudinal))
    span = maximum_s - minimum_s
    if span <= 0.0:
        raise RuntimeError("the reference silhouette has no longitudinal span")

    bin_count = max(2, int(np.ceil(span)) + 1)
    indices = np.floor((longitudinal - minimum_s) * (bin_count - 1) / span).astype(
        np.int32
    )
    indices = np.clip(indices, 0, bin_count - 1)
    minimum_t = np.full(bin_count, np.inf)
    maximum_t = np.full(bin_count, -np.inf)
    np.minimum.at(minimum_t, indices, transverse)
    np.maximum.at(maximum_t, indices, transverse)
    valid = np.isfinite(minimum_t) & np.isfinite(maximum_t)
    sample_s = np.linspace(minimum_s, maximum_s, bin_count)[valid]
    first_trace = (
        centroid
        + sample_s[:, None] * longitudinal_axis
        + minimum_t[valid, None] * transverse_axis
    )
    second_trace = (
        centroid
        + sample_s[:, None] * longitudinal_axis
        + maximum_t[valid, None] * transverse_axis
    )
    fit_minimum = minimum_s + SIDE_FIT_END_EXCLUSION_FRACTION * span
    fit_maximum = maximum_s - SIDE_FIT_END_EXCLUSION_FRACTION * span
    fit = (sample_s >= fit_minimum) & (sample_s <= fit_maximum)
    if np.count_nonzero(fit) < 2:
        raise RuntimeError("the middle silhouette span has too few side samples")
    first_side = _fit_line(first_trace[fit])
    second_side = _fit_line(second_trace[fit])

    distal_s = minimum_s if distal_orientation == "minimum_longitudinal" else maximum_s
    distal_limit = _cross_section_line(centroid, longitudinal_axis, distal_s)
    proximal_direction = (
        longitudinal_axis
        if distal_orientation == "minimum_longitudinal"
        else -longitudinal_axis
    )

    vanishing_point_h = np.cross(first_side, second_side).astype(np.float64)
    direction_norm = float(np.linalg.norm(vanishing_point_h[:2]))
    if direction_norm <= np.finfo(np.float64).eps:
        raise RuntimeError("the fitted side lines are coincident")
    vanishing_point_h /= direction_norm
    if abs(vanishing_point_h[2]) < _VANISHING_POINT_INFINITY_EPSILON:
        vanishing_point_h[2] = 0.0

    distal_midpoint = 0.5 * (
        _line_intersection(first_side, distal_limit)
        + _line_intersection(second_side, distal_limit)
    )
    projective_direction = (
        vanishing_point_h[:2] - vanishing_point_h[2] * distal_midpoint
    )
    if np.dot(projective_direction, proximal_direction) < 0.0:
        vanishing_point_h *= -1.0
    return first_side, second_side, distal_limit, vanishing_point_h


def _candidate_line(
    first_side: np.ndarray,
    second_side: np.ndarray,
    distal_limit: np.ndarray,
    vanishing_point_h: np.ndarray,
    alpha: float,
) -> np.ndarray:
    first_point = _line_intersection(first_side, distal_limit)
    second_point = _line_intersection(second_side, distal_limit)
    distal_point = (1.0 - alpha) * first_point + alpha * second_point
    return _normalized_line(
        np.cross(np.asarray((*distal_point, 1.0)), vanishing_point_h)
    )


def _project_from_distal(
    image_line: np.ndarray,
    distal_limit: np.ndarray,
    vanishing_point_h: np.ndarray,
    distances_mm: np.ndarray,
    initial_scale_px_per_mm: float,
) -> np.ndarray:
    """Project physical distances from the distal limit along one image line."""

    distances = np.asarray(distances_mm, dtype=np.float64)
    if distances.ndim != 1 or not len(distances) or not np.all(np.isfinite(distances)):
        raise ValueError("distances_mm must be a nonempty finite vector")
    if not np.isfinite(initial_scale_px_per_mm) or initial_scale_px_per_mm <= 0.0:
        raise ValueError("initial_scale_px_per_mm must be positive and finite")

    return _project_grid_from_distal(
        image_line,
        distal_limit,
        vanishing_point_h,
        distances,
        np.asarray((initial_scale_px_per_mm,)),
    )[0]


def _project_grid_from_distal(
    image_line: np.ndarray,
    distal_limit: np.ndarray,
    vanishing_point_h: np.ndarray,
    distances_mm: np.ndarray,
    scales_px_per_mm: np.ndarray,
) -> np.ndarray:
    """Project one distance vector for several distal image scales."""

    distal_point = _line_intersection(
        _normalized_line(image_line),
        _normalized_line(distal_limit),
    )
    vanishing = np.asarray(vanishing_point_h, dtype=np.float64)
    projective_direction = vanishing[:2] - vanishing[2] * distal_point
    direction_norm = float(np.linalg.norm(projective_direction))
    if direction_norm <= np.finfo(np.float64).eps:
        raise RuntimeError("distal anchor coincides with the longitudinal horizon")
    coefficient_per_mm = np.asarray(scales_px_per_mm, dtype=np.float64) / direction_norm
    points_h = (
        np.asarray((*distal_point, 1.0))[None, None, :]
        + coefficient_per_mm[:, None, None]
        * distances_mm[None, :, None]
        * vanishing[None, None, :]
    )
    if np.any(np.abs(points_h[:, :, 2]) <= np.finfo(np.float64).eps):
        raise RuntimeError("projective LED array crosses the camera horizon")
    return points_h[:, :, :2] / points_h[:, :, 2, None]


def _sample_periodic_profiles(
    red: np.ndarray,
    line: np.ndarray,
    distal_limit: np.ndarray,
    vanishing_point_h: np.ndarray,
    scales_px_per_mm: np.ndarray,
    distances_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    maps = _project_grid_from_distal(
        line,
        distal_limit,
        vanishing_point_h,
        distances_mm,
        scales_px_per_mm,
    )
    profile_sum = np.zeros((len(scales_px_per_mm), len(distances_mm)), dtype=np.float64)
    valid_count = np.zeros_like(profile_sum, dtype=np.int16)
    height, width = red.shape
    for transverse_offset_px in (-1.0, 0.0, 1.0):
        offset_maps = maps + transverse_offset_px * line[:2]
        map_x = offset_maps[:, :, 0].astype(np.float32)
        map_y = offset_maps[:, :, 1].astype(np.float32)
        values = cv2.remap(
            red,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = (
            (offset_maps[:, :, 0] >= 0.0)
            & (offset_maps[:, :, 0] <= width - 1)
            & (offset_maps[:, :, 1] >= 0.0)
            & (offset_maps[:, :, 1] <= height - 1)
        )
        profile_sum += np.where(valid, values, 0.0)
        valid_count += valid
    profiles = np.divide(
        profile_sum,
        valid_count,
        out=np.zeros_like(profile_sum),
        where=valid_count > 0,
    )
    return profiles, valid_count > 0


def _periodic_responses(
    distances_mm: np.ndarray,
    profiles: np.ndarray,
    valid: np.ndarray,
    led_positions_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return signed contrast, center/midpoint responses, and match scores."""

    distances = np.asarray(distances_mm, dtype=np.float64)
    samples = np.asarray(profiles, dtype=np.float64)
    sample_valid = np.asarray(valid, dtype=bool)
    centers_mm = np.asarray(led_positions_mm, dtype=np.float64)
    if samples.ndim != 2 or sample_valid.shape != samples.shape:
        raise ValueError("profiles and valid must be matching two-dimensional arrays")
    if len(distances) < 3 or samples.shape[1] != len(distances):
        raise ValueError("distances_mm must match the profile sample axis")
    if centers_mm.shape != (5,) or not np.all(np.diff(centers_mm) > 0.0):
        raise ValueError("led_positions_mm must contain five increasing positions")

    pitch_mm = float(np.mean(np.diff(centers_mm)))
    if not np.allclose(np.diff(centers_mm), pitch_mm):
        raise ValueError("led_positions_mm must be equally spaced")
    half_window_mm = PERIODIC_SAMPLE_HALF_WIDTH_IN_LED_PITCHES * pitch_mm
    sigma_samples = (
        RED_BACKGROUND_SIGMA_IN_LED_PITCHES
        * pitch_mm
        / float(distances[1] - distances[0])
    )
    radius = max(1, min(len(distances) - 1, int(np.ceil(4.0 * sigma_samples))))
    kernel = cv2.getGaussianKernel(2 * radius + 1, sigma_samples).reshape(1, -1)
    background = cv2.filter2D(
        samples.astype(np.float32),
        -1,
        kernel,
        borderType=cv2.BORDER_REFLECT101,
    ).astype(np.float64)
    contrast = samples - background

    center_responses, midpoint_responses, response_valid = _local_periodic_responses(
        distances,
        contrast,
        sample_valid,
        centers_mm,
        half_window_mm,
    )
    scores = np.mean(center_responses, axis=1) - np.mean(midpoint_responses, axis=1)
    scores[~response_valid] = -np.inf
    return contrast, center_responses, midpoint_responses, scores


def _local_periodic_responses(
    distances_mm: np.ndarray,
    contrast: np.ndarray,
    valid: np.ndarray,
    led_positions_mm: np.ndarray,
    half_window_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average signed contrast in equal center and inter-LED windows."""

    centers_mm = np.asarray(led_positions_mm, dtype=np.float64)
    midpoints_mm = 0.5 * (centers_mm[:-1] + centers_mm[1:])
    center_responses = np.zeros((len(contrast), 5), dtype=np.float64)
    midpoint_responses = np.zeros((len(contrast), 4), dtype=np.float64)
    response_valid = np.ones(len(contrast), dtype=bool)
    for response, positions_mm in (
        (center_responses, centers_mm),
        (midpoint_responses, midpoints_mm),
    ):
        for index, position_mm in enumerate(positions_mm):
            window = np.abs(distances_mm - position_mm) <= half_window_mm
            count = np.count_nonzero(valid[:, window], axis=1)
            response_valid &= count > 0
            response[:, index] = np.divide(
                np.sum(
                    np.where(valid[:, window], contrast[:, window], 0.0),
                    axis=1,
                ),
                count,
                out=np.zeros(len(contrast), dtype=np.float64),
                where=count > 0,
            )
    return center_responses, midpoint_responses, response_valid


def _scale_candidates(
    first_side: np.ndarray,
    second_side: np.ndarray,
    distal_limit: np.ndarray,
    distal_search_distance_mm: float,
) -> tuple[np.ndarray, float]:
    side_span_px = float(
        np.linalg.norm(
            _line_intersection(first_side, distal_limit)
            - _line_intersection(second_side, distal_limit)
        )
    )
    minimum_scale = 1.0 / float(solaris_physical_led_layout()[0])
    maximum_scale = side_span_px / distal_search_distance_mm
    if maximum_scale <= minimum_scale:
        raise RuntimeError(
            "the distal silhouette is too small to resolve the LED pitch"
        )
    return (
        np.linspace(
            minimum_scale,
            maximum_scale,
            LONGITUDINAL_SCALE_CANDIDATE_COUNT,
        ),
        side_span_px,
    )


def localize_solaris_leds(
    unloaded_rgb: np.ndarray,
    *,
    distal_orientation: str = "minimum_longitudinal",
) -> LedLocalizationResult:
    """Localize the rigid Solaris array from its full 11 mm periodic pattern."""

    image = np.asarray(unloaded_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("unloaded_rgb must be an H x W x 3 uint8 array")
    if distal_orientation not in _DISTAL_ORIENTATIONS:
        raise ValueError(
            "distal_orientation must be one of " + ", ".join(_DISTAL_ORIENTATIONS)
        )

    mask = segment_fingertip(image).final_mask
    first_side, second_side, distal_limit, vanishing_point_h = _side_geometry(
        mask,
        distal_orientation,
    )
    led_positions_mm = solaris_physical_led_layout()
    pitch_mm = float(np.mean(np.diff(led_positions_mm)))
    half_window_mm = PERIODIC_SAMPLE_HALF_WIDTH_IN_LED_PITCHES * pitch_mm
    distances_mm = np.linspace(
        led_positions_mm[0],
        led_positions_mm[-1],
        PERIODIC_PROFILE_SAMPLE_COUNT,
    )
    scales, distal_side_span_px = _scale_candidates(
        first_side,
        second_side,
        distal_limit,
        float(led_positions_mm[1] + half_window_mm),
    )
    red = image[:, :, 0].astype(np.float32)

    best: (
        tuple[
            float,
            float,
            float,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
        | None
    ) = None
    candidate_alphas = (
        np.arange(LED_LINE_CANDIDATE_COUNT, dtype=np.float64) + 0.5
    ) / LED_LINE_CANDIDATE_COUNT
    for alpha in candidate_alphas:
        line = _candidate_line(
            first_side,
            second_side,
            distal_limit,
            vanishing_point_h,
            float(alpha),
        )
        profiles, valid = _sample_periodic_profiles(
            red,
            line,
            distal_limit,
            vanishing_point_h,
            scales,
            distances_mm,
        )
        _, center_responses, midpoint_responses, scores = _periodic_responses(
            distances_mm,
            profiles,
            valid,
            led_positions_mm,
        )

        distal_support_points = _project_grid_from_distal(
            line,
            distal_limit,
            vanishing_point_h,
            np.asarray((led_positions_mm[1] + half_window_mm,)),
            scales,
        )[:, 0]
        distal_point = _line_intersection(line, distal_limit)
        inside_distal_support = (
            np.linalg.norm(distal_support_points - distal_point, axis=1)
            <= distal_side_span_px
        )
        scores[~inside_distal_support] = -np.inf
        scale_index = int(np.argmax(scores))
        score = float(scores[scale_index])
        if not np.isfinite(score):
            continue
        if best is None or score > best[0]:
            best = (
                score,
                float(alpha),
                float(scales[scale_index]),
                line,
                center_responses[scale_index],
                midpoint_responses[scale_index],
            )
    if best is None:
        raise RuntimeError("Solaris periodic LED search produced no valid candidate")
    (
        line_score,
        led_line_alpha,
        scale_px_per_mm,
        led_line,
        center_responses,
        midpoint_responses,
    ) = best

    first_distal = _line_intersection(first_side, distal_limit)
    second_distal = _line_intersection(second_side, distal_limit)
    led_distal = _line_intersection(led_line, distal_limit)
    if np.linalg.norm(led_distal - first_distal) <= np.linalg.norm(
        led_distal - second_distal
    ):
        dorsal_line, palmar_line = first_side, second_side
    else:
        dorsal_line, palmar_line = second_side, first_side
        led_line_alpha = 1.0 - led_line_alpha

    led_centers = _project_from_distal(
        led_line,
        distal_limit,
        vanishing_point_h,
        led_positions_mm,
        scale_px_per_mm,
    )
    return LedLocalizationResult(
        image_shape=image.shape[:2],
        led_centers_xy_px=led_centers,
        longitudinal_positions_mm=led_positions_mm,
        led_line=led_line,
        dorsal_line=dorsal_line,
        palmar_line=palmar_line,
        distal_limit=distal_limit,
        vanishing_point_h=vanishing_point_h,
        led_center_responses=center_responses,
        inter_led_responses=midpoint_responses,
        reference_mask=mask,
        led_line_alpha=led_line_alpha,
        longitudinal_scale_px_per_mm=scale_px_per_mm,
        line_score=line_score,
    )


__all__ = [
    "LED_LINE_CANDIDATE_COUNT",
    "LONGITUDINAL_SCALE_CANDIDATE_COUNT",
    "PERIODIC_PROFILE_SAMPLE_COUNT",
    "PERIODIC_SAMPLE_HALF_WIDTH_IN_LED_PITCHES",
    "RED_BACKGROUND_SIGMA_IN_LED_PITCHES",
    "SIDE_FIT_END_EXCLUSION_FRACTION",
    "localize_solaris_leds",
    "solaris_physical_led_layout",
]
