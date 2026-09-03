"""One-image geometry calibration for the fixed physical fingertip experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lumo.fingertip.layout import ACTIVE_Y_BOUNDS_MM, LED_CENTERS_Y_MM

from .fingertip_segmentation import segment_fingertip


SIDE_FIT_END_EXCLUSION_FRACTION = 0.10
LED_BACKGROUND_SIGMA_IN_SPACINGS = 0.75
LED_REFINEMENT_HALF_WINDOW_IN_SPACINGS = 0.25

LED_LINE_CANDIDATE_COUNT = 101
PROFILE_SAMPLE_COUNT = 512
CANONICAL_HEIGHT = 256
CANONICAL_WIDTH = 128

_DISTAL_ORIENTATIONS = ("minimum_longitudinal", "maximum_longitudinal")
_VANISHING_POINT_INFINITY_DISTANCE_IN_FINGER_SPANS = 1000.0
_FORMAT_VERSION = 1


@dataclass(frozen=True)
class FixedFingerCalibration:
    """Fixed image geometry and remap arrays derived from one unloaded frame.

    Lines use normalized homogeneous image coefficients ``[a, b, c]`` for
    ``a*x + b*y + c = 0``. Longitudinal-limit lines bound the complete smooth
    silhouette support. The calibration is valid only for the fixed fingertip
    and camera pose represented by its unloaded reference image.
    ``led_line_alpha`` is zero at the dorsal line and one at the palmar line.
    """

    image_shape: tuple[int, int]
    dorsal_line: np.ndarray
    palmar_line: np.ndarray
    led_line: np.ndarray
    vanishing_point_h: np.ndarray
    distal_longitudinal_limit: np.ndarray
    proximal_longitudinal_limit: np.ndarray
    led_centers_xy_px: np.ndarray
    led_longitudinal_fractions: np.ndarray
    canonical_map_x: np.ndarray
    canonical_map_y: np.ndarray
    reference_mask: np.ndarray
    distal_orientation: str
    led_line_alpha: float
    led_line_score: float

    def __post_init__(self) -> None:
        height, width = self.image_shape
        if (
            not isinstance(height, int)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or isinstance(width, bool)
            or min(height, width) < 2
        ):
            raise ValueError("image_shape must contain two integer dimensions >= 2")
        if self.distal_orientation not in _DISTAL_ORIENTATIONS:
            raise ValueError(
                "distal_orientation must be one of "
                + ", ".join(_DISTAL_ORIENTATIONS)
            )
        for name, value in (
            ("led_line_alpha", self.led_line_alpha),
            ("led_line_score", self.led_line_score),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.led_line_alpha <= 1.0:
            raise ValueError("led_line_alpha must be in [0, 1]")
        if self.led_line_score < 0.0:
            raise ValueError("led_line_score must be nonnegative")

        for name in (
            "dorsal_line",
            "palmar_line",
            "led_line",
            "distal_longitudinal_limit",
            "proximal_longitudinal_limit",
        ):
            line = _normalized_line(getattr(self, name), name)
            line.setflags(write=False)
            object.__setattr__(self, name, line)

        vanishing_point = np.asarray(self.vanishing_point_h, dtype=np.float64)
        if vanishing_point.shape != (3,) or not np.all(np.isfinite(vanishing_point)):
            raise ValueError("vanishing_point_h must be a finite length-three vector")
        if np.linalg.norm(vanishing_point) <= np.finfo(np.float64).eps:
            raise ValueError("vanishing_point_h must be nonzero")
        vanishing_point = vanishing_point.copy()
        vanishing_point.setflags(write=False)
        object.__setattr__(self, "vanishing_point_h", vanishing_point)

        centers = np.asarray(self.led_centers_xy_px, dtype=np.float64)
        fractions = np.asarray(self.led_longitudinal_fractions, dtype=np.float64)
        if centers.shape != (5, 2) or not np.all(np.isfinite(centers)):
            raise ValueError("led_centers_xy_px must be a finite 5 x 2 array")
        if (
            fractions.shape != (5,)
            or not np.all(np.isfinite(fractions))
            or not np.all(np.diff(fractions) > 0.0)
            or fractions[0] < 0.0
            or fractions[-1] > 1.0
        ):
            raise ValueError(
                "led_longitudinal_fractions must be five ordered values in [0, 1]"
            )
        centers = centers.copy()
        fractions = fractions.copy()
        centers.setflags(write=False)
        fractions.setflags(write=False)
        object.__setattr__(self, "led_centers_xy_px", centers)
        object.__setattr__(self, "led_longitudinal_fractions", fractions)

        map_x = np.asarray(self.canonical_map_x, dtype=np.float32)
        map_y = np.asarray(self.canonical_map_y, dtype=np.float32)
        if (
            map_x.ndim != 2
            or map_y.shape != map_x.shape
            or min(map_x.shape) < 2
            or not np.all(np.isfinite(map_x))
            or not np.all(np.isfinite(map_y))
        ):
            raise ValueError("canonical maps must be equal finite 2-D arrays")
        mask = np.asarray(self.reference_mask, dtype=bool)
        if mask.shape != self.image_shape or not np.any(mask):
            raise ValueError("reference_mask must be nonempty and match image_shape")
        map_x = map_x.copy()
        map_y = map_y.copy()
        mask = mask.copy()
        map_x.setflags(write=False)
        map_y.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "canonical_map_x", map_x)
        object.__setattr__(self, "canonical_map_y", map_y)
        object.__setattr__(self, "reference_mask", mask)


def _normalized_line(values: np.ndarray, name: str = "line") -> np.ndarray:
    line = np.asarray(values, dtype=np.float64)
    if line.shape != (3,) or not np.all(np.isfinite(line)):
        raise ValueError(f"{name} must be a finite length-three vector")
    scale = float(np.linalg.norm(line[:2]))
    if scale <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} has no finite image-line direction")
    return line.copy() / scale


def _fit_homogeneous_line(points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2:
        raise ValueError("line fit requires at least two image points")
    vx, vy, x0, y0 = np.asarray(
        cv2.fitLine(points, cv2.DIST_HUBER, 0.0, 0.01, 0.01)
    ).reshape(-1)
    return _normalized_line(
        np.asarray((vy, -vx, vx * y0 - vy * x0), dtype=np.float64)
    )


def _line_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    point_h = np.cross(first, second)
    if abs(point_h[2]) <= np.finfo(np.float64).eps * np.linalg.norm(point_h[:2]):
        raise RuntimeError("finite calibration lines do not intersect in the image plane")
    return point_h[:2] / point_h[2]


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


def _projective_vanishing_point(
    first_line: np.ndarray,
    second_line: np.ndarray,
    finger_span_px: float,
) -> np.ndarray:
    point_h = np.cross(first_line, second_line).astype(np.float64)
    direction_norm = float(np.linalg.norm(point_h[:2]))
    if direction_norm <= np.finfo(np.float64).eps:
        raise RuntimeError("the fitted side lines are coincident")
    point_h /= direction_norm
    if (
        abs(point_h[2]) * finger_span_px
        < 1.0 / _VANISHING_POINT_INFINITY_DISTANCE_IN_FINGER_SPANS
    ):
        point_h[2] = 0.0
    elif point_h[2] < 0.0:
        point_h *= -1.0
    return point_h


def _line_through_projective_point(
    point_xy: np.ndarray,
    vanishing_point_h: np.ndarray,
) -> np.ndarray:
    point_h = np.asarray((point_xy[0], point_xy[1], 1.0), dtype=np.float64)
    return _normalized_line(np.cross(point_h, vanishing_point_h))


def project_longitudinal_positions(
    image_line: np.ndarray,
    distal_limit: np.ndarray,
    proximal_limit: np.ndarray,
    vanishing_point_h: np.ndarray,
    longitudinal_fractions: np.ndarray,
) -> np.ndarray:
    """Project physical distal-to-proximal fractions onto one image line."""

    line = _normalized_line(image_line)
    distal = _line_intersection(line, _normalized_line(distal_limit))
    proximal = _line_intersection(line, _normalized_line(proximal_limit))
    fractions = np.asarray(longitudinal_fractions, dtype=np.float64)
    if (
        fractions.ndim != 1
        or not len(fractions)
        or not np.all(np.isfinite(fractions))
        or np.any(fractions < 0.0)
        or np.any(fractions > 1.0)
    ):
        raise ValueError("longitudinal_fractions must be a finite vector in [0, 1]")

    segment = proximal - distal
    segment_norm_squared = float(np.dot(segment, segment))
    if segment_norm_squared <= np.finfo(np.float64).eps:
        raise RuntimeError("distal and proximal projected limits coincide")
    vanishing_point = np.asarray(vanishing_point_h, dtype=np.float64)
    if vanishing_point.shape != (3,) or not np.all(np.isfinite(vanishing_point)):
        raise ValueError("vanishing_point_h must be a finite length-three vector")
    if abs(vanishing_point[2]) <= np.finfo(np.float64).eps:
        image_fractions = fractions
    else:
        vanishing_xy = vanishing_point[:2] / vanishing_point[2]
        vanishing_fraction = float(
            np.dot(vanishing_xy - distal, segment) / segment_norm_squared
        )
        denominator = vanishing_fraction - 1.0 + fractions
        if np.any(
            np.abs(denominator)
            <= np.finfo(np.float64).eps * max(1.0, abs(vanishing_fraction))
        ):
            raise RuntimeError("projective longitudinal mapping crosses its horizon")
        image_fractions = vanishing_fraction * fractions / denominator
    return distal + image_fractions[:, None] * segment


def _side_traces_from_mask(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    y_coordinates, x_coordinates = np.nonzero(mask)
    pixels = np.column_stack((x_coordinates, y_coordinates)).astype(np.float64)
    centroid = np.mean(pixels, axis=0)
    centered = pixels - centroid
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
    longitudinal_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    dominant_component = int(np.argmax(np.abs(longitudinal_axis)))
    if longitudinal_axis[dominant_component] < 0.0:
        longitudinal_axis *= -1.0
    transverse_axis = np.asarray(
        (-longitudinal_axis[1], longitudinal_axis[0]),
        dtype=np.float64,
    )
    longitudinal = centered @ longitudinal_axis
    transverse = centered @ transverse_axis
    minimum = float(np.min(longitudinal))
    maximum = float(np.max(longitudinal))
    span = maximum - minimum
    bin_count = max(2, int(np.ceil(span)) + 1)
    indices = np.floor((longitudinal - minimum) * (bin_count - 1) / span).astype(
        np.int32
    )
    indices = np.clip(indices, 0, bin_count - 1)
    minimum_transverse = np.full(bin_count, np.inf, dtype=np.float64)
    maximum_transverse = np.full(bin_count, -np.inf, dtype=np.float64)
    np.minimum.at(minimum_transverse, indices, transverse)
    np.maximum.at(maximum_transverse, indices, transverse)
    valid = np.isfinite(minimum_transverse) & np.isfinite(maximum_transverse)
    sample_longitudinal = np.linspace(minimum, maximum, bin_count)[valid]
    first_trace = (
        centroid
        + sample_longitudinal[:, None] * longitudinal_axis
        + minimum_transverse[valid, None] * transverse_axis
    )
    second_trace = (
        centroid
        + sample_longitudinal[:, None] * longitudinal_axis
        + maximum_transverse[valid, None] * transverse_axis
    )
    fit_minimum = minimum + SIDE_FIT_END_EXCLUSION_FRACTION * span
    fit_maximum = maximum - SIDE_FIT_END_EXCLUSION_FRACTION * span
    fit = (sample_longitudinal >= fit_minimum) & (
        sample_longitudinal <= fit_maximum
    )
    if np.count_nonzero(fit) < 2:
        raise RuntimeError("the middle silhouette span has too few side samples")
    return (
        _fit_homogeneous_line(first_trace[fit]),
        _fit_homogeneous_line(second_trace[fit]),
        np.column_stack((centroid, longitudinal_axis)),
        minimum,
        maximum,
    )


def _candidate_line(
    first_side: np.ndarray,
    second_side: np.ndarray,
    reference_limit: np.ndarray,
    vanishing_point_h: np.ndarray,
    alpha: float,
) -> np.ndarray:
    first_reference = _line_intersection(first_side, reference_limit)
    second_reference = _line_intersection(second_side, reference_limit)
    reference_point = (1.0 - alpha) * first_reference + alpha * second_reference
    return _line_through_projective_point(reference_point, vanishing_point_h)


def _sample_line(
    channel: np.ndarray,
    mask: np.ndarray,
    image_line: np.ndarray,
    distal_limit: np.ndarray,
    proximal_limit: np.ndarray,
    vanishing_point_h: np.ndarray,
    longitudinal_fractions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = project_longitudinal_positions(
        image_line,
        distal_limit,
        proximal_limit,
        vanishing_point_h,
        longitudinal_fractions,
    )
    map_x = points[:, 0].astype(np.float32).reshape(-1, 1)
    map_y = points[:, 1].astype(np.float32).reshape(-1, 1)
    values = cv2.remap(
        channel,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).reshape(-1)
    valid = cv2.remap(
        mask.astype(np.uint8),
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).reshape(-1).astype(bool)
    return values.astype(np.float64), valid


def _positive_red_contrast(
    profile: np.ndarray,
    valid: np.ndarray,
    smoothing_sigma_samples: float,
) -> np.ndarray:
    if np.count_nonzero(valid) < 2:
        return np.zeros_like(profile)
    samples = np.arange(len(profile))
    filled = np.interp(samples, samples[valid], profile[valid])
    background = cv2.GaussianBlur(
        filled.astype(np.float32)[:, None],
        (1, 0),
        smoothing_sigma_samples,
    ).reshape(-1)
    contrast = np.maximum(filled - background, 0.0)
    contrast[~valid] = 0.0
    return contrast


def _candidate_profile(
    red: np.ndarray,
    mask: np.ndarray,
    first_side: np.ndarray,
    second_side: np.ndarray,
    reference_limit: np.ndarray,
    distal_limit: np.ndarray,
    proximal_limit: np.ndarray,
    vanishing_point_h: np.ndarray,
    alpha: float,
    longitudinal_fractions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half_grid_step = 0.5 / (LED_LINE_CANDIDATE_COUNT - 1)
    strip_alphas = np.clip(
        alpha + np.asarray((-half_grid_step, 0.0, half_grid_step)),
        0.0,
        1.0,
    )
    values = []
    validities = []
    for strip_alpha in strip_alphas:
        strip_line = _candidate_line(
            first_side,
            second_side,
            reference_limit,
            vanishing_point_h,
            float(strip_alpha),
        )
        strip_values, strip_valid = _sample_line(
            red,
            mask,
            strip_line,
            distal_limit,
            proximal_limit,
            vanishing_point_h,
            longitudinal_fractions,
        )
        values.append(strip_values)
        validities.append(strip_valid)
    value_array = np.asarray(values)
    valid_array = np.asarray(validities)
    counts = np.count_nonzero(valid_array, axis=0)
    profile = np.divide(
        np.sum(np.where(valid_array, value_array, 0.0), axis=0),
        counts,
        out=np.zeros(len(longitudinal_fractions), dtype=np.float64),
        where=counts > 0,
    )
    valid = counts > 0
    spacing_samples = (
        (len(longitudinal_fractions) - 1)
        * (LED_CENTERS_Y_MM[1] - LED_CENTERS_Y_MM[0])
        / (ACTIVE_Y_BOUNDS_MM[1] - ACTIVE_Y_BOUNDS_MM[0])
    )
    contrast = _positive_red_contrast(
        profile,
        valid,
        LED_BACKGROUND_SIGMA_IN_SPACINGS * spacing_samples,
    )
    center_line = _candidate_line(
        first_side,
        second_side,
        reference_limit,
        vanishing_point_h,
        alpha,
    )
    return center_line, contrast, valid


def _refine_led_fractions(
    predicted: np.ndarray,
    sampled_fractions: np.ndarray,
    contrast: np.ndarray,
) -> np.ndarray:
    spacing_fraction = float(predicted[1] - predicted[0])
    half_window = LED_REFINEMENT_HALF_WINDOW_IN_SPACINGS * spacing_fraction
    refined = predicted.copy()
    for led_index, center in enumerate(predicted):
        in_window = np.flatnonzero(np.abs(sampled_fractions - center) <= half_window)
        local_maxima = [
            index
            for index in in_window
            if 0 < index < len(contrast) - 1
            and contrast[index] > contrast[index - 1]
            and contrast[index] > contrast[index + 1]
        ]
        if len(local_maxima) == 1 and contrast[local_maxima[0]] > 0.0:
            refined[led_index] = sampled_fractions[local_maxima[0]]
    return refined


def _points_inside_mask(points_xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rounded = np.rint(points_xy).astype(np.int64)
    height, width = mask.shape
    inside_image = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    inside = np.zeros(len(rounded), dtype=bool)
    valid = np.flatnonzero(inside_image)
    inside[valid] = mask[rounded[valid, 1], rounded[valid, 0]]
    return inside


def _canonical_maps(
    dorsal_line: np.ndarray,
    palmar_line: np.ndarray,
    distal_limit: np.ndarray,
    proximal_limit: np.ndarray,
    vanishing_point_h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    longitudinal = np.linspace(0.0, 1.0, CANONICAL_HEIGHT)
    dorsal = project_longitudinal_positions(
        dorsal_line,
        distal_limit,
        proximal_limit,
        vanishing_point_h,
        longitudinal,
    )
    palmar = project_longitudinal_positions(
        palmar_line,
        distal_limit,
        proximal_limit,
        vanishing_point_h,
        longitudinal,
    )
    transverse = np.linspace(0.0, 1.0, CANONICAL_WIDTH)
    points = (
        dorsal[:, None, :] * (1.0 - transverse[None, :, None])
        + palmar[:, None, :] * transverse[None, :, None]
    )
    return points[:, :, 0].astype(np.float32), points[:, :, 1].astype(np.float32)


def calibrate_fixed_finger(
    unloaded_rgb: np.ndarray,
    *,
    distal_orientation: str = "minimum_longitudinal",
) -> FixedFingerCalibration:
    """Calibrate one fixed fingertip/camera geometry from an unloaded RGB image."""

    image = np.asarray(unloaded_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("unloaded_rgb must be an H x W x 3 uint8 array")
    if distal_orientation not in _DISTAL_ORIENTATIONS:
        raise ValueError(
            "distal_orientation must be one of " + ", ".join(_DISTAL_ORIENTATIONS)
        )

    segmentation = segment_fingertip(image)
    mask = segmentation.final_mask
    first_side, second_side, frame, minimum_s, maximum_s = _side_traces_from_mask(
        mask
    )
    centroid = frame[:, 0]
    longitudinal_axis = frame[:, 1]
    span = maximum_s - minimum_s
    first_limit = _cross_section_line(centroid, longitudinal_axis, minimum_s)
    second_limit = _cross_section_line(centroid, longitudinal_axis, maximum_s)
    if distal_orientation == "minimum_longitudinal":
        distal_limit, proximal_limit = first_limit, second_limit
    else:
        distal_limit, proximal_limit = second_limit, first_limit
    reference_limit = _cross_section_line(
        centroid,
        longitudinal_axis,
        0.5 * (minimum_s + maximum_s),
    )
    vanishing_point_h = _projective_vanishing_point(
        first_side,
        second_side,
        span,
    )

    finger_length_mm = ACTIVE_Y_BOUNDS_MM[1] - ACTIVE_Y_BOUNDS_MM[0]
    led_span_mm = LED_CENTERS_Y_MM[-1] - LED_CENTERS_Y_MM[0]
    edge_fraction = 0.5 * (finger_length_mm - led_span_mm) / finger_length_mm
    predicted_fractions = np.linspace(edge_fraction, 1.0 - edge_fraction, 5)

    red = image[:, :, 0].astype(np.float32)
    sampled_fractions = np.linspace(0.0, 1.0, PROFILE_SAMPLE_COUNT)
    best: tuple[float, float, np.ndarray, np.ndarray] | None = None
    candidate_alphas = (
        np.arange(LED_LINE_CANDIDATE_COUNT, dtype=np.float64) + 0.5
    ) / LED_LINE_CANDIDATE_COUNT
    for alpha in candidate_alphas:
        candidate_line, contrast, valid = _candidate_profile(
            red,
            mask,
            first_side,
            second_side,
            reference_limit,
            distal_limit,
            proximal_limit,
            vanishing_point_h,
            float(alpha),
            sampled_fractions,
        )
        predicted_centers = project_longitudinal_positions(
            candidate_line,
            distal_limit,
            proximal_limit,
            vanishing_point_h,
            predicted_fractions,
        )
        if not np.all(_points_inside_mask(predicted_centers, mask)):
            continue
        score = float(np.sum(contrast[valid]) / len(sampled_fractions))
        if best is None or score > best[0]:
            best = (score, float(alpha), candidate_line, contrast)
    if best is None:
        raise RuntimeError("LED-line search produced no candidate")
    led_line_score, led_line_alpha, led_line, led_contrast = best

    first_reference = _line_intersection(first_side, reference_limit)
    second_reference = _line_intersection(second_side, reference_limit)
    led_reference = _line_intersection(led_line, reference_limit)
    if np.linalg.norm(led_reference - first_reference) <= np.linalg.norm(
        led_reference - second_reference
    ):
        dorsal_line, palmar_line = first_side, second_side
    else:
        dorsal_line, palmar_line = second_side, first_side
        led_line_alpha = 1.0 - led_line_alpha

    led_fractions = _refine_led_fractions(
        predicted_fractions,
        sampled_fractions,
        led_contrast,
    )
    led_centers = project_longitudinal_positions(
        led_line,
        distal_limit,
        proximal_limit,
        vanishing_point_h,
        led_fractions,
    )
    refined_inside = _points_inside_mask(led_centers, mask)
    if not np.all(refined_inside):
        led_fractions[~refined_inside] = predicted_fractions[~refined_inside]
        led_centers = project_longitudinal_positions(
            led_line,
            distal_limit,
            proximal_limit,
            vanishing_point_h,
            led_fractions,
        )
    map_x, map_y = _canonical_maps(
        dorsal_line,
        palmar_line,
        distal_limit,
        proximal_limit,
        vanishing_point_h,
    )
    return FixedFingerCalibration(
        image_shape=image.shape[:2],
        dorsal_line=dorsal_line,
        palmar_line=palmar_line,
        led_line=led_line,
        vanishing_point_h=vanishing_point_h,
        distal_longitudinal_limit=distal_limit,
        proximal_longitudinal_limit=proximal_limit,
        led_centers_xy_px=led_centers,
        led_longitudinal_fractions=led_fractions,
        canonical_map_x=map_x,
        canonical_map_y=map_y,
        reference_mask=mask,
        distal_orientation=distal_orientation,
        led_line_alpha=led_line_alpha,
        led_line_score=led_line_score,
    )


def warp_with_fixed_finger_calibration(
    rgb: np.ndarray,
    calibration: FixedFingerCalibration,
) -> np.ndarray:
    """Remap one fixed-camera experiment frame without new geometry inference."""

    image = np.asarray(rgb)
    if image.shape != (*calibration.image_shape, 3) or image.dtype != np.uint8:
        raise ValueError("rgb must match the calibrated H x W x 3 uint8 image shape")
    return cv2.remap(
        image,
        calibration.canonical_map_x,
        calibration.canonical_map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def save_fixed_finger_calibration(
    path: str | Path,
    calibration: FixedFingerCalibration,
) -> None:
    """Save one fixed calibration as compressed arrays without pickle."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        format_version=np.asarray(_FORMAT_VERSION, dtype=np.int64),
        image_shape=np.asarray(calibration.image_shape, dtype=np.int64),
        dorsal_line=calibration.dorsal_line,
        palmar_line=calibration.palmar_line,
        led_line=calibration.led_line,
        vanishing_point_h=calibration.vanishing_point_h,
        distal_longitudinal_limit=calibration.distal_longitudinal_limit,
        proximal_longitudinal_limit=calibration.proximal_longitudinal_limit,
        led_centers_xy_px=calibration.led_centers_xy_px,
        led_longitudinal_fractions=calibration.led_longitudinal_fractions,
        canonical_map_x=calibration.canonical_map_x,
        canonical_map_y=calibration.canonical_map_y,
        reference_mask=calibration.reference_mask.astype(np.uint8),
        distal_orientation=np.asarray(calibration.distal_orientation),
        led_line_alpha=np.asarray(calibration.led_line_alpha),
        led_line_score=np.asarray(calibration.led_line_score),
    )


def load_fixed_finger_calibration(path: str | Path) -> FixedFingerCalibration:
    """Load a fixed calibration without enabling NumPy pickle support."""

    with np.load(Path(path), allow_pickle=False) as data:
        version = int(data["format_version"].item())
        if version != _FORMAT_VERSION:
            raise ValueError(f"unsupported fixed calibration format version: {version}")
        return FixedFingerCalibration(
            image_shape=tuple(int(value) for value in data["image_shape"].tolist()),
            dorsal_line=data["dorsal_line"],
            palmar_line=data["palmar_line"],
            led_line=data["led_line"],
            vanishing_point_h=data["vanishing_point_h"],
            distal_longitudinal_limit=data["distal_longitudinal_limit"],
            proximal_longitudinal_limit=data["proximal_longitudinal_limit"],
            led_centers_xy_px=data["led_centers_xy_px"],
            led_longitudinal_fractions=data["led_longitudinal_fractions"],
            canonical_map_x=data["canonical_map_x"],
            canonical_map_y=data["canonical_map_y"],
            reference_mask=data["reference_mask"].astype(bool),
            distal_orientation=str(data["distal_orientation"].item()),
            led_line_alpha=float(data["led_line_alpha"].item()),
            led_line_score=float(data["led_line_score"].item()),
        )


__all__ = [
    "CANONICAL_HEIGHT",
    "CANONICAL_WIDTH",
    "FixedFingerCalibration",
    "LED_BACKGROUND_SIGMA_IN_SPACINGS",
    "LED_LINE_CANDIDATE_COUNT",
    "LED_REFINEMENT_HALF_WINDOW_IN_SPACINGS",
    "PROFILE_SAMPLE_COUNT",
    "SIDE_FIT_END_EXCLUSION_FRACTION",
    "calibrate_fixed_finger",
    "load_fixed_finger_calibration",
    "project_longitudinal_positions",
    "save_fixed_finger_calibration",
    "warp_with_fixed_finger_calibration",
]
