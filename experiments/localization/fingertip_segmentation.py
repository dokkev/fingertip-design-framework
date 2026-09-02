"""Paired-LSD and smooth emissive-fingertip segmentation internals."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np


_LSD_SMOOTH_SIGMA_HEIGHT_FRACTION = 1.0 / 900.0
_MINIMUM_SEGMENT_LENGTH_HEIGHT_FRACTION = 0.018
_MINIMUM_SEGMENT_VERTICALITY = 0.70
_SIDE_SAMPLE_OFFSET_WIDTH_FRACTION = 0.007
_MINIMUM_SIDE_SAMPLE_OFFSET_PX = 3.0
_SIDE_SAMPLE_SPACING_PX = 5.0
_MINIMUM_SIDE_SAMPLE_COUNT = 12
_MAXIMUM_SIDE_SAMPLE_COUNT = 60
_SAMPLE_ENDPOINT_MARGIN_FRACTION = 0.12

_DORSAL_A_SCALE = 30.0
_DORSAL_SATURATION_SCALE = 150.0
_DORSAL_VALUE_SCALE = 100.0
_DORSAL_MINIMUM_A_DIFFERENCE = 8.0
_DORSAL_MINIMUM_SATURATION_DIFFERENCE = 25.0
_DORSAL_MINIMUM_VALUE_DIFFERENCE = 8.0
_DORSAL_WEAK_POLARITY_WEIGHT = 0.15
_MINIMUM_DORSAL_SCORE = 0.12
_DORSAL_CLUSTER_MAXIMUM_ANGLE_DEG = 18.0
_DORSAL_CLUSTER_MAXIMUM_DISTANCE_WIDTH_FRACTION = 0.035
_MINIMUM_DORSAL_SUPPORT_HEIGHT_FRACTION = 0.07

_PALMAR_MINIMUM_SEPARATION_SUPPORT_FRACTION = 0.28
_PALMAR_MAXIMUM_SEPARATION_SUPPORT_FRACTION = 1.30
_PALMAR_MINIMUM_OVERLAP_FRACTION = 0.10
_PALMAR_PREFERRED_SEPARATION_SUPPORT_FRACTION = 0.72
_PALMAR_SEPARATION_PRIOR_SIGMA = 0.32
_PALMAR_BRIGHTNESS_SCALE = 100.0
_PALMAR_FULL_POLARITY_VALUE_DIFFERENCE = -5.0
_PALMAR_WEAK_POLARITY_WEIGHT = 0.15
_MINIMUM_PALMAR_SCORE = 0.08
_PALMAR_CLUSTER_MAXIMUM_ANGLE_DEG = 25.0
_PALMAR_CLUSTER_MAXIMUM_SEPARATION_SUPPORT_FRACTION = 0.25
_PALMAR_CLUSTER_MAXIMUM_VALUE_DIFFERENCE = -3.0

_FIT_MINIMUM_ABS_VY = 0.40
_MINIMUM_MEDIAN_WIDTH_SUPPORT_FRACTION = 0.25
_MAXIMUM_MEDIAN_WIDTH_SUPPORT_FRACTION = 1.35
_MAXIMUM_WIDTH_CV = 0.35
_BOUNDARY_EXTENSION_WIDTH_FRACTION = 0.12
_SEARCH_MASK_INSET_WIDTH_FRACTION = 0.025

_GEOMETRY_MAXIMUM_HEIGHT_PX = 480

_EMISSIVE_ROW_PERCENTILE = 70.0
_EMISSIVE_CORRIDOR_HALF_WIDTH_FRACTION = 0.65
_EMISSIVE_ROW_CLOSE_WIDTH_FRACTION = 0.08
_SUPPORT_TOP_EXTENSION_WIDTH_FRACTION = 0.04
_SUPPORT_BOTTOM_EXTENSION_WIDTH_FRACTION = 0.02

_ENVELOPE_DILATION_WIDTH_FRACTION = 0.22
_ENVELOPE_DILATION_HEIGHT_FRACTION = 0.12
_ENVELOPE_LEFT_EXTENSION_WIDTH_FRACTION = 0.12
_ENVELOPE_RIGHT_EXTENSION_WIDTH_FRACTION = 0.16
_ENVELOPE_TOP_EXTENSION_WIDTH_FRACTION = 0.08
_ENVELOPE_BOTTOM_EXTENSION_WIDTH_FRACTION = 0.02

_CORE_CYAN_PERCENTILE = 40.0
_CORE_VALUE_PERCENTILE = 10.0
_CORE_LAB_A_PERCENTILE = 55.0
_CORE_MINIMUM_CYAN = 0.03
_CORE_MINIMUM_VALUE_DN = 50.0
_CORE_BOTTOM_EXCLUSION_FRACTION = 0.12
_CORE_EROSION_WIDTH_FRACTION = 0.025
_CORE_MINIMUM_AREA_PRIOR_FRACTION = 0.005
_CORE_FALLBACK_CYAN_PERCENTILE = 70.0
_GRABCUT_ITERATIONS = 4

_BRIDGE_OPEN_WIDTH_FRACTION = 0.035
_BRIDGE_OPEN_HEIGHT_FRACTION = 0.09
_COMPONENT_CORE_WEIGHT = 3.0
_COMPONENT_AREA_WEIGHT = 0.01
_COMPONENT_MINIMUM_PRIOR_OVERLAP_FRACTION = 0.03

_RADIAL_PRIOR_EROSION_WIDTH_FRACTION = 0.15
_RADIAL_MAXIMUM_LENGTH_WIDTH_FRACTION = 2.7
_RADIAL_HOLE_CLOSE_WIDTH_FRACTION = 0.025
_CONTOUR_ANGLE_COUNT = 256
_RADIAL_MINIMUM_PRIOR_FRACTION = 0.45
_RADIAL_INWARD_ALLOWANCE_WIDTH_FRACTION = 0.22
_RADIAL_OUTWARD_ALLOWANCE_WIDTH_FRACTION = 0.18
_CIRCULAR_MEDIAN_WINDOW = 9
_CIRCULAR_GAUSSIAN_SIGMA = 2.5
_CORE_MINIMUM_ROW_WIDTH_FRACTION = 0.50


@dataclass(frozen=True)
class FingertipBoundaryRegion:
    """Two side-view pad boundaries and the fingerpad interior between them."""

    dorsal_boundary_xy_px: np.ndarray
    palmar_boundary_xy_px: np.ndarray
    search_mask: np.ndarray
    core_y_span: tuple[int, int]
    estimated_pad_width_px: float

    def __post_init__(self) -> None:
        dorsal = np.asarray(self.dorsal_boundary_xy_px, dtype=np.float64)
        palmar = np.asarray(self.palmar_boundary_xy_px, dtype=np.float64)
        mask = np.asarray(self.search_mask, dtype=bool)
        if (
            dorsal.ndim != 2
            or dorsal.shape[1:] != (2,)
            or len(dorsal) < 2
            or not np.all(np.isfinite(dorsal))
        ):
            raise ValueError("dorsal_boundary_xy_px must be a finite N x 2 array")
        if palmar.shape != dorsal.shape or not np.all(np.isfinite(palmar)):
            raise ValueError("palmar_boundary_xy_px must match the dorsal boundary")
        if mask.ndim != 2 or not np.any(mask):
            raise ValueError("search_mask must be a nonempty H x W array")
        if not np.all(np.diff(dorsal[:, 1]) > 0.0):
            raise ValueError("boundary coordinates must be strictly ordered by y")
        if not np.array_equal(dorsal[:, 1], palmar[:, 1]):
            raise ValueError("dorsal and palmar boundaries must share image rows")
        if np.any(dorsal[:, 0] >= palmar[:, 0]):
            raise ValueError("the palmar boundary must remain right of the dorsal boundary")
        y_start, y_stop = self.core_y_span
        if not 0 <= y_start < y_stop <= mask.shape[0]:
            raise ValueError("core_y_span must be a valid half-open image-row interval")
        if (
            not np.isfinite(self.estimated_pad_width_px)
            or self.estimated_pad_width_px <= 0.0
        ):
            raise ValueError("estimated_pad_width_px must be finite and positive")
        dorsal = dorsal.copy()
        palmar = palmar.copy()
        mask = mask.copy()
        dorsal.setflags(write=False)
        palmar.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "dorsal_boundary_xy_px", dorsal)
        object.__setattr__(self, "palmar_boundary_xy_px", palmar)
        object.__setattr__(self, "search_mask", mask)


@dataclass(frozen=True)
class _Segment:
    endpoints_xy_px: np.ndarray
    length_px: float
    verticality: float
    midpoint_xy_px: np.ndarray
    unit_tangent_xy: np.ndarray
    a_left_minus_right: float
    saturation_right_minus_left: float
    value_right_minus_left: float

    @property
    def y_min(self) -> float:
        return float(np.min(self.endpoints_xy_px[:, 1]))

    @property
    def y_max(self) -> float:
        return float(np.max(self.endpoints_xy_px[:, 1]))


@dataclass(frozen=True)
class _LineFit:
    vx: float
    vy: float
    x0: float
    y0: float

    def x_at(self, y: np.ndarray | float) -> np.ndarray:
        y_array = np.asarray(y, dtype=np.float64)
        return self.x0 + (self.vx / self.vy) * (y_array - self.y0)


@dataclass(frozen=True)
class _PairedLsdPrior:
    region: FingertipBoundaryRegion
    dorsal_fit: _LineFit
    palmar_fit: _LineFit


@dataclass(frozen=True)
class FingertipSegmentation:
    """Final fingertip region plus geometry-only segmentation diagnostics."""

    region: FingertipBoundaryRegion
    coarse_prior_mask: np.ndarray
    raw_component_mask: np.ndarray
    final_mask: np.ndarray
    contour_xy_px: np.ndarray
    geometry_scale: float
    runtime_ms: float


def _prepare_channels(
    rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lab_a = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 1]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return gray, lab_a, hsv[:, :, 1], hsv[:, :, 2]


def _sample_channel(
    channel: np.ndarray,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    return cv2.remap(
        channel,
        x_coordinates.astype(np.float32).reshape(-1, 1),
        y_coordinates.astype(np.float32).reshape(-1, 1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    ).reshape(-1)


def _measure_segment(
    endpoints: np.ndarray,
    lab_a: np.ndarray,
    saturation: np.ndarray,
    value: np.ndarray,
    side_offset_px: float,
) -> _Segment | None:
    point_1, point_2 = np.asarray(endpoints, dtype=np.float64)
    delta = point_2 - point_1
    length = float(np.linalg.norm(delta))
    if length <= np.finfo(np.float64).eps:
        return None
    tangent = delta / length
    verticality = float(abs(tangent[1]))
    normal = np.array([tangent[1], -tangent[0]], dtype=np.float64)
    if normal[0] < 0.0:
        normal = -normal

    sample_count = int(
        np.clip(
            round(length / _SIDE_SAMPLE_SPACING_PX),
            _MINIMUM_SIDE_SAMPLE_COUNT,
            _MAXIMUM_SIDE_SAMPLE_COUNT,
        )
    )
    fractions = np.linspace(
        _SAMPLE_ENDPOINT_MARGIN_FRACTION,
        1.0 - _SAMPLE_ENDPOINT_MARGIN_FRACTION,
        sample_count,
    )
    centers = point_1 + fractions[:, None] * delta
    left_points = centers - side_offset_px * normal
    right_points = centers + side_offset_px * normal

    a_left = _sample_channel(lab_a, left_points[:, 0], left_points[:, 1])
    a_right = _sample_channel(lab_a, right_points[:, 0], right_points[:, 1])
    saturation_left = _sample_channel(
        saturation,
        left_points[:, 0],
        left_points[:, 1],
    )
    saturation_right = _sample_channel(
        saturation,
        right_points[:, 0],
        right_points[:, 1],
    )
    value_left = _sample_channel(
        value,
        left_points[:, 0],
        left_points[:, 1],
    )
    value_right = _sample_channel(
        value,
        right_points[:, 0],
        right_points[:, 1],
    )
    return _Segment(
        endpoints_xy_px=np.asarray((point_1, point_2), dtype=np.float64),
        length_px=length,
        verticality=verticality,
        midpoint_xy_px=0.5 * (point_1 + point_2),
        unit_tangent_xy=tangent,
        a_left_minus_right=float(np.median(a_left - a_right)),
        saturation_right_minus_left=float(
            np.median(saturation_right - saturation_left)
        ),
        value_right_minus_left=float(np.median(value_right - value_left)),
    )


def _detect_segments(
    gray: np.ndarray,
    lab_a: np.ndarray,
    saturation: np.ndarray,
    value: np.ndarray,
) -> tuple[_Segment, ...]:
    height, width = gray.shape
    sigma = max(0.7, height * _LSD_SMOOTH_SIGMA_HEIGHT_FRACTION)
    minimum_length = _MINIMUM_SEGMENT_LENGTH_HEIGHT_FRACTION * height
    side_offset = max(
        _MINIMUM_SIDE_SAMPLE_OFFSET_PX,
        _SIDE_SAMPLE_OFFSET_WIDTH_FRACTION * width,
    )
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lab_a_float = lab_a.astype(np.float32)
    saturation_float = saturation.astype(np.float32)
    value_float = value.astype(np.float32)
    segments: list[_Segment] = []
    for channel in (lab_a, gray):
        smoothed = cv2.GaussianBlur(channel, (0, 0), sigma)
        detected = detector.detect(smoothed)[0]
        if detected is None:
            continue
        for coordinates in detected.reshape(-1, 4):
            endpoints = coordinates.reshape(2, 2)
            delta = endpoints[1] - endpoints[0]
            length = float(np.linalg.norm(delta))
            if length < minimum_length:
                continue
            verticality = float(abs(delta[1]) / length)
            if verticality < _MINIMUM_SEGMENT_VERTICALITY:
                continue
            segment = _measure_segment(
                endpoints,
                lab_a_float,
                saturation_float,
                value_float,
                side_offset,
            )
            if segment is not None:
                segments.append(segment)
    if not segments:
        raise RuntimeError("LSD found no sufficiently long side-view line segments")
    return tuple(segments)


def _soft_polarity_weight(
    values: tuple[float, ...],
    thresholds: tuple[float, ...],
) -> float:
    support = min(
        float(np.clip(value / threshold, 0.0, 1.0))
        for value, threshold in zip(values, thresholds, strict=True)
    )
    return _DORSAL_WEAK_POLARITY_WEIGHT + (
        1.0 - _DORSAL_WEAK_POLARITY_WEIGHT
    ) * support


def _dorsal_score(segment: _Segment, image_height: int) -> float:
    a_evidence = max(segment.a_left_minus_right, 0.0) / _DORSAL_A_SCALE
    saturation_evidence = (
        max(segment.saturation_right_minus_left, 0.0)
        / _DORSAL_SATURATION_SCALE
    )
    value_evidence = (
        max(segment.value_right_minus_left, 0.0) / _DORSAL_VALUE_SCALE
    )
    polarity_weight = _soft_polarity_weight(
        (
            segment.a_left_minus_right,
            segment.saturation_right_minus_left,
            segment.value_right_minus_left,
        ),
        (
            _DORSAL_MINIMUM_A_DIFFERENCE,
            _DORSAL_MINIMUM_SATURATION_DIFFERENCE,
            _DORSAL_MINIMUM_VALUE_DIFFERENCE,
        ),
    )
    return float(
        (segment.length_px / image_height) ** 0.7
        * segment.verticality
        * (0.15 + a_evidence + saturation_evidence + value_evidence)
        * polarity_weight
    )


def _orientation_difference_deg(first: _Segment, second: _Segment) -> float:
    cosine = float(
        np.clip(
            abs(np.dot(first.unit_tangent_xy, second.unit_tangent_xy)),
            0.0,
            1.0,
        )
    )
    return float(np.degrees(np.arccos(cosine)))


def _segment_line_x_at(segment: _Segment, y_coordinate: float) -> float:
    delta = segment.endpoints_xy_px[1] - segment.endpoints_xy_px[0]
    if abs(delta[1]) <= np.finfo(np.float64).eps:
        raise RuntimeError("vertical segment query received a horizontal line")
    return float(
        segment.endpoints_xy_px[0, 0]
        + delta[0]
        / delta[1]
        * (y_coordinate - segment.endpoints_xy_px[0, 1])
    )


def _fit_line(segments: tuple[_Segment, ...], name: str) -> _LineFit:
    points = np.concatenate([segment.endpoints_xy_px for segment in segments], axis=0)
    fitted = cv2.fitLine(
        points.astype(np.float32).reshape(-1, 1, 2),
        cv2.DIST_HUBER,
        0,
        0.01,
        0.01,
    ).reshape(-1)
    vx, vy, x0, y0 = (float(value) for value in fitted)
    if not np.all(np.isfinite((vx, vy, x0, y0))) or abs(vy) < _FIT_MINIMUM_ABS_VY:
        raise RuntimeError(f"{name} robust line fit is not a valid side-view line")
    return _LineFit(vx=vx, vy=vy, x0=x0, y0=y0)


def _select_dorsal_segments(
    segments: tuple[_Segment, ...],
    image_shape: tuple[int, int],
) -> tuple[tuple[_Segment, ...], _LineFit, tuple[float, float]]:
    height, width = image_shape
    scores = np.asarray(
        [_dorsal_score(segment, height) for segment in segments],
        dtype=np.float64,
    )
    seed_index = int(np.argmax(scores))
    if scores[seed_index] < _MINIMUM_DORSAL_SCORE:
        raise RuntimeError(
            f"best dorsal LSD score is too weak: {scores[seed_index]:.3f}"
        )
    seed = segments[seed_index]
    maximum_distance = _DORSAL_CLUSTER_MAXIMUM_DISTANCE_WIDTH_FRACTION * width
    selected = [seed]
    for index, candidate in enumerate(segments):
        if index == seed_index:
            continue
        midpoint_y = float(candidate.midpoint_xy_px[1])
        distance = abs(
            float(candidate.midpoint_xy_px[0])
            - _segment_line_x_at(seed, midpoint_y)
        )
        if (
            _orientation_difference_deg(seed, candidate)
            <= _DORSAL_CLUSTER_MAXIMUM_ANGLE_DEG
            and distance <= maximum_distance
            and candidate.a_left_minus_right >= _DORSAL_MINIMUM_A_DIFFERENCE
            and candidate.saturation_right_minus_left
            >= _DORSAL_MINIMUM_SATURATION_DIFFERENCE
            and candidate.value_right_minus_left
            >= _DORSAL_MINIMUM_VALUE_DIFFERENCE
        ):
            selected.append(candidate)
    selected_tuple = tuple(selected)
    y_min = min(segment.y_min for segment in selected_tuple)
    y_max = max(segment.y_max for segment in selected_tuple)
    support = y_max - y_min
    if support < _MINIMUM_DORSAL_SUPPORT_HEIGHT_FRACTION * height:
        raise RuntimeError(f"dorsal LSD support is too short: {support:.1f} px")
    return selected_tuple, _fit_line(selected_tuple, "dorsal"), (y_min, y_max)


def _vertical_overlap(
    segment: _Segment,
    y_support: tuple[float, float],
) -> float:
    return max(
        0.0,
        min(segment.y_max, y_support[1]) - max(segment.y_min, y_support[0]),
    )


def _palmar_polarity_weight(value_right_minus_left: float) -> float:
    support = float(
        np.clip(
            -value_right_minus_left / -_PALMAR_FULL_POLARITY_VALUE_DIFFERENCE,
            0.0,
            1.0,
        )
    )
    return _PALMAR_WEAK_POLARITY_WEIGHT + (
        1.0 - _PALMAR_WEAK_POLARITY_WEIGHT
    ) * support


def _palmar_score(
    segment: _Segment,
    separation: float,
    dorsal_support: float,
) -> float:
    normalized_separation = separation / dorsal_support
    width_prior = np.exp(
        -0.5
        * (
            (
                normalized_separation
                - _PALMAR_PREFERRED_SEPARATION_SUPPORT_FRACTION
            )
            / _PALMAR_SEPARATION_PRIOR_SIGMA
        )
        ** 2
    )
    brightness_drop = max(-segment.value_right_minus_left, 0.0)
    return float(
        (segment.length_px / dorsal_support) ** 0.7
        * segment.verticality
        * (0.15 + brightness_drop / _PALMAR_BRIGHTNESS_SCALE)
        * width_prior
        * _palmar_polarity_weight(segment.value_right_minus_left)
    )


def _select_palmar_segments(
    segments: tuple[_Segment, ...],
    dorsal_fit: _LineFit,
    dorsal_y_support: tuple[float, float],
) -> tuple[tuple[_Segment, ...], _LineFit, tuple[float, float]]:
    dorsal_support = dorsal_y_support[1] - dorsal_y_support[0]
    candidates: list[tuple[_Segment, float, float]] = []
    for segment in segments:
        midpoint_y = float(segment.midpoint_xy_px[1])
        separation = float(segment.midpoint_xy_px[0]) - float(
            dorsal_fit.x_at(midpoint_y)
        )
        overlap = _vertical_overlap(segment, dorsal_y_support)
        if (
            _PALMAR_MINIMUM_SEPARATION_SUPPORT_FRACTION * dorsal_support
            <= separation
            <= _PALMAR_MAXIMUM_SEPARATION_SUPPORT_FRACTION * dorsal_support
            and overlap
            >= _PALMAR_MINIMUM_OVERLAP_FRACTION
            * min(segment.length_px, dorsal_support)
        ):
            candidates.append(
                (
                    segment,
                    separation,
                    _palmar_score(segment, separation, dorsal_support),
                )
            )
    if not candidates:
        raise RuntimeError("no palmar LSD segment overlaps the dorsal boundary")
    seed, seed_separation, best_score = max(candidates, key=lambda item: item[2])
    if best_score < _MINIMUM_PALMAR_SCORE:
        raise RuntimeError(f"best palmar LSD score is too weak: {best_score:.3f}")

    selected = [seed]
    maximum_separation_difference = (
        _PALMAR_CLUSTER_MAXIMUM_SEPARATION_SUPPORT_FRACTION * dorsal_support
    )
    for candidate, separation, _ in candidates:
        if candidate is seed:
            continue
        if (
            _orientation_difference_deg(seed, candidate)
            < _PALMAR_CLUSTER_MAXIMUM_ANGLE_DEG
            and abs(separation - seed_separation) < maximum_separation_difference
            and candidate.value_right_minus_left
            < _PALMAR_CLUSTER_MAXIMUM_VALUE_DIFFERENCE
        ):
            selected.append(candidate)
    selected_tuple = tuple(selected)
    y_min = min(segment.y_min for segment in selected_tuple)
    y_max = max(segment.y_max for segment in selected_tuple)
    return selected_tuple, _fit_line(selected_tuple, "palmar"), (y_min, y_max)


def _paired_boundaries(
    image_shape: tuple[int, int],
    dorsal_fit: _LineFit,
    dorsal_y_support: tuple[float, float],
    palmar_fit: _LineFit,
    palmar_y_support: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], float]:
    height, width = image_shape
    dorsal_support = dorsal_y_support[1] - dorsal_y_support[0]
    core_start = max(0, int(np.ceil(dorsal_y_support[0])))
    core_stop = min(height, int(np.floor(dorsal_y_support[1])) + 1)
    core_rows = np.arange(core_start, core_stop, dtype=np.float64)
    if core_rows.size < 2:
        raise RuntimeError("dorsal robust line has insufficient image-row support")
    core_widths = palmar_fit.x_at(core_rows) - dorsal_fit.x_at(core_rows)
    if np.any(core_widths <= 0.0):
        raise RuntimeError("paired fingertip boundaries cross")
    median_width = float(np.median(core_widths))
    mean_width = float(np.mean(core_widths))
    width_cv = float(np.std(core_widths) / mean_width)
    if not (
        _MINIMUM_MEDIAN_WIDTH_SUPPORT_FRACTION * dorsal_support
        <= median_width
        <= _MAXIMUM_MEDIAN_WIDTH_SUPPORT_FRACTION * dorsal_support
    ):
        raise RuntimeError(
            "paired fingertip width is implausible: "
            f"{median_width:.1f} px for {dorsal_support:.1f} px support"
        )
    if width_cv > _MAXIMUM_WIDTH_CV:
        raise RuntimeError(f"paired fingertip width varies too much: CV={width_cv:.3f}")

    y_start = max(
        0,
        int(np.floor(min(dorsal_y_support[0], palmar_y_support[0]))),
    )
    y_stop = min(
        height,
        int(
            np.ceil(
                max(dorsal_y_support[1], palmar_y_support[1])
                + _BOUNDARY_EXTENSION_WIDTH_FRACTION * median_width
            )
        )
        + 1,
    )
    rows = np.arange(y_start, y_stop, dtype=np.float64)
    dorsal_x = dorsal_fit.x_at(rows)
    palmar_x = palmar_fit.x_at(rows)
    if (
        np.any(dorsal_x < 0.0)
        or np.any(palmar_x >= width)
        or np.any(dorsal_x >= palmar_x)
    ):
        raise RuntimeError("fitted fingertip boundaries leave or cross the image")

    inset = max(1, round(_SEARCH_MASK_INSET_WIDTH_FRACTION * median_width))
    mask = np.zeros((height, width), dtype=bool)
    for row, left, right in zip(
        rows.astype(np.int32),
        dorsal_x,
        palmar_x,
        strict=True,
    ):
        start = max(0, int(np.ceil(left)) + inset)
        stop = min(width, int(np.floor(right)) - inset + 1)
        if start < stop:
            mask[row, start:stop] = True
    if not np.any(mask):
        raise RuntimeError("fingertip boundaries produced an empty search mask")
    return (
        np.column_stack((dorsal_x, rows)),
        np.column_stack((palmar_x, rows)),
        mask,
        (core_start, core_stop),
        median_width,
    )


def _detect_paired_lsd_prior(image: np.ndarray) -> _PairedLsdPrior:
    gray, lab_a, saturation, value = _prepare_channels(image)
    segments = _detect_segments(gray, lab_a, saturation, value)
    _, dorsal_fit, dorsal_support = _select_dorsal_segments(
        segments,
        gray.shape,
    )
    _, palmar_fit, palmar_support = _select_palmar_segments(
        segments,
        dorsal_fit,
        dorsal_support,
    )
    dorsal, palmar, mask, core_y_span, median_width = _paired_boundaries(
        gray.shape,
        dorsal_fit,
        dorsal_support,
        palmar_fit,
        palmar_support,
    )
    return _PairedLsdPrior(
        region=FingertipBoundaryRegion(
            dorsal_boundary_xy_px=dorsal,
            palmar_boundary_xy_px=palmar,
            search_mask=mask,
            core_y_span=core_y_span,
            estimated_pad_width_px=median_width,
        ),
        dorsal_fit=dorsal_fit,
        palmar_fit=palmar_fit,
    )


def _odd_kernel_size(value: float, *, minimum: int = 1) -> int:
    size = max(minimum, int(round(value)))
    return size if size % 2 == 1 else size + 1


def _emission_score(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.asarray(rgb, dtype=np.float32) / 255.0
    red, green, blue = np.moveaxis(normalized, -1, 0)
    green_blue = np.minimum(green, blue)
    cyan = np.clip(
        (green_blue - red) / (red + green + blue + 1.0e-6),
        0.0,
        0.5,
    ) / 0.5
    bright_core = np.clip((green_blue - 0.50) / 0.50, 0.0, 1.0)
    emission = green_blue * (0.25 + 0.75 * cyan) + 0.10 * bright_core
    return emission, cyan


def _true_runs(values: np.ndarray) -> tuple[tuple[int, int], ...]:
    active = np.asarray(values, dtype=bool)
    changes = np.diff(np.pad(active.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return tuple(zip(starts.tolist(), stops.tolist(), strict=True))


def _recover_vertical_support(
    emission: np.ndarray,
    prior: _PairedLsdPrior,
) -> tuple[int, int]:
    height, width = emission.shape
    pad_width = prior.region.estimated_pad_width_px
    rows = np.arange(height, dtype=np.float64)
    centers = 0.5 * (
        prior.dorsal_fit.x_at(rows) + prior.palmar_fit.x_at(rows)
    )
    corridor_half_width = _EMISSIVE_CORRIDOR_HALF_WIDTH_FRACTION * pad_width
    profile = np.zeros(height, dtype=np.float32)
    for row, center_x in enumerate(centers):
        start = max(0, int(np.floor(center_x - corridor_half_width)))
        stop = min(width, int(np.ceil(center_x + corridor_half_width)) + 1)
        if start < stop:
            profile[row] = np.percentile(
                emission[row, start:stop],
                _EMISSIVE_ROW_PERCENTILE,
            )
    profile_range = float(np.ptp(profile))
    if profile_range <= np.finfo(np.float32).eps:
        raise RuntimeError("emissive row profile has no usable contrast")
    normalized_profile = np.rint(
        255.0 * (profile - float(np.min(profile))) / profile_range
    ).astype(np.uint8)
    _, active_rows = cv2.threshold(
        normalized_profile[:, None],
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    close_length = _odd_kernel_size(
        _EMISSIVE_ROW_CLOSE_WIDTH_FRACTION * pad_width,
        minimum=3,
    )
    active_rows = cv2.morphologyEx(
        active_rows,
        cv2.MORPH_CLOSE,
        np.ones((close_length, 1), dtype=np.uint8),
    ).ravel() > 0

    lsd_start, lsd_stop = prior.region.core_y_span
    candidates = []
    for start, stop in _true_runs(active_rows):
        overlap = max(0, min(stop, lsd_stop) - max(start, lsd_start))
        if overlap:
            candidates.append((overlap, stop - start, start, stop))
    if not candidates:
        raise RuntimeError("emissive row support does not overlap the LSD prior")
    _, _, start, stop = max(candidates)
    start = max(
        0,
        start - round(_SUPPORT_TOP_EXTENSION_WIDTH_FRACTION * pad_width),
    )
    stop = min(
        height,
        stop + round(_SUPPORT_BOTTOM_EXTENSION_WIDTH_FRACTION * pad_width),
    )
    if stop - start < 2:
        raise RuntimeError("recovered emissive support is too short")
    return start, stop


def _mask_between_lines(
    image_shape: tuple[int, int],
    dorsal_fit: _LineFit,
    palmar_fit: _LineFit,
    vertical_support: tuple[int, int],
) -> np.ndarray:
    height, width = image_shape
    start, stop = vertical_support
    rows = np.arange(start, stop, dtype=np.float64)
    dorsal_x = dorsal_fit.x_at(rows)
    palmar_x = palmar_fit.x_at(rows)
    if np.any(dorsal_x >= palmar_x):
        raise RuntimeError("coarse fingertip prior boundaries cross")
    mask = np.zeros((height, width), dtype=bool)
    for row, left, right in zip(
        rows.astype(np.int32),
        dorsal_x,
        palmar_x,
        strict=True,
    ):
        left_index = max(0, int(np.ceil(left)))
        right_index = min(width, int(np.floor(right)) + 1)
        if left_index < right_index:
            mask[row, left_index:right_index] = True
    if not np.any(mask):
        raise RuntimeError("coarse fingertip prior is empty")
    return mask


def _segmentation_envelope(prior_mask: np.ndarray, pad_width: float) -> np.ndarray:
    height, width = prior_mask.shape
    kernel_width = _odd_kernel_size(
        _ENVELOPE_DILATION_WIDTH_FRACTION * pad_width,
        minimum=3,
    )
    kernel_height = _odd_kernel_size(
        _ENVELOPE_DILATION_HEIGHT_FRACTION * pad_width,
        minimum=3,
    )
    dilated = cv2.dilate(
        prior_mask.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_width, kernel_height),
        ),
    ).astype(bool)
    active_y, active_x = np.nonzero(prior_mask)
    left = max(
        0,
        int(np.min(active_x) - _ENVELOPE_LEFT_EXTENSION_WIDTH_FRACTION * pad_width),
    )
    right = min(
        width,
        int(
            np.max(active_x)
            + 1
            + _ENVELOPE_RIGHT_EXTENSION_WIDTH_FRACTION * pad_width
        ),
    )
    top = max(
        0,
        int(np.min(active_y) - _ENVELOPE_TOP_EXTENSION_WIDTH_FRACTION * pad_width),
    )
    bottom = min(
        height,
        int(
            np.max(active_y)
            + 1
            + _ENVELOPE_BOTTOM_EXTENSION_WIDTH_FRACTION * pad_width
        ),
    )
    rectangle = np.zeros_like(prior_mask)
    rectangle[top:bottom, left:right] = True
    return dilated & rectangle


def _definite_emissive_core(
    rgb: np.ndarray,
    cyan: np.ndarray,
    prior_mask: np.ndarray,
    vertical_support: tuple[int, int],
    pad_width: float,
) -> np.ndarray:
    hsv_value = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 2]
    lab_a = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 1]
    cyan_threshold = max(
        _CORE_MINIMUM_CYAN,
        float(np.percentile(cyan[prior_mask], _CORE_CYAN_PERCENTILE)),
    )
    value_threshold = max(
        _CORE_MINIMUM_VALUE_DN,
        float(np.percentile(hsv_value[prior_mask], _CORE_VALUE_PERCENTILE)),
    )
    lab_a_threshold = float(
        np.percentile(lab_a[prior_mask], _CORE_LAB_A_PERCENTILE)
    )
    core = (
        prior_mask
        & (hsv_value > value_threshold)
        & ((cyan > cyan_threshold) | (lab_a < lab_a_threshold - 1.0))
    )
    start, stop = vertical_support
    excluded_start = int(
        np.floor(stop - _CORE_BOTTOM_EXCLUSION_FRACTION * (stop - start))
    )
    core[excluded_start:stop] = False
    kernel_size = _odd_kernel_size(
        _CORE_EROSION_WIDTH_FRACTION * pad_width,
        minimum=3,
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    eroded = cv2.erode(core.astype(np.uint8), kernel).astype(bool)
    minimum_area = max(
        8,
        round(_CORE_MINIMUM_AREA_PRIOR_FRACTION * np.count_nonzero(prior_mask)),
    )
    if np.count_nonzero(eroded) >= minimum_area:
        return eroded

    fallback_threshold = float(
        np.percentile(cyan[prior_mask], _CORE_FALLBACK_CYAN_PERCENTILE)
    )
    fallback = prior_mask & (hsv_value >= value_threshold) & (
        cyan >= fallback_threshold
    )
    fallback[excluded_start:stop] = False
    fallback = cv2.erode(fallback.astype(np.uint8), kernel).astype(bool)
    if np.count_nonzero(fallback) < minimum_area:
        raise RuntimeError("emissive fingertip has no reliable foreground seed")
    return fallback


def _grabcut_foreground(
    rgb: np.ndarray,
    prior_mask: np.ndarray,
    envelope: np.ndarray,
    definite_core: np.ndarray,
) -> np.ndarray:
    labels = np.full(prior_mask.shape, cv2.GC_BGD, dtype=np.uint8)
    labels[envelope] = cv2.GC_PR_BGD
    labels[prior_mask] = cv2.GC_PR_FGD
    labels[definite_core] = cv2.GC_FGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        rgb,
        labels,
        None,
        background_model,
        foreground_model,
        _GRABCUT_ITERATIONS,
        cv2.GC_INIT_WITH_MASK,
    )
    foreground = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
    return foreground & envelope


def _remove_bridge_and_select_component(
    foreground: np.ndarray,
    prior_mask: np.ndarray,
    definite_core: np.ndarray,
    pad_width: float,
) -> np.ndarray:
    kernel_width = _odd_kernel_size(
        _BRIDGE_OPEN_WIDTH_FRACTION * pad_width,
        minimum=3,
    )
    kernel_height = _odd_kernel_size(
        _BRIDGE_OPEN_HEIGHT_FRACTION * pad_width,
        minimum=3,
    )
    opened = cv2.morphologyEx(
        foreground.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_width, kernel_height),
        ),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    minimum_prior_overlap = max(
        8,
        round(
            _COMPONENT_MINIMUM_PRIOR_OVERLAP_FRACTION
            * np.count_nonzero(prior_mask)
        ),
    )
    best: tuple[float, int] | None = None
    for component in range(1, component_count):
        component_mask = labels == component
        prior_overlap = int(np.count_nonzero(component_mask & prior_mask))
        if prior_overlap < minimum_prior_overlap:
            continue
        core_overlap = int(np.count_nonzero(component_mask & definite_core))
        area = int(stats[component, cv2.CC_STAT_AREA])
        score = (
            prior_overlap
            + _COMPONENT_CORE_WEIGHT * core_overlap
            + _COMPONENT_AREA_WEIGHT * area
        )
        if best is None or score > best[0]:
            best = (score, component)
    if best is None:
        raise RuntimeError("GrabCut found no component tied to the fingertip prior")
    return labels == best[1]


def _fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    exterior = (1 - padded).copy()
    cv2.floodFill(exterior, None, (0, 0), 2)
    holes = exterior == 1
    return (padded.astype(bool) | holes)[1:-1, 1:-1]


def _radial_center(prior_mask: np.ndarray, pad_width: float) -> tuple[float, float]:
    erosion_size = _odd_kernel_size(
        _RADIAL_PRIOR_EROSION_WIDTH_FRACTION * pad_width,
        minimum=3,
    )
    eroded = cv2.erode(
        prior_mask.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erosion_size, erosion_size),
        ),
    )
    center_mask = eroded if np.any(eroded) else prior_mask.astype(np.uint8)
    moments = cv2.moments(center_mask)
    if moments["m00"] <= 0.0:
        raise RuntimeError("coarse fingertip prior has no radial center")
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def _radial_extent(
    mask: np.ndarray,
    center_xy: tuple[float, float],
    angles: np.ndarray,
    radial_samples: np.ndarray,
    radial_close_size: int,
) -> np.ndarray:
    center_x, center_y = center_xy
    map_x = center_x + np.cos(angles)[:, None] * radial_samples[None, :]
    map_y = center_y + np.sin(angles)[:, None] * radial_samples[None, :]
    sampled = cv2.remap(
        mask.astype(np.uint8),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    sampled = cv2.morphologyEx(
        sampled,
        cv2.MORPH_CLOSE,
        np.ones((1, radial_close_size), dtype=np.uint8),
    ).astype(bool)
    connected = np.logical_and.accumulate(sampled, axis=1)
    sample_count = np.count_nonzero(connected, axis=1)
    indices = np.maximum(sample_count - 1, 0)
    return radial_samples[indices]


def _circular_smooth(values: np.ndarray) -> np.ndarray:
    half_median = _CIRCULAR_MEDIAN_WINDOW // 2
    padded = np.pad(values, (half_median, half_median), mode="wrap")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        _CIRCULAR_MEDIAN_WINDOW,
    )
    median_filtered = np.median(windows, axis=1)

    gaussian_size = _odd_kernel_size(6.0 * _CIRCULAR_GAUSSIAN_SIGMA, minimum=3)
    gaussian = cv2.getGaussianKernel(
        gaussian_size,
        _CIRCULAR_GAUSSIAN_SIGMA,
    ).ravel()
    half_gaussian = gaussian_size // 2
    padded_median = np.pad(
        median_filtered,
        (half_gaussian, half_gaussian),
        mode="wrap",
    )
    return np.convolve(padded_median, gaussian, mode="valid")


def _regularize_contour(
    component_mask: np.ndarray,
    prior_mask: np.ndarray,
    pad_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    center = _radial_center(prior_mask, pad_width)
    center_index = tuple(np.rint(center).astype(int)[::-1])
    if not component_mask[center_index]:
        raise RuntimeError("selected fingertip component excludes the coarse-prior center")
    angles = np.linspace(-np.pi, np.pi, _CONTOUR_ANGLE_COUNT, endpoint=False)
    radial_samples = np.arange(
        0.0,
        _RADIAL_MAXIMUM_LENGTH_WIDTH_FRACTION * pad_width + 1.0,
        1.0,
    )
    radial_close_size = _odd_kernel_size(
        _RADIAL_HOLE_CLOSE_WIDTH_FRACTION * pad_width,
        minimum=3,
    )
    prior_radius = _radial_extent(
        prior_mask,
        center,
        angles,
        radial_samples,
        radial_close_size,
    )
    observed_radius = _radial_extent(
        component_mask,
        center,
        angles,
        radial_samples,
        radial_close_size,
    )
    minimum_radius = np.maximum(
        _RADIAL_MINIMUM_PRIOR_FRACTION * prior_radius,
        prior_radius - _RADIAL_INWARD_ALLOWANCE_WIDTH_FRACTION * pad_width,
    )
    maximum_radius = (
        prior_radius + _RADIAL_OUTWARD_ALLOWANCE_WIDTH_FRACTION * pad_width
    )
    radius = _circular_smooth(
        np.clip(observed_radius, minimum_radius, maximum_radius)
    )
    contour = np.column_stack(
        (
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
        )
    )
    height, width = component_mask.shape
    contour[:, 0] = np.clip(contour[:, 0], 0.0, width - 1.0)
    contour[:, 1] = np.clip(contour[:, 1], 0.0, height - 1.0)
    final_mask = np.zeros_like(component_mask, dtype=np.uint8)
    cv2.fillPoly(final_mask, [np.rint(contour).astype(np.int32)], 1)
    return contour, final_mask.astype(bool)


def _longest_run(values: np.ndarray) -> tuple[int, int]:
    runs = _true_runs(values)
    if not runs:
        raise RuntimeError("fingertip mask has no stable central row run")
    return max(runs, key=lambda run: run[1] - run[0])


def _region_from_mask(mask: np.ndarray) -> FingertipBoundaryRegion:
    silhouette = np.asarray(mask, dtype=bool)
    height, width = silhouette.shape
    row_counts = np.count_nonzero(silhouette, axis=1)
    rows = np.flatnonzero(row_counts >= 2)
    if rows.size < 2:
        raise RuntimeError("smooth fingertip mask has insufficient row support")
    dorsal_x = np.empty(rows.size, dtype=np.float64)
    palmar_x = np.empty(rows.size, dtype=np.float64)
    for index, row in enumerate(rows):
        active_x = np.flatnonzero(silhouette[row])
        dorsal_x[index] = active_x[0]
        palmar_x[index] = active_x[-1]
    widths = palmar_x - dorsal_x + 1.0
    reference_width = float(np.median(widths))
    stable_on_rows = widths >= _CORE_MINIMUM_ROW_WIDTH_FRACTION * reference_width
    run_start, run_stop = _longest_run(stable_on_rows)
    core_rows = rows[run_start:run_stop]
    core_y_span = (int(core_rows[0]), int(core_rows[-1]) + 1)
    estimated_width = float(np.median(widths[run_start:run_stop]))

    inset_size = _odd_kernel_size(
        _SEARCH_MASK_INSET_WIDTH_FRACTION * estimated_width,
        minimum=3,
    )
    search_mask = cv2.erode(
        silhouette.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (inset_size, inset_size),
        ),
    ).astype(bool)
    if not np.any(search_mask):
        raise RuntimeError("smooth fingertip mask produced an empty LED search mask")
    return FingertipBoundaryRegion(
        dorsal_boundary_xy_px=np.column_stack((dorsal_x, rows)),
        palmar_boundary_xy_px=np.column_stack((palmar_x, rows)),
        search_mask=search_mask,
        core_y_span=core_y_span,
        estimated_pad_width_px=estimated_width,
    )


def _resize_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    return cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def segment_fingertip(
    rgb: np.ndarray,
) -> FingertipSegmentation:
    start_time = perf_counter()
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be an H x W x 3 uint8 array")
    geometry_scale = min(1.0, _GEOMETRY_MAXIMUM_HEIGHT_PX / image.shape[0])
    if geometry_scale < 1.0:
        geometry_image = cv2.resize(
            image,
            (
                max(1, round(image.shape[1] * geometry_scale)),
                max(1, round(image.shape[0] * geometry_scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        geometry_image = image

    prior = _detect_paired_lsd_prior(geometry_image)
    emission, cyan = _emission_score(geometry_image)
    vertical_support = _recover_vertical_support(emission, prior)
    prior_mask = _mask_between_lines(
        geometry_image.shape[:2],
        prior.dorsal_fit,
        prior.palmar_fit,
        vertical_support,
    )
    pad_width = prior.region.estimated_pad_width_px
    envelope = _segmentation_envelope(prior_mask, pad_width)
    definite_core = _definite_emissive_core(
        geometry_image,
        cyan,
        prior_mask,
        vertical_support,
        pad_width,
    )
    foreground = _grabcut_foreground(
        geometry_image,
        prior_mask,
        envelope,
        definite_core,
    )
    component = _remove_bridge_and_select_component(
        foreground,
        prior_mask,
        definite_core,
        pad_width,
    )
    component = _fill_internal_holes(component)
    contour, final_mask = _regularize_contour(component, prior_mask, pad_width)

    if geometry_scale < 1.0:
        prior_mask = _resize_mask(prior_mask, image.shape[:2])
        component = _resize_mask(component, image.shape[:2])
        final_mask = _resize_mask(final_mask, image.shape[:2])
        scale_x = image.shape[1] / geometry_image.shape[1]
        scale_y = image.shape[0] / geometry_image.shape[0]
        contour_scale = np.array((scale_x, scale_y), dtype=np.float64)
        contour = (contour + 0.5) * contour_scale - 0.5
    region = _region_from_mask(final_mask)
    return FingertipSegmentation(
        region=region,
        coarse_prior_mask=prior_mask,
        raw_component_mask=component,
        final_mask=final_mask,
        contour_xy_px=contour,
        geometry_scale=geometry_scale,
        runtime_ms=1000.0 * (perf_counter() - start_time),
    )


__all__ = [
    "FingertipBoundaryRegion",
    "FingertipSegmentation",
    "segment_fingertip",
]
