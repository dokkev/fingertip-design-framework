"""Learning-free paired-line fingertip-boundary detection from RGB images."""

from __future__ import annotations

from dataclasses import dataclass

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

_HIGH_RESOLUTION_HEIGHT_THRESHOLD_PX = 720
_HIGH_RESOLUTION_GEOMETRY_SCALE = 0.5


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


def _detect_native(image: np.ndarray) -> FingertipBoundaryRegion:
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
    return FingertipBoundaryRegion(
        dorsal_boundary_xy_px=dorsal,
        palmar_boundary_xy_px=palmar,
        search_mask=mask,
        core_y_span=core_y_span,
        estimated_pad_width_px=median_width,
    )


def _map_region_to_image(
    region: FingertipBoundaryRegion,
    image_shape: tuple[int, int],
) -> FingertipBoundaryRegion:
    target_height, target_width = image_shape
    source_height, source_width = region.search_mask.shape
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    coordinate_scale = np.array((scale_x, scale_y), dtype=np.float64)
    coordinate_offset = 0.5 * coordinate_scale - 0.5

    dorsal = region.dorsal_boundary_xy_px * coordinate_scale + coordinate_offset
    palmar = region.palmar_boundary_xy_px * coordinate_scale + coordinate_offset
    search_mask = cv2.resize(
        region.search_mask.astype(np.uint8),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    core_start, core_stop = region.core_y_span
    mapped_core_span = (
        max(0, int(np.floor(core_start * scale_y))),
        min(target_height, int(np.ceil(core_stop * scale_y))),
    )
    return FingertipBoundaryRegion(
        dorsal_boundary_xy_px=dorsal,
        palmar_boundary_xy_px=palmar,
        search_mask=search_mask,
        core_y_span=mapped_core_span,
        estimated_pad_width_px=region.estimated_pad_width_px * scale_x,
    )


def detect_fingertip_boundary(rgb: np.ndarray) -> FingertipBoundaryRegion:
    """Detect paired dorsal and palmar side-view boundaries from one RGB frame."""

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be an H x W x 3 uint8 array")
    if image.shape[0] <= _HIGH_RESOLUTION_HEIGHT_THRESHOLD_PX:
        return _detect_native(image)

    geometry_image = cv2.resize(
        image,
        (
            max(1, round(image.shape[1] * _HIGH_RESOLUTION_GEOMETRY_SCALE)),
            max(1, round(image.shape[0] * _HIGH_RESOLUTION_GEOMETRY_SCALE)),
        ),
        interpolation=cv2.INTER_AREA,
    )
    return _map_region_to_image(
        _detect_native(geometry_image),
        image.shape[:2],
    )


__all__ = ["FingertipBoundaryRegion", "detect_fingertip_boundary"]
