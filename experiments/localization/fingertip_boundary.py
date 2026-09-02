"""Learning-free side-view fingertip-boundary detection from RGB images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


_REFERENCE_IMAGE_WIDTH_PX = 640
_REFERENCE_IMAGE_HEIGHT_PX = 480
_CYAN_SMOOTH_SIGMA_PX = 1.0

_DORSAL_GRADIENT_THRESHOLD = 0.07
_DORSAL_CYAN_RIGHT_THRESHOLD = 0.24
_DORSAL_CYAN_LEFT_THRESHOLD = 0.10
_DORSAL_MINIMUM_BRIGHTNESS_DN = 80.0
_DORSAL_SAMPLE_OFFSET_WIDTH_FRACTION = 0.006
_DORSAL_CLOSE_KERNEL_WIDTH_FRACTION = 0.004
_DORSAL_CLOSE_KERNEL_HEIGHT_FRACTION = 0.025
_DORSAL_SUPPORT_NEAR_WIDTH_FRACTION = 0.012
_DORSAL_SUPPORT_FAR_WIDTH_FRACTION = 0.12
_DORSAL_BROAD_LEFT_THRESHOLD = 0.15
_DORSAL_BROAD_RIGHT_THRESHOLD = 0.14
_DORSAL_GROUP_KERNEL_WIDTH_FRACTION = 0.03
_DORSAL_GROUP_KERNEL_HEIGHT_FRACTION = 0.15
_DORSAL_MINIMUM_COMPONENT_HEIGHT_FRACTION = 0.18
_DORSAL_MINIMUM_COMPONENT_AREA_FRACTION = 0.00008
_DORSAL_MEDIAN_WINDOW_HEIGHT_FRACTION = 0.03

_LOW_CYAN_THRESHOLD = 0.12
_LOW_CYAN_RUN_WIDTH_FRACTION = 0.008
_PAD_WIDTH_INFLATION = 1.15
_MINIMUM_PAD_WIDTH_FRACTION = 0.08
_MAXIMUM_PAD_WIDTH_FRACTION = 0.28
_FALLBACK_PAD_WIDTH_FRACTION = 0.16

_EDGE_NORMALIZATION_PERCENTILE = 99.5
_GRAY_EDGE_WEIGHT = 0.75
_CYAN_EDGE_WEIGHT = 0.25
_PALMAR_MINIMUM_WIDTH_FACTOR = 0.45
_PALMAR_MAXIMUM_WIDTH_FACTOR = 1.55
_WIDTH_PRIOR_SCALE = 0.42
_WIDTH_PRIOR_WEIGHT = 0.28
_PALMAR_SMOOTHNESS_PENALTY = 0.06
_PALMAR_MAXIMUM_STEP_WIDTH_FRACTION = 0.08

_SEARCH_MASK_INSET_WIDTH_FRACTION = 0.01
_SEARCH_MASK_Y_PADDING_FRACTION = 0.08


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


def _odd_at_least(value: float, minimum: int = 3) -> int:
    size = max(minimum, round(value))
    return size if size % 2 else size + 1


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    window = _odd_at_least(window)
    if window > len(values):
        window = len(values) if len(values) % 2 else len(values) - 1
    if window < 3:
        return np.asarray(values, dtype=np.float64).copy()
    half = window // 2
    padded = np.pad(np.asarray(values, dtype=np.float64), half, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=1)


def _cyan_geometry(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = rgb.astype(np.float32)
    red, green, blue = np.moveaxis(image, -1, 0)
    cyan = (np.minimum(green, blue) - red) / (red + green + blue + 1.0)
    scale = min(
        rgb.shape[1] / _REFERENCE_IMAGE_WIDTH_PX,
        rgb.shape[0] / _REFERENCE_IMAGE_HEIGHT_PX,
    )
    smoothed = cv2.GaussianBlur(
        cyan,
        (0, 0),
        max(0.5, _CYAN_SMOOTH_SIGMA_PX * scale),
    )
    return smoothed, np.maximum(green, blue)


def _dorsal_boundary(
    cyan: np.ndarray,
    maximum_green_blue: np.ndarray,
) -> np.ndarray:
    height, width = cyan.shape
    resolution_scale = min(
        width / _REFERENCE_IMAGE_WIDTH_PX,
        height / _REFERENCE_IMAGE_HEIGHT_PX,
    )
    gradient = cv2.Scharr(cyan, cv2.CV_32F, 1, 0, scale=1.0 / 32.0)
    gradient_threshold = _DORSAL_GRADIENT_THRESHOLD / resolution_scale
    offset = max(2, round(_DORSAL_SAMPLE_OFFSET_WIDTH_FRACTION * width))
    left = np.full_like(cyan, np.inf)
    right = np.full_like(cyan, -np.inf)
    left[:, offset:] = cyan[:, :-offset]
    right[:, :-offset] = cyan[:, offset:]

    support_near = max(
        offset + 1,
        round(_DORSAL_SUPPORT_NEAR_WIDTH_FRACTION * width),
    )
    support_far = max(
        support_near + 1,
        round(_DORSAL_SUPPORT_FAR_WIDTH_FRACTION * width),
    )
    cumulative = np.concatenate(
        (
            np.zeros((height, 1), dtype=np.float64),
            np.cumsum(cyan, axis=1, dtype=np.float64),
        ),
        axis=1,
    )
    broad_left = np.full_like(cyan, np.inf)
    broad_right = np.full_like(cyan, -np.inf)
    supported_columns = np.arange(support_far, width - support_far)
    broad_left[:, supported_columns] = (
        cumulative[:, supported_columns - support_near]
        - cumulative[:, supported_columns - support_far]
    ) / (support_far - support_near)
    broad_right[:, supported_columns] = (
        cumulative[:, supported_columns + support_far]
        - cumulative[:, supported_columns + support_near]
    ) / (support_far - support_near)
    candidate = np.asarray(
        (gradient > gradient_threshold)
        & (right > _DORSAL_CYAN_RIGHT_THRESHOLD)
        & (left < _DORSAL_CYAN_LEFT_THRESHOLD)
        & (maximum_green_blue > _DORSAL_MINIMUM_BRIGHTNESS_DN)
        & (broad_left < _DORSAL_BROAD_LEFT_THRESHOLD)
        & (broad_right > _DORSAL_BROAD_RIGHT_THRESHOLD),
        dtype=np.uint8,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            _odd_at_least(_DORSAL_CLOSE_KERNEL_WIDTH_FRACTION * width),
            _odd_at_least(_DORSAL_CLOSE_KERNEL_HEIGHT_FRACTION * height),
        ),
    )
    connected = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    grouping_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            _odd_at_least(_DORSAL_GROUP_KERNEL_WIDTH_FRACTION * width),
            _odd_at_least(_DORSAL_GROUP_KERNEL_HEIGHT_FRACTION * height),
        ),
    )
    grouped = cv2.dilate(connected, grouping_kernel)
    component_count, labels, _, _ = cv2.connectedComponentsWithStats(grouped)

    best_component: int | None = None
    best_score = -np.inf
    for component in range(1, component_count):
        source_rows, source_columns = np.nonzero(
            (labels == component) & (candidate > 0)
        )
        area = int(source_rows.size)
        if not area:
            continue
        component_width = int(source_columns.max() - source_columns.min() + 1)
        component_height = int(source_rows.max() - source_rows.min() + 1)
        if (
            component_height < _DORSAL_MINIMUM_COMPONENT_HEIGHT_FRACTION * height
            or area < _DORSAL_MINIMUM_COMPONENT_AREA_FRACTION * height * width
        ):
            continue
        edge_strength = gradient[source_rows, source_columns]
        mean_edge_strength = float(np.mean(np.maximum(edge_strength, 0.0)))
        width_penalty = 1.0 + 0.2 * component_width / component_height
        score = (
            component_height
            * np.sqrt(area)
            * mean_edge_strength
            / width_penalty
        )
        if score > best_score:
            best_score = score
            best_component = component
    if best_component is None:
        raise RuntimeError("no long dorsal cyan-transition component was found")

    selected = (labels == best_component) & (candidate > 0)
    active_rows = np.flatnonzero(np.any(selected, axis=1))
    y_start = int(active_rows[0])
    y_stop = int(active_rows[-1]) + 1
    rows = np.arange(y_start, y_stop)
    dorsal_x = np.full(rows.size, np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        columns = np.flatnonzero(selected[row])
        if columns.size:
            dorsal_x[index] = float(columns[np.argmax(gradient[row, columns])])
    present = np.flatnonzero(np.isfinite(dorsal_x))
    if present.size < 2:
        raise RuntimeError("dorsal transition has too few usable image rows")
    dorsal_x = np.interp(np.arange(rows.size), present, dorsal_x[present])
    dorsal_x = _rolling_median(
        dorsal_x,
        _DORSAL_MEDIAN_WINDOW_HEIGHT_FRACTION * height,
    )
    return np.column_stack((dorsal_x, rows.astype(np.float64)))


def _estimate_pad_width(cyan: np.ndarray, dorsal: np.ndarray) -> float:
    _, width = cyan.shape
    start_offset = max(2, round(_DORSAL_SAMPLE_OFFSET_WIDTH_FRACTION * width))
    run_length = max(2, round(_LOW_CYAN_RUN_WIDTH_FRACTION * width))
    widths = []
    for dorsal_x, y_coordinate in dorsal:
        row = cyan[int(y_coordinate)]
        start = min(width, int(round(dorsal_x)) + start_offset)
        low_cyan = np.asarray(row[start:] < _LOW_CYAN_THRESHOLD, dtype=np.uint8)
        if low_cyan.size < run_length:
            continue
        sustained = np.convolve(
            low_cyan,
            np.ones(run_length, dtype=np.uint8),
            mode="valid",
        )
        starts = np.flatnonzero(sustained == run_length)
        if starts.size:
            widths.append(start + int(starts[0]) - dorsal_x)

    if len(widths) >= max(8, round(0.15 * len(dorsal))):
        estimated = _PAD_WIDTH_INFLATION * float(np.median(widths))
    else:
        estimated = _FALLBACK_PAD_WIDTH_FRACTION * width
    return float(
        np.clip(
            estimated,
            _MINIMUM_PAD_WIDTH_FRACTION * width,
            _MAXIMUM_PAD_WIDTH_FRACTION * width,
        )
    )


def _robust_edge_normalize(edge: np.ndarray) -> np.ndarray:
    scale = float(np.percentile(edge, _EDGE_NORMALIZATION_PERCENTILE))
    if scale <= np.finfo(np.float32).eps:
        return np.zeros_like(edge, dtype=np.float32)
    return np.clip(edge / scale, 0.0, 1.0).astype(np.float32)


def _palmar_boundary(
    rgb: np.ndarray,
    cyan: np.ndarray,
    dorsal: np.ndarray,
    estimated_width: float,
) -> np.ndarray:
    height, width = cyan.shape
    scale = min(
        width / _REFERENCE_IMAGE_WIDTH_PX,
        height / _REFERENCE_IMAGE_HEIGHT_PX,
    )
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), max(0.5, _CYAN_SMOOTH_SIGMA_PX * scale))
    gray_edge = np.abs(cv2.Scharr(gray, cv2.CV_32F, 1, 0, scale=1.0 / 32.0))
    cyan_edge = np.abs(cv2.Scharr(cyan, cv2.CV_32F, 1, 0, scale=1.0 / 32.0))
    edge = (
        _GRAY_EDGE_WEIGHT * _robust_edge_normalize(gray_edge)
        + _CYAN_EDGE_WEIGHT * _robust_edge_normalize(cyan_edge)
    )

    minimum_delta = max(1, round(_PALMAR_MINIMUM_WIDTH_FACTOR * estimated_width))
    maximum_delta = max(
        minimum_delta + 1,
        round(_PALMAR_MAXIMUM_WIDTH_FACTOR * estimated_width),
    )
    deltas = np.arange(minimum_delta, maximum_delta + 1, dtype=np.int32)
    rows = dorsal[:, 1].astype(np.int32)
    dorsal_x = dorsal[:, 0]
    local_score = np.full((len(rows), len(deltas)), -np.inf, dtype=np.float64)
    width_prior = -_WIDTH_PRIOR_WEIGHT * (
        (deltas - estimated_width) / (_WIDTH_PRIOR_SCALE * estimated_width)
    ) ** 2
    for row_index, (row, boundary_x) in enumerate(zip(rows, dorsal_x, strict=True)):
        columns = np.rint(boundary_x + deltas).astype(np.int32)
        valid = (columns >= 0) & (columns < width)
        local_score[row_index, valid] = edge[row, columns[valid]] + width_prior[valid]

    maximum_step = max(
        1,
        round(_PALMAR_MAXIMUM_STEP_WIDTH_FRACTION * estimated_width),
    )
    dp = np.full_like(local_score, -np.inf)
    predecessor = np.full(local_score.shape, -1, dtype=np.int32)
    dp[0] = local_score[0]
    for row_index in range(1, len(rows)):
        for state in range(len(deltas)):
            if not np.isfinite(local_score[row_index, state]):
                continue
            previous_start = max(0, state - maximum_step)
            previous_stop = min(len(deltas), state + maximum_step + 1)
            previous_states = np.arange(previous_start, previous_stop)
            transition = dp[row_index - 1, previous_start:previous_stop] - (
                _PALMAR_SMOOTHNESS_PENALTY * np.abs(previous_states - state)
            )
            best_local = int(np.argmax(transition))
            if not np.isfinite(transition[best_local]):
                continue
            predecessor[row_index, state] = int(previous_states[best_local])
            dp[row_index, state] = (
                local_score[row_index, state] + transition[best_local]
            )

    final_state = int(np.argmax(dp[-1]))
    if not np.isfinite(dp[-1, final_state]):
        raise RuntimeError("palmar smooth path has no valid image-spanning solution")
    path = np.empty(len(rows), dtype=np.int32)
    path[-1] = final_state
    for row_index in range(len(rows) - 1, 0, -1):
        path[row_index - 1] = predecessor[row_index, path[row_index]]
        if path[row_index - 1] < 0:
            raise RuntimeError("palmar smooth-path backtracking failed")
    palmar_x = dorsal_x + deltas[path]
    return np.column_stack((palmar_x, rows.astype(np.float64)))


def _search_mask(
    image_shape: tuple[int, int],
    dorsal: np.ndarray,
    palmar: np.ndarray,
    estimated_width: float,
) -> np.ndarray:
    height, width = image_shape
    core_y = dorsal[:, 1].astype(np.int32)
    core_height = int(core_y[-1] - core_y[0] + 1)
    padding = max(1, round(_SEARCH_MASK_Y_PADDING_FRACTION * core_height))
    y_start = max(0, int(core_y[0]) - padding)
    y_stop = min(height, int(core_y[-1]) + padding + 1)
    rows = np.arange(y_start, y_stop)
    dorsal_x = np.interp(rows, core_y, dorsal[:, 0])
    palmar_x = np.interp(rows, core_y, palmar[:, 0])
    inset = max(1, round(_SEARCH_MASK_INSET_WIDTH_FRACTION * estimated_width))
    mask = np.zeros((height, width), dtype=bool)
    for row, left, right in zip(rows, dorsal_x, palmar_x, strict=True):
        start = max(0, int(np.ceil(left)) + inset)
        stop = min(width, int(np.floor(right)) - inset + 1)
        if start < stop:
            mask[row, start:stop] = True
    if not np.any(mask):
        raise RuntimeError("fingertip boundaries produced an empty search mask")
    return mask


def detect_fingertip_boundary(rgb: np.ndarray) -> FingertipBoundaryRegion:
    """Detect the bonded-left and palmar-right boundaries in a side-view RGB frame."""

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be an H x W x 3 uint8 array")
    cyan, maximum_green_blue = _cyan_geometry(image)
    dorsal = _dorsal_boundary(cyan, maximum_green_blue)
    estimated_width = _estimate_pad_width(cyan, dorsal)
    palmar = _palmar_boundary(image, cyan, dorsal, estimated_width)
    search_mask = _search_mask(image.shape[:2], dorsal, palmar, estimated_width)
    core_y_span = (int(dorsal[0, 1]), int(dorsal[-1, 1]) + 1)
    return FingertipBoundaryRegion(
        dorsal_boundary_xy_px=dorsal,
        palmar_boundary_xy_px=palmar,
        search_mask=search_mask,
        core_y_span=core_y_span,
        estimated_pad_width_px=estimated_width,
    )


__all__ = ["FingertipBoundaryRegion", "detect_fingertip_boundary"]
