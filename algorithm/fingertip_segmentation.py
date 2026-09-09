"""Deterministic geometry-only segmentation of the visible LUMO fingertip."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


_MINIMUM_IMAGE_EXTENT_PX = 32
_MINIMUM_COMPONENT_AREA_FRACTION = 0.001
_MINIMUM_ROW_WIDTH_FRACTION = 0.35


@dataclass(frozen=True)
class FingertipRegion:
    """Visible finger silhouette and its image-row side boundaries.

    Boundary points are ordered by increasing image ``y``. ``dorsal`` is the
    smaller image-``x`` boundary and ``palmar`` is the larger image-``x``
    boundary. The current side-view mounting therefore assumes the distal end
    is toward smaller image ``y``; a 180-degree camera inversion is outside the
    stated hardware convention.
    """

    mask: np.ndarray
    dorsal_boundary_xy_px: np.ndarray
    palmar_boundary_xy_px: np.ndarray
    search_mask: np.ndarray
    longitudinal_span: tuple[int, int]
    estimated_width_px: float

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        search = np.asarray(self.search_mask, dtype=bool)
        dorsal = np.asarray(self.dorsal_boundary_xy_px, dtype=np.float64)
        palmar = np.asarray(self.palmar_boundary_xy_px, dtype=np.float64)
        if mask.ndim != 2 or not np.any(mask):
            raise ValueError("mask must be a nonempty H x W array")
        if search.shape != mask.shape or not np.any(search) or np.any(search & ~mask):
            raise ValueError("search_mask must be a nonempty subset of mask")
        if (
            dorsal.ndim != 2
            or dorsal.shape[1:] != (2,)
            or palmar.shape != dorsal.shape
            or len(dorsal) < 2
            or not np.all(np.isfinite(dorsal))
            or not np.all(np.isfinite(palmar))
        ):
            raise ValueError("boundaries must be equal finite N x 2 arrays")
        if not np.array_equal(dorsal[:, 1], palmar[:, 1]):
            raise ValueError("dorsal and palmar boundaries must share image rows")
        if not np.all(np.diff(dorsal[:, 1]) > 0.0):
            raise ValueError("boundary rows must be strictly increasing")
        if np.any(dorsal[:, 0] >= palmar[:, 0]):
            raise ValueError("dorsal boundary must remain left of palmar boundary")
        start, stop = self.longitudinal_span
        if not 0 <= start < stop <= mask.shape[0]:
            raise ValueError("longitudinal_span must be a valid half-open row interval")
        if not np.isfinite(self.estimated_width_px) or self.estimated_width_px <= 0.0:
            raise ValueError("estimated_width_px must be finite and positive")

        mask = mask.copy()
        search = search.copy()
        dorsal = dorsal.copy()
        palmar = palmar.copy()
        for array in (mask, search, dorsal, palmar):
            array.setflags(write=False)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "search_mask", search)
        object.__setattr__(self, "dorsal_boundary_xy_px", dorsal)
        object.__setattr__(self, "palmar_boundary_xy_px", palmar)


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be an H x W x 3 uint8 array")
    if min(image.shape[:2]) < _MINIMUM_IMAGE_EXTENT_PX:
        raise ValueError("rgb is too small for fingertip segmentation")
    return image


def _odd_size(value: float, minimum: int = 3) -> int:
    size = max(minimum, int(round(value)))
    return size if size % 2 else size + 1


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    exterior = (1 - padded).copy()
    cv2.floodFill(exterior, None, (0, 0), 2)
    holes = exterior == 1
    return (padded.astype(bool) | holes)[1:-1, 1:-1]


def _principal_frame(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = np.nonzero(mask)
    if len(rows) < 3:
        raise RuntimeError("fingertip core contains too few pixels")
    points = np.column_stack((columns, rows)).astype(np.float64)
    center = np.mean(points, axis=0)
    _, eigenvectors = np.linalg.eigh(np.cov(points.T))
    longitudinal = eigenvectors[:, -1]
    if longitudinal[1] < 0.0:
        longitudinal = -longitudinal
    transverse = np.asarray((-longitudinal[1], longitudinal[0]))
    return center, longitudinal, transverse


def _select_emissive_fingertip_component(
    candidate: np.ndarray,
    green_blue: np.ndarray,
) -> np.ndarray:
    """Select the bright straight fingertip corridor, not illuminated fixtures."""

    height, width = candidate.shape
    candidate_values = green_blue[candidate]
    if candidate_values.size < 24:
        raise RuntimeError("cyan geometry contains insufficient emissive support")
    core_threshold = float(np.percentile(candidate_values, 70.0))
    core = candidate & (green_blue >= core_threshold)
    core_close = _odd_size(0.012 * min(height, width))
    core = cv2.morphologyEx(
        core.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (core_close, core_close),
        ),
    ).astype(bool)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8))
    minimum_area = max(24, round(0.0002 * height * width))
    best: tuple[float, int] | None = None
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        component_mask = labels == component
        score = np.sqrt(area) * float(np.mean(green_blue[component_mask]))
        if best is None or score > best[0]:
            best = (score, component)
    if best is None:
        raise RuntimeError("cyan geometry contains no emissive fingertip core")

    selected_core = labels == best[1]
    center, longitudinal, transverse = _principal_frame(selected_core)
    rows, columns = np.nonzero(selected_core)
    centered = np.column_stack((columns, rows)).astype(np.float64) - center
    longitudinal_coordinates = centered @ longitudinal
    transverse_coordinates = centered @ transverse
    longitudinal_start, longitudinal_stop = np.percentile(
        longitudinal_coordinates,
        (2.0, 98.0),
    )
    longitudinal_length = float(longitudinal_stop - longitudinal_start)
    transverse_center = float(np.median(transverse_coordinates))
    transverse_width = float(
        np.percentile(transverse_coordinates, 80.0)
        - np.percentile(transverse_coordinates, 20.0)
    )
    if longitudinal_length <= 0.0 or transverse_width <= 0.0:
        raise RuntimeError("emissive fingertip core has invalid oriented extent")

    # The high-intensity core identifies the physical finger even when a dimmer
    # carrier or fin array is cyan.  Grow only an oriented corridor around that
    # core, then recover the lower-intensity silicone inside the same corridor.
    half_width = max(0.10 * longitudinal_length, 0.90 * transverse_width)
    longitudinal_margin = 0.35 * longitudinal_length
    grid_y, grid_x = np.indices(candidate.shape)
    grid = np.stack((grid_x - center[0], grid_y - center[1]), axis=-1)
    grid_longitudinal = grid @ longitudinal
    grid_transverse = grid @ transverse
    corridor = (
        (grid_longitudinal >= longitudinal_start - longitudinal_margin)
        & (grid_longitudinal <= longitudinal_stop + longitudinal_margin)
        & (np.abs(grid_transverse - transverse_center) <= half_width)
    )
    restricted = candidate & corridor
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        restricted.astype(np.uint8)
    )
    selected_component: tuple[int, int, int] | None = None
    for component in range(1, count):
        overlap = int(np.count_nonzero((labels == component) & selected_core))
        area = int(stats[component, cv2.CC_STAT_AREA])
        candidate_score = (overlap, area, component)
        if selected_component is None or candidate_score > selected_component:
            selected_component = candidate_score
    if selected_component is None or selected_component[0] == 0:
        raise RuntimeError("emissive core does not overlap a fingertip component")
    return labels == selected_component[2]


def segment_fingertip(rgb: np.ndarray) -> FingertipRegion:
    """Return a deterministic emissive-fingertip silhouette from one RGB frame.

    Color is used only to localize geometry. White reflected structures have
    near-zero cyan chromaticity and are excluded before the smooth filled
    component prior is applied. No contact response is computed here.
    """

    image = _validate_rgb(rgb)
    height, width = image.shape[:2]
    values = image.astype(np.float32)
    red, green, blue = np.moveaxis(values, -1, 0)
    cyan = (0.5 * (green + blue) - red) / (red + green + blue + 1.0)
    cyan_positive = cyan[cyan > 0.0]
    if cyan_positive.size < max(24, round(0.0005 * height * width)):
        raise RuntimeError("image contains insufficient cyan fingertip evidence")
    cyan_threshold = max(0.025, float(np.percentile(cyan_positive, 45.0)))
    cyan_support = cyan >= cyan_threshold
    green_blue = np.minimum(green, blue)
    brightness_threshold = max(
        25.0,
        float(np.percentile(green_blue[cyan_support], 12.0)),
    )
    candidate = cyan_support & (green_blue >= brightness_threshold)

    scale = min(height, width)
    open_size = _odd_size(0.006 * scale)
    close_size = _odd_size(0.018 * scale)
    candidate = cv2.morphologyEx(
        candidate.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    ).astype(bool)
    component = _fill_holes(
        _select_emissive_fingertip_component(candidate, green_blue)
    )

    contours, _ = cv2.findContours(
        component.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 1)
    mask = mask.astype(bool)

    rows = np.flatnonzero(np.count_nonzero(mask, axis=1) >= 2)
    if rows.size < 2:
        raise RuntimeError("fingertip silhouette has insufficient longitudinal support")
    left = np.empty(rows.size, dtype=np.float64)
    right = np.empty(rows.size, dtype=np.float64)
    for index, row in enumerate(rows):
        columns = np.flatnonzero(mask[row])
        left[index] = columns[0]
        right[index] = columns[-1]
    widths = right - left + 1.0
    center, _, transverse = _principal_frame(mask)
    mask_rows, mask_columns = np.nonzero(mask)
    transverse_coordinates = (
        np.column_stack((mask_columns, mask_rows)).astype(np.float64) - center
    ) @ transverse
    median_width = float(
        np.percentile(transverse_coordinates, 95.0)
        - np.percentile(transverse_coordinates, 5.0)
    )
    stable = widths >= _MINIMUM_ROW_WIDTH_FRACTION * median_width
    if np.count_nonzero(stable) < 2:
        raise RuntimeError("fingertip silhouette has no stable-width rows")

    erosion = _odd_size(0.035 * median_width)
    search = cv2.erode(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion, erosion)),
    ).astype(bool)
    if not np.any(search):
        raise RuntimeError("fingertip erosion produced an empty LED search mask")
    return FingertipRegion(
        mask=mask,
        dorsal_boundary_xy_px=np.column_stack((left, rows)),
        palmar_boundary_xy_px=np.column_stack((right, rows)),
        search_mask=search,
        longitudinal_span=(int(rows[0]), int(rows[-1]) + 1),
        estimated_width_px=median_width,
    )


__all__ = ["FingertipRegion", "segment_fingertip"]
