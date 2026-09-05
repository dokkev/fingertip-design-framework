"""Fixed-session image geometry and compact longitudinal optical profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROFILE_BINS = 128
CANONICAL_LONGITUDINAL_BINS = 256
CANONICAL_TRANSVERSE_BINS = 128
GREEN_EXCESS_THRESHOLD_DN = 8.0
INTERIOR_MARGIN_PX = 12.0


@dataclass(frozen=True)
class OpticalStrip:
    """One unloaded-derived sampling strip reused by every loaded frame."""

    source_mask: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray
    support_mask: np.ndarray

    def __post_init__(self) -> None:
        source = np.asarray(self.source_mask, dtype=bool)
        map_x = np.asarray(self.map_x, dtype=np.float32)
        map_y = np.asarray(self.map_y, dtype=np.float32)
        support = np.asarray(self.support_mask, dtype=bool)
        if source.ndim != 2 or not np.any(source):
            raise ValueError("source_mask must be a nonempty 2-D array")
        if map_x.ndim != 2 or map_x.shape != map_y.shape:
            raise ValueError("map_x and map_y must be equal 2-D arrays")
        if support.shape != map_x.shape or not np.any(support):
            raise ValueError("support_mask must be nonempty and match the maps")
        if np.count_nonzero(np.any(support, axis=1)) < 2:
            raise ValueError("support_mask must cover at least two longitudinal rows")
        object.__setattr__(self, "source_mask", source.copy())
        object.__setattr__(self, "map_x", map_x.copy())
        object.__setattr__(self, "map_y", map_y.copy())
        object.__setattr__(self, "support_mask", support.copy())


def load_rgb(path: str | Path) -> np.ndarray:
    """Decode one stored OpenCV PNG as owned RGB8."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def temporal_median_rgb(images: list[np.ndarray]) -> np.ndarray:
    """Return the rounded pixelwise median of equal RGB8 images."""

    if not images:
        raise ValueError("at least one image is required")
    stack = np.asarray(images)
    if stack.ndim != 4 or stack.shape[-1] != 3 or stack.dtype != np.uint8:
        raise ValueError("images must be equal H x W x 3 uint8 arrays")
    return np.rint(np.median(stack, axis=0)).astype(np.uint8)


def calibrate_optical_strip(
    reference_rgb: np.ndarray,
    *,
    green_excess_threshold_dn: float = GREEN_EXCESS_THRESHOLD_DN,
    interior_margin_px: float = INTERIOR_MARGIN_PX,
) -> OpticalStrip:
    """Locate one green-lit region and construct a fixed interior strip."""

    image = np.asarray(reference_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("reference_rgb must be H x W x 3 uint8")
    if not np.isfinite(green_excess_threshold_dn) or green_excess_threshold_dn <= 0:
        raise ValueError("green_excess_threshold_dn must be finite and positive")
    if not np.isfinite(interior_margin_px) or interior_margin_px <= 0:
        raise ValueError("interior_margin_px must be finite and positive")

    red, green, blue = np.moveaxis(image.astype(np.float32), -1, 0)
    mask = (green - 0.5 * (red + blue) > green_excess_threshold_dn).astype(
        np.uint8
    )
    scale = max(3, int(round(min(image.shape[:2]) / 100.0)))
    if scale % 2 == 0:
        scale += 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((scale, scale), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates: list[tuple[float, int]] = []
    minimum_area = 0.002 * image.shape[0] * image.shape[1]
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        component = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(max(contours, key=cv2.contourArea)))
        solidity = area / max(float(hull_area), 1.0)
        candidates.append((area * solidity, label))
    if not candidates:
        raise RuntimeError("reference has no usable green-lit fingertip region")

    source_mask = _fill_mask(labels == max(candidates)[1])
    map_x, map_y = _canonical_maps_from_mask(source_mask)
    distance_px = cv2.distanceTransform(
        source_mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    canonical_distance_px = cv2.remap(
        distance_px,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    support = canonical_distance_px >= interior_margin_px
    return OpticalStrip(source_mask, map_x, map_y, support)


def strip_centroid(strip: OpticalStrip) -> np.ndarray:
    """Return the source-mask centroid as image x,y coordinates."""

    y, x = np.nonzero(strip.source_mask)
    return np.asarray((np.mean(x), np.mean(y)), dtype=np.float64)


def strip_geometry(strip: OpticalStrip) -> dict[str, float]:
    """Return simple image-space pose diagnostics for one calibrated strip."""

    y, x = np.nonzero(strip.source_mask)
    points = np.column_stack((x, y)).astype(np.float64)
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, vectors = np.linalg.eigh(np.cov(centered.T))
    longitudinal = vectors[:, 1]
    angle_deg = float(np.degrees(np.arctan2(longitudinal[1], longitudinal[0])))
    if angle_deg >= 90.0:
        angle_deg -= 180.0
    elif angle_deg < -90.0:
        angle_deg += 180.0
    transverse = np.asarray((-longitudinal[1], longitudinal[0]))
    longitudinal_coordinate = centered @ longitudinal
    transverse_coordinate = centered @ transverse
    return {
        "optical_region_centroid_x_px": float(centroid[0]),
        "optical_region_centroid_y_px": float(centroid[1]),
        "optical_region_orientation_deg": angle_deg,
        "optical_region_longitudinal_extent_px": float(
            np.ptp(longitudinal_coordinate)
        ),
        "optical_region_transverse_extent_px": float(
            np.ptp(transverse_coordinate)
        ),
        "optical_region_area_px": float(len(points)),
    }


def warp_rgb(rgb: np.ndarray, strip: OpticalStrip) -> np.ndarray:
    """Sample one RGB frame through the fixed canonical strip."""

    image = np.asarray(rgb)
    if image.shape != (*strip.source_mask.shape, 3) or image.dtype != np.uint8:
        raise ValueError("rgb must match the calibrated RGB8 image shape")
    return cv2.remap(
        image,
        strip.map_x,
        strip.map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def longitudinal_green_profile(
    rgb: np.ndarray,
    strip: OpticalStrip,
    *,
    bins: int = PROFILE_BINS,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return raw Green DN versus longitudinal coordinate plus camera QC."""

    canonical = warp_rgb(rgb, strip)
    support = strip.support_mask
    green = canonical[:, :, 1].astype(np.float64)
    counts = np.count_nonzero(support, axis=1)
    valid = counts > 0
    if np.count_nonzero(valid) < 2 or bins < 2:
        raise ValueError("the strip must provide at least two supported rows and bins")
    row_profile = np.sum(green * support, axis=1)[valid] / counts[valid]
    coordinate = np.linspace(0.0, 1.0, len(green))[valid]
    coordinate = (coordinate - coordinate[0]) / (coordinate[-1] - coordinate[0])
    profile = np.interp(np.linspace(0.0, 1.0, bins), coordinate, row_profile)

    qc: dict[str, float] = {
        "optical_support_pixel_count": float(np.count_nonzero(support)),
        "optical_support_fraction": float(np.mean(support)),
    }
    for channel_index, channel in enumerate("RGB"):
        values = canonical[:, :, channel_index][support]
        qc[f"image_mean_{channel}_dn"] = float(np.mean(values))
        qc[f"saturation_ge250_{channel}_fraction"] = float(np.mean(values >= 250))
        qc[f"saturation_eq255_{channel}_fraction"] = float(np.mean(values == 255))
    return profile.astype(np.float64), qc


def rms_profile_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return RMS distance between equal finite one-dimensional profiles."""

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not np.all(np.isfinite(a + b)):
        raise ValueError("profiles must be equal finite one-dimensional arrays")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _fill_mask(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    return filled.astype(bool)


def _canonical_maps_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.nonzero(mask)
    points = np.column_stack((x, y)).astype(np.float64)
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, vectors = np.linalg.eigh(np.cov(centered.T))
    longitudinal = vectors[:, 1]
    if longitudinal[np.argmax(np.abs(longitudinal))] < 0.0:
        longitudinal *= -1.0
    transverse = np.asarray((-longitudinal[1], longitudinal[0]))
    s = centered @ longitudinal
    t = centered @ transverse
    s_min, s_max = float(s.min()), float(s.max())
    source_bins = max(2, int(np.ceil(s_max - s_min)) + 1)
    indices = np.clip(
        np.floor((s - s_min) * (source_bins - 1) / (s_max - s_min)).astype(
            np.int32
        ),
        0,
        source_bins - 1,
    )
    lower = np.full(source_bins, np.inf)
    upper = np.full(source_bins, -np.inf)
    np.minimum.at(lower, indices, t)
    np.maximum.at(upper, indices, t)
    valid = np.isfinite(lower) & np.isfinite(upper)
    bin_s = np.linspace(s_min, s_max, source_bins)
    query_s = np.linspace(s_min, s_max, CANONICAL_LONGITUDINAL_BINS)
    lower_query = np.interp(query_s, bin_s[valid], lower[valid])
    upper_query = np.interp(query_s, bin_s[valid], upper[valid])
    transverse_coordinate = np.linspace(0.0, 1.0, CANONICAL_TRANSVERSE_BINS)
    query_t = (
        lower_query[:, None] * (1.0 - transverse_coordinate)
        + upper_query[:, None] * transverse_coordinate
    )
    points_xy = (
        centroid[None, None, :]
        + query_s[:, None, None] * longitudinal[None, None, :]
        + query_t[:, :, None] * transverse[None, None, :]
    )
    return points_xy[:, :, 0].astype(np.float32), points_xy[:, :, 1].astype(
        np.float32
    )


__all__ = [
    "GREEN_EXCESS_THRESHOLD_DN",
    "INTERIOR_MARGIN_PX",
    "OpticalStrip",
    "PROFILE_BINS",
    "calibrate_optical_strip",
    "load_rgb",
    "longitudinal_green_profile",
    "rms_profile_distance",
    "strip_centroid",
    "strip_geometry",
    "temporal_median_rgb",
    "warp_rgb",
]
