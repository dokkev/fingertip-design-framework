"""Fixed-session optical reference, calibration, and response measurements."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


CANONICAL_LONGITUDINAL_BINS = 256
CANONICAL_TRANSVERSE_BINS = 128
STORED_SIGNATURE_BINS = 128
DEFAULT_GREEN_EXCESS_THRESHOLD_DN = 8.0


@dataclass(frozen=True)
class FixedAnalysisCalibration:
    """One fixed image-space sampling strip derived from an unloaded reference."""

    reference_mask: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray

    def __post_init__(self) -> None:
        mask = np.asarray(self.reference_mask, dtype=bool)
        map_x = np.asarray(self.map_x, dtype=np.float32)
        map_y = np.asarray(self.map_y, dtype=np.float32)
        if mask.ndim != 2 or not np.any(mask):
            raise ValueError("reference_mask must be a nonempty 2-D array")
        if map_x.shape != map_y.shape or map_x.ndim != 2:
            raise ValueError("canonical maps must be equal 2-D arrays")
        object.__setattr__(self, "reference_mask", mask.copy())
        object.__setattr__(self, "map_x", map_x.copy())
        object.__setattr__(self, "map_y", map_y.copy())


def actual_force_magnitude(fx_n: float, fy_n: float, fz_n: float) -> float:
    """Return synchronized three-axis force magnitude in newtons."""

    return float(np.sqrt(fx_n * fx_n + fy_n * fy_n + fz_n * fz_n))


def unloaded_median_rgb(images: list[np.ndarray]) -> np.ndarray:
    """Return the rounded temporal pixelwise RGB median for one session."""

    if not images:
        raise ValueError("at least one unloaded image is required")
    stack = np.asarray(images)
    if stack.ndim != 4 or stack.shape[-1] != 3 or stack.dtype != np.uint8:
        raise ValueError("images must be equal H x W x 3 uint8 arrays")
    return np.rint(np.median(stack, axis=0)).astype(np.uint8)


def calibrate_analysis_strip(
    reference_rgb: np.ndarray,
    *,
    green_excess_threshold_dn: float = DEFAULT_GREEN_EXCESS_THRESHOLD_DN,
) -> FixedAnalysisCalibration:
    """Build one deterministic fixed strip from the dominant green-lit object."""

    image = np.asarray(reference_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("reference_rgb must be H x W x 3 uint8")
    if not np.isfinite(green_excess_threshold_dn) or green_excess_threshold_dn <= 0.0:
        raise ValueError("green_excess_threshold_dn must be finite and positive")
    red, green, blue = np.moveaxis(image.astype(np.float32), -1, 0)
    green_excess = green - 0.5 * (red + blue)
    mask = (green_excess > green_excess_threshold_dn).astype(np.uint8)
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
        hull_area = cv2.contourArea(cv2.convexHull(max(contours, key=cv2.contourArea)))
        solidity = area / max(float(hull_area), 1.0)
        candidates.append((area * solidity, label))
    if not candidates:
        raise RuntimeError("unloaded reference has no green-lit fingertip component")
    selected = max(candidates)[1]
    fixed_mask = labels == selected
    fixed_mask = _fill_mask(fixed_mask)
    map_x, map_y = _canonical_maps_from_mask(fixed_mask)
    return FixedAnalysisCalibration(fixed_mask, map_x, map_y)


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
    bins = max(2, int(np.ceil(s_max - s_min)) + 1)
    indices = np.clip(
        np.floor((s - s_min) * (bins - 1) / (s_max - s_min)).astype(np.int32),
        0,
        bins - 1,
    )
    t_min = np.full(bins, np.inf)
    t_max = np.full(bins, -np.inf)
    np.minimum.at(t_min, indices, t)
    np.maximum.at(t_max, indices, t)
    valid = np.isfinite(t_min) & np.isfinite(t_max)
    bin_s = np.linspace(s_min, s_max, bins)
    query_s = np.linspace(s_min, s_max, CANONICAL_LONGITUDINAL_BINS)
    lower = np.interp(query_s, bin_s[valid], t_min[valid])
    upper = np.interp(query_s, bin_s[valid], t_max[valid])
    u = np.linspace(0.0, 1.0, CANONICAL_TRANSVERSE_BINS)
    query_t = lower[:, None] * (1.0 - u) + upper[:, None] * u
    points_xy = (
        centroid[None, None, :]
        + query_s[:, None, None] * longitudinal[None, None, :]
        + query_t[:, :, None] * transverse[None, None, :]
    )
    return points_xy[:, :, 0].astype(np.float32), points_xy[:, :, 1].astype(np.float32)


def warp_rgb(rgb: np.ndarray, calibration: FixedAnalysisCalibration) -> np.ndarray:
    image = np.asarray(rgb)
    if image.shape != (*calibration.reference_mask.shape, 3) or image.dtype != np.uint8:
        raise ValueError("rgb must match the calibrated image shape")
    return cv2.remap(
        image,
        calibration.map_x,
        calibration.map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def fixed_channel_differences(
    unloaded_rgb: np.ndarray,
    loaded_rgbs: np.ndarray,
    calibration: FixedAnalysisCalibration,
    *,
    channel_index: int,
) -> np.ndarray:
    """Return signed canonical differences for one RGB channel."""

    loaded = np.asarray(loaded_rgbs)
    if loaded.ndim != 4 or loaded.shape[1:] != unloaded_rgb.shape:
        raise ValueError("loaded_rgbs must have shape N x H x W x 3")
    if channel_index not in range(3):
        raise ValueError("channel_index must be 0, 1, or 2")
    reference = warp_rgb(unloaded_rgb, calibration)[:, :, channel_index].astype(
        np.float32
    )
    return np.asarray(
        [
            warp_rgb(frame, calibration)[:, :, channel_index].astype(np.float32)
            - reference
            for frame in loaded
        ],
        dtype=np.float32,
    )


def mean_absolute_response(differences: np.ndarray) -> np.ndarray:
    """Return mean absolute response for each image in an N x H x W stack."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 3 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("differences must be a finite nonempty N x H x W array")
    return np.mean(np.abs(values), axis=(1, 2))


def optical_metrics(
    delta_rgb: np.ndarray, loaded_canonical_rgb: np.ndarray
) -> dict[str, float]:
    """Return raw per-channel difference and saturation summaries."""

    delta = np.asarray(delta_rgb, dtype=np.float64)
    loaded = np.asarray(loaded_canonical_rgb)
    if delta.ndim != 3 or delta.shape[-1] != 3 or loaded.shape != delta.shape:
        raise ValueError("delta_rgb and loaded_canonical_rgb must be equal H x W x 3")
    values: dict[str, float] = {}
    for index, channel in enumerate("RGB"):
        difference = delta[:, :, index]
        current = loaded[:, :, index]
        values[f"optical_mae_{channel}_dn"] = float(np.mean(np.abs(difference)))
        values[f"optical_rms_{channel}_dn"] = float(np.sqrt(np.mean(difference**2)))
        values[f"optical_signed_mean_{channel}_dn"] = float(np.mean(difference))
        values[f"image_mean_{channel}_dn"] = float(np.mean(current))
        values[f"saturation_ge250_{channel}_fraction"] = float(np.mean(current >= 250))
        values[f"saturation_eq255_{channel}_fraction"] = float(np.mean(current == 255))
    return values


def longitudinal_signature(
    delta_green: np.ndarray, bins: int = STORED_SIGNATURE_BINS
) -> np.ndarray:
    """Return an unnormalized signed transverse-mean Delta-G profile."""

    values = np.asarray(delta_green, dtype=np.float64)
    if values.ndim != 2 or bins < 2 or not np.all(np.isfinite(values)):
        raise ValueError("delta_green must be finite 2-D and bins must be >= 2")
    profile = values.mean(axis=1)
    source = np.linspace(0.0, 1.0, len(profile))
    target = np.linspace(0.0, 1.0, bins)
    return np.interp(target, source, profile)


def pairwise_signature_distances(signatures: np.ndarray) -> np.ndarray:
    values = np.asarray(signatures, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("signatures must be a finite N x bins array with N >= 2")
    differences = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.mean(differences**2, axis=2))


__all__ = [
    "CANONICAL_LONGITUDINAL_BINS",
    "CANONICAL_TRANSVERSE_BINS",
    "DEFAULT_GREEN_EXCESS_THRESHOLD_DN",
    "FixedAnalysisCalibration",
    "STORED_SIGNATURE_BINS",
    "actual_force_magnitude",
    "calibrate_analysis_strip",
    "fixed_channel_differences",
    "longitudinal_signature",
    "mean_absolute_response",
    "optical_metrics",
    "pairwise_signature_distances",
    "unloaded_median_rgb",
    "warp_rgb",
]
