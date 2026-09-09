"""Finger-relative sampling maps and image-space similarity transforms."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .fingertip_segmentation import FingertipRegion


@dataclass(frozen=True)
class CanonicalFingerConfig:
    """Resolution and fixed interior inset of the canonical finger image."""

    longitudinal_samples: int = 256
    transverse_samples: int = 128
    transverse_inset_fraction: float = 0.04

    def __post_init__(self) -> None:
        if (
            not isinstance(self.longitudinal_samples, int)
            or isinstance(self.longitudinal_samples, bool)
            or self.longitudinal_samples < 2
        ):
            raise ValueError("longitudinal_samples must be an integer >= 2")
        if (
            not isinstance(self.transverse_samples, int)
            or isinstance(self.transverse_samples, bool)
            or self.transverse_samples < 2
        ):
            raise ValueError("transverse_samples must be an integer >= 2")
        if not 0.0 <= self.transverse_inset_fraction < 0.5:
            raise ValueError("transverse_inset_fraction must be in [0, 0.5)")


@dataclass(frozen=True)
class CanonicalFingerMap:
    """Source-image coordinates for a normalized finger rectangle."""

    map_x: np.ndarray
    map_y: np.ndarray

    def __post_init__(self) -> None:
        map_x = np.asarray(self.map_x, dtype=np.float32)
        map_y = np.asarray(self.map_y, dtype=np.float32)
        if (
            map_x.ndim != 2
            or map_y.shape != map_x.shape
            or min(map_x.shape) < 2
            or not np.all(np.isfinite(map_x))
            or not np.all(np.isfinite(map_y))
        ):
            raise ValueError("map_x and map_y must be equal finite 2-D arrays")
        map_x = map_x.copy()
        map_y = map_y.copy()
        map_x.setflags(write=False)
        map_y.setflags(write=False)
        object.__setattr__(self, "map_x", map_x)
        object.__setattr__(self, "map_y", map_y)

    @property
    def shape(self) -> tuple[int, int]:
        return self.map_x.shape


def build_canonical_map(
    fingertip: FingertipRegion,
    config: CanonicalFingerConfig = CanonicalFingerConfig(),
) -> CanonicalFingerMap:
    """Build one full-silhouette finger-relative sampling map."""

    if not isinstance(fingertip, FingertipRegion):
        raise TypeError("fingertip must be a FingertipRegion")
    rows, columns = np.nonzero(fingertip.mask)
    points = np.column_stack((columns, rows)).astype(np.float64)
    center = np.mean(points, axis=0)
    _, eigenvectors = np.linalg.eigh(np.cov(points.T))
    longitudinal_axis = eigenvectors[:, -1]
    if longitudinal_axis[1] < 0.0:
        longitudinal_axis = -longitudinal_axis
    transverse_axis = np.asarray((-longitudinal_axis[1], longitudinal_axis[0]))
    centered = points - center
    longitudinal_coordinates = centered @ longitudinal_axis
    transverse_coordinates = centered @ transverse_axis
    minimum_longitudinal = float(np.min(longitudinal_coordinates))
    maximum_longitudinal = float(np.max(longitudinal_coordinates))
    source_longitudinal = np.linspace(
        minimum_longitudinal,
        maximum_longitudinal,
        config.longitudinal_samples,
        dtype=np.float64,
    )
    half_window = max(
        1.5,
        1.5
        * (maximum_longitudinal - minimum_longitudinal)
        / (config.longitudinal_samples - 1),
    )
    left = np.full(config.longitudinal_samples, np.nan, dtype=np.float64)
    right = np.full(config.longitudinal_samples, np.nan, dtype=np.float64)
    for index, coordinate in enumerate(source_longitudinal):
        support = np.abs(longitudinal_coordinates - coordinate) <= half_window
        if np.count_nonzero(support) >= 2:
            left[index], right[index] = np.percentile(
                transverse_coordinates[support],
                (5.0, 95.0),
            )
    valid = np.isfinite(left) & np.isfinite(right) & (right > left)
    if np.count_nonzero(valid) < 2:
        raise RuntimeError("fingertip mask has insufficient oriented cross-sections")
    sample_indices = np.arange(config.longitudinal_samples)
    left = np.interp(sample_indices, sample_indices[valid], left[valid])
    right = np.interp(sample_indices, sample_indices[valid], right[valid])
    left = cv2.GaussianBlur(left[:, None], (1, 0), 1.5).ravel()
    right = cv2.GaussianBlur(right[:, None], (1, 0), 1.5).ravel()
    width = right - left
    if np.any(width <= 0.0):
        raise RuntimeError("fingertip oriented boundaries cross")
    left = left + config.transverse_inset_fraction * width
    right = right - config.transverse_inset_fraction * width
    transverse = np.linspace(
        0.0,
        1.0,
        config.transverse_samples,
        dtype=np.float64,
    )
    source_transverse = (
        left[:, None] + transverse[None, :] * (right - left)[:, None]
    )
    points = (
        center[None, None, :]
        + source_longitudinal[:, None, None] * longitudinal_axis[None, None, :]
        + source_transverse[:, :, None] * transverse_axis[None, None, :]
    )
    map_x = points[:, :, 0]
    map_y = points[:, :, 1]
    return CanonicalFingerMap(map_x=map_x, map_y=map_y)


def transform_canonical_map(
    reference_map: CanonicalFingerMap,
    reference_to_current_transform: np.ndarray,
) -> CanonicalFingerMap:
    """Move a reference sampling map into the current image frame."""

    transform = np.asarray(reference_to_current_transform, dtype=np.float64)
    if transform.shape != (2, 3) or not np.all(np.isfinite(transform)):
        raise ValueError("reference_to_current_transform must be finite 2 x 3")
    homogeneous = np.column_stack(
        (
            reference_map.map_x.ravel(),
            reference_map.map_y.ravel(),
            np.ones(reference_map.map_x.size),
        )
    )
    moved = homogeneous @ transform.T
    return CanonicalFingerMap(
        map_x=moved[:, 0].reshape(reference_map.shape),
        map_y=moved[:, 1].reshape(reference_map.shape),
    )


def warp_to_canonical(
    rgb: np.ndarray,
    canonical_map: CanonicalFingerMap,
) -> np.ndarray:
    """Sample one RGB image through a canonical map."""

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be an H x W x 3 uint8 array")
    return cv2.remap(
        image,
        canonical_map.map_x,
        canonical_map.map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def landmark_longitudinal_coordinates(
    canonical_map: CanonicalFingerMap,
    landmarks_xy_px: np.ndarray,
) -> np.ndarray:
    """Return normalized canonical row coordinates nearest image landmarks."""

    landmarks = np.asarray(landmarks_xy_px, dtype=np.float64)
    if (
        landmarks.ndim != 2
        or landmarks.shape[1:] != (2,)
        or not len(landmarks)
        or not np.all(np.isfinite(landmarks))
    ):
        raise ValueError("landmarks_xy_px must be a finite N x 2 array")
    rows = []
    for landmark in landmarks:
        distance_squared = (
            (canonical_map.map_x - landmark[0]) ** 2
            + (canonical_map.map_y - landmark[1]) ** 2
        )
        row, _ = np.unravel_index(
            int(np.argmin(distance_squared)),
            distance_squared.shape,
        )
        rows.append(row)
    coordinates = np.asarray(rows, dtype=np.float64) / (canonical_map.shape[0] - 1)
    if not np.all(np.diff(coordinates) > 0.0):
        raise RuntimeError("LED landmarks do not preserve distal-to-proximal order")
    return coordinates


__all__ = [
    "CanonicalFingerConfig",
    "CanonicalFingerMap",
    "build_canonical_map",
    "landmark_longitudinal_coordinates",
    "transform_canonical_map",
    "warp_to_canonical",
]
