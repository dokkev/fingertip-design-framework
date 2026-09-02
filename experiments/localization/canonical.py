"""Finger-relative image coordinates shared by offline and online observers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .fingertip_boundary import FingertipBoundaryRegion


@dataclass(frozen=True)
class CanonicalFingerMap:
    """Source-image coordinates sampled on a normalized finger rectangle."""

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
    def output_height(self) -> int:
        return self.map_x.shape[0]

    @property
    def output_width(self) -> int:
        return self.map_x.shape[1]


def build_canonical_finger_map(
    region: FingertipBoundaryRegion,
    *,
    output_height: int = 256,
    output_width: int = 128,
    transverse_inset_fraction: float = 0.04,
) -> CanonicalFingerMap:
    """Map normalized longitudinal/transverse samples into one finger image."""

    for name, value in (
        ("output_height", output_height),
        ("output_width", output_width),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 2:
            raise ValueError(f"{name} must be an integer of at least two")
    if not 0.0 <= transverse_inset_fraction < 0.5:
        raise ValueError("transverse_inset_fraction must be in [0, 0.5)")

    y_start, y_stop = region.core_y_span
    source_y = np.linspace(y_start, y_stop - 1, output_height, dtype=np.float64)
    boundary_y = region.dorsal_boundary_xy_px[:, 1]
    dorsal_x = np.interp(
        source_y,
        boundary_y,
        region.dorsal_boundary_xy_px[:, 0],
    )
    palmar_x = np.interp(
        source_y,
        boundary_y,
        region.palmar_boundary_xy_px[:, 0],
    )
    width = palmar_x - dorsal_x
    if np.any(width <= 0.0):
        raise RuntimeError("canonical finger boundaries cross")

    left = dorsal_x + transverse_inset_fraction * width
    right = palmar_x - transverse_inset_fraction * width
    transverse = np.linspace(0.0, 1.0, output_width, dtype=np.float64)
    map_x = left[:, None] + transverse[None, :] * (right - left)[:, None]
    map_y = np.broadcast_to(source_y[:, None], map_x.shape)
    return CanonicalFingerMap(map_x=map_x, map_y=map_y)


def warp_to_canonical(
    rgb: np.ndarray,
    canonical_map: CanonicalFingerMap,
) -> np.ndarray:
    """Sample an RGB image on a previously constructed canonical finger map."""

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


def similarity_from_landmarks(
    reference_xy: np.ndarray,
    current_xy: np.ndarray,
) -> np.ndarray:
    """Return the least-squares orientation-preserving 2-D similarity transform."""

    reference = np.asarray(reference_xy, dtype=np.float64)
    current = np.asarray(current_xy, dtype=np.float64)
    if (
        reference.ndim != 2
        or reference.shape[1:] != (2,)
        or current.shape != reference.shape
        or len(reference) < 2
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(current))
    ):
        raise ValueError("reference_xy and current_xy must be equal finite N x 2 arrays")

    reference_center = np.mean(reference, axis=0)
    current_center = np.mean(current, axis=0)
    reference_centered = reference - reference_center
    current_centered = current - current_center
    reference_energy = float(np.sum(reference_centered**2))
    if reference_energy <= np.finfo(np.float64).eps:
        raise ValueError("reference_xy does not define a spatial extent")

    left, singular_values, right_transpose = np.linalg.svd(
        reference_centered.T @ current_centered
    )
    row_rotation = left @ right_transpose
    if np.linalg.det(row_rotation) < 0.0:
        left[:, -1] *= -1.0
        row_rotation = left @ right_transpose
        singular_values[-1] *= -1.0
    scale = float(np.sum(singular_values) / reference_energy)
    linear = scale * row_rotation.T
    translation = current_center - linear @ reference_center
    transform = np.column_stack((linear, translation))
    if not np.all(np.isfinite(transform)) or scale <= 0.0:
        raise RuntimeError("landmarks produced an invalid similarity transform")
    return transform


def transform_canonical_map(
    reference_map: CanonicalFingerMap,
    reference_to_current_transform: np.ndarray,
) -> CanonicalFingerMap:
    """Move a reference sampling map with a reference-to-current image transform."""

    transform = np.asarray(reference_to_current_transform, dtype=np.float64)
    if transform.shape != (2, 3) or not np.all(np.isfinite(transform)):
        raise ValueError("reference_to_current_transform must be a finite 2 x 3 array")
    points = np.column_stack(
        (
            reference_map.map_x.ravel(),
            reference_map.map_y.ravel(),
            np.ones(reference_map.map_x.size),
        )
    )
    moved = points @ transform.T
    return CanonicalFingerMap(
        map_x=moved[:, 0].reshape(reference_map.map_x.shape),
        map_y=moved[:, 1].reshape(reference_map.map_y.shape),
    )


__all__ = [
    "CanonicalFingerMap",
    "build_canonical_finger_map",
    "similarity_from_landmarks",
    "transform_canonical_map",
    "warp_to_canonical",
]
