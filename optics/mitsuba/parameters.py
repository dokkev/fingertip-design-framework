"""Camera and numerical controls for optional Mitsuba rendering."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np

from mesh import PadMesh
from model.fingertip import Fingertip


def _finite_vector3(
    value: tuple[float, float, float],
    *,
    name: str,
) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values")
    return vector


@dataclass(frozen=True)
class Camera:
    """Fixed diagnostic camera framing for one optical render session."""

    position_mm: tuple[float, float, float]
    target_mm: tuple[float, float, float]
    up: tuple[float, float, float]
    resolution_px: tuple[int, int] = (640, 640)
    projection: Literal["orthographic", "perspective"] = "orthographic"
    orthographic_scale_mm: float | None = None
    fov_deg: float = 45.0

    def __post_init__(self) -> None:
        position = _finite_vector3(self.position_mm, name="position_mm")
        target = _finite_vector3(self.target_mm, name="target_mm")
        up = _finite_vector3(self.up, name="up")
        view = target - position
        if np.linalg.norm(view) <= 0.0:
            raise ValueError("camera position and target must differ")
        if np.linalg.norm(up) <= 0.0:
            raise ValueError("camera up vector must be nonzero")
        if np.linalg.norm(np.cross(view, up)) <= 1.0e-12:
            raise ValueError("camera up vector must not be parallel to its view")
        if (
            not isinstance(self.resolution_px, tuple)
            or len(self.resolution_px) != 2
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in self.resolution_px
            )
        ):
            raise ValueError("resolution_px must contain two positive integers")
        if self.projection not in ("orthographic", "perspective"):
            raise ValueError("projection must be 'orthographic' or 'perspective'")
        if self.projection == "orthographic":
            if (
                self.orthographic_scale_mm is None
                or not isfinite(self.orthographic_scale_mm)
                or self.orthographic_scale_mm <= 0.0
            ):
                raise ValueError(
                    "orthographic_scale_mm must be finite and positive "
                    "for an orthographic camera"
                )
        if not isfinite(self.fov_deg) or not 0.0 < self.fov_deg < 180.0:
            raise ValueError("fov_deg must lie strictly between 0 and 180")


@dataclass(frozen=True)
class RenderSettings:
    """Sampling and renderer-scale settings, separate from physical data."""

    variant: str = "scalar_rgb"
    spp: int = 256
    max_depth: int = 12
    optical_depth_mm: float = 10.0
    point_emitter_scale: float = 30.0
    source_epsilon_mm: float = 1.0e-3

    def __post_init__(self) -> None:
        if not isinstance(self.variant, str) or not self.variant.strip():
            raise ValueError("variant must be a nonempty string")
        for name, value, minimum in (
            ("spp", self.spp, 1),
            ("max_depth", self.max_depth, 1),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if not isfinite(self.optical_depth_mm) or self.optical_depth_mm <= 0.0:
            raise ValueError(
                "optical_depth_mm must be finite and greater than zero"
            )
        if not isfinite(self.source_epsilon_mm) or self.source_epsilon_mm < 0.0:
            raise ValueError("source_epsilon_mm must be finite and nonnegative")
        if not isfinite(self.point_emitter_scale) or self.point_emitter_scale < 0.0:
            raise ValueError(
                "point_emitter_scale must be finite and nonnegative"
            )


def _default_camera(
    tip: Fingertip,
    mesh: PadMesh,
) -> Camera:
    """Frame reference geometry once for comparable no-load/loaded renders."""
    coordinates = mesh.coordinates
    min_x, min_y = np.min(coordinates, axis=0)
    max_x, max_y = np.max(coordinates, axis=0)
    rigid_min_x, rigid_min_y, rigid_max_x, rigid_max_y = (
        tip.geometry.link_geometry.bounds
    )
    min_x = min(float(min_x), rigid_min_x)
    min_y = min(float(min_y), rigid_min_y)
    max_x = max(float(max_x), rigid_max_x)
    max_y = max(float(max_y), rigid_max_y)
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    span = 1.16 * max(max_x - min_x, max_y - min_y, 1.0)
    return Camera(
        position_mm=(center_x, center_y, 40.0),
        target_mm=(center_x, center_y, 0.0),
        up=(0.0, 1.0, 0.0),
        orthographic_scale_mm=span / 2.0,
    )
