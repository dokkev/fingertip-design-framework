"""Small public API for deterministic fingertip light transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry.base import BaseGeometry

from model import Fingertip
from optics.cross_section.domain import (
    _build_mesh_domain,
    _build_no_load_domain,
)
from optics.cross_section.result import OpticalMedium, _RawRaySegment
from optics.cross_section.settings import TraceSettings
from optics.cross_section.transport import _trace_transport


@dataclass(frozen=True)
class RaySegment:
    """One weighted straight portion of a traced ray."""

    start: tuple[float, float]
    end: tuple[float, float]
    medium: OpticalMedium
    start_weight: float
    end_weight: float
    ray_index: int
    interaction_index: int

    @classmethod
    def _from_engine(cls, segment: _RawRaySegment) -> RaySegment:
        return cls(
            start=segment.start_mm,
            end=segment.end_mm,
            medium=segment.medium,
            start_weight=segment.start_weight,
            end_weight=segment.end_weight,
            ray_index=segment.primary_ray_index,
            interaction_index=segment.interaction_index,
        )


@dataclass(frozen=True)
class TransportResult:
    """A self-contained light-transport proxy and its plotting geometry."""

    source: tuple[float, float]
    x_edges: np.ndarray
    y_edges: np.ndarray
    density: np.ndarray
    optical_mask: np.ndarray
    segments: tuple[RaySegment, ...]
    outer_envelope: BaseGeometry
    silicone_region: BaseGeometry
    air_region: BaseGeometry
    rigid_region: BaseGeometry
    led_region: BaseGeometry
    launched_ray_count: int
    launched_weight: float
    escaped_weight: float
    absorbed_weight: float
    terminated_weight: float

    def __post_init__(self) -> None:
        """Own immutable copies of the neutral numerical result."""
        x_edges = np.array(self.x_edges, dtype=float, copy=True)
        y_edges = np.array(self.y_edges, dtype=float, copy=True)
        density = np.array(self.density, dtype=float, copy=True)
        optical_mask = np.array(self.optical_mask, dtype=bool, copy=True)
        if density.shape != optical_mask.shape:
            raise ValueError("density and optical_mask shapes differ")
        if len(x_edges) != density.shape[1] + 1:
            raise ValueError("x_edges does not match the density width")
        if len(y_edges) != density.shape[0] + 1:
            raise ValueError("y_edges does not match the density height")
        if not all(
            np.all(np.isfinite(array))
            for array in (x_edges, y_edges, density)
        ):
            raise ValueError("transport arrays must contain finite values")
        if np.any(density < 0.0):
            raise ValueError("density must be nonnegative")
        if np.any(np.diff(x_edges) <= 0.0) or np.any(np.diff(y_edges) <= 0.0):
            raise ValueError("transport grid edges must be strictly increasing")
        weights = np.asarray(
            [
                self.launched_weight,
                self.escaped_weight,
                self.absorbed_weight,
                self.terminated_weight,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("transport weights must be finite and nonnegative")
        for array in (x_edges, y_edges, density, optical_mask):
            array.setflags(write=False)
        object.__setattr__(self, "x_edges", x_edges)
        object.__setattr__(self, "y_edges", y_edges)
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "optical_mask", optical_mask)

def _pad_view(mesh: Any) -> Any:
    """Accept either the full mechanical mesh or a pad mesh view."""
    if hasattr(mesh, "coordinates") and hasattr(mesh, "triangles"):
        return mesh
    try:
        return mesh.pad
    except AttributeError as exc:
        raise TypeError(
            "mesh must be a FingertipMesh, PadMesh, or deformed pad mesh"
        ) from exc


def trace(
    tip: Fingertip,
    mesh: Any | None = None,
    settings: TraceSettings | None = None,
) -> TransportResult:
    """Trace a reference, loaded, or analytic no-load fingertip state."""
    if not isinstance(tip, Fingertip):
        raise TypeError("tip must be a Fingertip")
    domain = (
        _build_no_load_domain(tip)
        if mesh is None
        else _build_mesh_domain(tip, _pad_view(mesh))
    )
    raw = _trace_transport(
        domain,
        led=tip.led,
        material=tip.optical,
        settings=settings,
    )
    return TransportResult(
        source=raw.source_position_mm,
        x_edges=raw.x_edges_mm,
        y_edges=raw.y_edges_mm,
        density=raw.weighted_path_density,
        optical_mask=raw.optical_mask,
        segments=tuple(RaySegment._from_engine(item) for item in raw.segments),
        outer_envelope=domain.outer_envelope,
        silicone_region=domain.silicone_region,
        air_region=domain.accessible_region.difference(domain.silicone_region),
        rigid_region=domain.rigid_region,
        led_region=tip.led_package_geometry,
        launched_ray_count=raw.launched_ray_count,
        launched_weight=raw.launched_weight,
        escaped_weight=raw.escaped_weight,
        absorbed_weight=raw.absorbed_weight,
        terminated_weight=raw.terminated_weight,
    )


__all__ = ["RaySegment", "TraceSettings", "TransportResult", "trace"]
