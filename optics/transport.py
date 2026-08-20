"""Small public API for deterministic fingertip light transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry.base import BaseGeometry

from model import Fingertip
from mesh.indenter import IndenterPose2D
from optics.contact_object import IndenterOptics
from optics.cross_section.domain import (
    build_mesh_domain,
    build_no_load_domain,
)
from optics.cross_section.result import (
    OpticalMedium,
    RawExitEvent,
    RawRaySegment,
)
from optics.cross_section.settings import TraceSettings
from optics.cross_section.transport import trace_transport


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
    def _from_engine(cls, segment: RawRaySegment) -> RaySegment:
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
class ExitEvent:
    """One outgoing escape event from the reduced optical domain."""

    position: tuple[float, float]
    direction: tuple[float, float]
    weight: float
    boundary_tag: str | None
    ray_index: int
    interaction_index: int

    @classmethod
    def _from_engine(cls, event: RawExitEvent) -> ExitEvent:
        return cls(
            position=event.position_mm,
            direction=event.direction,
            weight=event.weight,
            boundary_tag=event.boundary_tag,
            ray_index=event.primary_ray_index,
            interaction_index=event.interaction_index,
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
    exit_events: tuple[ExitEvent, ...]
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
    object_absorbed_weight: float = 0.0
    object_transmitted_weight: float = 0.0
    object_interface_incident_weight: float = 0.0
    object_reflected_weight: float = 0.0

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
                self.object_absorbed_weight,
                self.object_transmitted_weight,
                self.object_interface_incident_weight,
                self.object_reflected_weight,
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

    @property
    def bulk_absorbed_weight(self) -> float:
        """Bulk silicone absorption; ``absorbed_weight`` remains the legacy name."""
        return self.absorbed_weight

    @property
    def terminal_weight(self) -> float:
        """Sum of all terminal energy channels, including the object."""
        return (
            self.escaped_weight
            + self.bulk_absorbed_weight
            + self.terminated_weight
            + self.object_absorbed_weight
            + self.object_transmitted_weight
        )

    @property
    def energy_balance_error(self) -> float:
        """Absolute launched-to-terminal weight residual."""
        return self.launched_weight - self.terminal_weight

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
    indenter_pose: IndenterPose2D | None = None,
    indenter_optics: IndenterOptics | None = None,
) -> TransportResult:
    """Trace a reference, loaded, or analytic no-load fingertip state."""
    if not isinstance(tip, Fingertip):
        raise TypeError("tip must be a Fingertip")
    domain = (
        build_no_load_domain(
            tip,
            indenter_pose=indenter_pose,
            indenter_optics=indenter_optics,
        )
        if mesh is None
        else build_mesh_domain(
            tip,
            _pad_view(mesh),
            indenter_pose=indenter_pose,
            indenter_optics=indenter_optics,
        )
    )
    raw = trace_transport(
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
        exit_events=tuple(ExitEvent._from_engine(item) for item in raw.exit_events),
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
        object_absorbed_weight=raw.object_absorbed_weight,
        object_transmitted_weight=raw.object_transmitted_weight,
        object_interface_incident_weight=raw.object_interface_incident_weight,
        object_reflected_weight=raw.object_reflected_weight,
    )


__all__ = [
    "ExitEvent",
    "RaySegment",
    "TraceSettings",
    "TransportResult",
    "trace",
]
