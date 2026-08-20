"""Neutral immutable results for deterministic FULL_3D optical transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


class Transport3DResultError(ValueError):
    """Raised when a 3D transport result violates its neutral contract."""


def _owned_array(value: Any, *, dtype: Any, name: str) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if not np.all(np.isfinite(array)):
        raise Transport3DResultError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True)
        array.setflags(write=False)
        return array
    return value


@dataclass(frozen=True)
class Transport3DResult:
    """Camera-free outgoing field and complete FULL_3D energy bookkeeping.

    ``escaped_weight`` includes both physical silicone-surface escape and
    virtual-envelope escape. ``outgoing_surface_weight`` and the recorded
    escape arrays contain only events with a silicone surface ``(u, z)``
    coordinate, so their mass may be smaller than ``escaped_weight``.
    """

    source_position_mm: tuple[float, float, float]
    extrusion_depth_mm: float
    launched_ray_count: int
    launched_weight: float
    escaped_weight: float
    absorbed_weight: float
    terminated_weight: float
    outgoing_surface_weight: float
    surface_u_edges: np.ndarray
    surface_z_edges: np.ndarray
    outgoing_surface_field: np.ndarray
    escape_positions_mm: np.ndarray
    escape_directions: np.ndarray
    escape_surface_normals: np.ndarray
    escape_surface_u: np.ndarray
    escape_surface_z: np.ndarray
    escape_surface_tags: tuple[str, ...]
    escape_surface_primitive_indices: np.ndarray
    escape_weights: np.ndarray
    escape_primary_ray_indices: np.ndarray
    escape_path_lengths_mm: np.ndarray
    escape_interaction_counts: np.ndarray
    energy_balance_error: float
    energy_balance_tolerance: float
    processed_segment_count: int = 0
    periodic_wrap_termination_count: int = 0
    periodic_wrap_termination_weight: float = 0.0
    no_event_termination_count: int = 0
    no_event_termination_weight: float = 0.0
    branch_cutoff_termination_count: int = 0
    branch_cutoff_termination_weight: float = 0.0
    max_interaction_termination_count: int = 0
    max_interaction_termination_weight: float = 0.0
    segment_budget_termination_count: int = 0
    segment_budget_termination_weight: float = 0.0
    rigid_surface_termination_count: int = 0
    rigid_surface_termination_weight: float = 0.0
    interface_normal_fallback_count: int = 0
    carrier_contact_triangle_count: int = 0
    escape_event_count: int = field(init=False)
    escaped_primary_count: int = field(init=False)
    object_absorbed_weight: float = 0.0
    object_transmitted_weight: float = 0.0
    object_interface_incident_weight: float = 0.0
    object_reflected_weight: float = 0.0
    carrier_absorbed_weight: float = 0.0
    carrier_transmitted_weight: float = 0.0
    carrier_interface_incident_weight: float = 0.0
    carrier_reflected_weight: float = 0.0
    field_x_edges_mm: np.ndarray | None = None
    field_y_edges_mm: np.ndarray | None = None
    field_z_edges_mm: np.ndarray | None = None
    field_density_3d: np.ndarray | None = None
    geometry_metadata: Mapping[str, Any] = field(default_factory=dict)
    timings_seconds: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source = tuple(float(value) for value in self.source_position_mm)
        if len(source) != 3 or not np.all(np.isfinite(source)):
            raise Transport3DResultError(
                "source_position_mm must contain three finite values"
            )
        if (
            not isinstance(self.launched_ray_count, int)
            or isinstance(self.launched_ray_count, bool)
            or self.launched_ray_count < 1
        ):
            raise Transport3DResultError(
                "launched_ray_count must be a positive integer"
            )
        scalar_names = (
            "extrusion_depth_mm",
            "launched_weight",
            "escaped_weight",
            "absorbed_weight",
            "terminated_weight",
            "outgoing_surface_weight",
            "energy_balance_error",
            "energy_balance_tolerance",
            "object_absorbed_weight",
            "object_transmitted_weight",
            "object_interface_incident_weight",
            "object_reflected_weight",
            "carrier_absorbed_weight",
            "carrier_transmitted_weight",
            "carrier_interface_incident_weight",
            "carrier_reflected_weight",
            "periodic_wrap_termination_weight",
            "no_event_termination_weight",
            "branch_cutoff_termination_weight",
            "max_interaction_termination_weight",
            "segment_budget_termination_weight",
            "rigid_surface_termination_weight",
        )
        scalars = {name: float(getattr(self, name)) for name in scalar_names}
        if any(not np.isfinite(value) for value in scalars.values()):
            raise Transport3DResultError("result scalars must be finite")
        if scalars["extrusion_depth_mm"] <= 0.0 or scalars[
            "energy_balance_tolerance"
        ] <= 0.0:
            raise Transport3DResultError(
                "depth and energy tolerance must be positive"
            )
        if any(
            value < 0.0
            for name, value in scalars.items()
            if name != "energy_balance_error"
        ):
            raise Transport3DResultError("result weights must be nonnegative")
        if scalars["energy_balance_error"] < 0.0:
            raise Transport3DResultError("energy_balance_error must be nonnegative")
        if scalars["energy_balance_error"] > scalars["energy_balance_tolerance"]:
            raise Transport3DResultError("energy balance exceeds its declared tolerance")
        count_names = (
            "processed_segment_count",
            "periodic_wrap_termination_count",
            "no_event_termination_count",
            "branch_cutoff_termination_count",
            "max_interaction_termination_count",
            "segment_budget_termination_count",
            "rigid_surface_termination_count",
            "interface_normal_fallback_count",
            "carrier_contact_triangle_count",
        )
        for name in count_names:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise Transport3DResultError(f"{name} must be a non-negative integer")
        for name in (
            "periodic_wrap_termination_weight",
            "no_event_termination_weight",
            "branch_cutoff_termination_weight",
            "max_interaction_termination_weight",
            "segment_budget_termination_weight",
            "rigid_surface_termination_weight",
        ):
            if scalars[name] < 0.0:
                raise Transport3DResultError(f"{name} must be non-negative")
        terminal_weight = sum(
            scalars[name]
            for name in (
                "escaped_weight",
                "absorbed_weight",
                "terminated_weight",
                "object_absorbed_weight",
                "object_transmitted_weight",
                "carrier_absorbed_weight",
                "carrier_transmitted_weight",
            )
        )
        calculated_error = abs(scalars["launched_weight"] - terminal_weight) / max(
            scalars["launched_weight"], 1.0e-30
        )
        if calculated_error > scalars["energy_balance_tolerance"]:
            raise Transport3DResultError(
                "terminal energy channels do not match the declared balance"
            )
        if not np.isclose(
            calculated_error,
            scalars["energy_balance_error"],
            rtol=1.0e-9,
            atol=1.0e-12,
        ):
            raise Transport3DResultError(
                "energy_balance_error does not match the terminal channels"
            )
        detailed_termination_weight = sum(
            scalars[name]
            for name in (
                "periodic_wrap_termination_weight",
                "no_event_termination_weight",
                "branch_cutoff_termination_weight",
                "max_interaction_termination_weight",
                "segment_budget_termination_weight",
                "rigid_surface_termination_weight",
            )
        )
        termination_error = abs(
            scalars["terminated_weight"] - detailed_termination_weight
        ) / max(scalars["launched_weight"], 1.0e-30)
        if termination_error > scalars["energy_balance_tolerance"]:
            raise Transport3DResultError(
                "explicit termination channels do not match terminated_weight"
            )
        for prefix in ("object", "carrier"):
            accounted = sum(
                scalars[f"{prefix}_{channel}_weight"]
                for channel in ("absorbed", "transmitted", "reflected")
            )
            interface_error = abs(
                scalars[f"{prefix}_interface_incident_weight"] - accounted
            ) / max(scalars["launched_weight"], 1.0e-30)
            if interface_error > scalars["energy_balance_tolerance"]:
                raise Transport3DResultError(
                    f"{prefix} interface energy channels do not match incident weight"
                )

        u_edges = _owned_array(self.surface_u_edges, dtype=float, name="surface_u_edges")
        z_edges = _owned_array(self.surface_z_edges, dtype=float, name="surface_z_edges")
        field = _owned_array(
            self.outgoing_surface_field,
            dtype=float,
            name="outgoing_surface_field",
        )
        if (
            len(u_edges) < 2
            or len(z_edges) < 2
            or field.shape != (len(z_edges) - 1, len(u_edges) - 1)
        ):
            raise Transport3DResultError("surface field shape does not match its edges")
        if np.any(np.diff(u_edges) <= 0.0) or np.any(np.diff(z_edges) <= 0.0):
            raise Transport3DResultError("surface field edges must be increasing")
        if np.any(field < 0.0):
            raise Transport3DResultError("surface field must be nonnegative")

        positions = _owned_array(
            self.escape_positions_mm, dtype=float, name="escape_positions_mm"
        )
        directions = _owned_array(
            self.escape_directions, dtype=float, name="escape_directions"
        )
        normals = _owned_array(
            self.escape_surface_normals,
            dtype=float,
            name="escape_surface_normals",
        )
        surface_u = _owned_array(self.escape_surface_u, dtype=float, name="escape_surface_u")
        surface_z = _owned_array(self.escape_surface_z, dtype=float, name="escape_surface_z")
        surface_tags = tuple(str(tag) for tag in self.escape_surface_tags)
        primitive_indices = np.array(
            self.escape_surface_primitive_indices,
            dtype=np.int64,
            copy=True,
        )
        weights = _owned_array(self.escape_weights, dtype=float, name="escape_weights")
        primary = np.array(self.escape_primary_ray_indices, dtype=np.int64, copy=True)
        paths = _owned_array(
            self.escape_path_lengths_mm,
            dtype=float,
            name="escape_path_lengths_mm",
        )
        interactions = np.array(self.escape_interaction_counts, dtype=np.int64, copy=True)
        if (
            positions.ndim != 2
            or positions.shape[1:] != (3,)
            or directions.shape != positions.shape
            or normals.shape != positions.shape
        ):
            raise Transport3DResultError(
                "escape positions, directions, and normals must have shape (N, 3)"
            )
        if any(
            array.ndim != 1 or len(array) != len(positions)
            for array in (
                surface_u,
                surface_z,
                primitive_indices,
                weights,
                primary,
                paths,
                interactions,
            )
        ):
            raise Transport3DResultError("escape metadata lengths do not match positions")
        if len(surface_tags) != len(positions):
            raise Transport3DResultError("escape surface tags do not match positions")
        if (
            np.any(surface_u < u_edges[0])
            or np.any(surface_u > u_edges[-1])
            or np.any(surface_z < z_edges[0])
            or np.any(surface_z > z_edges[-1])
            or np.any(primitive_indices < 0)
            or np.any(weights < 0.0)
            or np.any(paths < 0.0)
            or np.any(primary < 0)
            or np.any(primary >= self.launched_ray_count)
            or np.any(interactions < 0)
        ):
            raise Transport3DResultError("escape metadata contains invalid values")
        normalization = max(scalars["launched_weight"], 1.0e-30)
        if abs(float(np.sum(weights)) - scalars["outgoing_surface_weight"]) / normalization > scalars[
            "energy_balance_tolerance"
        ]:
            raise Transport3DResultError(
                "escape weights do not match outgoing_surface_weight"
            )
        if abs(float(np.sum(field)) - scalars["outgoing_surface_weight"]) / normalization > scalars[
            "energy_balance_tolerance"
        ]:
            raise Transport3DResultError(
                "outgoing surface field does not match outgoing_surface_weight"
            )
        if (
            scalars["outgoing_surface_weight"] - scalars["escaped_weight"]
        ) / normalization > scalars["energy_balance_tolerance"]:
            raise Transport3DResultError(
                "outgoing_surface_weight cannot exceed escaped_weight"
            )
        observed_escape_events = len(weights)
        observed_escaped_primaries = len(np.unique(primary))
        object.__setattr__(self, "escape_event_count", observed_escape_events)
        object.__setattr__(self, "escaped_primary_count", observed_escaped_primaries)
        normals_norm = np.linalg.norm(normals, axis=1)
        if len(normals) and np.any(~np.isfinite(normals_norm) | (normals_norm <= 0.0)):
            raise Transport3DResultError("escape surface normals must be nonzero")
        directions_norm = np.linalg.norm(directions, axis=1)
        if len(directions) and np.any(
            ~np.isfinite(directions_norm) | (directions_norm <= 0.0)
        ):
            raise Transport3DResultError("escape directions must be nonzero")
        primary.setflags(write=False)
        interactions.setflags(write=False)
        primitive_indices.setflags(write=False)

        field_values = (
            self.field_x_edges_mm,
            self.field_y_edges_mm,
            self.field_z_edges_mm,
            self.field_density_3d,
        )
        field_arrays: list[np.ndarray] = []
        if any(value is not None for value in field_values):
            if any(value is None for value in field_values):
                raise Transport3DResultError(
                    "FULL_3D field arrays must be supplied together"
                )
            field_x = _owned_array(
                self.field_x_edges_mm,
                dtype=float,
                name="field_x_edges_mm",
            )
            field_y = _owned_array(
                self.field_y_edges_mm,
                dtype=float,
                name="field_y_edges_mm",
            )
            field_z = _owned_array(
                self.field_z_edges_mm,
                dtype=float,
                name="field_z_edges_mm",
            )
            field_density = _owned_array(
                self.field_density_3d,
                dtype=float,
                name="field_density_3d",
            )
            if (
                len(field_x) < 2
                or len(field_y) < 2
                or len(field_z) < 2
                or np.any(np.diff(field_x) <= 0.0)
                or np.any(np.diff(field_y) <= 0.0)
                or np.any(np.diff(field_z) <= 0.0)
                or field_density.shape
                != (len(field_x) - 1, len(field_y) - 1, len(field_z) - 1)
                or np.any(field_density < 0.0)
            ):
                raise Transport3DResultError("FULL_3D field shape is invalid")
            field_arrays = [field_x, field_y, field_z, field_density]

        for array in (
            u_edges,
            z_edges,
            field,
            positions,
            directions,
            normals,
            surface_u,
            surface_z,
            weights,
            paths,
        ):
            array.setflags(write=False)
        object.__setattr__(self, "source_position_mm", source)
        object.__setattr__(self, "surface_u_edges", u_edges)
        object.__setattr__(self, "surface_z_edges", z_edges)
        object.__setattr__(self, "outgoing_surface_field", field)
        object.__setattr__(self, "escape_positions_mm", positions)
        object.__setattr__(self, "escape_directions", directions)
        object.__setattr__(self, "escape_surface_normals", normals)
        object.__setattr__(self, "escape_surface_u", surface_u)
        object.__setattr__(self, "escape_surface_z", surface_z)
        object.__setattr__(self, "escape_surface_tags", surface_tags)
        object.__setattr__(self, "escape_surface_primitive_indices", primitive_indices)
        object.__setattr__(self, "escape_weights", weights)
        object.__setattr__(self, "escape_primary_ray_indices", primary)
        object.__setattr__(self, "escape_path_lengths_mm", paths)
        object.__setattr__(self, "escape_interaction_counts", interactions)
        if field_arrays:
            object.__setattr__(self, "field_x_edges_mm", field_arrays[0])
            object.__setattr__(self, "field_y_edges_mm", field_arrays[1])
            object.__setattr__(self, "field_z_edges_mm", field_arrays[2])
            object.__setattr__(self, "field_density_3d", field_arrays[3])
        object.__setattr__(self, "geometry_metadata", _freeze_metadata(self.geometry_metadata))
        timings = {str(key): float(value) for key, value in self.timings_seconds.items()}
        if any(
            not key or not np.isfinite(value) or value < 0.0
            for key, value in timings.items()
        ):
            raise Transport3DResultError(
                "timings_seconds must contain named finite non-negative values"
            )
        object.__setattr__(self, "timings_seconds", MappingProxyType(timings))

    @property
    def field(self) -> np.ndarray:
        """Return the authoritative native field in ``(x, y, z)`` order."""
        if self.field_density_3d is None:
            raise Transport3DResultError("FULL_3D result has no retained native field")
        return self.field_density_3d

    @property
    def total_transport(self) -> float:
        """Return the escaped transport mass used by optimization metrics."""
        return float(self.escaped_weight)

    @property
    def field_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return native field edges in public ``(x, y, z)`` order."""
        if (
            self.field_x_edges_mm is None
            or self.field_y_edges_mm is None
            or self.field_z_edges_mm is None
        ):
            raise Transport3DResultError("FULL_3D result has no native field axes")
        return self.field_x_edges_mm, self.field_y_edges_mm, self.field_z_edges_mm

__all__ = ["Transport3DResult", "Transport3DResultError"]
