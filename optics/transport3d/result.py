"""Neutral immutable results for deterministic 3D optical transport."""

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
    """Recursively freeze the small metadata trees carried by a result."""
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
    """Camera-free outgoing field and complete energy bookkeeping."""

    source_position_mm: tuple[float, float, float]
    source_mode: str
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
    object_absorbed_weight: float = 0.0
    object_transmitted_weight: float = 0.0
    object_interface_incident_weight: float = 0.0
    object_reflected_weight: float = 0.0
    projected_x_edges_mm: np.ndarray | None = None
    projected_y_edges_mm: np.ndarray | None = None
    projected_weighted_path_density: np.ndarray | None = None
    projected_optical_mask: np.ndarray | None = None
    internal_path_x_edges_mm: np.ndarray | None = None
    internal_path_y_edges_mm: np.ndarray | None = None
    internal_path_z_edges_mm: np.ndarray | None = None
    internal_weighted_path_density_3d: np.ndarray | None = None
    internal_z_integrated_path_density: np.ndarray | None = None
    retained_segment_lengths_mm: np.ndarray | None = None
    retained_segment_primary_ray_indices: np.ndarray | None = None
    retained_segment_interaction_counts: np.ndarray | None = None
    retained_segment_starts_mm: np.ndarray | None = None
    retained_segment_ends_mm: np.ndarray | None = None
    retained_segment_media: np.ndarray | None = None
    retained_segment_start_weights: np.ndarray | None = None
    retained_segment_end_weights: np.ndarray | None = None
    geometry_metadata: Mapping[str, Any] = field(default_factory=dict)
    timings_seconds: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source = tuple(float(value) for value in self.source_position_mm)
        if len(source) != 3 or not np.all(np.isfinite(source)):
            raise Transport3DResultError("source_position_mm must contain three finite values")
        if self.source_mode not in ("planar", "full3d"):
            raise Transport3DResultError("source_mode must be 'planar' or 'full3d'")
        if (
            not isinstance(self.launched_ray_count, int)
            or isinstance(self.launched_ray_count, bool)
            or self.launched_ray_count < 1
        ):
            raise Transport3DResultError("launched_ray_count must be a positive integer")
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
        )
        scalars = {name: float(getattr(self, name)) for name in scalar_names}
        if any(not np.isfinite(value) for value in scalars.values()):
            raise Transport3DResultError("result scalars must be finite")
        if scalars["extrusion_depth_mm"] <= 0.0 or scalars["energy_balance_tolerance"] <= 0.0:
            raise Transport3DResultError("depth and energy tolerance must be positive")
        if any(value < 0.0 for name, value in scalars.items() if name not in ("energy_balance_error",)):
            raise Transport3DResultError("result weights must be nonnegative")
        if scalars["energy_balance_error"] < 0.0:
            raise Transport3DResultError("energy_balance_error must be nonnegative")
        if scalars["energy_balance_error"] > scalars["energy_balance_tolerance"]:
            raise Transport3DResultError("energy balance exceeds its declared tolerance")
        terminal_weight = sum(
            scalars[name]
            for name in (
                "escaped_weight",
                "absorbed_weight",
                "terminated_weight",
                "object_absorbed_weight",
                "object_transmitted_weight",
            )
        )
        calculated_error = abs(scalars["launched_weight"] - terminal_weight) / max(
            scalars["launched_weight"], 1.0e-30
        )
        if calculated_error > scalars["energy_balance_tolerance"]:
            raise Transport3DResultError(
                "terminal energy channels do not match the declared balance"
            )

        u_edges = _owned_array(self.surface_u_edges, dtype=float, name="surface_u_edges")
        z_edges = _owned_array(self.surface_z_edges, dtype=float, name="surface_z_edges")
        field = _owned_array(self.outgoing_surface_field, dtype=float, name="outgoing_surface_field")
        if len(u_edges) < 2 or len(z_edges) < 2 or field.shape != (len(z_edges) - 1, len(u_edges) - 1):
            raise Transport3DResultError("surface field shape does not match its edges")
        if np.any(np.diff(u_edges) <= 0.0) or np.any(np.diff(z_edges) <= 0.0):
            raise Transport3DResultError("surface field edges must be increasing")
        if np.any(field < 0.0):
            raise Transport3DResultError("surface field must be nonnegative")

        positions = _owned_array(self.escape_positions_mm, dtype=float, name="escape_positions_mm")
        directions = _owned_array(self.escape_directions, dtype=float, name="escape_directions")
        normals = _owned_array(self.escape_surface_normals, dtype=float, name="escape_surface_normals")
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
        paths = _owned_array(self.escape_path_lengths_mm, dtype=float, name="escape_path_lengths_mm")
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
            or np.any(interactions < 0)
        ):
            raise Transport3DResultError("escape metadata contains invalid values")
        normals_norm = np.linalg.norm(normals, axis=1)
        if len(normals) and np.any(~np.isfinite(normals_norm) | (normals_norm <= 0.0)):
            raise Transport3DResultError("escape surface normals must be nonzero")
        directions_norm = np.linalg.norm(directions, axis=1)
        if len(directions) and np.any(~np.isfinite(directions_norm) | (directions_norm <= 0.0)):
            raise Transport3DResultError("escape directions must be nonzero")
        primary.setflags(write=False)
        interactions.setflags(write=False)

        projected = []
        if self.projected_x_edges_mm is not None or self.projected_y_edges_mm is not None or self.projected_weighted_path_density is not None:
            if self.projected_x_edges_mm is None or self.projected_y_edges_mm is None or self.projected_weighted_path_density is None:
                raise Transport3DResultError("projected diagnostic arrays must be supplied together")
            projected_x = _owned_array(self.projected_x_edges_mm, dtype=float, name="projected_x_edges_mm")
            projected_y = _owned_array(self.projected_y_edges_mm, dtype=float, name="projected_y_edges_mm")
            projected_density = _owned_array(self.projected_weighted_path_density, dtype=float, name="projected_weighted_path_density")
            projected_mask = (
                np.ones_like(projected_density, dtype=bool)
                if self.projected_optical_mask is None
                else np.array(self.projected_optical_mask, dtype=bool, copy=True)
            )
            if (
                projected_density.shape != (len(projected_y) - 1, len(projected_x) - 1)
                or projected_mask.shape != projected_density.shape
                or np.any(projected_density < 0.0)
            ):
                raise Transport3DResultError("projected diagnostic shape is invalid")
            projected_mask.setflags(write=False)
            projected = [projected_x, projected_y, projected_density, projected_mask]

        internal = []
        internal_values = (
            self.internal_path_x_edges_mm,
            self.internal_path_y_edges_mm,
            self.internal_path_z_edges_mm,
            self.internal_weighted_path_density_3d,
            self.internal_z_integrated_path_density,
        )
        if any(value is not None for value in internal_values):
            if any(value is None for value in internal_values):
                raise Transport3DResultError(
                    "internal path arrays must be supplied together"
                )
            internal_x = _owned_array(
                self.internal_path_x_edges_mm,
                dtype=float,
                name="internal_path_x_edges_mm",
            )
            internal_y = _owned_array(
                self.internal_path_y_edges_mm,
                dtype=float,
                name="internal_path_y_edges_mm",
            )
            internal_z = _owned_array(
                self.internal_path_z_edges_mm,
                dtype=float,
                name="internal_path_z_edges_mm",
            )
            internal_density = _owned_array(
                self.internal_weighted_path_density_3d,
                dtype=float,
                name="internal_weighted_path_density_3d",
            )
            integrated_density = _owned_array(
                self.internal_z_integrated_path_density,
                dtype=float,
                name="internal_z_integrated_path_density",
            )
            if (
                len(internal_x) < 2
                or len(internal_y) < 2
                or len(internal_z) < 2
                or np.any(np.diff(internal_x) <= 0.0)
                or np.any(np.diff(internal_y) <= 0.0)
                or np.any(np.diff(internal_z) <= 0.0)
                or internal_density.shape
                != (len(internal_z) - 1, len(internal_y) - 1, len(internal_x) - 1)
                or integrated_density.shape != (len(internal_y) - 1, len(internal_x) - 1)
                or np.any(internal_density < 0.0)
                or np.any(integrated_density < 0.0)
            ):
                raise Transport3DResultError("internal path field shape is invalid")
            if not np.allclose(
                integrated_density,
                np.sum(internal_density, axis=0),
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                raise Transport3DResultError(
                    "z-integrated internal path field does not match P3"
                )
            internal = [
                internal_x,
                internal_y,
                internal_z,
                internal_density,
                integrated_density,
            ]

        retained_segments = []
        retained_values = (
            self.retained_segment_lengths_mm,
            self.retained_segment_primary_ray_indices,
            self.retained_segment_interaction_counts,
            self.retained_segment_starts_mm,
            self.retained_segment_ends_mm,
            self.retained_segment_media,
            self.retained_segment_start_weights,
            self.retained_segment_end_weights,
        )
        if any(value is not None for value in retained_values):
            if any(value is None for value in retained_values):
                raise Transport3DResultError(
                    "retained segment metadata must be supplied together"
                )
            retained_lengths = _owned_array(
                self.retained_segment_lengths_mm,
                dtype=float,
                name="retained_segment_lengths_mm",
            )
            retained_primary = np.array(
                self.retained_segment_primary_ray_indices,
                dtype=np.int64,
                copy=True,
            )
            retained_interactions = np.array(
                self.retained_segment_interaction_counts,
                dtype=np.int64,
                copy=True,
            )
            retained_starts = _owned_array(
                self.retained_segment_starts_mm,
                dtype=float,
                name="retained_segment_starts_mm",
            )
            retained_ends = _owned_array(
                self.retained_segment_ends_mm,
                dtype=float,
                name="retained_segment_ends_mm",
            )
            retained_media = np.array(
                self.retained_segment_media,
                dtype=np.uint8,
                copy=True,
            )
            retained_start_weights = _owned_array(
                self.retained_segment_start_weights,
                dtype=float,
                name="retained_segment_start_weights",
            )
            retained_end_weights = _owned_array(
                self.retained_segment_end_weights,
                dtype=float,
                name="retained_segment_end_weights",
            )
            if (
                retained_lengths.ndim != 1
                or retained_primary.ndim != 1
                or retained_interactions.ndim != 1
                or retained_starts.ndim != 2
                or retained_starts.shape[1:] != (3,)
                or retained_ends.shape != retained_starts.shape
                or retained_media.ndim != 1
                or retained_start_weights.ndim != 1
                or retained_end_weights.ndim != 1
                or len(retained_primary) != len(retained_lengths)
                or len(retained_interactions) != len(retained_lengths)
                or len(retained_starts) != len(retained_lengths)
                or len(retained_ends) != len(retained_lengths)
                or len(retained_media) != len(retained_lengths)
                or len(retained_start_weights) != len(retained_lengths)
                or len(retained_end_weights) != len(retained_lengths)
                or np.any(retained_lengths < 0.0)
                or np.any(retained_primary < 0)
                or np.any(retained_interactions < 0)
                or np.any(retained_media > 1)
                or np.any(retained_start_weights < 0.0)
                or np.any(retained_end_weights < 0.0)
            ):
                raise Transport3DResultError("retained segment metadata is invalid")
            retained_primary.setflags(write=False)
            retained_interactions.setflags(write=False)
            retained_media.setflags(write=False)
            retained_segments = [
                retained_lengths,
                retained_primary,
                retained_interactions,
                retained_starts,
                retained_ends,
                retained_media,
                retained_start_weights,
                retained_end_weights,
            ]

        for name, array in (
            ("u_edges", u_edges),
            ("z_edges", z_edges),
            ("field", field),
            ("positions", positions),
            ("directions", directions),
            ("normals", normals),
            ("surface_u", surface_u),
            ("surface_z", surface_z),
            ("weights", weights),
            ("paths", paths),
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
        primitive_indices.setflags(write=False)
        object.__setattr__(self, "escape_surface_primitive_indices", primitive_indices)
        object.__setattr__(self, "escape_weights", weights)
        object.__setattr__(self, "escape_primary_ray_indices", primary)
        object.__setattr__(self, "escape_path_lengths_mm", paths)
        object.__setattr__(self, "escape_interaction_counts", interactions)
        if projected:
            object.__setattr__(self, "projected_x_edges_mm", projected[0])
            object.__setattr__(self, "projected_y_edges_mm", projected[1])
            object.__setattr__(self, "projected_weighted_path_density", projected[2])
            object.__setattr__(self, "projected_optical_mask", projected[3])
        if internal:
            object.__setattr__(self, "internal_path_x_edges_mm", internal[0])
            object.__setattr__(self, "internal_path_y_edges_mm", internal[1])
            object.__setattr__(self, "internal_path_z_edges_mm", internal[2])
            object.__setattr__(self, "internal_weighted_path_density_3d", internal[3])
            object.__setattr__(self, "internal_z_integrated_path_density", internal[4])
        if retained_segments:
            object.__setattr__(self, "retained_segment_lengths_mm", retained_segments[0])
            object.__setattr__(self, "retained_segment_primary_ray_indices", retained_segments[1])
            object.__setattr__(self, "retained_segment_interaction_counts", retained_segments[2])
            object.__setattr__(self, "retained_segment_starts_mm", retained_segments[3])
            object.__setattr__(self, "retained_segment_ends_mm", retained_segments[4])
            object.__setattr__(self, "retained_segment_media", retained_segments[5])
            object.__setattr__(self, "retained_segment_start_weights", retained_segments[6])
            object.__setattr__(self, "retained_segment_end_weights", retained_segments[7])
        object.__setattr__(self, "geometry_metadata", _freeze_metadata(self.geometry_metadata))
        object.__setattr__(self, "timings_seconds", MappingProxyType({key: float(value) for key, value in self.timings_seconds.items()}))


__all__ = ["Transport3DResult", "Transport3DResultError"]
