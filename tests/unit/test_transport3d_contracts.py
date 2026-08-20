"""Focused, dependency-light contracts for deterministic 3D transport."""

from __future__ import annotations

import sys
import numpy as np
import pytest

from model import LED
from optics.geometry.extrusion import _ExtrudedMesh
from optics.transport3d.physics import (
    attenuated_weight,
    periodic_plane_distance,
    wrapped_periodic_z,
)
from optics.transport3d.result import Transport3DResult
from optics.transport3d.settings import Transport3DSettings


def _accumulate_reference_segment_path_3d(
    density: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    optical_mask: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    start_weight: float,
    end_weight: float,
    *,
    maximum_spacing: float,
) -> None:
    """Accumulate one weighted straight path for dependency-light tests."""
    displacement = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(displacement))
    if length <= 0.0:
        return
    sample_count = max(1, int(np.ceil(length / maximum_spacing)))
    fractions = (np.arange(sample_count, dtype=float) + 0.5) / sample_count
    samples = (
        np.asarray(start, dtype=float)[None, :]
        + fractions[:, None] * displacement[None, :]
    )
    x_indices = np.searchsorted(x_edges, samples[:, 0], side="right") - 1
    y_indices = np.searchsorted(y_edges, samples[:, 1], side="right") - 1
    z_indices = np.searchsorted(z_edges, samples[:, 2], side="right") - 1
    valid = (
        (x_indices >= 0)
        & (x_indices < len(x_edges) - 1)
        & (y_indices >= 0)
        & (y_indices < len(y_edges) - 1)
        & (z_indices >= 0)
        & (z_indices < len(z_edges) - 1)
    )
    if not np.any(valid):
        return
    x_indices = x_indices[valid]
    y_indices = y_indices[valid]
    z_indices = z_indices[valid]
    inside = optical_mask[y_indices, x_indices]
    if not np.any(inside):
        return
    representative_weight = 0.5 * (float(start_weight) + float(end_weight))
    np.add.at(
        density,
        (z_indices[inside], y_indices[inside], x_indices[inside]),
        representative_weight * length / sample_count,
    )


def test_full_sampling_is_deterministic_and_three_dimensional() -> None:
    from optics.transport3d.sampling import sample_directions

    led = LED()
    full_first = sample_directions(led, (0.0, -1.0), ray_count=257)
    full_second = sample_directions(led, (0.0, -1.0), ray_count=257)
    assert np.array_equal(full_first, full_second)
    assert np.allclose(np.linalg.norm(full_first, axis=1), 1.0)
    assert np.any(np.abs(full_first[:, 2]) > 0.0)


def test_extrusion_has_exact_periodic_depth_and_side_selector() -> None:
    from mesh.fingertip.surface import PadMesh

    mesh = PadMesh.from_arrays(
        node_ids=np.arange(4),
        reference_coordinates_mm=np.asarray(
            [[-1.0, 0.0], [1.0, 0.0], [1.0, -1.0], [-1.0, -1.0]]
        ),
        element_connectivity_node_ids=np.asarray([[0, 1, 2], [0, 2, 3]]),
        boundary_edge_node_ids_by_tag={"outer": np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]])},
    )
    extrusion = _ExtrudedMesh.from_pad_mesh(mesh, depth_mm=11.0)
    vertices = extrusion.vertices_for_mesh(mesh)
    assert np.isclose(np.min(vertices[:, 2]), -5.5)
    assert np.isclose(np.max(vertices[:, 2]), 5.5)
    assert extrusion.side_faces_for_edges(mesh.boundary_edges).shape == (8, 3)


def test_periodic_travel_and_wrap_preserve_direction() -> None:
    distance = periodic_plane_distance(
        np,
        np.asarray([5.0, -5.0]),
        np.asarray([0.5, -0.25]),
        z_min_mm=-5.5,
        z_max_mm=5.5,
        epsilon_mm=1.0e-6,
    )
    assert np.allclose(distance, np.asarray([1.0, 2.0]))
    wrapped = wrapped_periodic_z(
        np,
        np.asarray([0.5, -0.25]),
        z_min_mm=-5.5,
        z_max_mm=5.5,
        offset_mm=1.0e-5,
    )
    assert np.allclose(wrapped, np.asarray([-5.5 + 5.0e-6, 5.5 - 2.5e-6]))


def test_attenuation_and_result_validation() -> None:
    end, removed = attenuated_weight(
        1.0,
        10.0,
        medium="silicone",
        absorption_per_mm=0.02,
    )
    assert np.isclose(end + removed, 1.0)
    assert attenuated_weight(1.0, 10.0, medium="air", absorption_per_mm=0.02) == (1.0, 0.0)
    settings = Transport3DSettings(ray_count=3)
    result = Transport3DResult(
        source_position_mm=(0.0, -6.0, 0.0),
        extrusion_depth_mm=11.0,
        launched_ray_count=3,
        launched_weight=1.0,
        escaped_weight=1.0,
        absorbed_weight=0.0,
        terminated_weight=0.0,
        outgoing_surface_weight=1.0,
        surface_u_edges=np.linspace(0.0, 1.0, 3),
        surface_z_edges=np.linspace(-5.5, 5.5, 3),
        outgoing_surface_field=np.ones((2, 2)),
        escape_positions_mm=np.asarray([[0.0, 0.0, 0.0]]),
        escape_directions=np.asarray([[0.0, -1.0, 0.0]]),
        escape_surface_normals=np.asarray([[0.0, 1.0, 0.0]]),
        escape_surface_u=np.asarray([0.5]),
        escape_surface_z=np.asarray([0.0]),
        escape_surface_tags=("pad_outer_arc",),
        escape_surface_primitive_indices=np.asarray([0]),
        escape_weights=np.asarray([1.0]),
        escape_primary_ray_indices=np.asarray([0]),
        escape_path_lengths_mm=np.asarray([1.0]),
        escape_interaction_counts=np.asarray([0]),
        energy_balance_error=0.0,
        energy_balance_tolerance=settings.energy_balance_tolerance,
    )
    assert not result.outgoing_surface_field.flags.writeable


def test_simple_internal_path_accumulation_and_z_integration() -> None:
    density = np.zeros((2, 2, 2), dtype=float)
    repeated_density = np.zeros((2, 2, 2), dtype=float)
    x_edges = np.asarray([0.0, 1.0, 2.0])
    y_edges = np.asarray([0.0, 1.0, 2.0])
    z_edges = np.asarray([-1.0, 0.0, 1.0])
    optical_mask = np.ones((2, 2), dtype=bool)
    _accumulate_reference_segment_path_3d(
        density,
        x_edges,
        y_edges,
        z_edges,
        optical_mask,
        np.asarray([0.25, 0.25, -0.5]),
        np.asarray([0.25, 0.25, 0.5]),
        1.0,
        0.5,
        maximum_spacing=0.25,
    )
    _accumulate_reference_segment_path_3d(
        repeated_density,
        x_edges,
        y_edges,
        z_edges,
        optical_mask,
        np.asarray([0.25, 0.25, -0.5]),
        np.asarray([0.25, 0.25, 0.5]),
        1.0,
        0.5,
        maximum_spacing=0.25,
    )

    assert np.sum(density) == pytest.approx(0.75)
    assert density[0, 0, 0] > 0.0
    assert density[1, 0, 0] > 0.0
    np.testing.assert_array_equal(density, repeated_density)
    np.testing.assert_allclose(np.sum(density, axis=0)[0, 0], np.sum(density[:, 0, 0]))


def test_periodic_segments_accumulate_path_and_attenuation() -> None:
    density = np.zeros((2, 1, 1), dtype=float)
    edges = np.asarray([0.0, 1.0])
    z_edges = np.asarray([-1.0, 0.0, 1.0])
    mask = np.ones((1, 1), dtype=bool)
    first_end, first_removed = attenuated_weight(
        1.0,
        1.0,
        medium="silicone",
        absorption_per_mm=0.2,
    )
    second_end, second_removed = attenuated_weight(
        first_end,
        1.0,
        medium="silicone",
        absorption_per_mm=0.2,
    )
    _accumulate_reference_segment_path_3d(
        density,
        edges,
        edges,
        z_edges,
        mask,
        np.asarray([0.5, 0.5, -0.5]),
        np.asarray([0.5, 0.5, 0.5]),
        1.0,
        first_end,
        maximum_spacing=0.25,
    )
    _accumulate_reference_segment_path_3d(
        density,
        edges,
        edges,
        z_edges,
        mask,
        np.asarray([0.5, 0.5, -0.5]),
        np.asarray([0.5, 0.5, 0.5]),
        first_end,
        second_end,
        maximum_spacing=0.25,
    )

    assert np.sum(density) == pytest.approx(0.5 * (1.0 + first_end) + 0.5 * (first_end + second_end))
    assert first_end + first_removed == pytest.approx(1.0)
    assert second_end + second_removed == pytest.approx(first_end)


def test_rigid_blocker_mask_prevents_internal_path_accumulation() -> None:
    density = np.zeros((1, 1, 2), dtype=float)
    x_edges = np.asarray([0.0, 1.0, 2.0])
    y_edges = np.asarray([0.0, 1.0])
    z_edges = np.asarray([-1.0, 1.0])
    optical_mask = np.asarray([[True, False]])
    _accumulate_reference_segment_path_3d(
        density,
        x_edges,
        y_edges,
        z_edges,
        optical_mask,
        np.asarray([0.25, 0.5, 0.0]),
        np.asarray([1.75, 0.5, 0.0]),
        1.0,
        1.0,
        maximum_spacing=0.1,
    )

    assert density[0, 0, 0] > 0.0
    assert density[0, 0, 1] == 0.0


def test_nested_result_metadata_is_immutable() -> None:
    result = Transport3DResult(
        source_position_mm=(0.0, -6.0, 0.0),
        extrusion_depth_mm=11.0,
        launched_ray_count=3,
        launched_weight=1.0,
        escaped_weight=1.0,
        absorbed_weight=0.0,
        terminated_weight=0.0,
        outgoing_surface_weight=1.0,
        surface_u_edges=np.linspace(0.0, 1.0, 3),
        surface_z_edges=np.linspace(-5.5, 5.5, 3),
        outgoing_surface_field=np.ones((2, 2)),
        escape_positions_mm=np.asarray([[0.0, 0.0, 0.0]]),
        escape_directions=np.asarray([[0.0, -1.0, 0.0]]),
        escape_surface_normals=np.asarray([[0.0, 1.0, 0.0]]),
        escape_surface_u=np.asarray([0.5]),
        escape_surface_z=np.asarray([0.0]),
        escape_surface_tags=("pad_outer_arc",),
        escape_surface_primitive_indices=np.asarray([0]),
        escape_weights=np.asarray([1.0]),
        escape_primary_ray_indices=np.asarray([0]),
        escape_path_lengths_mm=np.asarray([1.0]),
        escape_interaction_counts=np.asarray([0]),
        energy_balance_error=0.0,
        energy_balance_tolerance=1.0e-5,
        geometry_metadata={"nested": {"values": [1, 2]}},
    )

    with pytest.raises(TypeError):
        result.geometry_metadata["nested"]["values"] = (3,)  # type: ignore[index]


def test_normal_optics_import_does_not_load_optional_optix() -> None:
    import optics

    assert not hasattr(optics, "trace")
    assert hasattr(optics, "CarrierOptics")
    assert "optix" not in sys.modules
