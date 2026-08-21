"""Focused, dependency-light contracts for deterministic 3D transport."""

from __future__ import annotations

from dataclasses import replace
import sys
import numpy as np
import pytest

from lumo.finger import LED
from lumo.ray_tracing.optical_mechanics.path_field import PathFieldAccumulator
from lumo.ray_tracing.optical_mechanics.physics import (
    attenuated_weight,
    periodic_plane_distance,
    wrapped_periodic_z,
)
from lumo.ray_tracing.optical_mechanics.result import (
    Transport3DResult,
    Transport3DResultError,
)
from lumo.ray_tracing.optical_mechanics.settings import Transport3DSettings


def _accumulate_segment_path_3d(
    density: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    start_weight: float,
    end_weight: float,
    *,
    maximum_spacing: float,
) -> None:
    """Exercise the production accumulator with one dependency-light segment."""
    accumulator = PathFieldAccumulator(
        x_edges=x_edges,
        y_edges=y_edges,
        z_edges=z_edges,
        density_zyx=density,
        maximum_spacing_mm=maximum_spacing,
        maximum_samples_per_segment=1024,
    )
    accumulator.accumulate(
        np.asarray(start, dtype=float)[None, :],
        np.asarray(end, dtype=float)[None, :],
        np.asarray([start_weight], dtype=float),
        np.asarray([end_weight], dtype=float),
    )


def test_full_sampling_is_deterministic_and_three_dimensional() -> None:
    from lumo.ray_tracing.optical_mechanics.sampling import sample_directions

    led = LED()
    full_first = sample_directions(led, (0.0, -1.0), ray_count=257)
    full_second = sample_directions(led, (0.0, -1.0), ray_count=257)
    assert np.array_equal(full_first, full_second)
    assert np.allclose(np.linalg.norm(full_first, axis=1), 1.0)
    assert np.any(np.abs(full_first[:, 2]) > 0.0)


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
        outgoing_surface_field=np.full((2, 2), 0.25),
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
    assert result.escape_event_count == 1
    assert result.escaped_primary_count == 1

    with pytest.raises(Transport3DResultError, match="escape metadata"):
        replace(result, escape_primary_ray_indices=np.asarray([3]))
    with pytest.raises(Transport3DResultError, match="escape weights"):
        replace(result, escape_weights=np.asarray([0.5]))
    with pytest.raises(Transport3DResultError, match="surface field"):
        replace(result, outgoing_surface_field=np.full((2, 2), 0.125))
    with pytest.raises(Transport3DResultError, match="cannot exceed"):
        replace(
            result,
            outgoing_surface_weight=1.1,
            outgoing_surface_field=np.full((2, 2), 0.275),
            escape_weights=np.asarray([1.1]),
        )
    with pytest.raises(Transport3DResultError, match="energy_balance_error"):
        replace(result, energy_balance_error=1.0e-6)
    with pytest.raises(Transport3DResultError, match="object interface"):
        replace(result, object_interface_incident_weight=0.1)
    with pytest.raises(Transport3DResultError, match="zero segment_budget"):
        replace(
            result,
            escaped_weight=0.9,
            terminated_weight=0.1,
            segment_budget_termination_count=0,
            segment_budget_termination_weight=0.1,
            outgoing_surface_weight=0.9,
            outgoing_surface_field=np.full((2, 2), 0.225),
            escape_weights=np.asarray([0.9]),
        )

    envelope_escape = replace(
        result,
        outgoing_surface_weight=0.5,
        outgoing_surface_field=np.full((2, 2), 0.125),
        escape_weights=np.asarray([0.5]),
    )
    assert envelope_escape.escaped_weight == 1.0
    assert envelope_escape.outgoing_surface_weight == 0.5


def test_optical_xy_bounds_must_be_provided_as_a_pair() -> None:
    with pytest.raises(ValueError, match="x_bounds_mm and y_bounds_mm"):
        Transport3DSettings(x_bounds_mm=(-1.0, 1.0))
    with pytest.raises(ValueError, match="x_bounds_mm and y_bounds_mm"):
        Transport3DSettings(y_bounds_mm=(-1.0, 1.0))


def test_simple_internal_path_accumulation_and_z_integration() -> None:
    density = np.zeros((2, 2, 2), dtype=float)
    repeated_density = np.zeros((2, 2, 2), dtype=float)
    x_edges = np.asarray([0.0, 1.0, 2.0])
    y_edges = np.asarray([0.0, 1.0, 2.0])
    z_edges = np.asarray([-1.0, 0.0, 1.0])
    _accumulate_segment_path_3d(
        density,
        x_edges,
        y_edges,
        z_edges,
        np.asarray([0.25, 0.25, -0.5]),
        np.asarray([0.25, 0.25, 0.5]),
        1.0,
        0.5,
        maximum_spacing=0.25,
    )
    _accumulate_segment_path_3d(
        repeated_density,
        x_edges,
        y_edges,
        z_edges,
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
    _accumulate_segment_path_3d(
        density,
        edges,
        edges,
        z_edges,
        np.asarray([0.5, 0.5, -0.5]),
        np.asarray([0.5, 0.5, 0.5]),
        1.0,
        first_end,
        maximum_spacing=0.25,
    )
    _accumulate_segment_path_3d(
        density,
        edges,
        edges,
        z_edges,
        np.asarray([0.5, 0.5, -0.5]),
        np.asarray([0.5, 0.5, 0.5]),
        first_end,
        second_end,
        maximum_spacing=0.25,
    )

    assert np.sum(density) == pytest.approx(0.5 * (1.0 + first_end) + 0.5 * (first_end + second_end))
    assert first_end + first_removed == pytest.approx(1.0)
    assert second_end + second_removed == pytest.approx(first_end)


def test_path_field_sampling_cap_preserves_total_weighted_length() -> None:
    accumulator = PathFieldAccumulator(
        x_edges=np.asarray([0.0, 1.0, 2.0]),
        y_edges=np.asarray([0.0, 1.0]),
        z_edges=np.asarray([-1.0, 1.0]),
        density_zyx=np.zeros((1, 1, 2), dtype=float),
        maximum_spacing_mm=0.01,
        maximum_samples_per_segment=2,
    )
    accumulator.accumulate(
        np.asarray([[0.25, 0.5, 0.0]]),
        np.asarray([[1.75, 0.5, 0.0]]),
        np.asarray([1.0]),
        np.asarray([1.0]),
    )

    assert np.sum(accumulator.density_zyx) == pytest.approx(1.5)
    assert accumulator.processed_segment_count == 1


def test_path_field_clipping_diagnostics_conserve_active_weighted_length() -> None:
    accumulator = PathFieldAccumulator(
        x_edges=np.asarray([0.0, 1.0]),
        y_edges=np.asarray([0.0, 1.0]),
        z_edges=np.asarray([0.0, 1.0]),
        density_zyx=np.zeros((1, 1, 1), dtype=float),
        maximum_spacing_mm=0.5,
        maximum_samples_per_segment=16,
    )
    accumulator.accumulate(
        np.asarray([[0.25, 0.5, 0.5], [2.0, 0.5, 0.5]]),
        np.asarray([[1.75, 0.5, 0.5], [3.0, 0.5, 0.5]]),
        np.asarray([1.0, 1.0]),
        np.asarray([1.0, 1.0]),
    )

    diagnostics = accumulator.diagnostics
    assert diagnostics.processed_sample_count == 5
    assert diagnostics.clipped_sample_count == 3
    assert diagnostics.represented_weighted_path_length_mm == pytest.approx(1.0)
    assert diagnostics.clipped_weighted_path_length_mm == pytest.approx(1.5)
    assert diagnostics.processed_weighted_path_length_mm == pytest.approx(2.5)
    assert np.all(accumulator.density_zyx >= 0.0)
    assert np.sum(accumulator.density_zyx) == pytest.approx(
        diagnostics.represented_weighted_path_length_mm
    )


def test_path_field_reports_an_all_clipped_active_segment() -> None:
    accumulator = PathFieldAccumulator(
        x_edges=np.asarray([0.0, 1.0]),
        y_edges=np.asarray([0.0, 1.0]),
        z_edges=np.asarray([0.0, 1.0]),
        density_zyx=np.zeros((1, 1, 1), dtype=float),
        maximum_spacing_mm=0.5,
        maximum_samples_per_segment=16,
    )
    accumulator.accumulate(
        np.asarray([[2.0, 0.5, 0.5]]),
        np.asarray([[3.0, 0.5, 0.5]]),
        np.asarray([1.0]),
        np.asarray([1.0]),
    )

    diagnostics = accumulator.diagnostics
    assert diagnostics.processed_sample_count > 0
    assert diagnostics.clipped_sample_count == diagnostics.processed_sample_count
    assert diagnostics.represented_weighted_path_length_mm == pytest.approx(0.0)
    assert diagnostics.clipped_weighted_path_length_mm == pytest.approx(1.0)
    assert np.sum(accumulator.density_zyx) == pytest.approx(0.0)


def test_path_field_padding_is_inactive_and_nonfinite_batches_fail_closed() -> None:
    accumulator = PathFieldAccumulator(
        x_edges=np.asarray([0.0, 2.0]),
        y_edges=np.asarray([0.0, 2.0]),
        z_edges=np.asarray([0.0, 2.0]),
        density_zyx=np.zeros((1, 1, 1), dtype=float),
        maximum_spacing_mm=0.5,
        maximum_samples_per_segment=16,
    )
    accumulator.accumulate(
        np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        np.asarray([[1.1, 1.0, 1.0], [2.0, 1.0, 1.0]]),
        np.asarray([1.0, 1.0]),
        np.asarray([1.0, 1.0]),
    )
    assert accumulator.processed_sample_count == 3
    assert accumulator.clipped_sample_count == 0

    with pytest.raises(ValueError, match="finite"):
        accumulator.accumulate(
            np.asarray([[np.nan, 0.0, 0.0]]),
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([1.0]),
            np.asarray([1.0]),
        )


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
        outgoing_surface_field=np.full((2, 2), 0.25),
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
    )
    assert result.escape_event_count == 1
    assert result.escaped_primary_count == 1


def test_normal_optics_import_does_not_load_optional_optix() -> None:
    from lumo import ray_tracing

    assert not hasattr(ray_tracing, "trace")
    assert hasattr(ray_tracing, "CarrierOptics")
    assert "optix" not in sys.modules
