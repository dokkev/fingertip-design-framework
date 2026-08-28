"""Regression tests for the production fingertip objectives."""

from __future__ import annotations

from inspect import signature
from pathlib import Path

import numpy as np
import pytest

from lumo.fingertip import Fingertip
from lumo.optimization.evaluator import (
    _indenter_trajectory,
    _scenario_specifications,
    evaluate_fingertip,
)
from lumo.optimization.objective import (
    compute_contact_objective,
    compute_observation_objective,
)
from lumo.ray_tracing import longitudinal_side_view_power


def test_production_loading_is_gpu_force_threshold_crossing() -> None:
    parameters = signature(evaluate_fingertip).parameters
    for obsolete_option in (
        "use_cuda_graph",
        "reuse_finalized_models",
        "reuse_runtimes",
        "parallel_world_count",
        "loading_mode",
        "settle_duration_s",
    ):
        assert obsolete_option not in parameters


def _observation_inputs() -> dict[str, object]:
    diameters = np.repeat((10.0, 20.0), 3)
    locations = np.tile((-5.5, 0.0, 5.5), 2)
    response = np.zeros((6, 2, 5, 3), dtype=np.float64)
    for scenario_index, (diameter, location) in enumerate(
        zip(diameters, locations, strict=True)
    ):
        for force_index, force in enumerate((5.0, 10.0)):
            response[scenario_index, force_index, :, 0] = (
                0.01 * diameter + 0.1 * force
            ) / 5.0
            response[scenario_index, force_index, :, 1] = location / 5.0
    return {
        "response_matrix": response,
        "no_contact_response": np.zeros((5, 3)),
        "scenario_names": tuple(f"scenario_{index}" for index in range(6)),
        "sphere_diameters_mm": diameters,
        "contact_angles_deg": np.zeros(6),
        "contact_y_mm": locations,
        "force_targets_n": np.array((5.0, 10.0)),
        "emitted_power": 5.0,
    }


def test_observation_uses_smallest_same_force_location_distance() -> None:
    inputs = _observation_inputs()

    result = compute_observation_objective(**inputs)

    assert result.J_obs == pytest.approx(1.1)
    assert result.limiting_contact_y_pair_mm == (-5.5, 0.0)
    assert result.limiting_sphere_diameter_mm == 10.0
    assert result.limiting_contact_angle_deg == 0.0
    assert result.limiting_force_n == 5.0


def test_orientation_scenarios_are_the_complete_ordered_product() -> None:
    paths = tuple(Path(f"sphere_{diameter:g}mm.urdf") for diameter in (10, 15, 20))
    angles = (-30.0, -15.0, 0.0, 15.0, 30.0)
    locations = (-11.0, -5.5, 0.0, 5.5, 11.0)

    scenarios = _scenario_specifications(
        paths,
        (10.0, 15.0, 20.0),
        angles,
        locations,
    )

    assert len(scenarios) == 75
    assert len(set(scenarios)) == 75
    assert scenarios[:5] == tuple(
        (paths[0], 10.0, -30.0, location) for location in locations
    )
    assert scenarios[25] == (paths[1], 15.0, -30.0, -11.0)
    assert scenarios[-1] == (paths[2], 20.0, 30.0, 11.0)


def test_indenter_trajectory_uses_inverse_y_rotation_about_the_fixed_pivot() -> None:
    fingertip = Fingertip()
    ordinary_center, ordinary_direction = _indenter_trajectory(
        fingertip,
        sphere_diameter_mm=20.0,
        contact_y_mm=5.5,
        fingertip_angle_deg=0.0,
        initial_clearance_m=0.010,
    )
    np.testing.assert_array_equal(
        ordinary_center,
        np.array((0.0, 0.0055, fingertip.tip_z_m - 0.020)),
    )
    np.testing.assert_array_equal(ordinary_direction, np.array((0.0, 0.0, 1.0)))

    positive_center, positive_direction = _indenter_trajectory(
        fingertip,
        sphere_diameter_mm=20.0,
        contact_y_mm=5.5,
        fingertip_angle_deg=30.0,
        initial_clearance_m=0.010,
    )
    negative_center, negative_direction = _indenter_trajectory(
        fingertip,
        sphere_diameter_mm=20.0,
        contact_y_mm=5.5,
        fingertip_angle_deg=-30.0,
        initial_clearance_m=0.010,
    )
    relative_z = ordinary_center[2]
    expected_positive = np.array(
        (-np.sin(np.deg2rad(30.0)) * relative_z, 0.0055, np.cos(np.deg2rad(30.0)) * relative_z)
    )
    expected_negative = np.array(
        (np.sin(np.deg2rad(30.0)) * relative_z, 0.0055, np.cos(np.deg2rad(30.0)) * relative_z)
    )
    np.testing.assert_allclose(positive_center, expected_positive, atol=1.0e-15)
    np.testing.assert_allclose(negative_center, expected_negative, atol=1.0e-15)
    np.testing.assert_allclose(
        positive_direction,
        (-0.5, 0.0, np.sqrt(3.0) / 2.0),
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        negative_direction,
        (0.5, 0.0, np.sqrt(3.0) / 2.0),
        atol=1.0e-15,
    )


def test_observation_never_compares_across_contact_angles() -> None:
    response = np.zeros((4, 1, 5, 1), dtype=np.float64)
    response[1, :, :, 0] = 1.0
    response[2, :, :, 0] = 1.0001
    response[3, :, :, 0] = 2.0001

    result = compute_observation_objective(
        response_matrix=response,
        no_contact_response=np.zeros((5, 1)),
        scenario_names=("a0_y0", "a0_y1", "a1_y0", "a1_y1"),
        sphere_diameters_mm=np.full(4, 10.0),
        contact_angles_deg=np.repeat((-30.0, 30.0), 2),
        contact_y_mm=np.tile((0.0, 5.5), 2),
        force_targets_n=np.array((1.0,)),
        emitted_power=5.0,
    )

    assert result.J_obs == pytest.approx(1.0)
    assert result.location_separations.shape == (1, 2, 1, 2, 2)


def test_led_permutation_cannot_change_combined_field_or_objective() -> None:
    inputs = _observation_inputs()
    response = inputs["response_matrix"]
    baseline = inputs["no_contact_response"]
    permutation = (3, 0, 4, 1, 2)

    combined = response.sum(axis=-2)
    permuted_combined = response[:, :, permutation, :].sum(axis=-2)
    original = compute_observation_objective(**inputs)
    permuted = compute_observation_objective(
        **{
            **inputs,
            "response_matrix": response[:, :, permutation, :],
            "no_contact_response": baseline[permutation, :],
        }
    )

    np.testing.assert_allclose(permuted_combined, combined, rtol=0.0, atol=1.0e-14)
    assert permuted.J_obs == pytest.approx(original.J_obs, abs=1.0e-14)


def test_contact_objective_components_match_definition() -> None:
    reference_vertices = np.array(
        ((0.0, 0.0, 0.0), (0.001, 0.0, 0.0), (0.0, 0.001, 0.0))
    )
    contact_indices = np.tile((0, 1, 2, -1), (4, 1))
    offsets = np.array(((((0, 1), (1, 1), (2, 1), (3, 1))),))
    vertices = np.tile(reference_vertices, (1, 4, 1, 1))
    vertices[0, 1] *= 2.0
    formation_area_m2 = 2.0e-6

    result = compute_contact_objective(
        reference_vertices_m=reference_vertices,
        surface_triangles=np.array(((0, 1, 2),)),
        scenario_names=("nominal",),
        sphere_diameters_mm=np.array((10.0,)),
        contact_angles_deg=np.array((0.0,)),
        contact_y_mm=np.array((0.0,)),
        force_targets_n=np.array((1.0, 2.0, 5.0, 10.0)),
        actual_forces_n=np.array(((1.0, 2.0, 5.0, 10.0),)),
        indentations_m=np.array(((0.001, 0.002, 0.0025, 0.00275),)),
        contact_record_offsets=offsets,
        contact_particle_indices=contact_indices,
        contact_normals_W=np.tile((0.0, 0.0, 1.0), (4, 1)),
        silicone_vertices_m=vertices,
    )

    assert result.q_form[0] == pytest.approx(
        np.sqrt(formation_area_m2 / (np.pi * 0.005**2))
    )
    assert result.patch_area_formation_m2[0] == pytest.approx(formation_area_m2)
    assert result.q_stable[0] == pytest.approx(1.0)
    assert result.q_stiff[0] == pytest.approx(0.95)
    assert result.q_normal[0] == pytest.approx(1.0)
    assert result.J_contact == pytest.approx(
        np.cbrt(result.q_form[0] * result.q_stiff[0])
    )


def test_contact_objective_selects_the_worst_angle_condition() -> None:
    reference_vertices = np.array(
        ((0.0, 0.0, 0.0), (0.001, 0.0, 0.0), (0.0, 0.001, 0.0))
    )
    contact_indices = np.tile((0, 1, 2, -1), (8, 1))
    offsets = np.arange(8, dtype=np.int64).reshape(2, 4, 1)
    offsets = np.concatenate((offsets, np.ones_like(offsets)), axis=2)
    vertices = np.tile(reference_vertices, (2, 4, 1, 1))

    result = compute_contact_objective(
        reference_vertices_m=reference_vertices,
        surface_triangles=np.array(((0, 1, 2),)),
        scenario_names=("theta-30", "theta+30"),
        sphere_diameters_mm=np.array((20.0, 20.0)),
        contact_angles_deg=np.array((-30.0, 30.0)),
        contact_y_mm=np.array((5.5, 5.5)),
        force_targets_n=np.array((1.0, 2.0, 5.0, 10.0)),
        actual_forces_n=np.tile((1.0, 2.0, 5.0, 10.0), (2, 1)),
        indentations_m=np.array(
            ((0.001, 0.002, 0.0025, 0.00275), (0.001, 0.002, 0.003, 0.008))
        ),
        contact_record_offsets=offsets,
        contact_particle_indices=contact_indices,
        contact_normals_W=np.tile((0.0, 0.0, 1.0), (8, 1)),
        silicone_vertices_m=vertices,
    )

    assert result.limiting_scenario == "theta+30"
    assert result.limiting_contact_angle_deg == 30.0
    assert result.limiting_sphere_diameter_mm == 20.0
    assert result.limiting_contact_y_mm == 5.5
    assert result.J_contact == pytest.approx(0.0)


def test_longitudinal_roi_bins_plus_outside_equal_visible_power() -> None:
    ray_dtype = np.dtype(
        [
            ("origin_W_m", np.float64, (3,)),
            ("direction_W", np.float64, (3,)),
            ("power", np.float64),
        ]
    )
    rays = np.zeros(5, dtype=ray_dtype)
    rays["origin_W_m"][:, 1] = (-0.020, 0.030, 0.0275, 0.0, -0.030)
    rays["direction_W"][:, 0] = (1.0, 1.0, 1.0, -1.0, 1.0)
    rays["power"] = (1.0, 2.0, 3.0, 4.0, 5.0)

    bins, outside, visible = longitudinal_side_view_power(rays)

    assert bins.sum() == pytest.approx(4.0)
    assert outside == pytest.approx(7.0)
    assert visible == pytest.approx(11.0)
    assert bins.sum() + outside == pytest.approx(visible)
