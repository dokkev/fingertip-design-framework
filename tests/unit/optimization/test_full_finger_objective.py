"""Regression tests for the production full-finger objectives."""

from __future__ import annotations

from inspect import signature

import numpy as np
import pytest

from lumo.optimization.evaluator import evaluate_full_finger
from lumo.optimization.objective import (
    combine_led_responses,
    compute_contact_objective,
    compute_observation_objective,
)
from lumo.ray_tracing import longitudinal_side_view_power
from lumo.simulation import DesignStudy, REFERENCE_DWELL_LOADING


def test_production_loading_default_remains_reference_dwell() -> None:
    assert (
        signature(evaluate_full_finger).parameters["loading_mode"].default
        == REFERENCE_DWELL_LOADING
    )
    assert (
        signature(DesignStudy.run).parameters["loading_mode"].default
        == REFERENCE_DWELL_LOADING
    )


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
    assert result.limiting_force_n == 5.0


def test_led_permutation_cannot_change_combined_field_or_objective() -> None:
    inputs = _observation_inputs()
    response = inputs["response_matrix"]
    baseline = inputs["no_contact_response"]
    permutation = (3, 0, 4, 1, 2)

    combined = combine_led_responses(response)
    permuted_combined = combine_led_responses(response[:, :, permutation, :])
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
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    )
    contact_indices = np.tile((0, 1, 2, -1), (4, 1))
    offsets = np.array(((((0, 1), (1, 1), (2, 1), (3, 1))),))

    result = compute_contact_objective(
        reference_vertices_m=reference_vertices,
        surface_triangles=np.array(((0, 1, 2),)),
        scenario_names=("nominal",),
        sphere_diameters_mm=np.array((10.0,)),
        force_targets_n=np.array((5.0, 10.0, 15.0, 20.0)),
        actual_forces_n=np.array(((5.0, 10.0, 15.0, 20.0),)),
        indentations_m=np.array(((0.001, 0.002, 0.0025, 0.00275),)),
        contact_record_offsets=offsets,
        contact_particle_indices=contact_indices,
        contact_normals_W=np.tile((0.0, 0.0, 1.0), (4, 1)),
        silicone_vertices_m=np.tile(reference_vertices, (1, 4, 1, 1)),
    )

    assert result.q_form[0] == pytest.approx(1.0)
    assert result.q_stable[0] == pytest.approx(1.0)
    assert result.q_stiff[0] == pytest.approx(0.75)
    assert result.q_normal[0] == pytest.approx(1.0)
    assert result.J_contact == pytest.approx(np.cbrt(0.75))


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


def test_linear_longitudinal_splat_conserves_power_and_removes_bin_jump() -> None:
    ray_dtype = np.dtype(
        [
            ("origin_W_m", np.float64, (3,)),
            ("direction_W", np.float64, (3,)),
            ("power", np.float64),
        ]
    )

    def response_at(y_m: float, *, linear_splat: bool) -> np.ndarray:
        rays = np.zeros(1, dtype=ray_dtype)
        rays["origin_W_m"][0, 1] = y_m
        rays["direction_W"][0, 0] = 1.0
        rays["power"][0] = 1.0
        bins, outside, visible = longitudinal_side_view_power(
            rays,
            linear_splat=linear_splat,
        )
        assert bins.sum() + outside == pytest.approx(visible)
        return bins

    hard_left = response_at(-22.5001e-3, linear_splat=False)
    hard_right = response_at(-22.4999e-3, linear_splat=False)
    linear_left = response_at(-22.5001e-3, linear_splat=True)
    linear_right = response_at(-22.4999e-3, linear_splat=True)

    assert np.linalg.norm(hard_left - hard_right, ord=1) == pytest.approx(2.0)
    assert np.linalg.norm(linear_left - linear_right, ord=1) < 1.0e-3
