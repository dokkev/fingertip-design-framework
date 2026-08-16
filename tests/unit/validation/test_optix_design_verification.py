"""Focused contracts for the sensor-facing OptiX analysis layer."""

from __future__ import annotations

import numpy as np
import pytest

from validation.optics.optix_design_verification import (
    camera_discretization_diagnostic,
    centered_jacobian,
    derivative_diagnostics,
    fisher_from_response,
    pairwise_separation,
    project_escape_events,
)


def _camera() -> dict[str, object]:
    return {
        "camera_position_mm": [10.0, 0.0, 0.0],
        "camera_target_mm": [0.0, 0.0, 0.0],
        "camera_up": [0.0, 0.0, 1.0],
        "sensor_y_bounds_mm": [-1.0, 1.0],
        "sensor_z_bounds_mm": [-1.0, 1.0],
        "resolution_px": [4, 4],
    }


def test_camera_projection_uses_external_direction_and_preserves_power() -> None:
    response, record = project_escape_events(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        np.asarray([1.0, 2.0, 4.0, 8.0]),
        _camera(),
    )
    assert record["accepted_event_count"] == 2
    assert record["accepted_camera_power"] == pytest.approx(3.0)
    assert float(np.sum(response)) == pytest.approx(3.0)


def test_pairwise_separation_does_not_normalize_total_response() -> None:
    result = pairwise_separation(
        np.asarray([[2.0, 0.0]]),
        np.asarray([[1.0, 0.0]]),
        read_noise_sigma=0.5,
    )
    assert result["first_total_power"] == pytest.approx(2.0)
    assert result["second_total_power"] == pytest.approx(1.0)
    assert result["noise_normalized_d2"] == pytest.approx(4.0)


def test_pairwise_gain_profile_removes_global_brightness_only() -> None:
    result = pairwise_separation(
        np.asarray([[2.0, 4.0]]),
        np.asarray([[1.0, 2.0]]),
        read_noise_sigma=1.0,
    )
    assert result["best_global_gain_second_to_first"] == pytest.approx(2.0)
    assert result["gain_profiled_shape_d2"] == pytest.approx(0.0)
    assert result["gain_profiled_interpretation"] == "photometric_dominant"


def test_camera_discretization_rebins_without_changing_accepted_power() -> None:
    states = (
        "unloaded",
        "left_contact",
        "right_contact",
        "near_left_contact",
        "near_right_contact",
        "center_low",
        "center_shallow",
        "center_high",
        "center_deep",
    )
    event_data = {}
    for morphology, offset in (("nominal", 0.0), ("candidate49", 0.1)):
        event_data[morphology] = {}
        for index, state in enumerate(states):
            event_data[morphology][state] = {
                "positions_mm": np.asarray([[0.0, offset + 0.01 * index, 0.0]]),
                "directions": np.asarray([[1.0, 0.0, 0.0]]),
                "weights": np.asarray([1.0]),
            }
    diagnostic, arrays = camera_discretization_diagnostic(
        event_data,
        resolutions=((4, 4), (2, 2)),
        read_noise_sigma=1.0,
    )
    assert diagnostic["assessment"]["total_accepted_camera_power_invariant"]["nominal"]["pass"]
    assert diagnostic["assessment"]["total_accepted_camera_power_invariant"]["candidate49"]["pass"]
    assert "4x4_nominal_center_shallow_mu" in arrays
    assert diagnostic["resolutions"]["4x4"]["morphologies"]["nominal"]["states"]["center_shallow"]["response_shape"] == [4, 4]


def test_centered_jacobian_uses_physical_stencils() -> None:
    responses = {
        "left_contact": np.asarray([[1.0, 2.0]]),
        "center_shallow": np.asarray([[2.0, 2.0]]),
        "right_contact": np.asarray([[3.0, 2.0]]),
        "unloaded": np.asarray([[2.0, 1.0]]),
        "center_deep": np.asarray([[2.0, 3.0]]),
        "near_left_contact": np.asarray([[1.5, 2.0]]),
        "near_right_contact": np.asarray([[2.5, 2.0]]),
        "center_low": np.asarray([[2.0, 1.5]]),
        "center_high": np.asarray([[2.0, 2.5]]),
    }
    jacobian, metadata = centered_jacobian(responses)
    np.testing.assert_allclose(
        jacobian,
        np.asarray([[1.0 / 3.0, 0.0], [0.0, 2.0]]),
    )
    assert metadata["method"] == "centered finite differences"


def test_gain_marginalization_cannot_increase_contact_information() -> None:
    response = np.asarray([[1.0, 2.0, 3.0, 4.0]])
    jacobian = np.asarray([[0.0, 1.0, 0.0, -1.0], [1.0, 0.0, -1.0, 0.0]]).T
    result = fisher_from_response(
        response,
        jacobian,
        read_noise_sigma=1.0,
    )
    assert result["fisher"]["symmetry_check_pass"]
    assert result["fisher"]["psd_check_pass"]
    assert result["gain_nuisance"]["marginalization_does_not_increase_information"]
    unmarginalized = np.asarray(result["fisher"]["fisher_physical"])
    marginalized = np.asarray(result["gain_nuisance"]["effective_contact_fisher"])
    assert np.min(np.linalg.eigvalsh(unmarginalized - marginalized)) >= -1.0e-12


def test_derivative_gate_surfaces_inconsistent_stencil_directions() -> None:
    responses = {
        "left_contact": np.asarray([[0.0, 0.0]]),
        "center_shallow": np.asarray([[1.0, 0.0]]),
        "right_contact": np.asarray([[0.0, 0.0]]),
        "near_left_contact": np.asarray([[0.0, 0.0]]),
        "near_right_contact": np.asarray([[0.0, 0.0]]),
        "unloaded": np.asarray([[0.0, 0.0]]),
        "center_low": np.asarray([[0.0, 0.0]]),
        "center_high": np.asarray([[0.0, 0.0]]),
        "center_deep": np.asarray([[0.0, 0.0]]),
    }
    result = derivative_diagnostics(responses)
    assert not result["validity"]["pass"]
    assert not result["validity"]["checks"]["position_outer_one_sided"]


def test_singular_fisher_is_reported_without_hidden_regularization() -> None:
    result = fisher_from_response(
        np.asarray([[1.0, 1.0]]),
        np.asarray([[1.0, 1.0], [2.0, 2.0]]),
        read_noise_sigma=1.0,
    )
    assert result["fisher"]["status"] == "rank_deficient_or_non_psd"
    assert result["fisher"]["condition_number"] is None
    assert result["fisher"]["log_determinant"] is None
