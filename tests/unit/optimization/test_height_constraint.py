"""Regression tests for the physical 30 mm fingertip-height envelope."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.optimization.ax_bo import (
    _DEFAULT_CONTACT_Y_MM,
    _DISCRETE_MAX_PAD_DEPTH_STEPS,
    _OBJECTIVE_NAMES,
    _campaign_definition,
    _decode_ax_parameters,
    _new_client,
    _run_config,
    _validate_campaign_parameters,
)
from lumo.optimization.design_space import MAX_FINGERTIP_HEIGHT_MM


_PRODUCTION_BOUNDS_MM = {
    "flat_pad_height_mm": (2.0, 29.0),
    "semiellipse_height_mm": (1.0, 20.0),
    "stem_width_mm": (4.0, 15.0),
    "stem_height_mm": (2.0, 15.0),
    "void_width_mm": (0.0, 4.0),
}


def _fingertip(
    flat_height_mm: float,
    ellipse_height_mm: float,
    *,
    stem_width_mm: float = 7.6,
    stem_height_mm: float = 6.0,
    void_width_mm: float = 2.0,
    void_height_mm: float = 0.0,
) -> Fingertip:
    parameters = FingertipParameters()
    geometry = replace(
        parameters.geometry,
        flat_pad_height_mm=flat_height_mm,
        semiellipse_height_mm=ellipse_height_mm,
        stem_width_mm=stem_width_mm,
        stem_height_mm=stem_height_mm,
        void_width_mm=void_width_mm,
        void_height_mm=void_height_mm,
    )
    return Fingertip(replace(parameters, geometry=geometry))


def _campaign():
    return _campaign_definition(
        "discrete-05mm",
        parameter_bounds_mm=_PRODUCTION_BOUNDS_MM,
    )


def test_full_finger_campaign_is_five_dimensional() -> None:
    campaign = _campaign()

    assert campaign.space.variable_names == (
        "geometry.flat_pad_height_mm",
        "geometry.semiellipse_height_mm",
        "geometry.stem_width_mm",
        "geometry.stem_height_mm",
        "geometry.void_width_mm",
    )
    assert dict(campaign.fixed_geometry) == {
        "geometry.flat_pad_width_mm": 30.0,
        "geometry.void_height_mm": 0.0,
    }
    geometry = campaign.space.parameter_bounds.parameters.geometry
    assert geometry.flat_pad_width_mm == 30.0
    assert geometry.void_height_mm == 0.0


def test_production_scientific_contract_is_explicitly_serialized() -> None:
    campaign = _campaign()
    config = _run_config(campaign)["scientific_contract"]

    assert campaign.contact_y_mm == (
        -22.0,
        -11.0,
        -5.5,
        0.0,
        5.5,
        11.0,
        22.0,
    )
    assert campaign.contact_y_mm == _DEFAULT_CONTACT_Y_MM
    assert config["scenarios"]["contact_y_mm"] == list(campaign.contact_y_mm)
    assert config["mechanics"]["force_targets_n"] == [5.0, 10.0, 15.0, 20.0]
    assert config["mechanics"]["loading_mode"] == "reference_dwell"
    assert config["mechanics"]["sim_frequency_hz"] == 100.0
    assert config["mechanics"]["vbd_iterations"] == 10
    assert config["mechanics"]["fixed_servo_dwell_s"] == 5.0
    assert config["mechanics"]["force_tolerance_fraction"] == 0.1
    assert config["optics"]["led_centers_y_mm"] == [-22.0, -11.0, 0.0, 11.0, 22.0]
    assert config["optics"]["observation_view_direction"] == "+X"
    assert config["optics"]["spatial_roi_y_mm"] == [-27.5, 27.5]
    assert config["optics"]["spatial_bin_count"] == 11
    assert config["optics"]["spatial_bin_width_mm"] == 5.0
    assert config["design_space"]["fixed"]["geometry.void_height_mm"] == 0.0
    assert config["design_space"]["full_fingertip_height_max_mm"] == 30.0
    assert tuple(config["objectives"]["names"]) == _OBJECTIVE_NAMES


def test_ax_uses_only_the_frozen_full_finger_objectives() -> None:
    optimization = _new_client(_campaign())._experiment.optimization_config

    assert optimization.objective.metric_names == ["J_contact", "J_obs"]
    assert optimization.objective.expression == "J_contact, J_obs"
    assert optimization.objective_thresholds == []


def _candidate(
    flat_height_mm: float,
    ellipse_height_mm: float,
    *,
    stem_width_mm: float = 8.0,
    stem_height_mm: float = 4.0,
    void_width_mm: float = 0.0,
) -> dict[str, float]:
    return {
        "geometry.flat_pad_height_mm": flat_height_mm,
        "geometry.semiellipse_height_mm": ellipse_height_mm,
        "geometry.stem_width_mm": stem_width_mm,
        "geometry.stem_height_mm": stem_height_mm,
        "geometry.void_width_mm": void_width_mm,
    }


def test_physical_height_boundary_and_historical_designs() -> None:
    campaign = _campaign()

    nominal = _fingertip(5.0, 9.0)
    assert nominal.full_height_mm == 24.0

    boundary = _candidate(12.0, 8.0)
    boundary_fingertip = _fingertip(
        12.0,
        8.0,
        stem_width_mm=8.0,
        stem_height_mm=4.0,
        void_width_mm=0.0,
    )
    assert boundary_fingertip.full_height_mm == 30.0
    assert campaign.space.is_feasible(boundary)

    over_boundary = _candidate(12.5, 8.0)
    over_fingertip = _fingertip(
        12.5,
        8.0,
        stem_width_mm=8.0,
        stem_height_mm=4.0,
        void_width_mm=0.0,
    )
    assert over_fingertip.full_height_mm == 30.5
    assert not campaign.space.is_feasible(over_boundary)

    dragon_123 = _candidate(
        13.5,
        14.0,
        stem_width_mm=7.0,
        stem_height_mm=4.0,
        void_width_mm=3.0,
    )
    solaris_107 = _candidate(
        20.0,
        5.5,
        stem_width_mm=13.0,
        stem_height_mm=2.5,
        void_width_mm=0.5,
    )
    assert not campaign.space.is_feasible(dragon_123)
    assert not campaign.space.is_feasible(solaris_107)


def test_ax_step_constraint_matches_physical_height_on_full_lattice() -> None:
    campaign = _campaign()
    step_bounds = {
        step_name: (lower, upper)
        for step_name, _, lower, upper in campaign.discrete_step_to_physical
    }
    flat_lower, flat_upper = step_bounds["flat_pad_height_step"]
    ellipse_lower, ellipse_upper = step_bounds["semiellipse_height_step"]

    for flat_step in range(flat_lower, flat_upper + 1):
        for ellipse_step in range(ellipse_lower, ellipse_upper + 1):
            ax_feasible = (
                flat_step + ellipse_step <= _DISCRETE_MAX_PAD_DEPTH_STEPS
            )
            fingertip = _fingertip(
                0.5 * flat_step,
                0.5 * ellipse_step,
                stem_width_mm=4.0,
                stem_height_mm=2.0,
                void_width_mm=0.0,
            )
            physical_feasible = (
                fingertip.full_height_mm <= MAX_FINGERTIP_HEIGHT_MM
            )
            assert ax_feasible == physical_feasible


def test_encoded_boundary_is_accepted_and_next_step_is_rejected() -> None:
    campaign = _campaign()
    boundary = {
        "flat_pad_height_step": 24,
        "semiellipse_height_step": 16,
        "stem_width_step": 16,
        "stem_height_step": 8,
        "void_width_step": 0,
    }
    parameters = _decode_ax_parameters(campaign, boundary)
    _validate_campaign_parameters(campaign, boundary, parameters)
    assert campaign.space.is_feasible(parameters)

    over_boundary = dict(boundary, flat_pad_height_step=25)
    over_parameters = _decode_ax_parameters(campaign, over_boundary)
    with pytest.raises(ValueError, match="full-height constraint"):
        _validate_campaign_parameters(campaign, over_boundary, over_parameters)


@pytest.mark.parametrize(
    ("flat_height_mm", "ellipse_height_mm", "expected_valid"),
    (
        (12.0, 8.0, True),
        (12.5, 7.5, True),
        (12.5, 8.0, False),
        (19.0, 1.0, True),
        (19.5, 1.0, False),
    ),
)
def test_encoded_half_millimetre_boundary_examples(
    flat_height_mm: float,
    ellipse_height_mm: float,
    expected_valid: bool,
) -> None:
    campaign = _campaign()
    raw_parameters = {
        "flat_pad_height_step": round(2.0 * flat_height_mm),
        "semiellipse_height_step": round(2.0 * ellipse_height_mm),
        "stem_width_step": 16,
        "stem_height_step": 8,
        "void_width_step": 0,
    }
    parameters = _decode_ax_parameters(campaign, raw_parameters)

    if expected_valid:
        _validate_campaign_parameters(campaign, raw_parameters, parameters)
        assert campaign.space.is_feasible(parameters)
    else:
        with pytest.raises(ValueError, match="full-height constraint"):
            _validate_campaign_parameters(campaign, raw_parameters, parameters)
        assert not campaign.space.is_feasible(parameters)
