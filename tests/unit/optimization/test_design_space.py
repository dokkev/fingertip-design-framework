"""Regression tests for the current five-dimensional Ax design space."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.optimization.ax_bo import (
    build_campaign,
    feasible_candidate_pool,
    new_client,
    propose_next_trial,
    propose_feasible_trial,
)
from lumo.optimization.campaign_io import build_run_config
from lumo.optimization.design_space import MAX_FINGERTIP_HEIGHT_MM


_BOUNDS_MM = {
    "flat_pad_height_mm": (2.0, 29.0),
    "semiellipse_height_mm": (1.0, 20.0),
    "stem_width_mm": (4.0, 15.0),
    "stem_height_mm": (2.0, 15.0),
    "void_width_mm": (0.0, 7.5),
}
_INDENTER_URDFS = (
    "sphere_10mm.urdf",
    "sphere_15mm.urdf",
    "sphere_20mm.urdf",
)
_SPHERE_DIAMETERS_MM = (10.0, 15.0, 20.0)
_INDENTATION_ANGLES_DEG = (-30.0, -15.0, 0.0, 15.0, 30.0)
_FORCE_TARGETS_N = (1.0, 2.0, 5.0, 10.0)
_CONTACT_Y_MM = (-11.0, -5.5, 0.0, 5.5, 11.0)
_INITIAL_MORPHOLOGIES_MM = (
    (14.5, 4.0, 5.0, 12.5, 5.0),
    (13.5, 1.5, 7.0, 9.5, 5.5),
    (5.5, 2.0, 9.0, 2.0, 0.5),
    (13.0, 7.0, 7.5, 14.5, 0.0),
    (2.5, 17.5, 7.0, 2.0, 3.5),
)


def _campaign(*, initial_morphologies_mm=()):
    return build_campaign(
        parameter_bounds_mm=_BOUNDS_MM,
        indenter_urdfs=_INDENTER_URDFS,
        sphere_diameters_mm=_SPHERE_DIAMETERS_MM,
        indentation_angles_deg=_INDENTATION_ANGLES_DEG,
        force_targets_n=_FORCE_TARGETS_N,
        initial_clearance_m=1.0e-3,
        contact_y_mm=_CONTACT_Y_MM,
        mechanics_preset="silicone",
        optical_preset="dragon_skin_10_nv_nominal",
        initial_morphologies_mm=initial_morphologies_mm,
    )


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


def test_campaign_is_the_fixed_five_dimensional_half_millimetre_space() -> None:
    campaign = _campaign()

    assert campaign.space.variable_names == tuple(_candidate(5.0, 9.0))
    assert campaign.space.base_parameters.geometry.flat_pad_width_mm == 30.0
    assert campaign.ax_parameter_constraints == (
        "flat_pad_height_step + semiellipse_height_step <= 40",
        "stem_width_step + 2 * void_width_step <= 39",
    )
    config = build_run_config(campaign)["scientific_contract"]
    assert config["design_space"]["resolution_mm"] == 0.5
    assert config["design_space"]["fixed"] == {"geometry.flat_pad_width_mm": 30.0}
    assert config["objectives"]["names"] == ["J_contact", "J_obs"]
    assert config["scenarios"]["theta_deg"] == list(_INDENTATION_ANGLES_DEG)
    assert config["scenarios"]["sphere_diameters_mm"] == list(
        _SPHERE_DIAMETERS_MM
    )
    assert config["scenarios"]["contact_y_mm"] == list(_CONTACT_Y_MM)
    assert config["scenarios"]["orientation_axis"] == "world +Y"
    assert config["scenarios"]["orientation_pivot"] == {"x_m": 0.0, "z_m": 0.0}


def test_design_space_owns_the_complete_height_limit() -> None:
    campaign = _campaign()
    boundary = _candidate(12.0, 8.0)
    beyond = _candidate(12.5, 8.0)

    assert campaign.space.is_feasible(boundary)
    assert not campaign.space.is_feasible(beyond)
    fingertip = Fingertip(campaign.space.to_parameters(boundary))
    assert fingertip.full_height_mm == MAX_FINGERTIP_HEIGHT_MM


def test_integer_encoding_matches_the_physical_candidate() -> None:
    campaign = _campaign()
    physical = _candidate(
        12.0,
        8.0,
        stem_width_mm=8.0,
        stem_height_mm=4.0,
        void_width_mm=0.5,
    )
    encoded = campaign.encode(physical)

    assert encoded == {
        "flat_pad_height_step": 24,
        "semiellipse_height_step": 16,
        "stem_width_step": 16,
        "stem_height_step": 8,
        "void_width_step": 1,
    }
    assert campaign.decode(encoded) == physical
    campaign.validate(encoded, physical)

    invalid = dict(encoded, flat_pad_height_step=25)
    with pytest.raises(ValueError, match="full-height constraint"):
        campaign.validate(invalid, campaign.decode(invalid))


def test_candidate_generation_attaches_only_exact_feasible_points() -> None:
    campaign = _campaign()
    first = feasible_candidate_pool(
        campaign,
        count=12,
        seed=1234,
        excluded=set(),
    )
    second = feasible_candidate_pool(
        campaign,
        count=12,
        seed=1234,
        excluded=set(),
    )
    assert first == second
    assert len({tuple(candidate.values()) for candidate in first}) == len(first)
    for raw in first:
        campaign.validate(raw, campaign.decode(raw))

    client = new_client(campaign)
    trial_index, raw, physical, generation_node = propose_feasible_trial(
        client, campaign
    )
    assert trial_index == 0
    assert generation_node == "FEASIBLE_Sobol"
    campaign.validate(raw, physical)


@pytest.mark.filterwarnings(
    "ignore:Encountered a `MultiObjective` without objective thresholds"
)
@pytest.mark.filterwarnings(
    "ignore:To copy construct from a tensor, it is recommended to use"
)
def test_initial_morphologies_count_toward_ax_initialization_budget() -> None:
    campaign = replace(
        _campaign(initial_morphologies_mm=_INITIAL_MORPHOLOGIES_MM),
        acquisition_pool_size=8,
    )
    config = build_run_config(campaign)["scientific_contract"]["initialization"]
    assert config["initial_morphology_count"] == 5
    assert config["fresh_sobol_count"] == 8
    assert config["old_objective_values_imported"] is False

    client = new_client(campaign)
    generation_nodes = []
    for index in range(14):
        trial_index, _, _, generation_node = propose_next_trial(client, campaign)
        generation_nodes.append(generation_node)
        if index < 13:
            client.complete_trial(
                trial_index=trial_index,
                raw_data={
                    "J_contact": 0.1 + 0.01 * index,
                    "J_obs": 0.001 + 0.0001 * ((7 * index) % 13),
                },
            )

    assert generation_nodes[:5] == ["INITIAL_MORPHOLOGY"] * 5
    assert generation_nodes[5:13] == ["FEASIBLE_Sobol"] * 8
    assert generation_nodes[13] == "FEASIBLE_MBM"


def test_fixed_parameters_are_preserved_when_constructing_a_candidate() -> None:
    campaign = _campaign()
    base = FingertipParameters()
    parameters = campaign.space.to_parameters(_candidate(5.0, 9.0))

    assert parameters.led == base.led
    assert parameters.mechanics == campaign.space.base_parameters.mechanics
    assert parameters.optics == campaign.space.base_parameters.optics
    assert parameters.geometry.flat_pad_width_mm == 30.0
