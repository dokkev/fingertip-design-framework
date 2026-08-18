"""Current immutable production-study contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from optics import IndenterOptics
from optimization import (
    PRODUCTION_EVALUATION_CONTRACT,
    PRODUCTION_EVALUATION_CONTRACT_ID,
    PRODUCTION_SEARCH_BOUNDS,
    create_production_study,
)
from optimization.evaluator import DesignEvaluator


def test_production_study_owns_the_frozen_scientific_configuration() -> None:
    study = create_production_study()

    assert study.scenario_grid.locations_x_mm == (0.0, 1.5, 3.0)
    assert study.scenario_grid.indenter_radii_mm == (3.0, 5.0, 7.0, 10.0)
    assert study.scenario_grid.captured_depths_mm == (0.5, 1.0, 1.5, 2.0)
    assert study.mesh_settings.level == "medium"
    assert study.trace_settings.mode == "planar"
    assert study.indenter_optics == IndenterOptics("absorber")
    assert study.fem_steps == 48
    assert study.basal_interface == "bonded"
    assert study.internal_contact == "sides_separate"
    with pytest.raises(AttributeError):
        study.fem_steps = 12  # type: ignore[misc]


def test_study_validates_current_independent_mechanics_contracts() -> None:
    study = create_production_study()
    with pytest.raises(ValueError, match="fem_steps=48"):
        replace(study, fem_steps=12)
    diagnostic = replace(
        study,
        basal_interface="explicit_contact",
        internal_contact="three_pairs",
    )
    assert diagnostic.basal_interface == "explicit_contact"
    assert diagnostic.internal_contact == "three_pairs"
    with pytest.raises(ValueError, match="requires"):
        replace(
            study,
            basal_interface="explicit_contact",
            internal_contact="sides_separate",
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        replace(
            study,
            basal_interface="bonded",
            internal_contact="three_pairs",
        )


def test_create_evaluator_binds_the_exact_study_configuration() -> None:
    study = create_production_study()
    evaluator = study.create_evaluator()

    assert isinstance(evaluator, DesignEvaluator)
    assert evaluator.scenario_grid is study.scenario_grid
    assert evaluator.mesh_settings is study.mesh_settings
    assert evaluator.trace_settings is study.trace_settings
    assert evaluator.led is study.led
    assert evaluator.optical is study.optical
    assert evaluator.indenter_optics is study.indenter_optics
    assert evaluator.fem_steps == 48
    assert evaluator.basal_interface == "bonded"
    assert evaluator.internal_contact == "sides_separate"


def test_contract_id_fingerprints_the_frozen_production_inputs() -> None:
    assert PRODUCTION_EVALUATION_CONTRACT["bounds_mm"] == PRODUCTION_SEARCH_BOUNDS
    assert PRODUCTION_EVALUATION_CONTRACT["fem"] == {
        "steps": 48,
        "basal_interface": "bonded",
        "internal_contact": "sides_separate",
    }
    assert PRODUCTION_EVALUATION_CONTRACT["objective"] == {
        "direction": "maximize",
        "metric": "minimum_auc",
        "state_metric": "lateral-L1/launched-v1",
        "trajectory_aggregation": "trapz-J0-normalized-2mm-v1",
        "study_aggregation": "min-trajectory-v1",
    }
    assert PRODUCTION_EVALUATION_CONTRACT_ID.startswith("production-evaluation-v1-")
    assert len(PRODUCTION_EVALUATION_CONTRACT_ID.rsplit("-", 1)[1]) == 16
