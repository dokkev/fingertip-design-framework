"""Status and setup contracts for the bounded validation BO report."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import validation.optimization.lumo6d_test_bo as test_bo
from lumo.config import load_lumo_execution_config
from lumo.mechanics_contract import MechanicsContract
from validation.optimization.lumo6d_test_bo import (
    _bounded_gate_status,
    _optical_grid,
    _search_mechanics,
    _status_contract,
    _trial_payload,
)


ROOT = Path(__file__).resolve().parents[3]


def test_status_contract_preserves_current_evaluation_taxonomy() -> None:
    assert _status_contract("success") == "valid_success"
    assert _status_contract("invalid_design") == "geometry_rejected"
    assert _status_contract("domain_incompatible") == "domain_incompatible"
    assert _status_contract("mesh_failure") == "geometry_rejected"
    assert _status_contract("mechanics_failure") == "mechanics_failed"
    assert _status_contract("optics_failure") == "optics_failed"
    assert _status_contract("duplicate_skipped") == "duplicate_skipped"


def test_removed_failure_vocabulary_is_not_silently_current() -> None:
    assert _status_contract("fea_failure") == "infrastructure_failed"


def test_test_bo_serializes_the_mechanics_contract_at_the_setup_boundary() -> None:
    contract = MechanicsContract()
    evaluator = SimpleNamespace(mechanics_contract=contract)

    search_mechanics = _search_mechanics(evaluator)

    assert search_mechanics == contract.to_dict()


def test_test_bo_optical_grid_comes_from_typed_execution_config() -> None:
    execution = load_lumo_execution_config(ROOT / "config" / "lumo_execution.yaml")

    grid = _optical_grid(execution)

    assert grid["x_bounds_mm"] == list(execution.transport.x_bounds_mm)
    assert grid["y_bounds_mm"] == list(execution.transport.y_bounds_mm)
    assert grid["fingerprint"]


def test_trial_payload_uses_the_resolved_optical_grid_fingerprint() -> None:
    design_space = test_bo._production_design_space()
    record = SimpleNamespace(
        parameters=design_space.encode(design_space.nominal_parameters),
        phase="initialization",
        trial_index=7,
        evaluation=None,
        status="feasibility_rejected",
        failure_message="rejected before evaluation",
        wall_time_seconds=0.0,
        registry_key=None,
        feasibility_rejection=True,
        feasibility_constraint="test_constraint",
    )

    payload = _trial_payload(
        record,
        design_space,
        0,
        optical_grid_fingerprint="resolved-grid-fingerprint",
    )

    assert payload["optical_grid_fingerprint"] == "resolved-grid-fingerprint"


def test_bounded_bo_cli_returns_nonzero_for_controlled_gate_failure(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        test_bo,
        "run_lumo6d_test_bo",
        lambda *_args, **_kwargs: {"status": "FAIL"},
    )

    assert test_bo.main([]) == 3
    assert '"status": "FAIL"' in capsys.readouterr().out


def test_bounded_bo_cli_returns_zero_only_for_full_pass(monkeypatch) -> None:
    monkeypatch.setattr(
        test_bo,
        "run_lumo6d_test_bo",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )

    assert test_bo.main([]) == 0


def test_bounded_gate_rejects_zero_objective_variation_without_fake_threshold() -> None:
    diagnostics = {
        "objective_extinction_pathology": False,
        "bo_successful_count": 4,
        "mechanics_failure_count": 0,
        "optics_failure_count": 0,
        "successful_count": 10,
        "objective_variation": {
            "range": [0.5, 0.5],
            "span": 0.0,
            "nonzero": False,
            "scientific_threshold": None,
            "magnitude_assessment": "FAIL",
        },
    }

    assert _bounded_gate_status(diagnostics, ax_status="COMPLETE") == "FAIL"
    diagnostics["objective_variation"] = {
        "range": [0.5, 0.5000000000000001],
        "span": 1.0e-16,
        "nonzero": True,
        "scientific_threshold": None,
        "magnitude_assessment": "INCONCLUSIVE",
    }
    assert _bounded_gate_status(diagnostics, ax_status="COMPLETE") == "PASS"


def test_bounded_bo_rejects_unavailable_source_before_evaluator_construction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        test_bo,
        "_source_provenance",
        lambda: {"status": "unavailable", "source_id": None},
    )
    monkeypatch.setattr(
        test_bo,
        "Lumo3DTrajectoryEvaluator",
        lambda *_args, **_kwargs: pytest.fail("expensive evaluator was constructed"),
    )

    with pytest.raises(RuntimeError, match="provenance is unavailable"):
        test_bo.run_lumo6d_test_bo(tmp_path / "unavailable")
    assert not (tmp_path / "unavailable").exists()
