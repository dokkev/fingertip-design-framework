from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("gmsh")
pytest.importorskip("newton")
pytest.importorskip("warp")
pytest.importorskip("cupy")
pytest.importorskip("optix")
pytest.importorskip("cuda.bindings.nvrtc")

import scripts.optimization.run_bo as run_bo
from lumo.optimization.adapters.ax import AxTerminationReason
from lumo.optimization.evaluation_registry import REGISTRY_SCHEMA_VERSION
from lumo.optimization.objectives import TRAJECTORY_SEPARATION_OBJECTIVE


@pytest.mark.smoke
def test_one_trial_uses_the_real_bo_campaign_boundary(tmp_path) -> None:
    output = tmp_path / "bo-smoke"

    summary = run_bo.run_campaign(output, trials=1, smoke=True)

    preflight = json.loads((output / "preflight.json").read_text())
    config = json.loads((output / "config.json").read_text())
    registry = json.loads((output / "registry.json").read_text())
    trials = json.loads((output / "trials.json").read_text())
    assert preflight["status"] == "PASS"
    assert config["evaluation_schema"] == run_bo.TRAJECTORY_EVALUATION_SCHEMA
    assert config["campaign_mode"] == "smoke"
    assert config["trajectory_protocol"] == run_bo.SMOKE_PROTOCOL.to_dict()
    assert summary["ax_proposal_count"] == 1
    assert summary["new_evaluation_count"] >= 2
    assert summary["status"] == "PASS"
    assert summary["campaign_acceptance"] == "PASS"
    assert summary["ax_status"] == "COMPLETE"
    assert summary["ax_termination_reason"] == (
        AxTerminationReason.REQUESTED_BUDGET_REACHED.value
    )
    assert summary["nominal_successful"] is True
    assert summary["successful_initialization_count"] >= 1
    assert summary["feasible_proposal_count"] >= 1
    assert registry["schema_version"] == REGISTRY_SCHEMA_VERSION
    assert trials == summary["records"]

    generated_successes = [
        record
        for record in trials
        if record["phase"] != "nominal" and record["status"] == "success"
    ]
    assert generated_successes
    design_space = run_bo._design_space(
        run_bo.USER_PARAMETERS,
        run_bo.USER_SEARCH_BOUNDS,
    )
    for record in generated_successes:
        assert record["physical_parameters"] is not None
        design_space.validate_physical_parameters(
            design_space.from_physical_values(record["physical_parameters"])
        )

    expected_objective = TRAJECTORY_SEPARATION_OBJECTIVE
    assert config["objective"]["name"] == expected_objective.serialized_name
    assert summary["objective_name"] == expected_objective.serialized_name
    registry_records = list(registry["records"].values())
    assert registry_records
    assert all(
        record["contract_id"] == config["contract_id"]
        for record in registry_records
    )
    assert all(
        record["objective"]
        == {"name": expected_objective.name, "version": expected_objective.version}
        for record in registry_records
    )
    for record in generated_successes:
        artifact = json.loads(
            Path(record["result_artifact_path"]).read_text(encoding="utf-8")
        )
        assert artifact["objective_name"] == expected_objective.serialized_name
        assert artifact["objective"]["objective_name"] == (
            expected_objective.serialized_name
        )
        assert artifact["evaluation_identity"]["evaluation_contract_id"] == (
            config["contract_id"]
        )
