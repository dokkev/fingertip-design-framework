from __future__ import annotations

import json

import pytest

pytest.importorskip("gmsh")
pytest.importorskip("newton")
pytest.importorskip("warp")
pytest.importorskip("cupy")
pytest.importorskip("optix")
pytest.importorskip("cuda.bindings.nvrtc")

import scripts.optimization.run_bo as run_bo


@pytest.mark.smoke
def test_one_trial_uses_the_real_bo_campaign_boundary(tmp_path) -> None:
    output = tmp_path / "bo-smoke"

    summary = run_bo.run_campaign(output, trials=1, smoke=True)

    preflight = json.loads((output / "preflight.json").read_text())
    config = json.loads((output / "config.json").read_text())
    registry = json.loads((output / "registry.json").read_text())
    assert preflight["status"] == "PASS"
    assert config["evaluation_schema"] == run_bo.TRAJECTORY_EVALUATION_SCHEMA
    assert config["campaign_mode"] == "smoke"
    assert config["trajectory_protocol"] == run_bo.SMOKE_PROTOCOL.to_dict()
    assert summary["ax_proposal_count"] == 1
    assert summary["new_evaluation_count"] >= 1
    assert summary["status"] in {"COMPLETE", "proposal_budget_exhausted"}
    assert registry["schema_version"] == 2
