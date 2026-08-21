from __future__ import annotations

from validation.optimization.lumo3d_ax_smoke import run_lumo3d_ax_smoke
from lumo.optimization.objectives import TRAJECTORY_SEPARATION_OBJECTIVE


def test_lumo3d_ax_smoke_uses_real_ax_and_named_objective(tmp_path) -> None:
    summary = run_lumo3d_ax_smoke(tmp_path)

    assert summary["status"] == "PASS"
    assert summary["objective_name"] == (
        TRAJECTORY_SEPARATION_OBJECTIVE.serialized_name
    )
    assert summary["phases"][:2] == ["nominal", "initialization"]
    assert summary["phases"][-1] == "search"
    assert summary["statuses"][0] == "success"
    assert summary["statuses"][1] == "success"
    assert summary["statuses"][-1] == "success"
    assert summary["fe_backend_invoked"] is False
    assert summary["optix_backend_invoked"] is False
    assert summary["evaluator_call_count"] == 3
