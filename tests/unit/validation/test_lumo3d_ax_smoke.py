from __future__ import annotations

from validation.optimization.lumo3d_ax_smoke import run_lumo3d_ax_smoke


def test_lumo3d_ax_smoke_uses_real_ax_and_named_objective(tmp_path) -> None:
    summary = run_lumo3d_ax_smoke(tmp_path)

    assert summary["status"] == "PASS"
    assert summary["objective_name"] == "contact_state_separation"
    assert summary["phases"] == ["nominal", "initialization", "search"]
    assert summary["statuses"] == ["success", "success", "success"]
    assert summary["fe_backend_invoked"] is False
    assert summary["optix_backend_invoked"] is False
    assert summary["evaluator_call_count"] == 3

