from __future__ import annotations

import pytest

from validation.ray_tracing import lumo3d_optix_stage


def test_full3d_stage_rejects_nonpassing_preflight() -> None:
    with pytest.raises(RuntimeError, match="requires a passing production preflight"):
        lumo3d_optix_stage._validated_preflight({"status": "FAIL", "stage": "cuda_device"})


def test_full3d_stage_runs_shared_smoke_when_preflight_is_omitted(monkeypatch) -> None:
    calls: list[str] = []

    class _Smoke:
        def to_dict(self) -> dict[str, object]:
            return {"metadata": {"device": "test"}, "ray_count": 2}

    def fake_smoke() -> _Smoke:
        calls.append("smoke")
        return _Smoke()

    monkeypatch.setattr(lumo3d_optix_stage, "run_optix_smoke", fake_smoke)
    evidence = lumo3d_optix_stage._validated_preflight(None)

    assert calls == ["smoke"]
    assert evidence["status"] == "PASS"
    assert evidence["evidence"]["ray_count"] == 2
