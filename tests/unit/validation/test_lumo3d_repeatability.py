from __future__ import annotations

import pytest

from validation.optimization.lumo3d_repeatability import _compare_run_summaries


def _run(signature: str, *, status: str = "PASS", count: int = 18) -> dict:
    return {"status": status, "checkpoint_count": count, "signature": signature}


def test_repeatability_requires_three_independent_runs() -> None:
    with pytest.raises(ValueError, match="at least three"):
        _compare_run_summaries((_run("same"), _run("same")))


def test_repeatability_passes_only_exact_complete_successes() -> None:
    result = _compare_run_summaries(tuple(_run("same") for _ in range(3)))

    assert result == {
        "worker_count": 3,
        "all_worker_processes_exited_zero": True,
        "all_workers_passed": True,
        "all_workers_have_18_states": True,
        "bit_exact": True,
        "unique_signature_count": 1,
        "status": "PASS",
    }


@pytest.mark.parametrize(
    "runs",
    (
        (_run("a"), _run("b"), _run("a")),
        (_run("a", status="FAIL"), _run("a"), _run("a")),
        (_run("a", count=17), _run("a"), _run("a")),
        (_run("a") | {"returncode": 2}, _run("a"), _run("a")),
    ),
)
def test_repeatability_fails_on_any_scientific_or_identity_difference(runs) -> None:
    assert _compare_run_summaries(runs)["status"] == "FAIL"
