"""Lightweight tests for Newton sphere convergence sweep bookkeeping."""

from __future__ import annotations

import numpy as np

from validation.common.io import strict_read_json
from validation.mechanics3d.sweep_newton_sphere_parameters import (
    SweepConfig,
    comparison_metrics,
    failed_record,
    load_steps_for_increment,
    write_sweep_artifacts,
)


def test_load_steps_policy_uses_smallest_satisfying_integer() -> None:
    assert load_steps_for_increment(0.6, 0.05) == 12
    assert load_steps_for_increment(2.0, 0.05) == 40
    assert load_steps_for_increment(3.0, 0.05) == 60
    assert load_steps_for_increment(1.0, 0.3) == 4


def test_comparison_metrics_are_geometry_focused() -> None:
    reference = np.zeros((2, 3), dtype=float)
    candidate = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    reference_displacement = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    candidate_displacement = np.array([[1.0, 0.0, 0.0], [0.0, 1.5, 0.0]])

    metrics = comparison_metrics(
        candidate,
        reference,
        candidate_displacement,
        reference_displacement,
    )

    assert metrics["max_abs_vertex_difference_mm"] == 2.0
    assert metrics["rms_vertex_difference_mm"] == np.sqrt(5.0 / 6.0)
    assert metrics["max_displacement_difference_mm"] == 0.5
    assert metrics["max_displacement_field_difference_mm"] == 0.5
    assert metrics["relative_max_displacement_difference"] == 0.5


def test_failed_run_is_serializable_without_solver_execution(tmp_path) -> None:
    config = SweepConfig("synthetic", 10.0, 3.0, 2, 60, 10)
    record = failed_record(config, "synthetic solver failure", runtime_s=1.25)
    summary = {
        "schema": "test",
        "selection": {},
        "noise_floor": {},
        "successful_runs": 0,
        "failed_runs": 1,
        "records": [record],
        "stress_case_healthy_under_search": False,
    }

    paths = write_sweep_artifacts(summary, tmp_path)
    restored = strict_read_json(paths["json"])
    assert restored["records"][0]["status"] == "failed"
    assert restored["records"][0]["failure_reason"] == "synthetic solver failure"
    assert "run_id" in tmp_path.joinpath("newton_sphere_sweep.csv").read_text()
