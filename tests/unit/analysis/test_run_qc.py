"""Behavioral tests for baseline-free suspect-run QC."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from experiments.analysis.run_qc import analyze_run_qc


FORCES = (2.0, 5.0, 10.0, 15.0)
HOLE_1_PATTERN = np.linspace(0.2, 1.0, 8)
HOLE_2_PATTERN = -2.0 * HOLE_1_PATTERN


def test_normal_repeat_is_closest_to_own_hole_and_not_suspect() -> None:
    rows, signatures = _two_hole_data()
    result = analyze_run_qc(rows, signatures, [])
    row = _row(result, "h1_r3")

    assert row["possible_hole_mislabel"] is False
    assert row["repeat_outlier"] is False
    assert row["qc_interpretation"] == "no automatic QC flag"
    assert row["recorded_hole_distance_median_dn"] < row[
        "nearest_alt_distance_median_dn"
    ]


def test_same_hole_outlier_without_good_alternative_is_repeat_outlier() -> None:
    rows, signatures = _two_hole_data(
        hole_1_scales=(0.96, 0.98, 1.0, 1.02, 1.04, 4.0)
    )
    result = analyze_run_qc(rows, signatures, [])
    row = _row(result, "h1_r6")

    assert row["repeat_outlier"] is True
    assert row["possible_hole_mislabel"] is False


def test_consistent_alternative_hole_affinity_marks_possible_mislabel() -> None:
    rows, signatures = _two_hole_data()
    _replace_run_pattern(rows, signatures, "h1_r5", HOLE_2_PATTERN)
    result = analyze_run_qc(rows, signatures, [])
    row = _row(result, "h1_r5")

    assert row["nearest_alternative_hole"] == 2
    assert row["alt_preferred_count"] == row["comparison_count"] == 4
    assert row["possible_hole_mislabel"] is True
    assert row["repeat_outlier"] is False


def test_one_force_anomaly_is_not_called_a_hole_mislabel() -> None:
    rows, signatures = _two_hole_data()
    _replace_signature(rows, signatures, "h1_r5", 15.0, 13.0 * HOLE_2_PATTERN)
    result = analyze_run_qc(rows, signatures, [])
    row = _row(result, "h1_r5")

    assert row["nearest_alternative_hole_15n"] == 2
    assert row["possible_hole_mislabel"] is False


def test_same_hole_template_leaves_evaluated_run_out() -> None:
    rows, signatures = _two_hole_data(
        hole_1_scales=(0.0, 1.0, 2.0, 100.0),
        hole_2_scales=(0.98, 1.0, 1.02, 1.04),
    )
    result = analyze_run_qc(rows, signatures, [])
    row = _row(result, "h1_r2")

    expected = 3.0 * np.sqrt(np.mean(HOLE_1_PATTERN**2))
    assert row["same_hole_distance_5n_dn"] == pytest.approx(expected)


def test_missing_repeat_metadata_is_independent_of_optical_qc() -> None:
    rows, signatures = _two_hole_data()
    coverage = [
        {
            "specimen_id": "specimen",
            "indenter": "sphere_10mm",
            "hole_index": 1,
            "repetition_index": 6,
            "target_force_n": 2.0,
            "run_ids": "",
            "validity": "missing_run",
        }
    ]
    result = analyze_run_qc(rows, signatures, coverage)
    row = _row(result, "h1_r3")

    assert row["metadata_anomaly"] is True
    assert row["metadata_anomaly_reason"] == "missing repetition 6"
    assert row["possible_hole_mislabel"] is False
    assert row["repeat_outlier"] is False
    assert row["qc_interpretation"] == "metadata-only anomaly"


def test_unequal_four_and_five_repeat_coverage_remains_comparable() -> None:
    rows, signatures = _two_hole_data(
        hole_1_scales=(0.97, 0.99, 1.01, 1.03),
        hole_2_scales=(0.96, 0.98, 1.0, 1.02, 1.04),
    )
    result = analyze_run_qc(rows, signatures, [])

    assert len(result["rows"]) == 9
    assert all(
        np.isfinite(float(row["same_hole_distance_10n_dn"]))
        for row in result["rows"]
    )


def test_run_static_additive_offset_does_not_change_baseline_free_qc() -> None:
    rows, signatures = _two_hole_data()
    original = analyze_run_qc(rows, signatures, [])
    shifted = signatures.copy()
    for run_id in sorted({str(row["run_id"]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["run_id"] == run_id]
        offset = np.linspace(-17.0, 23.0, shifted.shape[1])
        shifted[indices] += offset
    changed = analyze_run_qc(rows, shifted, [])

    keys = (
        "same_hole_distance_median_dn",
        "same_hole_slope_distance_dn_per_n",
        "nearest_alt_distance_median_dn",
        "worst_force_progression_residual_dn",
    )
    for first, second in zip(original["rows"], changed["rows"], strict=True):
        assert first["run_id"] == second["run_id"]
        for key in keys:
            assert float(first[key]) == pytest.approx(float(second[key]), abs=1e-12)
        assert first["possible_hole_mislabel"] == second["possible_hole_mislabel"]
        assert first["repeat_outlier"] == second["repeat_outlier"]


def _two_hole_data(
    *,
    hole_1_scales: tuple[float, ...] = (0.96, 0.98, 1.0, 1.02, 1.04),
    hole_2_scales: tuple[float, ...] = (0.96, 0.98, 1.0, 1.02, 1.04),
) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows: list[dict[str, Any]] = []
    signatures = []
    for hole, pattern, scales in (
        (1, HOLE_1_PATTERN, hole_1_scales),
        (2, HOLE_2_PATTERN, hole_2_scales),
    ):
        for repetition, scale in enumerate(scales, start=1):
            run_id = f"h{hole}_r{repetition}"
            static_offset = 0.1 * repetition * np.cos(np.linspace(0.0, np.pi, 8))
            for force in FORCES:
                rows.append(
                    {
                        "specimen_id": "specimen",
                        "material": "solaris",
                        "morphology": "test",
                        "indenter": "sphere_10mm",
                        "run_id": run_id,
                        "hole_index": hole,
                        "repetition_index": repetition,
                        "target_force_n": force,
                        "actual_force_median_n": force,
                    }
                )
                signatures.append(static_offset + scale * (force - 2.0) * pattern)
    return rows, np.asarray(signatures, dtype=np.float64)


def _replace_run_pattern(
    rows: list[dict[str, Any]],
    signatures: np.ndarray,
    run_id: str,
    pattern: np.ndarray,
) -> None:
    indices = [index for index, row in enumerate(rows) if row["run_id"] == run_id]
    reference = signatures[indices[0]].copy()
    for index in indices:
        force = float(rows[index]["target_force_n"])
        signatures[index] = reference + (force - 2.0) * pattern


def _replace_signature(
    rows: list[dict[str, Any]],
    signatures: np.ndarray,
    run_id: str,
    force: float,
    signature: np.ndarray,
) -> None:
    index = next(
        index
        for index, row in enumerate(rows)
        if row["run_id"] == run_id and float(row["target_force_n"]) == force
    )
    signatures[index] = signature


def _row(result: dict[str, Any], run_id: str) -> dict[str, Any]:
    return next(row for row in result["rows"] if row["run_id"] == run_id)
