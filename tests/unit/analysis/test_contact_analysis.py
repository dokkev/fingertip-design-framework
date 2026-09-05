"""Functional tests for baseline-free physical morphology analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pytest

from experiments.analysis.dataset import camera_consistency_warnings, index_session
from experiments.analysis.metrics import (
    actual_force_magnitude,
    aggregate_run_force_frames,
    fit_load_responses,
    fit_profile_slopes,
    morphology_metrics,
    spatial_metrics,
)
from experiments.analysis.summary import AnalysisConfig, analyze_morphologies
from experiments.data_collection.contact_dataset import (
    FRAME_CSV_COLUMNS,
    RunMetadata,
    SessionMetadata,
    format_force_directory,
)
from experiments.data_collection.force_sequence import ForceSequenceConfig
from experiments.hardware.bota import BotaTareOffsets


def test_actual_force_magnitude_uses_all_three_axes() -> None:
    assert actual_force_magnitude(3.0, 4.0, 12.0) == pytest.approx(13.0)


def test_five_frames_aggregate_to_one_run_force_and_preserve_identity() -> None:
    rows = [_frame_row(index, force=2.0 + index) for index in range(5)]
    profiles = np.asarray(
        [np.full(128, index, dtype=np.float64) for index in range(5)]
    )

    aggregated, profile = aggregate_run_force_frames(rows, profiles)

    assert len(aggregated) == 1
    assert aggregated[0]["run_id"] == "run_0042"
    assert aggregated[0]["hole_index"] == 3
    assert aggregated[0]["repetition_index"] == 2
    assert aggregated[0]["frame_count"] == 5
    assert aggregated[0]["actual_force_median_n"] == pytest.approx(4.0)
    assert np.array_equal(profile[0], np.full(128, 2.0))


def test_profile_slope_and_s_load_use_actual_force_and_ignore_static_offset() -> None:
    targets = (2.0, 5.0, 10.0, 15.0)
    actual = np.asarray((2.0, 4.0, 8.0, 12.0))
    expected_slope = np.linspace(-2.0, 3.0, 128)
    static = np.linspace(30.0, 50.0, 128)
    profiles = static[None, :] + actual[:, None] * expected_slope[None, :]
    rows = [_run_force_row(target, force) for target, force in zip(targets, actual)]

    slope, _, residual, r_squared = fit_profile_slopes(actual, profiles)
    run_rows, fitted = fit_load_responses(rows, profiles)
    shifted_rows, shifted = fit_load_responses(rows, profiles + 71.0)

    assert np.allclose(slope, expected_slope)
    assert residual == pytest.approx(0.0, abs=1e-12)
    assert r_squared == pytest.approx(1.0)
    assert np.allclose(fitted[0], expected_slope)
    assert np.allclose(shifted[0], expected_slope)
    expected_s_load = np.sqrt(np.mean(expected_slope**2))
    assert run_rows[0]["S_load_DN_per_N"] == pytest.approx(expected_s_load)
    assert shifted_rows[0]["S_load_DN_per_N"] == pytest.approx(expected_s_load)
    target_fit = np.polyfit(np.asarray(targets), profiles[:, 0], 1)[0]
    assert fitted[0, 0] != pytest.approx(target_fit)


def test_neighbor_distance_repeat_variability_and_missing_repeat() -> None:
    rows = [
        _load_row("h1_r1", 1, 1),
        _load_row("h1_r3", 1, 3),
        _load_row("h2_r1", 2, 1),
        _load_row("h2_r2", 2, 2),
    ]
    profiles = np.asarray(
        ((0.0, 0.0), (2.0, 2.0), (4.0, 5.0), (6.0, 7.0))
    )

    neighboring, variability = spatial_metrics(rows, profiles)
    headline = morphology_metrics(rows, neighboring, variability)

    assert len(neighboring) == 1
    assert neighboring[0]["D_neighbor_DN_per_N"] == pytest.approx(
        np.sqrt((4.0**2 + 5.0**2) / 2.0)
    )
    assert [row["repeat_variability_DN_per_N"] for row in variability] == (
        pytest.approx([1.0, 1.0, 1.0, 1.0])
    )
    assert headline[0]["W_median_DN_per_N"] == pytest.approx(1.0)
    assert headline[0]["independent_run_count"] == 4


def test_one_repeat_has_unavailable_variability_without_crashing() -> None:
    rows = [_load_row("h1_r1", 1, 1), _load_row("h2_r1", 2, 1)]
    neighboring, variability = spatial_metrics(rows, np.asarray(((0.0,), (3.0,))))

    assert neighboring[0]["D_neighbor_DN_per_N"] == pytest.approx(3.0)
    assert all(np.isnan(row["repeat_variability_DN_per_N"]) for row in variability)


def test_incomplete_force_run_is_retained_with_unavailable_slope() -> None:
    row = _run_force_row(2.0, 2.2)

    runs, slopes = fit_load_responses([row], np.ones((1, 128)))

    assert len(runs) == 1
    assert np.isnan(runs[0]["S_load_DN_per_N"])
    assert "missing_force" in runs[0]["qc_flags"]
    assert np.all(np.isnan(slopes[0]))


def test_index_reports_camera_mismatch_and_missing_repetitions(tmp_path: Path) -> None:
    first = _make_session(tmp_path / "first", "one", "baseline", 1500.0)
    second = _make_session(tmp_path / "second", "two", "flat_opt", 1600.0)
    indexes = [index_session(path, expected_repetitions=2) for path in (first, second)]

    assert any(
        row["validity"] == "missing_run" for row in indexes[0].coverage_rows
    )
    assert camera_consistency_warnings(indexes) == [
        "camera mismatch for camera_exposure_us: ['1500.0', '1600.0']"
    ]


def test_n_sessions_write_self_describing_image_free_summary(tmp_path: Path) -> None:
    sessions = [
        _make_session(tmp_path / "raw_a", "one", "baseline", 1500.0),
        _make_session(tmp_path / "raw_b", "two", "flat_opt", 1600.0),
    ]
    output = analyze_morphologies(
        sessions,
        tmp_path / "analysis",
        config=AnalysisConfig(expected_repetitions=1),
    )
    summary = output / "raw_data_summary"

    assert (output / "results" / "morphology_metrics.csv").is_file()
    assert (output / "figures" / "optical_load_sensitivity.png").is_file()
    assert not any(path.suffix.lower() == ".png" for path in summary.rglob("*"))
    assert not any(
        "deformation" in path.read_text(encoding="utf-8").splitlines()[0].lower()
        for path in summary.glob("*.csv")
        if path.stat().st_size
    )
    with np.load(summary / "longitudinal_profiles.npz", allow_pickle=False) as data:
        assert data["profiles"].shape == (8, 128)
        assert set(data["specimen_id"].tolist()) == {"one", "two"}
        assert set(data["run_id"].tolist()) == {"run_0001"}
        assert set(data["target_force_n"].tolist()) == {2.0, 5.0, 10.0, 15.0}
        assert np.all(data["frame_count"] == 5)
    with np.load(summary / "load_response_profiles.npz", allow_pickle=False) as data:
        assert data["slope_profiles"].shape == (2, 128)
        assert set(data["morphology"].tolist()) == {"baseline", "flat_opt"}
    with (summary / "qc_summary.csv").open(newline="", encoding="utf-8") as stream:
        qc = list(csv.DictReader(stream))
    assert any(row["qc_code"] == "camera_setting_mismatch" for row in qc)
    with zipfile.ZipFile(output / "raw_data_summary.zip") as archive:
        assert all(not name.lower().endswith(".png") for name in archive.namelist())
        assert "raw_data_summary/load_response_profiles.npz" in archive.namelist()


def _frame_row(index: int, *, force: float) -> dict[str, object]:
    row: dict[str, object] = {
        "specimen_id": "specimen",
        "material": "solaris",
        "morphology": "baseline",
        "run_id": "run_0042",
        "run_status": "complete",
        "indenter": "sphere_10mm",
        "hole_index": 3,
        "repetition_index": 2,
        "target_force_n": 5.0,
        "expected_frame_count": 5,
        "force_tolerance_n": 1.0,
        "acquisition_target_forces_n": "2;5;10;15",
        "actual_force_n": force,
        "camera_bota_time_delta_ms": index,
        "Fx_N": force,
        "Fy_N": 0.0,
        "Fz_N": 0.0,
        "Mx_Nm": 0.0,
        "My_Nm": 0.0,
        "Mz_Nm": 0.0,
    }
    for channel in "RGB":
        row[f"image_mean_{channel}_dn"] = 100.0
        row[f"saturation_ge250_{channel}_fraction"] = 0.0
        row[f"saturation_eq255_{channel}_fraction"] = 0.0
    return row


def _run_force_row(target: float, force: float) -> dict[str, object]:
    return {
        "specimen_id": "specimen",
        "material": "solaris",
        "morphology": "baseline",
        "run_id": "run_0001",
        "run_status": "complete",
        "indenter": "sphere_10mm",
        "hole_index": 1,
        "repetition_index": 1,
        "target_force_n": target,
        "actual_force_median_n": force,
        "acquisition_target_forces_n": "2;5;10;15",
        "qc_flags": "",
    }


def _load_row(run_id: str, hole: int, repetition: int) -> dict[str, object]:
    return {
        "specimen_id": "specimen",
        "material": "solaris",
        "morphology": "baseline",
        "run_id": run_id,
        "run_status": "complete",
        "indenter": "sphere_10mm",
        "hole_index": hole,
        "repetition_index": repetition,
        "S_load_DN_per_N": 2.0,
    }


def _make_session(
    root: Path,
    specimen_id: str,
    morphology: str,
    exposure_us: float,
) -> Path:
    unloaded_root = root / "unloaded" / "capture_001"
    run_root = root / "runs" / "run_0001"
    unloaded_root.mkdir(parents=True)
    run_root.mkdir(parents=True)
    session = SessionMetadata(
        material="solaris",
        morphology=morphology,
        specimen_id=specimen_id,
        camera_model="D435",
        camera_width=96,
        camera_height=96,
        camera_fps=30,
        camera_exposure_us=exposure_us,
        camera_gain=0.0,
        camera_white_balance_k=4600.0,
        bota_serial_port="mock",
        bota_tare_offsets=BotaTareOffsets(),
        force_sequence=ForceSequenceConfig(
            target_forces_n=(2.0, 5.0, 10.0, 15.0),
            settle_duration_s=1.0,
            record_duration_s=1.0,
            capture_rate_hz=5.0,
        ),
        sensor_mode="mock",
    )
    (root / "session.json").write_text(json.dumps(session.to_dict()))
    run = RunMetadata("run_0001", "sphere_10mm", 1, 1, "start", "end", "complete")
    (run_root / "run.json").write_text(json.dumps(run.to_dict()))
    reference = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.ellipse(reference, (48, 48), (18, 38), 0, 0, 360, (20, 100, 50), -1)
    _write_segment(unloaded_root, [reference] * 5, force=0.1)
    for target in session.force_sequence.target_forces_n:
        loaded = reference.copy()
        loaded[20:77, 35:62, 1] = np.clip(
            loaded[20:77, 35:62, 1].astype(np.int16) + int(target), 0, 255
        ).astype(np.uint8)
        _write_segment(
            run_root / format_force_directory(target), [loaded] * 5, force=target
        )
    return root


def _write_segment(root: Path, images: list[np.ndarray], *, force: float) -> None:
    (root / "frames").mkdir(parents=True, exist_ok=True)
    with (root / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FRAME_CSV_COLUMNS)
        writer.writeheader()
        for index, image in enumerate(images):
            filename = f"frames/{index:06d}.png"
            cv2.imwrite(str(root / filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            row = {name: 0 for name in FRAME_CSV_COLUMNS}
            row.update(
                {
                    "frame_index": index,
                    "rgb_filename": filename,
                    "capture_elapsed_s": index / 5.0,
                    "camera_frame_number": index,
                    "Fx_N": force,
                    "Fy_N": 0.0,
                    "Fz_N": 0.0,
                    "temperature_C": 25.0,
                    "bota_status": 0,
                }
            )
            writer.writerow(row)
