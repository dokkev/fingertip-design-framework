"""Functional tests for compact physical contact-dataset analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from experiments.analysis.aggregation import aggregate_analysis
from experiments.analysis.dataset_index import (
    camera_consistency_warnings,
    index_session,
)
from experiments.analysis.deformation import (
    build_contour_reference,
    contour_deformation,
)
from experiments.analysis.optical_response import (
    actual_force_magnitude,
    longitudinal_signature,
    optical_metrics,
    pairwise_signature_distances,
    unloaded_median_rgb,
)
from experiments.analysis.pipeline import AnalysisConfig, analyze_sessions
from experiments.analysis.spatial_signature import repeat_variability
from experiments.data_collection.contact_dataset import (
    FRAME_CSV_COLUMNS,
    RunMetadata,
    SessionMetadata,
    format_force_directory,
)
from experiments.data_collection.force_sequence import ForceSequenceConfig
from experiments.hardware.bota import BotaTareOffsets


def test_force_reference_optical_and_spatial_numerics() -> None:
    assert actual_force_magnitude(3.0, 4.0, 12.0) == pytest.approx(13.0)
    images = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (0, 10, 20)]
    assert np.array_equal(unloaded_median_rgb(images), images[1])

    delta = np.asarray(
        (((1.0, -2.0, 3.0), (-1.0, 2.0, -3.0)),) * 2,
        dtype=np.float32,
    )
    loaded = np.full(delta.shape, 255, dtype=np.uint8)
    metrics = optical_metrics(delta, loaded)
    assert metrics["optical_mae_R_dn"] == pytest.approx(1.0)
    assert metrics["optical_rms_G_dn"] == pytest.approx(2.0)
    assert metrics["optical_signed_mean_B_dn"] == pytest.approx(0.0)
    assert metrics["saturation_eq255_G_fraction"] == pytest.approx(1.0)

    profile = longitudinal_signature(np.asarray(((1.0, 3.0), (5.0, 7.0))), 2)
    assert np.array_equal(profile, np.asarray((2.0, 6.0)))
    distances = pairwise_signature_distances(np.asarray(((0.0, 0.0), (3.0, 4.0))))
    assert distances[0, 1] == pytest.approx(np.sqrt(12.5))


def test_five_hold_frames_become_one_run_force_and_use_actual_force() -> None:
    rows: list[dict[str, object]] = []
    signatures = []
    for frame in range(5):
        rows.append(_aggregate_frame(frame, force=2.0 + frame, response=6.0))
        signatures.append(np.full(128, frame, dtype=np.float64))
    result = aggregate_analysis(rows, np.asarray(signatures))
    assert len(result["run_rows"]) == 1
    run = result["run_rows"][0]
    assert run["frame_count"] == 5
    assert run["actual_force_mean_n"] == pytest.approx(4.0)
    assert run["optical_mae_G_dn_per_n"] == pytest.approx(
        np.median(6.0 / np.arange(2.0, 7.0))
    )
    assert np.array_equal(result["run_signatures"][0], np.full(128, 2.0))


def test_repeat_variability_uses_run_signatures() -> None:
    signatures = np.asarray(((0.0, 0.0), (2.0, 2.0), (4.0, 4.0)))
    template, variability = repeat_variability(signatures)
    assert np.array_equal(template, np.asarray((2.0, 2.0)))
    assert np.array_equal(variability, np.asarray((2.0, 0.0, 2.0)))


def test_contour_deformation_reports_image_space_motion() -> None:
    reference = np.zeros((100, 100, 3), dtype=np.uint8)
    loaded = np.zeros_like(reference)
    cv2.rectangle(reference, (30, 20), (70, 80), (100, 100, 100), -1)
    cv2.rectangle(loaded, (32, 20), (72, 80), (100, 100, 100), -1)
    mask = np.zeros(reference.shape[:2], dtype=bool)
    mask[20:81, 30:71] = True
    calibration = build_contour_reference(
        reference, mask, sample_count=64, search_radius_px=5
    )

    result = contour_deformation(loaded, calibration)

    assert result["deformation_valid"] is True
    assert 1.0 < result["deformation_rms_px"] < 3.0
    assert result["deformation_max_px"] <= 3.0


def test_index_exposes_missing_repetition_and_camera_mismatch(tmp_path: Path) -> None:
    first = _make_session(tmp_path / "first", "one", exposure_us=1500.0)
    second = _make_session(tmp_path / "second", "two", exposure_us=1600.0)
    third = _make_session(tmp_path / "third", "three", exposure_us=1500.0)
    indexes = [
        index_session(path, expected_repetitions=2) for path in (first, second, third)
    ]
    missing = [
        row for row in indexes[0].coverage_rows if row["validity"] == "missing_run"
    ]
    assert missing
    assert {index.session_id for index in indexes} == {"one", "two", "three"}
    assert camera_consistency_warnings(indexes) == [
        "camera mismatch for camera_exposure_us: ['1500.0', '1600.0']"
    ]


def test_cache_reproduces_bundle_without_decoding_raw_pngs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions = [
        _make_session(tmp_path / "raw_a", "a", morphology="baseline"),
        _make_session(tmp_path / "raw_b", "b", morphology="flat_opt"),
    ]
    output = tmp_path / "analysis"
    config = AnalysisConfig(
        expected_repetitions=1,
        deformation_contour_samples=64,
        deformation_search_radius_px=3,
    )
    bundle = analyze_sessions(sessions, output, recompute=True, config=config)
    first_summary = (bundle / "run_force_summary.csv").read_text()

    monkeypatch.setattr(
        cv2,
        "imread",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("raw PNG reopened")
        ),
    )
    analyze_sessions(sessions, output, recompute=False, config=config)
    assert (bundle / "run_force_summary.csv").read_text() == first_summary

    expected = {
        "README.md",
        "manifest.json",
        "session_summary.csv",
        "coverage.csv",
        "frame_features.csv",
        "run_force_summary.csv",
        "condition_summary.csv",
        "spatial_signatures.csv",
        "pairwise_separability.csv",
        "force_response_fits.csv",
        "unloaded_stability.csv",
    }
    assert expected <= {path.name for path in bundle.iterdir()}
    assert not any(path.suffix == ".png" for path in bundle.iterdir())
    assert (output / "analysis_bundle.zip").is_file()


def _aggregate_frame(frame: int, *, force: float, response: float) -> dict[str, object]:
    row: dict[str, object] = {
        "specimen_id": "specimen",
        "material": "solaris",
        "morphology": "baseline",
        "run_id": "run_0001",
        "indenter": "sphere_10mm",
        "hole_index": 1,
        "repetition_index": 1,
        "target_force_n": 5.0,
        "actual_force_n": force,
        "deformation_valid": True,
        "deformation_invalid_reason": "",
        "deformation_rms_px": 2.0,
        "deformation_p95_px": 3.0,
        "deformation_max_px": 4.0,
        "optical_mae_G_dn_per_deformation_px": response / 2.0,
        "optical_rms_G_dn_per_deformation_px": response / 2.0,
    }
    for channel in "RGB":
        row[f"optical_mae_{channel}_dn"] = response
        row[f"optical_rms_{channel}_dn"] = response
        row[f"optical_signed_mean_{channel}_dn"] = response
        row[f"optical_mae_{channel}_dn_per_n"] = response / force
        row[f"optical_rms_{channel}_dn_per_n"] = response / force
        row[f"saturation_ge250_{channel}_fraction"] = 0.0
        row[f"saturation_eq255_{channel}_fraction"] = 0.0
    return row


def _make_session(
    root: Path,
    specimen_id: str,
    *,
    morphology: str = "baseline",
    exposure_us: float = 1500.0,
) -> Path:
    (root / "unloaded" / "capture_001" / "frames").mkdir(parents=True)
    (root / "runs" / "run_0001").mkdir(parents=True)
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
            target_forces_n=(2.0,),
            settle_duration_s=1.0,
            record_duration_s=1.0,
            capture_rate_hz=5.0,
        ),
        sensor_mode="mock",
    )
    (root / "session.json").write_text(json.dumps(session.to_dict()))
    run = RunMetadata("run_0001", "sphere_10mm", 1, 1, "start", "end", "complete")
    run_root = root / "runs" / "run_0001"
    (run_root / "run.json").write_text(json.dumps(run.to_dict()))
    reference = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.ellipse(reference, (48, 48), (15, 35), 0, 0, 360, (20, 100, 50), -1)
    _write_segment(root / "unloaded" / "capture_001", [reference] * 5, force=0.1)
    loaded = reference.copy()
    loaded[35:61, 35:61, 1] = np.clip(loaded[35:61, 35:61, 1] + 20, 0, 255)
    _write_segment(run_root / format_force_directory(2.0), [loaded] * 5, force=2.0)
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
