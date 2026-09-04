from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from experiments.data_collection.contact_dataset import (
    FRAME_CSV_COLUMNS,
    ContactDatasetWriter,
    SessionMetadata,
    SynchronizedFrame,
    format_force_directory,
    iter_dataset_frames,
    parse_force_directory,
)
from experiments.data_collection.force_sequence import ForceSequenceConfig
from experiments.hardware import BotaSample, BotaTareOffsets


def _metadata(
    *,
    target_forces_n: tuple[float, ...] = (2.0,),
) -> SessionMetadata:
    return SessionMetadata(
        material="solaris",
        morphology="nominal",
        specimen_id="solaris_nominal_01",
        camera_model="RealSense D435",
        camera_width=8,
        camera_height=6,
        camera_fps=30,
        bota_serial_port="MOCK",
        bota_tare_offsets=BotaTareOffsets(fx_n=0.1, mz_nm=-0.2),
        force_sequence=ForceSequenceConfig(
            target_forces_n=target_forces_n,
            settle_duration_s=0.5,
            record_duration_s=1.0,
            capture_rate_hz=5.0,
        ),
        sensor_mode="mock",
        camera_serial_number="camera-01",
        bota_model="Mock Rokubi",
        created_utc="2026-09-03T12:00:00.000+00:00",
        git_commit="abc123",
    )


def _frame(index: int, force_n: float = 2.0) -> SynchronizedFrame:
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 17 + index
    sample = BotaSample(
        host_time_s=1.0 + index / 30.0,
        sensor_timestamp=100 + index,
        status=0,
        temperature_c=24.0,
        fx_n=0.1,
        fy_n=0.2,
        fz_n=force_n,
        mx_nm=0.01,
        my_nm=0.02,
        mz_nm=0.03,
        force_magnitude_n=float(np.linalg.norm((0.1, 0.2, force_n))),
        torque_magnitude_nm=float(np.linalg.norm((0.01, 0.02, 0.03))),
        fz_share=force_n / float(np.linalg.norm((0.1, 0.2, force_n))),
    )
    return SynchronizedFrame(
        rgb=rgb,
        camera_host_time_s=sample.host_time_s + 0.001,
        camera_device_timestamp_ms=1000.0 + index * 33.0,
        camera_frame_number=index,
        bota_sample=sample,
    )


def _run(
    writer: ContactDatasetWriter,
    *,
    hole_index: int = 3,
    repeat_index: int = 2,
):
    return writer.start_loaded_run(
        indenter="sphere_15mm",
        hole_index=hole_index,
        repeat_index=repeat_index,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_session_metadata_round_trip_uses_explicit_v2_hierarchy(tmp_path: Path) -> None:
    metadata = _metadata(target_forces_n=(2.0, 5.0, 10.0, 15.0))
    with ContactDatasetWriter(tmp_path, metadata) as writer:
        stored = json.loads(
            (writer.session_path / "session.json").read_text(encoding="utf-8")
        )

    assert stored["format_version"] == 2
    assert stored["specimen"] == {
        "material": "solaris",
        "morphology": "nominal",
        "specimen_id": "solaris_nominal_01",
    }
    assert stored["camera"] == {
        "model": "RealSense D435",
        "serial_number": "camera-01",
        "width": 8,
        "height": 6,
        "fps": 30,
    }
    assert stored["acquisition"]["target_forces_n"] == [2.0, 5.0, 10.0, 15.0]
    assert stored["acquisition"]["record_duration_s"] == 1.0
    assert SessionMetadata.from_dict(stored) == metadata


def test_manual_tare_updates_only_session_sensor_metadata(tmp_path: Path) -> None:
    with ContactDatasetWriter(tmp_path, _metadata()) as writer:
        writer.update_tare_offsets(BotaTareOffsets(fy_n=1.25))
        writer.flush()
        stored = json.loads(
            (writer.session_path / "session.json").read_text(encoding="utf-8")
        )

    assert stored["force_sensor"]["tare_offsets"] == {
        "fx_n": 0.0,
        "fy_n": 1.25,
        "fz_n": 0.0,
        "mx_nm": 0.0,
        "my_nm": 0.0,
        "mz_nm": 0.0,
    }


def test_run_json_contains_only_independent_trial_identity_and_lifecycle(
    tmp_path: Path,
) -> None:
    with ContactDatasetWriter(tmp_path, _metadata()) as writer:
        run = _run(writer)
        writer.abort_loaded_run(run)
        writer.flush()
        stored = json.loads((run.path / "run.json").read_text(encoding="utf-8"))

    assert set(stored) == {
        "run_id",
        "indenter",
        "hole_index",
        "repeat_index",
        "started_utc",
        "ended_utc",
        "status",
    }
    assert stored["indenter"] == "sphere_15mm"
    assert stored["hole_index"] == 3
    assert stored["repeat_index"] == 2
    assert stored["status"] == "aborted"
    assert "material" not in stored
    assert "target_forces_n" not in stored
    assert "completed_targets_n" not in stored


def test_completed_force_segment_contains_only_frames_and_csv(tmp_path: Path) -> None:
    with ContactDatasetWriter(tmp_path, _metadata()) as writer:
        run = _run(writer)
        segment = writer.begin_force_target(run, 2.0)
        assert writer.submit_frame(segment, _frame(0), capture_elapsed_s=0.0)
        assert writer.submit_frame(segment, _frame(1), capture_elapsed_s=0.2)
        writer.finalize_segment(segment)
        writer.complete_loaded_run(run)
        writer.flush()

        final = run.path / "force_02N"
        assert sorted(path.name for path in final.iterdir()) == ["frames", "frames.csv"]
        assert sorted(path.name for path in (final / "frames").glob("*.png")) == [
            "000000.png",
            "000001.png",
        ]
        stored_bgr = cv2.imread(str(final / "frames" / "000000.png"))
        assert stored_bgr is not None
        assert tuple(stored_bgr[0, 0]) == (0, 0, 17)


def test_failed_attempt_is_removed_and_retry_restarts_at_frame_zero(
    tmp_path: Path,
) -> None:
    with ContactDatasetWriter(tmp_path, _metadata()) as writer:
        run = _run(writer)
        failed = writer.begin_force_target(run, 2.0)
        writer.submit_frame(failed, _frame(0), capture_elapsed_s=0.0)
        writer.discard_segment(failed)

        retry = writer.begin_force_target(run, 2.0)
        writer.submit_frame(retry, _frame(2), capture_elapsed_s=0.0)
        writer.finalize_segment(retry)
        writer.complete_loaded_run(run)
        writer.flush()

        assert not failed.partial_path.exists()
        assert not list(run.path.glob("*.partial"))
        assert sorted(path.name for path in retry.final_path.iterdir()) == [
            "frames",
            "frames.csv",
        ]
        rows = _read_csv(retry.final_path / "frames.csv")
        assert rows[0]["frame_index"] == "0"
        assert rows[0]["camera_frame_number"] == "2"


def test_frames_csv_contains_raw_facts_and_omits_derived_values(tmp_path: Path) -> None:
    with ContactDatasetWriter(tmp_path, _metadata()) as writer:
        run = _run(writer)
        segment = writer.begin_force_target(run, 2.0)
        writer.submit_frame(segment, _frame(0), capture_elapsed_s=0.4)
        writer.finalize_segment(segment)
        writer.complete_loaded_run(run)
        writer.flush()
        rows = _read_csv(segment.final_path / "frames.csv")

    assert tuple(rows[0]) == FRAME_CSV_COLUMNS
    assert float(rows[0]["capture_elapsed_s"]) == 0.4
    assert float(rows[0]["Fx_N"]) == 0.1
    assert float(rows[0]["Mz_Nm"]) == 0.03
    assert float(rows[0]["camera_bota_time_delta_ms"]) == pytest.approx(1.0)
    for removed in ("target_force_n", "F_mag_N", "torque_mag_Nm", "Fz_share"):
        assert removed not in rows[0]


def test_reader_resolves_specimen_run_force_and_raw_frame_context(tmp_path: Path) -> None:
    with ContactDatasetWriter(tmp_path, _metadata()) as writer:
        run = _run(writer, hole_index=4, repeat_index=3)
        segment = writer.begin_force_target(run, 2.0)
        writer.submit_frame(segment, _frame(0), capture_elapsed_s=0.0)
        writer.finalize_segment(segment)
        writer.complete_loaded_run(run)
        writer.flush()
        session_path = writer.session_path

    records = list(iter_dataset_frames(session_path))
    assert len(records) == 1
    record = records[0]
    assert record.session.material == "solaris"
    assert record.session.morphology == "nominal"
    assert record.session.specimen_id == "solaris_nominal_01"
    assert record.run is not None
    assert record.run.indenter == "sphere_15mm"
    assert record.run.hole_index == 4
    assert record.run.repeat_index == 3
    assert record.target_force_n == 2.0
    assert record.measurements["Fz_N"] == "2.0"
    assert record.rgb_path.is_file()


def test_aborted_run_keeps_completed_segments_and_removes_active_attempt(
    tmp_path: Path,
) -> None:
    with ContactDatasetWriter(tmp_path, _metadata(target_forces_n=(2.0, 5.0))) as writer:
        run = _run(writer)
        complete = writer.begin_force_target(run, 2.0)
        writer.submit_frame(complete, _frame(0), capture_elapsed_s=0.0)
        writer.finalize_segment(complete)
        partial = writer.begin_force_target(run, 5.0)
        writer.submit_frame(partial, _frame(1, 5.0), capture_elapsed_s=0.0)
        writer.abort_loaded_run(run)
        writer.flush()
        stored = json.loads((run.path / "run.json").read_text(encoding="utf-8"))

    assert complete.final_path.is_dir()
    assert not partial.partial_path.exists()
    assert stored["status"] == "aborted"
    assert "completed_targets_n" not in stored


def test_unloaded_capture_inherits_session_specimen_without_own_metadata(
    tmp_path: Path,
) -> None:
    with ContactDatasetWriter(tmp_path, _metadata()) as writer:
        capture = writer.begin_unloaded_capture()
        writer.submit_frame(capture, _frame(0, 0.1), capture_elapsed_s=0.0)
        writer.finalize_segment(capture)
        writer.flush()
        session_path = writer.session_path

    assert sorted(path.name for path in capture.final_path.iterdir()) == [
        "frames",
        "frames.csv",
    ]
    assert not (capture.final_path / "metadata.json").exists()
    assert not (capture.final_path / "summary.json").exists()
    record = next(iter_dataset_frames(session_path))
    assert record.run is None
    assert record.target_force_n is None
    assert record.session.specimen_id == "solaris_nominal_01"


@pytest.mark.parametrize(
    ("target", "name"),
    ((1.0, "force_01N"), (2.0, "force_02N"), (5.0, "force_05N"), (10.0, "force_10N"), (15.0, "force_15N")),
)
def test_force_directory_has_one_formatter_and_parser(target: float, name: str) -> None:
    assert format_force_directory(target) == name
    assert parse_force_directory(name) == target
