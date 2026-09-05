from __future__ import annotations

import csv
import json
from pathlib import Path
import threading

import cv2
import numpy as np
import pytest

from experiments.data_collection.force_trajectory import (
    ForceTrajectoryConfig,
    ForceTrajectoryPhase,
    ForceTrajectoryState,
    ForceTrajectoryUpdate,
)
from experiments.data_collection.history_dataset import (
    HISTORY_FORMAT_VERSION,
    TRAJECTORY_FRAME_CSV_COLUMNS,
    HistoryDatasetWriter,
    HistorySessionMetadata,
    HistorySynchronizedFrame,
)
from experiments.hardware import BotaSample, BotaTareOffsets


def _metadata(*, sensor_mode: str = "mock") -> HistorySessionMetadata:
    return HistorySessionMetadata(
        material="solaris",
        morphology="flat_opt",
        specimen_id="solaris_flat_opt_history_01",
        camera_model="Mock RGB",
        camera_width=8,
        camera_height=6,
        camera_fps=30,
        camera_exposure_us=1500.0,
        camera_gain=0.0,
        camera_white_balance_k=4600.0,
        bota_serial_port="MOCK",
        bota_tare_offsets=BotaTareOffsets(fx_n=0.1),
        trajectory=ForceTrajectoryConfig(
            min_force_n=2.0,
            max_force_n=3.0,
            ramp_rate_n_per_s=2.0,
            conditioning_cycles=1,
            measurement_cycles=1,
        ),
        sensor_mode=sensor_mode,
        camera_serial_number="mock-camera-01",
        bota_model="Mock Rokubi",
        created_utc="2026-09-05T12:00:00.000+00:00",
        git_commit="abc123",
    )


def _frame(index: int, force_n: float) -> HistorySynchronizedFrame:
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[..., 1] = 20 + index
    sample = BotaSample(
        host_time_s=10.0 + index * 0.2,
        sensor_timestamp=100 + index,
        status=0,
        temperature_c=25.0,
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
    return HistorySynchronizedFrame(
        rgb=rgb,
        camera_host_time_s=sample.host_time_s + 0.001,
        camera_device_timestamp_ms=1000.0 + index * 200.0,
        camera_frame_number=index,
        bota_sample=sample,
    )


def _update(index: int, force_n: float) -> ForceTrajectoryUpdate:
    target = 2.0 + 0.2 * index
    return ForceTrajectoryUpdate(
        state=ForceTrajectoryState.CYCLING,
        phase=ForceTrajectoryPhase.LOADING,
        cycle_index=1,
        cycle_role="conditioning",
        cycle_role_index=1,
        target_force_n=target,
        target_ramp_n_per_s=1.0,
        tracking_error_n=force_n - target,
        trajectory_elapsed_s=0.2 * index,
        phase_elapsed_s=0.2 * index,
        should_capture_frame=True,
        missed_capture_deadlines=0,
        events=(),
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _complete_run(writer: HistoryDatasetWriter):
    run = writer.start_run(indenter="sphere_10mm", hole_index=3)
    for index, force in enumerate((2.1, 2.4)):
        assert writer.submit_trajectory_frame(
            run,
            _frame(index, force),
            _update(index, force),
        )
    writer.complete_run(
        run,
        trajectory_start_host_time_s=10.0,
        trajectory_end_host_time_s=12.0,
        dropped_camera_frame_count=1,
        missed_capture_deadline_count=2,
    )
    writer.flush()
    return run


def test_session_metadata_persists_history_format_and_trajectory(
    tmp_path: Path,
) -> None:
    with HistoryDatasetWriter(tmp_path, _metadata()) as writer:
        stored = json.loads(
            (writer.session_path / "session.json").read_text(encoding="utf-8")
        )

    assert stored["format_version"] == HISTORY_FORMAT_VERSION
    assert stored["specimen"]["specimen_id"] == "solaris_flat_opt_history_01"
    assert stored["camera"]["auto_exposure"] is False
    assert stored["camera"]["exposure_us"] == 1500.0
    assert stored["trajectory"] == {
        "capture_rate_hz": 5.0,
        "conditioning_cycles": 1,
        "high_dwell_s": 1.0,
        "low_dwell_s": 1.0,
        "max_force_n": 3.0,
        "measurement_cycles": 1,
        "min_force_n": 2.0,
        "preload_settle_s": 0.5,
        "preload_tolerance_n": 1.0,
        "ramp_rate_n_per_s": 2.0,
        "release_max_force_n": 1.0,
        "release_settle_s": 0.5,
    }


def test_completed_run_is_atomic_and_contains_continuous_trajectory(
    tmp_path: Path,
) -> None:
    with HistoryDatasetWriter(tmp_path, _metadata()) as writer:
        run = _complete_run(writer)
        session_path = writer.session_path

    assert not run.partial_path.exists()
    assert run.final_path.is_dir()
    assert sorted(path.name for path in run.final_path.iterdir()) == [
        "run.json",
        "trajectory",
    ]
    trajectory = run.final_path / "trajectory"
    assert sorted(path.name for path in trajectory.iterdir()) == [
        "frames",
        "frames.csv",
        "trajectory.json",
    ]
    assert not list(session_path.rglob("*.partial"))
    stored_run = json.loads((run.final_path / "run.json").read_text(encoding="utf-8"))
    assert stored_run["status"] == "complete"
    assert stored_run["hole_index"] == 3
    assert stored_run["contact_position_mm"] == 20.0


def test_trajectory_csv_preserves_actual_target_cycle_and_phase(tmp_path: Path) -> None:
    with HistoryDatasetWriter(tmp_path, _metadata()) as writer:
        run = _complete_run(writer)

    rows = _rows(run.final_path / "trajectory" / "frames.csv")
    assert tuple(rows[0]) == TRAJECTORY_FRAME_CSV_COLUMNS
    assert len(rows) == 2
    assert rows[0]["cycle_role"] == "conditioning"
    assert rows[0]["phase"] == "loading"
    assert float(rows[0]["target_force_N"]) == 2.0
    assert float(rows[0]["force_magnitude_N"]) > 2.1
    assert float(rows[1]["tracking_error_N"]) == pytest.approx(0.2)
    assert float(rows[0]["camera_bota_time_delta_ms"]) == pytest.approx(1.0)
    image = cv2.imread(str(run.final_path / "trajectory" / rows[0]["rgb_filename"]))
    assert image is not None
    assert tuple(image[0, 0]) == (0, 20, 0)


def test_trajectory_json_records_saved_and_missing_observations(tmp_path: Path) -> None:
    with HistoryDatasetWriter(tmp_path, _metadata()) as writer:
        run = _complete_run(writer)

    stored = json.loads(
        (run.final_path / "trajectory" / "trajectory.json").read_text(encoding="utf-8")
    )
    assert stored["saved_frame_count"] == 2
    assert stored["dropped_camera_frame_count"] == 1
    assert stored["dropped_writer_frame_count"] == 0
    assert stored["missed_capture_deadline_count"] == 2
    assert stored["total_conditioning_cycles"] == 1
    assert stored["total_measurement_cycles"] == 1


def test_abort_deletes_partial_run_and_restores_repetition_index(
    tmp_path: Path,
) -> None:
    with HistoryDatasetWriter(tmp_path, _metadata()) as writer:
        aborted = writer.start_run(indenter="sphere_30mm", hole_index=6)
        writer.submit_trajectory_frame(aborted, _frame(0, 2.0), _update(0, 2.0))
        writer.abort_run(aborted)
        writer.flush()
        retry = writer.start_run(indenter="sphere_30mm", hole_index=6)

        assert not aborted.partial_path.exists()
        assert retry.metadata.repetition_index == 1
        writer.abort_run(retry)


def test_mock_sessions_are_isolated_from_physical_namespace(tmp_path: Path) -> None:
    with HistoryDatasetWriter(tmp_path, _metadata(sensor_mode="mock")) as writer:
        assert writer.session_path.parent == tmp_path / "mock"


def test_unloaded_capture_is_separate_from_trajectory_runs(tmp_path: Path) -> None:
    with HistoryDatasetWriter(tmp_path, _metadata()) as writer:
        capture = writer.begin_unloaded_capture(expected_frame_count=2)
        assert writer.submit_unloaded_frame(
            capture, _frame(0, 0.1), capture_elapsed_s=0.0
        )
        assert writer.submit_unloaded_frame(
            capture, _frame(1, 0.1), capture_elapsed_s=0.2
        )
        writer.finalize_unloaded_capture(capture)
        writer.flush()

    assert capture.final_path.is_dir()
    assert len(_rows(capture.final_path / "frames.csv")) == 2
    assert not list((capture.final_path.parents[1] / "runs").iterdir())


def test_writer_overrun_preserves_run_and_records_missing_frame(
    tmp_path: Path, monkeypatch
) -> None:
    entered = threading.Event()
    release = threading.Event()
    real_imwrite = cv2.imwrite

    def blocked_imwrite(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=2.0)
        return real_imwrite(*args, **kwargs)

    monkeypatch.setattr(cv2, "imwrite", blocked_imwrite)
    with HistoryDatasetWriter(tmp_path, _metadata(), frame_queue_capacity=1) as writer:
        run = writer.start_run(indenter="sphere_10mm", hole_index=1)
        assert writer.submit_trajectory_frame(run, _frame(0, 2.0), _update(0, 2.0))
        assert entered.wait(timeout=2.0)
        assert not writer.submit_trajectory_frame(run, _frame(1, 2.2), _update(1, 2.2))
        writer.complete_run(
            run,
            trajectory_start_host_time_s=10.0,
            trajectory_end_host_time_s=12.0,
            dropped_camera_frame_count=0,
            missed_capture_deadline_count=0,
        )
        release.set()
        writer.flush()

    trajectory = json.loads(
        (run.final_path / "trajectory" / "trajectory.json").read_text(encoding="utf-8")
    )
    assert trajectory["saved_frame_count"] == 1
    assert trajectory["dropped_writer_frame_count"] == 1
