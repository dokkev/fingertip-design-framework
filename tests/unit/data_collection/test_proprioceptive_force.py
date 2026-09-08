from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np

from experiments.data_collection.proprioceptive_force import (
    CameraAcquisition,
    CameraContactWorker,
    FTAcquisition,
    ForceEstimator,
    FTSensorWorker,
    MotorAcquisition,
    MotorImpedanceWorker,
    OpticalAcquisition,
    ProprioceptiveRecorder,
    _LatestState,
)
from experiments.hardware.ak40_10 import MotorState
from experiments.hardware.bota import BotaSample, BotaTareOffsets


def _wait_until(predicate, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for worker state")
        time.sleep(0.002)


def test_uncalibrated_force_estimator_returns_none() -> None:
    assert ForceEstimator().predict(0.2, 15.0) is None


def test_camera_contact_gate_uses_normal_force_or_ft_vector_magnitude() -> None:
    normal_sample = FTAcquisition(
        1, 1, 0, 25.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, -2.0
    )
    magnitude_sample = FTAcquisition(
        1, 1, 0, 25.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, None
    )
    assert CameraContactWorker._contact_force(normal_sample) == 2.0
    assert CameraContactWorker._contact_force(magnitude_sample) == 5.0


def test_recorder_writes_independent_streams_and_lossless_rgb(tmp_path: Path) -> None:
    recorder = ProprioceptiveRecorder(tmp_path)
    run_path = recorder.start(
        {
            "motor": {"model": "AK40-10", "id": 13, "kp": 5.0, "kd": 0.1},
            "experiment": {
                "trial": 2,
                "contact_location_gt_mm": None,
                "notes": "synthetic",
            },
        }
    )
    recorder.submit_motor(MotorAcquisition(10, 0.1, 0.2, 0.3, 25.0, 0))
    recorder.submit_ft(
        FTAcquisition(11, 7, 0, 24.0, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, -3.0)
    )
    recorder.submit_optical(
        OpticalAcquisition(
            12,
            "contact",
            True,
            20.0,
            6.5,
            3.0,
            (1.0, 2.0, 3.0, 2.0, 1.0),
        )
    )
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[..., 0] = 73
    recorder.submit_camera(
        CameraAcquisition(13, 1.5, 9, 1.2, 12, image, "unloaded_reference")
    )
    recorder.stop()

    assert sorted(path.name for path in run_path.iterdir()) == [
        "camera",
        "camera_timestamps.csv",
        "ft.csv",
        "metadata.json",
        "motor.csv",
        "optical.csv",
    ]
    stored = cv2.imread(str(run_path / "camera" / "frame_000000.png"))
    assert stored is not None
    assert tuple(stored[0, 0]) == (0, 0, 73)
    with (run_path / "motor.csv").open(newline="", encoding="utf-8") as stream:
        motor_rows = list(csv.DictReader(stream))
    assert motor_rows[0]["torque_Nm"] == "0.3"
    with (run_path / "camera_timestamps.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        camera_rows = list(csv.DictReader(stream))
    assert camera_rows[0]["capture_kind"] == "unloaded_reference"
    metadata = json.loads((run_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == "run_001"
    assert metadata["status"] == "complete"
    assert metadata["sample_counts"] == {
        "camera": 1,
        "ft": 1,
        "motor": 1,
        "optical": 1,
    }
    assert "git_commit" not in metadata
    assert "morphology" not in metadata
    assert "material" not in metadata


class _FailingTracker:
    def process(self, rgb: np.ndarray):
        del rgb
        raise RuntimeError("synthetic detector failure")

    def request_recalibration(self) -> None:
        pass

    def request_unloaded_baseline(self) -> None:
        pass


def test_online_detector_failure_does_not_become_camera_failure(tmp_path: Path) -> None:
    recorder = ProprioceptiveRecorder(tmp_path)
    worker = CameraContactWorker(
        object(),  # type: ignore[arg-type]
        _FailingTracker(),  # type: ignore[arg-type]
        _LatestState(recorder),
        recorder,
    )
    result = worker._process_online(np.zeros((3, 4, 3), dtype=np.uint8))
    assert result.contact_detected is None
    assert "acquisition continues" in result.status


def test_offline_mode_skips_online_detector(tmp_path: Path) -> None:
    recorder = ProprioceptiveRecorder(tmp_path)
    worker = CameraContactWorker(
        object(),  # type: ignore[arg-type]
        _FailingTracker(),  # type: ignore[arg-type]
        _LatestState(recorder),
        recorder,
        contact_processing_mode="offline",
    )
    result = worker._process_online(np.zeros((3, 4, 3), dtype=np.uint8))
    assert result.contact_detected is None
    assert result.status == "online contact disabled; offline processing pending"


def test_offline_ready_recording_copies_reference_and_rate_limits_contact(
    tmp_path: Path,
) -> None:
    recorder = ProprioceptiveRecorder(tmp_path)
    worker = CameraContactWorker(
        object(),  # type: ignore[arg-type]
        _FailingTracker(),  # type: ignore[arg-type]
        _LatestState(recorder),
        recorder,
        contact_processing_mode="both",
        contact_record_rate_hz=5.0,
        offline_reference_frame_count=2,
    )
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    with worker._reference_lock:
        worker._reference_frames.extend(
            [
            CameraAcquisition(1, 1.0, 1, 0.0, 1, image, "unloaded_reference"),
            CameraAcquisition(2, 2.0, 2, 0.0, 2, image, "unloaded_reference"),
            ]
        )
    run_path = recorder.start({"experiment": {"trial": 1}})
    for timestamp_ns in (1_000_000_000, 1_100_000_000, 1_250_000_000):
        worker._record_camera_frame(
            CameraAcquisition(
                timestamp_ns,
                3.0,
                3,
                2.0,
                timestamp_ns,
                image,
                "contact",
            )
        )
    recorder.stop()

    with (run_path / "camera_timestamps.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert [row["capture_kind"] for row in rows] == [
        "unloaded_reference",
        "unloaded_reference",
        "contact",
        "contact",
    ]


class _FakeMotor:
    motor_id = 13

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[float, ...]]] = []

    def enable(self) -> None:
        self.calls.append(("enable", ()))

    def disable(self) -> None:
        self.calls.append(("disable", ()))

    def set_zero(self) -> None:
        self.calls.append(("zero", ()))

    def send_command(
        self,
        position: float,
        velocity: float,
        kp: float,
        kd: float,
        torque: float,
    ) -> None:
        self.calls.append(("command", (position, velocity, kp, kd, torque)))

    def read_state(self, timeout: float) -> MotorState:
        del timeout
        return MotorState(13, 0.01, 0.02, 0.03, 25.0, 0)


def test_motor_worker_is_passive_until_explicit_enable(tmp_path: Path) -> None:
    recorder = ProprioceptiveRecorder(tmp_path)
    state = _LatestState(recorder)
    motor = _FakeMotor()
    worker = MotorImpedanceWorker(
        motor,  # type: ignore[arg-type]
        state,
        recorder,
        kp=12.0,
        kd=0.4,
        command_rate_hz=100.0,
    )
    worker.start()
    time.sleep(0.03)
    assert motor.calls == []

    worker.request_enable()
    _wait_until(lambda: state.snapshot().motor is not None)
    worker.request_disable()
    _wait_until(lambda: state.snapshot().motor_status == "disabled")
    worker.stop()

    assert motor.calls[0] == ("enable", ())
    commands = [call for call in motor.calls if call[0] == "command"]
    assert commands
    assert commands[0][1] == (0.0, 0.0, 12.0, 0.4, 0.0)
    assert motor.calls[-1] == ("disable", ())


def test_motor_zero_is_sent_only_after_explicit_request(tmp_path: Path) -> None:
    recorder = ProprioceptiveRecorder(tmp_path)
    state = _LatestState(recorder)
    motor = _FakeMotor()
    worker = MotorImpedanceWorker(motor, state, recorder)  # type: ignore[arg-type]
    worker.start()
    time.sleep(0.02)
    assert motor.calls == []

    worker.request_set_zero()
    _wait_until(lambda: motor.calls == [("zero", ())])
    worker.stop()

    assert motor.calls == [("zero", ())]


class _FakeFTSensor:
    port = "MOCK"

    def __init__(self) -> None:
        self.tare_offsets = BotaTareOffsets()
        self.sequence = 0
        self.stopped = False

    @staticmethod
    def _sample(sequence: int, force_n: float) -> BotaSample:
        return BotaSample(
            host_time_s=time.monotonic(),
            sensor_timestamp=sequence,
            status=0,
            temperature_c=25.0,
            fx_n=0.0,
            fy_n=0.0,
            fz_n=force_n,
            mx_nm=0.0,
            my_nm=0.0,
            mz_nm=0.0,
            force_magnitude_n=abs(force_n),
            torque_magnitude_nm=0.0,
            fz_share=1.0,
        )

    def start(self) -> None:
        self.stopped = False
        self.sequence = 1

    def stop(self) -> None:
        self.stopped = True

    def wait_for_samples(
        self, after_sequence: int, *, timeout_s: float
    ) -> tuple[tuple[int, BotaSample], ...]:
        if after_sequence + 1 < self.sequence:
            raise RuntimeError("stale sequence cursor")
        if after_sequence < self.sequence:
            return ((self.sequence, self._sample(self.sequence, 4.0)),)
        time.sleep(min(timeout_s, 0.005))
        return ()

    def tare(self) -> BotaTareOffsets:
        self.sequence += 250
        self.tare_offsets = BotaTareOffsets(fz_n=4.0)
        return self.tare_offsets

    def latest_sequenced_sample(self) -> tuple[int, BotaSample]:
        return self.sequence, self._sample(self.sequence, 0.0)


def test_ft_worker_resumes_from_new_sequence_after_tare(tmp_path: Path) -> None:
    recorder = ProprioceptiveRecorder(tmp_path)
    state = _LatestState(recorder)
    sensor = _FakeFTSensor()
    worker = FTSensorWorker(
        sensor,  # type: ignore[arg-type]
        state,
        recorder,
        normal_axis="fz",
    )
    worker.start()
    _wait_until(
        lambda: state.snapshot().ft is not None
        and state.snapshot().ft.sensor_timestamp == 1
    )
    worker.request_tare()
    _wait_until(
        lambda: state.snapshot().ft is not None
        and state.snapshot().ft.sensor_timestamp == 251
    )
    worker.stop()

    assert state.snapshot().ft_error is None
    assert state.snapshot().ft is not None
    assert state.snapshot().ft.force_normal_n == 0.0
