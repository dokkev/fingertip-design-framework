"""Headless runtime and lossless recorder for proprioceptive force experiments."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import queue
import threading
from time import monotonic, monotonic_ns
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from experiments.force_estimation import OnlineForceEstimate, OnlineForceEstimator
from experiments.hardware import BotaSample, BotaSerialSensor, RealSenseColorCamera
from experiments.hardware.ak40_10 import AK40_10, KD_MAX, KD_MIN, KP_MAX, KP_MIN
from experiments.hardware.can_io import CanIO
from experiments.localization import LiveLedContactResult, LiveLedContactTracker


MOTOR_COLUMNS = (
    "timestamp_ns",
    "position_rad",
    "velocity_rad_s",
    "torque_Nm",
    "temperature_C",
    "error",
)
FT_COLUMNS = (
    "timestamp_ns",
    "sensor_timestamp",
    "status",
    "fx_N",
    "fy_N",
    "fz_N",
    "tx_Nm",
    "ty_Nm",
    "tz_Nm",
    "temperature_C",
    "force_normal_N",
)
OPTICAL_COLUMNS = (
    "timestamp_ns",
    "contact_detected",
    "contact_location_mm",
    "contact_score_z",
    "top_two_margin_DN",
    "response_led_1_DN",
    "response_led_2_DN",
    "response_led_3_DN",
    "response_led_4_DN",
    "response_led_5_DN",
    "detector_status",
)
CAMERA_COLUMNS = (
    "frame_index",
    "timestamp_ns",
    "filename",
    "capture_kind",
    "camera_device_timestamp_ms",
    "camera_frame_number",
    "ft_contact_force_N",
    "ft_timestamp_ns",
    "camera_ft_time_delta_ms",
)
FORCE_ESTIMATE_COLUMNS = (
    "timestamp_ns",
    "valid",
    "contact",
    "estimated_force_N",
    "torque_Nm",
    "torque_bias_Nm",
    "estimated_contact_location_mm",
    "optical_weights_json",
    "optical_timestamp_ns",
    "optical_age_ms",
    "status",
    "processing_time_ms",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _atomic_json(path: Path, values: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass(frozen=True)
class MotorAcquisition:
    timestamp_ns: int
    position_rad: float
    velocity_rad_s: float
    torque_nm: float
    temperature_c: float
    error: int


@dataclass(frozen=True)
class FTAcquisition:
    timestamp_ns: int
    sensor_timestamp: int
    status: int
    temperature_c: float
    fx_n: float
    fy_n: float
    fz_n: float
    tx_nm: float
    ty_nm: float
    tz_nm: float
    force_normal_n: float | None


@dataclass(frozen=True)
class OpticalAcquisition:
    timestamp_ns: int
    detector_status: str
    contact_detected: bool | None
    contact_location_mm: float | None
    contact_score_z: float | None
    top_two_margin_dn: float | None
    optical_response_dn: tuple[float, ...] | None


@dataclass(frozen=True)
class CameraAcquisition:
    timestamp_ns: int
    device_timestamp_ms: float
    frame_number: int
    ft_contact_force_n: float
    ft_timestamp_ns: int
    rgb: np.ndarray
    capture_kind: str = "contact"

    def __post_init__(self) -> None:
        if self.capture_kind not in {"unloaded_reference", "contact"}:
            raise ValueError(
                "capture_kind must be 'unloaded_reference' or 'contact'"
            )
        image = np.asarray(self.rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("rgb must be an H x W x 3 uint8 image")
        if image.flags.writeable:
            image = image.copy()
            image.setflags(write=False)
        object.__setattr__(self, "rgb", image)


@dataclass(frozen=True)
class RecorderSnapshot:
    status: str
    run_id: str | None
    run_path: Path | None
    started_ns: int | None
    motor_samples: int
    ft_samples: int
    optical_samples: int
    force_estimates: int
    camera_frames: int
    error: str | None


@dataclass(frozen=True)
class RuntimeSnapshot:
    can_status: str
    motor_status: str
    camera_status: str
    ft_status: str
    motor_error: str | None
    camera_error: str | None
    ft_error: str | None
    force_estimator_error: str | None
    motor: MotorAcquisition | None
    ft: FTAcquisition | None
    optical: OpticalAcquisition | None
    force_estimate: OnlineForceEstimate | None
    camera_jpeg: bytes | None
    recorder: RecorderSnapshot


ForceEstimator = OnlineForceEstimator


@dataclass(frozen=True)
class _RecordItem:
    stream: str
    sample: object


class ProprioceptiveRecorder:
    """Write independent native-rate streams through one non-blocking queue."""

    def __init__(
        self,
        output_root: str | Path = "output/experiments/proprioceptive_force",
        *,
        png_compression: int = 1,
    ) -> None:
        if not 0 <= png_compression <= 9:
            raise ValueError("png_compression must be in [0, 9]")
        self.output_root = Path(output_root)
        self.png_compression = png_compression
        self._lock = threading.Lock()
        self._queue: queue.SimpleQueue[_RecordItem | None] | None = None
        self._thread: threading.Thread | None = None
        self._accepting = False
        self._status = "idle"
        self._run_id: str | None = None
        self._run_path: Path | None = None
        self._started_ns: int | None = None
        self._metadata: dict[str, Any] | None = None
        self._counts = {
            "motor": 0,
            "ft": 0,
            "optical": 0,
            "force_estimate": 0,
            "camera": 0,
        }
        self._error: str | None = None

    def start(self, metadata: Mapping[str, Any]) -> Path:
        """Create a new run and begin accepting samples."""

        with self._lock:
            if self._accepting or self._thread is not None:
                raise RuntimeError("recorder is already active")
            self.output_root.mkdir(parents=True, exist_ok=True)
            run_number = 1
            while (self.output_root / f"run_{run_number:03d}").exists():
                run_number += 1
            run_id = f"run_{run_number:03d}"
            run_path = self.output_root / run_id
            (run_path / "camera").mkdir(parents=True)
            started_ns = monotonic_ns()
            stored = dict(metadata)
            stored["run_id"] = run_id
            stored["started_utc"] = _utc_now()
            stored["started_timestamp_ns"] = started_ns
            self._queue = queue.SimpleQueue()
            self._accepting = True
            self._status = "recording"
            self._run_id = run_id
            self._run_path = run_path
            self._started_ns = started_ns
            self._metadata = stored
            self._counts = {
                "motor": 0,
                "ft": 0,
                "optical": 0,
                "force_estimate": 0,
                "camera": 0,
            }
            self._error = None
            _atomic_json(run_path / "metadata.json", stored)
            thread = threading.Thread(
                target=self._writer_loop,
                args=(run_path,),
                name=f"proprioceptive-recorder-{run_id}",
            )
            self._thread = thread
            thread.start()
            return run_path

    def submit_motor(self, sample: MotorAcquisition) -> None:
        self._submit("motor", sample)

    def submit_ft(self, sample: FTAcquisition) -> None:
        self._submit("ft", sample)

    def submit_optical(self, sample: OpticalAcquisition) -> None:
        self._submit("optical", sample)

    def submit_force_estimate(self, sample: OnlineForceEstimate) -> None:
        self._submit("force_estimate", sample)

    def submit_camera(self, sample: CameraAcquisition) -> None:
        self._submit("camera", sample)

    def _submit(self, stream: str, sample: object) -> None:
        with self._lock:
            if not self._accepting:
                return
            work_queue = self._queue
            if work_queue is None:
                raise RuntimeError("active recorder has no writer queue")
            work_queue.put(_RecordItem(stream, sample))

    def stop(self) -> Path | None:
        """Stop admission, flush queued work, and close the current run."""

        with self._lock:
            if self._thread is None:
                return self._run_path
            self._accepting = False
            self._status = "saving"
            work_queue = self._queue
            thread = self._thread
            if work_queue is None:
                raise RuntimeError("active recorder has no writer queue")
            work_queue.put(None)
        thread.join()
        with self._lock:
            self._thread = None
            self._queue = None
            if self._error is None:
                self._status = "idle"
            else:
                self._status = "error"
            path = self._run_path
            error = self._error
        if error is not None:
            raise RuntimeError(error)
        return path

    def snapshot(self) -> RecorderSnapshot:
        with self._lock:
            return RecorderSnapshot(
                status=self._status,
                run_id=self._run_id,
                run_path=self._run_path,
                started_ns=self._started_ns,
                motor_samples=self._counts["motor"],
                ft_samples=self._counts["ft"],
                optical_samples=self._counts["optical"],
                force_estimates=self._counts["force_estimate"],
                camera_frames=self._counts["camera"],
                error=self._error,
            )

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._accepting

    def _writer_loop(self, run_path: Path) -> None:
        streams: dict[str, Any] = {}
        writers: dict[str, csv.DictWriter] = {}
        try:
            specifications = {
                "motor": ("motor.csv", MOTOR_COLUMNS),
                "ft": ("ft.csv", FT_COLUMNS),
                "optical": ("optical.csv", OPTICAL_COLUMNS),
                "force_estimate": (
                    "force_estimate.csv",
                    FORCE_ESTIMATE_COLUMNS,
                ),
                "camera": ("camera_timestamps.csv", CAMERA_COLUMNS),
            }
            for key, (filename, columns) in specifications.items():
                stream = (run_path / filename).open("w", newline="", encoding="utf-8")
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                streams[key] = stream
                writers[key] = writer

            frame_index = 0
            while True:
                assert self._queue is not None
                item = self._queue.get()
                if item is None:
                    break
                if item.stream == "motor":
                    sample = item.sample
                    assert isinstance(sample, MotorAcquisition)
                    writers["motor"].writerow(
                        {
                            "timestamp_ns": sample.timestamp_ns,
                            "position_rad": sample.position_rad,
                            "velocity_rad_s": sample.velocity_rad_s,
                            "torque_Nm": sample.torque_nm,
                            "temperature_C": sample.temperature_c,
                            "error": sample.error,
                        }
                    )
                elif item.stream == "ft":
                    sample = item.sample
                    assert isinstance(sample, FTAcquisition)
                    writers["ft"].writerow(
                        {
                            "timestamp_ns": sample.timestamp_ns,
                            "sensor_timestamp": sample.sensor_timestamp,
                            "status": sample.status,
                            "fx_N": sample.fx_n,
                            "fy_N": sample.fy_n,
                            "fz_N": sample.fz_n,
                            "tx_Nm": sample.tx_nm,
                            "ty_Nm": sample.ty_nm,
                            "tz_Nm": sample.tz_nm,
                            "temperature_C": sample.temperature_c,
                            "force_normal_N": ""
                            if sample.force_normal_n is None
                            else sample.force_normal_n,
                        }
                    )
                elif item.stream == "optical":
                    sample = item.sample
                    assert isinstance(sample, OpticalAcquisition)
                    response = sample.optical_response_dn or ()
                    row: dict[str, object] = {
                        "timestamp_ns": sample.timestamp_ns,
                        "contact_detected": ""
                        if sample.contact_detected is None
                        else int(sample.contact_detected),
                        "contact_location_mm": ""
                        if sample.contact_location_mm is None
                        else sample.contact_location_mm,
                        "contact_score_z": ""
                        if sample.contact_score_z is None
                        else sample.contact_score_z,
                        "top_two_margin_DN": ""
                        if sample.top_two_margin_dn is None
                        else sample.top_two_margin_dn,
                        "detector_status": sample.detector_status,
                    }
                    for index in range(5):
                        row[f"response_led_{index + 1}_DN"] = (
                            response[index] if index < len(response) else ""
                        )
                    writers["optical"].writerow(row)
                elif item.stream == "force_estimate":
                    sample = item.sample
                    assert isinstance(sample, OnlineForceEstimate)
                    writers["force_estimate"].writerow(
                        {
                            "timestamp_ns": sample.timestamp_ns,
                            "valid": int(sample.valid),
                            "contact": int(sample.contact),
                            "estimated_force_N": ""
                            if sample.estimated_force_n is None
                            else sample.estimated_force_n,
                            "torque_Nm": sample.torque_nm,
                            "torque_bias_Nm": ""
                            if sample.torque_bias_nm is None
                            else sample.torque_bias_nm,
                            "estimated_contact_location_mm": ""
                            if sample.contact_location_mm is None
                            else sample.contact_location_mm,
                            "optical_weights_json": ""
                            if sample.optical_weights is None
                            else json.dumps(sample.optical_weights),
                            "optical_timestamp_ns": ""
                            if sample.optical_timestamp_ns is None
                            else sample.optical_timestamp_ns,
                            "optical_age_ms": ""
                            if sample.optical_age_ms is None
                            else sample.optical_age_ms,
                            "status": sample.status,
                            "processing_time_ms": sample.processing_time_ms,
                        }
                    )
                elif item.stream == "camera":
                    sample = item.sample
                    assert isinstance(sample, CameraAcquisition)
                    filename = f"frame_{frame_index:06d}.png"
                    destination = run_path / "camera" / filename
                    written = cv2.imwrite(
                        str(destination),
                        cv2.cvtColor(sample.rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression],
                    )
                    if not written:
                        raise RuntimeError(f"failed to write lossless frame {destination}")
                    writers["camera"].writerow(
                        {
                            "frame_index": frame_index,
                            "timestamp_ns": sample.timestamp_ns,
                            "filename": f"camera/{filename}",
                            "capture_kind": sample.capture_kind,
                            "camera_device_timestamp_ms": sample.device_timestamp_ms,
                            "camera_frame_number": sample.frame_number,
                            "ft_contact_force_N": sample.ft_contact_force_n,
                            "ft_timestamp_ns": sample.ft_timestamp_ns,
                            "camera_ft_time_delta_ms": (
                                sample.timestamp_ns - sample.ft_timestamp_ns
                            )
                            / 1.0e6,
                        }
                    )
                    frame_index += 1
                else:
                    raise RuntimeError(f"unknown recorder stream {item.stream!r}")
                with self._lock:
                    self._counts[item.stream] += 1
        except BaseException as error:
            with self._lock:
                self._error = f"{type(error).__name__}: {error}"
        finally:
            for stream in streams.values():
                stream.flush()
                stream.close()
            with self._lock:
                if self._metadata is not None and self._run_path == run_path:
                    self._metadata["ended_utc"] = _utc_now()
                    self._metadata["ended_timestamp_ns"] = monotonic_ns()
                    self._metadata["sample_counts"] = dict(self._counts)
                    self._metadata["status"] = (
                        "complete" if self._error is None else "error"
                    )
                    if self._error is not None:
                        self._metadata["error"] = self._error
                    _atomic_json(run_path / "metadata.json", self._metadata)


class _LatestState:
    def __init__(self, recorder: ProprioceptiveRecorder) -> None:
        self._lock = threading.Lock()
        self.can_status = "disconnected"
        self.motor_status = "disconnected"
        self.camera_status = "disconnected"
        self.ft_status = "disconnected"
        self.motor_error: str | None = None
        self.camera_error: str | None = None
        self.ft_error: str | None = None
        self.force_estimator_error: str | None = None
        self.motor: MotorAcquisition | None = None
        self.ft: FTAcquisition | None = None
        self.optical: OpticalAcquisition | None = None
        self.force_estimate: OnlineForceEstimate | None = None
        self.camera_jpeg: bytes | None = None
        self.recorder = recorder

    def update(self, **values: object) -> None:
        with self._lock:
            for name, value in values.items():
                setattr(self, name, value)

    def snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            return RuntimeSnapshot(
                can_status=self.can_status,
                motor_status=self.motor_status,
                camera_status=self.camera_status,
                ft_status=self.ft_status,
                motor_error=self.motor_error,
                camera_error=self.camera_error,
                ft_error=self.ft_error,
                force_estimator_error=self.force_estimator_error,
                motor=self.motor,
                ft=self.ft,
                optical=self.optical,
                force_estimate=self.force_estimate,
                camera_jpeg=self.camera_jpeg,
                recorder=self.recorder.snapshot(),
            )

    def latest_ft(self) -> FTAcquisition | None:
        with self._lock:
            return self.ft


class MotorImpedanceWorker:
    """Own explicit AK40-10 actions and the periodic fixed-impedance loop."""

    def __init__(
        self,
        motor: AK40_10,
        state: _LatestState,
        recorder: ProprioceptiveRecorder,
        *,
        kp: float = 0.0,
        kd: float = 0.0,
        command_rate_hz: float = 100.0,
        feedback_timeout_s: float = 0.05,
        force_estimator: OnlineForceEstimator | None = None,
    ) -> None:
        self._motor = motor
        self._state = state
        self._recorder = recorder
        self._force_estimator = force_estimator
        self.command_rate_hz = float(command_rate_hz)
        self.feedback_timeout_s = float(feedback_timeout_s)
        if not math.isfinite(self.command_rate_hz) or self.command_rate_hz <= 0.0:
            raise ValueError("command_rate_hz must be finite and positive")
        if not math.isfinite(self.feedback_timeout_s) or self.feedback_timeout_s <= 0.0:
            raise ValueError("feedback_timeout_s must be finite and positive")
        self._condition = threading.Condition()
        self._kp = self._gain("kp", kp, KP_MIN, KP_MAX)
        self._kd = self._gain("kd", kd, KD_MIN, KD_MAX)
        self._enable_requested = False
        self._disable_requested = False
        self._zero_requested = False
        self._enabled = False
        self._stop = False
        self._thread: threading.Thread | None = None

    @staticmethod
    def _gain(name: str, value: float, minimum: float, maximum: float) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
            raise ValueError(f"{name} must be within [{minimum:g}, {maximum:g}]")
        return parsed

    @property
    def gains(self) -> tuple[float, float]:
        with self._condition:
            return self._kp, self._kd

    @property
    def enabled(self) -> bool:
        with self._condition:
            return self._enabled

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("motor worker is already running")
        self._stop = False
        self._state.update(can_status="connected", motor_status="disabled")
        self._thread = threading.Thread(
            target=self._run,
            name=f"ak40-impedance-{self._motor.motor_id}",
        )
        self._thread.start()

    def set_gains(self, kp: float, kd: float) -> None:
        new_kp = self._gain("kp", kp, KP_MIN, KP_MAX)
        new_kd = self._gain("kd", kd, KD_MIN, KD_MAX)
        with self._condition:
            self._kp = new_kp
            self._kd = new_kd
            self._condition.notify_all()

    def request_enable(self) -> None:
        with self._condition:
            self._enable_requested = True
            self._disable_requested = False
            self._condition.notify_all()

    def request_disable(self) -> None:
        with self._condition:
            self._disable_requested = True
            self._enable_requested = False
            self._condition.notify_all()

    def request_set_zero(self) -> None:
        with self._condition:
            self._zero_requested = True
            self._condition.notify_all()

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        thread.join(timeout=self.feedback_timeout_s + 2.0)
        if thread.is_alive():
            raise RuntimeError("motor worker did not stop within its bounded wait")
        self._thread = None

    def _run(self) -> None:
        period_s = 1.0 / self.command_rate_hz
        next_command_s = monotonic()
        try:
            while True:
                with self._condition:
                    if (
                        not self._stop
                        and not self._enabled
                        and not self._enable_requested
                        and not self._disable_requested
                        and not self._zero_requested
                    ):
                        self._condition.wait()
                    if self._stop:
                        break
                    zero_requested = self._zero_requested
                    enable_requested = self._enable_requested
                    disable_requested = self._disable_requested
                    self._zero_requested = False
                    self._enable_requested = False
                    self._disable_requested = False
                    kp, kd = self._kp, self._kd

                if disable_requested:
                    self._safe_disable()
                    continue
                if zero_requested:
                    self._motor.set_zero()
                if enable_requested and not self._enabled:
                    self._motor.enable()
                    with self._condition:
                        self._enabled = True
                    self._state.update(motor_status="enabled", motor_error=None)
                    next_command_s = monotonic()
                if not self._enabled:
                    continue

                self._motor.send_command(
                    position=0.0,
                    velocity=0.0,
                    kp=kp,
                    kd=kd,
                    torque=0.0,
                )
                motor_state = self._motor.read_state(self.feedback_timeout_s)
                if motor_state is None:
                    raise TimeoutError(
                        f"no AK40-10 feedback within {self.feedback_timeout_s:g} s"
                    )
                if motor_state.error != 0:
                    raise RuntimeError(
                        f"AK40-10 reported error code {motor_state.error}"
                    )
                values = (
                    motor_state.position,
                    motor_state.velocity,
                    motor_state.torque,
                    motor_state.temperature,
                )
                if not all(math.isfinite(value) for value in values):
                    raise RuntimeError("AK40-10 returned non-finite feedback")
                sample = MotorAcquisition(
                    timestamp_ns=monotonic_ns(),
                    position_rad=motor_state.position,
                    velocity_rad_s=motor_state.velocity,
                    torque_nm=motor_state.torque,
                    temperature_c=motor_state.temperature,
                    error=motor_state.error,
                )
                force_estimate = None
                if self._force_estimator is not None:
                    try:
                        force_estimate = self._force_estimator.update_torque(
                            sample.timestamp_ns,
                            sample.torque_nm,
                        )
                    except Exception as error:
                        self._state.update(
                            force_estimator_error=(
                                f"{type(error).__name__}: {error}"
                            )
                        )
                self._state.update(motor=sample, force_estimate=force_estimate)
                self._recorder.submit_motor(sample)
                if force_estimate is not None:
                    self._recorder.submit_force_estimate(force_estimate)

                next_command_s += period_s
                delay_s = next_command_s - monotonic()
                with self._condition:
                    if delay_s > 0.0:
                        self._condition.wait(timeout=delay_s)
                    else:
                        next_command_s = monotonic()
        except BaseException as error:
            message = f"{type(error).__name__}: {error}"
            self._state.update(motor_status="error", motor_error=message)
        finally:
            self._safe_disable()

    def _safe_disable(self) -> None:
        with self._condition:
            was_enabled = self._enabled
            self._enabled = False
        if was_enabled:
            try:
                self._motor.disable()
            except BaseException as error:
                self._state.update(
                    motor_status="error",
                    motor_error=f"{type(error).__name__}: {error}",
                )
                return
        current = self._state.snapshot()
        if current.motor_status != "error":
            self._state.update(motor_status="disabled")


class FTSensorWorker:
    """Forward each native Rokubi sample into runtime state and the recorder."""

    _AXES = {"fx": "fx_n", "fy": "fy_n", "fz": "fz_n"}

    def __init__(
        self,
        sensor: BotaSerialSensor,
        state: _LatestState,
        recorder: ProprioceptiveRecorder,
        *,
        normal_axis: str | None,
        normal_sign: float = 1.0,
    ) -> None:
        if normal_axis is not None and normal_axis not in self._AXES:
            raise ValueError("normal_axis must be fx, fy, fz, or None")
        normal_sign = float(normal_sign)
        if normal_sign not in {-1.0, 1.0}:
            raise ValueError("normal_sign must be +1 or -1")
        self._sensor = sensor
        self._state = state
        self._recorder = recorder
        self.normal_axis = normal_axis
        self.normal_sign = normal_sign
        self._stop = threading.Event()
        self._tare_requested = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("F/T worker is already running")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rokubi-acquisition")
        self._thread.start()

    def request_tare(self) -> None:
        self._tare_requested.set()

    def stop(self) -> None:
        self._stop.set()
        self._sensor.stop()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)
            if thread.is_alive():
                raise RuntimeError("F/T worker did not stop within its bounded wait")
        self._thread = None

    def _run(self) -> None:
        sequence = 0
        try:
            self._state.update(ft_status="connecting", ft_error=None)
            self._sensor.start()
            self._state.update(ft_status="connected")
            while not self._stop.is_set():
                if self._tare_requested.is_set():
                    self._tare_requested.clear()
                    self._state.update(ft_status="taring")
                    self._sensor.tare()
                    sequenced = self._sensor.latest_sequenced_sample()
                    if sequenced is not None:
                        sequence, source = sequenced
                        sample = self._sample(source)
                        self._state.update(ft=sample)
                    self._state.update(ft_status="connected")
                samples = self._sensor.wait_for_samples(sequence, timeout_s=0.2)
                for sequence, source in samples:
                    sample = self._sample(source)
                    self._state.update(ft=sample)
                    self._recorder.submit_ft(sample)
        except BaseException as error:
            if not self._stop.is_set():
                self._state.update(
                    ft_status="error",
                    ft_error=f"{type(error).__name__}: {error}",
                )
        finally:
            self._sensor.stop()
            if self._state.snapshot().ft_status != "error":
                self._state.update(ft_status="disconnected")

    def _sample(self, source: BotaSample) -> FTAcquisition:
        normal = None
        if self.normal_axis is not None:
            normal = self.normal_sign * float(
                getattr(source, self._AXES[self.normal_axis])
            )
        return FTAcquisition(
            timestamp_ns=round(source.host_time_s * 1.0e9),
            sensor_timestamp=source.sensor_timestamp,
            status=source.status,
            temperature_c=source.temperature_c,
            fx_n=source.fx_n,
            fy_n=source.fy_n,
            fz_n=source.fz_n,
            tx_nm=source.mx_nm,
            ty_nm=source.my_nm,
            tz_nm=source.mz_nm,
            force_normal_n=normal,
        )


class CameraContactWorker:
    """Acquire RGB independently of optional online contact processing."""

    def __init__(
        self,
        camera: RealSenseColorCamera,
        tracker: LiveLedContactTracker,
        state: _LatestState,
        recorder: ProprioceptiveRecorder,
        *,
        warmup_frame_count: int = 30,
        read_timeout_ms: int = 2000,
        preview_width: int = 960,
        contact_frame_threshold_n: float = 0.5,
        contact_processing_mode: str = "both",
        contact_record_rate_hz: float = 5.0,
        offline_reference_frame_count: int = 30,
        force_estimator: OnlineForceEstimator | None = None,
    ) -> None:
        if warmup_frame_count < 0:
            raise ValueError("warmup_frame_count must be nonnegative")
        if read_timeout_ms <= 0 or preview_width <= 0:
            raise ValueError("camera timeout and preview width must be positive")
        contact_frame_threshold_n = float(contact_frame_threshold_n)
        if not math.isfinite(contact_frame_threshold_n) or contact_frame_threshold_n < 0.0:
            raise ValueError("contact_frame_threshold_n must be finite and nonnegative")
        if contact_processing_mode not in {"online", "offline", "both"}:
            raise ValueError(
                "contact_processing_mode must be online, offline, or both"
            )
        contact_record_rate_hz = float(contact_record_rate_hz)
        if not math.isfinite(contact_record_rate_hz) or contact_record_rate_hz <= 0.0:
            raise ValueError("contact_record_rate_hz must be finite and positive")
        if (
            not isinstance(offline_reference_frame_count, int)
            or isinstance(offline_reference_frame_count, bool)
            or offline_reference_frame_count < 0
        ):
            raise ValueError(
                "offline_reference_frame_count must be a nonnegative integer"
            )
        self._camera = camera
        self._tracker = tracker
        self._force_estimator = force_estimator
        self._state = state
        self._recorder = recorder
        self.warmup_frame_count = warmup_frame_count
        self.read_timeout_ms = read_timeout_ms
        self.preview_width = preview_width
        self.contact_frame_threshold_n = contact_frame_threshold_n
        self.contact_processing_mode = contact_processing_mode
        self.contact_record_rate_hz = contact_record_rate_hz
        self.offline_reference_frame_count = offline_reference_frame_count
        self._reference_frames: deque[CameraAcquisition] = deque(
            maxlen=offline_reference_frame_count
        )
        self._reference_lock = threading.Lock()
        self._contact_frame_gate: Callable[[CameraAcquisition], bool] | None = None
        self._contact_frame_gate_lock = threading.Lock()
        self._recording_run_id: str | None = None
        self._last_contact_record_ns: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def online_contact_enabled(self) -> bool:
        return self.contact_processing_mode in {"online", "both"}

    @property
    def offline_contact_enabled(self) -> bool:
        return self.contact_processing_mode in {"offline", "both"}

    @property
    def offline_reference_count(self) -> int:
        with self._reference_lock:
            return len(self._reference_frames)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("camera worker is already running")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="realsense-contact-acquisition",
        )
        self._thread.start()

    def request_recalibration(self) -> None:
        if not self.online_contact_enabled and self._force_estimator is None:
            raise RuntimeError("online contact processing is disabled")
        if self.online_contact_enabled:
            self._tracker.request_recalibration()
        if self._force_estimator is not None:
            self._force_estimator.begin_geometry_initialization()

    def request_unloaded_baseline(self) -> None:
        if not self.online_contact_enabled and self._force_estimator is None:
            raise RuntimeError("online contact processing is disabled")
        if self._force_estimator is not None:
            self._force_estimator.begin_unloaded_baseline()
        if self.online_contact_enabled:
            self._tracker.request_unloaded_baseline()

    def set_contact_frame_gate(
        self,
        gate: Callable[[CameraAcquisition], bool] | None,
    ) -> None:
        """Optionally restrict recorded contact frames without gating raw sensors."""

        with self._contact_frame_gate_lock:
            self._contact_frame_gate = gate

    def stop(self) -> None:
        self._stop.set()
        self._camera.stop()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self.read_timeout_ms / 1000.0 + 2.0)
            if thread.is_alive():
                raise RuntimeError("camera worker did not stop within its bounded wait")
        self._thread = None

    def _run(self) -> None:
        try:
            self._state.update(camera_status="connecting", camera_error=None)
            self._camera.start()
            for _ in range(self.warmup_frame_count):
                if self._stop.is_set():
                    return
                self._camera.read(timeout_ms=self.read_timeout_ms)
            self._state.update(camera_status="calibrating")
            while not self._stop.is_set():
                source = self._camera.read(timeout_ms=self.read_timeout_ms)
                timestamp_ns = monotonic_ns()
                if self._force_estimator is not None:
                    try:
                        self._force_estimator.update_optical(timestamp_ns, source.rgb)
                        self._state.update(force_estimator_error=None)
                    except Exception as error:
                        self._state.update(
                            force_estimator_error=(
                                f"{type(error).__name__}: {error}"
                            )
                        )
                ft_sample = self._state.latest_ft()
                if ft_sample is not None:
                    contact_force_n = self._contact_force(ft_sample)
                    camera_sample = CameraAcquisition(
                        timestamp_ns=timestamp_ns,
                        device_timestamp_ms=source.timestamp_ms,
                        frame_number=source.frame_number,
                        ft_contact_force_n=contact_force_n,
                        ft_timestamp_ns=ft_sample.timestamp_ns,
                        rgb=source.rgb,
                        capture_kind=(
                            "contact"
                            if contact_force_n >= self.contact_frame_threshold_n
                            else "unloaded_reference"
                        ),
                    )
                    if (
                        self.offline_contact_enabled
                        and contact_force_n < self.contact_frame_threshold_n
                    ):
                        with self._reference_lock:
                            self._reference_frames.append(camera_sample)
                    self._record_camera_frame(camera_sample)
                result = self._process_online(source.rgb)
                optical = OpticalAcquisition(
                    timestamp_ns=timestamp_ns,
                    detector_status=result.status,
                    contact_detected=result.contact_detected,
                    contact_location_mm=result.contact_location_mm,
                    contact_score_z=result.contact_score_z,
                    top_two_margin_dn=result.top_two_margin_dn,
                    optical_response_dn=result.optical_response_dn,
                )
                self._recorder.submit_optical(optical)
                self._state.update(
                    camera_status="connected",
                    camera_error=None,
                    optical=optical,
                    camera_jpeg=self._preview(source.rgb, result),
                )
        except BaseException as error:
            if not self._stop.is_set():
                self._state.update(
                    camera_status="error",
                    camera_error=f"{type(error).__name__}: {error}",
                )
        finally:
            self._camera.stop()
            if self._state.snapshot().camera_status != "error":
                self._state.update(camera_status="disconnected")

    def _process_online(self, rgb: np.ndarray) -> LiveLedContactResult:
        if not self.online_contact_enabled:
            return self._empty_result(
                "online contact disabled; offline processing pending"
            )
        try:
            return self._tracker.process(rgb)
        except Exception as error:
            return self._empty_result(
                "online detector error; acquisition continues: "
                f"{type(error).__name__}: {error}"
            )

    @staticmethod
    def _empty_result(status: str) -> LiveLedContactResult:
        return LiveLedContactResult(
            status=status,
            geometry_ready=False,
            baseline_ready=False,
            contact_detected=None,
            contact_location_mm=None,
            contact_score_z=None,
            top_two_margin_dn=None,
            optical_response_dn=None,
            landmarks_xy_px=None,
            contact_point_xy_px=None,
        )

    def _record_camera_frame(self, sample: CameraAcquisition) -> None:
        recorder = self._recorder.snapshot()
        run_id = recorder.run_id if recorder.status == "recording" else None
        if run_id != self._recording_run_id:
            self._recording_run_id = run_id
            self._last_contact_record_ns = None
            if run_id is not None and self.offline_contact_enabled:
                with self._reference_lock:
                    references = tuple(self._reference_frames)
                for reference in references:
                    self._recorder.submit_camera(reference)
        if run_id is None:
            return
        with self._contact_frame_gate_lock:
            gate = self._contact_frame_gate
        if sample.capture_kind != "contact" or (
            gate is not None and not gate(sample)
        ):
            if gate is not None:
                self._last_contact_record_ns = None
            return
        period_ns = round(1.0e9 / self.contact_record_rate_hz)
        if (
            self._last_contact_record_ns is None
            or sample.timestamp_ns - self._last_contact_record_ns >= period_ns
        ):
            self._recorder.submit_camera(sample)
            self._last_contact_record_ns = sample.timestamp_ns

    @staticmethod
    def _contact_force(sample: FTAcquisition) -> float:
        if sample.force_normal_n is not None:
            return abs(sample.force_normal_n)
        return math.sqrt(sample.fx_n**2 + sample.fy_n**2 + sample.fz_n**2)

    def _preview(self, rgb: np.ndarray, result: LiveLedContactResult) -> bytes:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        scale = min(1.0, self.preview_width / bgr.shape[1])
        if scale < 1.0:
            bgr = cv2.resize(
                bgr,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        if result.landmarks_xy_px is not None:
            for index, point in enumerate(result.landmarks_xy_px, start=1):
                center = (round(scale * point[0]), round(scale * point[1]))
                cv2.circle(bgr, center, 4, (0, 220, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    bgr,
                    str(index),
                    (center[0] + 5, center[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        if result.contact_point_xy_px is not None:
            point = result.contact_point_xy_px
            center = (round(scale * point[0]), round(scale * point[1]))
            cv2.circle(bgr, center, 10, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(bgr, center, 7, (0, 70, 255), -1, cv2.LINE_AA)
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 32), (20, 20, 20), -1)
        cv2.putText(
            bgr,
            result.status,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        success, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 82],
        )
        if not success:
            raise RuntimeError("failed to encode the NiceGUI preview")
        return encoded.tobytes()


class ProprioceptiveExperimentRuntime:
    """Assemble device workers, shared state, recording, and explicit actions."""

    def __init__(
        self,
        *,
        can_io: CanIO,
        motor: AK40_10,
        camera: RealSenseColorCamera,
        ft_sensor: BotaSerialSensor,
        output_root: str | Path,
        normal_axis: str | None,
        normal_sign: float = 1.0,
        kp: float = 0.0,
        kd: float = 0.0,
        motor_rate_hz: float = 100.0,
        motor_feedback_timeout_s: float = 0.05,
        camera_contact_threshold_n: float = 0.5,
        contact_processing_mode: str = "both",
        camera_record_rate_hz: float = 5.0,
        offline_reference_frame_count: int = 30,
        force_estimator: OnlineForceEstimator | None = None,
    ) -> None:
        self.can_io = can_io
        self.motor_device = motor
        self.camera_device = camera
        self.ft_sensor = ft_sensor
        self.recorder = ProprioceptiveRecorder(output_root)
        self._state = _LatestState(self.recorder)
        self.motor = MotorImpedanceWorker(
            motor,
            self._state,
            self.recorder,
            kp=kp,
            kd=kd,
            command_rate_hz=motor_rate_hz,
            feedback_timeout_s=motor_feedback_timeout_s,
            force_estimator=force_estimator,
        )
        self.ft = FTSensorWorker(
            ft_sensor,
            self._state,
            self.recorder,
            normal_axis=normal_axis,
            normal_sign=normal_sign,
        )
        self.camera = CameraContactWorker(
            camera,
            LiveLedContactTracker(),
            self._state,
            self.recorder,
            contact_frame_threshold_n=camera_contact_threshold_n,
            contact_processing_mode=contact_processing_mode,
            contact_record_rate_hz=camera_record_rate_hz,
            offline_reference_frame_count=offline_reference_frame_count,
            force_estimator=force_estimator,
        )
        self.estimator = force_estimator
        self._shutdown_lock = threading.Lock()
        self._shutdown = False

    def start(self) -> None:
        """Start passive workers; this sends no motor command."""

        self.motor.start()
        self.ft.start()
        self.camera.start()

    def snapshot(self) -> RuntimeSnapshot:
        return self._state.snapshot()

    def set_motor_gains(self, kp: float, kd: float) -> None:
        if self.recorder.is_recording:
            raise RuntimeError("Kp/Kd are locked while recording")
        self.motor.set_gains(kp, kd)

    def set_zero(self) -> None:
        self.motor.request_set_zero()

    def enable_motor(self) -> None:
        self.motor.request_enable()

    def disable_motor(self) -> None:
        self.motor.request_disable()

    def tare_ft(self) -> None:
        if self.recorder.is_recording:
            raise RuntimeError("F/T tare is disabled while recording")
        self.ft.request_tare()

    def recalibrate_contact(self) -> None:
        if self.recorder.is_recording:
            raise RuntimeError("contact recalibration is disabled while recording")
        self.camera.request_recalibration()

    def acquire_unloaded_baseline(self) -> None:
        if self.recorder.is_recording:
            raise RuntimeError("unloaded baseline is disabled while recording")
        self.camera.request_unloaded_baseline()

    def set_force_torque_bias(self, torque_nm: float | None = None) -> None:
        """Set force-estimator bias explicitly, optionally from latest feedback."""

        if self.estimator is None:
            raise RuntimeError("no force estimator is configured")
        if torque_nm is None:
            motor = self.snapshot().motor
            if motor is None:
                raise RuntimeError("motor feedback is unavailable")
            torque_nm = motor.torque_nm
        self.estimator.set_torque_bias(torque_nm)

    def start_recording(
        self,
        *,
        trial: int,
        contact_location_gt_mm: float | None,
        notes: str | None,
        experiment_context: Mapping[str, Any] | None = None,
    ) -> Path:
        snapshot = self.snapshot()
        if not self.motor.enabled or snapshot.motor is None:
            raise RuntimeError("enable the motor and verify feedback before recording")
        if snapshot.ft_status != "connected" or snapshot.ft is None:
            raise RuntimeError("F/T sensor is not providing data")
        if snapshot.camera_jpeg is None or snapshot.camera_status == "error":
            raise RuntimeError("camera is not providing data")
        if (
            self.camera.offline_contact_enabled
            and self.camera.offline_reference_count
            < self.camera.offline_reference_frame_count
        ):
            raise RuntimeError(
                "offline unloaded reference is not ready: "
                f"{self.camera.offline_reference_count}/"
                f"{self.camera.offline_reference_frame_count} frames; keep the "
                "fingertip below the contact threshold before recording"
            )
        if not isinstance(trial, int) or isinstance(trial, bool) or trial < 1:
            raise ValueError("trial must be a positive integer")
        if contact_location_gt_mm is not None and not math.isfinite(
            float(contact_location_gt_mm)
        ):
            raise ValueError("contact_location_gt_mm must be finite when supplied")
        kp, kd = self.motor.gains
        experiment_metadata: dict[str, Any] = {
            "trial": trial,
            "contact_location_gt_mm": contact_location_gt_mm,
            "notes": None if notes is None or not notes.strip() else notes.strip(),
        }
        if experiment_context is not None:
            context = dict(experiment_context)
            collisions = sorted(set(experiment_metadata).intersection(context))
            if collisions:
                raise ValueError(
                    "experiment_context contains reserved keys: "
                    + ", ".join(collisions)
                )
            try:
                json.dumps(context)
            except (TypeError, ValueError) as error:
                raise ValueError("experiment_context must be JSON serializable") from error
            experiment_metadata.update(context)
        metadata = {
            "motor": {
                "model": "AK40-10",
                "id": self.motor_device.motor_id,
                "kp": kp,
                "kd": kd,
                "target_position_rad": 0.0,
                "target_velocity_rad_s": 0.0,
                "feedforward_torque_Nm": 0.0,
                "command_rate_hz": self.motor.command_rate_hz,
            },
            "experiment": experiment_metadata,
            "acquisition": {
                "clock": "time.monotonic_ns",
                "camera": {
                    "model": self.camera_device.device_name,
                    "serial_number": self.camera_device.serial_number,
                    "width": self.camera_device.width,
                    "height": self.camera_device.height,
                    "fps": self.camera_device.fps,
                    "exposure_us": self.camera_device.exposure_us,
                    "gain": self.camera_device.gain,
                    "white_balance_k": self.camera_device.white_balance_k,
                    "frame_recording_policy": (
                        "rolling_unloaded_reference_and_ft_contact"
                        if self.camera.offline_contact_enabled
                        else "ft_contact_only"
                    ),
                    "contact_threshold_n": self.camera.contact_frame_threshold_n,
                    "contact_record_rate_hz": self.camera.contact_record_rate_hz,
                    "unloaded_reference_frame_count": (
                        self.camera.offline_reference_frame_count
                        if self.camera.offline_contact_enabled
                        else 0
                    ),
                    "contact_force_signal": (
                        f"abs({self.ft.normal_axis})"
                        if self.ft.normal_axis is not None
                        else "force_vector_magnitude"
                    ),
                },
                "ft_sensor": {
                    "model": "Bota Rokubi",
                    "serial_port": self.ft_sensor.port,
                    "normal_axis": self.ft.normal_axis,
                    "normal_sign": self.ft.normal_sign,
                    "tare_offsets": asdict(self.ft_sensor.tare_offsets),
                },
                "optical_detector": {
                    "method": "five-LED top-10% red response",
                    "processing_mode": self.camera.contact_processing_mode,
                    "online_failure_policy": "record acquisition and report unavailable",
                },
                "force_estimator": None
                if self.estimator is None
                else {
                    "method": "optical-weighted affine torque-to-force models",
                    "calibration": self.estimator.calibration.to_mapping(),
                    "contact_enter_threshold_Nm": (
                        self.estimator.contact_enter_threshold_nm
                    ),
                    "contact_exit_threshold_Nm": (
                        self.estimator.contact_exit_threshold_nm
                    ),
                    "optical_to_motor_offset_ns": (
                        self.estimator.optical_to_motor_offset_ns
                    ),
                    "torque_bias_Nm": self.estimator.torque_bias_nm,
                },
            },
        }
        return self.recorder.start(metadata)

    def stop_recording(self) -> Path | None:
        return self.recorder.stop()

    def shutdown(self) -> None:
        """Flush recording and stop all devices, disabling the motor best-effort."""

        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True
        errors: list[str] = []
        for action in (
            self.motor.stop,
            self.camera.stop,
            self.ft.stop,
            self.recorder.stop,
            self.can_io.close,
        ):
            try:
                action()
            except BaseException as error:
                errors.append(f"{type(error).__name__}: {error}")
        if errors:
            self._state.update(motor_error="; ".join(errors))


__all__ = [
    "CameraAcquisition",
    "FTAcquisition",
    "ForceEstimator",
    "OnlineForceEstimate",
    "OnlineForceEstimator",
    "MotorAcquisition",
    "ProprioceptiveExperimentRuntime",
    "ProprioceptiveRecorder",
    "RecorderSnapshot",
    "RuntimeSnapshot",
]
