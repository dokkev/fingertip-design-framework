"""Format-v1 storage for continuous contact-history trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
from time import monotonic, sleep
from typing import Any, Callable, Mapping

import numpy as np

from experiments.hardware import BotaSample, BotaTareOffsets

from .force_trajectory import (
    ForceTrajectoryConfig,
    ForceTrajectoryState,
    ForceTrajectoryUpdate,
)


HISTORY_FORMAT_VERSION = 1
HOLE_CONTACT_POSITIONS_MM = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
TRAJECTORY_FRAME_CSV_COLUMNS = (
    "frame_index",
    "rgb_filename",
    "capture_elapsed_s",
    "camera_host_time_s",
    "camera_device_timestamp_ms",
    "camera_frame_number",
    "bota_host_time_s",
    "bota_sensor_timestamp",
    "camera_bota_time_delta_ms",
    "Fx_N",
    "Fy_N",
    "Fz_N",
    "Mx_Nm",
    "My_Nm",
    "Mz_Nm",
    "temperature_C",
    "bota_status",
    "force_magnitude_N",
    "torque_magnitude_Nm",
    "fz_share",
    "trajectory_elapsed_s",
    "cycle_index",
    "cycle_role",
    "phase",
    "target_force_N",
    "target_ramp_N_per_s",
    "tracking_error_N",
)
UNLOADED_FRAME_CSV_COLUMNS = TRAJECTORY_FRAME_CSV_COLUMNS[:20]
_MACHINE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _machine_name(name: str, value: str) -> str:
    normalized = value.strip()
    if not _MACHINE_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(f"{name} must use lowercase letters, numbers, and underscores")
    return normalized


def _tare_dict(offsets: BotaTareOffsets) -> dict[str, float]:
    return {
        "fx_n": offsets.fx_n,
        "fy_n": offsets.fy_n,
        "fz_n": offsets.fz_n,
        "mx_nm": offsets.mx_nm,
        "my_nm": offsets.my_nm,
        "mz_nm": offsets.mz_nm,
    }


@dataclass(frozen=True)
class HistorySessionMetadata:
    """Session identity and fixed hardware settings for history format v1."""

    material: str
    morphology: str
    specimen_id: str
    camera_model: str
    camera_width: int
    camera_height: int
    camera_fps: int
    camera_exposure_us: float
    camera_gain: float
    camera_white_balance_k: float
    bota_serial_port: str
    bota_tare_offsets: BotaTareOffsets
    trajectory: ForceTrajectoryConfig
    sensor_mode: str = "physical"
    camera_serial_number: str | None = None
    bota_model: str = "Bota Rokubi"
    created_utc: str = ""
    git_commit: str | None = None

    def __post_init__(self) -> None:
        for name in ("material", "morphology", "specimen_id"):
            object.__setattr__(self, name, _machine_name(name, getattr(self, name)))
        if not self.camera_model.strip():
            raise ValueError("camera_model must be nonempty")
        for name in ("camera_width", "camera_height", "camera_fps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("camera_exposure_us", "camera_white_balance_k"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        gain = float(self.camera_gain)
        if not math.isfinite(gain) or gain < 0.0:
            raise ValueError("camera_gain must be finite and nonnegative")
        object.__setattr__(self, "camera_gain", gain)
        if not self.bota_serial_port.strip() or not self.bota_model.strip():
            raise ValueError("force-sensor model and serial port must be nonempty")
        if self.sensor_mode not in {"physical", "mock"}:
            raise ValueError("sensor_mode must be 'physical' or 'mock'")
        if not self.created_utc:
            object.__setattr__(self, "created_utc", _utc_now())

    def to_dict(self) -> dict[str, Any]:
        config = self.trajectory
        return {
            "format_version": HISTORY_FORMAT_VERSION,
            "created_utc": self.created_utc,
            "specimen": {
                "material": self.material,
                "morphology": self.morphology,
                "specimen_id": self.specimen_id,
            },
            "camera": {
                "model": self.camera_model,
                "serial_number": self.camera_serial_number,
                "width": self.camera_width,
                "height": self.camera_height,
                "fps": self.camera_fps,
                "auto_exposure": False,
                "exposure_us": self.camera_exposure_us,
                "gain": self.camera_gain,
                "auto_white_balance": False,
                "white_balance_k": self.camera_white_balance_k,
            },
            "force_sensor": {
                "model": self.bota_model,
                "serial_port": self.bota_serial_port,
                "mode": self.sensor_mode,
                "tare_offsets": _tare_dict(self.bota_tare_offsets),
            },
            "trajectory": {
                "min_force_n": config.min_force_n,
                "max_force_n": config.max_force_n,
                "ramp_rate_n_per_s": config.ramp_rate_n_per_s,
                "low_dwell_s": config.low_dwell_s,
                "high_dwell_s": config.high_dwell_s,
                "conditioning_cycles": config.conditioning_cycles,
                "measurement_cycles": config.measurement_cycles,
                "preload_tolerance_n": config.preload_tolerance_n,
                "preload_settle_s": config.preload_settle_s,
                "release_max_force_n": config.release_max_force_n,
                "release_settle_s": config.release_settle_s,
                "capture_rate_hz": config.capture_rate_hz,
            },
            "git_commit": self.git_commit,
        }


@dataclass(frozen=True)
class HistorySynchronizedFrame:
    """One RGB frame paired to the nearest host-time force sample."""

    rgb: np.ndarray
    camera_host_time_s: float
    camera_device_timestamp_ms: float
    camera_frame_number: int
    bota_sample: BotaSample

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must be an HxWx3 uint8 array")
        if not rgb.flags.c_contiguous:
            rgb = np.ascontiguousarray(rgb)
        if rgb.flags.writeable:
            rgb = rgb.copy()
            rgb.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)
        if not math.isfinite(self.camera_host_time_s):
            raise ValueError("camera_host_time_s must be finite")
        if not math.isfinite(self.camera_device_timestamp_ms):
            raise ValueError("camera_device_timestamp_ms must be finite")
        if self.camera_frame_number < 0:
            raise ValueError("camera_frame_number must be nonnegative")


@dataclass(frozen=True)
class HistoryRunMetadata:
    run_id: str
    indenter: str
    hole_index: int
    contact_position_mm: float
    repetition_index: int
    started_utc: str
    ended_utc: str | None = None
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "indenter": self.indenter,
            "hole_index": self.hole_index,
            "contact_position_mm": self.contact_position_mm,
            "repetition_index": self.repetition_index,
            "started_utc": self.started_utc,
            "ended_utc": self.ended_utc,
            "status": self.status,
        }


@dataclass(frozen=True)
class HistoryRunHandle:
    run_id: str
    partial_path: Path
    final_path: Path
    metadata: HistoryRunMetadata


@dataclass(frozen=True)
class HistoryUnloadedHandle:
    key: str
    partial_path: Path
    final_path: Path
    expected_frame_count: int


@dataclass(frozen=True)
class _Task:
    action: Callable[[], None]
    frame_slot: bool = False


class HistoryDatasetWriter:
    """Write one history-format session with atomic runs and bounded PNG work."""

    def __init__(
        self,
        output_root: str | Path,
        session_metadata: HistorySessionMetadata,
        *,
        frame_queue_capacity: int = 256,
        png_compression: int = 1,
    ) -> None:
        if frame_queue_capacity < 1:
            raise ValueError("frame_queue_capacity must be positive")
        if not 0 <= png_compression <= 9:
            raise ValueError("png_compression must be in [0, 9]")
        base = Path(output_root)
        if session_metadata.sensor_mode == "mock":
            base = base / "mock"
        date = datetime.now().strftime("%Y-%m-%d")
        name = f"{date}_{session_metadata.material}_{session_metadata.morphology}"
        self.session_path = self._unique_path(base, name)
        self.session_path.mkdir(parents=True)
        (self.session_path / "runs").mkdir()
        (self.session_path / "unloaded").mkdir()
        self._session = session_metadata
        self._session_dict = session_metadata.to_dict()
        if self._session_dict["git_commit"] is None:
            self._session_dict["git_commit"] = _git_commit()
        _write_json(self.session_path / "session.json", self._session_dict)

        self._png_compression = png_compression
        self._tasks: queue.SimpleQueue[_Task | None] = queue.SimpleQueue()
        self._frame_slots = threading.BoundedSemaphore(frame_queue_capacity)
        self._lock = threading.Lock()
        self._rows: dict[str, list[dict[str, Any]]] = {}
        self._frame_indices: dict[str, int] = {}
        self._writer_drops: dict[str, int] = {}
        self._failed_keys: set[str] = set()
        self._run_count = 0
        self._unloaded_count = 0
        self._repetition_counts: dict[tuple[str, int], int] = {}
        self._dropped_frame_count = 0
        self._pending_tasks = 0
        self._first_error: BaseException | None = None
        self._closed = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="contact_history_writer",
            daemon=True,
        )
        self._worker.start()

    @staticmethod
    def _unique_path(parent: Path, name: str) -> Path:
        candidate = parent / name
        suffix = 1
        while candidate.exists():
            candidate = parent / f"{name}_{suffix:02d}"
            suffix += 1
        return candidate

    @property
    def dropped_frame_count(self) -> int:
        with self._lock:
            return self._dropped_frame_count

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._first_error

    def update_tare_offsets(self, offsets: BotaTareOffsets) -> None:
        self._ensure_open()

        def update() -> None:
            sensor = self._session_dict["force_sensor"]
            assert isinstance(sensor, dict)
            sensor["tare_offsets"] = _tare_dict(offsets)
            _write_json(self.session_path / "session.json", self._session_dict)

        self._enqueue(_Task(update))

    def start_run(self, *, indenter: str, hole_index: int) -> HistoryRunHandle:
        self._ensure_open()
        indenter = _machine_name("indenter", indenter)
        if not isinstance(hole_index, int) or hole_index not in range(1, 7):
            raise ValueError("hole_index must be an integer from 1 through 6")
        condition = (indenter, hole_index)
        with self._lock:
            repetition = self._repetition_counts.get(condition, 0) + 1
            self._repetition_counts[condition] = repetition
            self._run_count += 1
            run_id = f"run_{self._run_count:04d}"
            self._rows[run_id] = []
            self._frame_indices[run_id] = 0
            self._writer_drops[run_id] = 0
        final_path = self.session_path / "runs" / run_id
        partial_path = final_path.with_name(f"{run_id}.partial")
        if partial_path.exists() or final_path.exists():
            raise FileExistsError(f"run path already exists for {run_id}")
        (partial_path / "trajectory" / "frames").mkdir(parents=True)
        metadata = HistoryRunMetadata(
            run_id=run_id,
            indenter=indenter,
            hole_index=hole_index,
            contact_position_mm=HOLE_CONTACT_POSITIONS_MM[hole_index - 1],
            repetition_index=repetition,
            started_utc=_utc_now(),
        )
        _write_json(partial_path / "run.json", metadata.to_dict())
        return HistoryRunHandle(run_id, partial_path, final_path, metadata)

    def submit_trajectory_frame(
        self,
        run: HistoryRunHandle,
        frame: HistorySynchronizedFrame,
        trajectory: ForceTrajectoryUpdate,
    ) -> bool:
        self._ensure_open()
        if trajectory.state is not ForceTrajectoryState.CYCLING:
            raise ValueError("trajectory frames may only be saved while CYCLING")
        if (
            trajectory.phase is None
            or trajectory.cycle_index is None
            or trajectory.cycle_role is None
            or trajectory.target_force_n is None
            or trajectory.tracking_error_n is None
        ):
            raise ValueError("trajectory update is missing active-cycle metadata")
        if not self._frame_slots.acquire(blocking=False):
            with self._lock:
                self._dropped_frame_count += 1
                self._writer_drops[run.run_id] += 1
            return False
        with self._lock:
            frame_index = self._frame_indices[run.run_id]
            self._frame_indices[run.run_id] = frame_index + 1
        filename = f"frame_{frame_index + 1:06d}.png"
        row = {
            **self._raw_frame_row(
                frame_index, filename, trajectory.trajectory_elapsed_s, frame
            ),
            "trajectory_elapsed_s": trajectory.trajectory_elapsed_s,
            "cycle_index": trajectory.cycle_index,
            "cycle_role": trajectory.cycle_role,
            "phase": trajectory.phase.value,
            "target_force_N": trajectory.target_force_n,
            "target_ramp_N_per_s": trajectory.target_ramp_n_per_s,
            "tracking_error_N": trajectory.tracking_error_n,
        }
        output = run.partial_path / "trajectory" / "frames" / filename
        self._queue_png(run.run_id, output, row, frame)
        return True

    def complete_run(
        self,
        run: HistoryRunHandle,
        *,
        trajectory_start_host_time_s: float,
        trajectory_end_host_time_s: float,
        dropped_camera_frame_count: int,
        missed_capture_deadline_count: int,
    ) -> None:
        self._ensure_open()
        start = float(trajectory_start_host_time_s)
        end = float(trajectory_end_host_time_s)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("trajectory host times must be finite and increasing")
        for name, value in (
            ("dropped_camera_frame_count", dropped_camera_frame_count),
            ("missed_capture_deadline_count", missed_capture_deadline_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        with self._lock:
            writer_drops = self._writer_drops[run.run_id]

        def complete() -> None:
            if run.run_id in self._failed_keys:
                raise RuntimeError(f"cannot finalize failed run {run.run_id}")
            rows = self._rows.pop(run.run_id)
            trajectory_path = run.partial_path / "trajectory"
            self._write_csv(
                trajectory_path / "frames.csv",
                TRAJECTORY_FRAME_CSV_COLUMNS,
                rows,
            )
            actual = [float(row["force_magnitude_N"]) for row in rows]
            error = [float(row["tracking_error_N"]) for row in rows]
            config = self._session.trajectory
            _write_json(
                trajectory_path / "trajectory.json",
                {
                    "total_conditioning_cycles": config.conditioning_cycles,
                    "total_measurement_cycles": config.measurement_cycles,
                    "total_expected_cycles": config.total_cycles,
                    "nominal_cycle_duration_s": config.nominal_cycle_duration_s,
                    "min_force_n": config.min_force_n,
                    "max_force_n": config.max_force_n,
                    "ramp_rate_n_per_s": config.ramp_rate_n_per_s,
                    "high_dwell_s": config.high_dwell_s,
                    "low_dwell_s": config.low_dwell_s,
                    "trajectory_start_host_time_s": start,
                    "trajectory_end_host_time_s": end,
                    "saved_frame_count": len(rows),
                    "dropped_camera_frame_count": dropped_camera_frame_count,
                    "dropped_writer_frame_count": writer_drops,
                    "missed_capture_deadline_count": missed_capture_deadline_count,
                    "observed_min_actual_force_n": min(actual) if actual else None,
                    "observed_max_actual_force_n": max(actual) if actual else None,
                    "observed_rms_tracking_error_n": (
                        math.sqrt(sum(value * value for value in error) / len(error))
                        if error
                        else None
                    ),
                },
            )
            completed = HistoryRunMetadata(
                run_id=run.metadata.run_id,
                indenter=run.metadata.indenter,
                hole_index=run.metadata.hole_index,
                contact_position_mm=run.metadata.contact_position_mm,
                repetition_index=run.metadata.repetition_index,
                started_utc=run.metadata.started_utc,
                ended_utc=_utc_now(),
                status="complete",
            )
            _write_json(run.partial_path / "run.json", completed.to_dict())
            run.partial_path.replace(run.final_path)
            with self._lock:
                self._frame_indices.pop(run.run_id, None)
                self._writer_drops.pop(run.run_id, None)

        self._enqueue(_Task(complete))

    def abort_run(self, run: HistoryRunHandle) -> None:
        self._ensure_open()
        condition = (run.metadata.indenter, run.metadata.hole_index)
        with self._lock:
            if self._repetition_counts.get(condition) == run.metadata.repetition_index:
                if run.metadata.repetition_index == 1:
                    self._repetition_counts.pop(condition)
                else:
                    self._repetition_counts[condition] = (
                        run.metadata.repetition_index - 1
                    )

        def abort() -> None:
            self._rows.pop(run.run_id, None)
            self._failed_keys.discard(run.run_id)
            with self._lock:
                self._frame_indices.pop(run.run_id, None)
                self._writer_drops.pop(run.run_id, None)
            if run.partial_path.exists():
                shutil.rmtree(run.partial_path)

        self._enqueue(_Task(abort))

    def begin_unloaded_capture(
        self, expected_frame_count: int
    ) -> HistoryUnloadedHandle:
        self._ensure_open()
        if expected_frame_count < 1:
            raise ValueError("expected_frame_count must be positive")
        with self._lock:
            self._unloaded_count += 1
            capture_id = f"capture_{self._unloaded_count:03d}"
            key = f"unloaded:{capture_id}"
            self._rows[key] = []
            self._frame_indices[key] = 0
            self._writer_drops[key] = 0
        final_path = self.session_path / "unloaded" / capture_id
        partial_path = final_path.with_name(f"{capture_id}.partial")
        (partial_path / "frames").mkdir(parents=True)
        return HistoryUnloadedHandle(
            key, partial_path, final_path, expected_frame_count
        )

    def submit_unloaded_frame(
        self,
        capture: HistoryUnloadedHandle,
        frame: HistorySynchronizedFrame,
        *,
        capture_elapsed_s: float,
    ) -> bool:
        self._ensure_open()
        elapsed = float(capture_elapsed_s)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("capture_elapsed_s must be finite and nonnegative")
        if not self._frame_slots.acquire(blocking=False):
            with self._lock:
                self._dropped_frame_count += 1
                self._writer_drops[capture.key] += 1
            return False
        with self._lock:
            frame_index = self._frame_indices[capture.key]
            if frame_index >= capture.expected_frame_count:
                self._frame_slots.release()
                raise RuntimeError("unloaded capture already has its expected frames")
            self._frame_indices[capture.key] = frame_index + 1
        filename = f"frame_{frame_index + 1:06d}.png"
        row = self._raw_frame_row(frame_index, filename, elapsed, frame)
        self._queue_png(
            capture.key,
            capture.partial_path / "frames" / filename,
            row,
            frame,
        )
        return True

    def finalize_unloaded_capture(self, capture: HistoryUnloadedHandle) -> None:
        self._ensure_open()
        with self._lock:
            submitted = self._frame_indices[capture.key]
        if submitted != capture.expected_frame_count:
            raise RuntimeError(
                f"cannot finalize {capture.key}: expected {capture.expected_frame_count} "
                f"frames, received {submitted}"
            )

        def finalize() -> None:
            if capture.key in self._failed_keys:
                raise RuntimeError(f"cannot finalize failed capture {capture.key}")
            rows = self._rows.pop(capture.key)
            if len(rows) != capture.expected_frame_count:
                raise RuntimeError(
                    f"cannot finalize {capture.key}: expected "
                    f"{capture.expected_frame_count} written frames, received {len(rows)}"
                )
            self._write_csv(
                capture.partial_path / "frames.csv",
                UNLOADED_FRAME_CSV_COLUMNS,
                rows,
            )
            capture.partial_path.replace(capture.final_path)
            with self._lock:
                self._frame_indices.pop(capture.key, None)
                self._writer_drops.pop(capture.key, None)

        self._enqueue(_Task(finalize))

    def discard_unloaded_capture(self, capture: HistoryUnloadedHandle) -> None:
        self._ensure_open()

        def discard() -> None:
            self._rows.pop(capture.key, None)
            self._failed_keys.discard(capture.key)
            with self._lock:
                self._frame_indices.pop(capture.key, None)
                self._writer_drops.pop(capture.key, None)
            if capture.partial_path.exists():
                shutil.rmtree(capture.partial_path)

        self._enqueue(_Task(discard))

    @staticmethod
    def _raw_frame_row(
        frame_index: int,
        filename: str,
        capture_elapsed_s: float,
        frame: HistorySynchronizedFrame,
    ) -> dict[str, Any]:
        sample = frame.bota_sample
        return {
            "frame_index": frame_index,
            "rgb_filename": f"frames/{filename}",
            "capture_elapsed_s": capture_elapsed_s,
            "camera_host_time_s": frame.camera_host_time_s,
            "camera_device_timestamp_ms": frame.camera_device_timestamp_ms,
            "camera_frame_number": frame.camera_frame_number,
            "bota_host_time_s": sample.host_time_s,
            "bota_sensor_timestamp": sample.sensor_timestamp,
            "camera_bota_time_delta_ms": (frame.camera_host_time_s - sample.host_time_s)
            * 1000.0,
            "Fx_N": sample.fx_n,
            "Fy_N": sample.fy_n,
            "Fz_N": sample.fz_n,
            "Mx_Nm": sample.mx_nm,
            "My_Nm": sample.my_nm,
            "Mz_Nm": sample.mz_nm,
            "temperature_C": sample.temperature_c,
            "bota_status": sample.status,
            "force_magnitude_N": sample.force_magnitude_n,
            "torque_magnitude_Nm": sample.torque_magnitude_nm,
            "fz_share": sample.fz_share,
        }

    def _queue_png(
        self,
        key: str,
        output: Path,
        row: dict[str, Any],
        frame: HistorySynchronizedFrame,
    ) -> None:
        def write() -> None:
            try:
                import cv2

                bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                ok = cv2.imwrite(
                    str(output),
                    bgr,
                    (cv2.IMWRITE_PNG_COMPRESSION, self._png_compression),
                )
                if not ok:
                    raise OSError(f"OpenCV could not write {output}")
                self._rows[key].append(row)
            except BaseException:
                self._failed_keys.add(key)
                raise

        self._enqueue(_Task(write, frame_slot=True))

    @staticmethod
    def _write_csv(
        path: Path,
        columns: tuple[str, ...],
        rows: list[dict[str, Any]],
    ) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def flush(self, timeout_s: float = 30.0) -> None:
        deadline = monotonic() + timeout_s
        while True:
            with self._lock:
                pending = self._pending_tasks
                error = self._first_error
            if pending == 0:
                if error is not None:
                    raise RuntimeError("contact history writer failed") from error
                return
            if monotonic() >= deadline:
                raise TimeoutError("timed out waiting for contact history writer")
            sleep(0.005)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._closed = True
            self._tasks.put(None)
            self._worker.join(timeout=10.0)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("contact history writer is closed")
        error = self.error
        if error is not None:
            raise RuntimeError("contact history writer failed") from error

    def _enqueue(self, task: _Task) -> None:
        with self._lock:
            self._pending_tasks += 1
        self._tasks.put(task)

    def _worker_loop(self) -> None:
        while True:
            task = self._tasks.get()
            if task is None:
                return
            try:
                task.action()
            except BaseException as error:
                with self._lock:
                    if self._first_error is None:
                        self._first_error = error
            finally:
                if task.frame_slot:
                    self._frame_slots.release()
                with self._lock:
                    self._pending_tasks -= 1

    def __enter__(self) -> HistoryDatasetWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = [
    "HISTORY_FORMAT_VERSION",
    "HOLE_CONTACT_POSITIONS_MM",
    "HistoryDatasetWriter",
    "HistoryRunHandle",
    "HistorySessionMetadata",
    "HistorySynchronizedFrame",
    "HistoryUnloadedHandle",
    "TRAJECTORY_FRAME_CSV_COLUMNS",
    "UNLOADED_FRAME_CSV_COLUMNS",
]
