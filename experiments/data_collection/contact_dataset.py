"""Format-v3 storage and reading for raw physical contact data."""

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
from typing import Any, Callable, Iterator, Mapping

import numpy as np

from experiments.hardware import BotaSample, BotaTareOffsets

from .force_sequence import ForceSequenceConfig


FORMAT_VERSION = 3
FRAME_CSV_COLUMNS = (
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
)
_FORCE_DIRECTORY_PATTERN = re.compile(r"^force_(\d+(?:\.\d+)?)N$")
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


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


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


def _validate_machine_name(name: str, value: str) -> str:
    normalized = value.strip()
    if not _MACHINE_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(f"{name} must use lowercase letters, numbers, and underscores")
    return normalized


def _tare_offsets_dict(offsets: BotaTareOffsets) -> dict[str, float]:
    return {
        "fx_n": offsets.fx_n,
        "fy_n": offsets.fy_n,
        "fz_n": offsets.fz_n,
        "mx_nm": offsets.mx_nm,
        "my_nm": offsets.my_nm,
        "mz_nm": offsets.mz_nm,
    }


def format_force_directory(target_force_n: float) -> str:
    """Return the one canonical directory name for an experimental force target."""

    target = float(target_force_n)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_force_n must be finite and positive")
    return f"force_{target:02g}N"


def parse_force_directory(name: str) -> float:
    """Parse and validate a canonical ``force_*N`` directory name."""

    match = _FORCE_DIRECTORY_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid force directory name: {name}")
    target = float(match.group(1))
    if format_force_directory(target) != name:
        raise ValueError(f"noncanonical force directory name: {name}")
    return target


@dataclass(frozen=True)
class SessionMetadata:
    """Session-wide format-v3 identity and acquisition configuration."""

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
    force_sequence: ForceSequenceConfig
    sensor_mode: str = "physical"
    camera_serial_number: str | None = None
    bota_model: str = "Bota Rokubi"
    created_utc: str = ""
    git_commit: str | None = None

    def __post_init__(self) -> None:
        for name in ("material", "morphology", "specimen_id"):
            object.__setattr__(self, name, _validate_machine_name(name, getattr(self, name)))
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
        if not self.bota_serial_port.strip():
            raise ValueError("bota_serial_port must be nonempty")
        if not self.bota_model.strip():
            raise ValueError("bota_model must be nonempty")
        if self.sensor_mode not in {"physical", "mock"}:
            raise ValueError("sensor_mode must be 'physical' or 'mock'")
        if not self.created_utc:
            object.__setattr__(self, "created_utc", _utc_now())

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit persisted v2 session schema."""

        config = self.force_sequence
        return {
            "format_version": FORMAT_VERSION,
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
                "tare_offsets": _tare_offsets_dict(self.bota_tare_offsets),
            },
            "acquisition": {
                "target_forces_n": list(config.target_forces_n),
                "settle_duration_s": config.settle_duration_s,
                "record_duration_s": config.record_duration_s,
                "capture_rate_hz": config.capture_rate_hz,
                "minimum_tolerance_n": config.minimum_tolerance_n,
                "low_force_relative_tolerance": config.low_force_relative_tolerance,
                "high_force_relative_tolerance": config.high_force_relative_tolerance,
                "high_force_threshold_n": config.high_force_threshold_n,
                "unloaded_max_force_n": config.unloaded_max_force_n,
                "unloaded_settle_duration_s": config.unloaded_settle_duration_s,
                "unloaded_record_duration_s": config.unloaded_record_duration_s,
            },
            "git_commit": self.git_commit,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SessionMetadata:
        """Load only the explicit format-v3 session schema."""

        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"session format_version must be {FORMAT_VERSION}")
        specimen = data["specimen"]
        camera = data["camera"]
        sensor = data["force_sensor"]
        acquisition = data["acquisition"]
        tare = sensor["tare_offsets"]
        if camera.get("auto_exposure") is not False:
            raise ValueError("session camera auto_exposure must be false")
        if camera.get("auto_white_balance") is not False:
            raise ValueError("session camera auto_white_balance must be false")
        return cls(
            material=str(specimen["material"]),
            morphology=str(specimen["morphology"]),
            specimen_id=str(specimen["specimen_id"]),
            camera_model=str(camera["model"]),
            camera_serial_number=camera["serial_number"],
            camera_width=int(camera["width"]),
            camera_height=int(camera["height"]),
            camera_fps=int(camera["fps"]),
            camera_exposure_us=float(camera["exposure_us"]),
            camera_gain=float(camera["gain"]),
            camera_white_balance_k=float(camera["white_balance_k"]),
            bota_serial_port=str(sensor["serial_port"]),
            bota_tare_offsets=BotaTareOffsets(
                fx_n=float(tare["fx_n"]),
                fy_n=float(tare["fy_n"]),
                fz_n=float(tare["fz_n"]),
                mx_nm=float(tare["mx_nm"]),
                my_nm=float(tare["my_nm"]),
                mz_nm=float(tare["mz_nm"]),
            ),
            force_sequence=ForceSequenceConfig(
                target_forces_n=tuple(float(value) for value in acquisition["target_forces_n"]),
                settle_duration_s=float(acquisition["settle_duration_s"]),
                record_duration_s=float(acquisition["record_duration_s"]),
                capture_rate_hz=float(acquisition["capture_rate_hz"]),
                unloaded_max_force_n=float(acquisition["unloaded_max_force_n"]),
                unloaded_settle_duration_s=float(acquisition["unloaded_settle_duration_s"]),
                unloaded_record_duration_s=float(acquisition["unloaded_record_duration_s"]),
                minimum_tolerance_n=float(acquisition["minimum_tolerance_n"]),
                low_force_relative_tolerance=float(
                    acquisition["low_force_relative_tolerance"]
                ),
                high_force_relative_tolerance=float(
                    acquisition["high_force_relative_tolerance"]
                ),
                high_force_threshold_n=float(acquisition["high_force_threshold_n"]),
            ),
            sensor_mode=str(sensor["mode"]),
            bota_model=str(sensor["model"]),
            created_utc=str(data["created_utc"]),
            git_commit=data.get("git_commit"),
        )


@dataclass(frozen=True)
class RunMetadata:
    """Identity and lifecycle state for one independent contact trial."""

    run_id: str
    indenter: str
    hole_index: int
    repetition_index: int
    started_utc: str
    ended_utc: str | None = None
    status: str = "active"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"run_\d{4}", self.run_id):
            raise ValueError("run_id must use run_NNNN")
        object.__setattr__(self, "indenter", _validate_machine_name("indenter", self.indenter))
        if not isinstance(self.hole_index, int) or self.hole_index not in range(1, 7):
            raise ValueError("hole_index must be an integer from 1 through 6")
        if not isinstance(self.repetition_index, int) or self.repetition_index < 1:
            raise ValueError("repetition_index must be a positive integer")
        if self.status not in {"active", "complete", "aborted"}:
            raise ValueError("status must be active, complete, or aborted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "indenter": self.indenter,
            "hole_index": self.hole_index,
            "repetition_index": self.repetition_index,
            "started_utc": self.started_utc,
            "ended_utc": self.ended_utc,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunMetadata:
        return cls(
            run_id=str(data["run_id"]),
            indenter=str(data["indenter"]),
            hole_index=int(data["hole_index"]),
            repetition_index=int(data["repetition_index"]),
            started_utc=str(data["started_utc"]),
            ended_utc=None if data["ended_utc"] is None else str(data["ended_utc"]),
            status=str(data["status"]),
        )


@dataclass(frozen=True)
class SynchronizedFrame:
    """One untouched RGB frame paired to the nearest host-time Rokubi sample."""

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
        for name in ("camera_host_time_s", "camera_device_timestamp_ms"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.camera_frame_number < 0:
            raise ValueError("camera_frame_number must be nonnegative")


@dataclass(frozen=True)
class LoadedRunHandle:
    run_id: str
    path: Path
    metadata: RunMetadata


@dataclass(frozen=True)
class SegmentHandle:
    key: str
    partial_path: Path
    final_path: Path
    target_force_n: float | None
    expected_frame_count: int


@dataclass(frozen=True)
class DatasetFrameRecord:
    """One frame with session, run, and force context already resolved."""

    session: SessionMetadata
    run: RunMetadata | None
    target_force_n: float | None
    segment_path: Path
    rgb_path: Path
    measurements: Mapping[str, str]


@dataclass(frozen=True)
class _Task:
    action: Callable[[], None]
    frame_slot: bool = False


class ContactDatasetWriter:
    """Write one specimen-scoped format-v3 session with atomic segments."""

    def __init__(
        self,
        output_root: str | Path,
        session_metadata: SessionMetadata,
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
        self._session_metadata = session_metadata.to_dict()
        if self._session_metadata["git_commit"] is None:
            self._session_metadata["git_commit"] = _git_commit()
        _write_json(self.session_path / "session.json", self._session_metadata)

        self._png_compression = png_compression
        self._tasks: queue.SimpleQueue[_Task | None] = queue.SimpleQueue()
        self._frame_slots = threading.BoundedSemaphore(frame_queue_capacity)
        self._lock = threading.Lock()
        self._frame_indices: dict[str, int] = {}
        self._rows: dict[str, list[dict[str, Any]]] = {}
        self._failed_segments: set[str] = set()
        self._run_count = 0
        self._unloaded_count = 0
        self._attempt_counts: dict[str, int] = {}
        self._repetition_counts: dict[tuple[str, int], int] = {}
        self._dropped_frame_count = 0
        self._pending_tasks = 0
        self._first_error: BaseException | None = None
        self._closed = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="contact_dataset_writer",
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
        """Persist a manual tare once in session-level sensor metadata."""

        self._ensure_open()

        def update() -> None:
            force_sensor = self._session_metadata["force_sensor"]
            assert isinstance(force_sensor, dict)
            force_sensor["tare_offsets"] = _tare_offsets_dict(offsets)
            _write_json(self.session_path / "session.json", self._session_metadata)

        self._enqueue_control(update)

    def start_loaded_run(
        self,
        *,
        indenter: str,
        hole_index: int,
    ) -> LoadedRunHandle:
        """Create one independent run; specimen and acquisition stay session-owned."""

        self._ensure_open()
        indenter = _validate_machine_name("indenter", indenter)
        if not isinstance(hole_index, int) or hole_index not in range(1, 7):
            raise ValueError("hole_index must be an integer from 1 through 6")
        condition = (indenter, hole_index)
        with self._lock:
            repetition_index = self._repetition_counts.get(condition, 0) + 1
            self._repetition_counts[condition] = repetition_index
            self._run_count += 1
            run_id = f"run_{self._run_count:04d}"
        metadata = RunMetadata(
            run_id=run_id,
            indenter=indenter,
            hole_index=hole_index,
            repetition_index=repetition_index,
            started_utc=_utc_now(),
        )
        path = self.session_path / "runs" / run_id
        path.mkdir(parents=True, exist_ok=False)
        _write_json(path / "run.json", metadata.to_dict())
        return LoadedRunHandle(run_id=run_id, path=path, metadata=metadata)

    def begin_force_target(
        self,
        run: LoadedRunHandle,
        target_force_n: float,
    ) -> SegmentHandle:
        target = float(target_force_n)
        if target not in self._session.force_sequence.target_forces_n:
            raise ValueError("target_force_n is not part of the session acquisition config")
        return self._begin_segment(
            base_key=f"{run.run_id}:force:{target:g}",
            final_path=run.path / format_force_directory(target),
            target_force_n=target,
            expected_frame_count=(
                self._session.force_sequence.expected_record_frame_count
            ),
        )

    def begin_unloaded_capture(self) -> SegmentHandle:
        with self._lock:
            self._unloaded_count += 1
            capture_id = f"capture_{self._unloaded_count:03d}"
        return self._begin_segment(
            base_key=f"unloaded:{capture_id}",
            final_path=self.session_path / "unloaded" / capture_id,
            target_force_n=None,
            expected_frame_count=(
                self._session.force_sequence.expected_unloaded_frame_count
            ),
        )

    def _begin_segment(
        self,
        *,
        base_key: str,
        final_path: Path,
        target_force_n: float | None,
        expected_frame_count: int,
    ) -> SegmentHandle:
        self._ensure_open()
        if final_path.exists():
            raise FileExistsError(f"completed segment already exists: {final_path}")
        with self._lock:
            attempt = self._attempt_counts.get(base_key, 0) + 1
            self._attempt_counts[base_key] = attempt
        key = f"{base_key}:attempt:{attempt}"
        partial = final_path.parent / f"{final_path.name}.attempt_{attempt:03d}.partial"
        handle = SegmentHandle(
            key=key,
            partial_path=partial,
            final_path=final_path,
            target_force_n=target_force_n,
            expected_frame_count=expected_frame_count,
        )

        def begin() -> None:
            if partial.exists():
                shutil.rmtree(partial)
            (partial / "frames").mkdir(parents=True)
            self._rows[key] = []

        with self._lock:
            self._frame_indices[key] = 0
        self._enqueue_control(begin)
        return handle

    def submit_frame(
        self,
        segment: SegmentHandle,
        frame: SynchronizedFrame,
        *,
        capture_elapsed_s: float = 0.0,
    ) -> bool:
        """Queue one frame, returning ``False`` rather than silently dropping it."""

        self._ensure_open()
        elapsed = float(capture_elapsed_s)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("capture_elapsed_s must be finite and nonnegative")
        if not self._frame_slots.acquire(blocking=False):
            with self._lock:
                self._dropped_frame_count += 1
            return False
        with self._lock:
            frame_index = self._frame_indices[segment.key]
            if frame_index >= segment.expected_frame_count:
                self._frame_slots.release()
                raise RuntimeError(
                    f"segment {segment.key} already has its expected "
                    f"{segment.expected_frame_count} frames"
                )
            self._frame_indices[segment.key] = frame_index + 1
        filename = f"{frame_index:06d}.png"
        row = self._frame_row(frame_index, filename, elapsed, frame)

        def write_frame() -> None:
            try:
                import cv2

                output = segment.partial_path / "frames" / filename
                bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                ok = cv2.imwrite(
                    str(output),
                    bgr,
                    (cv2.IMWRITE_PNG_COMPRESSION, self._png_compression),
                )
                if not ok:
                    raise OSError(f"OpenCV could not write {output}")
                self._rows[segment.key].append(row)
            except BaseException:
                self._failed_segments.add(segment.key)
                raise

        self._enqueue(_Task(write_frame, frame_slot=True))
        return True

    @staticmethod
    def _frame_row(
        frame_index: int,
        filename: str,
        capture_elapsed_s: float,
        frame: SynchronizedFrame,
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
            "camera_bota_time_delta_ms": (
                frame.camera_host_time_s - sample.host_time_s
            )
            * 1000.0,
            "Fx_N": sample.fx_n,
            "Fy_N": sample.fy_n,
            "Fz_N": sample.fz_n,
            "Mx_Nm": sample.mx_nm,
            "My_Nm": sample.my_nm,
            "Mz_Nm": sample.mz_nm,
            "temperature_C": sample.temperature_c,
            "bota_status": sample.status,
        }

    def finalize_segment(self, segment: SegmentHandle) -> None:
        """Publish a successful segment containing only frames and frames.csv."""

        self._ensure_open()
        with self._lock:
            submitted_count = self._frame_indices.get(segment.key, 0)
        if submitted_count != segment.expected_frame_count:
            raise RuntimeError(
                f"cannot finalize {segment.key}: expected "
                f"{segment.expected_frame_count} frames, received {submitted_count}"
            )

        def finalize() -> None:
            rows = self._rows.pop(segment.key, [])
            if segment.key in self._failed_segments:
                return
            if len(rows) != segment.expected_frame_count:
                self._failed_segments.add(segment.key)
                raise RuntimeError(
                    f"cannot finalize {segment.key}: expected "
                    f"{segment.expected_frame_count} written frames, received {len(rows)}"
                )
            self._write_frames_csv(segment.partial_path / "frames.csv", rows)
            if segment.final_path.exists():
                raise FileExistsError(f"completed segment already exists: {segment.final_path}")
            segment.partial_path.replace(segment.final_path)
            with self._lock:
                self._frame_indices.pop(segment.key, None)

        self._enqueue_control(finalize)

    def discard_segment(self, segment: SegmentHandle) -> None:
        """Discard one incomplete attempt without touching completed target data."""

        self._ensure_open()

        def discard() -> None:
            self._rows.pop(segment.key, None)
            self._failed_segments.discard(segment.key)
            with self._lock:
                self._frame_indices.pop(segment.key, None)
            if segment.partial_path.exists():
                shutil.rmtree(segment.partial_path)

        self._enqueue_control(discard)

    def complete_loaded_run(self, run: LoadedRunHandle) -> None:
        self._set_run_status(run, "complete")

    def abort_loaded_run(self, run: LoadedRunHandle) -> None:
        self._set_run_status(run, "aborted")

    def _set_run_status(self, run: LoadedRunHandle, status: str) -> None:
        self._ensure_open()
        if status not in {"complete", "aborted"}:
            raise ValueError("invalid run status")

        def finish() -> None:
            if status == "complete":
                missing = [
                    target
                    for target in self._session.force_sequence.target_forces_n
                    if not (run.path / format_force_directory(target)).is_dir()
                ]
                if missing:
                    raise RuntimeError(
                        f"cannot complete {run.run_id}: missing targets are {missing}"
                    )
            else:
                for partial in run.path.glob("*.partial"):
                    shutil.rmtree(partial)
            metadata = RunMetadata(
                run_id=run.metadata.run_id,
                indenter=run.metadata.indenter,
                hole_index=run.metadata.hole_index,
                repetition_index=run.metadata.repetition_index,
                started_utc=run.metadata.started_utc,
                ended_utc=_utc_now(),
                status=status,
            )
            _write_json(run.path / "run.json", metadata.to_dict())

        self._enqueue_control(finish)

    @staticmethod
    def _write_frames_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FRAME_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def flush(self, timeout_s: float = 10.0) -> None:
        """Wait for queued work; primarily useful at shutdown and in tests."""

        deadline = monotonic() + timeout_s
        while True:
            with self._lock:
                pending = self._pending_tasks
                error = self._first_error
            if pending == 0:
                if error is not None:
                    raise RuntimeError("contact dataset writer failed") from error
                return
            if monotonic() >= deadline:
                raise TimeoutError("timed out waiting for contact dataset writer")
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
            raise RuntimeError("contact dataset writer is closed")
        error = self.error
        if error is not None:
            raise RuntimeError("contact dataset writer failed") from error

    def _enqueue_control(self, action: Callable[[], None]) -> None:
        self._enqueue(_Task(action))

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

    def __enter__(self) -> ContactDatasetWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _segment_rows(segment_path: Path) -> Iterator[dict[str, str]]:
    csv_path = segment_path / "frames.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FRAME_CSV_COLUMNS:
            raise ValueError(f"{csv_path} does not use the format-v3 frame schema")
        yield from reader


def iter_dataset_frames(session_path: str | Path) -> Iterator[DatasetFrameRecord]:
    """Yield all finalized v2 frames with inherited context already resolved."""

    root = Path(session_path)
    session = SessionMetadata.from_dict(_read_json(root / "session.json"))

    for capture_path in sorted((root / "unloaded").glob("capture_*")):
        if not capture_path.is_dir() or not (capture_path / "frames.csv").is_file():
            continue
        for row in _segment_rows(capture_path):
            yield DatasetFrameRecord(
                session=session,
                run=None,
                target_force_n=None,
                segment_path=capture_path,
                rgb_path=capture_path / row["rgb_filename"],
                measurements=row,
            )

    for run_path in sorted((root / "runs").glob("run_*")):
        if not run_path.is_dir() or not (run_path / "run.json").is_file():
            continue
        run = RunMetadata.from_dict(_read_json(run_path / "run.json"))
        for segment_path in sorted(run_path.glob("force_*N")):
            if not segment_path.is_dir() or not (segment_path / "frames.csv").is_file():
                continue
            target = parse_force_directory(segment_path.name)
            for row in _segment_rows(segment_path):
                yield DatasetFrameRecord(
                    session=session,
                    run=run,
                    target_force_n=target,
                    segment_path=segment_path,
                    rgb_path=segment_path / row["rgb_filename"],
                    measurements=row,
                )


__all__ = [
    "ContactDatasetWriter",
    "DatasetFrameRecord",
    "FORMAT_VERSION",
    "FRAME_CSV_COLUMNS",
    "LoadedRunHandle",
    "RunMetadata",
    "SegmentHandle",
    "SessionMetadata",
    "SynchronizedFrame",
    "format_force_directory",
    "iter_dataset_frames",
    "parse_force_directory",
]
