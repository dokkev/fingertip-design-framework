"""Structured, asynchronous storage for raw contact-acquisition data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import queue
import shutil
import subprocess
import threading
from time import monotonic, sleep
from typing import Any, Callable, Iterator, Mapping

import numpy as np

from experiments.hardware import BotaSample, BotaTareOffsets

from .force_sequence import ForceSequenceConfig


HOLE_INDEX_DEFINITION = "1=distal, 6=proximal"

FRAME_CSV_COLUMNS = (
    "frame_index",
    "rgb_filename",
    "target_force_n",
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
    "F_mag_N",
    "torque_mag_Nm",
    "Fz_share",
    "temperature_C",
    "bota_status",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=_json_ready) + "\n",
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


@dataclass(frozen=True)
class SessionMetadata:
    camera_model: str
    camera_width: int
    camera_height: int
    camera_fps: int
    bota_serial_port: str
    bota_tare_offsets: BotaTareOffsets
    force_sequence: ForceSequenceConfig
    sensor_mode: str = "physical"
    camera_serial_number: str | None = None
    bota_model: str = "Rokubi"
    created_utc: str = ""
    git_commit: str | None = None

    def __post_init__(self) -> None:
        if not self.camera_model.strip():
            raise ValueError("camera_model must be nonempty")
        for name in ("camera_width", "camera_height", "camera_fps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.bota_serial_port.strip():
            raise ValueError("bota_serial_port must be nonempty")
        if not self.bota_model.strip():
            raise ValueError("bota_model must be nonempty")
        if self.sensor_mode not in {"physical", "mock"}:
            raise ValueError("sensor_mode must be 'physical' or 'mock'")
        if not self.created_utc:
            object.__setattr__(self, "created_utc", _utc_now())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


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
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class SegmentHandle:
    key: str
    partial_path: Path
    final_path: Path
    target_force_n: float | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CompletedRunRecord:
    path: Path
    metadata: Mapping[str, Any]
    force_segments: tuple[Path, ...]


@dataclass(frozen=True)
class _Task:
    action: Callable[[], None]
    frame_slot: bool = False


class ContactDatasetWriter:
    """Own a session directory and one lossless-PNG writer thread."""

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
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "MOCK_" if session_metadata.sensor_mode == "mock" else ""
        self.session_path = self._unique_path(base, prefix + stamp)
        self.session_path.mkdir(parents=True)
        (self.session_path / "runs").mkdir()
        (self.session_path / "unloaded").mkdir()
        self._session_metadata = session_metadata.to_dict()
        if self._session_metadata.get("git_commit") is None:
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
        """Persist a manual tare performed while no acquisition is active."""

        self._ensure_open()

        def update() -> None:
            self._session_metadata["bota_tare_offsets"] = asdict(offsets)
            _write_json(self.session_path / "session.json", self._session_metadata)

        self._enqueue_control(update)

    def start_loaded_run(
        self,
        *,
        morphology: str,
        indenter_type: str,
        hole_index: int,
        config: ForceSequenceConfig,
        start_host_time_s: float,
        hole_position_mm: float | None = None,
    ) -> LoadedRunHandle:
        self._ensure_open()
        if not morphology.strip() or not indenter_type.strip():
            raise ValueError("morphology and indenter_type must be nonempty")
        if hole_index not in range(1, 7):
            raise ValueError("hole_index must be an integer from 1 through 6")
        if hole_position_mm is not None and not math.isfinite(hole_position_mm):
            raise ValueError("hole_position_mm must be finite when provided")
        with self._lock:
            self._run_count += 1
            run_id = f"run_{self._run_count:04d}"
        path = self.session_path / "runs" / run_id
        metadata: dict[str, Any] = {
            "run_id": run_id,
            "morphology": morphology.strip(),
            "indenter_type": indenter_type.strip(),
            "hole_index": hole_index,
            "hole_index_definition": HOLE_INDEX_DEFINITION,
            "hole_position_mm": hole_position_mm,
            "target_forces_n": list(config.target_forces_n),
            "tolerance_rule": (
                f"max({config.minimum_tolerance_n:g} N, "
                f"{config.relative_tolerance:g} * target_force_n)"
            ),
            "settle_duration_s": config.settle_duration_s,
            "record_duration_s": config.record_duration_s,
            "run_start_host_time_s": float(start_host_time_s),
            "run_end_host_time_s": None,
            "status": "active",
            "completed_targets_n": [],
        }
        path.mkdir(parents=True, exist_ok=False)
        _write_json(path / "metadata.json", metadata)
        return LoadedRunHandle(run_id=run_id, path=path, metadata=metadata)

    def begin_force_target(
        self,
        run: LoadedRunHandle,
        target_force_n: float,
    ) -> SegmentHandle:
        target = float(target_force_n)
        if not math.isfinite(target) or target <= 0:
            raise ValueError("target_force_n must be finite and positive")
        final_path = run.path / f"force_{target:02g}N"
        return self._begin_segment(
            base_key=f"{run.run_id}:force:{target:g}",
            final_path=final_path,
            target_force_n=target,
            metadata={
                "kind": "loaded_force_target",
                "run_id": run.run_id,
                "target_force_n": target,
                "status": "recording",
                "started_utc": _utc_now(),
            },
        )

    def begin_unloaded_capture(
        self,
        *,
        morphology: str,
        indenter_type: str,
        config: ForceSequenceConfig,
    ) -> SegmentHandle:
        if not morphology.strip() or not indenter_type.strip():
            raise ValueError("morphology and indenter_type must be nonempty")
        with self._lock:
            self._unloaded_count += 1
            capture_id = f"capture_{self._unloaded_count:03d}"
        final_path = self.session_path / "unloaded" / capture_id
        return self._begin_segment(
            base_key=f"unloaded:{capture_id}",
            final_path=final_path,
            target_force_n=None,
            metadata={
                "kind": "unloaded",
                "capture_id": capture_id,
                "morphology": morphology.strip(),
                "indenter_type": indenter_type.strip(),
                "hole_index": None,
                "maximum_force_n": config.unloaded_max_force_n,
                "settle_duration_s": config.unloaded_settle_duration_s,
                "record_duration_s": config.unloaded_record_duration_s,
                "status": "recording",
                "started_utc": _utc_now(),
            },
        )

    def _begin_segment(
        self,
        *,
        base_key: str,
        final_path: Path,
        target_force_n: float | None,
        metadata: Mapping[str, Any],
    ) -> SegmentHandle:
        self._ensure_open()
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
            metadata=dict(metadata),
        )

        def begin() -> None:
            if partial.exists():
                shutil.rmtree(partial)
            (partial / "frames").mkdir(parents=True)
            _write_json(partial / "metadata.json", dict(metadata))
            self._rows[key] = []

        with self._lock:
            self._frame_indices[key] = 0
        self._enqueue_control(begin)
        return handle

    def submit_frame(self, segment: SegmentHandle, frame: SynchronizedFrame) -> bool:
        """Queue one frame, returning ``False`` rather than silently dropping it."""

        self._ensure_open()
        if not self._frame_slots.acquire(blocking=False):
            with self._lock:
                self._dropped_frame_count += 1
            return False
        with self._lock:
            frame_index = self._frame_indices[segment.key]
            self._frame_indices[segment.key] = frame_index + 1
        filename = f"{frame_index:06d}.png"
        row = self._frame_row(frame_index, filename, segment.target_force_n, frame)

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
        target_force_n: float | None,
        frame: SynchronizedFrame,
    ) -> dict[str, Any]:
        sample = frame.bota_sample
        return {
            "frame_index": frame_index,
            "rgb_filename": f"frames/{filename}",
            "target_force_n": "" if target_force_n is None else target_force_n,
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
            "F_mag_N": sample.force_magnitude_n,
            "torque_mag_Nm": sample.torque_magnitude_nm,
            "Fz_share": sample.fz_share,
            "temperature_C": sample.temperature_c,
            "bota_status": sample.status,
        }

    def finalize_segment(self, segment: SegmentHandle) -> None:
        """Atomically expose a complete segment after all queued frames are written."""

        self._ensure_open()

        def finalize() -> None:
            rows = self._rows.pop(segment.key, [])
            if segment.key in self._failed_segments:
                return
            if not rows:
                self._failed_segments.add(segment.key)
                raise RuntimeError(f"cannot finalize empty segment {segment.key}")
            self._write_frames_csv(segment.partial_path / "frames.csv", rows)
            summary = self._summary(segment.target_force_n, rows)
            _write_json(segment.partial_path / "summary.json", summary)
            metadata = dict(segment.metadata)
            metadata.update(
                {
                    "status": "complete",
                    "completed_utc": _utc_now(),
                    "frame_count": len(rows),
                }
            )
            _write_json(segment.partial_path / "metadata.json", metadata)
            if segment.final_path.exists():
                raise FileExistsError(f"completed segment already exists: {segment.final_path}")
            segment.partial_path.replace(segment.final_path)

        self._enqueue_control(finalize)

    def discard_segment(self, segment: SegmentHandle) -> None:
        """Discard one incomplete attempt without touching completed target data."""

        self._ensure_open()

        def discard() -> None:
            self._rows.pop(segment.key, None)
            self._failed_segments.discard(segment.key)
            if segment.partial_path.exists():
                shutil.rmtree(segment.partial_path)

        self._enqueue_control(discard)

    def complete_loaded_run(self, run: LoadedRunHandle, end_host_time_s: float) -> None:
        self._set_run_status(run, "complete", end_host_time_s)

    def abort_loaded_run(self, run: LoadedRunHandle, end_host_time_s: float) -> None:
        self._set_run_status(run, "aborted", end_host_time_s)

    def _set_run_status(
        self,
        run: LoadedRunHandle,
        status: str,
        end_host_time_s: float,
    ) -> None:
        self._ensure_open()
        if status not in {"complete", "aborted"}:
            raise ValueError("invalid run status")

        def finish() -> None:
            metadata = dict(run.metadata)
            completed = []
            for target in metadata["target_forces_n"]:
                if (run.path / f"force_{float(target):02g}N").is_dir():
                    completed.append(float(target))
            if status == "complete" and completed != [
                float(target) for target in metadata["target_forces_n"]
            ]:
                raise RuntimeError(
                    f"cannot complete {run.run_id}: completed targets are {completed}"
                )
            metadata.update(
                {
                    "status": status,
                    "run_end_host_time_s": float(end_host_time_s),
                    "completed_targets_n": completed,
                }
            )
            _write_json(run.path / "metadata.json", metadata)

        self._enqueue_control(finish)

    @staticmethod
    def _write_frames_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FRAME_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _summary(
        target_force_n: float | None,
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        forces = np.asarray([row["F_mag_N"] for row in rows], dtype=np.float64)
        fz = np.asarray([row["Fz_N"] for row in rows], dtype=np.float64)
        shares = np.asarray([row["Fz_share"] for row in rows], dtype=np.float64)
        times = np.asarray([row["camera_host_time_s"] for row in rows], dtype=np.float64)
        return {
            "target_force_n": target_force_n,
            "number_of_frames": len(rows),
            "mean_force_magnitude_n": float(np.mean(forces)),
            "std_force_magnitude_n": float(np.std(forces)),
            "min_force_magnitude_n": float(np.min(forces)),
            "max_force_magnitude_n": float(np.max(forces)),
            "mean_fz_n": float(np.mean(fz)),
            "mean_fz_share": float(np.mean(shares)),
            "recording_duration_s": float(times[-1] - times[0]),
        }

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


def iter_completed_runs(
    dataset_root: str | Path,
    *,
    include_aborted: bool = False,
) -> Iterator[CompletedRunRecord]:
    """Yield metadata-backed completed runs, ignoring partial segments."""

    root = Path(dataset_root)
    for metadata_path in sorted(root.rglob("runs/run_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        status = metadata.get("status")
        if status != "complete" and not (include_aborted and status == "aborted"):
            continue
        run_path = metadata_path.parent
        segments = tuple(
            path
            for path in sorted(run_path.glob("force_*N"))
            if path.is_dir()
            and not path.name.endswith(".partial")
            and (path / "metadata.json").is_file()
            and json.loads((path / "metadata.json").read_text(encoding="utf-8")).get(
                "status"
            )
            == "complete"
        )
        yield CompletedRunRecord(run_path, metadata, segments)


__all__ = [
    "CompletedRunRecord",
    "ContactDatasetWriter",
    "FRAME_CSV_COLUMNS",
    "HOLE_INDEX_DEFINITION",
    "LoadedRunHandle",
    "SegmentHandle",
    "SessionMetadata",
    "SynchronizedFrame",
    "iter_completed_runs",
]
