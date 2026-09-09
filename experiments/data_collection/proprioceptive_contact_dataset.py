"""Force-checkpoint acquisition on top of the proprioceptive hardware runtime."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import threading
from time import monotonic, sleep
from typing import Any

from .force_sequence import (
    ForceSequenceConfig,
    ForceSequenceController,
    ForceSequenceState,
    ForceSequenceUpdate,
)
from .proprioceptive_force import CameraAcquisition, ProprioceptiveExperimentRuntime


TARGET_FORCES_N = (2.0, 5.0, 10.0, 15.0, 20.0)
RELEASE_FORCE_THRESHOLD_N = 2.0
LED1_ROTATION_AXIS_DISTANCE_MM = 103.6
LED_SPACING_MM = 11.0
HOLE_SPACING_MM = 10.0
LED_ROTATION_AXIS_DISTANCES_MM = tuple(
    LED1_ROTATION_AXIS_DISTANCE_MM + index * LED_SPACING_MM for index in range(5)
)
HOLE_ROTATION_AXIS_DISTANCES_MM = tuple(
    LED1_ROTATION_AXIS_DISTANCE_MM + index * HOLE_SPACING_MM for index in range(6)
)

SEQUENCE_COLUMNS = (
    "timestamp_ns",
    "ft_timestamp_ns",
    "actual_force_N",
    "target_force_N",
    "target_tolerance_N",
    "state",
    "band_position",
    "phase_elapsed_s",
    "scheduled_record_observation",
    "events",
    "motor_timestamp_ns",
    "motor_torque_Nm",
)


def contact_location_mm(hole_index: int) -> float:
    """Return the fixture coordinate measured proximally from Hole 1."""

    if not isinstance(hole_index, int) or isinstance(hole_index, bool):
        raise ValueError("hole_index must be an integer from 1 through 6")
    if hole_index not in range(1, 7):
        raise ValueError("hole_index must be an integer from 1 through 6")
    return float((hole_index - 1) * HOLE_SPACING_MM)


def geometry_prior_metadata(hole_index: int) -> dict[str, Any]:
    """Return the fixed physical LED/hole geometry recorded with every run."""

    location_mm = contact_location_mm(hole_index)
    return {
        "axis_origin": "motor_rotation_axis",
        "positive_direction": "proximal",
        "led_order": "LED1_distal_to_LED5_proximal",
        "led1_rotation_axis_distance_mm": LED1_ROTATION_AXIS_DISTANCE_MM,
        "led_spacing_mm": LED_SPACING_MM,
        "led_rotation_axis_distances_mm": list(LED_ROTATION_AXIS_DISTANCES_MM),
        "hole1_alignment": "LED1",
        "hole_spacing_mm": HOLE_SPACING_MM,
        "hole_rotation_axis_distances_mm": list(HOLE_ROTATION_AXIS_DISTANCES_MM),
        "selected_hole_index": hole_index,
        "selected_contact_location_mm": location_mm,
        "selected_rotation_axis_distance_mm": (
            LED1_ROTATION_AXIS_DISTANCE_MM + location_mm
        ),
    }


@dataclass(frozen=True)
class ContactDatasetSnapshot:
    status: str
    run_path: Path | None
    indenter: str | None
    hole_index: int | None
    repetition_index: int | None
    sequence_state: ForceSequenceState
    current_target_n: float | None
    completed_targets_n: tuple[float, ...]
    actual_force_n: float | None
    motor_torque_nm: float | None
    session_elapsed_s: float
    error: str | None


class ProprioceptiveContactDatasetController:
    """Guide one manual force sequence while the runtime records raw streams."""

    _POLL_INTERVAL_S = 0.005

    def __init__(
        self,
        runtime: ProprioceptiveExperimentRuntime,
        config: ForceSequenceConfig | None = None,
    ) -> None:
        self.runtime = runtime
        self.config = config or ForceSequenceConfig(target_forces_n=TARGET_FORCES_N)
        if self.config.target_forces_n != TARGET_FORCES_N:
            raise ValueError(f"target_forces_n must be exactly {TARGET_FORCES_N}")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sequence = ForceSequenceController(self.config)
        self._latest_update = ForceSequenceUpdate(
            state=ForceSequenceState.IDLE,
            events=(),
            band_position=None,
            current_target_n=None,
            current_target_index=0,
            completed_targets_n=(),
            phase_elapsed_s=0.0,
        )
        self._status = "idle"
        self._run_path: Path | None = None
        self._indenter: str | None = None
        self._hole_index: int | None = None
        self._repetition_index: int | None = None
        self._actual_force_n: float | None = None
        self._motor_torque_nm: float | None = None
        self._run_started_s: float | None = None
        self._run_elapsed_offset_s = 0.0
        self._session_elapsed_s = 0.0
        self._rows: list[dict[str, object]] = []
        self._error: str | None = None
        self.runtime.camera.set_contact_frame_gate(self._admit_camera_frame)

    def start_run(self, *, indenter: str, hole_index: int, notes: str = "") -> Path:
        if indenter not in {"sphere_10mm", "sphere_30mm"}:
            raise ValueError("indenter must be sphere_10mm or sphere_30mm")
        location_mm = contact_location_mm(hole_index)
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("a contact-dataset run is already active")
        repetition_index = self._next_repetition_index(indenter, hole_index)
        context = {
            "dataset_type": "proprioceptive_contact_dataset",
            "indenter": indenter,
            "hole_index": hole_index,
            "repetition_index": repetition_index,
            "target_forces_N": list(self.config.target_forces_n),
            "force_sequence": {
                "settle_duration_s": self.config.settle_duration_s,
                "record_duration_s": self.config.record_duration_s,
                "observation_rate_hz": self.config.capture_rate_hz,
                "minimum_tolerance_N": self.config.minimum_tolerance_n,
                "low_force_relative_tolerance": (
                    self.config.low_force_relative_tolerance
                ),
                "high_force_relative_tolerance": (
                    self.config.high_force_relative_tolerance
                ),
                "high_force_threshold_N": self.config.high_force_threshold_n,
                "release_force_threshold_N": RELEASE_FORCE_THRESHOLD_N,
                "rgb_recording_scope": "entire_run_through_release",
            },
            "geometry_prior": geometry_prior_metadata(hole_index),
            "torque_source": "AK40-10 MIT feedback torque",
        }
        run_path = self.runtime.start_recording(
            trial=repetition_index,
            contact_location_gt_mm=location_mm,
            notes=notes,
            experiment_context=context,
        )
        sequence = ForceSequenceController(self.config)
        started_s = monotonic()
        update = sequence.start(started_s)
        with self._lock:
            self._sequence = sequence
            self._latest_update = update
            self._status = "running"
            self._run_path = run_path
            self._indenter = indenter
            self._hole_index = hole_index
            self._repetition_index = repetition_index
            self._actual_force_n = None
            self._motor_torque_nm = None
            self._run_started_s = started_s
            self._run_elapsed_offset_s = self._session_elapsed_s
            self._rows = []
            self._error = None
            self._stop.clear()
            thread = threading.Thread(
                target=self._run,
                name="proprioceptive-contact-sequence",
            )
            self._thread = thread
            thread.start()
        return run_path

    def abort(self) -> bool:
        """Abort and delete only the currently active incomplete run."""

        with self._lock:
            thread = self._thread
            if thread is not None:
                self._status = "aborting"
        if thread is None:
            return False
        self._stop.set()
        thread.join(timeout=3.0)
        if thread.is_alive():
            raise RuntimeError("contact-dataset controller did not stop")
        return True

    def shutdown(self) -> None:
        self.abort()
        self.runtime.camera.set_contact_frame_gate(None)

    def snapshot(self) -> ContactDatasetSnapshot:
        with self._lock:
            return ContactDatasetSnapshot(
                status=self._status,
                run_path=self._run_path,
                indenter=self._indenter,
                hole_index=self._hole_index,
                repetition_index=self._repetition_index,
                sequence_state=self._latest_update.state,
                current_target_n=self._latest_update.current_target_n,
                completed_targets_n=self._latest_update.completed_targets_n,
                actual_force_n=self._actual_force_n,
                motor_torque_nm=self._motor_torque_nm,
                session_elapsed_s=self._session_elapsed_s,
                error=self._error,
            )

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                snapshot = self.runtime.snapshot()
                if snapshot.recorder.status == "error":
                    raise RuntimeError(snapshot.recorder.error or "recorder failed")
                device_errors = tuple(
                    value
                    for value in (
                        snapshot.motor_error,
                        snapshot.ft_error,
                        snapshot.camera_error,
                    )
                    if value
                )
                if device_errors:
                    raise RuntimeError("; ".join(device_errors))
                with self._lock:
                    sequence_complete = (
                        self._latest_update.state is ForceSequenceState.RUN_COMPLETE
                    )
                    released = (
                        self._actual_force_n is not None
                        and self._actual_force_n < RELEASE_FORCE_THRESHOLD_N
                    )
                    if sequence_complete and not released:
                        self._status = "waiting_for_release"
                if sequence_complete and released:
                    self._complete_run()
                    return
                sleep(self._POLL_INTERVAL_S)

            now_s = monotonic()
            with self._lock:
                if self._sequence.state is not ForceSequenceState.RUN_COMPLETE:
                    update = self._sequence.abort(now_s)
                    self._latest_update = update
            self.runtime.stop_recording()
            self._delete_active_run()
            with self._lock:
                self._status = "aborted"
                self._thread = None
        except BaseException as error:
            message = f"{type(error).__name__}: {error}"
            try:
                self.runtime.stop_recording()
                self._write_outputs(status="error", error=message)
            except BaseException as stop_error:
                message += f"; finalize failed: {type(stop_error).__name__}: {stop_error}"
            with self._lock:
                self._status = "error"
                self._error = message
                self._thread = None

    def _complete_run(self) -> None:
        with self._lock:
            self._status = "saving"
        self.runtime.stop_recording()
        self._write_outputs(status="complete", error=None)
        with self._lock:
            self._status = "complete"
            self._thread = None

    def _write_outputs(self, *, status: str, error: str | None) -> None:
        with self._lock:
            run_path = self._run_path
            rows = tuple(self._rows)
            completed_targets = self._latest_update.completed_targets_n
            final_actual_force_n = self._actual_force_n
        if run_path is None or not run_path.is_dir():
            return
        csv_path = run_path / "force_sequence.csv"
        temporary_csv = csv_path.with_suffix(".csv.tmp")
        with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=SEQUENCE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temporary_csv.replace(csv_path)

        metadata_path = run_path / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["status"] = status
        if error is not None:
            metadata["error"] = error
        metadata["force_sequence_result"] = {
            "status": status,
            "completed_targets_N": list(completed_targets),
            "release_force_threshold_N": RELEASE_FORCE_THRESHOLD_N,
            "final_actual_force_N": final_actual_force_n,
            "observation_count": len(rows),
            "error": error,
        }
        temporary_json = metadata_path.with_suffix(".json.tmp")
        temporary_json.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_json.replace(metadata_path)

    def _delete_active_run(self) -> None:
        with self._lock:
            run_path = self._run_path
        if run_path is None or not run_path.exists():
            return
        output_root = self.runtime.recorder.output_root.resolve()
        resolved = run_path.resolve()
        if resolved.parent != output_root or not resolved.name.startswith("run_"):
            raise RuntimeError(f"refusing to delete unexpected run path: {resolved}")
        shutil.rmtree(resolved)

    def _next_repetition_index(self, indenter: str, hole_index: int) -> int:
        maximum = 0
        root = self.runtime.recorder.output_root
        if not root.exists():
            return 1
        for path in root.glob("run_*/metadata.json"):
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                experiment = metadata["experiment"]
                if (
                    experiment.get("dataset_type") == (
                        "proprioceptive_contact_dataset"
                    )
                    and experiment.get("indenter") == indenter
                    and int(experiment.get("hole_index")) == hole_index
                ):
                    maximum = max(maximum, int(experiment["repetition_index"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
                continue
        return maximum + 1

    def _admit_camera_frame(self, sample: CameraAcquisition) -> bool:
        """Advance the force sequence and admit RGB throughout the active run."""

        motor = self.runtime.snapshot().motor
        with self._lock:
            if self._status not in {"running", "waiting_for_release"}:
                return False
            target = self._sequence.current_target_n
            update = self._sequence.update(
                sample.timestamp_ns / 1.0e9,
                sample.ft_contact_force_n,
            )
            self._latest_update = update
            self._actual_force_n = sample.ft_contact_force_n
            self._motor_torque_nm = None if motor is None else motor.torque_nm
            if update.state is ForceSequenceState.RUN_COMPLETE:
                self._status = "waiting_for_release"
            if self._run_started_s is not None:
                self._session_elapsed_s = max(
                    self._session_elapsed_s,
                    self._run_elapsed_offset_s
                    + sample.timestamp_ns / 1.0e9
                    - self._run_started_s,
                )
            self._rows.append(
                {
                    "timestamp_ns": sample.timestamp_ns,
                    "ft_timestamp_ns": sample.ft_timestamp_ns,
                    "actual_force_N": sample.ft_contact_force_n,
                    "target_force_N": "" if target is None else target,
                    "target_tolerance_N": ""
                    if target is None
                    else self.config.tolerance_n(target),
                    "state": (
                        "release"
                        if update.state is ForceSequenceState.RUN_COMPLETE
                        else update.state.name.lower()
                    ),
                    "band_position": ""
                    if update.band_position is None
                    else update.band_position.name.lower(),
                    "phase_elapsed_s": update.phase_elapsed_s,
                    "scheduled_record_observation": int(
                        update.should_record_frame
                    ),
                    "events": "|".join(
                        event.name.lower() for event in update.events
                    ),
                    "motor_timestamp_ns": ""
                    if motor is None
                    else motor.timestamp_ns,
                    "motor_torque_Nm": "" if motor is None else motor.torque_nm,
                }
            )
            return True


__all__ = [
    "ContactDatasetSnapshot",
    "HOLE_ROTATION_AXIS_DISTANCES_MM",
    "HOLE_SPACING_MM",
    "LED1_ROTATION_AXIS_DISTANCE_MM",
    "LED_ROTATION_AXIS_DISTANCES_MM",
    "LED_SPACING_MM",
    "ProprioceptiveContactDatasetController",
    "RELEASE_FORCE_THRESHOLD_N",
    "SEQUENCE_COLUMNS",
    "TARGET_FORCES_N",
    "contact_location_mm",
    "geometry_prior_metadata",
]
