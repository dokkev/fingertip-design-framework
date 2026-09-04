"""Collect synchronized RealSense RGB and Bota Rokubi contact data in a GUI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import queue
import sys
import threading
from time import perf_counter
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

import cv2


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiments.data_collection import (  # noqa: E402
    ContactDatasetWriter,
    ForceBandPosition,
    ForceSequenceConfig,
    ForceSequenceController,
    ForceSequenceEvent,
    ForceSequenceState,
    ForceSequenceUpdate,
    LoadedRunHandle,
    SegmentHandle,
    SessionMetadata,
    SynchronizedFrame,
    UnloadedCaptureController,
    UnloadedCaptureEvent,
    UnloadedCaptureState,
    UnloadedCaptureUpdate,
)
from experiments.hardware import (  # noqa: E402
    BotaSample,
    BotaSerialSensor,
    BotaTareOffsets,
    ColorFrame,
    RealSenseColorCamera,
)


CAMERA_READ_TIMEOUT_MS = 2000
CAMERA_QUEUE_CAPACITY = 4
WRITER_FRAME_CAPACITY = 32
GUI_REFRESH_MS = 10
PREVIEW_WIDTH = 960
PREVIEW_HEIGHT = 540


@dataclass(frozen=True)
class _TimedColorFrame:
    frame: ColorFrame
    host_time_s: float


class _CameraReader:
    """Keep blocking RealSense reads outside the Tk event loop."""

    def __init__(self, camera: RealSenseColorCamera) -> None:
        self._camera = camera
        self._frames: queue.Queue[_TimedColorFrame] = queue.Queue(
            maxsize=CAMERA_QUEUE_CAPACITY
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._dropped = 0

    @property
    def dropped_frame_count(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop,
            name="realsense_collection_reader",
            daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                frame = self._camera.read(timeout_ms=CAMERA_READ_TIMEOUT_MS)
                timed = _TimedColorFrame(frame=frame, host_time_s=perf_counter())
                try:
                    self._frames.put_nowait(timed)
                except queue.Full:
                    try:
                        self._frames.get_nowait()
                    except queue.Empty:
                        pass
                    with self._lock:
                        self._dropped += 1
                    self._frames.put_nowait(timed)
        except BaseException as error:
            if not self._stop.is_set():
                with self._lock:
                    self._error = error

    def drain(self) -> list[_TimedColorFrame]:
        frames: list[_TimedColorFrame] = []
        while True:
            try:
                frames.append(self._frames.get_nowait())
            except queue.Empty:
                return frames

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=CAMERA_READ_TIMEOUT_MS / 1000.0 + 0.5)


class _MockBotaSensor:
    """GUI-only manual force source, isolated under a MOCK dataset namespace."""

    def __init__(self, force_getter: Callable[[], float]) -> None:
        self.port = "MOCK"
        self.tare_offsets = BotaTareOffsets()
        self._force_getter = force_getter

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def tare(self, **_: object) -> BotaTareOffsets:
        return self.tare_offsets

    def latest_sample(self) -> BotaSample:
        return self.nearest_sample(perf_counter())

    def nearest_sample(self, host_time_s: float) -> BotaSample:
        force = max(0.0, float(self._force_getter()))
        return BotaSample(
            host_time_s=host_time_s,
            sensor_timestamp=round(host_time_s * 1000.0),
            status=0,
            temperature_c=25.0,
            fx_n=0.0,
            fy_n=0.0,
            fz_n=force,
            mx_nm=0.0,
            my_nm=0.0,
            mz_nm=0.0,
            force_magnitude_n=force,
            torque_magnitude_nm=0.0,
            fz_share=0.0 if force == 0.0 else 1.0,
        )


class ContactCollectorApp:
    """One concrete Tkinter interface for the LUMO physical experiment."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        camera: RealSenseColorCamera,
        camera_reader: _CameraReader,
        sensor: BotaSerialSensor | _MockBotaSensor,
        output_root: Path,
        config: ForceSequenceConfig,
        mock_sensor: bool,
        mock_force_variable: tk.DoubleVar | None = None,
    ) -> None:
        self.root = root
        self.camera = camera
        self.camera_reader = camera_reader
        self.sensor = sensor
        self.output_root = output_root
        self.config = config
        self.mock_sensor = mock_sensor
        self.mock_force = mock_force_variable
        self.writer: ContactDatasetWriter | None = None
        self.force_controller: ForceSequenceController | None = None
        self.unloaded_controller: UnloadedCaptureController | None = None
        self.active_mode: str | None = None
        self.active_run: LoadedRunHandle | None = None
        self.active_segment: SegmentHandle | None = None
        self.unloaded_capture_count = 0
        self.current_tare_offsets: BotaTareOffsets | None = None
        self.tare_in_progress = False
        self.tare_results: queue.SimpleQueue[BotaTareOffsets | BaseException] = (
            queue.SimpleQueue()
        )
        self.last_camera_drop_count = 0
        self.last_preview: tk.PhotoImage | None = None
        self.closed = False

        self._build_ui()
        self._set_inputs_enabled(False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._begin_tare)
        self.root.after(GUI_REFRESH_MS, self._tick)

    def _build_ui(self) -> None:
        self.root.title("LUMO contact dataset collector")
        self.root.minsize(1250, 700)
        outer = ttk.Frame(self.root, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)

        title = "LUMO contact dataset acquisition — Bota Rokubi"
        ttk.Label(outer, text=title, font=("TkDefaultFont", 15, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        if self.mock_sensor:
            tk.Label(
                outer,
                text="MOCK SENSOR — NOT PHYSICAL EXPERIMENT DATA",
                bg="#b00020",
                fg="white",
                font=("TkDefaultFont", 13, "bold"),
                padx=10,
                pady=5,
            ).grid(row=0, column=1, sticky="e", pady=(0, 8))

        self.preview = ttk.Label(outer, anchor="center")
        self.preview.grid(row=1, column=0, sticky="nsew", padx=(0, 12))

        side = ttk.Frame(outer)
        side.grid(row=1, column=1, sticky="nsew")
        self.material = tk.StringVar(value="dragon_skin")
        self.morphology = tk.StringVar(value="nominal")
        self.specimen_id = tk.StringVar(value="dragon_skin_nominal_01")
        self.indenter = tk.StringVar(value="sphere_15mm")
        self.hole = tk.IntVar(value=1)
        self.repeat = tk.IntVar(value=1)
        self.session_widgets: list[tk.Widget] = []
        self.input_widgets: list[tk.Widget] = []

        specimen = ttk.LabelFrame(side, text="Session specimen", padding=10)
        specimen.grid(row=0, column=0, sticky="ew")
        specimen.columnconfigure(1, weight=1)
        ttk.Label(specimen, text="Material").grid(row=0, column=0, sticky="w")
        material = ttk.Combobox(
            specimen,
            textvariable=self.material,
            values=("dragon_skin", "solaris"),
        )
        material.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=2)
        ttk.Label(specimen, text="Morphology").grid(row=1, column=0, sticky="w")
        morphology = ttk.Combobox(
            specimen,
            textvariable=self.morphology,
            values=("nominal", "optimized"),
        )
        morphology.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=2)
        ttk.Label(specimen, text="Specimen ID").grid(row=2, column=0, sticky="w")
        specimen_id = ttk.Entry(specimen, textvariable=self.specimen_id)
        specimen_id.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=2)
        self.create_session_button = ttk.Button(
            specimen,
            text="CREATE SESSION",
            command=self._create_session,
        )
        self.create_session_button.grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )
        self.session_widgets.extend((material, morphology, specimen_id))

        conditions = ttk.LabelFrame(side, text="Run conditions", padding=10)
        conditions.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        conditions.columnconfigure(1, weight=1)
        ttk.Label(conditions, text="Indenter").grid(row=0, column=0, sticky="w")
        indenter = ttk.Combobox(
            conditions,
            textvariable=self.indenter,
            values=("sphere_10mm", "sphere_15mm", "sphere_20mm"),
        )
        indenter.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=2)
        ttk.Label(conditions, text="Hole number").grid(row=1, column=0, sticky="w")
        hole = ttk.Spinbox(conditions, from_=1, to=6, textvariable=self.hole, width=6)
        hole.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=2)
        ttk.Label(conditions, text="Repeat index").grid(row=2, column=0, sticky="w")
        repeat = ttk.Spinbox(
            conditions,
            from_=1,
            to=999,
            textvariable=self.repeat,
            width=6,
        )
        repeat.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=2)
        ttk.Label(conditions, text="Hole 1 = distal · Hole 6 = proximal").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        self.input_widgets.extend((indenter, hole, repeat))

        force_box = ttk.LabelFrame(side, text="Live Bota Rokubi", padding=10)
        force_box.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.force_text = tk.StringVar(value="F magnitude: -- N")
        self.fz_text = tk.StringVar(value="Fz: -- N")
        self.share_text = tk.StringVar(value="|Fz| / F: -- %")
        for row, variable in enumerate((self.force_text, self.fz_text, self.share_text)):
            ttk.Label(force_box, textvariable=variable, font=("TkFixedFont", 12)).grid(
                row=row, column=0, sticky="w"
            )
        if self.mock_sensor:
            assert self.mock_force is not None
            ttk.Scale(force_box, from_=0.0, to=17.0, variable=self.mock_force).grid(
                row=3, column=0, sticky="ew", pady=(8, 0)
            )
            ttk.Label(force_box, text="Manual MOCK force: 0–17 N").grid(
                row=4, column=0, sticky="w"
            )

        sequence = ttk.LabelFrame(side, text="Progressive loading", padding=10)
        sequence.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.sequence_labels: list[ttk.Label] = []
        for column, target in enumerate(self.config.target_forces_n):
            label = ttk.Label(sequence, text=f"○ {target:g} N", padding=4)
            label.grid(row=0, column=column, sticky="w")
            self.sequence_labels.append(label)
        self.state_text = tk.StringVar(value="Starting sensor tare…")
        ttk.Label(
            sequence,
            textvariable=self.state_text,
            font=("TkDefaultFont", 11, "bold"),
            wraplength=330,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self.force_progress = ttk.Progressbar(sequence, maximum=1.0, value=0.0)
        self.force_progress.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(6, 0))

        self.unloaded_text = tk.StringVar(value="Unloaded reference: NOT CAPTURED")
        ttk.Label(side, textvariable=self.unloaded_text).grid(
            row=4, column=0, sticky="w", pady=(10, 0)
        )

        controls = ttk.Frame(side)
        controls.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        self.start_button = ttk.Button(controls, text="START RUN", command=self._start_run)
        self.start_button.grid(row=0, column=0, padx=(0, 5))
        self.unloaded_button = ttk.Button(
            controls, text="CAPTURE UNLOADED", command=self._start_unloaded
        )
        self.unloaded_button.grid(row=0, column=1, padx=5)
        self.abort_button = ttk.Button(controls, text="ABORT", command=self._abort)
        self.abort_button.grid(row=0, column=2, padx=5)
        self.tare_button = ttk.Button(controls, text="TARE", command=self._begin_tare)
        self.tare_button.grid(row=0, column=3, padx=(5, 0))

        self.io_text = tk.StringVar(value="Writer: waiting for tare")
        ttk.Label(side, textvariable=self.io_text, wraplength=360).grid(
            row=6, column=0, sticky="w", pady=(10, 0)
        )

    def _set_inputs_enabled(self, enabled: bool) -> None:
        session_state = "normal" if enabled and self.writer is None else "disabled"
        for widget in self.session_widgets:
            widget.configure(state=session_state)
        input_state = "normal" if enabled and self.writer is not None else "disabled"
        for widget in self.input_widgets:
            widget.configure(state=input_state)
        create_enabled = (
            enabled
            and not self.tare_in_progress
            and self.writer is None
            and self.current_tare_offsets is not None
        )
        self.create_session_button.configure(
            state="normal" if create_enabled else "disabled"
        )
        controls_enabled = (
            enabled
            and self.writer is not None
            and not self.tare_in_progress
        )
        self.start_button.configure(state="normal" if controls_enabled else "disabled")
        self.unloaded_button.configure(
            state="normal" if controls_enabled else "disabled"
        )
        self.tare_button.configure(state="normal" if enabled else "disabled")
        self.abort_button.configure(state="normal" if self.active_mode else "disabled")

    def _begin_tare(self) -> None:
        if self.active_mode is not None or self.tare_in_progress:
            return
        self.tare_in_progress = True
        self.state_text.set("TARING — keep the Bota Rokubi completely unloaded")
        self._set_inputs_enabled(False)

        def work() -> None:
            try:
                self.tare_results.put(self.sensor.tare())
            except BaseException as error:
                self.tare_results.put(error)

        threading.Thread(target=work, name="bota_tare", daemon=True).start()

    def _poll_tare(self) -> None:
        try:
            result = self.tare_results.get_nowait()
        except queue.Empty:
            return
        self.tare_in_progress = False
        if isinstance(result, BaseException):
            self.state_text.set(f"TARE FAILED: {result}")
            self.current_tare_offsets = None
            self._set_inputs_enabled(True)
            messagebox.showerror("Bota Rokubi tare failed", str(result))
            return
        self.current_tare_offsets = result
        if self.writer is not None:
            self.writer.update_tare_offsets(result)
            self.state_text.set("TARE COMPLETE — session tare metadata updated")
        else:
            self.state_text.set("TARE COMPLETE — enter specimen identity and create session")
            self.io_text.set("Writer: create a specimen session")
        self._set_inputs_enabled(True)

    def _create_session(self) -> None:
        if self.writer is not None or self.current_tare_offsets is None:
            return
        try:
            session = SessionMetadata(
                material=self.material.get().strip(),
                morphology=self.morphology.get().strip(),
                specimen_id=self.specimen_id.get().strip(),
                camera_model=self.camera.device_name or "Intel RealSense",
                camera_width=self.camera.width,
                camera_height=self.camera.height,
                camera_fps=self.camera.fps,
                bota_serial_port=self.sensor.port,
                bota_tare_offsets=self.current_tare_offsets,
                force_sequence=self.config,
                sensor_mode="mock" if self.mock_sensor else "physical",
                camera_serial_number=self.camera.serial_number,
                bota_model="Mock Rokubi" if self.mock_sensor else "Bota Rokubi",
            )
            self.writer = ContactDatasetWriter(
                self.output_root,
                session,
                frame_queue_capacity=WRITER_FRAME_CAPACITY,
            )
        except (OSError, RuntimeError, ValueError) as error:
            messagebox.showerror("Cannot create session", str(error))
            return
        self.state_text.set("READY — configure a loaded run or unloaded capture")
        self.io_text.set(f"Session: {self.writer.session_path}")
        self._set_inputs_enabled(True)

    def _conditions(self) -> tuple[str, int, int]:
        indenter = self.indenter.get().strip()
        try:
            hole = int(self.hole.get())
            repeat = int(self.repeat.get())
        except (TypeError, ValueError) as error:
            raise ValueError("hole and repeat must be integers") from error
        if not indenter:
            raise ValueError("indenter is required")
        if hole not in range(1, 7):
            raise ValueError("hole number must be from 1 through 6")
        if repeat < 1:
            raise ValueError("repeat index must be positive")
        return indenter, hole, repeat

    def _start_run(self) -> None:
        if self.writer is None or self.active_mode is not None:
            return
        try:
            indenter, hole, repeat = self._conditions()
            self.active_run = self.writer.start_loaded_run(
                indenter=indenter,
                hole_index=hole,
                repeat_index=repeat,
            )
        except (RuntimeError, ValueError) as error:
            messagebox.showerror("Cannot start run", str(error))
            return
        self.force_controller = ForceSequenceController(self.config)
        self.active_mode = "loaded"
        self.active_segment = None
        self.state_text.set(f"Increase force to {self.config.target_forces_n[0]:g} N")
        self._set_inputs_enabled(False)
        self._update_sequence_labels()

    def _start_unloaded(self) -> None:
        if self.writer is None or self.active_mode is not None:
            return
        self.unloaded_controller = UnloadedCaptureController(self.config)
        self.active_mode = "unloaded"
        self.active_segment = None
        self.state_text.set(
            f"Remove load and hold at ≤ {self.config.unloaded_max_force_n:g} N"
        )
        self._set_inputs_enabled(False)

    def _abort(self) -> None:
        if self.active_mode is None or self.writer is None:
            return
        now = perf_counter()
        if self.active_segment is not None:
            self.writer.discard_segment(self.active_segment)
            self.active_segment = None
        if self.active_mode == "loaded" and self.active_run is not None:
            assert self.force_controller is not None
            if self.force_controller.state is not ForceSequenceState.RUN_COMPLETE:
                self.force_controller.abort(now)
            self.writer.abort_loaded_run(self.active_run)
        elif self.active_mode == "unloaded":
            assert self.unloaded_controller is not None
            if self.unloaded_controller.state is not UnloadedCaptureState.COMPLETE:
                self.unloaded_controller.abort(now)
        self.state_text.set("ABORTED — completed force targets were preserved")
        self._finish_active()

    def _tick(self) -> None:
        if self.closed:
            return
        self._poll_tare()
        camera_error = self.camera_reader.error
        if camera_error is not None:
            self.state_text.set(f"CAMERA ERROR: {camera_error}")
            self._abort()
        frames = self.camera_reader.drain()
        drop_count = self.camera_reader.dropped_frame_count
        if drop_count != self.last_camera_drop_count and self.active_segment is not None:
            self._abort_for_writer_loss("camera delivery queue overflow")
        self.last_camera_drop_count = drop_count
        for timed in frames:
            self._process_frame(timed)
        if frames:
            self._show_preview(frames[-1].frame)
        self._update_live_force()
        self._update_io_status()
        self.root.after(GUI_REFRESH_MS, self._tick)

    def _process_frame(self, timed: _TimedColorFrame) -> None:
        sample = self.sensor.nearest_sample(timed.host_time_s)
        if sample is None or self.active_mode is None or self.writer is None:
            return
        frame = SynchronizedFrame(
            rgb=timed.frame.rgb,
            camera_host_time_s=timed.host_time_s,
            camera_device_timestamp_ms=timed.frame.timestamp_ms,
            camera_frame_number=timed.frame.frame_number,
            bota_sample=sample,
        )
        if self.active_mode == "loaded":
            self._process_loaded_frame(frame)
        else:
            self._process_unloaded_frame(frame)

    def _process_loaded_frame(self, frame: SynchronizedFrame) -> None:
        assert self.writer is not None
        assert self.force_controller is not None
        assert self.active_run is not None
        if self.force_controller.state is ForceSequenceState.IDLE:
            self.force_controller.start(frame.camera_host_time_s)
        update = self.force_controller.update(
            frame.camera_host_time_s, frame.bota_sample.force_magnitude_n
        )
        if ForceSequenceEvent.ATTEMPT_RESET in update.events:
            if self.active_segment is not None:
                self.writer.discard_segment(self.active_segment)
                self.active_segment = None
        if ForceSequenceEvent.RECORDING_STARTED in update.events:
            assert update.record_target_n is not None
            self.active_segment = self.writer.begin_force_target(
                self.active_run, update.record_target_n
            )
        if update.should_record_frame:
            assert self.active_segment is not None
            if not self.writer.submit_frame(
                self.active_segment,
                frame,
                capture_elapsed_s=update.phase_elapsed_s,
            ):
                self._abort_for_writer_loss("PNG writer queue overflow")
                return
        if ForceSequenceEvent.TARGET_COMPLETED in update.events:
            assert self.active_segment is not None
            self.writer.finalize_segment(self.active_segment)
            self.active_segment = None
        if ForceSequenceEvent.RUN_COMPLETED in update.events:
            self.writer.complete_loaded_run(self.active_run)
            self.state_text.set("RUN COMPLETE — release indenter")
            self._finish_active()
        else:
            self._show_loaded_update(update)
        self._update_sequence_labels()

    def _process_unloaded_frame(self, frame: SynchronizedFrame) -> None:
        assert self.writer is not None
        assert self.unloaded_controller is not None
        if self.unloaded_controller.state is UnloadedCaptureState.IDLE:
            self.unloaded_controller.start(frame.camera_host_time_s)
        update = self.unloaded_controller.update(
            frame.camera_host_time_s, frame.bota_sample.force_magnitude_n
        )
        if UnloadedCaptureEvent.ATTEMPT_RESET in update.events:
            if self.active_segment is not None:
                self.writer.discard_segment(self.active_segment)
                self.active_segment = None
        if UnloadedCaptureEvent.RECORDING_STARTED in update.events:
            self.active_segment = self.writer.begin_unloaded_capture()
        if update.should_record_frame:
            assert self.active_segment is not None
            if not self.writer.submit_frame(
                self.active_segment,
                frame,
                capture_elapsed_s=update.phase_elapsed_s,
            ):
                self._abort_for_writer_loss("PNG writer queue overflow")
                return
        if UnloadedCaptureEvent.CAPTURE_COMPLETED in update.events:
            assert self.active_segment is not None
            self.writer.finalize_segment(self.active_segment)
            self.active_segment = None
            self.unloaded_capture_count += 1
            self.state_text.set("UNLOADED CAPTURE COMPLETE")
            self._update_unloaded_label()
            self._finish_active()
        else:
            self._show_unloaded_update(update)

    def _show_loaded_update(self, update: ForceSequenceUpdate) -> None:
        assert self.force_controller is not None
        state = self.force_controller.state
        target = self.force_controller.current_target_n
        phase_elapsed = update.phase_elapsed_s
        band = update.band_position
        if state is ForceSequenceState.WAITING_FOR_TARGET:
            prefix = "ABOVE TARGET" if band is ForceBandPosition.ABOVE else "Increase force"
            self.state_text.set(f"{prefix} — establish {target:g} N inside tolerance")
            self.force_progress["value"] = 0.0
        elif state is ForceSequenceState.SETTLING:
            self.state_text.set(
                f"HOLDING {phase_elapsed:.2f} / {self.config.settle_duration_s:.2f} s"
            )
            self.force_progress["value"] = min(
                1.0, phase_elapsed / max(self.config.settle_duration_s, 1.0e-9)
            )
        elif state is ForceSequenceState.RECORDING:
            self.state_text.set(
                f"RECORDING {phase_elapsed:.2f} / {self.config.record_duration_s:.2f} s"
            )
            self.force_progress["value"] = min(
                1.0, phase_elapsed / self.config.record_duration_s
            )

    def _show_unloaded_update(self, update: UnloadedCaptureUpdate) -> None:
        assert self.unloaded_controller is not None
        state = self.unloaded_controller.state
        elapsed = update.phase_elapsed_s
        if state is UnloadedCaptureState.WAITING_FOR_UNLOADED:
            self.state_text.set(
                f"Remove load and hold at ≤ {self.config.unloaded_max_force_n:g} N"
            )
        elif state is UnloadedCaptureState.SETTLING:
            self.state_text.set(
                "UNLOADED HOLDING "
                f"{elapsed:.2f} / {self.config.unloaded_settle_duration_s:.2f} s"
            )
        elif state is UnloadedCaptureState.RECORDING:
            self.state_text.set(
                "UNLOADED RECORDING "
                f"{elapsed:.2f} / {self.config.unloaded_record_duration_s:.2f} s"
            )

    def _abort_for_writer_loss(self, reason: str) -> None:
        assert self.writer is not None
        if self.active_segment is not None:
            self.writer.discard_segment(self.active_segment)
            self.active_segment = None
        if self.active_mode == "loaded" and self.active_run is not None:
            self.writer.abort_loaded_run(self.active_run)
        self.state_text.set(f"RUN ABORTED — {reason}; no frames were silently dropped")
        self._finish_active()

    def _finish_active(self) -> None:
        self.active_mode = None
        self.active_run = None
        self.force_controller = None
        self.unloaded_controller = None
        self.active_segment = None
        self.force_progress["value"] = 0.0
        self._set_inputs_enabled(True)
        self._update_sequence_labels()

    def _update_sequence_labels(self) -> None:
        completed = (
            set(self.force_controller.completed_targets_n)
            if self.force_controller is not None
            else set()
        )
        current = (
            self.force_controller.current_target_n
            if self.force_controller is not None
            else None
        )
        for label, target in zip(self.sequence_labels, self.config.target_forces_n):
            marker = "✓" if target in completed else "●" if target == current else "○"
            label.configure(text=f"{marker} {target:g} N")

    def _update_live_force(self) -> None:
        try:
            sample = self.sensor.latest_sample()
        except RuntimeError as error:
            self.state_text.set(f"BOTA ERROR: {error}")
            return
        if sample is None:
            return
        self.force_text.set(f"F magnitude: {sample.force_magnitude_n:7.3f} N")
        self.fz_text.set(f"Fz:          {sample.fz_n:7.3f} N")
        self.share_text.set(f"|Fz| / F:    {100.0 * sample.fz_share:6.1f} %")

    def _update_io_status(self) -> None:
        if self.writer is None:
            return
        error = self.writer.error
        if error is not None:
            self.state_text.set(f"WRITER ERROR: {error}")
            if self.active_mode is not None:
                self._finish_active()
        self.io_text.set(
            f"Session: {self.writer.session_path}\n"
            f"Dropped: camera={self.camera_reader.dropped_frame_count}, "
            f"writer={self.writer.dropped_frame_count}"
        )

    def _update_unloaded_label(self, *_: object) -> None:
        self.unloaded_text.set(
            f"Unloaded captures in session: {self.unloaded_capture_count}"
        )

    def _show_preview(self, frame: ColorFrame) -> None:
        resized = cv2.resize(
            frame.rgb,
            (PREVIEW_WIDTH, PREVIEW_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        ppm = (
            f"P6\n{PREVIEW_WIDTH} {PREVIEW_HEIGHT}\n255\n".encode("ascii")
            + resized.tobytes()
        )
        image = tk.PhotoImage(data=ppm, format="PPM")
        self.preview.configure(image=image)
        self.last_preview = image

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.active_mode is not None:
            self._abort()
        self.camera_reader.stop()
        self.camera.stop()
        self.sensor.stop()
        if self.writer is not None:
            try:
                self.writer.close()
            except RuntimeError as error:
                messagebox.showerror("Dataset writer error", str(error))
        self.root.destroy()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bota-port", default="/dev/ttyUSB0")
    parser.add_argument("--output", type=Path, default=_REPOSITORY_ROOT / "experiments")
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-serial")
    parser.add_argument(
        "--capture-rate-hz",
        type=float,
        default=5.0,
        help="saved RGB frame rate during loaded and unloaded recording bursts",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="replace the Rokubi with a manual force slider and isolate output as MOCK",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    root = tk.Tk()
    mock_force = tk.DoubleVar(master=root, value=0.0)
    camera = RealSenseColorCamera(
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        serial_number=args.camera_serial,
    )
    sensor: BotaSerialSensor | _MockBotaSensor
    sensor = _MockBotaSensor(mock_force.get) if args.mock else BotaSerialSensor(args.bota_port)
    try:
        camera.start()
        sensor.start()
    except BaseException:
        camera.stop()
        sensor.stop()
        root.destroy()
        raise
    reader = _CameraReader(camera)
    reader.start()
    app = ContactCollectorApp(
        root,
        camera=camera,
        camera_reader=reader,
        sensor=sensor,
        output_root=args.output,
        config=ForceSequenceConfig(capture_rate_hz=args.capture_rate_hz),
        mock_sensor=args.mock,
        mock_force_variable=mock_force if args.mock else None,
    )
    try:
        root.mainloop()
    finally:
        if not app.closed:
            app.close()


if __name__ == "__main__":
    main()
