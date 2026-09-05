"""Collect continuous cyclic contact histories from RealSense RGB and Rokubi."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import queue
import sys
import threading
from time import perf_counter, sleep
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

import cv2
import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiments.data_collection import (  # noqa: E402
    ForceSequenceConfig,
    ForceTrajectoryConfig,
    ForceTrajectoryController,
    ForceTrajectoryState,
    ForceTrajectoryUpdate,
    HistoryDatasetWriter,
    HistoryRunHandle,
    HistorySessionMetadata,
    HistorySynchronizedFrame,
    HistoryUnloadedHandle,
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
WRITER_FRAME_CAPACITY = 64
GUI_REFRESH_MS = 10
PREVIEW_WIDTH = 880
PREVIEW_HEIGHT = 495
GAUGE_WIDTH = 205
GAUGE_HEIGHT = 240
GAUGE_TOP = 16
GAUGE_BOTTOM = 222
GAUGE_LEFT = 26
GAUGE_RIGHT = 62


@dataclass(frozen=True)
class _TimedColorFrame:
    frame: ColorFrame
    host_time_s: float


class _MockColorCamera:
    """Static RGB source for an entirely hardware-free GUI exercise."""

    def __init__(self, *, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.device_name = "Static mock RGB"
        self.serial_number = "MOCK-CAMERA"
        self.exposure_us: float | None = None
        self.gain: float | None = None
        self.white_balance_k: float | None = None
        self._running = False
        self._frame_number = 0
        self._next_frame_s = 0.0
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = np.asarray(25.0 + 25.0 * x, dtype=np.uint8)
        image[..., 1] = np.asarray(45.0 + 45.0 * y, dtype=np.uint8)
        image[..., 2] = 32
        image.setflags(write=False)
        self._image = image

    def start(self) -> None:
        self._running = True
        self._next_frame_s = perf_counter()

    def set_manual_photometric_controls(
        self, *, exposure_us: float, gain: float, white_balance_k: float
    ) -> None:
        self.exposure_us = float(exposure_us)
        self.gain = float(gain)
        self.white_balance_k = float(white_balance_k)

    def read(self, *, timeout_ms: int = CAMERA_READ_TIMEOUT_MS) -> ColorFrame:
        del timeout_ms
        if not self._running:
            raise RuntimeError("mock camera is not running")
        now = perf_counter()
        if now < self._next_frame_s:
            sleep(self._next_frame_s - now)
        captured = perf_counter()
        self._next_frame_s = captured + 1.0 / self.fps
        self._frame_number += 1
        return ColorFrame(
            rgb=self._image,
            timestamp_ms=captured * 1000.0,
            frame_number=self._frame_number,
        )

    def stop(self) -> None:
        self._running = False


Camera = RealSenseColorCamera | _MockColorCamera


class _CameraReader:
    """Keep blocking camera reads outside the Tk event loop."""

    def __init__(self, camera: Camera) -> None:
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
            name="contact_history_camera_reader",
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
        if self._thread is not None:
            self._thread.join(timeout=CAMERA_READ_TIMEOUT_MS / 1000.0 + 0.5)


class _MockBotaSensor:
    """Manual force source for mock-only contact histories."""

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


Sensor = BotaSerialSensor | _MockBotaSensor


class ContactHistoryApp:
    """Dedicated guided-acquisition GUI for one continuous trajectory per run."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        camera: Camera,
        camera_reader: _CameraReader,
        sensor: Sensor,
        output_root: Path,
        config: ForceTrajectoryConfig,
        mock_sensor: bool,
        mock_force_variable: tk.DoubleVar | None = None,
        mock_smoke: bool = False,
    ) -> None:
        self.root = root
        self.camera = camera
        self.camera_reader = camera_reader
        self.sensor = sensor
        self.output_root = output_root
        self.config = config
        self.mock_sensor = mock_sensor
        self.mock_force = mock_force_variable
        self.mock_smoke = mock_smoke
        self.writer: HistoryDatasetWriter | None = None
        self.trajectory: ForceTrajectoryController | None = None
        self.last_trajectory_update: ForceTrajectoryUpdate | None = None
        self.unloaded: UnloadedCaptureController | None = None
        self.active_run: HistoryRunHandle | None = None
        self.active_unloaded: HistoryUnloadedHandle | None = None
        self.active_mode: str | None = None
        self.current_tare_offsets: BotaTareOffsets | None = None
        self.tare_in_progress = False
        self.tare_results: queue.SimpleQueue[BotaTareOffsets | BaseException] = (
            queue.SimpleQueue()
        )
        self.series_total = 0
        self.series_completed = 0
        self.series_condition: tuple[str, int] | None = None
        self.run_camera_drop_start = 0
        self.unloaded_capture_count = 0
        self.last_preview: tk.PhotoImage | None = None
        self.closed = False
        self._smoke_unloaded_requested = False
        self._smoke_started = False

        self._build_ui()
        self._set_inputs_enabled(False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self._begin_tare)
        self.root.after(GUI_REFRESH_MS, self._tick)
        if self.mock_smoke:
            self.root.after(25, self._drive_mock_smoke)

    def _build_ui(self) -> None:
        self.root.title("LUMO continuous contact-history collector")
        self.root.minsize(1240, 710)
        outer = ttk.Frame(self.root, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)
        ttk.Label(
            outer,
            text="LUMO continuous contact-history acquisition — Bota Rokubi",
            font=("TkDefaultFont", 15, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
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
        self.morphology = tk.StringVar(value="baseline")
        self.specimen_id = tk.StringVar(value="dragon_skin_baseline_history_01")
        self.indenter = tk.StringVar(value="sphere_10mm")
        self.hole = tk.IntVar(value=1)
        self.series_run_count = tk.IntVar(value=1)
        self.repetition_text = tk.StringVar(value="—")
        self.series_text = tk.StringVar(value="—")
        self.session_widgets: list[tk.Widget] = []
        self.condition_widgets: list[tk.Widget] = []

        specimen = ttk.LabelFrame(side, text="Session specimen", padding=8)
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
            values=("baseline", "flat_opt", "angled_opt"),
        )
        morphology.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=2)
        ttk.Label(specimen, text="Specimen ID").grid(row=2, column=0, sticky="w")
        specimen_id = ttk.Entry(specimen, textvariable=self.specimen_id)
        specimen_id.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=2)
        self.create_button = ttk.Button(
            specimen, text="CREATE SESSION", command=self._create_session
        )
        self.create_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.session_widgets.extend((material, morphology, specimen_id))

        conditions = ttk.LabelFrame(side, text="Run conditions", padding=8)
        conditions.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        conditions.columnconfigure(1, weight=1)
        ttk.Label(conditions, text="Indenter").grid(row=0, column=0, sticky="w")
        buttons = ttk.Frame(conditions)
        buttons.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        buttons.columnconfigure((0, 1), weight=1)
        indenter_widgets: list[tk.Widget] = []
        for column, (label, value) in enumerate(
            (("10 mm", "sphere_10mm"), ("30 mm", "sphere_30mm"))
        ):
            button = ttk.Radiobutton(
                buttons,
                text=label,
                variable=self.indenter,
                value=value,
                style="Toolbutton",
            )
            button.grid(row=0, column=column, sticky="ew", padx=3)
            indenter_widgets.append(button)
        ttk.Label(conditions, text="Hole number").grid(row=1, column=0, sticky="w")
        hole = ttk.Spinbox(conditions, from_=1, to=6, textvariable=self.hole, width=6)
        hole.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=2)
        ttk.Label(conditions, text="Runs in series").grid(row=2, column=0, sticky="w")
        count = ttk.Spinbox(
            conditions,
            from_=1,
            to=100,
            textvariable=self.series_run_count,
            width=6,
        )
        count.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=2)
        ttk.Label(conditions, text="Series progress").grid(row=3, column=0, sticky="w")
        ttk.Label(conditions, textvariable=self.series_text).grid(
            row=3, column=1, sticky="w", padx=(8, 0)
        )
        ttk.Label(conditions, text="Repetition Index").grid(row=4, column=0, sticky="w")
        ttk.Label(conditions, textvariable=self.repetition_text).grid(
            row=4, column=1, sticky="w", padx=(8, 0)
        )
        ttk.Label(conditions, text="Hole 1 = distal · Hole 6 = proximal").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(3, 0)
        )
        self.condition_widgets.extend((*indenter_widgets, hole, count))

        live = ttk.LabelFrame(side, text="Live Bota Rokubi", padding=8)
        live.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.force_text = tk.StringVar(value="F magnitude: -- N")
        self.fz_text = tk.StringVar(value="Fz: -- N")
        self.share_text = tk.StringVar(value="|Fz| / F: -- %")
        for row, variable in enumerate(
            (self.force_text, self.fz_text, self.share_text)
        ):
            ttk.Label(live, textvariable=variable, font=("TkFixedFont", 11)).grid(
                row=row, column=0, sticky="w"
            )
        self.force_gauge_max_n = max(20.0, self.config.max_force_n + 2.0)
        self.gauge = tk.Canvas(
            live,
            width=GAUGE_WIDTH,
            height=GAUGE_HEIGHT,
            background="white",
            highlightbackground="#b8b8b8",
            highlightthickness=1,
        )
        self.gauge.grid(row=0, column=1, rowspan=5, padx=(12, 0))
        self.gauge.create_rectangle(
            GAUGE_LEFT,
            GAUGE_TOP,
            GAUGE_RIGHT,
            GAUGE_BOTTOM,
            fill="#f0f0f0",
            outline="#666666",
        )
        self.current_bar = self.gauge.create_rectangle(
            GAUGE_LEFT + 1,
            GAUGE_BOTTOM,
            GAUGE_RIGHT - 1,
            GAUGE_BOTTOM - 1,
            fill="#3579b8",
            outline="",
        )
        self.target_line = self.gauge.create_line(
            GAUGE_LEFT - 8,
            GAUGE_BOTTOM,
            GAUGE_RIGHT + 8,
            GAUGE_BOTTOM,
            fill="#202020",
            width=2,
            state="hidden",
        )
        for tick_n in range(0, int(self.force_gauge_max_n) + 1, 5):
            tick_y = self._gauge_y(float(tick_n))
            self.gauge.create_line(
                GAUGE_RIGHT, tick_y, GAUGE_RIGHT + 5, tick_y, fill="#555555"
            )
            self.gauge.create_text(
                GAUGE_RIGHT + 8,
                tick_y,
                text=f"{tick_n:g}",
                anchor="w",
                fill="#404040",
                font=("TkDefaultFont", 8),
            )
        self.gauge_current_text = self.gauge.create_text(
            108,
            48,
            text="Actual\n-- N",
            anchor="w",
            fill="#245a8d",
            font=("TkDefaultFont", 10, "bold"),
        )
        self.gauge_target_text = self.gauge.create_text(
            108,
            103,
            text="Target\n-- N",
            anchor="w",
            fill="#202020",
            font=("TkDefaultFont", 10),
        )
        self.gauge_error_text = self.gauge.create_text(
            108,
            158,
            text="Error\n-- N",
            anchor="w",
            fill="#7b3d2b",
            font=("TkDefaultFont", 10),
        )
        if self.mock_sensor:
            assert self.mock_force is not None
            ttk.Scale(
                live,
                from_=0.0,
                to=self.force_gauge_max_n,
                variable=self.mock_force,
            ).grid(row=3, column=0, sticky="ew", pady=(7, 0))
            ttk.Label(live, text="Manual MOCK force").grid(row=4, column=0, sticky="w")

        status = ttk.LabelFrame(side, text="Continuous trajectory", padding=8)
        status.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.cycle_text = tk.StringVar(value="Cycle: —")
        self.phase_text = tk.StringVar(value="Phase: —")
        self.target_text = tk.StringVar(value="Target: — N")
        self.actual_text = tk.StringVar(value="Actual: — N")
        self.error_text = tk.StringVar(value="Error: — N")
        for row, variable in enumerate(
            (
                self.cycle_text,
                self.phase_text,
                self.target_text,
                self.actual_text,
                self.error_text,
            )
        ):
            ttk.Label(status, textvariable=variable, font=("TkFixedFont", 11)).grid(
                row=row, column=0, sticky="w"
            )
        self.state_text = tk.StringVar(value="Starting sensor tare…")
        ttk.Label(
            status,
            textvariable=self.state_text,
            font=("TkDefaultFont", 10, "bold"),
            wraplength=350,
        ).grid(row=5, column=0, sticky="w", pady=(5, 0))

        self.unloaded_text = tk.StringVar(value="Unloaded captures: 0")
        ttk.Label(side, textvariable=self.unloaded_text).grid(
            row=4, column=0, sticky="w", pady=(7, 0)
        )
        controls = ttk.Frame(side)
        controls.grid(row=5, column=0, sticky="ew", pady=(7, 0))
        self.start_button = ttk.Button(
            controls, text="START SERIES", command=self._start_series
        )
        self.start_button.grid(row=0, column=0, padx=(0, 4))
        self.unloaded_button = ttk.Button(
            controls, text="CAPTURE UNLOADED", command=self._start_unloaded
        )
        self.unloaded_button.grid(row=0, column=1, padx=4)
        self.abort_button = ttk.Button(controls, text="ABORT", command=self._abort)
        self.abort_button.grid(row=0, column=2, padx=4)
        self.tare_button = ttk.Button(controls, text="TARE", command=self._begin_tare)
        self.tare_button.grid(row=0, column=3, padx=(4, 0))
        self.io_text = tk.StringVar(value="Writer: waiting for tare")
        ttk.Label(side, textvariable=self.io_text, wraplength=360).grid(
            row=6, column=0, sticky="w", pady=(7, 0)
        )

    def _set_inputs_enabled(self, enabled: bool) -> None:
        for widget in self.session_widgets:
            widget.configure(
                state="normal" if enabled and self.writer is None else "disabled"
            )
        for widget in self.condition_widgets:
            widget.configure(
                state="normal"
                if enabled and self.writer is not None and self.active_mode is None
                else "disabled"
            )
        create = (
            enabled
            and not self.tare_in_progress
            and self.writer is None
            and self.current_tare_offsets is not None
        )
        controls = (
            enabled
            and self.writer is not None
            and not self.tare_in_progress
            and self.active_mode is None
        )
        self.create_button.configure(state="normal" if create else "disabled")
        self.start_button.configure(state="normal" if controls else "disabled")
        self.unloaded_button.configure(state="normal" if controls else "disabled")
        self.tare_button.configure(
            state="normal"
            if enabled and self.active_mode is None and not self.tare_in_progress
            else "disabled"
        )
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

        threading.Thread(target=work, name="contact_history_tare", daemon=True).start()

    def _poll_tare(self) -> None:
        try:
            result = self.tare_results.get_nowait()
        except queue.Empty:
            return
        self.tare_in_progress = False
        if isinstance(result, BaseException):
            self.current_tare_offsets = None
            self.state_text.set(f"TARE FAILED: {result}")
            self._set_inputs_enabled(True)
            if not self.mock_smoke:
                messagebox.showerror("Bota Rokubi tare failed", str(result))
            return
        self.current_tare_offsets = result
        if self.writer is None:
            self.state_text.set("TARE COMPLETE — create a specimen session")
        else:
            self.writer.update_tare_offsets(result)
            self.state_text.set("TARE COMPLETE — session metadata updated")
        self._set_inputs_enabled(True)

    def _create_session(self) -> None:
        if self.writer is not None or self.current_tare_offsets is None:
            return
        try:
            metadata = HistorySessionMetadata(
                material=self.material.get(),
                morphology=self.morphology.get(),
                specimen_id=self.specimen_id.get(),
                camera_model=self.camera.device_name or "Intel RealSense",
                camera_serial_number=self.camera.serial_number,
                camera_width=self.camera.width,
                camera_height=self.camera.height,
                camera_fps=self.camera.fps,
                camera_exposure_us=self._camera_setting("exposure_us"),
                camera_gain=self._camera_setting("gain"),
                camera_white_balance_k=self._camera_setting("white_balance_k"),
                bota_serial_port=self.sensor.port,
                bota_tare_offsets=self.current_tare_offsets,
                trajectory=self.config,
                sensor_mode="mock" if self.mock_sensor else "physical",
                bota_model="Mock Rokubi" if self.mock_sensor else "Bota Rokubi",
            )
            self.writer = HistoryDatasetWriter(
                self.output_root,
                metadata,
                frame_queue_capacity=WRITER_FRAME_CAPACITY,
            )
        except (OSError, RuntimeError, ValueError) as error:
            if self.mock_smoke:
                raise
            messagebox.showerror("Cannot create session", str(error))
            return
        self.state_text.set("READY — configure a continuous history run")
        self.io_text.set(f"Session: {self.writer.session_path}")
        self._set_inputs_enabled(True)

    def _camera_setting(self, name: str) -> float:
        value = getattr(self.camera, name)
        if value is None:
            raise RuntimeError(f"camera {name} was not fixed before session creation")
        return float(value)

    def _conditions(self) -> tuple[str, int]:
        indenter = self.indenter.get().strip()
        hole = int(self.hole.get())
        if indenter not in {"sphere_10mm", "sphere_30mm"}:
            raise ValueError("indenter must be 10 mm or 30 mm")
        if hole not in range(1, 7):
            raise ValueError("hole number must be from 1 through 6")
        return indenter, hole

    def _start_series(self) -> None:
        if self.writer is None or self.active_mode is not None:
            return
        try:
            condition = self._conditions()
            count = int(self.series_run_count.get())
            if count not in range(1, 101):
                raise ValueError("runs in series must be from 1 through 100")
        except (TypeError, ValueError, tk.TclError) as error:
            if self.mock_smoke:
                raise
            messagebox.showerror("Cannot start series", str(error))
            return
        self.series_condition = condition
        self.series_total = count
        self.series_completed = 0
        self._begin_run()

    def _begin_run(self) -> None:
        assert self.writer is not None
        assert self.series_condition is not None
        indenter, hole = self.series_condition
        self.active_run = self.writer.start_run(indenter=indenter, hole_index=hole)
        self.trajectory = ForceTrajectoryController(self.config)
        self.last_trajectory_update = None
        self.active_mode = "trajectory"
        self.run_camera_drop_start = self.camera_reader.dropped_frame_count
        self.repetition_text.set(str(self.active_run.metadata.repetition_index))
        self.series_text.set(
            f"{self.series_completed + 1} / {self.series_total} active"
        )
        self.state_text.set(f"Bring force to {self.config.min_force_n:g} N for preload")
        self._set_inputs_enabled(False)

    def _start_unloaded(self) -> None:
        if self.writer is None or self.active_mode is not None:
            return
        unloaded_config = ForceSequenceConfig(
            capture_rate_hz=self.config.capture_rate_hz
        )
        self.unloaded = UnloadedCaptureController(unloaded_config)
        self.active_mode = "unloaded"
        self.active_unloaded = None
        self.state_text.set(
            f"Remove load and hold at ≤ {unloaded_config.unloaded_max_force_n:g} N"
        )
        self._set_inputs_enabled(False)

    def _abort(self) -> None:
        if self.active_mode is None or self.writer is None:
            return
        now = perf_counter()
        if self.active_mode == "trajectory" and self.active_run is not None:
            assert self.trajectory is not None
            if self.trajectory.state is not ForceTrajectoryState.COMPLETE:
                self.trajectory.abort(now)
            self.writer.abort_run(self.active_run)
            self.state_text.set("SERIES ABORTED — incomplete run discarded")
        elif self.active_mode == "unloaded":
            if (
                self.unloaded is not None
                and self.unloaded.state is not UnloadedCaptureState.COMPLETE
            ):
                self.unloaded.abort(now)
            if self.active_unloaded is not None:
                self.writer.discard_unloaded_capture(self.active_unloaded)
            self.state_text.set("UNLOADED CAPTURE ABORTED")
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
        for timed in frames:
            self._process_frame(timed)
        if frames:
            self._show_preview(frames[-1].frame)
        self._update_live_force()
        self._update_io_status()
        self.root.after(GUI_REFRESH_MS, self._tick)

    def _process_frame(self, timed: _TimedColorFrame) -> None:
        if self.active_mode is None or self.writer is None:
            return
        sample = self.sensor.nearest_sample(timed.host_time_s)
        if sample is None:
            return
        frame = HistorySynchronizedFrame(
            rgb=timed.frame.rgb,
            camera_host_time_s=timed.host_time_s,
            camera_device_timestamp_ms=timed.frame.timestamp_ms,
            camera_frame_number=timed.frame.frame_number,
            bota_sample=sample,
        )
        if self.active_mode == "trajectory":
            self._process_trajectory(frame)
        else:
            self._process_unloaded(frame)

    def _process_trajectory(self, frame: HistorySynchronizedFrame) -> None:
        assert self.writer is not None
        assert self.active_run is not None
        assert self.trajectory is not None
        if self.trajectory.state is ForceTrajectoryState.IDLE:
            self.trajectory.start(frame.camera_host_time_s)
        update = self.trajectory.update(
            frame.camera_host_time_s, frame.bota_sample.force_magnitude_n
        )
        self.last_trajectory_update = update
        if update.should_capture_frame:
            self.writer.submit_trajectory_frame(self.active_run, frame, update)
        if update.state is ForceTrajectoryState.COMPLETE:
            start = self.trajectory.trajectory_start_time_s
            end = self.trajectory.trajectory_end_time_s
            assert start is not None and end is not None
            self.writer.complete_run(
                self.active_run,
                trajectory_start_host_time_s=start,
                trajectory_end_host_time_s=end,
                dropped_camera_frame_count=(
                    self.camera_reader.dropped_frame_count - self.run_camera_drop_start
                ),
                missed_capture_deadline_count=self.trajectory.missed_capture_deadlines,
            )
            self.series_completed += 1
            self.series_text.set(
                f"{self.series_completed} / {self.series_total} complete"
            )
            if self.series_completed < self.series_total:
                self._begin_run()
            else:
                self.state_text.set(
                    f"SERIES COMPLETE — {self.series_completed} histories saved"
                )
                self._finish_active()
                if self.mock_smoke:
                    self.root.after(100, self.close)
            return
        self._show_trajectory(update, frame.bota_sample.force_magnitude_n)

    def _process_unloaded(self, frame: HistorySynchronizedFrame) -> None:
        assert self.writer is not None
        assert self.unloaded is not None
        if self.unloaded.state is UnloadedCaptureState.IDLE:
            self.unloaded.start(frame.camera_host_time_s)
        update = self.unloaded.update(
            frame.camera_host_time_s, frame.bota_sample.force_magnitude_n
        )
        if UnloadedCaptureEvent.ATTEMPT_RESET in update.events:
            if self.active_unloaded is not None:
                self.writer.discard_unloaded_capture(self.active_unloaded)
                self.active_unloaded = None
        if UnloadedCaptureEvent.RECORDING_STARTED in update.events:
            expected = self.unloaded.config.expected_unloaded_frame_count
            self.active_unloaded = self.writer.begin_unloaded_capture(expected)
        if update.should_record_frame:
            assert self.active_unloaded is not None
            accepted = self.writer.submit_unloaded_frame(
                self.active_unloaded,
                frame,
                capture_elapsed_s=update.phase_elapsed_s,
            )
            if not accepted:
                self.writer.discard_unloaded_capture(self.active_unloaded)
                self.active_unloaded = None
                self.unloaded.reset_attempt(frame.camera_host_time_s)
                self.state_text.set("Writer overrun — unloaded attempt restarted")
                return
        if UnloadedCaptureEvent.CAPTURE_COMPLETED in update.events:
            assert self.active_unloaded is not None
            self.writer.finalize_unloaded_capture(self.active_unloaded)
            self.active_unloaded = None
            self.unloaded_capture_count += 1
            self.unloaded_text.set(f"Unloaded captures: {self.unloaded_capture_count}")
            self.state_text.set("UNLOADED CAPTURE COMPLETE")
            self._finish_active()
        else:
            self._show_unloaded(update)

    def _show_trajectory(
        self, update: ForceTrajectoryUpdate, actual_force_n: float
    ) -> None:
        state = update.state
        if state is ForceTrajectoryState.WAITING_FOR_PRELOAD:
            self.state_text.set(
                f"Establish preload {self.config.min_force_n:g} ± "
                f"{self.config.preload_tolerance_n:g} N"
            )
        elif state is ForceTrajectoryState.PRELOAD_SETTLING:
            self.state_text.set(
                f"PRELOAD HOLD {update.phase_elapsed_s:.2f} / "
                f"{self.config.preload_settle_s:.2f} s"
            )
        elif state is ForceTrajectoryState.CYCLING:
            self.state_text.set(
                "Follow the moving target; deviations are recorded, not rejected"
            )
        elif state is ForceTrajectoryState.WAITING_FOR_RELEASE:
            self.state_text.set(
                f"FULL RELEASE ≤ {self.config.release_max_force_n:g} N for "
                f"{self.config.release_settle_s:g} s"
            )
        self.actual_text.set(f"Actual: {actual_force_n:6.2f} N")
        if update.cycle_role is None:
            self.cycle_text.set("Cycle: —")
        else:
            role_total = (
                self.config.conditioning_cycles
                if update.cycle_role == "conditioning"
                else self.config.measurement_cycles
            )
            role = update.cycle_role.capitalize()
            self.cycle_text.set(
                f"Cycle: {role} {update.cycle_role_index} / {role_total}"
            )
        phase = update.phase.value.upper().replace("_", " ") if update.phase else "—"
        self.phase_text.set(f"Phase: {phase}")
        self.target_text.set(
            "Target: — N"
            if update.target_force_n is None
            else f"Target: {update.target_force_n:6.2f} N"
        )
        self.error_text.set(
            "Error: — N"
            if update.tracking_error_n is None
            else f"Error: {update.tracking_error_n:+6.2f} N"
        )

    def _show_unloaded(self, update: UnloadedCaptureUpdate) -> None:
        state = update.state
        if state is UnloadedCaptureState.WAITING_FOR_UNLOADED:
            text = "Remove load for unloaded diagnostic"
        elif state is UnloadedCaptureState.SETTLING:
            text = f"UNLOADED HOLD {update.phase_elapsed_s:.2f} s"
        elif state is UnloadedCaptureState.RECORDING:
            text = f"UNLOADED RECORDING {update.phase_elapsed_s:.2f} s"
        else:
            text = state.name
        self.state_text.set(text)

    def _finish_active(self) -> None:
        self.active_mode = None
        self.active_run = None
        self.active_unloaded = None
        self.trajectory = None
        self.last_trajectory_update = None
        self.unloaded = None
        self.series_condition = None
        self.series_total = 0
        self.series_completed = 0
        self._set_inputs_enabled(True)

    def _update_live_force(self) -> None:
        try:
            sample = self.sensor.latest_sample()
        except RuntimeError as error:
            self.state_text.set(f"BOTA ERROR: {error}")
            return
        if sample is None:
            return
        actual = sample.force_magnitude_n
        self.force_text.set(f"F magnitude: {actual:7.3f} N")
        self.fz_text.set(f"Fz:          {sample.fz_n:7.3f} N")
        self.share_text.set(f"|Fz| / F:    {100.0 * sample.fz_share:6.1f} %")
        target = None
        error = None
        if self.last_trajectory_update is not None:
            target = self.last_trajectory_update.target_force_n
            error = self.last_trajectory_update.tracking_error_n
        elif self.active_mode == "unloaded":
            target = 1.0
            error = actual - target
        self._update_gauge(actual, target, error)

    def _gauge_y(self, force_n: float) -> float:
        clipped = min(max(force_n, 0.0), self.force_gauge_max_n)
        return GAUGE_BOTTOM - (clipped / self.force_gauge_max_n) * (
            GAUGE_BOTTOM - GAUGE_TOP
        )

    def _update_gauge(
        self, actual_force_n: float, target_force_n: float | None, error_n: float | None
    ) -> None:
        actual_y = self._gauge_y(actual_force_n)
        self.gauge.coords(
            self.current_bar,
            GAUGE_LEFT + 1,
            actual_y,
            GAUGE_RIGHT - 1,
            GAUGE_BOTTOM - 1,
        )
        self.gauge.itemconfigure(
            self.gauge_current_text, text=f"Actual\n{actual_force_n:.2f} N"
        )
        if target_force_n is None:
            self.gauge.itemconfigure(self.target_line, state="hidden")
            self.gauge.itemconfigure(self.gauge_target_text, text="Target\n-- N")
            self.gauge.itemconfigure(self.gauge_error_text, text="Error\n-- N")
            return
        target_y = self._gauge_y(target_force_n)
        self.gauge.coords(
            self.target_line,
            GAUGE_LEFT - 8,
            target_y,
            GAUGE_RIGHT + 8,
            target_y,
        )
        self.gauge.itemconfigure(self.target_line, state="normal")
        self.gauge.itemconfigure(
            self.gauge_target_text, text=f"Target\n{target_force_n:.2f} N"
        )
        self.gauge.itemconfigure(
            self.gauge_error_text,
            text="Error\n-- N" if error_n is None else f"Error\n{error_n:+.2f} N",
        )

    def _update_io_status(self) -> None:
        if self.writer is None:
            return
        if self.writer.error is not None:
            self.state_text.set(f"WRITER ERROR: {self.writer.error}")
        self.io_text.set(
            f"Session: {self.writer.session_path}\n"
            f"Dropped: camera={self.camera_reader.dropped_frame_count}, "
            f"writer={self.writer.dropped_frame_count}"
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

    def _drive_mock_smoke(self) -> None:
        if self.closed or not self.mock_smoke:
            return
        assert self.mock_force is not None
        if self.current_tare_offsets is not None and self.writer is None:
            self._create_session()
            self._start_unloaded()
            self._smoke_unloaded_requested = True
        if (
            self._smoke_unloaded_requested
            and not self._smoke_started
            and self.active_mode is None
            and self.unloaded_capture_count == 1
        ):
            self.series_run_count.set(2)
            self._start_series()
            self._smoke_started = True
        if self.active_mode == "unloaded":
            self.mock_force.set(0.0)
        if self._smoke_started and self.active_mode == "trajectory":
            if self.trajectory is None:
                self.mock_force.set(self.config.min_force_n)
            elif self.trajectory.state in {
                ForceTrajectoryState.IDLE,
                ForceTrajectoryState.WAITING_FOR_PRELOAD,
                ForceTrajectoryState.PRELOAD_SETTLING,
            }:
                self.mock_force.set(self.config.min_force_n)
            elif self.trajectory.state is ForceTrajectoryState.CYCLING:
                update = self.last_trajectory_update
                target = (
                    self.config.min_force_n if update is None else update.target_force_n
                )
                self.mock_force.set(
                    self.config.min_force_n if target is None else target
                )
            elif self.trajectory.state is ForceTrajectoryState.WAITING_FOR_RELEASE:
                self.mock_force.set(0.0)
        self.root.after(10, self._drive_mock_smoke)

    def close(self) -> None:
        if self.closed:
            return
        if self.active_mode is not None:
            self._abort()
        self.closed = True
        self.camera_reader.stop()
        self.camera.stop()
        self.sensor.stop()
        if self.writer is not None:
            try:
                self.writer.close()
            except RuntimeError as error:
                if not self.mock_smoke:
                    messagebox.showerror("History writer error", str(error))
        self.root.destroy()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_REPOSITORY_ROOT / "output" / "contact_history",
    )
    parser.add_argument("--bota-port", default="/dev/ttyUSB0")
    parser.add_argument("--min-force-n", type=float, default=2.0)
    parser.add_argument("--max-force-n", type=float, default=15.0)
    parser.add_argument("--ramp-rate-n-per-s", type=float, default=1.0)
    parser.add_argument("--low-dwell-s", type=float, default=1.0)
    parser.add_argument("--high-dwell-s", type=float, default=1.0)
    parser.add_argument("--conditioning-cycles", type=int, default=2)
    parser.add_argument("--measurement-cycles", type=int, default=5)
    parser.add_argument("--capture-rate-hz", type=float, default=5.0)
    parser.add_argument("--exposure-us", type=float, default=1500.0)
    parser.add_argument("--gain", type=float, default=0.0)
    parser.add_argument("--white-balance-k", type=float, default=4600.0)
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-serial")
    parser.add_argument("--preload-tolerance-n", type=float, default=1.0)
    parser.add_argument("--preload-settle-s", type=float, default=0.5)
    parser.add_argument("--release-max-force-n", type=float, default=1.0)
    parser.add_argument("--release-settle-s", type=float, default=0.5)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mock-smoke", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mock_smoke and not args.mock:
        parser.error("--mock-smoke requires --mock")
    return args


def main() -> None:
    args = _arguments()
    config = ForceTrajectoryConfig(
        min_force_n=args.min_force_n,
        max_force_n=args.max_force_n,
        ramp_rate_n_per_s=args.ramp_rate_n_per_s,
        low_dwell_s=args.low_dwell_s,
        high_dwell_s=args.high_dwell_s,
        conditioning_cycles=args.conditioning_cycles,
        measurement_cycles=args.measurement_cycles,
        preload_tolerance_n=args.preload_tolerance_n,
        preload_settle_s=args.preload_settle_s,
        release_max_force_n=args.release_max_force_n,
        release_settle_s=args.release_settle_s,
        capture_rate_hz=args.capture_rate_hz,
    )
    root = tk.Tk()
    mock_force = tk.DoubleVar(master=root, value=0.0)
    camera: Camera
    if args.mock:
        camera = _MockColorCamera(
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        )
    else:
        camera = RealSenseColorCamera(
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            serial_number=args.camera_serial,
        )
    sensor: Sensor = (
        _MockBotaSensor(mock_force.get)
        if args.mock
        else BotaSerialSensor(args.bota_port)
    )
    try:
        camera.start()
        camera.set_manual_photometric_controls(
            exposure_us=args.exposure_us,
            gain=args.gain,
            white_balance_k=args.white_balance_k,
        )
        sensor.start()
    except BaseException:
        camera.stop()
        sensor.stop()
        root.destroy()
        raise
    reader = _CameraReader(camera)
    reader.start()
    app = ContactHistoryApp(
        root,
        camera=camera,
        camera_reader=reader,
        sensor=sensor,
        output_root=args.output_root,
        config=config,
        mock_sensor=args.mock,
        mock_force_variable=mock_force if args.mock else None,
        mock_smoke=args.mock_smoke,
    )
    try:
        root.mainloop()
    finally:
        if not app.closed:
            app.close()


if __name__ == "__main__":
    main()
