"""Run the NiceGUI proprioceptive force-sensing experiment console."""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
import math
from pathlib import Path
import sys
from time import monotonic_ns

from nicegui import app, run, ui
from starlette.responses import StreamingResponse


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiments.data_collection import ProprioceptiveExperimentRuntime  # noqa: E402
from experiments.hardware import BotaSerialSensor, RealSenseColorCamera  # noqa: E402
from experiments.hardware.ak40_10 import AK40_10, KD_MAX, KP_MAX  # noqa: E402
from experiments.hardware.can_io import CanIO  # noqa: E402


UI_REFRESH_S = 0.1
PLOT_WINDOW_S = 15.0


def _motor_id(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "motor ID must be decimal 13 or hexadecimal 0x0D"
        ) from error
    if not 0 <= parsed <= 0xFF:
        raise argparse.ArgumentTypeError("motor ID must be an unsigned 8-bit value")
    return parsed


class ExperimentConsole:
    """Read runtime snapshots and issue only explicit high-level actions."""

    def __init__(self, runtime: ProprioceptiveExperimentRuntime) -> None:
        self._runtime = runtime
        self._last_motor_timestamp_ns: int | None = None
        self._last_ft_timestamp_ns: int | None = None
        self._last_optical_timestamp_ns: int | None = None
        self._plot_start_ns = monotonic_ns()
        self._torque_points: deque[list[float]] = deque()
        self._force_points: deque[list[float]] = deque()
        self._location_points: deque[list[float]] = deque()

        ui.colors(primary="#2C758E", positive="#009E73", negative="#D62728")
        ui.page_title("LUMO proprioceptive force experiment")
        with ui.column().classes("w-full max-w-[1500px] mx-auto gap-3 p-4"):
            ui.label("LUMO proprioceptive force experiment").classes(
                "text-2xl font-medium"
            )
            ui.label(
                "Fixed MIT impedance hold with independent motor, Rokubi, camera, "
                "and optical acquisition. Motor control never starts automatically. "
                f"Contact processing: {self._runtime.camera.contact_processing_mode}."
            ).classes("text-sm text-gray-600")
            self._build_status()
            with ui.row().classes("w-full items-start gap-3 no-wrap"):
                self._build_camera()
                with ui.column().classes("w-[480px] gap-3"):
                    self._build_motor()
                    self._build_ft()
                    self._build_estimator()
            self._build_recording()
            self._build_plots()
        ui.timer(UI_REFRESH_S, self._refresh)

    def _build_status(self) -> None:
        with ui.card().classes("w-full"):
            ui.label("System status").classes("text-lg font-medium")
            self._status_labels: dict[str, object] = {}
            with ui.row().classes("w-full gap-5"):
                for key, title in (
                    ("can", "CAN"),
                    ("motor", "Motor"),
                    ("camera", "Camera"),
                    ("ft", "F/T"),
                    ("recorder", "Recorder"),
                ):
                    with ui.column().classes("gap-0 min-w-[150px]"):
                        ui.label(title).classes("text-xs text-gray-500")
                        self._status_labels[key] = ui.label("disconnected").classes(
                            "font-mono"
                        )
            self._fault_label = ui.label("").classes("text-sm text-red-700")

    def _build_camera(self) -> None:
        with ui.card().classes("grow min-w-0"):
            ui.label("Live camera and optical contact").classes(
                "text-lg font-medium"
            )
            ui.html(
                '<img src="/proprioceptive-camera.mjpeg" '
                'style="width:100%;max-height:540px;object-fit:contain;'
                'background:#000;display:block" alt="Live camera">',
                sanitize=False,
            ).classes(
                "w-full overflow-hidden bg-black"
            )
            with ui.row().classes("w-full justify-between gap-4"):
                self._contact_label = ui.label("Contact: unavailable")
                self._location_label = ui.label("Location: unavailable")
                self._confidence_label = ui.label("Score: unavailable")
            self._detector_label = ui.label("Detector: initializing").classes(
                "text-sm text-gray-600"
            )
            with ui.row().classes("gap-2"):
                recalibrate_button = ui.button(
                    "Recalibrate geometry",
                    icon="center_focus_strong",
                    on_click=lambda: self._action(
                        self._runtime.recalibrate_contact,
                        "Geometry recalibration requested",
                    ),
                ).props("outline")
                baseline_button = ui.button(
                    "Acquire unloaded baseline",
                    icon="exposure_zero",
                    on_click=lambda: self._action(
                        self._runtime.acquire_unloaded_baseline,
                        "Keep the fingertip unloaded for 30 frames",
                    ),
                )
                if not self._runtime.camera.online_contact_enabled:
                    recalibrate_button.props("disable")
                    baseline_button.props("disable")

    def _build_motor(self) -> None:
        with ui.card().classes("w-full"):
            ui.label("AK40-10 fixed impedance").classes("text-lg font-medium")
            self._motor_labels: dict[str, object] = {}
            for key, title, initial in (
                ("q", "q", "-- rad"),
                ("dq", "dq", "-- rad/s"),
                ("torque", "Torque", "-- N m"),
                ("temperature", "Temperature", "-- C"),
                ("error", "Error code", "--"),
            ):
                with ui.row().classes("w-full justify-between"):
                    ui.label(title).classes("text-gray-600")
                    self._motor_labels[key] = ui.label(initial).classes("font-mono")
            with ui.row().classes("w-full gap-3"):
                self._kp_input = ui.number(
                    "Kp",
                    value=self._runtime.motor.gains[0],
                    min=0.0,
                    max=KP_MAX,
                    step=1.0,
                ).classes("w-1/2")
                self._kd_input = ui.number(
                    "Kd",
                    value=self._runtime.motor.gains[1],
                    min=0.0,
                    max=KD_MAX,
                    step=0.01,
                ).classes("w-1/2")
            ui.button("Apply Kp / Kd", on_click=self._apply_gains).props("outline")
            with ui.column().classes("gap-0 text-sm text-gray-600"):
                ui.label("Position target = 0.0 rad")
                ui.label("Velocity target = 0.0 rad/s")
                ui.label("Feed-forward torque = 0.0 N m")
            with ui.row().classes("w-full gap-2"):
                ui.button(
                    "Set Motor Zero Position",
                    icon="exposure_zero",
                    on_click=lambda: self._action(
                        self._runtime.set_zero,
                        "Set-zero command requested",
                    ),
                    color="warning",
                )
                ui.button(
                    "Enable",
                    icon="play_arrow",
                    on_click=lambda: self._action(
                        self._runtime.enable_motor,
                        "Motor enable requested",
                    ),
                    color="positive",
                )
                ui.button(
                    "Disable",
                    icon="stop",
                    on_click=lambda: self._action(
                        self._runtime.disable_motor,
                        "Motor disable requested",
                    ),
                    color="negative",
                ).classes("grow")

    def _build_ft(self) -> None:
        with ui.card().classes("w-full"):
            ui.label("Bota Rokubi ground truth").classes("text-lg font-medium")
            self._normal_force_label = ui.label(
                "Ground-truth normal force: unavailable"
            ).classes("text-xl font-medium")
            self._ft_label = ui.label(
                "Fx -- | Fy -- | Fz -- N\nTx -- | Ty -- | Tz -- N m"
            ).classes("font-mono whitespace-pre-line text-sm")
            ui.button(
                "Tare F/T sensor",
                icon="balance",
                on_click=lambda: self._action(
                    self._runtime.tare_ft,
                    "F/T tare requested; keep the sensor unloaded",
                ),
            ).props("outline")

    def _build_estimator(self) -> None:
        with ui.card().classes("w-full"):
            ui.label("Force estimation").classes("text-lg font-medium")
            self._estimated_force = ui.label("Estimated force: not calibrated")
            self._ground_truth = ui.label("Ground truth: unavailable")
            self._estimation_error = ui.label("Estimation error: unavailable")

    def _build_recording(self) -> None:
        with ui.card().classes("w-full"):
            ui.label("Data collection").classes("text-lg font-medium")
            with ui.row().classes("w-full gap-3 items-end"):
                self._trial_input = ui.number(
                    "Trial", value=1, min=1, step=1, format="%.0f"
                ).classes("w-28")
                self._gt_location_input = ui.input(
                    "GT contact location [mm] (optional)"
                ).classes("w-64")
                self._notes_input = ui.input("Notes (optional)").classes("grow")
                self._record_button = ui.button(
                    "Start Recording",
                    icon="fiber_manual_record",
                    on_click=self._start_recording,
                    color="negative",
                )
                self._stop_record_button = ui.button(
                    "Stop Recording", icon="stop", on_click=self._stop_recording
                )
            self._record_label = ui.label("Recorder idle").classes("font-mono text-sm")
            ui.label(
                f"Contact PNG: {self._runtime.camera.contact_record_rate_hz:g} Hz "
                "while Rokubi force is at least "
                f"{self._runtime.camera.contact_frame_threshold_n:g} N. "
                + (
                    f"Each run also stores the preceding "
                    f"{self._runtime.camera.offline_reference_frame_count} unloaded "
                    "frames for offline processing."
                    if self._runtime.camera.offline_contact_enabled
                    else "No unloaded reference is stored in online-only mode."
                )
            ).classes("text-xs text-gray-600")

    def _build_plots(self) -> None:
        with ui.row().classes("w-full gap-3 no-wrap"):
            self._torque_chart = self._chart("Motor torque [N m]", "#2C758E")
            self._force_chart = self._chart("Normal force [N]", "#D62728")
            self._location_chart = self._chart("Contact location [mm]", "#009E73")

    @staticmethod
    def _chart(title: str, color: str):
        return ui.echart(
            {
                "animation": False,
                "title": {"text": title, "left": "center", "textStyle": {"fontSize": 14}},
                "grid": {"left": 52, "right": 12, "top": 42, "bottom": 38},
                "xAxis": {"type": "value", "name": "Recent time [s]"},
                "yAxis": {"type": "value", "scale": True},
                "series": [
                    {
                        "type": "line",
                        "showSymbol": False,
                        "lineStyle": {"width": 1.5, "color": color},
                        "data": [],
                    }
                ],
            }
        ).classes("w-1/3 h-56")

    def _apply_gains(self) -> None:
        self._action(
            lambda: self._runtime.set_motor_gains(
                float(self._kp_input.value),
                float(self._kd_input.value),
            ),
            "Kp/Kd updated",
        )

    def _start_recording(self) -> None:
        raw_location = str(self._gt_location_input.value or "").strip()
        try:
            location = None if not raw_location else float(raw_location)
            path = self._runtime.start_recording(
                trial=int(self._trial_input.value),
                contact_location_gt_mm=location,
                notes=str(self._notes_input.value or ""),
            )
        except Exception as error:
            ui.notify(str(error), type="negative", close_button=True)
            return
        ui.notify(f"Recording {path}", type="positive")

    async def _stop_recording(self) -> None:
        try:
            path = await run.io_bound(self._runtime.stop_recording)
        except Exception as error:
            ui.notify(str(error), type="negative", close_button=True)
            return
        ui.notify(f"Saved {path}", type="positive")

    @staticmethod
    def _display_status(label: object, text: str) -> None:
        label.set_text(text)
        if text in {"connected", "enabled", "recording"}:
            label.style("color: #087F5B")
        elif text == "error" or text.startswith("error"):
            label.style("color: #B00020")
        else:
            label.style("color: #4C5055")

    def _action(self, action, success_message: str) -> None:
        try:
            action()
        except Exception as error:
            ui.notify(str(error), type="negative", close_button=True)
            return
        ui.notify(success_message, type="positive")

    def _refresh(self) -> None:
        snapshot = self._runtime.snapshot()
        for key, value in (
            ("can", snapshot.can_status),
            ("motor", snapshot.motor_status),
            ("camera", snapshot.camera_status),
            ("ft", snapshot.ft_status),
            ("recorder", snapshot.recorder.status),
        ):
            self._display_status(self._status_labels[key], value)
        errors = [
            value
            for value in (
                snapshot.motor_error,
                snapshot.camera_error,
                snapshot.ft_error,
                snapshot.recorder.error,
            )
            if value
        ]
        self._fault_label.set_text(" | ".join(errors))

        if snapshot.motor is not None:
            motor = snapshot.motor
            self._motor_labels["q"].set_text(f"{motor.position_rad:+.5f} rad")
            self._motor_labels["dq"].set_text(f"{motor.velocity_rad_s:+.5f} rad/s")
            self._motor_labels["torque"].set_text(f"{motor.torque_nm:+.5f} N m")
            self._motor_labels["temperature"].set_text(
                f"{motor.temperature_c:.1f} C"
            )
            self._motor_labels["error"].set_text(str(motor.error))
            if motor.timestamp_ns != self._last_motor_timestamp_ns:
                self._append_point(self._torque_points, motor.timestamp_ns, motor.torque_nm)
                self._last_motor_timestamp_ns = motor.timestamp_ns

        if snapshot.ft is not None:
            ft = snapshot.ft
            self._ft_label.set_text(
                f"Fx {ft.fx_n:+.3f} | Fy {ft.fy_n:+.3f} | Fz {ft.fz_n:+.3f} N\n"
                f"Tx {ft.tx_nm:+.3f} | Ty {ft.ty_nm:+.3f} | Tz {ft.tz_nm:+.3f} N m"
            )
            if ft.force_normal_n is None:
                self._normal_force_label.set_text(
                    "Ground-truth normal force: axis not configured"
                )
                self._ground_truth.set_text("Ground truth: unavailable")
            else:
                self._normal_force_label.set_text(
                    f"Ground-truth normal force: {ft.force_normal_n:+.3f} N"
                )
                self._ground_truth.set_text(f"Ground truth: {ft.force_normal_n:+.3f} N")
                if ft.timestamp_ns != self._last_ft_timestamp_ns:
                    self._append_point(self._force_points, ft.timestamp_ns, ft.force_normal_n)
            self._last_ft_timestamp_ns = ft.timestamp_ns

        if snapshot.optical is not None:
            optical = snapshot.optical
            detected = optical.contact_detected
            self._contact_label.set_text(
                "Contact: unavailable"
                if detected is None
                else f"Contact: {'yes' if detected else 'no'}"
            )
            self._location_label.set_text(
                "Location: unavailable"
                if optical.contact_location_mm is None
                else f"Location: {optical.contact_location_mm:+.2f} mm"
            )
            self._confidence_label.set_text(
                "Score: unavailable"
                if optical.contact_score_z is None
                else f"Score: {optical.contact_score_z:.2f} z"
            )
            self._detector_label.set_text(f"Detector: {optical.detector_status}")
            if (
                optical.contact_location_mm is not None
                and optical.timestamp_ns != self._last_optical_timestamp_ns
            ):
                self._append_point(
                    self._location_points,
                    optical.timestamp_ns,
                    optical.contact_location_mm,
                )
            self._last_optical_timestamp_ns = optical.timestamp_ns

        recorder = snapshot.recorder
        duration_s = 0.0
        if recorder.started_ns is not None and recorder.status in {"recording", "saving"}:
            duration_s = (monotonic_ns() - recorder.started_ns) / 1.0e9
        self._record_label.set_text(
            f"{recorder.status} | {recorder.run_id or '--'} | {duration_s:.1f} s | "
            f"motor {recorder.motor_samples} | F/T {recorder.ft_samples} | "
            f"camera {recorder.camera_frames} | optical {recorder.optical_samples} | "
            + (
                f"reference {self._runtime.camera.offline_reference_count}/"
                f"{self._runtime.camera.offline_reference_frame_count} | "
                if self._runtime.camera.offline_contact_enabled
                else ""
            )
            + f"{recorder.run_path or '--'}"
        )
        self._update_chart(self._torque_chart, self._torque_points)
        self._update_chart(self._force_chart, self._force_points)
        self._update_chart(self._location_chart, self._location_points)

    def _append_point(
        self,
        points: deque[list[float]],
        timestamp_ns: int,
        value: float,
    ) -> None:
        time_s = (timestamp_ns - self._plot_start_ns) / 1.0e9
        points.append([time_s, float(value)])
        cutoff = time_s - PLOT_WINDOW_S
        while points and points[0][0] < cutoff:
            points.popleft()

    @staticmethod
    def _update_chart(chart, points: deque[list[float]]) -> None:
        chart.options["series"][0]["data"] = list(points)
        chart.update()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--motor-id", type=_motor_id, default=13)
    parser.add_argument("--motor-rate-hz", type=float, default=100.0)
    parser.add_argument("--motor-feedback-timeout-s", type=float, default=0.05)
    parser.add_argument("--kp", type=float, default=0.0)
    parser.add_argument("--kd", type=float, default=0.0)
    parser.add_argument("--bota-port", default="/dev/ttyUSB0")
    parser.add_argument("--normal-axis", choices=("fx", "fy", "fz"))
    parser.add_argument("--normal-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-serial")
    parser.add_argument(
        "--camera-contact-threshold-n",
        type=float,
        default=0.5,
        help="save lossless camera frames only above this Rokubi force (default: 0.5 N)",
    )
    parser.add_argument(
        "--contact-processing",
        choices=("online", "offline", "both"),
        default="both",
        help=(
            "contact processing path: live only, deferred offline only, or both "
            "(default: both)"
        ),
    )
    parser.add_argument(
        "--camera-record-rate-hz",
        type=float,
        default=5.0,
        help="lossless contact-frame rate (default: 5 Hz)",
    )
    parser.add_argument(
        "--offline-reference-frames",
        type=int,
        default=30,
        help="rolling unloaded frames copied into each offline-ready run (default: 30)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/experiments/proprioceptive_force"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.kp <= KP_MAX:
        parser.error(f"--kp must be within [0, {KP_MAX:g}]")
    if not 0.0 <= args.kd <= KD_MAX:
        parser.error(f"--kd must be within [0, {KD_MAX:g}]")
    if args.motor_rate_hz <= 0.0 or args.motor_feedback_timeout_s <= 0.0:
        parser.error("motor rate and feedback timeout must be positive")
    if (
        not math.isfinite(args.camera_contact_threshold_n)
        or args.camera_contact_threshold_n < 0.0
    ):
        parser.error("--camera-contact-threshold-n must be finite and nonnegative")
    if (
        not math.isfinite(args.camera_record_rate_hz)
        or args.camera_record_rate_hz <= 0.0
    ):
        parser.error("--camera-record-rate-hz must be finite and positive")
    if args.offline_reference_frames < 1:
        parser.error("--offline-reference-frames must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def main() -> None:
    args = _parse_args()
    can_io = CanIO(args.channel)
    runtime = ProprioceptiveExperimentRuntime(
        can_io=can_io,
        motor=AK40_10(can_io, motor_id=args.motor_id),
        camera=RealSenseColorCamera(
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            serial_number=args.camera_serial,
        ),
        ft_sensor=BotaSerialSensor(args.bota_port),
        output_root=args.output_root,
        normal_axis=args.normal_axis,
        normal_sign=args.normal_sign,
        kp=args.kp,
        kd=args.kd,
        motor_rate_hz=args.motor_rate_hz,
        motor_feedback_timeout_s=args.motor_feedback_timeout_s,
        camera_contact_threshold_n=args.camera_contact_threshold_n,
        contact_processing_mode=args.contact_processing,
        camera_record_rate_hz=args.camera_record_rate_hz,
        offline_reference_frame_count=args.offline_reference_frames,
    )
    runtime.start()

    @app.get("/proprioceptive-camera.mjpeg")
    async def camera_stream() -> StreamingResponse:
        async def frames():
            last_timestamp_ns: int | None = None
            while True:
                snapshot = runtime.snapshot()
                timestamp_ns = (
                    None
                    if snapshot.optical is None
                    else snapshot.optical.timestamp_ns
                )
                if (
                    snapshot.camera_jpeg is not None
                    and timestamp_ns != last_timestamp_ns
                ):
                    frame = snapshot.camera_jpeg
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                        + frame
                        + b"\r\n"
                    )
                    last_timestamp_ns = timestamp_ns
                await asyncio.sleep(1.0 / 60.0)

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    def build_page() -> None:
        ExperimentConsole(runtime)

    app.on_disconnect(runtime.disable_motor)
    app.on_shutdown(runtime.shutdown)
    try:
        ui.run(
            root=build_page,
            host=args.host,
            port=args.port,
            title="LUMO proprioceptive force experiment",
            show=not args.no_browser,
            reload=False,
        )
    finally:
        runtime.shutdown()


if __name__ in {"__main__", "__mp_main__"}:
    main()
