"""NiceGUI dashboard for bounded AK40-10 torque commands and shaft feedback."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from time import monotonic

from nicegui import app, ui


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiments.actuation import AK40TorqueSession  # noqa: E402
from experiments.hardware.ak40_10 import AK40_10, T_MAX  # noqa: E402
from experiments.hardware.can_io import CanIO  # noqa: E402


UI_REFRESH_S = 0.02


def _motor_id(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid motor ID {value!r}; use decimal 13 or hexadecimal 0x0D"
        ) from exc
    if not 0 <= parsed <= 0xFF:
        raise argparse.ArgumentTypeError("motor ID must be an unsigned 8-bit drive ID")
    return parsed


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid number {value!r}") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


class TorqueDashboard:
    """Render controls and immutable snapshots from the headless torque session."""

    def __init__(self, session: AK40TorqueSession) -> None:
        self._session = session
        self._running = False
        self._last_fault: str | None = None

        ui.colors(primary="#2C758E", positive="#009E73", negative="#D62728")
        ui.page_title("AK40-10 torque and state monitor")

        with ui.column().classes("w-full max-w-5xl mx-auto gap-3 p-4"):
            ui.label("CubeMars AK40-10 torque and state monitor").classes(
                "text-2xl font-medium"
            )
            ui.label(
                "The clock hand follows measured shaft position. "
                "Closing or reconnecting the page stops and disables the actuator."
            ).classes("text-sm text-gray-600")

            with ui.row().classes("w-full items-stretch gap-4 no-wrap"):
                with ui.card().classes("w-[430px] items-center"):
                    ui.label("Measured shaft angle").classes("text-lg")
                    self._clock = ui.echart(self._clock_options()).classes(
                        "w-[400px] h-[400px]"
                    )
                    self._angle_label = ui.label("No feedback").classes(
                        "text-base font-mono"
                    )

                with ui.column().classes("grow gap-4"):
                    with ui.card().classes("w-full"):
                        ui.label("Motor feedback").classes("text-lg")
                        self._state_labels: dict[str, object] = {}
                        self._state_row("Position", "position", "-- rad")
                        self._state_row("Velocity", "velocity", "-- rad/s")
                        self._state_row("Torque", "torque", "-- N m")
                        self._state_row("Temperature", "temperature", "-- degC")
                        self._state_row("Drive error", "error", "--")
                        self._state_row("Feedback", "feedback", "0 frames")

                    with ui.card().classes("w-full"):
                        ui.label("Torque command").classes("text-lg")
                        self._target_label = ui.label("Target: +0.000 N m").classes(
                            "font-mono"
                        )
                        display_limit = max(session.max_abs_torque_nm, 0.1)
                        self._slider = ui.slider(
                            min=-display_limit,
                            max=display_limit,
                            step=max(display_limit / 200.0, 0.001),
                            value=0.0,
                            on_change=self._torque_changed,
                        ).props("label-always")
                        self._slider.classes("w-full")
                        self._slider.disable()
                        with ui.row().classes("w-full justify-between text-xs text-gray-600"):
                            ui.label(f"-{session.max_abs_torque_nm:g} N m")
                            ui.label("operator limit")
                            ui.label(f"+{session.max_abs_torque_nm:g} N m")

                    with ui.card().classes("w-full"):
                        with ui.row().classes("w-full gap-2"):
                            self._start_button = ui.button(
                                "START", icon="play_arrow", on_click=self._start
                            )
                            self._zero_button = ui.button(
                                "ZERO TORQUE",
                                icon="exposure_zero",
                                on_click=self._zero_torque,
                                color="warning",
                            )
                            self._stop_button = ui.button(
                                "STOP / DISABLE",
                                icon="stop",
                                on_click=self._stop,
                                color="negative",
                            )
                            self._zero_button.disable()
                            self._stop_button.disable()
                        self._status = ui.label(
                            "Inactive; motor control is disabled"
                        ).classes("text-sm text-gray-700")

        ui.timer(UI_REFRESH_S, self._refresh)

    @staticmethod
    def _clock_options() -> dict:
        return {
            "animationDurationUpdate": 0,
            "series": [
                {
                    "type": "gauge",
                    "min": -180,
                    "max": 180,
                    "startAngle": 90,
                    "endAngle": -270,
                    "splitNumber": 12,
                    "axisLine": {
                        "lineStyle": {"width": 5, "color": [[1, "#59616B"]]}
                    },
                    "axisTick": {
                        "distance": -13,
                        "length": 7,
                        "lineStyle": {"color": "#59616B", "width": 1},
                    },
                    "splitLine": {
                        "distance": -16,
                        "length": 13,
                        "lineStyle": {"color": "#30343B", "width": 2},
                    },
                    "axisLabel": {"show": False},
                    "pointer": {"length": "73%", "width": 7},
                    "itemStyle": {"color": "#D62728"},
                    "anchor": {
                        "show": True,
                        "showAbove": True,
                        "size": 15,
                        "itemStyle": {"color": "#30343B"},
                    },
                    "title": {"show": False},
                    "detail": {"show": False},
                    "data": [{"value": 0.0}],
                }
            ],
        }

    def _state_row(self, title: str, key: str, initial: str) -> None:
        with ui.row().classes("w-full justify-between gap-8"):
            ui.label(title).classes("text-gray-600")
            self._state_labels[key] = ui.label(initial).classes("font-mono")

    def _start(self) -> None:
        self._running = False
        self._slider.set_value(0.0)
        self._target_label.set_text("Target: +0.000 N m")
        try:
            self._session.start()
        except Exception as exc:
            self._show_fault(str(exc))
            return

        self._running = True
        self._last_fault = None
        self._start_button.disable()
        self._stop_button.enable()
        self._zero_button.enable()
        if self._session.max_abs_torque_nm > 0.0:
            self._slider.enable()
            self._set_status("Active; operator torque limit is enabled", "#146B3A")
        else:
            self._set_status("Active in zero-torque monitor-only mode", "#146B3A")

    def _stop(self) -> None:
        self._running = False
        self._slider.set_value(0.0)
        try:
            self._session.stop()
        except Exception as exc:
            self._show_fault(str(exc))
            return
        self._set_inactive_controls()
        self._set_status(
            "Stopped; zero torque sent and motor control disabled", "#333333"
        )

    def _zero_torque(self) -> None:
        self._slider.set_value(0.0)
        if not self._running:
            return
        try:
            self._session.set_target_torque(0.0)
        except Exception as exc:
            self._show_fault(str(exc))

    def _torque_changed(self, event: object) -> None:
        value = getattr(event, "value", 0.0)
        torque_nm = 0.0 if value is None else float(value)
        self._target_label.set_text(f"Target: {torque_nm:+.3f} N m")
        if not self._running:
            return
        try:
            self._session.set_target_torque(torque_nm)
        except Exception as exc:
            self._running = False
            self._slider.set_value(0.0)
            self._show_fault(str(exc))

    def _refresh(self) -> None:
        snapshot = self._session.snapshot()
        state = snapshot.state
        if state is not None:
            self._state_labels["position"].set_text(f"{state.position:+.6f} rad")
            self._state_labels["velocity"].set_text(f"{state.velocity:+.6f} rad/s")
            self._state_labels["torque"].set_text(f"{state.torque:+.6f} N m")
            self._state_labels["temperature"].set_text(
                f"{state.temperature:.1f} degC"
            )
            self._state_labels["error"].set_text(str(state.error))
            age_s = 0.0
            if snapshot.feedback_monotonic_s is not None:
                age_s = max(0.0, monotonic() - snapshot.feedback_monotonic_s)
            self._state_labels["feedback"].set_text(
                f"{snapshot.feedback_count} frames; age {age_s * 1000.0:.0f} ms"
            )
            self._update_clock(state.position)

        if snapshot.fault is not None and snapshot.fault != self._last_fault:
            self._last_fault = snapshot.fault
            self._show_fault(snapshot.fault)
            self._set_inactive_controls()
            ui.notify(snapshot.fault, type="negative", close_button=True)
        elif self._running and not snapshot.active:
            self._set_inactive_controls()
            self._set_status("Stopped; motor control disabled", "#333333")

    def _update_clock(self, position_rad: float) -> None:
        degrees = math.degrees(position_rad)
        wrapped_degrees = (degrees + 180.0) % 360.0 - 180.0
        self._clock.options["series"][0]["data"][0]["value"] = wrapped_degrees
        self._clock.update()
        self._angle_label.set_text(f"{position_rad:+.3f} rad  |  {degrees:+.1f} deg")

    def _set_inactive_controls(self) -> None:
        self._running = False
        self._slider.set_value(0.0)
        self._slider.disable()
        self._start_button.enable()
        self._zero_button.disable()
        self._stop_button.disable()

    def _show_fault(self, message: str) -> None:
        self._set_status(f"FAULT: {message}", "#B00020")

    def _set_status(self, message: str, color: str) -> None:
        self._status.set_text(message)
        self._status.style(f"color: {color}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="can0", help="SocketCAN channel")
    parser.add_argument("--motor-id", type=_motor_id, default=13, help="default: 13")
    parser.add_argument(
        "--max-abs-torque-nm",
        type=_finite_float,
        default=0.0,
        help="operator-approved absolute torque limit; default 0 is monitor-only",
    )
    parser.add_argument("--rate-hz", type=_finite_float, default=50.0)
    parser.add_argument("--feedback-timeout-s", type=_finite_float, default=0.1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.max_abs_torque_nm <= T_MAX:
        parser.error(f"--max-abs-torque-nm must be between 0 and {T_MAX:g}")
    if args.rate_hz <= 0.0:
        parser.error("--rate-hz must be positive")
    if args.feedback_timeout_s <= 0.0:
        parser.error("--feedback-timeout-s must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def main() -> None:
    args = _parse_args()
    can_io = CanIO(args.channel)
    motor = AK40_10(can_io, motor_id=args.motor_id)
    session = AK40TorqueSession(
        motor,
        max_abs_torque_nm=args.max_abs_torque_nm,
        command_rate_hz=args.rate_hz,
        feedback_timeout_s=args.feedback_timeout_s,
    )

    def build_page() -> None:
        TorqueDashboard(session)

    def stop_hardware() -> None:
        try:
            session.stop()
        finally:
            can_io.close()

    app.on_disconnect(session.stop)
    app.on_shutdown(stop_hardware)
    try:
        ui.run(
            root=build_page,
            host=args.host,
            port=args.port,
            title="AK40-10 torque monitor",
            show=not args.no_browser,
            reload=False,
        )
    finally:
        stop_hardware()


if __name__ in {"__main__", "__mp_main__"}:
    main()
