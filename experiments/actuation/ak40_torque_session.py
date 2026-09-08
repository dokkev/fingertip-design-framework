"""Headless torque-command and feedback session for one AK40-10."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from time import monotonic

from experiments.hardware.ak40_10 import AK40_10, MotorState, T_MAX


@dataclass(frozen=True)
class TorqueSessionSnapshot:
    """One coherent status snapshot for a UI or headless caller."""

    active: bool
    target_torque_nm: float
    state: MotorState | None
    feedback_monotonic_s: float | None
    feedback_count: int
    fault: str | None


class AK40TorqueSession:
    """Run bounded torque commands and feedback reads outside a caller's UI."""

    def __init__(
        self,
        motor: AK40_10,
        *,
        max_abs_torque_nm: float,
        command_rate_hz: float = 50.0,
        feedback_timeout_s: float = 0.1,
    ) -> None:
        max_abs_torque_nm = float(max_abs_torque_nm)
        command_rate_hz = float(command_rate_hz)
        feedback_timeout_s = float(feedback_timeout_s)
        if not math.isfinite(max_abs_torque_nm) or not 0.0 <= max_abs_torque_nm <= T_MAX:
            raise ValueError(f"max_abs_torque_nm must be between 0 and {T_MAX:g}")
        if not math.isfinite(command_rate_hz) or command_rate_hz <= 0.0:
            raise ValueError("command_rate_hz must be positive")
        if not math.isfinite(feedback_timeout_s) or feedback_timeout_s <= 0.0:
            raise ValueError("feedback_timeout_s must be positive")

        self._motor = motor
        self.max_abs_torque_nm = max_abs_torque_nm
        self.command_rate_hz = command_rate_hz
        self.feedback_timeout_s = feedback_timeout_s

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self._target_torque_nm = 0.0
        self._state: MotorState | None = None
        self._feedback_monotonic_s: float | None = None
        self._feedback_count = 0
        self._fault: str | None = None

    def start(self) -> None:
        """Enable the actuator, verify zero-torque feedback, and start the loop."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("torque session is already running")
            self._stop.clear()
            self._active = False
            self._target_torque_nm = 0.0
            self._state = None
            self._feedback_monotonic_s = None
            self._feedback_count = 0
            self._fault = None

        enabled = False
        try:
            self._motor.enable()
            enabled = True
            self._motor.set_torque(0.0)
            state = self._motor.read_state(self.feedback_timeout_s)
            if state is None:
                raise TimeoutError(
                    f"no AK40-10 feedback within {self.feedback_timeout_s:g} s"
                )
            self._validate_state(state)

            thread = threading.Thread(
                target=self._run,
                name=f"ak40-torque-{self._motor.motor_id}",
            )
            with self._lock:
                self._active = True
                self._state = state
                self._feedback_monotonic_s = monotonic()
                self._feedback_count = 1
                self._thread = thread
            thread.start()
        except Exception as exc:
            self._record_fault(exc)
            if enabled:
                self._safe_deactivate()
            raise

    def set_target_torque(self, torque_nm: float) -> None:
        """Set the next complete torque command within the operator limit."""

        torque_nm = float(torque_nm)
        if not math.isfinite(torque_nm):
            raise ValueError("torque_nm must be finite")
        if abs(torque_nm) > self.max_abs_torque_nm:
            raise ValueError(
                f"torque_nm must be within +/-{self.max_abs_torque_nm:g} N m"
            )
        with self._lock:
            if not self._active:
                raise RuntimeError("torque session is not active")
            self._target_torque_nm = torque_nm

    def snapshot(self) -> TorqueSessionSnapshot:
        """Return the latest coherent command, feedback, and fault state."""

        with self._lock:
            return TorqueSessionSnapshot(
                active=self._active,
                target_torque_nm=self._target_torque_nm,
                state=self._state,
                feedback_monotonic_s=self._feedback_monotonic_s,
                feedback_count=self._feedback_count,
                fault=self._fault,
            )

    def stop(self) -> None:
        """Request zero torque and wait for the worker to disable the actuator."""

        with self._lock:
            thread = self._thread
            self._target_torque_nm = 0.0
        if thread is None:
            return
        self._stop.set()
        thread.join(timeout=self.feedback_timeout_s + 1.0)
        if thread.is_alive():
            raise RuntimeError("torque session did not stop within its bounded wait")

    def _run(self) -> None:
        period_s = 1.0 / self.command_rate_hz
        next_command_s = monotonic()
        try:
            while not self._stop.is_set():
                with self._lock:
                    target_torque_nm = self._target_torque_nm

                self._motor.set_torque(target_torque_nm)
                state = self._motor.read_state(self.feedback_timeout_s)
                if state is None:
                    raise TimeoutError(
                        f"no AK40-10 feedback within {self.feedback_timeout_s:g} s"
                    )
                self._validate_state(state)
                with self._lock:
                    self._state = state
                    self._feedback_monotonic_s = monotonic()
                    self._feedback_count += 1

                next_command_s += period_s
                delay_s = next_command_s - monotonic()
                if delay_s <= 0.0:
                    next_command_s = monotonic()
                    continue
                self._stop.wait(delay_s)
        except Exception as exc:
            self._record_fault(exc)
        finally:
            self._safe_deactivate()

    @staticmethod
    def _validate_state(state: MotorState) -> None:
        if state.error != 0:
            raise RuntimeError(f"AK40-10 reported error code {state.error}")
        values = (state.position, state.velocity, state.torque, state.temperature)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("AK40-10 returned non-finite feedback")

    def _safe_deactivate(self) -> None:
        try:
            self._motor.set_torque(0.0)
        except Exception as exc:
            self._record_fault(exc)
        try:
            self._motor.disable()
        except Exception as exc:
            self._record_fault(exc)
        with self._lock:
            self._active = False
            self._target_torque_nm = 0.0

    def _record_fault(self, exc: Exception) -> None:
        with self._lock:
            if self._fault is None:
                self._fault = f"{type(exc).__name__}: {exc}"


__all__ = ["AK40TorqueSession", "TorqueSessionSnapshot"]
