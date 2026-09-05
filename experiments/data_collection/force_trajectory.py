"""Monotonic-time force trajectory for continuous cyclic contact acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math


@dataclass(frozen=True)
class ForceTrajectoryConfig:
    """Timing contract for one continuously engaged cyclic-contact run."""

    min_force_n: float = 2.0
    max_force_n: float = 15.0
    ramp_rate_n_per_s: float = 1.0
    low_dwell_s: float = 1.0
    high_dwell_s: float = 1.0
    conditioning_cycles: int = 2
    measurement_cycles: int = 5
    preload_tolerance_n: float = 1.0
    preload_settle_s: float = 0.5
    release_max_force_n: float = 1.0
    release_settle_s: float = 0.5
    capture_rate_hz: float = 5.0

    def __post_init__(self) -> None:
        positive = (
            "min_force_n",
            "max_force_n",
            "ramp_rate_n_per_s",
            "low_dwell_s",
            "high_dwell_s",
            "preload_tolerance_n",
            "preload_settle_s",
            "release_settle_s",
            "capture_rate_hz",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        release = float(self.release_max_force_n)
        if not math.isfinite(release) or release < 0.0:
            raise ValueError("release_max_force_n must be finite and nonnegative")
        object.__setattr__(self, "release_max_force_n", release)
        if self.max_force_n <= self.min_force_n:
            raise ValueError("max_force_n must be greater than min_force_n")
        if self.release_max_force_n >= self.min_force_n:
            raise ValueError("release_max_force_n must be below min_force_n")
        for name in ("conditioning_cycles", "measurement_cycles"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.conditioning_cycles < 0:
            raise ValueError("conditioning_cycles must be nonnegative")
        if self.measurement_cycles < 1:
            raise ValueError("measurement_cycles must be at least one")

    @property
    def ramp_duration_s(self) -> float:
        return (self.max_force_n - self.min_force_n) / self.ramp_rate_n_per_s

    @property
    def nominal_cycle_duration_s(self) -> float:
        return 2.0 * self.ramp_duration_s + self.high_dwell_s + self.low_dwell_s

    @property
    def total_cycles(self) -> int:
        return self.conditioning_cycles + self.measurement_cycles

    @property
    def nominal_trajectory_duration_s(self) -> float:
        return self.total_cycles * self.nominal_cycle_duration_s

    @property
    def capture_period_s(self) -> float:
        return 1.0 / self.capture_rate_hz

    @property
    def expected_capture_count(self) -> int:
        """Scheduled trajectory observations, including the sample at time zero."""

        return math.ceil(
            self.nominal_trajectory_duration_s * self.capture_rate_hz - 1.0e-12
        )


class ForceTrajectoryState(Enum):
    IDLE = auto()
    WAITING_FOR_PRELOAD = auto()
    PRELOAD_SETTLING = auto()
    CYCLING = auto()
    WAITING_FOR_RELEASE = auto()
    COMPLETE = auto()
    ABORTED = auto()


class ForceTrajectoryPhase(Enum):
    LOW_DWELL = "low_dwell"
    LOADING = "loading"
    HIGH_DWELL = "high_dwell"
    UNLOADING = "unloading"


class ForceTrajectoryEvent(Enum):
    PRELOAD_SETTLING_STARTED = auto()
    PRELOAD_LOST = auto()
    CYCLING_STARTED = auto()
    PHASE_CHANGED = auto()
    CYCLE_CHANGED = auto()
    WAITING_FOR_RELEASE = auto()
    RELEASE_SETTLING_STARTED = auto()
    RELEASE_LOST = auto()
    COMPLETED = auto()
    ABORTED = auto()


@dataclass(frozen=True)
class ForceTrajectoryUpdate:
    """One deterministic trajectory observation at a monotonic timestamp."""

    state: ForceTrajectoryState
    phase: ForceTrajectoryPhase | None
    cycle_index: int | None
    cycle_role: str | None
    cycle_role_index: int | None
    target_force_n: float | None
    target_ramp_n_per_s: float
    tracking_error_n: float | None
    trajectory_elapsed_s: float
    phase_elapsed_s: float
    should_capture_frame: bool
    missed_capture_deadlines: int
    events: tuple[ForceTrajectoryEvent, ...]


class ForceTrajectoryController:
    """Generate one nominal trajectory without using measured force as control."""

    def __init__(self, config: ForceTrajectoryConfig | None = None) -> None:
        self.config = config or ForceTrajectoryConfig()
        self._state = ForceTrajectoryState.IDLE
        self._last_time_s: float | None = None
        self._phase: ForceTrajectoryPhase | None = None
        self._cycle_index: int | None = None
        self._preload_started_s: float | None = None
        self._release_started_s: float | None = None
        self._trajectory_started_s: float | None = None
        self._trajectory_ended_s: float | None = None
        self._next_capture_index: int | None = None
        self._missed_capture_deadlines = 0

    @property
    def state(self) -> ForceTrajectoryState:
        return self._state

    @property
    def trajectory_start_time_s(self) -> float | None:
        return self._trajectory_started_s

    @property
    def trajectory_end_time_s(self) -> float | None:
        return self._trajectory_ended_s

    @property
    def missed_capture_deadlines(self) -> int:
        return self._missed_capture_deadlines

    def start(self, now_s: float) -> ForceTrajectoryUpdate:
        now = self._validate_time(now_s, first=True)
        if self._state is not ForceTrajectoryState.IDLE:
            raise RuntimeError("force trajectory has already started")
        self._state = ForceTrajectoryState.WAITING_FOR_PRELOAD
        self._last_time_s = now
        return self._snapshot(actual_force_n=None)

    def update(self, now_s: float, actual_force_n: float) -> ForceTrajectoryUpdate:
        now = self._validate_time(now_s)
        actual = self._validate_force(actual_force_n)
        self._last_time_s = now
        events: list[ForceTrajectoryEvent] = []
        should_capture = False
        missed_now = 0

        if self._state in {
            ForceTrajectoryState.IDLE,
            ForceTrajectoryState.COMPLETE,
            ForceTrajectoryState.ABORTED,
        }:
            return self._snapshot(actual_force_n=actual)

        if self._state is ForceTrajectoryState.WAITING_FOR_PRELOAD:
            if self._in_preload_band(actual):
                self._state = ForceTrajectoryState.PRELOAD_SETTLING
                self._preload_started_s = now
                events.append(ForceTrajectoryEvent.PRELOAD_SETTLING_STARTED)

        elif self._state is ForceTrajectoryState.PRELOAD_SETTLING:
            if not self._in_preload_band(actual):
                self._state = ForceTrajectoryState.WAITING_FOR_PRELOAD
                self._preload_started_s = None
                events.append(ForceTrajectoryEvent.PRELOAD_LOST)
            elif now - self._preload_start() >= self.config.preload_settle_s:
                self._state = ForceTrajectoryState.CYCLING
                self._trajectory_started_s = now
                self._next_capture_index = 0
                self._set_schedule_position(0.0)
                events.append(ForceTrajectoryEvent.CYCLING_STARTED)

        if self._state is ForceTrajectoryState.CYCLING:
            elapsed = now - self._trajectory_start()
            if elapsed >= self.config.nominal_trajectory_duration_s:
                missed_now = self._consume_remaining_deadlines()
                self._missed_capture_deadlines += missed_now
                self._trajectory_ended_s = (
                    self._trajectory_start() + self.config.nominal_trajectory_duration_s
                )
                self._state = ForceTrajectoryState.WAITING_FOR_RELEASE
                self._phase = None
                self._cycle_index = None
                events.append(ForceTrajectoryEvent.WAITING_FOR_RELEASE)
            else:
                previous_phase = self._phase
                previous_cycle = self._cycle_index
                self._set_schedule_position(elapsed)
                if previous_cycle is not None and self._cycle_index != previous_cycle:
                    events.append(ForceTrajectoryEvent.CYCLE_CHANGED)
                elif previous_phase is not None and self._phase is not previous_phase:
                    events.append(ForceTrajectoryEvent.PHASE_CHANGED)
                should_capture, missed_now = self._capture_due(now)
                self._missed_capture_deadlines += missed_now

        if self._state is ForceTrajectoryState.WAITING_FOR_RELEASE:
            if actual <= self.config.release_max_force_n:
                if self._release_started_s is None:
                    self._release_started_s = now
                    events.append(ForceTrajectoryEvent.RELEASE_SETTLING_STARTED)
                elif now - self._release_started_s >= self.config.release_settle_s:
                    self._state = ForceTrajectoryState.COMPLETE
                    events.append(ForceTrajectoryEvent.COMPLETED)
            elif self._release_started_s is not None:
                self._release_started_s = None
                events.append(ForceTrajectoryEvent.RELEASE_LOST)

        return self._snapshot(
            actual_force_n=actual,
            should_capture_frame=should_capture,
            missed_capture_deadlines=missed_now,
            events=tuple(events),
        )

    def abort(self, now_s: float) -> ForceTrajectoryUpdate:
        now = self._validate_time(now_s)
        if self._state is ForceTrajectoryState.COMPLETE:
            raise RuntimeError("a completed trajectory cannot be aborted")
        self._last_time_s = now
        self._state = ForceTrajectoryState.ABORTED
        self._phase = None
        self._cycle_index = None
        return self._snapshot(
            actual_force_n=None,
            events=(ForceTrajectoryEvent.ABORTED,),
        )

    def _set_schedule_position(self, trajectory_elapsed_s: float) -> None:
        cycle_duration = self.config.nominal_cycle_duration_s
        cycle_zero_index = min(
            int(trajectory_elapsed_s / cycle_duration),
            self.config.total_cycles - 1,
        )
        within = trajectory_elapsed_s - cycle_zero_index * cycle_duration
        ramp = self.config.ramp_duration_s
        if within < ramp:
            phase = ForceTrajectoryPhase.LOADING
        elif within < ramp + self.config.high_dwell_s:
            phase = ForceTrajectoryPhase.HIGH_DWELL
        elif within < 2.0 * ramp + self.config.high_dwell_s:
            phase = ForceTrajectoryPhase.UNLOADING
        else:
            phase = ForceTrajectoryPhase.LOW_DWELL
        self._cycle_index = cycle_zero_index + 1
        self._phase = phase

    def _target(self, elapsed_s: float) -> tuple[float, float, float]:
        cycle_start = (
            self._cycle_index_value() - 1
        ) * self.config.nominal_cycle_duration_s
        within = elapsed_s - cycle_start
        ramp = self.config.ramp_duration_s
        if self._phase is ForceTrajectoryPhase.LOADING:
            return (
                min(
                    self.config.max_force_n,
                    self.config.min_force_n + self.config.ramp_rate_n_per_s * within,
                ),
                self.config.ramp_rate_n_per_s,
                within,
            )
        if self._phase is ForceTrajectoryPhase.HIGH_DWELL:
            return self.config.max_force_n, 0.0, within - ramp
        if self._phase is ForceTrajectoryPhase.UNLOADING:
            phase_elapsed = within - ramp - self.config.high_dwell_s
            return (
                max(
                    self.config.min_force_n,
                    self.config.max_force_n
                    - self.config.ramp_rate_n_per_s * phase_elapsed,
                ),
                -self.config.ramp_rate_n_per_s,
                phase_elapsed,
            )
        phase_elapsed = within - 2.0 * ramp - self.config.high_dwell_s
        return self.config.min_force_n, 0.0, phase_elapsed

    def _capture_due(self, now_s: float) -> tuple[bool, int]:
        if self._next_capture_index is None:
            return False, 0
        due = 0
        while self._next_capture_index < self.config.expected_capture_count:
            deadline = (
                self._trajectory_start()
                + self._next_capture_index * self.config.capture_period_s
            )
            if deadline > now_s:
                break
            due += 1
            self._next_capture_index += 1
        return due > 0, max(0, due - 1)

    def _consume_remaining_deadlines(self) -> int:
        if self._next_capture_index is None:
            return 0
        count = self.config.expected_capture_count - self._next_capture_index
        self._next_capture_index = self.config.expected_capture_count
        return count

    def _snapshot(
        self,
        *,
        actual_force_n: float | None,
        should_capture_frame: bool = False,
        missed_capture_deadlines: int = 0,
        events: tuple[ForceTrajectoryEvent, ...] = (),
    ) -> ForceTrajectoryUpdate:
        trajectory_elapsed = 0.0
        phase_elapsed = 0.0
        target = None
        target_ramp = 0.0
        tracking_error = None
        cycle_role = None
        cycle_role_index = None
        if self._state is ForceTrajectoryState.CYCLING:
            trajectory_elapsed = self._last_time() - self._trajectory_start()
            target, target_ramp, phase_elapsed = self._target(trajectory_elapsed)
            cycle = self._cycle_index_value()
            if cycle <= self.config.conditioning_cycles:
                cycle_role = "conditioning"
                cycle_role_index = cycle
            else:
                cycle_role = "measurement"
                cycle_role_index = cycle - self.config.conditioning_cycles
            if actual_force_n is not None:
                tracking_error = actual_force_n - target
        elif self._state is ForceTrajectoryState.PRELOAD_SETTLING:
            phase_elapsed = self._last_time() - self._preload_start()
            target = self.config.min_force_n
            if actual_force_n is not None:
                tracking_error = actual_force_n - target
        elif self._state is ForceTrajectoryState.WAITING_FOR_PRELOAD:
            target = self.config.min_force_n
            if actual_force_n is not None:
                tracking_error = actual_force_n - target
        elif self._state is ForceTrajectoryState.WAITING_FOR_RELEASE:
            trajectory_elapsed = self.config.nominal_trajectory_duration_s
            if self._release_started_s is not None:
                phase_elapsed = self._last_time() - self._release_started_s
        elif self._state is ForceTrajectoryState.COMPLETE:
            trajectory_elapsed = self.config.nominal_trajectory_duration_s
        return ForceTrajectoryUpdate(
            state=self._state,
            phase=self._phase,
            cycle_index=self._cycle_index,
            cycle_role=cycle_role,
            cycle_role_index=cycle_role_index,
            target_force_n=target,
            target_ramp_n_per_s=target_ramp,
            tracking_error_n=tracking_error,
            trajectory_elapsed_s=trajectory_elapsed,
            phase_elapsed_s=phase_elapsed,
            should_capture_frame=should_capture_frame,
            missed_capture_deadlines=missed_capture_deadlines,
            events=events,
        )

    def _in_preload_band(self, actual_force_n: float) -> bool:
        return (
            abs(actual_force_n - self.config.min_force_n)
            <= self.config.preload_tolerance_n
        )

    def _validate_time(self, now_s: float, *, first: bool = False) -> float:
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError("now_s must be finite")
        if not first and self._last_time_s is not None and now < self._last_time_s:
            raise ValueError("now_s must be monotonic nondecreasing")
        return now

    @staticmethod
    def _validate_force(actual_force_n: float) -> float:
        value = float(actual_force_n)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("actual_force_n must be finite and nonnegative")
        return value

    def _last_time(self) -> float:
        assert self._last_time_s is not None
        return self._last_time_s

    def _preload_start(self) -> float:
        assert self._preload_started_s is not None
        return self._preload_started_s

    def _trajectory_start(self) -> float:
        assert self._trajectory_started_s is not None
        return self._trajectory_started_s

    def _cycle_index_value(self) -> int:
        assert self._cycle_index is not None
        return self._cycle_index


__all__ = [
    "ForceTrajectoryConfig",
    "ForceTrajectoryController",
    "ForceTrajectoryEvent",
    "ForceTrajectoryPhase",
    "ForceTrajectoryState",
    "ForceTrajectoryUpdate",
]
