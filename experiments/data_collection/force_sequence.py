"""Hardware-independent force-hold state machines for contact acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math


@dataclass(frozen=True)
class ForceSequenceConfig:
    """Timing and force-band contract shared by loaded and unloaded capture."""

    target_forces_n: tuple[float, ...] = (2.0, 5.0, 10.0, 15.0)
    settle_duration_s: float = 1.0
    record_duration_s: float = 1.0
    capture_rate_hz: float = 5.0
    unloaded_max_force_n: float = 1.0
    unloaded_settle_duration_s: float = 1.0
    unloaded_record_duration_s: float = 1.0
    minimum_tolerance_n: float = 0.2
    low_force_relative_tolerance: float = 0.20
    high_force_relative_tolerance: float = 0.10
    high_force_threshold_n: float = 10.0

    def __post_init__(self) -> None:
        targets = tuple(float(value) for value in self.target_forces_n)
        if not targets or not all(math.isfinite(value) and value > 0 for value in targets):
            raise ValueError("target_forces_n must contain positive finite values")
        if any(right <= left for left, right in zip(targets, targets[1:])):
            raise ValueError("target_forces_n must be strictly increasing")
        object.__setattr__(self, "target_forces_n", targets)
        positive_fields = (
            "settle_duration_s",
            "record_duration_s",
            "capture_rate_hz",
            "unloaded_settle_duration_s",
            "unloaded_record_duration_s",
            "unloaded_max_force_n",
            "minimum_tolerance_n",
            "low_force_relative_tolerance",
            "high_force_relative_tolerance",
            "high_force_threshold_n",
        )
        for name in positive_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

    @property
    def capture_period_s(self) -> float:
        return 1.0 / self.capture_rate_hz

    def tolerance_n(self, target_force_n: float) -> float:
        """Return the absolute tolerance for one low- or high-force target."""

        target = float(target_force_n)
        if not math.isfinite(target) or target <= 0:
            raise ValueError("target_force_n must be finite and positive")
        relative_tolerance = (
            self.high_force_relative_tolerance
            if target >= self.high_force_threshold_n
            else self.low_force_relative_tolerance
        )
        return max(self.minimum_tolerance_n, relative_tolerance * target)


class ForceSequenceState(Enum):
    IDLE = auto()
    WAITING_FOR_TARGET = auto()
    SETTLING = auto()
    RECORDING = auto()
    RUN_COMPLETE = auto()
    ABORTED = auto()


class ForceBandPosition(Enum):
    BELOW = auto()
    IN_BAND = auto()
    ABOVE = auto()


class ForceSequenceEvent(Enum):
    SETTLING_STARTED = auto()
    RECORDING_STARTED = auto()
    ATTEMPT_RESET = auto()
    TARGET_COMPLETED = auto()
    RUN_COMPLETED = auto()
    ABORTED = auto()


@dataclass(frozen=True)
class ForceSequenceUpdate:
    """One state-machine observation returned after an input sample."""

    state: ForceSequenceState
    events: tuple[ForceSequenceEvent, ...]
    band_position: ForceBandPosition | None
    current_target_n: float | None
    current_target_index: int
    completed_targets_n: tuple[float, ...]
    phase_elapsed_s: float
    should_record_frame: bool = False
    record_target_n: float | None = None
    completed_target_n: float | None = None


class ForceSequenceController:
    """Progress through continuously held target-force bands without hardware I/O."""

    def __init__(self, config: ForceSequenceConfig | None = None) -> None:
        self.config = config or ForceSequenceConfig()
        self._state = ForceSequenceState.IDLE
        self._target_index = 0
        self._phase_started_s: float | None = None
        self._next_capture_time_s: float | None = None
        self._last_time_s: float | None = None

    @property
    def state(self) -> ForceSequenceState:
        return self._state

    @property
    def current_target_n(self) -> float | None:
        if self._target_index >= len(self.config.target_forces_n):
            return None
        return self.config.target_forces_n[self._target_index]

    @property
    def completed_targets_n(self) -> tuple[float, ...]:
        return self.config.target_forces_n[: self._target_index]

    def start(self, now_s: float) -> ForceSequenceUpdate:
        now = self._validate_time(now_s, allow_before_start=True)
        self._state = ForceSequenceState.WAITING_FOR_TARGET
        self._target_index = 0
        self._phase_started_s = None
        self._next_capture_time_s = None
        self._last_time_s = now
        return self._snapshot()

    def update(self, now_s: float, force_magnitude_n: float) -> ForceSequenceUpdate:
        now = self._validate_time(now_s)
        force = self._validate_force(force_magnitude_n)
        self._last_time_s = now
        if self._state in (
            ForceSequenceState.IDLE,
            ForceSequenceState.RUN_COMPLETE,
            ForceSequenceState.ABORTED,
        ):
            return self._snapshot()

        target = self.current_target_n
        assert target is not None
        band = self._band_position(force, target)
        events: list[ForceSequenceEvent] = []
        should_record = False
        record_target: float | None = None
        completed_target: float | None = None

        if self._state is ForceSequenceState.WAITING_FOR_TARGET:
            if band is ForceBandPosition.IN_BAND:
                self._state = ForceSequenceState.SETTLING
                self._phase_started_s = now
                events.append(ForceSequenceEvent.SETTLING_STARTED)
        elif self._state is ForceSequenceState.SETTLING:
            if band is not ForceBandPosition.IN_BAND:
                self._reset_to_waiting()
                events.append(ForceSequenceEvent.ATTEMPT_RESET)
            elif now - self._phase_start() >= self.config.settle_duration_s:
                self._state = ForceSequenceState.RECORDING
                self._phase_started_s = now
                self._next_capture_time_s = now + self.config.capture_period_s
                should_record = True
                record_target = target
                events.append(ForceSequenceEvent.RECORDING_STARTED)
        elif self._state is ForceSequenceState.RECORDING:
            if band is not ForceBandPosition.IN_BAND:
                self._reset_to_waiting()
                events.append(ForceSequenceEvent.ATTEMPT_RESET)
            elif now - self._phase_start() >= self.config.record_duration_s:
                completed_target = target
                events.append(ForceSequenceEvent.TARGET_COMPLETED)
                self._target_index += 1
                self._phase_started_s = None
                self._next_capture_time_s = None
                if self._target_index == len(self.config.target_forces_n):
                    self._state = ForceSequenceState.RUN_COMPLETE
                    events.append(ForceSequenceEvent.RUN_COMPLETED)
                else:
                    self._state = ForceSequenceState.WAITING_FOR_TARGET
            elif now >= self._next_capture_time():
                should_record = True
                record_target = target
                while self._next_capture_time() <= now:
                    self._next_capture_time_s += self.config.capture_period_s

        return self._snapshot(
            events=tuple(events),
            band=band,
            should_record=should_record,
            record_target=record_target,
            completed_target=completed_target,
        )

    def abort(self, now_s: float) -> ForceSequenceUpdate:
        now = self._validate_time(now_s)
        self._last_time_s = now
        if self._state is ForceSequenceState.RUN_COMPLETE:
            raise RuntimeError("a completed force sequence cannot be aborted")
        self._state = ForceSequenceState.ABORTED
        self._phase_started_s = None
        self._next_capture_time_s = None
        return self._snapshot(events=(ForceSequenceEvent.ABORTED,))

    def _reset_to_waiting(self) -> None:
        self._state = ForceSequenceState.WAITING_FOR_TARGET
        self._phase_started_s = None
        self._next_capture_time_s = None

    def _band_position(self, force_n: float, target_n: float) -> ForceBandPosition:
        tolerance = self.config.tolerance_n(target_n)
        if force_n < target_n - tolerance:
            return ForceBandPosition.BELOW
        if force_n > target_n + tolerance:
            return ForceBandPosition.ABOVE
        return ForceBandPosition.IN_BAND

    def _phase_start(self) -> float:
        assert self._phase_started_s is not None
        return self._phase_started_s

    def _next_capture_time(self) -> float:
        assert self._next_capture_time_s is not None
        return self._next_capture_time_s

    def _snapshot(
        self,
        *,
        events: tuple[ForceSequenceEvent, ...] = (),
        band: ForceBandPosition | None = None,
        should_record: bool = False,
        record_target: float | None = None,
        completed_target: float | None = None,
    ) -> ForceSequenceUpdate:
        elapsed = 0.0
        if self._phase_started_s is not None and self._last_time_s is not None:
            elapsed = max(0.0, self._last_time_s - self._phase_started_s)
        return ForceSequenceUpdate(
            state=self._state,
            events=events,
            band_position=band,
            current_target_n=self.current_target_n,
            current_target_index=self._target_index,
            completed_targets_n=self.completed_targets_n,
            phase_elapsed_s=elapsed,
            should_record_frame=should_record,
            record_target_n=record_target,
            completed_target_n=completed_target,
        )

    def _validate_time(self, now_s: float, *, allow_before_start: bool = False) -> float:
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError("now_s must be finite")
        if (
            not allow_before_start
            and self._last_time_s is not None
            and now < self._last_time_s
        ):
            raise ValueError("now_s must be monotonic")
        return now

    @staticmethod
    def _validate_force(force_magnitude_n: float) -> float:
        force = float(force_magnitude_n)
        if not math.isfinite(force) or force < 0:
            raise ValueError("force_magnitude_n must be finite and nonnegative")
        return force


class UnloadedCaptureState(Enum):
    IDLE = auto()
    WAITING_FOR_UNLOADED = auto()
    SETTLING = auto()
    RECORDING = auto()
    COMPLETE = auto()
    ABORTED = auto()


class UnloadedCaptureEvent(Enum):
    SETTLING_STARTED = auto()
    RECORDING_STARTED = auto()
    ATTEMPT_RESET = auto()
    CAPTURE_COMPLETED = auto()
    ABORTED = auto()


@dataclass(frozen=True)
class UnloadedCaptureUpdate:
    state: UnloadedCaptureState
    events: tuple[UnloadedCaptureEvent, ...]
    phase_elapsed_s: float
    should_record_frame: bool = False


class UnloadedCaptureController:
    """Capture one low-rate burst while the unloaded condition remains valid."""

    def __init__(self, config: ForceSequenceConfig | None = None) -> None:
        self.config = config or ForceSequenceConfig()
        self._state = UnloadedCaptureState.IDLE
        self._phase_started_s: float | None = None
        self._next_capture_time_s: float | None = None
        self._last_time_s: float | None = None

    @property
    def state(self) -> UnloadedCaptureState:
        return self._state

    def start(self, now_s: float) -> UnloadedCaptureUpdate:
        now = self._validate_time(now_s, allow_before_start=True)
        self._state = UnloadedCaptureState.WAITING_FOR_UNLOADED
        self._phase_started_s = None
        self._next_capture_time_s = None
        self._last_time_s = now
        return self._snapshot()

    def update(self, now_s: float, force_magnitude_n: float) -> UnloadedCaptureUpdate:
        now = self._validate_time(now_s)
        force = ForceSequenceController._validate_force(force_magnitude_n)
        self._last_time_s = now
        if self._state in (
            UnloadedCaptureState.IDLE,
            UnloadedCaptureState.COMPLETE,
            UnloadedCaptureState.ABORTED,
        ):
            return self._snapshot()

        events: list[UnloadedCaptureEvent] = []
        should_record = False
        is_unloaded = force <= self.config.unloaded_max_force_n
        if self._state is UnloadedCaptureState.WAITING_FOR_UNLOADED:
            if is_unloaded:
                self._state = UnloadedCaptureState.SETTLING
                self._phase_started_s = now
                events.append(UnloadedCaptureEvent.SETTLING_STARTED)
        elif self._state is UnloadedCaptureState.SETTLING:
            if not is_unloaded:
                self._reset_to_waiting()
                events.append(UnloadedCaptureEvent.ATTEMPT_RESET)
            elif now - self._phase_start() >= self.config.unloaded_settle_duration_s:
                self._state = UnloadedCaptureState.RECORDING
                self._phase_started_s = now
                self._next_capture_time_s = now + self.config.capture_period_s
                should_record = True
                events.append(UnloadedCaptureEvent.RECORDING_STARTED)
        elif self._state is UnloadedCaptureState.RECORDING:
            if not is_unloaded:
                self._reset_to_waiting()
                events.append(UnloadedCaptureEvent.ATTEMPT_RESET)
            elif now - self._phase_start() >= self.config.unloaded_record_duration_s:
                self._state = UnloadedCaptureState.COMPLETE
                self._phase_started_s = None
                self._next_capture_time_s = None
                events.append(UnloadedCaptureEvent.CAPTURE_COMPLETED)
            elif now >= self._next_capture_time():
                should_record = True
                while self._next_capture_time() <= now:
                    self._next_capture_time_s += self.config.capture_period_s
        return self._snapshot(tuple(events), should_record)

    def abort(self, now_s: float) -> UnloadedCaptureUpdate:
        now = self._validate_time(now_s)
        self._last_time_s = now
        if self._state is UnloadedCaptureState.COMPLETE:
            raise RuntimeError("a completed unloaded capture cannot be aborted")
        self._state = UnloadedCaptureState.ABORTED
        self._phase_started_s = None
        self._next_capture_time_s = None
        return self._snapshot((UnloadedCaptureEvent.ABORTED,))

    def _reset_to_waiting(self) -> None:
        self._state = UnloadedCaptureState.WAITING_FOR_UNLOADED
        self._phase_started_s = None
        self._next_capture_time_s = None

    def _phase_start(self) -> float:
        assert self._phase_started_s is not None
        return self._phase_started_s

    def _next_capture_time(self) -> float:
        assert self._next_capture_time_s is not None
        return self._next_capture_time_s

    def _snapshot(
        self,
        events: tuple[UnloadedCaptureEvent, ...] = (),
        should_record: bool = False,
    ) -> UnloadedCaptureUpdate:
        elapsed = 0.0
        if self._phase_started_s is not None and self._last_time_s is not None:
            elapsed = max(0.0, self._last_time_s - self._phase_started_s)
        return UnloadedCaptureUpdate(
            state=self._state,
            events=events,
            phase_elapsed_s=elapsed,
            should_record_frame=should_record,
        )

    def _validate_time(self, now_s: float, *, allow_before_start: bool = False) -> float:
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError("now_s must be finite")
        if (
            not allow_before_start
            and self._last_time_s is not None
            and now < self._last_time_s
        ):
            raise ValueError("now_s must be monotonic")
        return now


__all__ = [
    "ForceBandPosition",
    "ForceSequenceConfig",
    "ForceSequenceController",
    "ForceSequenceEvent",
    "ForceSequenceState",
    "ForceSequenceUpdate",
    "UnloadedCaptureController",
    "UnloadedCaptureEvent",
    "UnloadedCaptureState",
    "UnloadedCaptureUpdate",
]
