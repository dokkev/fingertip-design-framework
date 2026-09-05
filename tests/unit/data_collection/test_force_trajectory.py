from __future__ import annotations

import pytest

from experiments.data_collection.force_trajectory import (
    ForceTrajectoryConfig,
    ForceTrajectoryController,
    ForceTrajectoryEvent,
    ForceTrajectoryPhase,
    ForceTrajectoryState,
)


CONFIG = ForceTrajectoryConfig(
    min_force_n=2.0,
    max_force_n=4.0,
    ramp_rate_n_per_s=2.0,
    low_dwell_s=0.5,
    high_dwell_s=0.5,
    conditioning_cycles=1,
    measurement_cycles=2,
    preload_tolerance_n=0.5,
    preload_settle_s=0.25,
    release_max_force_n=1.0,
    release_settle_s=0.25,
    capture_rate_hz=5.0,
)


def test_default_trajectory_timing_contract() -> None:
    config = ForceTrajectoryConfig()

    assert config.ramp_duration_s == 13.0
    assert config.nominal_cycle_duration_s == 28.0
    assert config.total_cycles == 7
    assert config.nominal_trajectory_duration_s == 196.0
    assert config.expected_capture_count == 980


@pytest.mark.parametrize(
    "changes",
    (
        {"min_force_n": 0.0},
        {"max_force_n": 2.0},
        {"ramp_rate_n_per_s": 0.0},
        {"low_dwell_s": 0.0},
        {"high_dwell_s": 0.0},
        {"conditioning_cycles": -1},
        {"measurement_cycles": 0},
        {"capture_rate_hz": 0.0},
        {"release_max_force_n": 2.0},
    ),
)
def test_invalid_config_is_rejected(changes: dict[str, float | int]) -> None:
    values = {
        "min_force_n": 2.0,
        "max_force_n": 4.0,
        "ramp_rate_n_per_s": 2.0,
        "low_dwell_s": 0.5,
        "high_dwell_s": 0.5,
        "conditioning_cycles": 1,
        "measurement_cycles": 1,
        "capture_rate_hz": 5.0,
    }
    values.update(changes)
    with pytest.raises(ValueError):
        ForceTrajectoryConfig(**values)


def _start_cycling(
    controller: ForceTrajectoryController, start_s: float = 0.0
) -> float:
    controller.start(start_s)
    waiting = controller.update(start_s + 0.1, 2.0)
    assert waiting.state is ForceTrajectoryState.PRELOAD_SETTLING
    started_at = start_s + 0.36
    started = controller.update(started_at, 2.0)
    assert started.state is ForceTrajectoryState.CYCLING
    assert started.phase is ForceTrajectoryPhase.LOADING
    assert started.should_capture_frame
    return started_at


def test_preload_gate_and_loss_before_cycling() -> None:
    controller = ForceTrajectoryController(CONFIG)
    controller.start(0.0)

    waiting = controller.update(0.1, 0.0)
    settling = controller.update(0.2, 2.0)
    lost = controller.update(0.3, 2.6)

    assert waiting.state is ForceTrajectoryState.WAITING_FOR_PRELOAD
    assert settling.events == (ForceTrajectoryEvent.PRELOAD_SETTLING_STARTED,)
    assert lost.state is ForceTrajectoryState.WAITING_FOR_PRELOAD
    assert lost.events == (ForceTrajectoryEvent.PRELOAD_LOST,)


def test_loading_high_dwell_unloading_and_low_dwell_targets() -> None:
    controller = ForceTrajectoryController(CONFIG)
    start = _start_cycling(controller)

    loading = controller.update(start + 0.5, 0.0)
    high = controller.update(start + 1.01, 9.0)
    unloading = controller.update(start + 1.75, 0.0)
    low = controller.update(start + 2.51, 9.0)

    assert loading.phase is ForceTrajectoryPhase.LOADING
    assert loading.target_force_n == pytest.approx(3.0)
    assert loading.target_ramp_n_per_s == 2.0
    assert high.phase is ForceTrajectoryPhase.HIGH_DWELL
    assert high.target_force_n == 4.0
    assert high.target_ramp_n_per_s == 0.0
    assert unloading.phase is ForceTrajectoryPhase.UNLOADING
    assert unloading.target_force_n == pytest.approx(3.5)
    assert unloading.target_ramp_n_per_s == -2.0
    assert low.phase is ForceTrajectoryPhase.LOW_DWELL
    assert low.target_force_n == 2.0
    assert low.target_ramp_n_per_s == 0.0


def test_conditioning_and_measurement_cycle_labels_are_exact() -> None:
    controller = ForceTrajectoryController(CONFIG)
    start = _start_cycling(controller)
    cycle_duration = CONFIG.nominal_cycle_duration_s

    conditioning = controller.update(start + 0.1, 2.0)
    first_measurement = controller.update(start + cycle_duration + 0.1, 2.0)
    second_measurement = controller.update(start + 2.0 * cycle_duration + 0.1, 2.0)

    assert (
        conditioning.cycle_index,
        conditioning.cycle_role,
        conditioning.cycle_role_index,
    ) == (1, "conditioning", 1)
    assert (
        first_measurement.cycle_index,
        first_measurement.cycle_role,
        first_measurement.cycle_role_index,
    ) == (2, "measurement", 1)
    assert (
        second_measurement.cycle_index,
        second_measurement.cycle_role,
        second_measurement.cycle_role_index,
    ) == (3, "measurement", 2)


def test_actual_force_tracking_error_never_resets_cycling() -> None:
    controller = ForceTrajectoryController(CONFIG)
    start = _start_cycling(controller)

    far_below = controller.update(start + 0.5, 0.0)
    far_above = controller.update(start + 1.01, 100.0)

    assert far_below.state is ForceTrajectoryState.CYCLING
    assert far_below.tracking_error_n == pytest.approx(-3.0)
    assert far_above.state is ForceTrajectoryState.CYCLING
    assert far_above.phase is ForceTrajectoryPhase.HIGH_DWELL
    assert far_above.tracking_error_n == pytest.approx(96.0)


def test_exact_cycle_end_waits_for_release_then_completes() -> None:
    controller = ForceTrajectoryController(CONFIG)
    start = _start_cycling(controller)
    end = start + CONFIG.nominal_trajectory_duration_s

    waiting = controller.update(end, 2.0)
    release_started = controller.update(end + 0.1, 0.5)
    release_lost = controller.update(end + 0.2, 1.1)
    release_restarted = controller.update(end + 0.3, 0.5)
    complete = controller.update(end + 0.55, 0.5)

    assert waiting.state is ForceTrajectoryState.WAITING_FOR_RELEASE
    assert ForceTrajectoryEvent.WAITING_FOR_RELEASE in waiting.events
    assert release_started.events == (ForceTrajectoryEvent.RELEASE_SETTLING_STARTED,)
    assert release_lost.events == (ForceTrajectoryEvent.RELEASE_LOST,)
    assert release_restarted.events == (ForceTrajectoryEvent.RELEASE_SETTLING_STARTED,)
    assert complete.state is ForceTrajectoryState.COMPLETE
    assert complete.events == (ForceTrajectoryEvent.COMPLETED,)


def test_abort_and_monotonic_time_validation() -> None:
    controller = ForceTrajectoryController(CONFIG)
    controller.start(1.0)
    with pytest.raises(ValueError, match="monotonic"):
        controller.update(0.9, 2.0)

    aborted = controller.abort(1.1)

    assert aborted.state is ForceTrajectoryState.ABORTED
    assert aborted.events == (ForceTrajectoryEvent.ABORTED,)


def test_capture_schedule_uses_elapsed_time_and_counts_missed_deadlines() -> None:
    controller = ForceTrajectoryController(CONFIG)
    start = _start_cycling(controller)

    delayed = controller.update(start + 0.65, 2.0)

    assert delayed.should_capture_frame
    assert delayed.missed_capture_deadlines == 2
    assert controller.missed_capture_deadlines == 2


def test_capture_deadlines_do_not_gain_a_false_sample_at_exact_end() -> None:
    config = ForceTrajectoryConfig(
        min_force_n=2.0,
        max_force_n=3.0,
        ramp_rate_n_per_s=5.0,
        low_dwell_s=0.1,
        high_dwell_s=0.1,
        conditioning_cycles=1,
        measurement_cycles=1,
        preload_settle_s=0.05,
        capture_rate_hz=10.0,
    )
    controller = ForceTrajectoryController(config)
    controller.start(0.0)
    controller.update(0.01, 2.0)
    start_s = 0.07
    started = controller.update(start_s, 2.0)
    assert started.state is ForceTrajectoryState.CYCLING
    captured = int(started.should_capture_frame)
    for index in range(1, 36):
        update = controller.update(start_s + index / 30.0, 2.0)
        captured += int(update.should_capture_frame)
    ended = controller.update(start_s + config.nominal_trajectory_duration_s, 2.0)

    assert config.expected_capture_count == 12
    assert captured == 12
    assert ended.state is ForceTrajectoryState.WAITING_FOR_RELEASE
    assert controller.missed_capture_deadlines == 0
