from __future__ import annotations

from experiments.data_collection.force_sequence import (
    ForceBandPosition,
    ForceSequenceConfig,
    ForceSequenceController,
    ForceSequenceEvent,
    ForceSequenceState,
    UnloadedCaptureController,
    UnloadedCaptureEvent,
    UnloadedCaptureState,
)


CONFIG = ForceSequenceConfig(
    settle_duration_s=0.5,
    record_duration_s=1.0,
    capture_rate_hz=4.0,
    unloaded_settle_duration_s=0.5,
    unloaded_record_duration_s=1.0,
)


def _complete_target(
    controller: ForceSequenceController,
    target_n: float,
    start_s: float,
) -> float:
    update = controller.update(start_s, target_n)
    assert update.state is ForceSequenceState.SETTLING
    recording_start = start_s + 0.5
    update = controller.update(recording_start, target_n)
    assert update.state is ForceSequenceState.RECORDING
    assert ForceSequenceEvent.RECORDING_STARTED in update.events
    assert update.should_record_frame
    for elapsed_s in (0.26, 0.51, 0.76):
        update = controller.update(recording_start + elapsed_s, target_n)
        assert update.should_record_frame
    update = controller.update(recording_start + 1.0, target_n)
    assert ForceSequenceEvent.TARGET_COMPLETED in update.events
    assert not update.should_record_frame
    assert update.completed_target_n == target_n
    return recording_start + 1.0


def test_full_valid_recording_captures_five_scheduled_frames() -> None:
    config = ForceSequenceConfig(
        target_forces_n=(2.0,),
        settle_duration_s=0.1,
        record_duration_s=1.0,
        capture_rate_hz=5.0,
    )
    controller = ForceSequenceController(config)
    controller.start(0.0)
    controller.update(0.1, 2.0)
    updates = [controller.update(0.2, 2.0)]
    updates.extend(
        controller.update(0.2 + index / 100.0, 2.0) for index in range(1, 101)
    )

    assert sum(update.should_record_frame for update in updates) == 5
    assert config.expected_record_frame_count == 5
    assert updates[-1].state is ForceSequenceState.RUN_COMPLETE
    assert ForceSequenceEvent.TARGET_COMPLETED in updates[-1].events


def test_correct_progressive_sequence_completes_in_order() -> None:
    controller = ForceSequenceController(CONFIG)
    controller.start(0.0)
    now = 0.1
    for target in CONFIG.target_forces_n:
        now = _complete_target(controller, target, now)
        now += 0.1

    assert controller.state is ForceSequenceState.RUN_COMPLETE
    assert controller.completed_targets_n == CONFIG.target_forces_n


def test_force_spike_crossing_target_does_not_count() -> None:
    controller = ForceSequenceController(CONFIG)
    controller.start(0.0)

    update = controller.update(0.1, 1.0)
    assert update.state is ForceSequenceState.WAITING_FOR_TARGET
    update = controller.update(0.2, 3.0)
    assert update.state is ForceSequenceState.WAITING_FOR_TARGET
    assert update.completed_targets_n == ()


def test_leaving_band_during_settling_resets() -> None:
    controller = ForceSequenceController(CONFIG)
    controller.start(0.0)
    controller.update(0.1, 2.0)

    update = controller.update(0.4, 1.0)

    assert update.state is ForceSequenceState.WAITING_FOR_TARGET
    assert ForceSequenceEvent.ATTEMPT_RESET in update.events


def test_leaving_band_at_point_nine_seconds_discards_attempt() -> None:
    controller = ForceSequenceController(CONFIG)
    controller.start(0.0)
    controller.update(0.1, 2.0)
    controller.update(0.6, 2.0)
    controller.update(0.86, 2.0)

    update = controller.update(1.5, 1.0)

    assert update.state is ForceSequenceState.WAITING_FOR_TARGET
    assert ForceSequenceEvent.ATTEMPT_RESET in update.events
    assert update.completed_targets_n == ()


def test_reentry_after_failed_recording_starts_new_schedule() -> None:
    controller = ForceSequenceController(CONFIG)
    controller.start(0.0)
    controller.update(0.1, 2.0)
    controller.update(0.6, 2.0)
    controller.update(1.5, 1.0)

    controller.update(1.6, 2.0)
    restarted = controller.update(2.1, 2.0)

    assert restarted.state is ForceSequenceState.RECORDING
    assert ForceSequenceEvent.RECORDING_STARTED in restarted.events
    assert restarted.should_record_frame
    assert restarted.phase_elapsed_s == 0.0


def test_overshoot_during_recording_resets_attempt() -> None:
    controller = ForceSequenceController(CONFIG)
    controller.start(0.0)
    controller.update(0.1, 2.0)
    controller.update(0.6, 2.0)

    update = controller.update(1.0, 2.5)

    assert update.band_position is ForceBandPosition.ABOVE
    assert update.state is ForceSequenceState.WAITING_FOR_TARGET
    assert ForceSequenceEvent.ATTEMPT_RESET in update.events


def test_one_camera_frame_is_not_reused_for_multiple_missed_deadlines() -> None:
    controller = ForceSequenceController(CONFIG)
    controller.start(0.0)
    controller.update(0.1, 2.0)
    started = controller.update(0.6, 2.0)
    delayed = controller.update(1.36, 2.0)
    immediate_next = controller.update(1.361, 2.0)

    assert started.should_record_frame
    assert delayed.should_record_frame
    assert not immediate_next.should_record_frame

    incomplete = controller.update(1.6, 2.0)
    assert incomplete.state is ForceSequenceState.WAITING_FOR_TARGET
    assert ForceSequenceEvent.ATTEMPT_RESET in incomplete.events
    assert incomplete.completed_targets_n == ()


def test_external_camera_drop_resets_only_current_target_attempt() -> None:
    controller = ForceSequenceController(CONFIG)
    controller.start(0.0)
    first_target_end = _complete_target(controller, 2.0, 0.1)
    controller.update(first_target_end + 0.1, 5.0)
    controller.update(first_target_end + 0.6, 5.0)

    reset = controller.reset_attempt(first_target_end + 0.7)

    assert reset.state is ForceSequenceState.WAITING_FOR_TARGET
    assert reset.current_target_n == 5.0
    assert reset.completed_targets_n == (2.0,)
    assert reset.events == (ForceSequenceEvent.ATTEMPT_RESET,)
    assert controller.captured_frame_count == 0


def test_two_newtons_success_advances_without_release() -> None:
    controller = ForceSequenceController(CONFIG)
    controller.start(0.0)
    end = _complete_target(controller, 2.0, 0.1)

    update = controller.update(end + 0.1, 5.0)

    assert update.state is ForceSequenceState.SETTLING
    assert update.current_target_n == 5.0


def test_fifteen_newtons_success_emits_run_complete() -> None:
    config = ForceSequenceConfig(
        target_forces_n=(15.0,),
        settle_duration_s=0.5,
        record_duration_s=1.0,
        capture_rate_hz=4.0,
    )
    controller = ForceSequenceController(config)
    controller.start(0.0)
    _complete_target(controller, 15.0, 0.1)

    assert controller.state is ForceSequenceState.RUN_COMPLETE


def test_abort_produces_aborted_state() -> None:
    controller = ForceSequenceController(CONFIG)
    controller.start(0.0)

    update = controller.abort(0.1)

    assert update.state is ForceSequenceState.ABORTED
    assert update.events == (ForceSequenceEvent.ABORTED,)


def test_overshoot_does_not_advance_five_newton_target() -> None:
    config = ForceSequenceConfig(
        target_forces_n=(5.0,),
        settle_duration_s=0.5,
    )
    controller = ForceSequenceController(config)
    controller.start(0.0)

    update = controller.update(0.1, 7.0)

    assert update.band_position is ForceBandPosition.ABOVE
    assert update.state is ForceSequenceState.WAITING_FOR_TARGET


def test_unloaded_continuous_hold_captures_scheduled_burst() -> None:
    controller = UnloadedCaptureController(CONFIG)
    controller.start(0.0)
    controller.update(0.1, 0.1)
    started = controller.update(0.6, 0.1)
    captures = [started]
    captures.extend(
        controller.update(0.6 + elapsed, 0.1) for elapsed in (0.26, 0.51, 0.76)
    )
    completed = controller.update(1.6, 0.1)

    assert UnloadedCaptureEvent.RECORDING_STARTED in started.events
    assert sum(update.should_record_frame for update in captures) == 4
    assert CONFIG.expected_unloaded_frame_count == 4
    assert completed.state is UnloadedCaptureState.COMPLETE
    assert UnloadedCaptureEvent.CAPTURE_COMPLETED in completed.events
    assert not completed.should_record_frame


def test_unloaded_force_excursion_resets() -> None:
    controller = UnloadedCaptureController(CONFIG)
    controller.start(0.0)
    controller.update(0.1, 0.1)
    controller.update(0.6, 0.1)

    update = controller.update(1.0, 1.01)

    assert update.state is UnloadedCaptureState.WAITING_FOR_UNLOADED
    assert UnloadedCaptureEvent.ATTEMPT_RESET in update.events


def test_unloaded_missing_scheduled_frames_discards_attempt() -> None:
    controller = UnloadedCaptureController(CONFIG)
    controller.start(0.0)
    controller.update(0.1, 0.1)
    controller.update(0.6, 0.1)
    controller.update(1.36, 0.1)

    update = controller.update(1.6, 0.1)

    assert update.state is UnloadedCaptureState.WAITING_FOR_UNLOADED
    assert update.events == (UnloadedCaptureEvent.ATTEMPT_RESET,)
    assert controller.captured_frame_count == 0
