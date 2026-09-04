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
    unloaded_settle_duration_s=0.5,
)


def _complete_target(
    controller: ForceSequenceController,
    target_n: float,
    start_s: float,
) -> float:
    update = controller.update(start_s, target_n)
    assert update.state is ForceSequenceState.SETTLING
    update = controller.update(start_s + 0.5, target_n)
    assert ForceSequenceEvent.TARGET_COMPLETED in update.events
    assert update.should_record_frame
    assert update.record_target_n == target_n
    return start_s + 0.5


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
    )
    controller = ForceSequenceController(config)
    controller.start(0.0)
    controller.update(0.1, 15.0)
    update = controller.update(0.6, 15.0)

    assert update.state is ForceSequenceState.RUN_COMPLETE
    assert ForceSequenceEvent.RUN_COMPLETED in update.events


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


def test_unloaded_continuous_hold_captures_one_snapshot() -> None:
    controller = UnloadedCaptureController(CONFIG)
    controller.start(0.0)
    controller.update(0.1, 0.1)
    completed = controller.update(0.6, 0.1)

    assert completed.state is UnloadedCaptureState.COMPLETE
    assert UnloadedCaptureEvent.CAPTURE_COMPLETED in completed.events
    assert completed.should_record_frame


def test_unloaded_force_excursion_resets() -> None:
    controller = UnloadedCaptureController(CONFIG)
    controller.start(0.0)
    controller.update(0.1, 0.1)

    update = controller.update(0.4, 1.01)

    assert update.state is UnloadedCaptureState.WAITING_FOR_UNLOADED
    assert UnloadedCaptureEvent.ATTEMPT_RESET in update.events
