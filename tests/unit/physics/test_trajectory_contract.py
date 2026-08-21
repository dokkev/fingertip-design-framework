from __future__ import annotations

import numpy as np

from lumo.physics import CheckpointStep, checkpoint_step_schedule


def test_checkpoint_schedule_lands_exactly_and_respects_increment() -> None:
    schedule = checkpoint_step_schedule((0.5, 1.0, 1.5), max_load_increment_mm=0.05)
    assert len(schedule) == 30
    assert [
        step.travel_mm
        for step in schedule
        if step.travel_mm in {0.5, 1.0, 1.5}
    ] == [0.5, 1.0, 1.5]
    increments = np.diff([0.0, *[step.travel_mm for step in schedule]])
    assert float(np.max(increments)) <= 0.05 + 1.0e-12
    assert schedule[-1] == CheckpointStep(1.5, 10, 30)


def test_checkpoint_schedule_rejects_non_monotonic_path() -> None:
    try:
        checkpoint_step_schedule((0.5, 0.4), max_load_increment_mm=0.05)
    except ValueError:
        pass
    else:
        raise AssertionError("non-monotonic checkpoint path was accepted")
