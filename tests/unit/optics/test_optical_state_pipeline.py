from __future__ import annotations

import numpy as np
import pytest

from model import Fingertip, FingertipParameters, LED
from optics import TraceSettings, evaluate, field_difference, trace


def test_evaluate_is_camera_independent_and_zero_for_identical_results() -> None:
    settings = TraceSettings(
        ray_count=21,
        grid_width=32,
        grid_height=32,
        maximum_segment_count=3000,
    )
    reference = trace(
        Fingertip(
            FingertipParameters(),
            led=LED(emission_half_angle_deg=60.0),
        ),
        settings=settings,
    )

    identical = evaluate(reference, reference)

    assert identical["field_difference"] == pytest.approx(0.0)
    assert field_difference(reference, reference) == pytest.approx(0.0)
    assert identical["centroid_shift_mm"] == pytest.approx(0.0)
    assert identical["escaped_fraction_change"] == pytest.approx(0.0)
    assert identical["absorbed_fraction_change"] == pytest.approx(0.0)

    loaded = trace(
        Fingertip(
            FingertipParameters(),
            led=LED(emission_half_angle_deg=80.0),
        ),
        settings=settings,
    )
    metrics = evaluate(reference, loaded)
    assert field_difference(reference, loaded) == pytest.approx(
        field_difference(loaded, reference)
    )
    assert metrics["field_difference"] == pytest.approx(
        field_difference(reference, loaded)
    )
    assert set(metrics) == {
        "field_difference",
        "centroid_shift_mm",
        "escaped_fraction_change",
        "absorbed_fraction_change",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["field_difference"] > 0.0
    assert metrics["centroid_shift_mm"] > 0.0
