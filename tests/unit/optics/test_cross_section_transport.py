from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point

from model import Fingertip, FingertipParameters, LED
from optics import TraceSettings, trace


@pytest.fixture(scope="module")
def reference_mesh():
    tip = Fingertip(FingertipParameters())
    return tip, tip.mesh()


@pytest.fixture(scope="module")
def settings() -> TraceSettings:
    return TraceSettings(
        ray_count=31,
        grid_width=48,
        grid_height=48,
        maximum_segment_count=5000,
    )


def test_transport_is_deterministic_and_conserves_weight(
    settings: TraceSettings,
) -> None:
    tip = Fingertip(FingertipParameters(void_width=1.0, void_height=2.0))

    first = trace(tip, settings=settings)
    second = trace(tip, settings=settings)

    assert first.density.shape == (48, 48)
    assert first.x_edges.shape == (49,)
    assert first.y_edges.shape == (49,)
    assert np.all(np.isfinite(first.density))
    assert np.all(first.density >= 0.0)
    assert float(np.max(first.density)) > 0.0
    assert any(segment.medium == "air" for segment in first.segments)
    assert any(segment.medium == "silicone" for segment in first.segments)
    np.testing.assert_allclose(
        first.escaped_weight
        + first.absorbed_weight
        + first.terminated_weight,
        first.launched_weight,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    np.testing.assert_array_equal(first.density, second.density)
    np.testing.assert_array_equal(first.optical_mask, second.optical_mask)
    assert first.segments == second.segments
    assert first.outer_envelope.equals(second.outer_envelope)
    assert first.silicone_region.equals(second.silicone_region)
    assert first.launched_weight == second.launched_weight
    assert first.escaped_weight == second.escaped_weight
    assert first.absorbed_weight == second.absorbed_weight
    assert first.terminated_weight == second.terminated_weight


def test_led_power_scales_raw_transport() -> None:
    parameters = FingertipParameters()
    baseline_tip = Fingertip(parameters, led=LED(relative_radiant_power=1.0))
    doubled_tip = Fingertip(parameters, led=LED(relative_radiant_power=2.0))
    settings = TraceSettings(
        ray_count=31,
        grid_width=48,
        grid_height=48,
        maximum_segment_count=5000,
    )

    baseline = trace(baseline_tip, settings=settings)
    doubled = trace(doubled_tip, settings=settings)

    np.testing.assert_allclose(
        doubled.density,
        2.0 * baseline.density,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        [
            doubled.launched_weight,
            doubled.escaped_weight,
            doubled.absorbed_weight,
            doubled.terminated_weight,
        ],
        2.0
        * np.asarray(
            [
                baseline.launched_weight,
                baseline.escaped_weight,
                baseline.absorbed_weight,
                baseline.terminated_weight,
            ]
        ),
        rtol=1.0e-12,
        atol=1.0e-14,
    )


def test_loaded_cutout_gap_starts_in_air_and_reaches_silicone(
    reference_mesh,
    settings: TraceSettings,
) -> None:
    tip, mesh = reference_mesh
    displacement = np.zeros_like(mesh.coordinates)
    cutout_bottom = mesh.boundary_node_indices_for("pad_cutout_bottom")
    displacement[cutout_bottom, 1] = -0.05
    loaded_mesh = mesh.deformed(displacement)

    loaded = trace(tip, loaded_mesh, settings)

    assert loaded.segments[0].medium == "air"
    assert any(segment.medium == "silicone" for segment in loaded.segments)
    assert loaded.air_region.covers(
        Point(tip.led_source[0], tip.led_source[1] - 0.025)
    )
    tolerance = tip.parameters.geometry_tolerance
    assert loaded.outer_envelope.buffer(tolerance).covers(
        loaded.silicone_region
    )
