from __future__ import annotations

import numpy as np
from shapely.geometry import Point

from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters
from model.fingertip_sensor_model import FingertipSensorModel
from optics.cross_section import (
    CrossSectionTraceSettings,
    build_no_load_optical_domain,
    trace_cross_section_transport,
)


def _u_clearance_sensor() -> FingertipSensorModel:
    geometry = FingertipModel(
        FingertipParameters(void_width=1.0, void_height=2.0)
    )
    return FingertipSensorModel.from_geometry(geometry)


def test_led_source_is_at_the_stem_tip() -> None:
    sensor = _u_clearance_sensor()
    assert sensor.led_source_position_2d == (
        0.0,
        sensor.geometry.parameters.stem_tip_y,
    )


def test_u_clearance_transport_is_deterministic_and_preserves_geometry() -> None:
    sensor = _u_clearance_sensor()
    model = sensor.geometry
    domain = build_no_load_optical_domain(sensor)
    settings = CrossSectionTraceSettings(
        ray_count=41,
        grid_width=64,
        grid_height=72,
        maximum_segment_count=5000,
    )
    trace_origin = Point(
        domain.source_position_mm[0],
        domain.source_position_mm[1] - settings.source_epsilon_mm,
    )
    assert domain.outer_envelope.covers(trace_origin)
    assert domain.accessible_region.covers(trace_origin)
    assert not domain.silicone_region.covers(trace_origin)
    assert not domain.rigid_region.covers(trace_origin)

    original_areas = (
        model.outer_pad_geometry.area,
        model.pad_material_geometry.area,
        model.link_geometry.area,
    )
    original_bounds = (
        model.outer_pad_geometry.bounds,
        model.pad_material_geometry.bounds,
        model.link_geometry.bounds,
    )
    original_boundary_tags = tuple(model.boundaries.segments)

    first = trace_cross_section_transport(
        domain,
        led=sensor.led,
        material=sensor.optical_material,
        settings=settings,
    )
    second = trace_cross_section_transport(
        domain,
        led=sensor.led,
        material=sensor.optical_material,
        settings=settings,
    )

    assert first.weighted_path_density.shape == (72, 64)
    assert first.optical_mask.shape == first.weighted_path_density.shape
    assert len(first.x_edges_mm) == 65
    assert len(first.y_edges_mm) == 73
    assert np.all(np.isfinite(first.weighted_path_density))
    assert np.all(first.weighted_path_density >= 0.0)
    assert float(np.max(first.weighted_path_density)) > 0.0
    assert any(segment.medium == "air" for segment in first.segments)
    assert any(segment.medium == "silicone" for segment in first.segments)
    assert first.segments[0].medium == "air"
    assert first.launched_weight == 1.0
    np.testing.assert_allclose(
        first.escaped_weight
        + first.absorbed_weight
        + first.terminated_weight,
        first.launched_weight,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    np.testing.assert_array_equal(
        first.weighted_path_density,
        second.weighted_path_density,
    )
    np.testing.assert_array_equal(first.optical_mask, second.optical_mask)
    assert first.segments == second.segments
    assert first.escaped_weight == second.escaped_weight
    assert first.absorbed_weight == second.absorbed_weight
    assert first.terminated_weight == second.terminated_weight

    assert original_areas == (
        model.outer_pad_geometry.area,
        model.pad_material_geometry.area,
        model.link_geometry.area,
    )
    assert original_bounds == (
        model.outer_pad_geometry.bounds,
        model.pad_material_geometry.bounds,
        model.link_geometry.bounds,
    )
    assert original_boundary_tags == tuple(model.boundaries.segments)
