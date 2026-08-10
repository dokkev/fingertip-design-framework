from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from model import Fingertip, FingertipParameters, LED
from model.optical import OpticalMaterial
from optics.cross_section.domain import _OpticalDomain
from optics.cross_section.transport import (
    _interface_directions_and_reflectance,
    _trace_transport,
)
from optics import TraceSettings, trace


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


def test_silicone_air_fresnel_normal_incidence() -> None:
    material = OpticalMaterial()
    incident = np.asarray([1.0, 0.0])
    normal = np.asarray([1.0, 0.0])

    _, transmitted, reflectance = _interface_directions_and_reflectance(
        incident,
        normal,
        material.refractive_index_silicone,
        material.refractive_index_air,
    )

    expected = (
        (material.refractive_index_silicone - material.refractive_index_air)
        / (material.refractive_index_silicone + material.refractive_index_air)
    ) ** 2
    assert transmitted is not None
    assert 0.0 < reflectance < 1.0
    assert reflectance == pytest.approx(expected)


def test_silicone_air_fresnel_total_internal_reflection() -> None:
    reflected, transmitted, reflectance = (
        _interface_directions_and_reflectance(
            np.asarray([0.5, np.sqrt(3.0) / 2.0]),
            np.asarray([1.0, 0.0]),
            1.5,
            1.0,
        )
    )

    assert np.all(np.isfinite(reflected))
    assert transmitted is None
    assert reflectance == pytest.approx(1.0)


def _square_external_silicone_domain() -> _OpticalDomain:
    square = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    return _OpticalDomain(
        outer_envelope=square,
        silicone_region=square,
        rigid_region=Polygon(),
        accessible_region=square,
        source_position_mm=(5.0, 5.0),
        source_emission_axis_2d=(1.0, 0.0),
        geometry_tolerance_mm=1.0e-9,
    )


def _transport_test_settings(**overrides: object) -> TraceSettings:
    return TraceSettings(
        ray_count=3,
        max_interactions=1,
        minimum_ray_weight=1.0e-6,
        maximum_segment_count=32,
        grid_width=16,
        grid_height=16,
        **overrides,
    )


def test_external_silicone_boundary_splits_before_escape() -> None:
    result = _trace_transport(
        _square_external_silicone_domain(),
        led=LED(emission_half_angle_deg=1.0e-3),
        material=OpticalMaterial(absorption_per_mm=0.0),
        settings=_transport_test_settings(),
    )

    first_incident_weight = sum(
        segment.end_weight
        for segment in result.segments
        if segment.interaction_index == 0
    )
    assert first_incident_weight == pytest.approx(result.launched_weight)
    assert result.escaped_weight < first_incident_weight
    assert result.terminated_weight > 0.0
    assert result.escaped_weight + result.terminated_weight == pytest.approx(
        result.launched_weight
    )


def test_air_crossing_outer_envelope_escapes_without_fresnel() -> None:
    outer = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    silicone = Polygon([(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)])
    domain = _OpticalDomain(
        outer_envelope=outer,
        silicone_region=silicone,
        rigid_region=Polygon(),
        accessible_region=outer,
        source_position_mm=(1.0, 5.0),
        source_emission_axis_2d=(-1.0, 0.0),
        geometry_tolerance_mm=1.0e-9,
    )

    result = _trace_transport(
        domain,
        led=LED(emission_half_angle_deg=1.0e-3),
        material=OpticalMaterial(absorption_per_mm=0.0),
        settings=_transport_test_settings(),
    )

    assert all(segment.medium == "air" for segment in result.segments)
    assert result.escaped_weight == pytest.approx(result.launched_weight)
    assert result.absorbed_weight == pytest.approx(0.0)
    assert result.terminated_weight == pytest.approx(0.0)
