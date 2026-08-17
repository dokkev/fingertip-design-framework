from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon

from mesh.indenter import (
    IndenterSettings,
    build_normal_indenter_fixture_at_x,
    pose_from_fixture,
)
from model import Fingertip, FingertipParameters, LED
from model.optical import OpticalMaterial
from optics import IndenterOptics, TraceSettings
from optics.cross_section.domain import (
    CrossSectionOpticsError,
    _OpticalDomain,
    _build_no_load_domain,
)
from optics.cross_section.transport import _trace_transport
from optics.physics import interface_directions_and_reflectance
from optics.transport3d.physics import object_interface_split


def _object_domain() -> _OpticalDomain:
    outer = Polygon([(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)])
    silicone = Polygon([(-5.0, -5.0), (1.0, -5.0), (1.0, 5.0), (-5.0, 5.0)])
    object_region = Point(2.0, 0.0).buffer(1.0, quad_segs=64)
    return _OpticalDomain(
        outer_envelope=outer,
        silicone_region=silicone,
        rigid_region=Polygon(),
        accessible_region=outer.difference(object_region),
        source_position_mm=(0.0, 0.0),
        source_emission_axis_2d=(1.0, 0.0),
        geometry_tolerance_mm=1.0e-9,
        indenter_region=object_region,
        contact_patch=LineString([(1.0, -0.1), (1.0, 0.1)]),
        indenter_center_mm=(2.0, 0.0),
    )


def _object_settings() -> TraceSettings:
    return TraceSettings(
        ray_count=3,
        max_interactions=1,
        minimum_ray_weight=1.0e-6,
        maximum_segment_count=32,
        grid_width=16,
        grid_height=16,
    )


def test_canonical_helper_reproduces_air_silicone_normal_incidence() -> None:
    reflected, transmitted, reflectance, tir = interface_directions_and_reflectance(
        np.asarray([1.0, 0.0]),
        np.asarray([1.0, 0.0]),
        1.5,
        1.0,
    )
    expected = ((1.5 - 1.0) / (1.5 + 1.0)) ** 2
    assert not tir
    assert transmitted is not None
    assert reflectance == pytest.approx(expected)
    np.testing.assert_allclose(reflected, [-1.0, 0.0])


def test_canonical_helper_preserves_tir() -> None:
    reflected, transmitted, reflectance, tir = interface_directions_and_reflectance(
        np.asarray([0.5, np.sqrt(3.0) / 2.0]),
        np.asarray([1.0, 0.0]),
        1.5,
        1.0,
    )
    assert tir
    assert transmitted is None
    assert reflectance == pytest.approx(1.0)
    assert np.all(np.isfinite(reflected))


def test_absorber_object_is_terminal_and_not_air_escape() -> None:
    domain = _object_domain()
    domain = replace(domain, indenter_optics=IndenterOptics("absorber"))
    result = _trace_transport(
        domain,
        led=LED(emission_half_angle_deg=1.0e-3),
        material=OpticalMaterial(absorption_per_mm=0.0),
        settings=_object_settings(),
    )
    assert result.object_absorbed_weight == pytest.approx(result.launched_weight)
    assert result.object_transmitted_weight == pytest.approx(0.0)
    assert result.object_reflected_weight == pytest.approx(0.0)
    assert result.escaped_weight == pytest.approx(0.0)


def test_dielectric_object_matches_fresnel_and_closes_interface_weight() -> None:
    domain = _object_domain()
    domain = replace(
        domain,
        indenter_optics=IndenterOptics("dielectric", 2.0),
    )
    result = _trace_transport(
        domain,
        led=LED(emission_half_angle_deg=1.0e-3),
        material=OpticalMaterial(absorption_per_mm=0.0),
        settings=_object_settings(),
    )
    expected_r = ((1.41 - 2.0) / (1.41 + 2.0)) ** 2
    assert result.object_interface_incident_weight == pytest.approx(
        result.object_reflected_weight + result.object_transmitted_weight
    )
    assert result.object_reflected_weight / result.object_interface_incident_weight == pytest.approx(
        expected_r, rel=1.0e-6
    )
    assert result.object_transmitted_weight > 0.0
    assert result.escaped_weight == pytest.approx(0.0)
    assert (
        result.escaped_weight
        + result.absorbed_weight
        + result.terminated_weight
        + result.object_absorbed_weight
        + result.object_transmitted_weight
    ) == pytest.approx(result.launched_weight)


def test_noncontacting_indenter_surface_remains_silicone_air() -> None:
    domain = replace(
        _object_domain(),
        contact_patch=LineString([(1.0, 0.5), (1.0, 0.6)]),
        indenter_optics=IndenterOptics("absorber"),
    )
    result = _trace_transport(
        domain,
        led=LED(emission_half_angle_deg=1.0e-3),
        material=OpticalMaterial(absorption_per_mm=0.0),
        settings=_object_settings(),
    )
    assert result.object_interface_incident_weight == pytest.approx(0.0)
    assert result.object_absorbed_weight == pytest.approx(0.0)


def test_planar_object_interface_directions_have_zero_longitudinal_component() -> None:
    reflected, transmitted, _reflectance, tir = object_interface_split(
        np,
        np.asarray([[0.0, -1.0, 0.0]]),
        np.asarray([[0.0, -1.0, 0.0]]),
        1.41,
        2.0,
    )
    assert not bool(tir[0])
    assert reflected[0, 2] == pytest.approx(0.0)
    assert transmitted[0, 2] == pytest.approx(0.0)


def test_indenter_optics_contract_rejects_implicit_dielectric_index() -> None:
    with pytest.raises(ValueError):
        IndenterOptics("dielectric")
    with pytest.raises(ValueError):
        IndenterOptics("absorber", 1.5)


def test_pose_reuses_fixture_geometry_and_travel() -> None:
    tip = Fingertip(FingertipParameters())
    fixture = build_normal_indenter_fixture_at_x(
        tip.geometry,
        0.0,
        IndenterSettings(initial_gap_mm=0.0),
    )
    pose = pose_from_fixture(fixture, 0.5)
    assert pose.prescribed_travel_mm == pytest.approx(0.5)
    translation = fixture.displacement_for_travel(0.5)
    assert pose.carrier_geometry.equals(
        affinity.translate(
            fixture.carrier_geometry,
            xoff=translation[0],
            yoff=translation[1],
        )
    )
    assert pose.center_mm == pytest.approx(
        tuple(
            fixture.center_mm[index] + fixture.displacement_for_travel(0.5)[index]
            for index in range(2)
        )
    )


def test_indenter_optics_requires_mechanical_contact_patch() -> None:
    tip = Fingertip(FingertipParameters())
    fixture = build_normal_indenter_fixture_at_x(
        tip.geometry,
        0.0,
        IndenterSettings(initial_gap_mm=0.0),
    )
    pose = pose_from_fixture(fixture, 0.5)
    with pytest.raises(CrossSectionOpticsError, match="BLOCKED_CONTACT_INTERFACE_MAPPING"):
        _build_no_load_domain(
            tip,
            indenter_pose=pose,
            indenter_optics=IndenterOptics("absorber"),
        )
