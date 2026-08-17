from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon

from mesh.indenter import (
    IndenterSettings,
    build_normal_indenter_fixture_at_x,
    pose_from_fixture,
)
from mesh import mesh_settings_for_level
from model import Fingertip, FingertipParameters, LED
from model.optical import OpticalMaterial
from optics import IndenterOptics, TraceSettings
from optics.cross_section.domain import (
    CrossSectionOpticsError,
    _OpticalDomain,
    _build_no_load_domain,
    _validate_domain,
)
from optics.cross_section.transport import _trace_transport
from optics.physics import interface_directions_and_reflectance
from optics.transport3d.physics import (
    Transport3DPhysicsError,
    object_interface_split,
)
from optics.transport3d.geometry import (
    AIR_INTERFACE,
    OBJECT_CONTACT_INTERFACE,
    build_transport_geometry,
)


# Synthetic unit-test value only; this is not a measured indenter property.
DEMONSTRATION_OBJECT_REFRACTIVE_INDEX = 2.0


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
        indenter_optics=IndenterOptics(
            "dielectric",
            DEMONSTRATION_OBJECT_REFRACTIVE_INDEX,
        ),
    )
    result = _trace_transport(
        domain,
        led=LED(emission_half_angle_deg=1.0e-3),
        material=OpticalMaterial(absorption_per_mm=0.0),
        settings=_object_settings(),
    )
    expected_r = (
        (1.41 - DEMONSTRATION_OBJECT_REFRACTIVE_INDEX)
        / (1.41 + DEMONSTRATION_OBJECT_REFRACTIVE_INDEX)
    ) ** 2
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
        DEMONSTRATION_OBJECT_REFRACTIVE_INDEX,
    )
    assert not bool(tir[0])
    assert reflected[0, 2] == pytest.approx(0.0)
    assert transmitted[0, 2] == pytest.approx(0.0)


def test_object_total_internal_reflection_has_no_transmitted_terminal_weight() -> None:
    angle = np.deg2rad(50.0)
    incident = np.asarray([[np.cos(angle), np.sin(angle), 0.0]])
    normal = np.asarray([[1.0, 0.0, 0.0]])
    reflected, transmitted, reflectance, tir = object_interface_split(
        np,
        incident,
        normal,
        1.41,
        1.0,
    )
    assert bool(tir[0])
    assert reflectance[0] == pytest.approx(1.0)
    assert np.allclose(transmitted[0], 0.0)
    assert np.all(np.isfinite(reflected))
    assert reflected[0, 2] == pytest.approx(0.0)


def test_reversed_object_normal_is_corrected_before_fresnel_split() -> None:
    incident = np.asarray([[0.0, 1.0, 0.0]])
    reversed_normal = np.asarray([[0.0, -1.0, 0.0]])
    with pytest.raises(Transport3DPhysicsError, match="normal"):
        object_interface_split(
            np,
            incident,
            reversed_normal,
            1.41,
            DEMONSTRATION_OBJECT_REFRACTIVE_INDEX,
        )

    corrected_normal = -reversed_normal
    reflected, transmitted, _reflectance, tir = object_interface_split(
        np,
        incident,
        corrected_normal,
        1.41,
        DEMONSTRATION_OBJECT_REFRACTIVE_INDEX,
    )
    assert not bool(tir[0])
    assert np.all(np.isfinite(reflected))
    assert np.all(np.isfinite(transmitted))


def test_contact_only_indenter_does_not_mask_air_side_p2_domain() -> None:
    tip = Fingertip(FingertipParameters())
    object_region = Point(0.0, -10.0).buffer(0.5, quad_segs=32)
    pose = SimpleNamespace(
        carrier_geometry=object_region,
        contact_patch=LineString([(0.0, -9.5), (0.0, -9.4)]),
        center_mm=(0.0, -10.0),
    )
    domain = _validate_domain(
        tip,
        outer_envelope=tip.geometry.outer_pad_geometry,
        silicone_region=tip.geometry.pad_material_geometry,
        indenter_pose=pose,
        indenter_optics=IndenterOptics("absorber"),
    )
    assert domain.accessible_region.covers(Point(0.0, -10.0))


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


def test_production_geometry_tags_only_the_mechanical_contact_edge() -> None:
    tip = Fingertip(FingertipParameters())
    mesh = tip.mesh(mesh_settings_for_level("medium"))
    arc_edges = mesh.pad.boundary_edges_for("pad_outer_arc")
    selected_edge = arc_edges[0]
    selected_key = tuple(sorted(int(value) for value in selected_edge))
    selected_index = next(
        index
        for index, edge in enumerate(mesh.pad.boundary_edges)
        if tuple(sorted(int(value) for value in edge)) == selected_key
    )
    active_node_ids = tuple(int(mesh.pad.node_ids[index]) for index in selected_edge)
    external_keys = {
        tuple(sorted(int(value) for value in edge))
        for tag in ("pad_outer_arc", "pad_outer_left", "pad_outer_right")
        for edge in mesh.pad.boundary_edges_for(tag)
    }
    arc_keys = {
        tuple(sorted(int(value) for value in edge))
        for edge in mesh.pad.boundary_edges_for("pad_outer_arc")
    }
    fixture = build_normal_indenter_fixture_at_x(
        tip.geometry,
        0.0,
        IndenterSettings(initial_gap_mm=0.0),
    )
    pose = pose_from_fixture(
        fixture,
        0.5,
        contact_patch=LineString(mesh.pad.coordinates[selected_edge]),
        active_contact_node_ids=active_node_ids,
    )

    control = build_transport_geometry(tip, mesh.pad, mesh)
    loaded = build_transport_geometry(
        tip,
        mesh.pad,
        mesh,
        indenter_pose=pose,
        indenter_optics=IndenterOptics("absorber"),
    )

    assert all(
        control.silicone.interface_tags[2 * index] == AIR_INTERFACE
        for index, edge in enumerate(mesh.pad.boundary_edges)
        if tuple(sorted(int(value) for value in edge)) in external_keys
    )
    assert loaded.silicone.interface_tags[2 * selected_index] == OBJECT_CONTACT_INTERFACE
    assert loaded.silicone.interface_tags[2 * selected_index + 1] == OBJECT_CONTACT_INTERFACE
    for index, edge in enumerate(mesh.pad.boundary_edges):
        edge_key = tuple(sorted(int(value) for value in edge))
        if edge_key in arc_keys and index != selected_index:
            assert loaded.silicone.interface_tags[2 * index] == AIR_INTERFACE
            assert loaded.silicone.interface_tags[2 * index + 1] == AIR_INTERFACE


def test_production_geometry_rejects_missing_active_contact_provenance() -> None:
    tip = Fingertip(FingertipParameters())
    mesh = tip.mesh(mesh_settings_for_level("medium"))
    fixture = build_normal_indenter_fixture_at_x(
        tip.geometry,
        0.0,
        IndenterSettings(initial_gap_mm=0.0),
    )
    pose = pose_from_fixture(
        fixture,
        0.5,
        contact_patch=LineString([(0.0, 0.0), (1.0, 0.0)]),
    )
    with pytest.raises(
        ValueError,
        match="BLOCKED_CONTACT_INTERFACE_MAPPING",
    ):
        build_transport_geometry(
            tip,
            mesh.pad,
            mesh,
            indenter_pose=pose,
            indenter_optics=IndenterOptics("absorber"),
        )
