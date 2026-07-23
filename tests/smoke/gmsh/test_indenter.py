"""Geometry-only tests for the FEM-owned Phase 4I rigid fixture."""

from __future__ import annotations

import math

import pytest

from mesh.indenter import (
    IndenterSettings,
    InvalidIndenterSettings,
    build_indenter_fixture,
    generate_indenter_mesh,
)
from fem.indentation import (
    IndentationSettings,
    InvalidIndentationSettings,
)
from fem.kratos_settings import INDENTATION_CONTACT_GROUPS
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


@pytest.fixture(scope="module")
def fixture_geometry():
    model = FingertipModel(FingertipParameters())
    return model, build_indenter_fixture(model)


def test_crown_point_and_outward_normal_come_from_pad_boundary(
    fixture_geometry,
) -> None:
    model, fixture = fixture_geometry
    crown = fixture.frame.point_mm
    assert model.boundaries.segments["pad_outer_arc"].geometry.distance(
        __import__("shapely.geometry", fromlist=["Point"]).Point(crown)
    ) <= model.parameters.geometry_tolerance
    assert crown == pytest.approx((0.0, -18.0))
    assert fixture.frame.pad_outward_normal == pytest.approx((0.0, -1.0))
    assert fixture.frame.loading_direction == pytest.approx((0.0, 1.0))
    assert math.isclose(
        fixture.frame.tangent[0] * fixture.frame.pad_outward_normal[0]
        + fixture.frame.tangent[1] * fixture.frame.pad_outward_normal[1],
        0.0,
        abs_tol=1.0e-12,
    )


def test_indenter_center_tangent_and_initial_gap_contract(fixture_geometry) -> None:
    model, fixture = fixture_geometry
    expected = (
        fixture.frame.point_mm[0]
        + fixture.settings.radius_mm * fixture.frame.pad_outward_normal[0],
        fixture.frame.point_mm[1]
        + fixture.settings.radius_mm * fixture.frame.pad_outward_normal[1],
    )
    assert fixture.center_mm == pytest.approx(expected)
    assert model.boundaries.segments["pad_outer_arc"].geometry.distance(
        fixture.contact_arc
    ) == pytest.approx(0.0, abs=1.0e-10)


def test_indenter_contact_mesh_orientation_points_toward_pad(
    fixture_geometry,
) -> None:
    _, fixture = fixture_geometry
    mesh = generate_indenter_mesh(fixture, 0.35)
    outward = [0.0, 0.0]
    for edge in mesh.contact_edges:
        first, second = (mesh.nodes[node_id] for node_id in edge.node_ids)
        outward[0] += second.y_mm - first.y_mm
        outward[1] -= second.x_mm - first.x_mm
    length = math.hypot(*outward)
    normal = (outward[0] / length, outward[1] / length)
    assert sum(
        normal[index] * fixture.frame.loading_direction[index]
        for index in range(2)
    ) > 0.99
    assert mesh.maximum_contact_edge_length_mm <= 0.35 * 1.01


def test_prescribed_motion_is_one_rigid_translation(fixture_geometry) -> None:
    _, fixture = fixture_geometry
    assert fixture.displacement_for_travel(0.25) == pytest.approx((0.0, 0.25))
    with pytest.raises(InvalidIndenterSettings):
        fixture.displacement_for_travel(-0.1)


def test_four_contact_pair_mapping_is_explicit() -> None:
    assert INDENTATION_CONTACT_GROUPS == (
        ("external_pad_indenter", "PadOuterArc", "IndenterContactArc"),
        ("internal_left", "PadCutoutLeft", "StemLeft"),
        ("internal_right", "PadCutoutRight", "StemRight"),
        ("internal_bottom", "PadCutoutBottom", "StemBottom"),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius_mm": 0.0},
        {"radius_mm": -1.0},
        {"initial_gap_mm": -1.0e-3},
        {"thickness_mm": 0.0},
        {"contact_half_angle_degrees": 90.0},
    ],
)
def test_invalid_indenter_settings_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(InvalidIndenterSettings):
        IndenterSettings(**kwargs)


@pytest.mark.parametrize(
    "indentation, steps",
    [(0.0, 48), (-0.1, 48), (0.25, 0), (math.inf, 48)],
)
def test_invalid_indentation_settings_are_rejected(
    indentation: float, steps: int
) -> None:
    with pytest.raises(InvalidIndentationSettings):
        IndentationSettings(indentation, steps)


def test_baseline_capture_depths_are_exact_solution_steps() -> None:
    settings = IndentationSettings(1.5, 48)
    assert settings.capture_step(0.5) == 16
    assert settings.capture_step(1.0) == 32
    assert settings.capture_step(1.5) == 48

