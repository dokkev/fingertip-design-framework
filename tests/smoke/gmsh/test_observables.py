"""Extractor-level tests for the Phase 4K mechanical transfer map."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Point

from fem.fingertip_mesher import generate_fingertip_mesh
from fem.indenter_fixture import build_indenter_fixture_at_location
from fem.mechanical_transfer_map import (
    ReferenceBoundaryChain,
    TransferMapSettings,
    integrate_nodal_contact_distribution,
    interpolate_linear_chain_field,
    order_open_edge_chain,
    reference_outer_arc_chain,
    sample_observation_sidewalls,
    strict_json_round_trip,
)
from fem.mesh_types import mesh_settings_for_level
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


def test_open_chain_order_is_independent_of_edge_order_direction_and_ids() -> None:
    coordinates = {
        91: (0.0, 0.0),
        7: (1.0, 0.0),
        400: (2.0, 0.0),
        13: (3.0, 0.0),
    }
    edges = [(400, 13), (400, 7), (91, 7)]
    assert order_open_edge_chain(edges, coordinates, (0.0, 0.0)) == (
        91,
        7,
        400,
        13,
    )
    assert order_open_edge_chain(
        list(reversed([(second, first) for first, second in edges])),
        coordinates,
        (3.0, 0.0),
    ) == (13, 400, 7, 91)


def test_linear_segment_interpolation_reproduces_affine_field() -> None:
    chain = ReferenceBoundaryChain(
        node_ids=(40, 3, 91),
        points_mm=((0.0, 0.0), (1.0, 0.0), (3.0, 0.0)),
        cumulative_length_mm=(0.0, 1.0, 3.0),
        total_length_mm=3.0,
    )
    values = {
        node_id: (2.0 * point[0] + 1.0, -3.0 * point[0] + 4.0)
        for node_id, point in zip(chain.node_ids, chain.points_mm)
    }
    value, point, tangent = interpolate_linear_chain_field(chain, values, 0.5)
    assert point == pytest.approx((1.5, 0.0))
    assert value == pytest.approx((4.0, -0.5))
    assert tangent == pytest.approx((1.0, 0.0))


@pytest.fixture(scope="module")
def model_mesh():
    model = FingertipModel(FingertipParameters())
    mesh = generate_fingertip_mesh(model, mesh_settings_for_level("medium"))
    return model, mesh


def test_two_sidewall_normals_are_outward_and_bulging_is_positive(
    model_mesh,
) -> None:
    model, mesh = model_mesh
    chain = reference_outer_arc_chain(model, mesh)
    zero = {node_id: (0.0, 0.0) for node_id in mesh.nodes}
    samples = sample_observation_sidewalls(
        model, chain, zero, TransferMapSettings()
    )
    for side in ("left", "right"):
        assert len(samples[side]) == 41
        for row in samples[side]:
            normal = (
                row["reference_outward_normal_x"],
                row["reference_outward_normal_y"],
            )
            assert math.hypot(*normal) == pytest.approx(1.0)
            probe = Point(
                row["reference_x_mm"] + 1.0e-4 * normal[0],
                row["reference_y_mm"] + 1.0e-4 * normal[1],
            )
            assert not model.pad_material_geometry.covers(probe)
            assert row["u_normal_mm"] == pytest.approx(0.0)
        assert samples[side][0]["eta"] == 0.0
        assert samples[side][-1]["eta"] == 1.0


def test_uniform_and_linear_pressure_centroids_match_quadrature() -> None:
    uniform = integrate_nodal_contact_distribution(
        xi=(0.0, 0.5, 1.0),
        pressure=(2.0, 2.0, 2.0),
        nodal_area=(1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0),
        active=(True, True, True),
        pressure_tolerance=1.0e-12,
    )
    assert uniform["integrated_contact_resultant_n"] == pytest.approx(2.0)
    assert uniform["xi_centroid"] == pytest.approx(0.5)
    linear = integrate_nodal_contact_distribution(
        xi=(0.0, 0.5, 1.0),
        pressure=(0.0, 1.0, 2.0),
        nodal_area=(1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0),
        active=(True, True, True),
        pressure_tolerance=1.0e-12,
    )
    assert linear["integrated_contact_resultant_n"] == pytest.approx(1.0)
    assert linear["xi_centroid"] == pytest.approx(2.0 / 3.0)


def test_zero_contact_is_safe_and_strictly_serializable() -> None:
    result = integrate_nodal_contact_distribution(
        xi=(0.0, 1.0),
        pressure=(0.0, 0.0),
        nodal_area=(0.5, 0.5),
        active=(False, False),
        pressure_tolerance=1.0e-12,
    )
    assert result["integrated_contact_resultant_n"] == 0.0
    assert result["contact_length_mm"] == 0.0
    assert result["xi_centroid"] is None
    assert strict_json_round_trip(result) == result


def test_contact_resultant_projects_to_the_global_loading_direction() -> None:
    result = integrate_nodal_contact_distribution(
        xi=(0.0, 1.0),
        pressure=(2.0, 2.0),
        nodal_area=(0.5, 0.5),
        active=(True, True),
        pressure_tolerance=1.0e-12,
        global_projection_factors=(0.5, 0.5),
    )
    assert result["integrated_contact_normal_magnitude_n"] == pytest.approx(2.0)
    assert result["integrated_contact_resultant_n"] == pytest.approx(1.0)
    assert result["xi_centroid"] == pytest.approx(0.5)


@pytest.mark.parametrize("xi", [0.2, 0.5, 0.8])
def test_location_fixture_preserves_global_load_and_targets_requested_arc(
    model_mesh,
    xi: float,
) -> None:
    model, _ = model_mesh
    fixture = build_indenter_fixture_at_location(model, xi)
    arc = model.boundaries.segments["pad_outer_arc"].geometry
    assert fixture.frame.arc_distance_mm / arc.length == pytest.approx(xi)
    assert fixture.frame.loading_direction == pytest.approx((0.0, 1.0))
    assert fixture.contact_arc.distance(
        Point(fixture.frame.point_mm)
    ) == pytest.approx(0.0, abs=1.0e-9)
