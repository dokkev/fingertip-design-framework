"""Unit tests for Phase 4I solver-independent measurements."""

from __future__ import annotations

import numpy as np
import pytest

from fem.fingertip_mesher import generate_fingertip_mesh
from fem.indenter_fixture import build_indenter_fixture
from fem.indentation_postprocess import (
    compressive_indenter_reaction,
    contact_width_metrics,
    extract_outer_arc_profile,
    interpolate_profile,
    ordered_boundary_node_ids,
    pad_strain_det_f_statistics,
    profile_error_metrics,
    unique_projected_reaction,
)
from fem.mesh_types import mesh_settings_for_level
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


@pytest.fixture(scope="module")
def meshed_model():
    model = FingertipModel(FingertipParameters())
    mesh = generate_fingertip_mesh(model, mesh_settings_for_level("medium"))
    return model, mesh


def test_reaction_sign_and_unique_node_summation() -> None:
    reactions = {1: (0.0, 2.0), 2: (0.0, 3.0)}
    assert unique_projected_reaction(reactions, [1, 1, 2], (0.0, 1.0)) == 5.0
    assert compressive_indenter_reaction(
        reactions, [1, 1, 2], (0.0, 1.0)
    ) == 5.0


def test_contact_width_uses_active_nodes_and_source_edges(meshed_model) -> None:
    model, mesh = meshed_model
    ordered = ordered_boundary_node_ids(
        mesh,
        "pad_outer_arc",
        model.boundaries.segments["pad_outer_arc"].geometry,
    )
    center = len(ordered) // 2
    active = ordered[center - 1 : center + 2]
    fixture = build_indenter_fixture(model)
    result = contact_width_metrics(mesh, active, fixture.frame.tangent)
    assert result["active_node_count"] == 3
    assert result["active_edge_count"] == 2
    assert result["chord_width_mm"] > 0.0
    assert result["arc_length_mm"] > 0.0


def test_outer_arc_profile_is_ordered_and_complete(meshed_model) -> None:
    model, mesh = meshed_model
    fixture = build_indenter_fixture(model)
    displacements = {node_id: (0.0, 0.0) for node_id in mesh.nodes}
    profile = extract_outer_arc_profile(
        model, mesh, displacements, fixture.frame
    )
    assert len(profile) == len(mesh.boundary_edges["pad_outer_arc"]) + 1
    assert profile[0]["normalized_arc_coordinate"] == 0.0
    assert profile[-1]["normalized_arc_coordinate"] == 1.0
    assert all(
        second["normalized_arc_coordinate"] > first["normalized_arc_coordinate"]
        for first, second in zip(profile, profile[1:])
    )
    assert all(record["local_normal_displacement_mm"] == 0.0 for record in profile)


def test_profile_interpolation_and_synthetic_convergence_metric() -> None:
    profile = [
        {
            "normalized_arc_coordinate": coordinate,
            "local_normal_displacement_mm": 2.0 * coordinate,
            "local_tangential_displacement_mm": -coordinate,
        }
        for coordinate in (0.0, 0.5, 1.0)
    ]
    interpolated = interpolate_profile(profile, (0.0, 0.25, 0.5, 0.75, 1.0))
    assert interpolated["normal_displacement_mm"] == pytest.approx(
        (0.0, 0.5, 1.0, 1.5, 2.0)
    )
    metric = profile_error_metrics(
        [value * 1.01 for value in interpolated["normal_displacement_mm"]],
        interpolated["normal_displacement_mm"],
        1.0e-5,
    )
    assert metric["relative_l2_error"] == pytest.approx(0.01)


def test_pad_strain_statistics_exclude_both_rigid_domains(meshed_model) -> None:
    _, mesh = meshed_model
    displacements = {
        node_id: ((100.0, -100.0) if node.domain == "rigid_carrier" else (0.0, 0.0))
        for node_id, node in mesh.nodes.items()
    }
    result = pad_strain_det_f_statistics(mesh, displacements)
    assert result["rigid_domains_excluded"]
    assert result["pad_element_count"] == len(mesh.pad_elements)
    assert result["det_f"]["min"] == pytest.approx(1.0)
    assert result["det_f"]["max"] == pytest.approx(1.0)
    assert result["maximum_absolute_green_lagrange_component"]["value"] == pytest.approx(0.0)
