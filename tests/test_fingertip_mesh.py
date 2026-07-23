"""Tests for Gmsh topology derived only from ``FingertipModel``."""

from __future__ import annotations

import math
import os
import subprocess
import sys

import pytest

from fem.fingertip_mesher import generate_fingertip_mesh
from fem.mesh_types import InvalidMeshSettings, MeshSettings, mesh_settings_for_level
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


def test_geometry_mesh_api_imports_without_kratos() -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import sys; import fem; "
                "assert not any(name.startswith('KratosMultiphysics') "
                "for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.fixture(scope="module")
def zero_clearance_medium():
    model = FingertipModel(FingertipParameters())
    return model, generate_fingertip_mesh(model, mesh_settings_for_level("medium"))


@pytest.fixture(scope="module")
def zero_clearance_fine():
    model = FingertipModel(FingertipParameters())
    return model, generate_fingertip_mesh(model, mesh_settings_for_level("fine"))


@pytest.fixture(scope="module")
def u_clearance_medium():
    model = FingertipModel(
        FingertipParameters(void_width=2.5, void_height=3.0)
    )
    return model, generate_fingertip_mesh(model, mesh_settings_for_level("medium"))


def test_default_zero_clearance_mesh_passes_validation(
    zero_clearance_medium,
) -> None:
    model, mesh = zero_clearance_medium
    assert mesh.validation.passed, mesh.validation.errors
    assert mesh.settings.level == "medium"


def test_nonzero_u_clearance_mesh_preserves_unpaired_void_boundary(
    u_clearance_medium,
) -> None:
    _, mesh = u_clearance_medium
    assert mesh.validation.passed, mesh.validation.errors
    assert mesh.boundary_edges["pad_void_unpaired"]


@pytest.mark.parametrize("fixture_name", ["zero_clearance_medium", "zero_clearance_fine"])
def test_medium_and_fine_have_positive_t3_area(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    _, mesh = request.getfixturevalue(fixture_name)
    assert mesh.quality.nonpositive_area_element_count == 0
    assert mesh.quality.minimum_triangle_angle_degrees >= 15.0


def test_pad_and_link_area_are_preserved(zero_clearance_medium) -> None:
    model, mesh = zero_clearance_medium
    assert mesh.quality.pad_mesh_area_mm2 == pytest.approx(
        model.pad_material_geometry.area, rel=1.0e-3
    )
    assert mesh.quality.carrier_mesh_area_mm2 == pytest.approx(
        model.link_geometry.area, rel=1.0e-3
    )


def test_all_source_and_adapter_boundary_tags_are_preserved(
    zero_clearance_medium,
) -> None:
    model, mesh = zero_clearance_medium
    expected = set(model.boundaries.segments) | {
        "pad_void_unpaired",
        "rigid_link_outer",
        "rigid_bond_interface",
    }
    assert set(mesh.boundary_edges) == expected
    assert all(mesh.boundary_edges[tag] for tag in model.boundaries.segments)
    assert mesh.validation.checks["semantic_edges_lie_on_source_segments"]
    assert mesh.validation.checks["no_edge_has_multiple_semantic_tags"]


def test_nonzero_contact_gaps_match_contact_pair_metadata(
    u_clearance_medium,
) -> None:
    model, mesh = u_clearance_medium
    expected = {
        pair.name: pair.initial_normal_gap for pair in model.contact_pairs
    }
    actual = {
        pair.name: pair.measured_mesh_gap_mm for pair in mesh.contact_pairs
    }
    assert actual == pytest.approx(expected, abs=mesh.settings.classification_tolerance_mm)


def test_zero_clearance_contact_nodes_are_distinct_and_coincident(
    zero_clearance_medium,
) -> None:
    model, mesh = zero_clearance_medium
    for pair in mesh.contact_pairs:
        pad_edges = mesh.boundary_edges[pair.pad_boundary_tag]
        stem_edges = mesh.boundary_edges[pair.stem_boundary_tag]
        pad_ids = {node_id for edge in pad_edges for node_id in edge.node_ids}
        stem_ids = {node_id for edge in stem_edges for node_id in edge.node_ids}
        pad_coordinates = [
            (mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm)
            for node_id in pad_ids
        ]
        stem_coordinates = [
            (mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm)
            for node_id in stem_ids
        ]
        assert pad_ids.isdisjoint(stem_ids)
        assert len(pad_coordinates) == len(stem_coordinates)
        assert max(
            min(math.dist(pad, stem) for stem in stem_coordinates)
            for pad in pad_coordinates
        ) <= model.parameters.geometry_tolerance


def test_mesh_generation_is_deterministic(zero_clearance_medium) -> None:
    model, first = zero_clearance_medium
    second = generate_fingertip_mesh(model, mesh_settings_for_level("medium"))
    assert second.canonical_signature() == first.canonical_signature()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bulk_target_size_mm": 0.0},
        {"contact_boundary_target_size_mm": -0.1},
        {"bulk_target_size_mm": 0.2, "contact_boundary_target_size_mm": 0.3},
        {"minimum_angle_target_degrees": 60.0},
    ],
)
def test_invalid_mesh_settings_are_rejected(kwargs: dict[str, float]) -> None:
    values = {
        "level": "medium",
        "bulk_target_size_mm": 0.75,
        "contact_boundary_target_size_mm": 0.35,
    }
    values.update(kwargs)
    with pytest.raises(InvalidMeshSettings):
        MeshSettings(**values)
