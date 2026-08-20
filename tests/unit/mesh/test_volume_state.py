from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("gmsh")

from mesh import FingertipVolumeState, generate_volume_mesh, volume_mesh_settings_for_tier
from mesh.volume_types import SurfaceTriangle
from model import Fingertip


@pytest.fixture(scope="module")
def search_mesh():
    tip = Fingertip()
    settings = volume_mesh_settings_for_tier("search")
    return tip, generate_volume_mesh(tip.solid(), settings)


def test_direct_volume_mesh_matches_repeated_generation(search_mesh) -> None:
    tip, facade_mesh = search_mesh
    manual_mesh = generate_volume_mesh(tip.solid(), volume_mesh_settings_for_tier("search"))

    assert facade_mesh.morphology_fingerprint == manual_mesh.morphology_fingerprint
    assert len(facade_mesh.nodes) == len(manual_mesh.nodes)
    assert len(facade_mesh.tetrahedra) == len(manual_mesh.tetrahedra)
    assert facade_mesh.semantic_surface_tags == manual_mesh.semantic_surface_tags
    assert {
        tag: len(triangles) for tag, triangles in facade_mesh.surface_triangles.items()
    } == {
        tag: len(triangles) for tag, triangles in manual_mesh.surface_triangles.items()
    }


def test_reference_state_uses_sorted_source_order_and_identity_coordinates(search_mesh) -> None:
    _, volume_mesh = search_mesh
    state = FingertipVolumeState.reference(volume_mesh)

    assert state.source_node_ids == tuple(sorted(volume_mesh.nodes))
    np.testing.assert_array_equal(state.reference_coordinates_mm, state.deformed_coordinates_mm)
    np.testing.assert_array_equal(state.displacement_mm, np.zeros_like(state.displacement_mm))
    assert state.tetrahedra == volume_mesh.tetrahedra
    assert state.surface_triangles == volume_mesh.surface_triangles
    assert state.morphology_fingerprint == volume_mesh.morphology_fingerprint
    assert not state.deformed_coordinates_mm.flags.writeable


@pytest.mark.parametrize("bad_coordinates", [
    np.zeros((1, 3)),
    np.asarray([[np.nan, 0.0, 0.0]]),
])
def test_state_rejects_bad_coordinate_shape_or_finiteness(search_mesh, bad_coordinates) -> None:
    _, volume_mesh = search_mesh
    with pytest.raises(ValueError):
        FingertipVolumeState.from_deformed_coordinates(volume_mesh, bad_coordinates)


def test_state_rejects_degenerate_semantic_surface(search_mesh) -> None:
    _, volume_mesh = search_mesh
    tag = volume_mesh.semantic_surface_tags[0]
    triangle = volume_mesh.surface_triangles[tag][0]
    bad_triangle = SurfaceTriangle(
        triangle.id,
        (triangle.node_ids[0], triangle.node_ids[0], triangle.node_ids[2]),
        tag,
        triangle.domain,
    )
    bad_surfaces = dict(volume_mesh.surface_triangles)
    bad_surfaces[tag] = (bad_triangle, *bad_surfaces[tag][1:])
    invalid_mesh = replace(volume_mesh, surface_triangles=bad_surfaces)

    with pytest.raises(ValueError, match="degenerate"):
        FingertipVolumeState.reference(invalid_mesh)


def test_state_rejects_invalid_source_node_correspondence(search_mesh) -> None:
    _, volume_mesh = search_mesh
    first_id = min(volume_mesh.nodes)
    first = volume_mesh.nodes[first_id]
    invalid_nodes = dict(volume_mesh.nodes)
    invalid_nodes[first_id] = replace(first, id=first_id + 1000000)
    invalid_mesh = replace(volume_mesh, nodes=invalid_nodes)

    with pytest.raises(ValueError, match="node keys and source"):
        FingertipVolumeState.reference(invalid_mesh)
