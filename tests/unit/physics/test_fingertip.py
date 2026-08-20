"""Contracts for preserving the authoritative volume-mesh topology."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("gmsh")

from mesh.volume.mesh import generate_volume_mesh
from mesh.volume.contracts import VolumeMeshValidation, volume_mesh_settings_for_tier
from finger.fingertip_geometry import FingertipModel
from finger.fingertip_parameters import FingertipParameters
from finger.extrusion import build_fingertip_solid
from physics import (
    NewtonResult,
    make_fingertip_volume_state,
    prepare_fingertip_mesh,
)


@pytest.fixture(scope="module")
def volume_mesh():
    model = FingertipModel(FingertipParameters())
    solid = build_fingertip_solid(model)
    return generate_volume_mesh(solid, volume_mesh_settings_for_tier("search"))


def test_adapter_preserves_fea_node_order_and_surface_provenance(volume_mesh) -> None:
    prepared = prepare_fingertip_mesh(volume_mesh)

    assert tuple(prepared.source_node_ids) == tuple(sorted(volume_mesh.nodes))
    assert prepared.morphology_fingerprint == volume_mesh.morphology_fingerprint
    assert set(prepared.surface_triangles) == set(volume_mesh.surface_triangles)
    assert prepared.tet_mesh.vertices.shape[0] == len(volume_mesh.nodes)
    assert prepared.tet_mesh.tetrahedra.shape[0] == len(volume_mesh.tetrahedra)


def test_adapter_translates_all_connectivity_through_one_local_mapping(volume_mesh) -> None:
    prepared = prepare_fingertip_mesh(volume_mesh)
    local = {node_id: index for index, node_id in enumerate(sorted(volume_mesh.nodes))}

    expected_tetrahedra = np.asarray(
        [[local[node_id] for node_id in tetrahedron.node_ids] for tetrahedron in volume_mesh.tetrahedra],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(prepared.tet_mesh.tetrahedra, expected_tetrahedra)
    for tag, triangles in volume_mesh.surface_triangles.items():
        expected = np.asarray(
            [[local[node_id] for node_id in triangle.node_ids] for triangle in triangles],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(prepared.surface_triangles[tag], expected)

    support_source_ids = {
        node_id
        for tag in ("support_bond_left", "support_bond_right")
        for triangle in volume_mesh.surface_triangles[tag]
        for node_id in triangle.node_ids
    }
    assert prepared.support_vertex_indices == tuple(sorted(local[node_id] for node_id in support_source_ids))


def test_adapter_rejects_invalid_source_volume_mesh(volume_mesh) -> None:
    invalid = replace(
        volume_mesh,
        validation=VolumeMeshValidation(False, {"synthetic_failure": False}, ("synthetic_failure",)),
    )
    with pytest.raises(ValueError, match="invalid FingertipVolumeMesh"):
        prepare_fingertip_mesh(invalid)


def test_mechanics_result_promotes_without_reordering_or_remeshing(volume_mesh) -> None:
    prepared = prepare_fingertip_mesh(volume_mesh)
    result = NewtonResult(
        rest_vertices=prepared.tet_mesh.vertices,
        deformed_vertices=prepared.tet_mesh.vertices.copy(),
        tetrahedra=prepared.tet_mesh.tetrahedra,
        steps=1,
    )

    state = make_fingertip_volume_state(volume_mesh, prepared, result)

    assert state.source_node_ids == tuple(prepared.source_node_ids.tolist())
    np.testing.assert_array_equal(state.deformed_coordinates_mm, prepared.tet_mesh.vertices)
    np.testing.assert_allclose(
        state.reference_coordinates_mm,
        prepared.tet_mesh.vertices,
        rtol=0.0,
        atol=1.0e-6,
    )
    assert state.tetrahedra == volume_mesh.tetrahedra
    assert state.surface_triangles == volume_mesh.surface_triangles
    assert state.morphology_fingerprint == volume_mesh.morphology_fingerprint


def test_mechanics_result_promotion_rejects_topology_mismatch(volume_mesh) -> None:
    prepared = prepare_fingertip_mesh(volume_mesh)
    mismatched = prepared.tet_mesh.tetrahedra.copy()
    mismatched[0] = mismatched[0, ::-1]
    result = NewtonResult(
        rest_vertices=prepared.tet_mesh.vertices,
        deformed_vertices=prepared.tet_mesh.vertices,
        tetrahedra=mismatched,
        steps=1,
    )

    with pytest.raises(ValueError, match="topology"):
        make_fingertip_volume_state(volume_mesh, prepared, result)
