"""Nominal authoritative 3D fingertip to Newton VBD smoke path."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("warp")
pytest.importorskip("newton")

import warp as wp

from physics import NewtonSettings, prepare_fingertip_mesh
from physics.newton.solve import solve
from mesh.volume.mesh import generate_volume_mesh
from mesh.volume.contracts import volume_mesh_settings_for_tier
from model.fingertip_model import FingertipModel
from model.fingertip_model import FingertipParameters
from model.solid import build_fingertip_solid


@pytest.mark.smoke
@pytest.mark.physics
def test_nominal_fingertip_volume_mesh_advances_on_cuda() -> None:
    if not wp.is_device_available("cuda:0"):
        pytest.skip("nominal physics smoke requires cuda:0")

    model = FingertipModel(FingertipParameters())
    solid = build_fingertip_solid(model)
    volume_mesh = generate_volume_mesh(solid, volume_mesh_settings_for_tier("search"))
    prepared = prepare_fingertip_mesh(volume_mesh)

    assert volume_mesh.fingertip.validation.passed, volume_mesh.fingertip.validation.errors
    assert volume_mesh.nodes
    assert volume_mesh.tetrahedra
    assert prepared.tet_mesh.vertices.shape[0] == len(volume_mesh.nodes)
    assert prepared.tet_mesh.tetrahedra.shape[0] == len(volume_mesh.tetrahedra)
    assert np.all(
        (prepared.tet_mesh.tetrahedra >= 0)
        & (prepared.tet_mesh.tetrahedra < prepared.tet_mesh.vertices.shape[0])
    )
    np.testing.assert_array_equal(
        prepared.source_node_ids,
        np.asarray(sorted(volume_mesh.nodes), dtype=np.int64),
    )
    assert set(prepared.surface_triangles) == set(volume_mesh.surface_triangles)
    assert prepared.support_vertex_indices

    result = solve(
        prepared.tet_mesh,
        settings=NewtonSettings(
            device="cuda:0",
            gravity=0.0,
            steps=1,
            iterations=5,
            fixed_vertex_indices=prepared.support_vertex_indices,
        ),
    )

    assert result.deformed_vertices.shape == prepared.tet_mesh.vertices.shape
    np.testing.assert_array_equal(result.tetrahedra, prepared.tet_mesh.tetrahedra)
    assert np.all(np.isfinite(result.deformed_vertices))
    assert np.all(np.isfinite(result.displacement))

    zero_load_tolerance_mm = 1.0e-5
    np.testing.assert_allclose(
        result.rest_vertices,
        prepared.tet_mesh.vertices,
        atol=zero_load_tolerance_mm,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.deformed_vertices,
        prepared.tet_mesh.vertices,
        atol=zero_load_tolerance_mm,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.deformed_vertices[list(prepared.support_vertex_indices)],
        prepared.tet_mesh.vertices[list(prepared.support_vertex_indices)],
        atol=zero_load_tolerance_mm,
        rtol=0.0,
    )
