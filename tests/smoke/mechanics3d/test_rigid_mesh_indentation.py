"""Real CUDA smoke tests for the triangle-mesh indentation path."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("warp")
pytest.importorskip("newton")

import warp as wp

from mechanics3d import (
    IndentationSettings,
    Mechanics3DSettings,
    RigidIndenter3D,
    RigidPose3D,
    make_fingertip_volume_state,
    prepare_fingertip_mechanics_mesh,
    solve_fingertip_indentation,
)
from mesh.rigid_object import make_box_mesh, make_cylinder_mesh, make_sphere_mesh
from mesh.volume3d import generate_volume_mesh
from mesh.volume_types import volume_mesh_settings_for_tier
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters
from model.solid import build_fingertip_solid


def _six_volumes(vertices: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    points = vertices[tetrahedra]
    return np.einsum(
        "ij,ij->i",
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
        points[:, 3] - points[:, 0],
    )


@pytest.mark.smoke
@pytest.mark.mechanics3d
def test_nominal_triangle_mesh_indenter_deforms_and_promotes_volume_state() -> None:
    if not wp.is_device_available("cuda:0"):
        pytest.skip("rigid-mesh indentation smoke requires cuda:0")

    volume_mesh = generate_volume_mesh(
        build_fingertip_solid(FingertipModel(FingertipParameters())),
        volume_mesh_settings_for_tier("search"),
    )
    prepared = prepare_fingertip_mechanics_mesh(volume_mesh)
    object_mesh = make_sphere_mesh(2.0, subdivisions=1)
    surface_candidates = np.unique(prepared.surface_triangles["outer_compliant_arc"])
    surface_vertices = surface_candidates[
        np.abs(prepared.tet_mesh.vertices[surface_candidates, 0] - 10.0) < 1.0
    ]
    contact_y_mm = float(prepared.tet_mesh.vertices[surface_vertices, 1].max())
    object_top_extent_mm = float(object_mesh.vertices_mm[:, 1].max())
    indenter = RigidIndenter3D(
        object_mesh,
        RigidPose3D(
            (10.0, contact_y_mm + object_top_extent_mm + 0.5, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        (0.0, -1.0, 0.0),
    )
    mechanics_settings = Mechanics3DSettings(
        device="cuda:0",
        gravity=0.0,
        dt=1.0e-3,
        steps=1,
        iterations=5,
        fixed_vertex_indices=prepared.support_vertex_indices,
    )

    def run(travel_mm: float):
        return solve_fingertip_indentation(
            prepared,
            indenter,
            mechanics_settings,
            IndentationSettings(
                travel_mm=travel_mm,
                load_steps=4,
                soft_contact_margin_mm=0.02,
                soft_contact_ke=1.0e3,
                soft_contact_kd=10.0,
                rigid_body_particle_contact_buffer_size=8192,
            ),
        )

    reference_indenter = RigidIndenter3D(
        object_mesh,
        RigidPose3D(
            (10.0, contact_y_mm + object_top_extent_mm + 1.5, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        (0.0, -1.0, 0.0),
    )
    reference = solve_fingertip_indentation(
        prepared,
        reference_indenter,
        mechanics_settings,
        IndentationSettings(
            travel_mm=0.0,
            load_steps=4,
            soft_contact_margin_mm=0.02,
            soft_contact_ke=1.0e3,
            soft_contact_kd=10.0,
            rigid_body_particle_contact_buffer_size=8192,
        ),
    )
    smaller = run(0.5)
    loaded = run(0.6)
    with pytest.raises(RuntimeError, match="buffer"):
        solve_fingertip_indentation(
            prepared,
            indenter,
            mechanics_settings,
            IndentationSettings(
                travel_mm=0.6,
                load_steps=4,
                soft_contact_margin_mm=0.02,
                soft_contact_ke=1.0e3,
                soft_contact_kd=10.0,
                rigid_body_particle_contact_buffer_size=1,
            ),
        )

    np.testing.assert_allclose(
        reference.mechanics_result.rest_vertices,
        prepared.tet_mesh.vertices,
        atol=1.0e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        reference.mechanics_result.deformed_vertices,
        prepared.tet_mesh.vertices,
        atol=1.0e-5,
        rtol=0.0,
    )
    assert loaded.diagnostics["full_surface_contact"] is True
    assert loaded.diagnostics["contact_buffer_safe"] is True
    assert loaded.diagnostics["rigid_sdf_target_voxel_mm"] == 0.125
    assert int(loaded.diagnostics["max_soft_contact_count"]) > 0
    assert np.all(np.isfinite(loaded.mechanics_result.deformed_vertices))
    assert np.max(np.linalg.norm(loaded.mechanics_result.displacement, axis=1)) > 1.0e-5
    assert np.max(np.linalg.norm(loaded.mechanics_result.displacement, axis=1)) > np.max(
        np.linalg.norm(smaller.mechanics_result.displacement, axis=1)
    )
    assert np.min(_six_volumes(loaded.mechanics_result.deformed_vertices, loaded.mechanics_result.tetrahedra)) > 0.0

    state = make_fingertip_volume_state(volume_mesh, prepared, loaded.mechanics_result)
    assert state.morphology_fingerprint == volume_mesh.morphology_fingerprint
    assert state.deformed_coordinates_mm.shape == prepared.tet_mesh.vertices.shape


@pytest.mark.parametrize(
    "object_mesh",
    (
        make_sphere_mesh(0.35, subdivisions=1),
        make_cylinder_mesh(0.35, 0.7, radial_segments=8),
        make_box_mesh(0.7, 0.7, 0.7),
    ),
    ids=("sphere", "cylinder", "box"),
)
@pytest.mark.smoke
@pytest.mark.mechanics3d
def test_triangle_mesh_model_path_accepts_primitive_family(object_mesh) -> None:
    if not wp.is_device_available("cuda:0"):
        pytest.skip("rigid-mesh indentation smoke requires cuda:0")

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    tetrahedra = np.array(
        [[0, 1, 3, 4], [1, 2, 3, 6], [1, 3, 4, 6], [1, 4, 5, 6], [3, 4, 6, 7]],
        dtype=np.int32,
    )
    from mechanics3d.fingertip import FingertipMechanicsMesh
    from mechanics3d.types import TetMeshData

    prepared = FingertipMechanicsMesh(
        TetMeshData(vertices, tetrahedra),
        np.arange(8, dtype=np.int64),
        (0, 1, 2, 3),
        {},
        "primitive-family-smoke",
    )
    center = np.array((0.5, 0.5, 1.0 + float(object_mesh.vertices_mm[:, 2].max()) + 0.02))
    indenter = RigidIndenter3D(
        object_mesh,
        RigidPose3D(tuple(center), (0.0, 0.0, 0.0, 1.0)),
        (0.0, 0.0, -1.0),
    )
    result = solve_fingertip_indentation(
        prepared,
        indenter,
        Mechanics3DSettings(
            device="cuda:0",
            gravity=0.0,
            dt=1.0e-3,
            steps=1,
            iterations=5,
            fixed_vertex_indices=prepared.support_vertex_indices,
        ),
        IndentationSettings(
            travel_mm=0.05,
            load_steps=2,
            soft_contact_margin_mm=0.01,
            soft_contact_ke=1.0e3,
            soft_contact_kd=10.0,
            rigid_body_particle_contact_buffer_size=2048,
        ),
    )
    assert result.diagnostics["full_surface_contact"] is True
    assert np.all(np.isfinite(result.mechanics_result.deformed_vertices))
