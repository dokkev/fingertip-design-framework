"""Real CUDA smoke tests for the triangle-mesh indentation path."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("warp")
pytest.importorskip("newton")

import warp as wp

from contact import (
    FirstContactSettings,
    canonical_sphere_alignment,
    find_first_contact,
    intersects,
    make_outer_compliant_surface,
)
from physics import (
    IndentationSettings,
    NewtonSettings,
    RigidIndenter3D,
    make_fingertip_volume_state,
    prepare_fingertip_mesh,
    solve_fingertip_indentation,
)
from mesh.rigid.object import RigidPose3D
from physics.newton.vbd import solve_newton_vbd_indentation
from mesh.rigid.carrier import make_distal_phalanx_mesh
from mesh.rigid.object import make_box_mesh, make_cylinder_mesh, make_sphere_mesh
from mesh.volume.mesh import generate_volume_mesh
from mesh.volume.contracts import volume_mesh_settings_for_tier
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
@pytest.mark.physics
def test_nominal_triangle_mesh_indenter_deforms_and_promotes_volume_state() -> None:
    if not wp.is_device_available("cuda:0"):
        pytest.skip("rigid-mesh indentation smoke requires cuda:0")

    volume_mesh = generate_volume_mesh(
        build_fingertip_solid(FingertipModel(FingertipParameters())),
        volume_mesh_settings_for_tier("search"),
    )
    prepared = prepare_fingertip_mesh(volume_mesh)
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
    mechanics_settings = NewtonSettings(
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
        ),
    )
    smaller = run(0.5)
    loaded = run(0.6)
    loaded_with_visual_carrier = solve_newton_vbd_indentation(
        prepared,
        indenter,
        mechanics_settings,
        IndentationSettings(
            travel_mm=0.6,
            load_steps=4,
            soft_contact_margin_mm=0.02,
            soft_contact_ke=1.0e3,
            soft_contact_kd=10.0,
        ),
        visual_carrier_mesh=make_distal_phalanx_mesh(volume_mesh.solid).surface_mesh,
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
    assert loaded.diagnostics["contact_buffer_status"] == "not_applicable_for_kinematic_indenter"
    assert loaded.diagnostics["rigid_sdf_target_voxel_mm"] == 0.125
    assert int(loaded.diagnostics["max_soft_contact_count"]) > 0
    assert int(loaded.diagnostics["max_soft_contact_overflow"]) == 0
    assert int(loaded.diagnostics["max_rigid_contact_overflow"]) == 0
    assert np.all(np.isfinite(loaded.mechanics_result.deformed_vertices))
    assert np.max(np.linalg.norm(loaded.mechanics_result.displacement, axis=1)) > 1.0e-5
    assert np.max(np.linalg.norm(loaded.mechanics_result.displacement, axis=1)) > np.max(
        np.linalg.norm(smaller.mechanics_result.displacement, axis=1)
    )
    assert np.min(_six_volumes(loaded.mechanics_result.deformed_vertices, loaded.mechanics_result.tetrahedra)) > 0.0
    np.testing.assert_allclose(
        loaded_with_visual_carrier.mechanics_result.deformed_vertices,
        loaded.mechanics_result.deformed_vertices,
        atol=1.0e-7,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        loaded_with_visual_carrier.mechanics_result.displacement,
        loaded.mechanics_result.displacement,
        atol=1.0e-7,
        rtol=0.0,
    )
    assert loaded_with_visual_carrier.diagnostics["max_soft_contact_count"] == loaded.diagnostics[
        "max_soft_contact_count"
    ]

    state = make_fingertip_volume_state(volume_mesh, prepared, loaded.mechanics_result)
    assert state.morphology_fingerprint == volume_mesh.morphology_fingerprint
    assert state.deformed_coordinates_mm.shape == prepared.tet_mesh.vertices.shape


@pytest.mark.smoke
@pytest.mark.physics
def test_sphere_first_contact_normalization_is_start_distance_invariant() -> None:
    if not wp.is_device_available("cuda:0"):
        pytest.skip("sphere first-contact normalization smoke requires cuda:0")

    model = FingertipModel(FingertipParameters())
    solid = build_fingertip_solid(model)
    volume_mesh = generate_volume_mesh(
        solid,
        volume_mesh_settings_for_tier("search"),
    )
    prepared = prepare_fingertip_mesh(volume_mesh)
    object_mesh = make_sphere_mesh(2.0, subdivisions=1)
    surface = make_outer_compliant_surface(solid)
    contact_settings = FirstContactSettings(
        coarse_step_mm=0.25,
        tolerance_mm=1.0e-5,
        spawn_clearance_mm=0.02,
        max_travel_mm=20.0,
    )
    far_spawn_settings = FirstContactSettings(
        coarse_step_mm=contact_settings.coarse_step_mm,
        tolerance_mm=contact_settings.tolerance_mm,
        spawn_clearance_mm=0.10,
        max_travel_mm=contact_settings.max_travel_mm,
    )
    alignments = tuple(
        canonical_sphere_alignment(
            model,
            radius_mm=2.0,
            initial_gap_mm=gap,
        )
        for gap in (1.0, 10.0)
    )
    first_contacts = tuple(
        find_first_contact(
            surface,
            object_mesh,
            alignment.nominal_pose,
            alignment.approach_direction,
            settings,
        )
        for alignment, settings in zip(
            alignments,
            (contact_settings, far_spawn_settings),
            strict=True,
        )
    )
    for result in first_contacts:
        assert not intersects(surface, object_mesh, result.spawn_pose)

    mechanics_settings = NewtonSettings(
        device="cuda:0",
        gravity=0.0,
        dt=1.0e-3,
        steps=1,
        iterations=5,
        fixed_vertex_indices=prepared.support_vertex_indices,
    )

    def run(alignment, first_contact):
        return solve_fingertip_indentation(
            prepared,
            RigidIndenter3D(
                object_mesh,
                alignment.nominal_pose,
                alignment.approach_direction,
            ),
            mechanics_settings,
            IndentationSettings(
                travel_mm=0.6,
                load_steps=4,
                soft_contact_margin_mm=0.02,
                soft_contact_ke=1.0e3,
                soft_contact_kd=10.0,
            ),
            first_contact=first_contact,
        )

    near_result = run(alignments[0], first_contacts[0])
    far_result = run(alignments[1], first_contacts[1])

    np.testing.assert_allclose(
        first_contacts[0].contact_pose.translation_mm,
        first_contacts[1].contact_pose.translation_mm,
        atol=contact_settings.tolerance_mm,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        near_result.final_indenter_pose.translation_mm,
        far_result.final_indenter_pose.translation_mm,
        atol=contact_settings.tolerance_mm,
        rtol=0.0,
    )
    assert near_result.diagnostics["first_contact_normalized"] is True
    assert far_result.diagnostics["first_contact_normalized"] is True
    # Newton's GPU contact reduction is not bitwise deterministic across two
    # freshly built contexts.  Compare the actual deformation fields at a
    # tight 0.02 mm mesh-scale tolerance, while requiring the same normalized
    # prescribed pose and contact/load signature above.
    max_abs_difference_mm = float(
        np.max(
            np.abs(
                near_result.mechanics_result.deformed_vertices
                - far_result.mechanics_result.deformed_vertices
            )
        )
    )
    rms_difference_mm = float(
        np.sqrt(
            np.mean(
                np.square(
                    near_result.mechanics_result.deformed_vertices
                    - far_result.mechanics_result.deformed_vertices
                )
            )
        )
    )
    max_displacement_difference_mm = abs(
        float(
            np.max(np.linalg.norm(near_result.mechanics_result.displacement, axis=1))
        )
        - float(
            np.max(np.linalg.norm(far_result.mechanics_result.displacement, axis=1))
        )
    )
    comparison_diagnostic = (
        "spawn-clearance invariance diagnostics: "
        f"max_abs_difference_mm={max_abs_difference_mm:.6g}, "
        f"rms_difference_mm={rms_difference_mm:.6g}, "
        f"max_displacement_difference_mm={max_displacement_difference_mm:.6g}"
    )
    print(comparison_diagnostic)
    np.testing.assert_allclose(
        near_result.mechanics_result.deformed_vertices,
        far_result.mechanics_result.deformed_vertices,
        atol=2.0e-2,
        rtol=0.0,
        err_msg=comparison_diagnostic,
    )
    np.testing.assert_allclose(
        near_result.mechanics_result.displacement,
        far_result.mechanics_result.displacement,
        atol=2.0e-2,
        rtol=0.0,
        err_msg=comparison_diagnostic,
    )
    for result in (near_result, far_result):
        assert np.all(np.isfinite(result.mechanics_result.deformed_vertices))
        assert np.min(
            _six_volumes(
                result.mechanics_result.deformed_vertices,
                result.mechanics_result.tetrahedra,
            )
        ) > 0.0
        assert int(result.diagnostics["max_soft_contact_count"]) > 0
    near_displacement_mm = float(
        np.max(np.linalg.norm(near_result.mechanics_result.displacement, axis=1))
    )
    far_displacement_mm = float(
        np.max(np.linalg.norm(far_result.mechanics_result.displacement, axis=1))
    )
    assert near_displacement_mm > 0.0
    assert far_displacement_mm > 0.0
    assert abs(near_displacement_mm - far_displacement_mm) <= 0.05 * max(
        near_displacement_mm,
        far_displacement_mm,
    )


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
@pytest.mark.physics
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
    from physics.trajectory.fingertip import PreparedFingertipMesh
    from physics.contracts.types import TetMeshData

    prepared = PreparedFingertipMesh(
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
        NewtonSettings(
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
        ),
    )
    assert result.diagnostics["full_surface_contact"] is True
    assert np.all(np.isfinite(result.mechanics_result.deformed_vertices))
