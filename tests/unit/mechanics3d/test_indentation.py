"""Solver-neutral rigid indentation contracts."""

from __future__ import annotations

import numpy as np
import pytest

from mechanics3d import (
    IndentationResult,
    IndentationSettings,
    Mechanics3DSettings,
    Mechanics3DResult,
    RigidIndenter3D,
    RigidPose3D,
    solve_fingertip_indentation,
)
from mechanics3d.fingertip import FingertipMechanicsMesh
from mechanics3d.types import TetMeshData
from mesh.rigid_object import make_cube_mesh


def test_pose_and_direction_are_normalized_without_solver_imports() -> None:
    pose = RigidPose3D((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 2.0))
    indenter = RigidIndenter3D(make_cube_mesh(2.0), pose, (0.0, -2.0, 0.0))

    assert pose.quaternion_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert indenter.approach_direction == (0.0, -1.0, 0.0)
    assert indenter.pose_at_travel(1.5).translation_mm == (1.0, 0.5, 3.0)


@pytest.mark.parametrize(
    "value",
    ((0.0, 0.0, 0.0, 0.0), (np.nan, 0.0, 0.0, 1.0)),
)
def test_pose_rejects_invalid_quaternion(value) -> None:
    with pytest.raises(ValueError):
        RigidPose3D((0.0, 0.0, 0.0), value)


def test_indentation_settings_and_result_are_neutral() -> None:
    settings = IndentationSettings(travel_mm=0.5, load_steps=4)
    assert settings.travel_mm == 0.5
    assert settings.load_steps == 4

    result = Mechanics3DResult(
        rest_vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        deformed_vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.1]], dtype=np.float32),
        tetrahedra=np.array([[0, 1, 2, 3]], dtype=np.int32),
        steps=4,
    )
    indentation = IndentationResult(result, RigidPose3D((0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0)), {"full_surface_contact": True})
    assert indentation.diagnostics["full_surface_contact"] is True
    with pytest.raises(ValueError):
        IndentationSettings(travel_mm=-0.1)
    with pytest.raises(ValueError):
        IndentationSettings(travel_mm=0.1, rigid_sdf_target_voxel_mm=0.0)


def test_indentation_rejects_non_authoritative_fixed_vertices_before_backend_load() -> None:
    prepared = FingertipMechanicsMesh(
        TetMeshData(
            np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            np.array([[0, 1, 2, 3]], dtype=np.int32),
        ),
        np.arange(4, dtype=np.int64),
        (0, 1),
        {},
        "support-contract",
    )
    with pytest.raises(ValueError, match="fixed_vertex_indices"):
        solve_fingertip_indentation(
            prepared,
            RigidIndenter3D(
                make_cube_mesh(1.0),
                RigidPose3D((0.0, 2.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                (0.0, -1.0, 0.0),
            ),
            Mechanics3DSettings(fixed_vertex_indices=(0, 2)),
            IndentationSettings(travel_mm=0.1),
        )
