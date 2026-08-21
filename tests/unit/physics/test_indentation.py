"""Solver-neutral rigid indentation contracts."""

from __future__ import annotations

import numpy as np
import pytest

from lumo.physics import (
    IndentationResult,
    IndentationSettings,
    NewtonSettings,
    NewtonResult,
    RigidIndenter3D,
)
from lumo.physics.trajectory.fingertip_adapter import PreparedFingertipMesh
from lumo.physics.contracts.types import TetMeshData
from lumo.physics.trajectory.indentation import _validate_support_constraints
from lumo.mesh.rigid.object import RigidPose3D, make_cube_mesh


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

    result = NewtonResult(
        rest_vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        deformed_vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.1]], dtype=np.float32),
        tetrahedra=np.array([[0, 1, 2, 3]], dtype=np.int32),
        steps=4,
    )
    indentation = IndentationResult(
        result,
        RigidPose3D((0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0)),
        diagnostics={"full_surface_contact": True},
    )
    assert indentation.diagnostics["full_surface_contact"] is True
    with pytest.raises(ValueError):
        IndentationSettings(travel_mm=-0.1)
    with pytest.raises(ValueError):
        IndentationSettings(travel_mm=0.1, rigid_sdf_target_voxel_mm=0.0)


def _prepared_with_support(support: tuple[int, ...]) -> PreparedFingertipMesh:
    prepared = PreparedFingertipMesh(
        TetMeshData(
            np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            np.array([[0, 1, 2, 3]], dtype=np.int32),
        ),
        np.arange(4, dtype=np.int64),
        support,
        {},
        "support-contract",
    )
    return prepared


def test_indentation_accepts_exact_authoritative_fixed_vertices() -> None:
    _validate_support_constraints(
        _prepared_with_support((0, 1)),
        NewtonSettings(fixed_vertex_indices=(1, 0)),
    )


@pytest.mark.parametrize("value", (1.5, True))
def test_newton_settings_reject_non_integer_step_counts(value) -> None:
    with pytest.raises(TypeError, match="steps"):
        NewtonSettings(steps=value)
    with pytest.raises(TypeError, match="iterations"):
        NewtonSettings(iterations=value)


def test_newton_settings_reject_non_integer_fixed_vertices() -> None:
    with pytest.raises(TypeError, match="fixed_vertex_indices"):
        NewtonSettings(fixed_vertex_indices=(1.5,))


@pytest.mark.parametrize("fixed", ((), (0, 2)))
def test_indentation_rejects_any_support_mismatch_before_backend_load(
    fixed: tuple[int, ...],
) -> None:
    prepared = _prepared_with_support((0, 1))
    with pytest.raises(ValueError, match="fixed_vertex_indices"):
        _validate_support_constraints(prepared, NewtonSettings(fixed_vertex_indices=fixed))
