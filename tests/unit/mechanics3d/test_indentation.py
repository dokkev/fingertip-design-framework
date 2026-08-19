"""Solver-neutral rigid indentation contracts."""

from __future__ import annotations

import numpy as np
import pytest

from mechanics3d import (
    IndentationResult,
    IndentationSettings,
    Mechanics3DResult,
    RigidIndenter3D,
    RigidPose3D,
)
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
