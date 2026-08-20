"""Persistent Newton VBD reset and deterministic repeated-solve smoke."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

import warp as wp

from physics import NewtonSettings, TetMeshData
from physics.contracts.load import ParticleLoad
from physics.newton.session import NewtonSession


def _cube_mesh() -> TetMeshData:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    tetrahedra = np.asarray(
        [[0, 1, 3, 4], [1, 2, 3, 6], [1, 3, 4, 6], [1, 4, 5, 6], [3, 4, 6, 7]],
        dtype=np.int32,
    )
    return TetMeshData(vertices, tetrahedra)


@pytest.mark.smoke
@pytest.mark.physics
def test_persistent_session_reset_is_deterministic() -> None:
    if not wp.is_device_available("cuda:0"):
        pytest.skip("persistent physics smoke requires cuda:0")

    mesh = _cube_mesh()
    session = NewtonSession(
        mesh,
        NewtonSettings(
            device="cuda:0",
            gravity=0.0,
            steps=2,
            iterations=3,
            fixed_vertex_indices=(0, 1, 2, 3),
        ),
    )
    load = ParticleLoad(
        vertex_indices=np.asarray([4, 5, 6, 7], dtype=np.int32),
        forces_n=np.tile(np.asarray([0.0, 1.0e-6, 0.0]), (4, 1)),
        load_steps=2,
    )

    first = session.solve(load)
    session.reset()
    second = session.solve(load)
    session.reset()
    zero = session.solve(ParticleLoad.zero(load_steps=2))

    np.testing.assert_allclose(first.deformed_vertices, second.deformed_vertices, atol=1.0e-6, rtol=0.0)
    np.testing.assert_allclose(zero.deformed_vertices, zero.rest_vertices, atol=2.0e-5, rtol=0.0)
