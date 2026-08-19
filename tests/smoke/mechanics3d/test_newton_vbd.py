"""Minimal real Newton VBD launch contract."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

import warp as wp

from mechanics3d import Mechanics3DSettings, TetMeshData, solve


@pytest.mark.smoke
@pytest.mark.mechanics3d
def test_newton_vbd_deforms_tiny_tet_block_on_cuda() -> None:
    if not wp.is_device_available("cuda:0"):
        pytest.skip("mechanics3d smoke requires cuda:0")

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
        [
            [0, 1, 3, 4],
            [1, 2, 3, 6],
            [1, 3, 4, 6],
            [1, 4, 5, 6],
            [3, 4, 6, 7],
        ],
        dtype=np.int32,
    )

    fixed = np.array([0, 1, 2, 3])
    free = np.array([4, 5, 6, 7])

    result = solve(
        TetMeshData(vertices, tetrahedra),
        settings=Mechanics3DSettings(
            device="cuda:0",
            steps=1,
            iterations=5,
            fixed_vertex_indices=(0, 1, 2, 3),
        ),
    )

    assert result.deformed_vertices.shape == vertices.shape
    assert result.tetrahedra.shape == tetrahedra.shape
    assert np.all(np.isfinite(result.deformed_vertices))
    assert np.all(np.isfinite(result.displacement))
    np.testing.assert_allclose(
        result.deformed_vertices[fixed],
        vertices[fixed],
        atol=1.0e-6,
        rtol=0.0,
    )
    assert np.max(np.linalg.norm(result.displacement[free], axis=1)) > 1.0e-6
