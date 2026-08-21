"""Minimal real Newton VBD launch contract."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

import warp as wp

from lumo.physics import NewtonSettings, TetMeshData
from lumo.physics.newton.solve import solve
from validation.physics.multi_location_sphere_contact import (
    run_multi_location_sphere_contact,
)


@pytest.mark.smoke
@pytest.mark.physics
def test_newton_vbd_deforms_tiny_tet_block_on_cuda() -> None:
    if not wp.is_device_available("cuda:0"):
        pytest.skip("physics smoke requires cuda:0")

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
        settings=NewtonSettings(
            # Wiring-smoke parameters only; not a calibrated silicone baseline.
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
        result.rest_vertices,
        vertices,
        atol=1.0e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.deformed_vertices[fixed],
        vertices[fixed],
        atol=1.0e-6,
        rtol=0.0,
    )
    assert np.max(np.linalg.norm(result.displacement[free], axis=1)) > 1.0e-6


@pytest.mark.smoke
@pytest.mark.physics
def test_nominal_carrier_contact_is_bit_exact_on_repeated_cuda_solve() -> None:
    if not wp.is_device_available("cuda:0"):
        pytest.skip("physics smoke requires cuda:0")

    first = run_multi_location_sphere_contact(
        normalized_locations=(0.5,),
        carrier_contact=True,
    ).locations[0].indentation
    second = run_multi_location_sphere_contact(
        normalized_locations=(0.5,),
        carrier_contact=True,
    ).locations[0].indentation

    np.testing.assert_array_equal(
        first.mechanics_result.rest_vertices,
        second.mechanics_result.rest_vertices,
    )
    np.testing.assert_array_equal(
        first.mechanics_result.deformed_vertices,
        second.mechanics_result.deformed_vertices,
    )
    assert first.diagnostics["vbd_deterministic_mode"] == "run_to_run"
    assert second.diagnostics["vbd_deterministic_mode"] == "run_to_run"
    assert first.diagnostics["max_soft_contact_count"] == (
        second.diagnostics["max_soft_contact_count"]
    )
    assert first.diagnostics["carrier_interface_contact_count"] == (
        second.diagnostics["carrier_interface_contact_count"]
    )
