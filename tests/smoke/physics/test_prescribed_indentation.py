"""Real nominal fingertip prescribed-indentation VBD smoke."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("warp")
pytest.importorskip("newton")

import warp as wp

from mechanics3d import (
    Mechanics3DSettings,
    outer_compliant_timing_patch,
    prepare_fingertip_mechanics_mesh,
    solve_prescribed_indentation,
)
from mesh.volume3d import generate_volume_mesh
from mesh.volume_types import volume_mesh_settings_for_tier
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters
from model.solid import build_fingertip_solid


@pytest.mark.smoke
@pytest.mark.mechanics3d
def test_nominal_fingertip_prescribed_patch_runs_on_cuda() -> None:
    if not wp.is_device_available("cuda:0"):
        pytest.skip("prescribed mechanics3d smoke requires cuda:0")

    model = FingertipModel(FingertipParameters())
    solid = build_fingertip_solid(model)
    volume_mesh = generate_volume_mesh(solid, volume_mesh_settings_for_tier("search"))
    prepared = prepare_fingertip_mechanics_mesh(volume_mesh)
    patch = outer_compliant_timing_patch(prepared, load_steps=2)
    result, timing = solve_prescribed_indentation(
        prepared,
        Mechanics3DSettings(
            device="cuda:0",
            gravity=0.0,
            steps=patch.load_steps,
            iterations=2,
            fixed_vertex_indices=prepared.support_vertex_indices,
        ),
        patch,
    )

    selected = np.asarray(patch.vertex_indices, dtype=np.int64)
    np.testing.assert_allclose(
        result.displacement[selected],
        np.broadcast_to(np.asarray(patch.displacement_mm), (selected.size, 3)),
        atol=2.0e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.deformed_vertices[np.asarray(prepared.support_vertex_indices)],
        result.rest_vertices[np.asarray(prepared.support_vertex_indices)],
        atol=2.0e-5,
        rtol=0.0,
    )
    assert np.all(np.isfinite(result.deformed_vertices))
    assert timing["load_steps"] == 2
    assert timing["solver_loop_wall_s"] >= 0.0
