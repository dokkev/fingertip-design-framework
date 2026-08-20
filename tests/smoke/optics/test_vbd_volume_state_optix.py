from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("warp")
pytest.importorskip("newton")
pytest.importorskip("cupy")
pytest.importorskip("optix")
pytest.importorskip("cuda")

import warp as wp

from physics import (
    NewtonSettings,
    make_fingertip_volume_state,
    prepare_fingertip_mesh,
    solve,
)
from model import Fingertip
from optics.transport3d import (
    Transport3DSettings,
    build_fingertip_volume_state_geometry,
    trace_geometry,
)
from optics.transport3d.optix_backend import create_runtime


@pytest.mark.smoke
@pytest.mark.physics
def test_vbd_volume_state_reaches_full3d_optix_without_fea_artifact() -> None:
    if not wp.is_device_available("cuda:0"):
        pytest.skip("VBD→FULL_3D OptiX smoke requires cuda:0")

    tip = Fingertip()
    volume_mesh = tip.volume_mesh()
    prepared = prepare_fingertip_mesh(volume_mesh)
    mechanics_result = solve(
        prepared.tet_mesh,
        settings=NewtonSettings(
            device="cuda:0",
            gravity=0.0,
            steps=1,
            iterations=5,
            fixed_vertex_indices=prepared.support_vertex_indices,
        ),
    )
    state = make_fingertip_volume_state(volume_mesh, prepared, mechanics_result)
    geometry = build_fingertip_volume_state_geometry(tip, state, reference_mesh=tip.mesh())
    runtime = create_runtime()
    result = trace_geometry(
        tip,
        geometry,
        settings=Transport3DSettings(
            mode="full3d",
            ray_count=16,
            max_interactions=4,
            maximum_segment_count=512,
            maximum_periodic_wraps=4,
            surface_u_bins=8,
            surface_z_bins=4,
            projected_grid_width=16,
            projected_grid_height=16,
            internal_grid_width=16,
            internal_grid_height=16,
            internal_z_bins=4,
            terminate_on_periodic_wrap_limit=True,
            terminate_on_no_event=True,
        ),
        runtime=runtime,
    )

    assert geometry.geometry_mode == "full3d_surface"
    assert result.launched_ray_count == 16
    assert np.isfinite(result.energy_balance_error)
    assert result.energy_balance_error >= 0.0
