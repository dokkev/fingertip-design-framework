"""Real nominal localized FEA-load translation into Newton VBD."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("warp")
pytest.importorskip("newton")

import warp as wp

from physics import NewtonSettings, prepare_fingertip_mesh
from physics.newton.session import NewtonSession
from mesh.volume.mesh import generate_volume_mesh
from mesh.volume.contracts import volume_mesh_settings_for_tier
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters
from model.solid import build_fingertip_solid
from validation.physics.correspondence import (
    VBD_CORRESPONDENCE_DT,
    VBD_CORRESPONDENCE_ITERATIONS,
    _selected_reference,
    build_localized_particle_load,
    verify_exact_mesh_correspondence,
)


@pytest.mark.smoke
@pytest.mark.physics
def test_nominal_localized_load_runs_on_exact_mesh() -> None:
    if not wp.is_device_available("cuda:0"):
        pytest.skip("localized physics smoke requires cuda:0")

    payload, reference, _ = _selected_reference()
    parameters = FingertipParameters(**payload["parameters"])
    volume_mesh = generate_volume_mesh(
        build_fingertip_solid(FingertipModel(parameters)),
        volume_mesh_settings_for_tier(payload["mesh"]["tier"]),
    )
    prepared = prepare_fingertip_mesh(volume_mesh)
    correspondence = verify_exact_mesh_correspondence(volume_mesh, prepared, reference)
    load, construction = build_localized_particle_load(prepared, reference, payload)
    result = NewtonSession(
        prepared.tet_mesh,
        NewtonSettings(
            device="cuda:0",
            gravity=0.0,
            dt=VBD_CORRESPONDENCE_DT,
            steps=load.load_steps,
            iterations=VBD_CORRESPONDENCE_ITERATIONS,
            fixed_vertex_indices=prepared.support_vertex_indices,
        ),
    ).solve(load)

    assert correspondence["source_node_ids_exact"]
    assert construction["resultant_magnitude_error_n"] < 1.0e-5
    assert np.all(np.isfinite(result.deformed_vertices))
    assert result.steps == 12
