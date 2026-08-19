from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gmsh")

from fem import SolidFEAResult
from mesh import volume_mesh_settings_for_tier
from model import Fingertip


@pytest.fixture(scope="module")
def volume_mesh():
    return Fingertip().volume_mesh(volume_mesh_settings_for_tier("search"))


def test_solid_fea_result_promotes_through_neutral_state_factory(volume_mesh) -> None:
    node_ids = tuple(sorted(volume_mesh.nodes))
    reference = np.asarray(
        [
            [
                volume_mesh.nodes[node_id].x_mm,
                volume_mesh.nodes[node_id].y_mm,
                volume_mesh.nodes[node_id].z_mm,
            ]
            for node_id in node_ids
        ]
    )
    deformed = reference.copy()
    displacement = np.zeros_like(reference)
    result = SolidFEAResult(
        volume_mesh=volume_mesh,
        reference_coordinates_mm=reference,
        deformed_coordinates_mm=deformed,
        displacement_mm=displacement,
        reaction_force_n=0.0,
        contact_state={},
        configuration={},
        converged=True,
    )

    state = result.volume_state

    assert state.volume_mesh is volume_mesh
    np.testing.assert_array_equal(state.reference_coordinates_mm, reference)
    np.testing.assert_array_equal(state.deformed_coordinates_mm, deformed)
    np.testing.assert_array_equal(state.displacement_mm, displacement)
    assert state.morphology_fingerprint == volume_mesh.morphology_fingerprint


def test_failed_solid_fea_result_has_no_volume_state(volume_mesh) -> None:
    node_ids = tuple(sorted(volume_mesh.nodes))
    reference = np.asarray(
        [
            [
                volume_mesh.nodes[node_id].x_mm,
                volume_mesh.nodes[node_id].y_mm,
                volume_mesh.nodes[node_id].z_mm,
            ]
            for node_id in node_ids
        ]
    )
    result = SolidFEAResult(
        volume_mesh=volume_mesh,
        reference_coordinates_mm=reference,
        deformed_coordinates_mm=None,
        displacement_mm=None,
        reaction_force_n=None,
        contact_state={},
        configuration={},
        converged=False,
        failure_message="synthetic failure",
    )

    with pytest.raises(RuntimeError, match="failed"):
        _ = result.volume_state
