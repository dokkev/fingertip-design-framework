from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gmsh")

from mesh import FingertipVolumeState, volume_mesh_settings_for_tier
from model import Fingertip
from optics.transport3d import build_fingertip_volume_state_geometry


def test_volume_state_direct_adapter_builds_full3d_geometry_without_fea_artifact() -> None:
    tip = Fingertip()
    volume_mesh = tip.volume_mesh(volume_mesh_settings_for_tier("search"))
    state = FingertipVolumeState.reference(volume_mesh)

    geometry = build_fingertip_volume_state_geometry(
        tip,
        state,
        reference_mesh=tip.mesh(),
    )

    assert geometry.geometry_mode == "full3d_surface"
    assert geometry.depth_mm == pytest.approx(11.0)
    assert geometry.z_min_mm == pytest.approx(-5.5)
    assert geometry.z_max_mm == pytest.approx(5.5)
    assert geometry.metadata["full3d_surface_provenance"] == "actual_deformed_3d_volume_state"
    assert set(geometry.silicone.semantic_tags or ()) == set(volume_mesh.surface_triangles) - {
        "longitudinal_end_minus",
        "longitudinal_end_plus",
    }
    assert geometry.silicone.external_surface is not None
    assert np.all(geometry.silicone.external_surface == np.asarray([
        tag.startswith("outer_compliant_")
        for tag in geometry.silicone.semantic_tags or ()
    ]))
    geometric_normals = np.cross(
        geometry.silicone.vertices[geometry.silicone.faces[:, 1]]
        - geometry.silicone.vertices[geometry.silicone.faces[:, 0]],
        geometry.silicone.vertices[geometry.silicone.faces[:, 2]]
        - geometry.silicone.vertices[geometry.silicone.faces[:, 0]],
    )
    alignment = np.sum(
        geometric_normals * geometry.silicone.normals,
        axis=1,
    )
    assert np.all(alignment > 0.0)
