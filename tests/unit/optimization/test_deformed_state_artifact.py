"""Persisted Newton-state to optical-geometry handoff tests."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

pytest.importorskip("gmsh")

from physics import prepare_fingertip_mesh
from mesh import volume_mesh_settings_for_tier
from mesh.rigid.carrier import make_distal_phalanx_mesh
from mesh.volume.mesh import generate_volume_mesh
from finger import Fingertip
from validation.ray_tracing.deformed_state_restore import restore_deformed_optical_state


def _write_artifact(path, prepared, deformed) -> str:
    arrays = {
        "rest_vertices_mm": np.asarray(prepared.tet_mesh.vertices, dtype=np.float32),
        "deformed_vertices_mm": np.asarray(deformed, dtype=np.float32),
        "tetrahedra": np.asarray(prepared.tet_mesh.tetrahedra, dtype=np.int32),
        "source_node_ids": np.asarray(prepared.source_node_ids, dtype=np.int64),
        "carrier_contact_vertex_indices": np.asarray([], dtype=np.int64),
        "carrier_contact_source_node_ids": np.asarray([], dtype=np.int64),
    }
    arrays.update(
        {
            f"surface_{tag}": np.asarray(triangles, dtype=np.int32)
            for tag, triangles in prepared.surface_triangles.items()
        }
    )
    np.savez_compressed(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_persisted_state_restores_exact_deformed_optical_geometry(tmp_path) -> None:
    tip = Fingertip()
    volume_mesh = generate_volume_mesh(
        tip.solid(),
        volume_mesh_settings_for_tier("search"),
    )
    prepared = prepare_fingertip_mesh(volume_mesh)
    deformed = np.asarray(prepared.tet_mesh.vertices, dtype=np.float32).copy()
    deformed[:, 1] += 0.05
    artifact = tmp_path / "state.npz"
    digest = _write_artifact(artifact, prepared, deformed)

    restored = restore_deformed_optical_state(
        tip,
        volume_mesh,
        prepared,
        artifact,
        digest,
        carrier_mesh=make_distal_phalanx_mesh(volume_mesh.solid),
    )

    assert restored.geometry.full3d_surface_provenance == "actual_deformed_3d_volume_state"
    assert not hasattr(restored.geometry, "metadata")
    assert restored.state_id
    assert np.all(np.isfinite(restored.geometry.silicone.vertices))
    assert float(np.max(np.abs(restored.state.displacement_mm))) == pytest.approx(
        0.05, abs=1.0e-6
    )

def test_persisted_state_hash_is_verified(tmp_path) -> None:
    tip = Fingertip()
    volume_mesh = generate_volume_mesh(
        tip.solid(),
        volume_mesh_settings_for_tier("search"),
    )
    prepared = prepare_fingertip_mesh(volume_mesh)
    artifact = tmp_path / "state.npz"
    _write_artifact(artifact, prepared, prepared.tet_mesh.vertices)

    with pytest.raises(ValueError, match="hash mismatch"):
        restore_deformed_optical_state(
            tip,
            volume_mesh,
            prepared,
            artifact,
            "0" * 64,
            carrier_mesh=make_distal_phalanx_mesh(volume_mesh.solid),
        )
