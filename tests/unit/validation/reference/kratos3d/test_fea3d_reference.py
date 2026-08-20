"""Focused tests for the validation-owned native FEA3D loader."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from validation.reference.kratos3d import FEA3DReferenceError, load_fea3d_reference


def _write_fixture(tmp_path, *, node_ids=True, malformed=None, inconsistent=False):
    reference = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    displacement = np.full((4, 3), 0.25, dtype=np.float64)
    deformed = reference + displacement
    arrays = {
        "undeformed_nodes_xyz": reference,
        "deformed_nodes_xyz": deformed,
        "displacement_xyz": displacement,
        "metadata_json": np.asarray(json.dumps({"solver": "synthetic"})),
    }
    if node_ids:
        arrays["node_ids"] = np.asarray([10, 20, 30, 40], dtype=np.int64)
    if malformed == "shape":
        arrays["undeformed_nodes_xyz"] = np.zeros((4, 2))
    if malformed == "finite":
        arrays["deformed_nodes_xyz"][0, 0] = np.nan
    if inconsistent:
        arrays["displacement_xyz"] = np.zeros((4, 3))
    state_path = tmp_path / "state.npz"
    np.savez(state_path, **arrays)
    manifest = {
        "schema": "native-3d-fea-state-v1",
        "native_state_artifact": state_path.name,
        "native_state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "morphology_fingerprint": "synthetic-morphology-v1",
        "force_target_n": 2.0,
        "surface_provenance": "synthetic fixture",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_loader_reads_native_state_and_proves_sorted_node_order(tmp_path) -> None:
    state = load_fea3d_reference(_write_fixture(tmp_path))

    assert state.node_count == 4
    assert state.direct_node_correspondence_provable
    assert state.provenance["node_correspondence"] == "provable"
    np.testing.assert_array_equal(state.source_node_ids, [10, 20, 30, 40])
    np.testing.assert_allclose(state.deformed_coordinates_mm - state.reference_coordinates_mm, state.displacement_mm)
    assert not state.reference_coordinates_mm.flags.writeable
    assert state.morphology_fingerprint == "synthetic-morphology-v1"
    assert state.load_metadata["force_target_n"] == 2.0


@pytest.mark.parametrize("malformed", ["shape", "finite"])
def test_loader_rejects_malformed_coordinate_arrays(tmp_path, malformed) -> None:
    with pytest.raises(FEA3DReferenceError):
        load_fea3d_reference(_write_fixture(tmp_path, malformed=malformed))


def test_loader_rejects_inconsistent_displacement(tmp_path) -> None:
    with pytest.raises(FEA3DReferenceError, match="displacement"):
        load_fea3d_reference(_write_fixture(tmp_path, inconsistent=True))


def test_loader_marks_missing_source_ids_unsupported(tmp_path) -> None:
    state = load_fea3d_reference(_write_fixture(tmp_path, node_ids=False))

    assert state.source_node_ids is None
    assert not state.direct_node_correspondence_provable
    assert state.provenance["node_correspondence"] == "unsupported"
