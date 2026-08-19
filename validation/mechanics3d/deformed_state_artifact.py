"""Restore exact Newton-deformed volume states for optical handoff."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from mechanics3d.fingertip import FingertipMechanicsMesh
from mesh.volume_types import FingertipVolumeMesh
from mesh.volume_state import FingertipVolumeState
from model.fingertip import Fingertip
from optics.transport3d import build_fingertip_volume_state_geometry
from optics.contact_object import CarrierOptics


@dataclass(frozen=True)
class RestoredDeformedOpticalState:
    """Validated neutral state and its actual deformed FULL_3D geometry."""

    artifact_path: Path
    artifact_sha256: str
    state_id: str
    state: FingertipVolumeState
    geometry: object


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_array(archive, name: str) -> np.ndarray:
    if name not in archive.files:
        raise ValueError(f"deformed mechanics artifact is missing {name!r}")
    return np.asarray(archive[name])


def _load_state(
    volume_mesh: FingertipVolumeMesh,
    prepared: FingertipMechanicsMesh,
    artifact_path: Path,
    expected_sha256: str,
) -> tuple[FingertipVolumeState, str, tuple[int, ...]]:
    actual_sha256 = _sha256(artifact_path)
    if actual_sha256 != str(expected_sha256):
        raise ValueError(
            f"deformed mechanics artifact hash mismatch for {artifact_path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    with np.load(artifact_path, allow_pickle=False) as archive:
        rest = _required_array(archive, "rest_vertices_mm").astype(np.float32, copy=False)
        deformed = _required_array(archive, "deformed_vertices_mm").astype(np.float32, copy=False)
        tetrahedra = _required_array(archive, "tetrahedra").astype(np.int32, copy=False)
        source_node_ids = _required_array(archive, "source_node_ids").astype(np.int64, copy=False)
        if rest.shape != prepared.tet_mesh.vertices.shape:
            raise ValueError("deformed mechanics artifact rest shape does not match prepared mesh")
        if deformed.shape != rest.shape or not np.all(np.isfinite(deformed)):
            raise ValueError("deformed mechanics artifact coordinates are invalid")
        if not np.allclose(rest, prepared.tet_mesh.vertices, atol=1.0e-5, rtol=0.0):
            raise ValueError("deformed mechanics artifact rest coordinates do not match mesh")
        if not np.array_equal(tetrahedra, prepared.tet_mesh.tetrahedra):
            raise ValueError("deformed mechanics artifact tetrahedra do not match mesh")
        if not np.array_equal(source_node_ids, prepared.source_node_ids):
            raise ValueError("deformed mechanics artifact source node IDs do not match mesh")
        expected_tags = set(prepared.surface_triangles)
        stored_tags = {
            name.removeprefix("surface_")
            for name in archive.files
            if name.startswith("surface_")
        }
        if stored_tags != expected_tags:
            raise ValueError(
                "deformed mechanics artifact semantic surface tags do not match mesh: "
                f"expected {sorted(expected_tags)!r}, got {sorted(stored_tags)!r}"
            )
        for tag, expected in prepared.surface_triangles.items():
            stored = _required_array(archive, f"surface_{tag}").astype(np.int32, copy=False)
            if not np.array_equal(stored, expected):
                raise ValueError(f"deformed mechanics artifact surface {tag!r} does not match mesh")
        stored_contact_ids = tuple(
            int(value)
            for value in np.asarray(
                archive["carrier_contact_source_node_ids"], dtype=np.int64
            ).reshape(-1)
        ) if "carrier_contact_source_node_ids" in archive.files else ()
        if any(node_id not in set(int(value) for value in prepared.source_node_ids)
               for node_id in stored_contact_ids):
            raise ValueError(
                "persisted carrier contact provenance references an unknown source node"
            )
    state = FingertipVolumeState.from_deformed_coordinates(volume_mesh, deformed)
    state_id = hashlib.sha256(np.asarray(deformed, dtype=np.float32).tobytes()).hexdigest()
    return state, state_id, stored_contact_ids


def restore_deformed_optical_state(
    tip: Fingertip,
    volume_mesh: FingertipVolumeMesh,
    prepared: FingertipMechanicsMesh,
    artifact_path: str | Path,
    expected_sha256: str,
    *,
    carrier_contact_source_node_ids: Iterable[int] | None = None,
    carrier_optics: CarrierOptics | None = None,
    metadata: Mapping[str, object] | None = None,
) -> RestoredDeformedOpticalState:
    """Validate one persisted Newton state and build its optical geometry."""

    if not isinstance(tip, Fingertip):
        raise TypeError("tip must be Fingertip")
    if not isinstance(volume_mesh, FingertipVolumeMesh):
        raise TypeError("volume_mesh must be a FingertipVolumeMesh")
    if not isinstance(prepared, FingertipMechanicsMesh):
        raise TypeError("prepared must be a FingertipMechanicsMesh")
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    state, state_id, stored_contact_ids = _load_state(
        volume_mesh, prepared, path, expected_sha256
    )
    selected_contact_ids = (
        tuple(sorted(int(value) for value in carrier_contact_source_node_ids))
        if carrier_contact_source_node_ids is not None
        else stored_contact_ids
    )
    if carrier_contact_source_node_ids is not None and stored_contact_ids:
        if selected_contact_ids != tuple(sorted(stored_contact_ids)):
            raise ValueError(
                "requested carrier contact provenance does not match the persisted mechanics artifact"
            )
    geometry_metadata = {
        "optical_state_id": state_id,
        "mechanics_artifact_path": str(path),
        "mechanics_artifact_sha256": str(expected_sha256),
        "mechanics_source": "persisted_newton_vbd_deformed_volume_state",
        "morphology_fingerprint": volume_mesh.morphology_fingerprint,
        "carrier_contact_source_node_ids": list(selected_contact_ids),
    }
    if metadata is not None:
        geometry_metadata.update(dict(metadata))
    geometry = build_fingertip_volume_state_geometry(
        tip,
        state,
        reference_mesh=tip.mesh(),
        carrier_contact_source_node_ids=frozenset(selected_contact_ids),
        carrier_optics=carrier_optics,
        metadata=geometry_metadata,
    )
    return RestoredDeformedOpticalState(
        artifact_path=path,
        artifact_sha256=str(expected_sha256),
        state_id=state_id,
        state=state,
        geometry=geometry,
    )


__all__ = ["RestoredDeformedOpticalState", "restore_deformed_optical_state"]
