"""Validation-only restoration of persisted Newton states for optical replay."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np

from lumo.mesh.rigid.carrier import RigidCarrierMesh
from lumo.mesh.volume.contracts import FingertipVolumeMesh
from lumo.mesh.volume.state import FingertipVolumeState
from lumo.finger.fingertip import Fingertip
from lumo.ray_tracing.contracts.objects import CarrierOptics
from lumo.ray_tracing.optical_mechanics import build_fingertip_volume_state_geometry
from lumo.ray_tracing.optical_mechanics.geometry import TransportGeometry
from lumo.physics.trajectory.fingertip_adapter import PreparedFingertipMesh


@dataclass(frozen=True)
class RestoredDeformedOpticalState:
    """Validated neutral state and its actual deformed FULL_3D geometry."""

    artifact_path: Path
    artifact_sha256: str
    state_id: str
    state: FingertipVolumeState
    geometry: TransportGeometry


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_array(archive, name: str) -> np.ndarray:
    if name not in archive.files:
        raise ValueError(f"deformed mechanics artifact is missing {name!r}")
    return np.asarray(archive[name])


def _load_state(
    volume_mesh: FingertipVolumeMesh,
    prepared: PreparedFingertipMesh,
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
        rest = _required_array(archive, "rest_vertices_mm").astype(
            np.float32, copy=False
        )
        deformed = _required_array(archive, "deformed_vertices_mm").astype(
            np.float32, copy=False
        )
        tetrahedra = _required_array(archive, "tetrahedra").astype(
            np.int32, copy=False
        )
        source_node_ids = _required_array(archive, "source_node_ids").astype(
            np.int64, copy=False
        )
        if rest.shape != prepared.tet_mesh.vertices.shape:
            raise ValueError(
                "deformed mechanics artifact rest shape does not match prepared mesh"
            )
        if deformed.shape != rest.shape or not np.all(np.isfinite(deformed)):
            raise ValueError("deformed mechanics artifact coordinates are invalid")
        if not np.allclose(
            rest, prepared.tet_mesh.vertices, atol=1.0e-5, rtol=0.0
        ):
            raise ValueError(
                "deformed mechanics artifact rest coordinates do not match mesh"
            )
        if not np.array_equal(tetrahedra, prepared.tet_mesh.tetrahedra):
            raise ValueError(
                "deformed mechanics artifact tetrahedra do not match mesh"
            )
        if not np.array_equal(source_node_ids, prepared.source_node_ids):
            raise ValueError(
                "deformed mechanics artifact source node IDs do not match mesh"
            )
        expected_tags = set(prepared.surface_triangles)
        stored_tags = {
            name.removeprefix("surface_")
            for name in archive.files
            if name.startswith("surface_")
        }
        if stored_tags != expected_tags:
            raise ValueError(
                "deformed mechanics artifact semantic surface tags do not match "
                f"mesh: expected {sorted(expected_tags)!r}, "
                f"got {sorted(stored_tags)!r}"
            )
        for tag, expected in prepared.surface_triangles.items():
            stored = _required_array(archive, f"surface_{tag}").astype(
                np.int32, copy=False
            )
            if not np.array_equal(stored, expected):
                raise ValueError(
                    f"deformed mechanics artifact surface {tag!r} does not match mesh"
                )
        stored_contact_local_indices = _required_array(
            archive,
            "carrier_contact_vertex_indices",
        )
        stored_contact_source_ids = _required_array(
            archive,
            "carrier_contact_source_node_ids",
        )
        for name, values in (
            ("carrier_contact_vertex_indices", stored_contact_local_indices),
            ("carrier_contact_source_node_ids", stored_contact_source_ids),
        ):
            if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"deformed mechanics artifact {name} must be 1D integers")
            if len(np.unique(values)) != len(values):
                raise ValueError(f"deformed mechanics artifact {name} contains duplicates")
        stored_contact_local_indices = np.asarray(
            stored_contact_local_indices,
            dtype=np.int64,
        )
        stored_contact_source_ids = np.asarray(
            stored_contact_source_ids,
            dtype=np.int64,
        )
        if np.any(stored_contact_local_indices < 0) or np.any(
            stored_contact_local_indices >= len(prepared.source_node_ids)
        ):
            raise ValueError(
                "persisted carrier contact provenance references an unknown local vertex"
            )
        expected_contact_source_ids = np.asarray(
            prepared.source_node_ids[stored_contact_local_indices],
            dtype=np.int64,
        )
        if not np.array_equal(
            stored_contact_source_ids,
            expected_contact_source_ids,
        ):
            raise ValueError(
                "persisted carrier contact local and source node IDs disagree"
            )
        stored_contact_ids = tuple(int(value) for value in stored_contact_source_ids)
        source_id_set = set(int(value) for value in prepared.source_node_ids)
        if any(node_id not in source_id_set for node_id in stored_contact_ids):
            raise ValueError(
                "persisted carrier contact provenance references an unknown source node"
            )
    state = FingertipVolumeState.from_deformed_coordinates(volume_mesh, deformed)
    state_id = hashlib.sha256(
        np.asarray(deformed, dtype=np.float32).tobytes()
    ).hexdigest()
    return state, state_id, stored_contact_ids


def restore_deformed_optical_state(
    tip: Fingertip,
    volume_mesh: FingertipVolumeMesh,
    prepared: PreparedFingertipMesh,
    artifact_path: str | Path,
    expected_sha256: str,
    *,
    carrier_mesh: RigidCarrierMesh,
    carrier_contact_source_node_ids: Iterable[int] | None = None,
    carrier_optics: CarrierOptics | None = None,
    carrier_mapping_tolerance_mm: float | None = None,
) -> RestoredDeformedOpticalState:
    """Validate one persisted Newton state and build replay geometry."""

    if not isinstance(tip, Fingertip):
        raise TypeError("tip must be Fingertip")
    if not isinstance(volume_mesh, FingertipVolumeMesh):
        raise TypeError("volume_mesh must be a FingertipVolumeMesh")
    if not isinstance(prepared, PreparedFingertipMesh):
        raise TypeError("prepared must be a PreparedFingertipMesh")
    if not isinstance(carrier_mesh, RigidCarrierMesh):
        raise TypeError("carrier_mesh must be a RigidCarrierMesh")
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
    if carrier_contact_source_node_ids is not None:
        if selected_contact_ids != tuple(sorted(stored_contact_ids)):
            raise ValueError(
                "requested carrier contact provenance does not match the "
                "persisted mechanics artifact"
            )
    geometry = build_fingertip_volume_state_geometry(
        tip,
        state,
        carrier_mesh=carrier_mesh,
        carrier_contact_source_node_ids=frozenset(selected_contact_ids),
        carrier_optics=carrier_optics,
        carrier_mapping_tolerance_mm=carrier_mapping_tolerance_mm,
        full3d_surface_provenance="actual_deformed_3d_volume_state",
    )
    return RestoredDeformedOpticalState(
        artifact_path=path,
        artifact_sha256=str(expected_sha256),
        state_id=state_id,
        state=state,
        geometry=geometry,
    )


__all__ = ["RestoredDeformedOpticalState", "restore_deformed_optical_state"]
