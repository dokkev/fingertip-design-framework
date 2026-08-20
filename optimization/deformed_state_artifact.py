"""Restore exact Newton-deformed volume states for the optical handoff."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from physics.trajectory.fingertip import PreparedFingertipMesh
from mesh.fingertip.geometry import generate_fingertip_mesh
from mesh.fingertip.contracts import mesh_settings_for_level
from mesh.volume.contracts import FingertipVolumeMesh
from mesh.volume.state import FingertipVolumeState
from model.fingertip import Fingertip
from optics.transport3d import build_fingertip_volume_state_geometry
from optimization.optical_artifact import fingerprint_mapping
from optics.contracts.objects import CarrierOptics
from optimization.protocol import TrajectoryEvaluationProtocol


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


def _state_identity(
    morphology_fingerprint: str,
    protocol: TrajectoryEvaluationProtocol,
    *,
    location_u: float,
    radius_mm: float,
    checkpoint_depth_mm: float,
    checkpoint_fraction: float,
    normalized_indentation_ratio: float,
    post_contact_travel_mm: float,
    unintended_boundary_clearance_mm: float,
    mechanics_artifact_sha256: str,
) -> dict[str, Any]:
    return {
        "morphology_fingerprint": morphology_fingerprint,
        "protocol_fingerprint": protocol.fingerprint,
        "contact_location_u": float(location_u),
        "indenter_radius_mm": float(radius_mm),
        "checkpoint_depth_mm": float(checkpoint_depth_mm),
        "checkpoint_fraction": float(checkpoint_fraction),
        "normalized_indentation_ratio": float(normalized_indentation_ratio),
        "post_contact_travel_mm": float(post_contact_travel_mm),
        "unintended_boundary_clearance_mm": float(unintended_boundary_clearance_mm),
        "mechanics_artifact_sha256": mechanics_artifact_sha256,
        "mechanics_artifact_fingerprint": mechanics_artifact_sha256,
    }


def build_contact_state_record(
    *,
    morphology_fingerprint: str,
    protocol: TrajectoryEvaluationProtocol,
    location_u: float,
    radius_mm: float,
    checkpoint_depth_mm: float,
    checkpoint_fraction: float,
    normalized_indentation_ratio: float,
    post_contact_travel_mm: float,
    unintended_boundary_clearance_mm: float,
    checkpoint_diagnostics: Mapping[str, Any],
    source_node_ids: tuple[int, ...],
    mechanics_artifact_sha256: str,
) -> dict[str, Any]:
    """Build the persisted mechanics-to-optics contact-state certificate."""
    local_indices = tuple(
        int(index)
        for index in checkpoint_diagnostics.get("active_carrier_contact_vertex_indices", ())
    )
    source_ids = tuple(
        int(source_node_ids[index])
        for index in local_indices
        if 0 <= index < len(source_node_ids)
    )
    identity = _state_identity(
        morphology_fingerprint,
        protocol,
        location_u=location_u,
        radius_mm=radius_mm,
        checkpoint_depth_mm=checkpoint_depth_mm,
        checkpoint_fraction=checkpoint_fraction,
        normalized_indentation_ratio=normalized_indentation_ratio,
        post_contact_travel_mm=post_contact_travel_mm,
        unintended_boundary_clearance_mm=unintended_boundary_clearance_mm,
        mechanics_artifact_sha256=mechanics_artifact_sha256,
    )
    return {
        "state_identity": identity,
        "contact_state_fingerprint": fingerprint_mapping(
            identity | {"carrier_contact_source_node_ids": source_ids}
        ),
        "normalized_location": float(location_u),
        "indenter_radius_mm": float(radius_mm),
        "initial_gap_mm": protocol.initial_gap_mm,
        "checkpoint_depth_mm": float(checkpoint_depth_mm),
        "checkpoint_fraction": float(checkpoint_fraction),
        "normalized_indentation_ratio": float(normalized_indentation_ratio),
        "post_contact_travel_mm": float(post_contact_travel_mm),
        "unintended_boundary_clearance_mm": float(unintended_boundary_clearance_mm),
        "first_contact_travel_mm": float(
            checkpoint_diagnostics.get("first_contact_travel_mm", 0.0)
        ),
        "spawn_clearance_mm": float(checkpoint_diagnostics.get("spawn_clearance_mm", 0.0)),
        "carrier_contact_active": bool(
            checkpoint_diagnostics.get("carrier_contact_active", False)
        ),
        "carrier_contact_occurred": bool(
            checkpoint_diagnostics.get("carrier_contact_occurred", False)
        ),
        "carrier_mechanical_contact_count": int(
            checkpoint_diagnostics.get("carrier_interface_contact_count", 0)
        ),
        "carrier_mechanical_contact_vertex_count": len(source_ids),
        "first_carrier_contact_step": checkpoint_diagnostics.get(
            "first_carrier_contact_step"
        ),
        "carrier_contact_source_node_ids": list(source_ids),
        "carrier_mapping_tolerance_mm": 0.5
        * float(checkpoint_diagnostics.get("rigid_sdf_target_voxel_mm", 0.125)),
        "mechanics_artifact_sha256": mechanics_artifact_sha256,
    }


def write_mechanics_artifact(
    path: Path,
    checkpoint: Any,
    prepared: PreparedFingertipMesh,
) -> str:
    """Persist one checkpoint and return its content hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "rest_vertices_mm": np.asarray(
            checkpoint.mechanics_result.rest_vertices, dtype=np.float32
        ),
        "deformed_vertices_mm": np.asarray(
            checkpoint.mechanics_result.deformed_vertices, dtype=np.float32
        ),
        "tetrahedra": np.asarray(checkpoint.mechanics_result.tetrahedra, dtype=np.int32),
        "source_node_ids": np.asarray(prepared.source_node_ids, dtype=np.int64),
        "carrier_contact_vertex_indices": np.asarray(
            checkpoint.diagnostics.get("active_carrier_contact_vertex_indices", ()),
            dtype=np.int64,
        ),
    }
    arrays["carrier_contact_source_node_ids"] = np.asarray(
        [prepared.source_node_ids[index] for index in arrays["carrier_contact_vertex_indices"]],
        dtype=np.int64,
    )
    arrays.update(
        {
            f"surface_{tag}": np.asarray(triangles, dtype=np.int32)
            for tag, triangles in prepared.surface_triangles.items()
        }
    )
    np.savez_compressed(path, **arrays)
    return _sha256(path)


def restore_deformed_optical_state(
    tip: Fingertip,
    volume_mesh: FingertipVolumeMesh,
    prepared: PreparedFingertipMesh,
    artifact_path: str | Path,
    expected_sha256: str,
    *,
    carrier_contact_source_node_ids: Iterable[int] | None = None,
    carrier_optics: CarrierOptics | None = None,
    carrier_mapping_tolerance_mm: float | None = None,
    metadata: Mapping[str, object] | None = None,
) -> RestoredDeformedOpticalState:
    """Validate one persisted Newton state and build its optical geometry."""

    if not isinstance(tip, Fingertip):
        raise TypeError("tip must be Fingertip")
    if not isinstance(volume_mesh, FingertipVolumeMesh):
        raise TypeError("volume_mesh must be a FingertipVolumeMesh")
    if not isinstance(prepared, PreparedFingertipMesh):
        raise TypeError("prepared must be a PreparedFingertipMesh")
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
        reference_mesh=generate_fingertip_mesh(
            tip.geometry,
            mesh_settings_for_level("medium"),
        ),
        carrier_contact_source_node_ids=frozenset(selected_contact_ids),
        carrier_optics=carrier_optics,
        carrier_mapping_tolerance_mm=carrier_mapping_tolerance_mm,
        full3d_surface_provenance="actual_deformed_3d_volume_state",
        metadata=geometry_metadata,
    )
    return RestoredDeformedOpticalState(
        artifact_path=path,
        artifact_sha256=str(expected_sha256),
        state_id=state_id,
        state=state,
        geometry=geometry,
    )


__all__ = [
    "RestoredDeformedOpticalState",
    "build_contact_state_record",
    "restore_deformed_optical_state",
    "write_mechanics_artifact",
]
