"""Persist exact Newton checkpoints and their contact-state provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from physics.trajectory.fingertip import PreparedFingertipMesh
from optimization.optical_artifact import fingerprint_mapping
from optimization.protocol import TrajectoryEvaluationProtocol


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    required_diagnostics = (
        "active_carrier_contact_vertex_indices",
        "first_contact_travel_mm",
        "spawn_clearance_mm",
        "carrier_contact_active",
        "carrier_contact_occurred",
        "carrier_interface_contact_count",
        "first_carrier_contact_step",
        "rigid_sdf_target_voxel_mm",
    )
    missing = tuple(
        name for name in required_diagnostics if name not in checkpoint_diagnostics
    )
    if missing:
        raise RuntimeError(
            "mechanics checkpoint is missing persisted diagnostics: "
            f"{missing!r}"
        )
    raw_indices = np.asarray(
        checkpoint_diagnostics["active_carrier_contact_vertex_indices"]
    )
    if raw_indices.ndim != 1 or (
        raw_indices.size
        and not np.issubdtype(raw_indices.dtype, np.integer)
    ):
        raise RuntimeError(
            "active carrier-contact vertex indices must be a 1D integer sequence"
        )
    local_indices = tuple(int(index) for index in raw_indices)
    if len(set(local_indices)) != len(local_indices):
        raise RuntimeError(
            "mechanics checkpoint contains duplicate carrier-contact vertex indices"
        )
    invalid_indices = tuple(
        index
        for index in local_indices
        if index < 0 or index >= len(source_node_ids)
    )
    if invalid_indices:
        raise RuntimeError(
            "mechanics checkpoint contains out-of-range carrier-contact "
            f"vertex indices: {invalid_indices!r}"
        )
    source_ids = tuple(
        int(source_node_ids[index])
        for index in local_indices
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
            checkpoint_diagnostics["first_contact_travel_mm"]
        ),
        "spawn_clearance_mm": float(checkpoint_diagnostics["spawn_clearance_mm"]),
        "carrier_contact_active": bool(
            checkpoint_diagnostics["carrier_contact_active"]
        ),
        "carrier_contact_occurred": bool(
            checkpoint_diagnostics["carrier_contact_occurred"]
        ),
        "carrier_mechanical_contact_count": int(
            checkpoint_diagnostics["carrier_interface_contact_count"]
        ),
        "carrier_mechanical_contact_vertex_count": len(source_ids),
        "first_carrier_contact_step": checkpoint_diagnostics[
            "first_carrier_contact_step"
        ],
        "carrier_contact_source_node_ids": list(source_ids),
        "carrier_mapping_tolerance_mm": 0.5
        * float(checkpoint_diagnostics["rigid_sdf_target_voxel_mm"]),
        "mechanics_artifact_sha256": mechanics_artifact_sha256,
    }


def write_mechanics_artifact(
    path: Path,
    checkpoint: Any,
    prepared: PreparedFingertipMesh,
) -> str:
    """Persist one checkpoint and return its content hash."""
    if "active_carrier_contact_vertex_indices" not in checkpoint.diagnostics:
        raise RuntimeError(
            "mechanics checkpoint is missing "
            "active_carrier_contact_vertex_indices"
        )
    raw_contact_indices = np.asarray(
        checkpoint.diagnostics["active_carrier_contact_vertex_indices"]
    )
    if raw_contact_indices.ndim != 1 or (
        raw_contact_indices.size
        and not np.issubdtype(raw_contact_indices.dtype, np.integer)
    ):
        raise RuntimeError(
            "active carrier-contact vertex indices must be a 1D integer sequence"
        )
    contact_indices = np.asarray(raw_contact_indices, dtype=np.int64)
    if len(np.unique(contact_indices)) != len(contact_indices):
        raise RuntimeError(
            "mechanics checkpoint contains duplicate carrier-contact vertex indices"
        )
    if np.any(contact_indices < 0) or np.any(
        contact_indices >= len(prepared.source_node_ids)
    ):
        raise RuntimeError(
            "mechanics checkpoint contains an out-of-range carrier-contact "
            "vertex index"
        )
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
        "carrier_contact_vertex_indices": contact_indices,
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


__all__ = [
    "build_contact_state_record",
    "write_mechanics_artifact",
]
