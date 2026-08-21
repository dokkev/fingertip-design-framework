"""Persist exact Newton checkpoints and typed contact-state provenance."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from physics.trajectory.fingertip import PreparedFingertipMesh
from physics.trajectory.indentation import IndentationCheckpoint
from optimization.optical_contract import fingerprint_mapping
from optimization.protocol import TrajectoryEvaluationProtocol


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_nonnegative(name: str, value: object) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True, init=False)
class ContactStateIdentity:
    """Stable identity inputs for one mechanics-to-optics handoff."""

    morphology_fingerprint: str
    protocol_fingerprint: str
    contact_location_u: float
    indenter_radius_mm: float
    checkpoint_depth_mm: float
    checkpoint_fraction: float
    normalized_indentation_ratio: float
    post_contact_travel_mm: float
    unintended_boundary_clearance_mm: float
    mechanics_artifact_sha256: str

    def __init__(
        self,
        *,
        morphology_fingerprint: str,
        protocol_fingerprint: str,
        contact_location_u: float,
        indenter_radius_mm: float,
        checkpoint_depth_mm: float,
        checkpoint_fraction: float,
        normalized_indentation_ratio: float,
        post_contact_travel_mm: float,
        unintended_boundary_clearance_mm: float,
        mechanics_artifact_sha256: str,
    ) -> None:
        object.__setattr__(self, "morphology_fingerprint", str(morphology_fingerprint))
        object.__setattr__(self, "protocol_fingerprint", str(protocol_fingerprint))
        location = _finite_nonnegative(
            "contact_location_u", contact_location_u
        )
        if location > 1.0:
            raise ValueError("contact_location_u must lie in [0, 1]")
        object.__setattr__(self, "contact_location_u", location)
        object.__setattr__(self, "indenter_radius_mm", _finite_nonnegative(
            "indenter_radius_mm", indenter_radius_mm
        ))
        object.__setattr__(self, "checkpoint_depth_mm", _finite_nonnegative(
            "checkpoint_depth_mm", checkpoint_depth_mm
        ))
        object.__setattr__(self, "checkpoint_fraction", _finite_nonnegative(
            "checkpoint_fraction", checkpoint_fraction
        ))
        object.__setattr__(self, "normalized_indentation_ratio", _finite_nonnegative(
            "normalized_indentation_ratio", normalized_indentation_ratio
        ))
        object.__setattr__(self, "post_contact_travel_mm", _finite_nonnegative(
            "post_contact_travel_mm", post_contact_travel_mm
        ))
        object.__setattr__(self, "unintended_boundary_clearance_mm", _finite_nonnegative(
            "unintended_boundary_clearance_mm",
            unintended_boundary_clearance_mm,
        ))
        object.__setattr__(self, "mechanics_artifact_sha256", str(mechanics_artifact_sha256))
        if not self.morphology_fingerprint or not self.protocol_fingerprint:
            raise ValueError("contact-state identity fingerprints must be non-empty")
        if not self.mechanics_artifact_sha256:
            raise ValueError("mechanics_artifact_sha256 must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "morphology_fingerprint": self.morphology_fingerprint,
            "protocol_fingerprint": self.protocol_fingerprint,
            "contact_location_u": self.contact_location_u,
            "indenter_radius_mm": self.indenter_radius_mm,
            "checkpoint_depth_mm": self.checkpoint_depth_mm,
            "checkpoint_fraction": self.checkpoint_fraction,
            "normalized_indentation_ratio": self.normalized_indentation_ratio,
            "post_contact_travel_mm": self.post_contact_travel_mm,
            "unintended_boundary_clearance_mm": self.unintended_boundary_clearance_mm,
            "mechanics_artifact_sha256": self.mechanics_artifact_sha256,
        }


@dataclass(frozen=True, init=False)
class ContactState:
    """Immutable certificate for one deformed checkpoint."""

    identity: ContactStateIdentity
    contact_state_fingerprint: str
    normalized_location: float
    indenter_radius_mm: float
    initial_gap_mm: float
    checkpoint_depth_mm: float
    checkpoint_fraction: float
    normalized_indentation_ratio: float
    post_contact_travel_mm: float
    unintended_boundary_clearance_mm: float
    first_contact_travel_mm: float
    spawn_clearance_mm: float
    carrier_contact_active: bool
    carrier_contact_occurred: bool
    carrier_mechanical_contact_count: int
    carrier_mechanical_contact_vertex_count: int
    first_carrier_contact_step: int | None
    carrier_contact_source_node_ids: tuple[int, ...]
    carrier_mapping_tolerance_mm: float
    mechanics_artifact_sha256: str

    def __init__(
        self,
        *,
        identity: ContactStateIdentity,
        contact_state_fingerprint: str,
        normalized_location: float,
        indenter_radius_mm: float,
        initial_gap_mm: float,
        checkpoint_depth_mm: float,
        checkpoint_fraction: float,
        normalized_indentation_ratio: float,
        post_contact_travel_mm: float,
        unintended_boundary_clearance_mm: float,
        first_contact_travel_mm: float,
        spawn_clearance_mm: float,
        carrier_contact_active: bool,
        carrier_contact_occurred: bool,
        carrier_mechanical_contact_count: int,
        carrier_mechanical_contact_vertex_count: int,
        first_carrier_contact_step: int | None,
        carrier_contact_source_node_ids: tuple[int, ...],
        carrier_mapping_tolerance_mm: float,
        mechanics_artifact_sha256: str,
    ) -> None:
        if not isinstance(identity, ContactStateIdentity):
            raise TypeError("identity must be ContactStateIdentity")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "contact_state_fingerprint", str(contact_state_fingerprint))
        if not self.contact_state_fingerprint:
            raise ValueError("contact_state_fingerprint must be non-empty")
        object.__setattr__(self, "normalized_location", _finite_nonnegative("normalized_location", normalized_location))
        object.__setattr__(self, "indenter_radius_mm", _finite_nonnegative("indenter_radius_mm", indenter_radius_mm))
        object.__setattr__(self, "initial_gap_mm", _finite_nonnegative("initial_gap_mm", initial_gap_mm))
        object.__setattr__(self, "checkpoint_depth_mm", _finite_nonnegative("checkpoint_depth_mm", checkpoint_depth_mm))
        object.__setattr__(self, "checkpoint_fraction", _finite_nonnegative("checkpoint_fraction", checkpoint_fraction))
        object.__setattr__(self, "normalized_indentation_ratio", _finite_nonnegative(
            "normalized_indentation_ratio", normalized_indentation_ratio
        ))
        object.__setattr__(self, "post_contact_travel_mm", _finite_nonnegative(
            "post_contact_travel_mm", post_contact_travel_mm
        ))
        object.__setattr__(self, "unintended_boundary_clearance_mm", _finite_nonnegative(
            "unintended_boundary_clearance_mm", unintended_boundary_clearance_mm
        ))
        object.__setattr__(self, "first_contact_travel_mm", _finite_nonnegative(
            "first_contact_travel_mm", first_contact_travel_mm
        ))
        object.__setattr__(self, "spawn_clearance_mm", _finite_nonnegative("spawn_clearance_mm", spawn_clearance_mm))
        if not isinstance(carrier_contact_active, bool) or not isinstance(
            carrier_contact_occurred, bool
        ):
            raise TypeError("carrier contact flags must be bool")
        object.__setattr__(self, "carrier_contact_active", carrier_contact_active)
        object.__setattr__(self, "carrier_contact_occurred", carrier_contact_occurred)
        for name, value in (
            ("carrier_mechanical_contact_count", carrier_mechanical_contact_count),
            (
                "carrier_mechanical_contact_vertex_count",
                carrier_mechanical_contact_vertex_count,
            ),
        ):
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))
        if self.carrier_mechanical_contact_count < 0 or self.carrier_mechanical_contact_vertex_count < 0:
            raise ValueError("carrier contact counts must be non-negative")
        if first_carrier_contact_step is not None:
            if int(first_carrier_contact_step) != first_carrier_contact_step or int(first_carrier_contact_step) < 1:
                raise ValueError("first_carrier_contact_step must be None or positive")
            first_carrier_contact_step = int(first_carrier_contact_step)
        object.__setattr__(self, "first_carrier_contact_step", first_carrier_contact_step)
        source_ids = tuple(int(value) for value in carrier_contact_source_node_ids)
        if len(set(source_ids)) != len(source_ids) or any(value < 0 for value in source_ids):
            raise ValueError("carrier_contact_source_node_ids must be unique and non-negative")
        object.__setattr__(self, "carrier_contact_source_node_ids", source_ids)
        object.__setattr__(self, "carrier_mapping_tolerance_mm", _finite_nonnegative(
            "carrier_mapping_tolerance_mm", carrier_mapping_tolerance_mm
        ))
        object.__setattr__(self, "mechanics_artifact_sha256", str(mechanics_artifact_sha256))
        if not self.mechanics_artifact_sha256:
            raise ValueError("mechanics_artifact_sha256 must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_identity": self.identity.to_dict(),
            "contact_state_fingerprint": self.contact_state_fingerprint,
            "normalized_location": self.normalized_location,
            "indenter_radius_mm": self.indenter_radius_mm,
            "initial_gap_mm": self.initial_gap_mm,
            "checkpoint_depth_mm": self.checkpoint_depth_mm,
            "checkpoint_fraction": self.checkpoint_fraction,
            "normalized_indentation_ratio": self.normalized_indentation_ratio,
            "post_contact_travel_mm": self.post_contact_travel_mm,
            "unintended_boundary_clearance_mm": self.unintended_boundary_clearance_mm,
            "first_contact_travel_mm": self.first_contact_travel_mm,
            "spawn_clearance_mm": self.spawn_clearance_mm,
            "carrier_contact_active": self.carrier_contact_active,
            "carrier_contact_occurred": self.carrier_contact_occurred,
            "carrier_mechanical_contact_count": self.carrier_mechanical_contact_count,
            "carrier_mechanical_contact_vertex_count": self.carrier_mechanical_contact_vertex_count,
            "first_carrier_contact_step": self.first_carrier_contact_step,
            "carrier_contact_source_node_ids": list(self.carrier_contact_source_node_ids),
            "carrier_mapping_tolerance_mm": self.carrier_mapping_tolerance_mm,
            "mechanics_artifact_sha256": self.mechanics_artifact_sha256,
        }


def build_contact_state_record(
    *,
    morphology_fingerprint: str,
    protocol: TrajectoryEvaluationProtocol,
    location_u: float,
    radius_mm: float,
    checkpoint: IndentationCheckpoint,
    post_contact_travel_mm: float,
    unintended_boundary_clearance_mm: float,
    source_node_ids: tuple[int, ...],
    mechanics_artifact_sha256: str,
) -> ContactState:
    """Build the typed mechanics-to-optics contact-state certificate."""

    state = checkpoint.state
    if state.first_contact_travel_mm is None or state.spawn_clearance_mm is None:
        raise RuntimeError("mechanics checkpoint lacks complete contact provenance")
    invalid_indices = tuple(
        index
        for index in state.active_carrier_contact_vertex_indices
        if index < 0 or index >= len(source_node_ids)
    )
    if invalid_indices:
        raise RuntimeError(
            "mechanics checkpoint contains out-of-range carrier-contact vertex "
            f"indices: {invalid_indices!r}"
        )
    source_ids = tuple(
        int(source_node_ids[index])
        for index in state.active_carrier_contact_vertex_indices
    )
    identity = ContactStateIdentity(
        morphology_fingerprint=morphology_fingerprint,
        protocol_fingerprint=protocol.fingerprint,
        contact_location_u=location_u,
        indenter_radius_mm=radius_mm,
        checkpoint_depth_mm=post_contact_travel_mm,
        checkpoint_fraction=checkpoint.checkpoint_fraction,
        normalized_indentation_ratio=checkpoint.normalized_indentation_ratio,
        post_contact_travel_mm=post_contact_travel_mm,
        unintended_boundary_clearance_mm=unintended_boundary_clearance_mm,
        mechanics_artifact_sha256=mechanics_artifact_sha256,
    )
    contact_fingerprint = fingerprint_mapping(
        identity.to_dict() | {"carrier_contact_source_node_ids": source_ids}
    )
    return ContactState(
        identity=identity,
        contact_state_fingerprint=contact_fingerprint,
        normalized_location=location_u,
        indenter_radius_mm=radius_mm,
        initial_gap_mm=protocol.initial_gap_mm,
        checkpoint_depth_mm=post_contact_travel_mm,
        checkpoint_fraction=checkpoint.checkpoint_fraction,
        normalized_indentation_ratio=checkpoint.normalized_indentation_ratio,
        post_contact_travel_mm=post_contact_travel_mm,
        unintended_boundary_clearance_mm=unintended_boundary_clearance_mm,
        first_contact_travel_mm=state.first_contact_travel_mm,
        spawn_clearance_mm=state.spawn_clearance_mm,
        carrier_contact_active=state.carrier_contact_active,
        carrier_contact_occurred=state.carrier_contact_occurred,
        carrier_mechanical_contact_count=state.carrier_interface_contact_count,
        carrier_mechanical_contact_vertex_count=len(source_ids),
        first_carrier_contact_step=state.first_carrier_contact_step,
        carrier_contact_source_node_ids=source_ids,
        carrier_mapping_tolerance_mm=0.5 * state.rigid_sdf_target_voxel_mm,
        mechanics_artifact_sha256=mechanics_artifact_sha256,
    )


def write_mechanics_artifact(
    path: Path,
    checkpoint: IndentationCheckpoint,
    prepared: PreparedFingertipMesh,
) -> str:
    """Persist one checkpoint and its scalar state certificate."""

    state = checkpoint.state
    contact_indices = np.asarray(
        state.active_carrier_contact_vertex_indices,
        dtype=np.int64,
    )
    if np.any(contact_indices >= len(prepared.source_node_ids)):
        raise RuntimeError("mechanics checkpoint contact index is out of range")
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
        "carrier_contact_source_node_ids": np.asarray(
            [prepared.source_node_ids[index] for index in contact_indices],
            dtype=np.int64,
        ),
        "checkpoint_index": np.asarray(checkpoint.checkpoint_index, dtype=np.int64),
        "post_contact_travel_mm": np.asarray(checkpoint.post_contact_travel_mm, dtype=np.float64),
        "final_pose_error_mm": np.asarray(state.final_pose_error_mm, dtype=np.float64),
        "rigid_sdf_target_voxel_mm": np.asarray(state.rigid_sdf_target_voxel_mm, dtype=np.float64),
    }
    arrays.update(
        {
            f"surface_{tag}": np.asarray(triangles, dtype=np.int32)
            for tag, triangles in prepared.surface_triangles.items()
        }
    )
    np.savez_compressed(path, **arrays)
    return _sha256(path)


__all__ = [
    "ContactStateIdentity",
    "ContactState",
    "build_contact_state_record",
    "write_mechanics_artifact",
]
