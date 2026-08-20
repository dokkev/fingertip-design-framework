"""Fail-closed loader for persisted native Kratos 3D states."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from validation.common.io import strict_read_json


class FEA3DReferenceError(ValueError):
    """Raised when a persisted 3D FEA reference state is not self-consistent."""


def _readonly_array(value: np.ndarray, *, dtype: np.dtype | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _resolve_state_path(manifest_path: Path, raw_path: str) -> Path:
    reference = Path(raw_path)
    candidates = [
        reference,
        Path.cwd() / reference,
        manifest_path.parent / reference,
        manifest_path.parent / reference.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FEA3DReferenceError(
        f"native state artifact does not exist for {manifest_path}: {raw_path}"
    )


def _metadata_json(archive: Any) -> dict[str, Any]:
    if "metadata_json" not in archive.files:
        return {}
    try:
        value = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exception:
        raise FEA3DReferenceError("native state metadata_json is not valid JSON") from exception
    if not isinstance(value, dict):
        raise FEA3DReferenceError("native state metadata_json must contain an object")
    return value


def _validate_field(name: str, value: Any, shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise FEA3DReferenceError(f"{name} must have shape (N, 3), got {array.shape}")
    if shape is not None and array.shape != shape:
        raise FEA3DReferenceError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise FEA3DReferenceError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class FEA3DReferenceState:
    """Neutral persisted FEA3D state; this type never owns solver objects."""

    source_path: Path
    reference_coordinates_mm: np.ndarray
    deformed_coordinates_mm: np.ndarray
    displacement_mm: np.ndarray
    source_node_ids: np.ndarray | None
    morphology_fingerprint: str | None
    load_metadata: Mapping[str, Any]
    mechanics_metadata: Mapping[str, Any]
    provenance: Mapping[str, Any]
    tetrahedra_node_ids: np.ndarray | None = None
    surface_faces_node_ids: np.ndarray | None = None
    surface_semantic_tags: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        reference = np.asarray(self.reference_coordinates_mm, dtype=float)
        deformed = np.asarray(self.deformed_coordinates_mm, dtype=float)
        displacement = np.asarray(self.displacement_mm, dtype=float)
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError("reference_coordinates_mm must have shape (N, 3)")
        if deformed.shape != reference.shape or displacement.shape != reference.shape:
            raise ValueError("FEA3D coordinate fields must share shape (N, 3)")
        if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(deformed)):
            raise ValueError("FEA3D coordinates must be finite")
        if not np.all(np.isfinite(displacement)):
            raise ValueError("FEA3D displacement must be finite")
        if not np.allclose(deformed - reference, displacement, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("deformed coordinates and displacement are inconsistent")

        node_ids = None
        if self.source_node_ids is not None:
            raw_node_ids = np.asarray(self.source_node_ids)
            if raw_node_ids.shape != (reference.shape[0],):
                raise ValueError("source_node_ids must have one ID per coordinate row")
            if not np.issubdtype(raw_node_ids.dtype, np.integer):
                raise ValueError("source_node_ids must contain integers")
            node_ids = np.asarray(raw_node_ids, dtype=np.int64)
            if len(set(node_ids.tolist())) != len(node_ids):
                raise ValueError("source_node_ids must be unique")

        fingerprint = self.morphology_fingerprint
        if fingerprint is not None and (not isinstance(fingerprint, str) or not fingerprint):
            raise ValueError("morphology_fingerprint must be a non-empty string or None")
        tetrahedra = None
        if self.tetrahedra_node_ids is not None:
            raw_tetrahedra = np.asarray(self.tetrahedra_node_ids)
            if raw_tetrahedra.ndim != 2 or raw_tetrahedra.shape[1] != 4:
                raise ValueError("tetrahedra_node_ids must have shape (T, 4)")
            if not np.issubdtype(raw_tetrahedra.dtype, np.integer):
                raise ValueError("tetrahedra_node_ids must contain integers")
            tetrahedra = np.asarray(raw_tetrahedra, dtype=np.int64)
        surface_faces = None
        if self.surface_faces_node_ids is not None:
            raw_faces = np.asarray(self.surface_faces_node_ids)
            if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
                raise ValueError("surface_faces_node_ids must have shape (F, 3)")
            if not np.issubdtype(raw_faces.dtype, np.integer):
                raise ValueError("surface_faces_node_ids must contain integers")
            surface_faces = np.asarray(raw_faces, dtype=np.int64)
        surface_tags = None if self.surface_semantic_tags is None else tuple(self.surface_semantic_tags)
        if surface_tags is not None and surface_faces is not None and len(surface_tags) != len(surface_faces):
            raise ValueError("surface_semantic_tags must match surface_faces_node_ids")
        object.__setattr__(self, "source_path", Path(self.source_path).resolve())
        object.__setattr__(self, "reference_coordinates_mm", _readonly_array(reference))
        object.__setattr__(self, "deformed_coordinates_mm", _readonly_array(deformed))
        object.__setattr__(self, "displacement_mm", _readonly_array(displacement))
        object.__setattr__(self, "source_node_ids", None if node_ids is None else _readonly_array(node_ids, dtype=np.int64))
        object.__setattr__(self, "load_metadata", _mapping(self.load_metadata))
        object.__setattr__(self, "mechanics_metadata", _mapping(self.mechanics_metadata))
        object.__setattr__(self, "provenance", _mapping(self.provenance))
        object.__setattr__(
            self,
            "tetrahedra_node_ids",
            None if tetrahedra is None else _readonly_array(tetrahedra, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "surface_faces_node_ids",
            None if surface_faces is None else _readonly_array(surface_faces, dtype=np.int64),
        )
        object.__setattr__(self, "surface_semantic_tags", surface_tags)

    @property
    def node_count(self) -> int:
        return int(self.reference_coordinates_mm.shape[0])

    @property
    def direct_node_correspondence_provable(self) -> bool:
        """Whether persisted source IDs prove the canonical node-row order."""
        return self.provenance.get("node_correspondence") == "provable"


def load_fea3d_reference(
    manifest_path: str | Path,
    *,
    case_metadata: Mapping[str, Any] | None = None,
) -> FEA3DReferenceState:
    """Load one ``native-3d-fea-state-v1`` state without running FEA."""

    manifest = Path(manifest_path).resolve()
    if not manifest.is_file():
        raise FEA3DReferenceError(f"native state manifest does not exist: {manifest}")
    try:
        payload = strict_read_json(manifest)
    except (OSError, ValueError) as exception:
        raise FEA3DReferenceError(f"failed to read native state manifest {manifest}: {exception}") from exception
    if payload.get("schema") != "native-3d-fea-state-v1":
        raise FEA3DReferenceError(
            f"unsupported FEA3D artifact schema: {payload.get('schema')!r}"
        )
    raw_state_path = payload.get("native_state_artifact", payload.get("surface_artifact"))
    if not isinstance(raw_state_path, str) or not raw_state_path:
        raise FEA3DReferenceError("native state manifest has no state artifact path")
    state_path = _resolve_state_path(manifest, raw_state_path)

    expected_sha = payload.get("native_state_sha256", payload.get("surface_sha256"))
    sha_verified = None
    if expected_sha is not None:
        actual_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
        sha_verified = actual_sha == expected_sha
        if not sha_verified:
            raise FEA3DReferenceError(f"native state checksum mismatch: {state_path}")

    tetrahedra = None
    surface_faces = None
    tags = None
    try:
        with np.load(state_path, allow_pickle=False) as archive:
            required = {
                "undeformed_nodes_xyz",
                "deformed_nodes_xyz",
                "displacement_xyz",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise FEA3DReferenceError(f"native state is missing arrays: {missing}")
            reference = _validate_field("undeformed_nodes_xyz", archive["undeformed_nodes_xyz"])
            expected_shape = reference.shape
            deformed = _validate_field("deformed_nodes_xyz", archive["deformed_nodes_xyz"], expected_shape)
            displacement = _validate_field("displacement_xyz", archive["displacement_xyz"], expected_shape)
            if not np.allclose(deformed - reference, displacement, rtol=1.0e-10, atol=1.0e-12):
                raise FEA3DReferenceError("native state displacement does not match coordinate difference")

            source_node_ids = None
            node_order = "unsupported: source node IDs are absent"
            if "node_ids" in archive.files:
                raw_ids = np.asarray(archive["node_ids"])
                if raw_ids.shape != (expected_shape[0],) or not np.issubdtype(raw_ids.dtype, np.integer):
                    raise FEA3DReferenceError("native state node_ids do not match coordinate rows")
                source_node_ids = np.asarray(raw_ids, dtype=np.int64)
                if len(set(source_node_ids.tolist())) != len(source_node_ids):
                    raise FEA3DReferenceError("native state node_ids are not unique")
                if np.array_equal(source_node_ids, np.sort(source_node_ids)):
                    node_order = "explicit node_ids sorted; generator uses tuple(sorted(volume_mesh.nodes))"
                else:
                    node_order = "unsupported: explicit node_ids are not in canonical sorted order"

            tetrahedra_count = None
            if "tetrahedra_node_ids" in archive.files and source_node_ids is not None:
                tetrahedra = np.asarray(archive["tetrahedra_node_ids"])
                if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
                    raise FEA3DReferenceError("tetrahedra_node_ids must have shape (T, 4)")
                if not np.issubdtype(tetrahedra.dtype, np.integer):
                    raise FEA3DReferenceError("tetrahedra_node_ids must contain integers")
                if not np.isin(tetrahedra, source_node_ids).all():
                    raise FEA3DReferenceError("tetrahedra_node_ids reference unknown node IDs")
                tetrahedra_count = int(tetrahedra.shape[0])

            surface_tag_count = None
            if "surface_semantic_tags_json" in archive.files:
                try:
                    tags = json.loads(str(np.asarray(archive["surface_semantic_tags_json"]).item()))
                except (TypeError, ValueError, json.JSONDecodeError) as exception:
                    raise FEA3DReferenceError("surface_semantic_tags_json is not valid JSON") from exception
                if not isinstance(tags, list):
                    raise FEA3DReferenceError("surface_semantic_tags_json must contain a list")
                surface_tag_count = len(tags)
            if "surface_faces_node_ids" in archive.files:
                raw_faces = np.asarray(archive["surface_faces_node_ids"])
                if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
                    raise FEA3DReferenceError("surface_faces_node_ids must have shape (F, 3)")
                if not np.issubdtype(raw_faces.dtype, np.integer):
                    raise FEA3DReferenceError("surface_faces_node_ids must contain integers")
                if source_node_ids is not None and not np.isin(raw_faces, source_node_ids).all():
                    raise FEA3DReferenceError("surface_faces_node_ids reference unknown node IDs")
                surface_faces = np.asarray(raw_faces, dtype=np.int64)

            metadata = _metadata_json(archive)
    except (OSError, ValueError, KeyError) as exception:
        if isinstance(exception, FEA3DReferenceError):
            raise
        raise FEA3DReferenceError(f"failed to read native state {state_path}: {exception}") from exception

    case = dict(case_metadata or {})
    load_metadata = {
        key: value
        for key, value in {
            **case,
            **payload,
        }.items()
        if key
        in {
            "contact_location",
            "contact_location_mm",
            "force_target_n",
            "localized_load_only",
            "total_prescribed_travel_mm",
            "indenter_radius_mm",
            "initial_gap_mm",
            "load",
            "force_control",
            "reaction_force_n",
            "status",
            "outcome",
        }
    }
    morphology_fingerprint = payload.get("morphology_fingerprint") or case.get("morphology_fingerprint")
    provenance = {
        "artifact_schema": payload["schema"],
        "manifest_path": str(manifest),
        "state_path": str(state_path),
        "state_sha256_verified": sha_verified,
        "node_order_evidence": node_order,
        "node_correspondence": "provable" if node_order.startswith("explicit node_ids sorted") else "unsupported",
        "generator_contract": "validation exporters persist node_order=tuple(sorted(volume_mesh.nodes))",
        "surface_provenance": payload.get("surface_provenance"),
        "surface_semantic_tag_count": surface_tag_count,
        "tetrahedra_node_ids_available": tetrahedra_count is not None,
        "tetrahedra_count": tetrahedra_count,
        "native_artifact_repair": payload.get("native_artifact_repair"),
    }
    return FEA3DReferenceState(
        source_path=manifest,
        reference_coordinates_mm=reference,
        deformed_coordinates_mm=deformed,
        displacement_mm=displacement,
        source_node_ids=source_node_ids,
        morphology_fingerprint=(str(morphology_fingerprint) if morphology_fingerprint is not None else None),
        load_metadata=load_metadata,
        mechanics_metadata=metadata or {"configuration": payload.get("configuration")},
        provenance=provenance,
        tetrahedra_node_ids=tetrahedra,
        surface_faces_node_ids=surface_faces,
        surface_semantic_tags=(tuple(tags) if isinstance(tags, list) else None),
    )


__all__ = ["FEA3DReferenceError", "FEA3DReferenceState", "load_fea3d_reference"]
