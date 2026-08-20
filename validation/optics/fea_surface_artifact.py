"""Fail-closed loading of actual deformed 3D FEA optical surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from model.fingertip import Fingertip
from optics.transport3d.geometry import (
    TriangleSurface,
    Transport3DGeometryError,
    build_full3d_transport_geometry,
)


FULL3D_SURFACE_SCHEMA = "full3d-fea-surface-v1"
NATIVE_3D_FEA_STATE_SCHEMA = "native-3d-fea-state-v1"
_ACCEPTED_SCHEMAS = {FULL3D_SURFACE_SCHEMA, NATIVE_3D_FEA_STATE_SCHEMA}


def _json_scalar(archive: Any, key: str) -> Any:
    if key not in archive:
        raise ValueError(f"3D surface artifact is missing {key}")
    try:
        return json.loads(str(np.asarray(archive[key]).item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"3D surface artifact has invalid JSON field {key}") from exc


def _surface(
    archive: Any,
    prefix: str,
    *,
    semantic: bool,
    repair_normals: bool = False,
) -> TriangleSurface:
    required = ("vertices", "faces", "normals")
    arrays = {}
    for name in required:
        key = f"{prefix}_{name}"
        if key not in archive:
            raise ValueError(f"3D surface artifact is missing {key}")
        arrays[name] = np.asarray(archive[key])
    tags = None
    tags_key = f"{prefix}_semantic_tags_json"
    if semantic:
        raw_tags = _json_scalar(archive, tags_key)
        if not isinstance(raw_tags, list):
            raise ValueError(f"{tags_key} must contain a JSON list")
        tags = tuple(str(tag) for tag in raw_tags)
    elif tags_key in archive:
        raw_tags = _json_scalar(archive, tags_key)
        tags = tuple(str(tag) for tag in raw_tags)
    external = None
    external_key = f"{prefix}_external_surface"
    if external_key in archive:
        external = np.asarray(archive[external_key], dtype=bool)
    u_start = None
    u_end = None
    for name in ("u_start", "u_end"):
        key = f"{prefix}_{name}"
        if key in archive:
            if name == "u_start":
                u_start = np.asarray(archive[key], dtype=float)
            else:
                u_end = np.asarray(archive[key], dtype=float)
    if semantic and (external is None or u_start is None or u_end is None):
        raise ValueError(
            f"{prefix} full 3D surface must preserve external and material-coordinate metadata"
        )
    if repair_normals:
        faces = np.asarray(arrays["faces"], dtype=np.int64)
        geometric_cross = np.cross(
            arrays["vertices"][faces[:, 1]] - arrays["vertices"][faces[:, 0]],
            arrays["vertices"][faces[:, 2]] - arrays["vertices"][faces[:, 0]],
        )
        arrays["normals"] = geometric_cross
    return TriangleSurface(
        vertices=arrays["vertices"],
        faces=arrays["faces"],
        normals=arrays["normals"],
        external_surface=external,
        u_start=u_start,
        u_end=u_end,
        semantic_tags=tags,
    )


def _integer_array(archive: Any, key: str, *, ndim: int) -> np.ndarray:
    if key not in archive:
        raise ValueError(f"native 3D FEA artifact is missing {key}")
    values = np.asarray(archive[key])
    if values.ndim != ndim or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"native 3D FEA artifact field {key} has an invalid shape or type")
    result = np.asarray(values, dtype=np.int64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"native 3D FEA artifact field {key} is not finite")
    return result


def _finite_array(archive: Any, key: str, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if key not in archive:
        raise ValueError(f"native 3D FEA artifact is missing {key}")
    result = np.asarray(archive[key], dtype=float)
    if shape is not None and result.shape != shape:
        raise ValueError(f"native 3D FEA artifact field {key} has shape {result.shape}, expected {shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"native 3D FEA artifact field {key} is not finite")
    return result


def _validate_native_state(
    archive: Any,
    *,
    repair_derived_normals: bool = False,
) -> dict[str, Any]:
    """Validate the persisted mechanics state before any optical promotion."""
    node_ids = _integer_array(archive, "node_ids", ndim=1)
    if len(node_ids) == 0 or len(np.unique(node_ids)) != len(node_ids):
        raise ValueError("native 3D FEA node IDs must be nonempty and unique")
    if not np.array_equal(node_ids, np.sort(node_ids)):
        raise ValueError("native 3D FEA node IDs must be sorted deterministically")
    undeformed = _finite_array(archive, "undeformed_nodes_xyz", shape=(len(node_ids), 3))
    deformed = _finite_array(archive, "deformed_nodes_xyz", shape=undeformed.shape)
    displacement = _finite_array(archive, "displacement_xyz", shape=undeformed.shape)
    if not np.allclose(deformed - undeformed, displacement, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("native 3D FEA deformed coordinates and displacement disagree")

    surface_faces = _integer_array(archive, "surface_faces_node_ids", ndim=2)
    if surface_faces.shape[1:] != (3,) or len(surface_faces) == 0:
        raise ValueError("native 3D FEA surface faces must have shape (F, 3)")
    node_set = set(int(value) for value in node_ids)
    for face in surface_faces:
        if len(set(int(value) for value in face)) != 3 or any(int(value) not in node_set for value in face):
            raise ValueError("native 3D FEA surface contains an invalid or duplicate node ID")
    canonical_faces = {tuple(sorted(int(value) for value in face)) for face in surface_faces}
    if len(canonical_faces) != len(surface_faces):
        raise ValueError("native 3D FEA surface contains duplicate triangles")
    tags = _json_scalar(archive, "surface_semantic_tags_json")
    if not isinstance(tags, list) or len(tags) != len(surface_faces) or any(not isinstance(tag, str) or not tag for tag in tags):
        raise ValueError("native 3D FEA surface semantic IDs are invalid")
    reference_normals = _finite_array(
        archive, "surface_reference_normals", shape=(len(surface_faces), 3)
    )
    deformed_normals = _finite_array(
        archive, "surface_deformed_normals", shape=(len(surface_faces), 3)
    )
    if np.any(np.linalg.norm(reference_normals, axis=1) <= 0.0) or np.any(
        np.linalg.norm(deformed_normals, axis=1) <= 0.0
    ):
        raise ValueError("native 3D FEA surface normals must be finite and nonzero")
    node_index = {int(node_id): index for index, node_id in enumerate(node_ids)}
    reference_points = undeformed[
        np.asarray([[node_index[int(value)] for value in face] for face in surface_faces])
    ]
    deformed_points = deformed[
        np.asarray([[node_index[int(value)] for value in face] for face in surface_faces])
    ]
    reference_cross = np.cross(
        reference_points[:, 1] - reference_points[:, 0],
        reference_points[:, 2] - reference_points[:, 0],
    )
    deformed_cross = np.cross(
        deformed_points[:, 1] - deformed_points[:, 0],
        deformed_points[:, 2] - deformed_points[:, 0],
    )
    if np.any(np.linalg.norm(reference_cross, axis=1) <= 1.0e-12) or np.any(
        np.linalg.norm(deformed_cross, axis=1) <= 1.0e-12
    ):
        raise ValueError("native 3D FEA surface contains a zero-area triangle")
    reference_alignment = np.sum(reference_cross * reference_normals, axis=1)
    deformed_alignment = np.sum(deformed_cross * deformed_normals, axis=1)
    if np.any(reference_alignment <= 0.0) or np.any(deformed_alignment <= 0.0):
        if not repair_derived_normals:
            raise ValueError("native 3D FEA surface orientation metadata is inconsistent")
        reference_normals = reference_cross / np.linalg.norm(reference_cross, axis=1)[:, None]
        deformed_normals = deformed_cross / np.linalg.norm(deformed_cross, axis=1)[:, None]
    if np.any(np.sum(reference_cross * deformed_cross, axis=1) <= 0.0):
        raise ValueError("native 3D FEA surface orientation flipped after deformation")
    tetrahedra = _integer_array(archive, "tetrahedra_node_ids", ndim=2)
    if tetrahedra.shape[1:] != (4,) or len(tetrahedra) == 0:
        raise ValueError("native 3D FEA tetrahedra must have shape (T, 4)")
    if any(len(set(int(value) for value in tetrahedron)) != 4 or any(int(value) not in node_set for value in tetrahedron) for tetrahedron in tetrahedra):
        raise ValueError("native 3D FEA tetrahedra contain invalid node IDs")
    silicone_node_ids = _integer_array(archive, "silicone_node_ids", ndim=1)
    if len(silicone_node_ids) == 0 or len(np.unique(silicone_node_ids)) != len(silicone_node_ids):
        raise ValueError("native 3D silicone node IDs must be nonempty and unique")
    if any(int(value) not in node_set for value in silicone_node_ids):
        raise ValueError("native 3D silicone node IDs reference an unknown volume node")
    silicone_vertices = _finite_array(archive, "silicone_vertices")
    if silicone_vertices.shape != (len(silicone_node_ids), 3):
        raise ValueError("native 3D silicone vertices do not match silicone node IDs")
    silicone_faces = _integer_array(archive, "silicone_faces", ndim=2)
    if silicone_faces.shape[1:] != (3,) or len(silicone_faces) == 0 or np.any(silicone_faces >= len(silicone_node_ids)):
        raise ValueError("native 3D silicone faces are invalid")
    silicone_index = {int(node_id): index for index, node_id in enumerate(node_ids)}
    expected_silicone_deformed = deformed[
        [silicone_index[int(node_id)] for node_id in silicone_node_ids]
    ]
    if not np.allclose(silicone_vertices, expected_silicone_deformed, rtol=1.0e-6, atol=1.0e-6):
        raise ValueError("FULL_3D silicone surface is not the direct deformed-node surface")
    return {
        "node_ids": node_ids,
        "undeformed_nodes_xyz": undeformed,
        "deformed_nodes_xyz": deformed,
        "displacement_xyz": displacement,
        "surface_faces_node_ids": surface_faces,
        "surface_semantic_tags": tuple(str(tag) for tag in tags),
        "surface_reference_normals": reference_normals,
        "surface_deformed_normals": deformed_normals,
        "tetrahedra_node_ids": tetrahedra,
        "silicone_node_ids": silicone_node_ids,
    }


@dataclass(frozen=True)
class Full3DSurfaceArtifact:
    """Validated true-deformed-surface artifact and its FEA provenance."""

    artifact_path: Path
    morphology_id: str
    morphology_fingerprint: str
    contact_state_fingerprint: str
    mechanics_source: str
    source_position_mm: tuple[float, float, float]
    source_medium: int
    silicone: TriangleSurface
    rigid: TriangleSurface
    envelope: TriangleSurface
    metadata: Mapping[str, Any]
    node_ids: np.ndarray
    undeformed_nodes_xyz: np.ndarray
    deformed_nodes_xyz: np.ndarray
    displacement_xyz: np.ndarray
    surface_faces_node_ids: np.ndarray
    surface_semantic_tags: tuple[str, ...]
    surface_reference_normals: np.ndarray
    surface_deformed_normals: np.ndarray
    tetrahedra_node_ids: np.ndarray
    silicone_node_ids: np.ndarray
    mesh_fingerprint: str
    mechanics_config_fingerprint: str
    tier: str
    contact_location: str

    def geometry(self, tip: Fingertip):
        return build_full3d_transport_geometry(
            tip,
            silicone=self.silicone,
            rigid=self.rigid,
            envelope=self.envelope,
            source_position_mm=self.source_position_mm,
            source_medium=self.source_medium,
            metadata={
                **dict(self.metadata),
                "fea_artifact": str(self.artifact_path),
                "morphology_id": self.morphology_id,
                "morphology_fingerprint": self.morphology_fingerprint,
                "contact_state_fingerprint": self.contact_state_fingerprint,
                "mechanics_source": self.mechanics_source,
            },
            full3d_surface_provenance="actual_deformed_3d_fea_surface",
        )


def load_full3d_surface_artifact(
    manifest_path: Path,
    *,
    expected_morphology_fingerprint: str,
    expected_contact_state_fingerprint: str,
    repair_derived_normals: bool = False,
) -> Full3DSurfaceArtifact:
    """Load a true 3D surface only when its complete provenance matches."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") not in _ACCEPTED_SCHEMAS:
        raise ValueError(
            "the supplied artifact is not a persisted deformed 3D FEA surface; "
            "a validation summary or 2D state archive cannot be promoted"
        )
    if manifest.get("morphology_fingerprint") != expected_morphology_fingerprint:
        raise ValueError("3D surface morphology fingerprint mismatch")
    if manifest.get("contact_state_fingerprint") != expected_contact_state_fingerprint:
        raise ValueError("3D surface contact-state fingerprint mismatch")
    raw_path = Path(str(manifest.get("native_state_artifact", manifest.get("surface_artifact", ""))))
    surface_path = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
    if not surface_path.exists():
        raise ValueError("persisted deformed 3D surface arrays are missing")
    expected_sha = manifest.get("native_state_sha256", manifest.get("surface_sha256"))
    actual_sha = hashlib.sha256(surface_path.read_bytes()).hexdigest()
    if expected_sha != actual_sha:
        raise ValueError("persisted deformed 3D surface checksum mismatch")
    with np.load(surface_path, allow_pickle=False) as archive:
        metadata = _json_scalar(archive, "metadata_json")
        if not isinstance(metadata, Mapping):
            raise ValueError("3D surface metadata must be a JSON object")
        for field in (
            "morphology_fingerprint",
            "contact_state_fingerprint",
            "mesh_fingerprint",
            "mechanics_config_fingerprint",
            "mechanics_source",
        ):
            if metadata.get(field) != manifest.get(field):
                raise ValueError(f"native 3D FEA metadata mismatch for {field}")
        native_state = _validate_native_state(
            archive,
            repair_derived_normals=repair_derived_normals,
        )
        silicone = _surface(
            archive,
            "silicone",
            semantic=True,
            repair_normals=repair_derived_normals,
        )
        rigid = _surface(archive, "rigid", semantic=False)
        envelope = _surface(archive, "envelope", semantic=False)
    required_manifest_fields = (
        "morphology_id",
        "mechanics_source",
        "source_position_mm",
        "source_medium",
        "mesh_fingerprint",
        "mechanics_config_fingerprint",
        "tier",
        "contact_location",
        "total_prescribed_travel_mm",
        "indenter_radius_mm",
        "initial_gap_mm",
    )
    if any(field not in manifest for field in required_manifest_fields):
        raise ValueError("3D surface artifact is missing required source provenance")
    source = tuple(float(value) for value in manifest["source_position_mm"])
    if len(source) != 3 or not np.all(np.isfinite(source)):
        raise Transport3DGeometryError("3D surface source position is invalid")
    return Full3DSurfaceArtifact(
        artifact_path=surface_path,
        morphology_id=str(manifest["morphology_id"]),
        morphology_fingerprint=str(manifest["morphology_fingerprint"]),
        contact_state_fingerprint=str(manifest["contact_state_fingerprint"]),
        mechanics_source=str(manifest["mechanics_source"]),
        source_position_mm=source,
        source_medium=int(manifest["source_medium"]),
        silicone=silicone,
        rigid=rigid,
        envelope=envelope,
        metadata=dict(metadata),
        node_ids=native_state["node_ids"],
        undeformed_nodes_xyz=native_state["undeformed_nodes_xyz"],
        deformed_nodes_xyz=native_state["deformed_nodes_xyz"],
        displacement_xyz=native_state["displacement_xyz"],
        surface_faces_node_ids=native_state["surface_faces_node_ids"],
        surface_semantic_tags=native_state["surface_semantic_tags"],
        surface_reference_normals=native_state["surface_reference_normals"],
        surface_deformed_normals=native_state["surface_deformed_normals"],
        tetrahedra_node_ids=native_state["tetrahedra_node_ids"],
        silicone_node_ids=native_state["silicone_node_ids"],
        mesh_fingerprint=str(manifest["mesh_fingerprint"]),
        mechanics_config_fingerprint=str(manifest["mechanics_config_fingerprint"]),
        tier=str(manifest["tier"]),
        contact_location=str(manifest["contact_location"]),
    )


__all__ = [
    "FULL3D_SURFACE_SCHEMA",
    "NATIVE_3D_FEA_STATE_SCHEMA",
    "Full3DSurfaceArtifact",
    "load_full3d_surface_artifact",
]
