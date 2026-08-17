"""Small, verifiable persistence format for one :class:`FingertipCase`."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from shapely.wkt import loads as load_wkt

from fem import FEAResult
from mesh.indenter import (
    IndenterSettings,
    pose_from_fixture,
    build_normal_indenter_fixture_at_x,
)
from mesh.types import (
    BoundaryEdge,
    FingertipMesh,
    MeshQualityStatistics,
    MeshSettings,
    MeshValidationReport,
    MeshedContactPair,
    MeshNode,
    T3Element,
)
from model import Fingertip, FingertipParameters
from optics.transport3d.unified import UnifiedTransportResult, fingerprint_mapping

from case.core import CASE_SCHEMA, ContactState, FingertipCase


MECHANICS_SCHEMA = "fingertip-case-mechanics-v1"
OPTICAL_SCHEMA = "fingertip-case-optics-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _mesh_from_dict(payload: Mapping[str, Any]) -> FingertipMesh:
    parameters = FingertipParameters(**payload["parameters"])
    settings = MeshSettings(**payload["settings"])
    nodes = {
        int(node_id): MeshNode(
            id=int(record["id"]),
            x_mm=float(record["x_mm"]),
            y_mm=float(record["y_mm"]),
            domain=str(record["domain"]),
        )
        for node_id, record in payload["nodes"].items()
    }
    pad_elements = tuple(
        T3Element(
            id=int(record["id"]),
            node_ids=tuple(int(value) for value in record["node_ids"]),
            domain=str(record["domain"]),
        )
        for record in payload["pad_elements"]
    )
    carrier_elements = tuple(
        T3Element(
            id=int(record["id"]),
            node_ids=tuple(int(value) for value in record["node_ids"]),
            domain=str(record["domain"]),
        )
        for record in payload["carrier_elements"]
    )
    boundary_edges = {
        str(tag): tuple(
            BoundaryEdge(
                node_ids=tuple(int(value) for value in edge["node_ids"]),
                domain=str(edge["domain"]),
            )
            for edge in edges
        )
        for tag, edges in payload["boundary_edges"].items()
    }
    contact_pairs = tuple(
        MeshedContactPair(
            name=str(record["name"]),
            pad_boundary_tag=str(record["pad_boundary_tag"]),
            stem_boundary_tag=str(record["stem_boundary_tag"]),
            initial_normal_gap_mm=float(record["initial_normal_gap_mm"]),
            measured_mesh_gap_mm=float(record["measured_mesh_gap_mm"]),
        )
        for record in payload["contact_pairs"]
    )
    quality = MeshQualityStatistics(
        **{
            key: (
                tuple(float(value) for value in value)
                if key == "minimum_triangle_angle_centroid_mm"
                else value
            )
            for key, value in payload["quality"].items()
        }
    )
    validation_data = payload["validation"]
    validation = MeshValidationReport(
        passed=bool(validation_data["passed"]),
        checks={str(key): bool(value) for key, value in validation_data["checks"].items()},
        errors=tuple(str(value) for value in validation_data["errors"]),
    )
    return FingertipMesh(
        nodes=nodes,
        pad_elements=pad_elements,
        carrier_elements=carrier_elements,
        domain_node_ids={
            str(key): tuple(int(value) for value in values)
            for key, values in payload["domain_node_ids"].items()
        },
        domain_element_ids={
            str(key): tuple(int(value) for value in values)
            for key, values in payload["domain_element_ids"].items()
        },
        boundary_edges=boundary_edges,
        contact_pairs=contact_pairs,
        parameters=parameters,
        settings=settings,
        quality=quality,
        validation=validation,
        gmsh_version=str(payload["gmsh_version"]),
    )


def _pose_payload(case: FingertipCase) -> dict[str, Any]:
    pose = case.indenter_pose
    payload = pose.to_dict()
    payload["contact_patch_wkt"] = (
        None if pose.contact_patch is None else pose.contact_patch.wkt
    )
    return payload


def _save_mechanics(case: FingertipCase, directory: Path) -> Path:
    mesh_path = directory / "mesh.json"
    arrays_path = directory / "fea.npz"
    manifest_path = directory / "fea.json"
    _write_json(mesh_path, case.fea.mesh.to_dict())
    if case.fea.displacement is None:
        raise ValueError("a persisted FingertipCase requires displacement")
    _write_npz(arrays_path, displacement=case.fea.displacement)
    metadata = {
        "schema": MECHANICS_SCHEMA,
        "mesh_artifact": mesh_path.name,
        "mesh_sha256": _sha256(mesh_path),
        "arrays_artifact": arrays_path.name,
        "arrays_sha256": _sha256(arrays_path),
        "displacement_key": "displacement",
        "reaction_force_n": case.fea.reaction_force,
        "contact": _jsonable(case.fea.contact),
        "converged": case.fea.converged,
        "details": _jsonable(case.fea.details),
        "indenter_pose": _pose_payload(case),
    }
    _write_json(manifest_path, metadata)
    return manifest_path


def _save_optics(case: FingertipCase, directory: Path) -> Path:
    arrays_path = directory / "raytrace.npz"
    manifest_path = directory / "raytrace.json"
    _write_npz(
        arrays_path,
        field=case.raytrace.field,
        **{
            f"axis_{index}": axis
            for index, axis in enumerate(case.raytrace.field_axes)
        },
    )
    result = case.raytrace
    metadata = {
        "schema": OPTICAL_SCHEMA,
        "arrays_artifact": arrays_path.name,
        "arrays_sha256": _sha256(arrays_path),
        "field_dimension": result.field.ndim,
        "result": {
            "morphology_id": result.morphology_id,
            "morphology_fingerprint": result.morphology_fingerprint,
            "mechanics_source": result.mechanics_source,
            "mechanics_dimension": result.mechanics_dimension,
            "contact_state": _jsonable(result.contact_state),
            "optical_mode": result.optical_mode,
            "ray_count": result.ray_count,
            "transport_configuration_fingerprint": result.transport_configuration_fingerprint,
            "total_transport": result.total_transport,
            "launched_weight": result.launched_weight,
            "escaped_weight": result.escaped_weight,
            "absorbed_weight": result.absorbed_weight,
            "terminated_weight": result.terminated_weight,
            "valid_ray_count": result.valid_ray_count,
            "terminated_ray_count": result.terminated_ray_count,
            "energy_balance_error": result.energy_balance_error,
            "path_diagnostics": _jsonable(result.path_diagnostics),
        },
    }
    _write_json(manifest_path, metadata)
    return manifest_path


def save_case(case: FingertipCase, root: str | Path) -> Path:
    """Persist one case under ``<root>/<case_id>/case.json`` semantics."""
    if not isinstance(case, FingertipCase):
        raise TypeError("case must be a FingertipCase")
    requested = Path(root)
    if requested.suffix == ".json":
        case_directory = requested.parent
        manifest_path = requested
    elif requested.name == case.case_id:
        case_directory = requested
        manifest_path = case_directory / "case.json"
    else:
        case_directory = requested / case.case_id
        manifest_path = case_directory / "case.json"
    mechanics_directory = case_directory / "mechanics"
    optics_directory = case_directory / "optics"
    mechanics_directory.mkdir(parents=True, exist_ok=True)
    optics_directory.mkdir(parents=True, exist_ok=True)
    mechanics_path = _save_mechanics(case, mechanics_directory)
    optical_path = _save_optics(case, optics_directory)
    morphology_fingerprint = case.raytrace.morphology_fingerprint
    top_level = {
        "schema": CASE_SCHEMA,
        "case_id": case.case_id,
        "fingertip_parameters": asdict(case.fingertip_parameters),
        "fingertip_parameters_fingerprint": morphology_fingerprint,
        "indenter_parameters": asdict(case.indenter),
        "indenter_pose_fingerprint": fingerprint_mapping(
            case.indenter_pose.to_dict()
        ),
        "contact_state": asdict(case.contact_state),
        "provenance": dict(case.provenance),
        "optical_mode": case.raytrace.optical_mode,
        "configuration_fingerprint": case.raytrace.transport_configuration_fingerprint,
        "mechanics": {
            "artifact": str(mechanics_path.relative_to(case_directory)),
            "artifact_sha256": _sha256(mechanics_path),
        },
        "optical": {
            "artifact": str(optical_path.relative_to(case_directory)),
            "artifact_sha256": _sha256(optical_path),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, top_level)
    return manifest_path


def _checked_child(root: Path, record: Mapping[str, Any], key: str) -> Path:
    relative = Path(str(record[key]))
    path = root / relative
    if not path.exists() or not path.is_file():
        raise ValueError(f"case {key} artifact is missing: {path}")
    if _sha256(path) != record.get(f"{key}_sha256"):
        raise ValueError(f"case {key} artifact checksum mismatch: {path}")
    return path


def _load_mechanics(
    path: Path,
    *,
    parameters: FingertipParameters,
    indenter: IndenterSettings,
    contact_state: ContactState,
) -> FEAResult:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema") != MECHANICS_SCHEMA:
        raise ValueError("unsupported case mechanics schema")
    mesh_path = path.parent / str(metadata["mesh_artifact"])
    arrays_path = path.parent / str(metadata["arrays_artifact"])
    if not mesh_path.exists() or _sha256(mesh_path) != metadata.get("mesh_sha256"):
        raise ValueError("case mechanics mesh checksum mismatch")
    if not arrays_path.exists() or _sha256(arrays_path) != metadata.get("arrays_sha256"):
        raise ValueError("case mechanics array checksum mismatch")
    mesh = _mesh_from_dict(json.loads(mesh_path.read_text(encoding="utf-8")))
    if mesh.parameters != parameters:
        raise ValueError("case mechanics mesh parameters mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        key = str(metadata.get("displacement_key", "displacement"))
        if key not in archive:
            raise ValueError("case mechanics displacement is missing")
        displacement = np.asarray(archive[key], dtype=float)
    tip = Fingertip(parameters)
    fixture = build_normal_indenter_fixture_at_x(
        tip.geometry,
        contact_state.location_x_mm,
        indenter,
    )
    pose_metadata = metadata.get("indenter_pose")
    if not isinstance(pose_metadata, Mapping):
        raise ValueError("case mechanics indenter pose is missing")
    patch_wkt = pose_metadata.get("contact_patch_wkt")
    contact_patch = None if patch_wkt is None else load_wkt(str(patch_wkt))
    pose = pose_from_fixture(
        fixture,
        float(pose_metadata["prescribed_travel_mm"]),
        contact_patch=contact_patch,
        active_contact_node_ids=tuple(
            int(value) for value in pose_metadata.get("active_contact_node_ids", ())
        ),
    )
    return FEAResult(
        mesh=mesh,
        displacement=displacement,
        reaction_force=(
            None
            if metadata.get("reaction_force_n") is None
            else float(metadata["reaction_force_n"])
        ),
        contact=metadata.get("contact", {}),
        converged=bool(metadata["converged"]),
        details=metadata.get("details", {}),
        indenter_pose=pose,
    )


def _load_optics(path: Path) -> UnifiedTransportResult:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema") != OPTICAL_SCHEMA:
        raise ValueError("unsupported case optical schema")
    arrays_path = path.parent / str(metadata["arrays_artifact"])
    if not arrays_path.exists() or _sha256(arrays_path) != metadata.get("arrays_sha256"):
        raise ValueError("case optical array checksum mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        field = np.asarray(archive["field"], dtype=float)
        axes = tuple(
            np.asarray(archive[f"axis_{index}"], dtype=float)
            for index in range(field.ndim)
        )
    record = metadata.get("result")
    if not isinstance(record, Mapping):
        raise ValueError("case optical result metadata is missing")
    return UnifiedTransportResult(
        morphology_id=str(record["morphology_id"]),
        morphology_fingerprint=str(record["morphology_fingerprint"]),
        mechanics_source=str(record["mechanics_source"]),
        mechanics_dimension=str(record["mechanics_dimension"]),
        contact_state=record["contact_state"],
        optical_mode=str(record["optical_mode"]),
        ray_count=int(record["ray_count"]),
        transport_configuration_fingerprint=str(
            record["transport_configuration_fingerprint"]
        ),
        field=field,
        field_axes=axes,
        total_transport=float(record["total_transport"]),
        launched_weight=float(record["launched_weight"]),
        escaped_weight=float(record["escaped_weight"]),
        absorbed_weight=float(record["absorbed_weight"]),
        terminated_weight=float(record["terminated_weight"]),
        valid_ray_count=int(record["valid_ray_count"]),
        terminated_ray_count=int(record["terminated_ray_count"]),
        energy_balance_error=float(record["energy_balance_error"]),
        path_diagnostics=record.get("path_diagnostics", {}),
    )


def load_case(path: str | Path) -> FingertipCase:
    """Load and cross-check one case from its top-level ``case.json``."""
    requested = Path(path)
    manifest_path = requested / "case.json" if requested.is_dir() else requested
    if not manifest_path.exists():
        raise ValueError(f"case manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CASE_SCHEMA:
        raise ValueError("unsupported FingertipCase schema")
    root = manifest_path.parent
    mechanics_path = _checked_child(root, manifest["mechanics"], "artifact")
    optical_path = _checked_child(root, manifest["optical"], "artifact")
    parameters = FingertipParameters(**manifest["fingertip_parameters"])
    indenter = IndenterSettings(**manifest["indenter_parameters"])
    contact_state = ContactState(**manifest["contact_state"])
    fea = _load_mechanics(
        mechanics_path,
        parameters=parameters,
        indenter=indenter,
        contact_state=contact_state,
    )
    raytrace = _load_optics(optical_path)
    if manifest.get("fingertip_parameters_fingerprint") != raytrace.morphology_fingerprint:
        raise ValueError("case morphology fingerprint mismatch")
    case = FingertipCase(
        fingertip_parameters=parameters,
        indenter_parameters=indenter,
        contact_state=contact_state,
        fea=fea,
        raytrace=raytrace,
        case_id=str(manifest["case_id"]),
        provenance=manifest.get("provenance", {}),
    )
    if manifest.get("indenter_pose_fingerprint") != fingerprint_mapping(
        case.indenter_pose.to_dict()
    ):
        raise ValueError("case indenter-pose fingerprint mismatch")
    if manifest.get("optical_mode") != case.raytrace.optical_mode:
        raise ValueError("case optical mode mismatch")
    if manifest.get("configuration_fingerprint") != (
        case.raytrace.transport_configuration_fingerprint
    ):
        raise ValueError("case optical configuration fingerprint mismatch")
    return case


__all__ = ["load_case", "save_case"]
