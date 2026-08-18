"""Small, verifiable persistence format for one :class:`FingertipCase`."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.wkt import loads as load_wkt

from fem import FEAResult
from mesh.indenter import CrownFrame, IndenterFixture, IndenterPose2D, IndenterSettings
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
from model import Fingertip, FingertipParameters, LED, OpticalMaterial
from optics.contact_object import IndenterOptics
from optics.transport3d.result import Transport3DResult
from optics.transport3d.settings import Transport3DSettings
from optics.transport3d.unified import UnifiedTransportResult, fingerprint_mapping

from case.core import CASE_SCHEMA, FingertipCase
from case.fea2d import FEA2D
from case.raytracing2d import RayTracing2D
from case.state import ContactState


MECHANICS_SCHEMA = "fingertip-case-mechanics-v2"
OPTICAL_SCHEMA = "fingertip-case-optics-v3"


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
    if case.fea.result is None or case.fea.result.indenter_pose is None:
        raise ValueError("a persisted FingertipCase requires an FEA indenter pose")
    pose = case.fea.result.indenter_pose
    payload = pose.to_dict()
    payload.update(
        {
            "carrier_geometry_wkt": pose.carrier_geometry.wkt,
            "contact_arc_wkt": pose.contact_arc.wkt,
            "outer_remainder_wkt": pose.outer_remainder.wkt,
        }
    )
    fixture = pose.fixture
    fixture_payload = payload["fixture"]
    fixture_payload.update(
        {
            "carrier_geometry_wkt": fixture.carrier_geometry.wkt,
            "contact_arc_wkt": fixture.contact_arc.wkt,
            "outer_remainder_wkt": fixture.outer_remainder.wkt,
            "frame": {
                "point_mm": list(fixture.frame.point_mm),
                "tangent": list(fixture.frame.tangent),
                "pad_outward_normal": list(fixture.frame.pad_outward_normal),
                "loading_direction": list(fixture.frame.loading_direction),
                "arc_distance_mm": fixture.frame.arc_distance_mm,
            },
        }
    )
    # The old convenience distance is derived from geometry and can vary in
    # its final floating-point bit after WKT round-trip.  Exact WKT is the
    # persisted geometry contract, so do not use that derived scalar as an ID.
    fixture_payload.pop("pad_contact_arc_minimum_distance_mm", None)
    payload["contact_patch_wkt"] = (
        None if pose.contact_patch is None else pose.contact_patch.wkt
    )
    return payload


def _save_mechanics(case: FingertipCase, directory: Path) -> Path:
    mesh_path = directory / "mesh.json"
    arrays_path = directory / "fea.npz"
    manifest_path = directory / "fea.json"
    if case.fea.result is None:
        raise ValueError("a persisted FingertipCase requires an FEA result")
    result = case.fea.result
    if result.reference_mesh is None:
        raise ValueError("a persisted FingertipCase requires reference_mesh")
    _write_json(mesh_path, result.reference_mesh.to_dict())
    if result.displacement is None:
        raise ValueError("a persisted FingertipCase requires displacement")
    _write_npz(arrays_path, displacement=result.displacement)
    metadata = {
        "schema": MECHANICS_SCHEMA,
        "mesh_artifact": mesh_path.name,
        "mesh_sha256": _sha256(mesh_path),
        "arrays_artifact": arrays_path.name,
        "arrays_sha256": _sha256(arrays_path),
        "displacement_key": "displacement",
        "reaction_force_n": result.reaction_force,
        "contact": _jsonable(result.contact),
        "converged": result.converged,
        "details": _jsonable(result.details),
        "element_von_mises_stress_mpa": _jsonable(
            result.element_von_mises_stress_mpa
        ),
        "indenter_pose": _pose_payload(case),
    }
    _write_json(manifest_path, metadata)
    return manifest_path


def _save_optics(case: FingertipCase, directory: Path) -> Path:
    arrays_path = directory / "raytrace.npz"
    manifest_path = directory / "raytrace.json"
    raw = case.raytracing.raw
    result = case.raytracing.summary
    if raw is None or result is None:
        raise ValueError("a persisted FingertipCase requires completed optics")
    arrays: dict[str, Any] = {
        "source_position_mm": raw.source_position_mm,
        "surface_u_edges": raw.surface_u_edges,
        "surface_z_edges": raw.surface_z_edges,
        "outgoing_surface_field": raw.outgoing_surface_field,
        "escape_positions_mm": raw.escape_positions_mm,
        "escape_directions": raw.escape_directions,
        "escape_surface_normals": raw.escape_surface_normals,
        "escape_surface_u": raw.escape_surface_u,
        "escape_surface_z": raw.escape_surface_z,
        "escape_surface_primitive_indices": raw.escape_surface_primitive_indices,
        "escape_weights": raw.escape_weights,
        "escape_primary_ray_indices": raw.escape_primary_ray_indices,
        "escape_path_lengths_mm": raw.escape_path_lengths_mm,
        "escape_interaction_counts": raw.escape_interaction_counts,
        "field": result.field,
    }
    arrays.update(
        {
            f"axis_{index}": axis
            for index, axis in enumerate(result.field_axes)
        }
    )
    optional_arrays = (
        "projected_x_edges_mm",
        "projected_y_edges_mm",
        "projected_weighted_path_density",
        "projected_optical_mask",
        "internal_path_x_edges_mm",
        "internal_path_y_edges_mm",
        "internal_path_z_edges_mm",
        "internal_weighted_path_density_3d",
        "internal_z_integrated_path_density",
        "retained_segment_lengths_mm",
        "retained_segment_primary_ray_indices",
        "retained_segment_interaction_counts",
        "retained_segment_starts_mm",
        "retained_segment_ends_mm",
        "retained_segment_media",
        "retained_segment_start_weights",
        "retained_segment_end_weights",
    )
    arrays.update(
        {
            name: value
            for name in optional_arrays
            if (value := getattr(raw, name)) is not None
        }
    )
    _write_npz(arrays_path, **arrays)
    metadata = {
        "schema": OPTICAL_SCHEMA,
        "arrays_artifact": arrays_path.name,
        "arrays_sha256": _sha256(arrays_path),
        "raw": {
            "source_mode": raw.source_mode,
            "extrusion_depth_mm": raw.extrusion_depth_mm,
            "launched_ray_count": raw.launched_ray_count,
            "launched_weight": raw.launched_weight,
            "escaped_weight": raw.escaped_weight,
            "absorbed_weight": raw.absorbed_weight,
            "terminated_weight": raw.terminated_weight,
            "object_absorbed_weight": raw.object_absorbed_weight,
            "object_transmitted_weight": raw.object_transmitted_weight,
            "object_interface_incident_weight": raw.object_interface_incident_weight,
            "object_reflected_weight": raw.object_reflected_weight,
            "outgoing_surface_weight": raw.outgoing_surface_weight,
            "escape_surface_tags": list(raw.escape_surface_tags),
            "energy_balance_error": raw.energy_balance_error,
            "energy_balance_tolerance": raw.energy_balance_tolerance,
            "geometry_metadata": _jsonable(raw.geometry_metadata),
            "timings_seconds": _jsonable(raw.timings_seconds),
        },
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
            "object_absorbed_weight": result.object_absorbed_weight,
            "object_transmitted_weight": result.object_transmitted_weight,
            "object_interface_incident_weight": result.object_interface_incident_weight,
            "object_reflected_weight": result.object_reflected_weight,
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
    if (
        case.fea.result is None
        or case.raytracing.raw is None
        or case.raytracing.summary is None
    ):
        raise ValueError("a persisted FingertipCase requires completed FEA and optics")
    mechanics_path = _save_mechanics(case, mechanics_directory)
    optical_path = _save_optics(case, optics_directory)
    morphology_fingerprint = case.raytracing.summary.morphology_fingerprint
    top_level = {
        "schema": CASE_SCHEMA,
        "case_id": case.case_id,
        "fingertip_parameters": asdict(case.fingertip.parameters),
        "fingertip_parameters_fingerprint": morphology_fingerprint,
        "indenter_parameters": asdict(case.indenter),
        "mesh_settings": asdict(case.fea.mesh_settings),
        "fem_steps": case.fea.steps,
        "internal_contact": case.fea.internal_contact,
        "basal_interface": case.fea.basal_interface,
        "led": asdict(case.fingertip.led),
        "optical_material": asdict(case.fingertip.optical),
        "trace_settings": asdict(case.raytracing.settings),
        "indenter_optics": (
            None
            if case.raytracing.indenter_optics is None
            else asdict(case.raytracing.indenter_optics)
        ),
        "indenter_pose_fingerprint": fingerprint_mapping(
            _pose_payload(case)
        ),
        "contact_state": asdict(case.contact_state),
        "provenance": dict(case.provenance),
        "optical_mode": case.raytracing.summary.optical_mode,
        "configuration_fingerprint": case.raytracing.summary.transport_configuration_fingerprint,
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


def _load_pose(
    payload: Mapping[str, Any],
    *,
    indenter: IndenterSettings,
) -> IndenterPose2D:
    fixture_payload = payload.get("fixture")
    if not isinstance(fixture_payload, Mapping):
        raise ValueError("case mechanics fixture geometry is missing")
    settings = IndenterSettings(**fixture_payload["settings"])
    if settings != indenter:
        raise ValueError("case mechanics fixture settings mismatch")
    frame_payload = fixture_payload.get("frame")
    if not isinstance(frame_payload, Mapping):
        raise ValueError("case mechanics fixture frame is missing")

    def _vector(value: Any, *, length: int, name: str) -> tuple[float, ...]:
        values = tuple(float(item) for item in value)
        if len(values) != length or not np.all(np.isfinite(values)):
            raise ValueError(f"case mechanics {name} is invalid")
        return values

    def _geometry(value: Any, *, name: str, expected_type: type | tuple[type, ...] | None = None):
        if not isinstance(value, str):
            raise ValueError(f"case mechanics {name} geometry is missing")
        geometry = load_wkt(value)
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"case mechanics {name} geometry is invalid")
        if expected_type is not None and not isinstance(geometry, expected_type):
            raise ValueError(f"case mechanics {name} geometry type is invalid")
        return geometry

    frame = CrownFrame(
        point_mm=_vector(frame_payload["point_mm"], length=2, name="frame point"),
        tangent=_vector(frame_payload["tangent"], length=2, name="frame tangent"),
        pad_outward_normal=_vector(
            frame_payload["pad_outward_normal"],
            length=2,
            name="frame normal",
        ),
        loading_direction=_vector(
            frame_payload["loading_direction"],
            length=2,
            name="frame loading direction",
        ),
        arc_distance_mm=float(frame_payload["arc_distance_mm"]),
    )
    fixture = IndenterFixture(
        settings=settings,
        frame=frame,
        center_mm=_vector(fixture_payload["center_mm"], length=2, name="fixture center"),
        contact_direction=_vector(
            fixture_payload["contact_direction"],
            length=2,
            name="contact direction",
        ),
        carrier_geometry=_geometry(
            fixture_payload["carrier_geometry_wkt"],
            name="fixture carrier",
            expected_type=Polygon,
        ),
        contact_arc=_geometry(
            fixture_payload["contact_arc_wkt"],
            name="fixture contact arc",
            expected_type=LineString,
        ),
        outer_remainder=_geometry(
            fixture_payload["outer_remainder_wkt"],
            name="fixture outer remainder",
            expected_type=MultiLineString,
        ),
    )
    travel = float(payload["prescribed_travel_mm"])
    translation = _vector(payload["translation_mm"], length=2, name="translation")
    expected_translation = fixture.displacement_for_travel(travel)
    if not np.allclose(translation, expected_translation, rtol=0.0, atol=1.0e-12):
        raise ValueError("case mechanics pose translation is inconsistent with fixture")
    center = _vector(payload["center_mm"], length=2, name="pose center")
    expected_center = tuple(
        fixture.center_mm[index] + translation[index] for index in range(2)
    )
    if not np.allclose(center, expected_center, rtol=0.0, atol=1.0e-12):
        raise ValueError("case mechanics pose center is inconsistent with fixture")
    carrier = _geometry(
        payload["carrier_geometry_wkt"],
        name="posed carrier",
        expected_type=Polygon,
    )
    contact_arc = _geometry(
        payload["contact_arc_wkt"],
        name="posed contact arc",
        expected_type=LineString,
    )
    outer_remainder = _geometry(
        payload["outer_remainder_wkt"],
        name="posed outer remainder",
        expected_type=MultiLineString,
    )
    if not carrier.equals_exact(
        affinity.translate(fixture.carrier_geometry, xoff=translation[0], yoff=translation[1]),
        1.0e-9,
    ):
        raise ValueError("case mechanics carrier geometry is inconsistent with pose")
    if not contact_arc.equals_exact(
        affinity.translate(fixture.contact_arc, xoff=translation[0], yoff=translation[1]),
        1.0e-9,
    ):
        raise ValueError("case mechanics contact arc is inconsistent with pose")
    if not outer_remainder.equals_exact(
        affinity.translate(fixture.outer_remainder, xoff=translation[0], yoff=translation[1]),
        1.0e-9,
    ):
        raise ValueError("case mechanics outer remainder is inconsistent with pose")
    patch_wkt = payload.get("contact_patch_wkt")
    contact_patch = (
        None
        if patch_wkt is None
        else _geometry(
            patch_wkt,
            name="contact patch",
            expected_type=(LineString, MultiLineString),
        )
    )
    return IndenterPose2D(
        fixture=fixture,
        prescribed_travel_mm=travel,
        translation_mm=translation,
        center_mm=center,
        carrier_geometry=carrier,
        contact_arc=contact_arc,
        outer_remainder=outer_remainder,
        contact_patch=contact_patch,
        active_contact_node_ids=tuple(
            int(value) for value in payload.get("active_contact_node_ids", ())
        ),
    )


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
    pose_metadata = metadata.get("indenter_pose")
    if not isinstance(pose_metadata, Mapping):
        raise ValueError("case mechanics indenter pose is missing")
    pose = _load_pose(pose_metadata, indenter=indenter)
    return FEAResult(
        mesh=mesh.pad,
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
        reference_mesh=mesh,
        element_von_mises_stress_mpa=(
            None
            if metadata.get("element_von_mises_stress_mpa") is None
            else {
                int(element_id): float(value)
                for element_id, value in metadata[
                    "element_von_mises_stress_mpa"
                ].items()
            }
        ),
    )


def _load_optics(path: Path) -> tuple[Transport3DResult, UnifiedTransportResult]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema") != OPTICAL_SCHEMA:
        raise ValueError("unsupported case optical schema")
    arrays_path = path.parent / str(metadata["arrays_artifact"])
    if not arrays_path.exists() or _sha256(arrays_path) != metadata.get("arrays_sha256"):
        raise ValueError("case optical array checksum mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        loaded_arrays = {
            name: np.asarray(archive[name]) for name in archive.files
        }

    def required(name: str) -> np.ndarray:
        if name not in loaded_arrays:
            raise ValueError(f"case optical array is missing: {name}")
        return loaded_arrays[name]

    def optional(name: str) -> np.ndarray | None:
        return loaded_arrays.get(name)

    field = np.asarray(required("field"), dtype=float)
    axes = tuple(
        np.asarray(required(f"axis_{index}"), dtype=float)
        for index in range(field.ndim)
    )
    record = metadata.get("result")
    if not isinstance(record, Mapping):
        raise ValueError("case optical result metadata is missing")
    raw_record = metadata.get("raw")
    if not isinstance(raw_record, Mapping):
        raise ValueError("case raw optical result metadata is missing")
    raw = Transport3DResult(
        source_position_mm=tuple(
            float(value) for value in required("source_position_mm")
        ),
        source_mode=str(raw_record["source_mode"]),
        extrusion_depth_mm=float(raw_record["extrusion_depth_mm"]),
        launched_ray_count=int(raw_record["launched_ray_count"]),
        launched_weight=float(raw_record["launched_weight"]),
        escaped_weight=float(raw_record["escaped_weight"]),
        absorbed_weight=float(raw_record["absorbed_weight"]),
        terminated_weight=float(raw_record["terminated_weight"]),
        object_absorbed_weight=float(raw_record["object_absorbed_weight"]),
        object_transmitted_weight=float(raw_record["object_transmitted_weight"]),
        object_interface_incident_weight=float(
            raw_record["object_interface_incident_weight"]
        ),
        object_reflected_weight=float(raw_record["object_reflected_weight"]),
        outgoing_surface_weight=float(raw_record["outgoing_surface_weight"]),
        surface_u_edges=required("surface_u_edges"),
        surface_z_edges=required("surface_z_edges"),
        outgoing_surface_field=required("outgoing_surface_field"),
        escape_positions_mm=required("escape_positions_mm"),
        escape_directions=required("escape_directions"),
        escape_surface_normals=required("escape_surface_normals"),
        escape_surface_u=required("escape_surface_u"),
        escape_surface_z=required("escape_surface_z"),
        escape_surface_tags=tuple(str(value) for value in raw_record["escape_surface_tags"]),
        escape_surface_primitive_indices=required("escape_surface_primitive_indices"),
        escape_weights=required("escape_weights"),
        escape_primary_ray_indices=required("escape_primary_ray_indices"),
        escape_path_lengths_mm=required("escape_path_lengths_mm"),
        escape_interaction_counts=required("escape_interaction_counts"),
        energy_balance_error=float(raw_record["energy_balance_error"]),
        energy_balance_tolerance=float(raw_record["energy_balance_tolerance"]),
        projected_x_edges_mm=optional("projected_x_edges_mm"),
        projected_y_edges_mm=optional("projected_y_edges_mm"),
        projected_weighted_path_density=optional("projected_weighted_path_density"),
        projected_optical_mask=optional("projected_optical_mask"),
        internal_path_x_edges_mm=optional("internal_path_x_edges_mm"),
        internal_path_y_edges_mm=optional("internal_path_y_edges_mm"),
        internal_path_z_edges_mm=optional("internal_path_z_edges_mm"),
        internal_weighted_path_density_3d=optional("internal_weighted_path_density_3d"),
        internal_z_integrated_path_density=optional("internal_z_integrated_path_density"),
        retained_segment_lengths_mm=optional("retained_segment_lengths_mm"),
        retained_segment_primary_ray_indices=optional("retained_segment_primary_ray_indices"),
        retained_segment_interaction_counts=optional("retained_segment_interaction_counts"),
        retained_segment_starts_mm=optional("retained_segment_starts_mm"),
        retained_segment_ends_mm=optional("retained_segment_ends_mm"),
        retained_segment_media=optional("retained_segment_media"),
        retained_segment_start_weights=optional("retained_segment_start_weights"),
        retained_segment_end_weights=optional("retained_segment_end_weights"),
        geometry_metadata=raw_record.get("geometry_metadata", {}),
        timings_seconds=raw_record.get("timings_seconds", {}),
    )
    summary = UnifiedTransportResult(
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
        object_absorbed_weight=float(record["object_absorbed_weight"]),
        object_transmitted_weight=float(record["object_transmitted_weight"]),
        object_interface_incident_weight=float(
            record["object_interface_incident_weight"]
        ),
        object_reflected_weight=float(record["object_reflected_weight"]),
    )
    return raw, summary


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
    mesh_settings = MeshSettings(**manifest["mesh_settings"])
    led_payload = dict(manifest["led"])
    led_payload["emission_rgb"] = tuple(led_payload["emission_rgb"])
    led = LED(**led_payload)
    optical = OpticalMaterial(**manifest["optical_material"])
    trace_payload = dict(manifest["trace_settings"])
    for key in ("x_bounds_mm", "y_bounds_mm"):
        if trace_payload.get(key) is not None:
            trace_payload[key] = tuple(trace_payload[key])
    trace_settings = Transport3DSettings(**trace_payload)
    indenter_optics_payload = manifest.get("indenter_optics")
    indenter_optics = (
        None
        if indenter_optics_payload is None
        else IndenterOptics(**indenter_optics_payload)
    )
    fea = _load_mechanics(
        mechanics_path,
        parameters=parameters,
        indenter=indenter,
        contact_state=contact_state,
    )
    raytrace, optics = _load_optics(optical_path)
    if manifest.get("fingertip_parameters_fingerprint") != optics.morphology_fingerprint:
        raise ValueError("case morphology fingerprint mismatch")
    fingertip = Fingertip(
        parameters,
        led=led,
        optical=optical,
    )
    fea_config = FEA2D(
        indenter=indenter,
        contact=contact_state,
        mesh_settings=mesh_settings,
        steps=int(manifest["fem_steps"]),
        internal_contact=str(manifest["internal_contact"]),
        basal_interface=str(manifest.get("basal_interface", "bonded")),
    )
    fea_config.result = fea
    raytracing = RayTracing2D(
        settings=trace_settings,
        indenter_optics=indenter_optics,
    )
    raytracing.raw = raytrace
    raytracing.summary = optics
    case = FingertipCase(
        fingertip=fingertip,
        fea=fea_config,
        raytracing=raytracing,
        provenance=manifest.get("provenance", {}),
    )
    if manifest.get("case_id") != case.case_id:
        raise ValueError("case ID mismatch")
    if manifest.get("indenter_pose_fingerprint") != fingerprint_mapping(
        _pose_payload(case)
    ):
        raise ValueError("case indenter-pose fingerprint mismatch")
    if manifest.get("optical_mode") != case.raytracing.summary.optical_mode:
        raise ValueError("case optical mode mismatch")
    if manifest.get("configuration_fingerprint") != (
        case.raytracing.summary.transport_configuration_fingerprint
    ):
        raise ValueError("case optical configuration fingerprint mismatch")
    return case


__all__ = ["load_case", "save_case"]
