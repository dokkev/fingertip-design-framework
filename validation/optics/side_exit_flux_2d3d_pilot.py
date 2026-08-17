"""Native 2D/3D side-exit optical pilot.

The pilot consumes the existing localized-load mechanics states and runs only
the explicitly selected optical traces.  It persists exit-event weights
against reference/material surface elements so the front/side partition can be
changed without rerunning transport.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fem.solid3d import localized_profile
from mesh.fingertip import generate_fingertip_mesh
from mesh.types import mesh_settings_for_level
from model import Fingertip, FingertipParameters
from optics.transport3d import (
    UnifiedTransportResult,
    build_full3d_transport_geometry,
    fingerprint_mapping,
    load_full3d_surface_artifact,
    trace_geometry,
    transport_configuration,
)
from optics.transport3d.geometry import TriangleSurface, build_transport_geometry
from optics.transport3d.optix_backend import create_runtime
from optics.transport3d.settings import Transport3DSettings
from validation import overnight_force_localized_trend as localized
from validation.common.io import atomic_write_json, strict_read_json
from validation.common.provenance import sha256_file


OUTPUT = Path("output/validation/overnight_force_localized_trend/side_exit_flux_2d3d_pilot_v1")
TRACE_OUTPUT = OUTPUT / "traces"
PILOT_SCHEMA = "side-exit-flux-2d3d-pilot-v1"
TRACE_SCHEMA = "side-exit-surface-artifact-v1"
PILOT_ID = "side_exit_flux_2d3d_pilot_v1"
PILOT_MORPHOLOGY_IDS = (
    "base_00_nominal__FIXED",
    "base_01_candidate49__VARIED",
    "base_03_lhs_02__FIXED",
)
THETAS_DEG = (30.0, 45.0, 60.0)
EXTERNAL_2D_TAGS = frozenset({"pad_outer_left", "pad_outer_arc", "pad_outer_right"})
EXTERNAL_3D_TAG_PREFIX = "outer_compliant_"
RELATIVE_GAIN_MIN_BASELINE = 1.0e-14
ACCOUNTING_RTOL = 1.0e-10
ACCOUNTING_ATOL = 1.0e-12


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{__import__('time').time_ns()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _resolve_path(value: str | Path, *, relative_to: Path | None = None) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path]
    if relative_to is not None:
        candidates.extend((relative_to.parent / path, relative_to.parent / path.name))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _case_rows() -> list[dict[str, Any]]:
    morphologies = localized._morphologies()
    missing = [name for name in PILOT_MORPHOLOGY_IDS if name not in morphologies]
    if missing:
        raise RuntimeError(f"pilot morphology is missing from the authoritative manifest: {missing}")
    rows = [dict(morphologies[name]) for name in PILOT_MORPHOLOGY_IDS]
    fingerprints = [row["morphology_fingerprint"] for row in rows]
    if len(set(fingerprints)) != len(fingerprints):
        raise RuntimeError("pilot morphologies are not three unique physical configurations")
    return rows


def _mechanics_path(dimension: str, morphology_id: str, side: str) -> Path:
    return localized.OUTPUT / ("fea2d" if dimension == "2D" else "fea3d") / f"{morphology_id}__{side}.json"


def _loaded_mechanics_record(dimension: str, morphology: Mapping[str, Any], side: str) -> tuple[Path, dict[str, Any]]:
    path = _mechanics_path(dimension, str(morphology["morphology_id"]), side)
    record = strict_read_json(path)
    checks = {
        "status": record.get("status") == "PASS",
        "experiment_fingerprint": record.get("experiment_fingerprint") == localized._manifest()["experiment_fingerprint"],
        "morphology_fingerprint": record.get("morphology_fingerprint") == morphology["morphology_fingerprint"],
        "external_contact": record.get("external_contact") is False,
        "footprint_radius_mm": math.isclose(float(record.get("footprint_radius_mm")), localized.RADIUS_MM),
        "force_target_n": math.isclose(float(record.get("force_target_n")), float(localized._production_force())),
    }
    if not all(checks.values()):
        raise RuntimeError(f"invalid exact-fingerprint {dimension} mechanics artifact {path}: {checks}")
    return path, record


def _reference_geometry_checksum_2d(reference_mesh: Any) -> str:
    return _fingerprint(
        {
            "node_ids": np.asarray(reference_mesh.node_ids),
            "coordinates": np.asarray(reference_mesh.reference_coordinates_mm),
            "triangles": np.asarray(reference_mesh.triangles),
            "boundary_edges": np.asarray(reference_mesh.boundary_edges),
            "boundary_tags": {
                tag: np.asarray(reference_mesh.boundary_edges_for(tag))
                for tag in reference_mesh.semantic_boundary_tags
            },
        }
    )


def _reference_geometry_checksum_3d(artifact: Any) -> str:
    return _fingerprint(
        {
            "node_ids": np.asarray(artifact.node_ids),
            "undeformed_nodes_xyz": np.asarray(artifact.undeformed_nodes_xyz),
            "surface_faces_node_ids": np.asarray(artifact.surface_faces_node_ids),
            "surface_semantic_tags": artifact.surface_semantic_tags,
        }
    )


def _outward_normal_2d(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    direction = np.asarray(second, dtype=float) - np.asarray(first, dtype=float)
    length = float(np.linalg.norm(direction))
    if length <= 0.0 or not math.isfinite(length):
        raise ValueError("reference boundary edge has zero length")
    return np.asarray([direction[1], -direction[0]], dtype=float) / length


def _is_external_tag(dimension: str, tag: str) -> bool:
    return tag in EXTERNAL_2D_TAGS if dimension == "2D" else tag.startswith(EXTERNAL_3D_TAG_PREFIX)


def _planar_elements(reference_mesh: Any) -> tuple[list[dict[str, Any]], dict[int, int]]:
    pad = reference_mesh.pad
    edge_tags: dict[tuple[int, int], str] = {}
    for tag in pad.semantic_boundary_tags:
        for edge in pad.boundary_edges_for(tag):
            edge_tags[tuple(sorted((int(edge[0]), int(edge[1]))))] = tag
    elements = []
    for edge_id, edge in enumerate(np.asarray(pad.boundary_edges, dtype=int)):
        first, second = int(edge[0]), int(edge[1])
        p0, p1 = pad.reference_coordinates_mm[first], pad.reference_coordinates_mm[second]
        tag = edge_tags[(min(first, second), max(first, second))]
        length = float(np.linalg.norm(p1 - p0))
        elements.append(
            {
                "element_id": edge_id,
                "semantic_tag": tag,
                "reference_centroid_mm": ((p0 + p1) * 0.5).tolist(),
                "reference_normal": _outward_normal_2d(p0, p1).tolist(),
                "reference_measure": length,
                "eligible_external": _is_external_tag("2D", tag),
            }
        )
    # The 2D cross-section is extruded into two OptiX side triangles per
    # boundary edge.  Keep one reference/material element per edge while
    # mapping both runtime primitives back to that edge.
    primitive_to_element = {
        2 * element["element_id"] + local_triangle: element["element_id"]
        for element in elements
        for local_triangle in (0, 1)
    }
    return elements, primitive_to_element


def _lateral_indices(artifact: Any) -> list[int]:
    return [
        index
        for index, tag in enumerate(artifact.surface_semantic_tags)
        if not str(tag).startswith("longitudinal_end_")
    ]


def _three_d_elements(artifact: Any) -> tuple[list[dict[str, Any]], dict[int, int], list[int]]:
    node_index = {int(node_id): index for index, node_id in enumerate(artifact.node_ids)}
    lateral = _lateral_indices(artifact)
    elements = []
    primitive_to_element: dict[int, int] = {}
    for primitive, full_index in enumerate(lateral):
        face = np.asarray(artifact.surface_faces_node_ids[full_index], dtype=int)
        points = np.asarray([artifact.undeformed_nodes_xyz[node_index[int(node_id)]] for node_id in face])
        cross = np.cross(points[1] - points[0], points[2] - points[0])
        area = 0.5 * float(np.linalg.norm(cross))
        normal = np.asarray(artifact.surface_reference_normals[full_index], dtype=float)
        if area <= 0.0 or not np.all(np.isfinite(normal)) or np.linalg.norm(normal) <= 0.0:
            raise ValueError(f"invalid reference 3D surface triangle {full_index}")
        tag = str(artifact.surface_semantic_tags[full_index])
        element = {
            "element_id": int(full_index),
            "semantic_tag": tag,
            "reference_centroid_mm": points.mean(axis=0).tolist(),
            "reference_normal": (normal / np.linalg.norm(normal)).tolist(),
            "reference_measure": area,
            "eligible_external": _is_external_tag("3D", tag),
        }
        elements.append(element)
        primitive_to_element[primitive] = int(full_index)
    return elements, primitive_to_element, lateral


def _contact_direction_2d(reference_mesh: Any, side: str) -> np.ndarray:
    coefficient = np.zeros(2, dtype=float)
    center_x = localized.SIDES[side]
    pad = reference_mesh.pad
    for edge in pad.boundary_edges_for("pad_outer_arc"):
        first, second = (pad.reference_coordinates_mm[int(node)] for node in edge)
        length = float(np.linalg.norm(second - first))
        midpoint = 0.5 * (first + second)
        profile = localized_profile(abs(float(midpoint[0]) - center_x), localized.RADIUS_MM)
        inward = -_outward_normal_2d(first, second)
        coefficient += length * profile * inward
    magnitude = float(np.linalg.norm(coefficient))
    if magnitude <= 0.0:
        raise ValueError(f"2D localized load has no reference contact direction for {side}")
    # The load implementation stores the inward material force.  The
    # contact-facing direction is the opposite, outward direction.
    return -coefficient / magnitude


def _contact_direction_3d(elements: Sequence[Mapping[str, Any]], side: str) -> np.ndarray:
    coefficient = np.zeros(3, dtype=float)
    center_x = localized.SIDES[side]
    for element in elements:
        if not element["eligible_external"]:
            continue
        centroid = np.asarray(element["reference_centroid_mm"], dtype=float)
        distance = math.hypot(float(centroid[0]) - center_x, float(centroid[2]))
        profile = localized_profile(distance, localized.RADIUS_MM)
        normal = np.asarray(element["reference_normal"], dtype=float)
        coefficient += float(element["reference_measure"]) * profile * (-normal)
    magnitude = float(np.linalg.norm(coefficient))
    if magnitude <= 0.0:
        raise ValueError(f"3D localized load has no reference contact direction for {side}")
    return -coefficient / magnitude


def _partition(elements: Sequence[Mapping[str, Any]], contact_direction: np.ndarray, theta_deg: float) -> dict[str, Any]:
    if theta_deg not in THETAS_DEG:
        raise ValueError(f"unsupported region threshold: {theta_deg}")
    direction = np.asarray(contact_direction, dtype=float)
    direction /= np.linalg.norm(direction)
    cosine = math.cos(math.radians(theta_deg))
    front_ids: list[int] = []
    side_ids: list[int] = []
    eligible_ids: list[int] = []
    eligible_measure = 0.0
    front_measure = 0.0
    side_measure = 0.0
    for element in elements:
        if not element["eligible_external"]:
            continue
        element_id = int(element["element_id"])
        normal = np.asarray(element["reference_normal"], dtype=float)
        alignment = float(np.dot(normal / np.linalg.norm(normal), direction))
        eligible_ids.append(element_id)
        measure = float(element["reference_measure"])
        eligible_measure += measure
        if alignment >= cosine:
            front_ids.append(element_id)
            front_measure += measure
        else:
            side_ids.append(element_id)
            side_measure += measure
    if not eligible_ids or not side_ids:
        raise ValueError("reference surface partition has no eligible or side elements")
    return {
        "theta_deg": theta_deg,
        "contact_direction_reference": direction.tolist(),
        "eligible_external_element_ids": eligible_ids,
        "front_element_ids": front_ids,
        "side_element_ids": side_ids,
        "eligible_external_measure": eligible_measure,
        "front_measure": front_measure,
        "side_measure": side_measure,
        "measure_unit": "mm for 2D boundary length; mm2 for 3D surface area",
    }


def _load_2d_state(morphology: Mapping[str, Any], load_state: str) -> dict[str, Any]:
    tip = Fingertip(FingertipParameters(**morphology["parameters"]))
    full_mesh = generate_fingertip_mesh(tip.geometry, mesh_settings_for_level(localized.MESH_2D_LEVEL))
    reference_pad = full_mesh.pad
    elements, primitive_to_element = _planar_elements(full_mesh)
    if load_state == "no_load":
        current_pad = reference_pad
        source = "NO_LOAD_REFERENCE_GEOMETRY"
        mechanics_fp = "NO_LOAD_REFERENCE"
    else:
        path, record = _loaded_mechanics_record("2D", morphology, load_state)
        state_path = _resolve_path(record["state_artifact"], relative_to=path)
        if record.get("state_sha256") != sha256_file(state_path):
            raise RuntimeError(f"2D mechanics state checksum mismatch: {state_path}")
        with np.load(state_path, allow_pickle=False) as archive:
            displacement = np.asarray(archive["displacement"], dtype=float)
        if displacement.shape != reference_pad.coordinates.shape:
            raise RuntimeError(f"2D mechanics topology mismatch: {state_path}")
        current_pad = reference_pad.deformed(displacement)
        source = str(state_path)
        mechanics_fp = str(record["state_sha256"])
    geometry = build_transport_geometry(tip, current_pad, full_mesh, depth_mm=11.0)
    return {
        "tip": tip,
        "geometry": geometry,
        "elements": elements,
        "primitive_to_element": primitive_to_element,
        "reference_geometry_checksum": _reference_geometry_checksum_2d(reference_pad),
        "mechanics_source": source,
        "mechanics_fingerprint": mechanics_fp,
        "reference_mesh": full_mesh,
    }


def _reference_3d_geometry(tip: Fingertip, artifact: Any) -> Any:
    node_index = {int(node_id): index for index, node_id in enumerate(artifact.node_ids)}
    lateral = _lateral_indices(artifact)
    silicone_node_ids = np.unique(
        np.asarray(artifact.surface_faces_node_ids[lateral], dtype=np.int64).reshape(-1)
    )
    if len(silicone_node_ids) != len(artifact.silicone.vertices):
        raise ValueError("3D silicone reference vertex mapping is inconsistent")
    silicone_vertices = np.asarray(
        [artifact.undeformed_nodes_xyz[node_index[int(node_id)]] for node_id in silicone_node_ids],
        dtype=np.float32,
    )
    silicone_normals = np.asarray(artifact.surface_reference_normals[lateral], dtype=np.float32)
    silicone = TriangleSurface(
        vertices=silicone_vertices,
        faces=artifact.silicone.faces,
        normals=silicone_normals,
        external_surface=artifact.silicone.external_surface,
        u_start=artifact.silicone.u_start,
        u_end=artifact.silicone.u_end,
        semantic_tags=artifact.silicone.semantic_tags,
    )
    return build_full3d_transport_geometry(
        tip,
        silicone=silicone,
        rigid=artifact.rigid,
        envelope=artifact.envelope,
        source_position_mm=artifact.source_position_mm,
        source_medium=artifact.source_medium,
        metadata={
            **dict(artifact.metadata),
            "surface_provenance": "direct persisted undeformed native 3D reference coordinates; no FEA rerun",
            "mechanics_source": "NO_LOAD_REFERENCE_GEOMETRY",
        },
        depth_mm=11.0,
    )


def _load_3d_state(morphology: Mapping[str, Any], load_state: str) -> dict[str, Any]:
    tip = Fingertip(FingertipParameters(**morphology["parameters"]))
    path, record = _loaded_mechanics_record("3D", morphology, "left" if load_state == "no_load" else load_state)
    manifest_path = _resolve_path(record["native_manifest"], relative_to=path)
    manifest = strict_read_json(manifest_path)
    artifact = load_full3d_surface_artifact(
        manifest_path,
        expected_morphology_fingerprint=str(morphology["morphology_fingerprint"]),
        expected_contact_state_fingerprint=str(manifest["contact_state_fingerprint"]),
    )
    elements, primitive_to_element, _ = _three_d_elements(artifact)
    geometry = artifact.geometry(tip) if load_state != "no_load" else _reference_3d_geometry(tip, artifact)
    source = "NO_LOAD_REFERENCE_GEOMETRY" if load_state == "no_load" else str(manifest_path)
    mechanics_fp = "NO_LOAD_REFERENCE" if load_state == "no_load" else str(manifest.get("native_state_sha256"))
    return {
        "tip": tip,
        "geometry": geometry,
        "elements": elements,
        "primitive_to_element": primitive_to_element,
        "reference_geometry_checksum": _reference_geometry_checksum_3d(artifact),
        "mechanics_source": source,
        "mechanics_fingerprint": mechanics_fp,
        "native_artifact": artifact,
    }


def _surface_mapping_checksum(elements: Sequence[Mapping[str, Any]]) -> str:
    return _fingerprint(
        [
            {
                key: element[key]
                for key in (
                    "element_id",
                    "semantic_tag",
                    "reference_centroid_mm",
                    "reference_normal",
                    "reference_measure",
                    "eligible_external",
                )
            }
            for element in elements
        ]
    )


def _settings(mode: str, bounds: tuple[tuple[float, float], tuple[float, float]]) -> Transport3DSettings:
    return localized._optix_settings("planar" if mode == "2D" else "full3d", bounds)


def _trace_contract(
    morphology: Mapping[str, Any],
    mode: str,
    load_state: str,
    prepared: Mapping[str, Any],
    settings: Transport3DSettings,
) -> dict[str, Any]:
    optical_mode = "PLANAR_2D" if mode == "2D" else "FULL_3D"
    material = {
        "refractive_index_air": prepared["tip"].optical.refractive_index_air,
        "refractive_index_silicone": prepared["tip"].optical.refractive_index_silicone,
        "absorption_per_mm": prepared["tip"].optical.absorption_per_mm,
        "scattering_per_mm": prepared["tip"].optical.scattering_per_mm,
    }
    configuration = transport_configuration(settings, material=material)
    return {
        "schema": TRACE_SCHEMA,
        "accounting_contract_revision": 2,
        "pilot_id": PILOT_ID,
        "experiment_fingerprint": localized._manifest()["experiment_fingerprint"],
        "morphology_id": morphology["morphology_id"],
        "morphology_fingerprint": morphology["morphology_fingerprint"],
        "mechanics_dimension": mode,
        "optical_mode": optical_mode,
        "load_state": load_state,
        "mechanics_source": prepared["mechanics_source"],
        "mechanics_fingerprint": prepared["mechanics_fingerprint"],
        "reference_geometry_checksum": prepared["reference_geometry_checksum"],
        "surface_mapping_checksum": _surface_mapping_checksum(prepared["elements"]),
        "ray_count": settings.ray_count,
        "transport_configuration": configuration,
        "transport_configuration_fingerprint": fingerprint_mapping(configuration),
        "source_sampling": "shared OptiXTransport deterministic sample_directions",
    }


def _trace_path(morphology_id: str, mode: str, load_state: str) -> Path:
    return TRACE_OUTPUT / f"{morphology_id}__{mode}__{load_state}.json"


def _valid_trace(path: Path, contract: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = strict_read_json(path)
        if payload.get("contract") != dict(contract):
            return None
        if payload.get("contract_fingerprint") != fingerprint_mapping(dict(contract)):
            return None
        array_path = _resolve_path(payload["array_artifact"], relative_to=path)
        if payload.get("array_sha256") != sha256_file(array_path):
            return None
        result = payload["result"]
        accounting = payload["accounting"]
        if not accounting.get("passed"):
            return None
        if any(
            float(accounting.get(key, math.inf)) > ACCOUNTING_ATOL
            for key in (
                "event_to_material_surface_abs_error",
                "surface_to_material_surface_abs_error",
                "total_escape_reconciliation_abs_error",
            )
        ):
            return None
        with np.load(array_path, allow_pickle=False) as archive:
            required_arrays = {
                "escape_weights",
                "escape_element_ids",
                "escape_primitive_indices",
                "element_ids",
                "element_exit_weight_sum",
                "element_exit_count",
            }
            if not required_arrays.issubset(set(archive.files)):
                return None
            escape_weights = np.asarray(archive["escape_weights"], dtype=float)
            element_weights = np.asarray(archive["element_exit_weight_sum"], dtype=float)
            element_counts = np.asarray(archive["element_exit_count"], dtype=np.int64)
            if (
                not np.all(np.isfinite(escape_weights))
                or np.any(escape_weights < 0.0)
                or not np.all(np.isfinite(element_weights))
                or np.any(element_weights < 0.0)
                or np.any(element_counts < 0)
                or int(len(escape_weights)) != int(result["escaped_event_count"])
                or not math.isclose(
                    float(escape_weights.sum()),
                    float(result["outgoing_surface_weight"]),
                    rel_tol=ACCOUNTING_RTOL,
                    abs_tol=ACCOUNTING_ATOL,
                )
                or not math.isclose(
                    float(element_weights.sum()),
                    float(result["runtime_material_surface_weight"]),
                    rel_tol=ACCOUNTING_RTOL,
                    abs_tol=ACCOUNTING_ATOL,
                )
            ):
                return None
        return payload
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _aggregate_exit_events(
    elements: Sequence[Mapping[str, Any]],
    primitive_to_element: Mapping[int, int],
    primitive_indices: Sequence[int],
    event_tags: Sequence[str],
    weights: Sequence[float],
    escaped_weight: float,
    outgoing_surface_weight: float | None = None,
) -> dict[str, Any]:
    """Map integrated runtime exit events to reference surface elements."""
    primitive_array = np.asarray(primitive_indices, dtype=np.int64)
    weight_array = np.asarray(weights, dtype=float)
    tags = tuple(str(tag) for tag in event_tags)
    if primitive_array.ndim != 1 or weight_array.ndim != 1 or len(primitive_array) != len(weight_array) or len(tags) != len(weight_array):
        raise ValueError("exit-event arrays have inconsistent lengths")
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("exit-event weights must be finite and nonnegative")
    element_by_id = {int(element["element_id"]): element for element in elements}
    event_element_ids: list[int] = []
    excluded_weight = 0.0
    for primitive, weight, tag in zip(primitive_array, weight_array, tags, strict=True):
        try:
            element_id = int(primitive_to_element[int(primitive)])
            element = element_by_id[element_id]
        except (KeyError, IndexError) as exc:
            raise ValueError(f"escape event cannot map to reference surface primitive {primitive}") from exc
        if str(tag) != str(element["semantic_tag"]):
            raise ValueError("escape event semantic tag disagrees with reference surface mapping")
        if not element["eligible_external"]:
            excluded_weight += float(weight)
        event_element_ids.append(element_id)
    if excluded_weight > ACCOUNTING_ATOL:
        raise ValueError("runtime escaped transport reached an ineligible surface category")
    event_weight_sum = float(weight_array.sum())
    material_surface_weight = float(
        escaped_weight if outgoing_surface_weight is None else outgoing_surface_weight
    )
    if material_surface_weight < -ACCOUNTING_ATOL or material_surface_weight > float(escaped_weight) + ACCOUNTING_ATOL:
        raise ValueError("runtime escaped-weight accounting has an invalid surface total")
    if not math.isclose(event_weight_sum, material_surface_weight, rel_tol=ACCOUNTING_RTOL, abs_tol=ACCOUNTING_ATOL):
        raise ValueError("exit-event weight does not reconcile with runtime material-surface escaped weight")
    element_ids = np.asarray(sorted(element_by_id), dtype=np.int64)
    element_weights = np.zeros(len(element_ids), dtype=float)
    element_counts = np.zeros(len(element_ids), dtype=np.int64)
    position = {int(element_id): index for index, element_id in enumerate(element_ids)}
    for element_id, weight in zip(event_element_ids, weight_array, strict=True):
        index = position[element_id]
        element_weights[index] += float(weight)
        element_counts[index] += 1
    eligible_mask = np.asarray(
        [bool(element_by_id[int(value)]["eligible_external"]) for value in element_ids],
        dtype=bool,
    )
    phi_external = float(element_weights[eligible_mask].sum())
    if not math.isclose(phi_external, material_surface_weight, rel_tol=ACCOUNTING_RTOL, abs_tol=ACCOUNTING_ATOL):
        raise ValueError("eligible external surface exit weight does not reconcile with runtime material-surface escaped weight")
    return {
        "primitive_indices": primitive_array,
        "weights": weight_array,
        "event_element_ids": np.asarray(event_element_ids, dtype=np.int64),
        "excluded_weight": excluded_weight,
        "event_weight_sum": event_weight_sum,
        "runtime_escaped_weight": float(escaped_weight),
        "material_surface_escaped_weight": material_surface_weight,
        "excluded_escape_weight": float(escaped_weight) - material_surface_weight,
        "excluded_escape_categories": {
            "virtual_envelope": float(escaped_weight) - material_surface_weight,
        },
        "element_ids": element_ids,
        "element_weights": element_weights,
        "element_counts": element_counts,
        "phi_external": phi_external,
    }


def _run_trace(
    morphology: Mapping[str, Any],
    mode: str,
    load_state: str,
    prepared: Mapping[str, Any],
    settings: Transport3DSettings,
    runtime: Any,
) -> dict[str, Any]:
    contract = _trace_contract(morphology, mode, load_state, prepared, settings)
    path = _trace_path(str(morphology["morphology_id"]), mode, load_state)
    existing = _valid_trace(path, contract)
    if existing is not None:
        return existing
    raw = trace_geometry(prepared["tip"], prepared["geometry"], settings=settings, runtime=runtime)
    aggregation = _aggregate_exit_events(
        prepared["elements"],
        prepared["primitive_to_element"],
        raw.escape_surface_primitive_indices,
        raw.escape_surface_tags,
        raw.escape_weights,
        raw.escaped_weight,
        raw.outgoing_surface_weight,
    )
    primitive_indices = aggregation["primitive_indices"]
    weights = aggregation["weights"]
    event_element_ids = aggregation["event_element_ids"]
    excluded_weight = aggregation["excluded_weight"]
    event_weight_sum = aggregation["event_weight_sum"]
    element_ids = aggregation["element_ids"]
    element_weights = aggregation["element_weights"]
    element_counts = aggregation["element_counts"]
    phi_external = aggregation["phi_external"]
    element_by_id = {int(element["element_id"]): element for element in prepared["elements"]}
    unified = UnifiedTransportResult.from_transport_result(
        raw,
        morphology_id=str(morphology["morphology_id"]),
        morphology_fingerprint=str(morphology["morphology_fingerprint"]),
        mechanics_source=str(prepared["mechanics_source"]),
        mechanics_dimension=mode,
        contact_state={"pilot_load_state": load_state, "localized_load_only": True},
        transport_configuration_fingerprint=contract["transport_configuration_fingerprint"],
    )
    array_path = path.with_suffix(".npz")
    arrays = {
        "field": np.asarray(unified.field),
        **{f"axis_{index}": axis for index, axis in enumerate(unified.field_axes)},
        "escape_weights": weights,
        "escape_element_ids": np.asarray(event_element_ids, dtype=np.int64),
        "escape_primitive_indices": primitive_indices,
        "element_ids": element_ids,
        "element_exit_weight_sum": element_weights,
        "element_exit_count": element_counts,
    }
    _atomic_npz(array_path, **arrays)
    elements = []
    for element_id, weight, count in zip(element_ids, element_weights, element_counts, strict=True):
        element = dict(element_by_id[int(element_id)])
        element["exit_weight_sum"] = float(weight)
        element["exit_count"] = int(count)
        elements.append(element)
    payload = {
        "schema": TRACE_SCHEMA,
        "contract": contract,
        "contract_fingerprint": fingerprint_mapping(contract),
        "array_artifact": str(array_path),
        "array_sha256": sha256_file(array_path),
        "result": {
            "mode": mode,
            "mechanics_dimension": mode,
            "optical_mode": unified.optical_mode,
            "load_state": load_state,
            "launched_weight": float(raw.launched_weight),
            "escaped_weight": float(raw.escaped_weight),
            "runtime_material_surface_weight": float(raw.outgoing_surface_weight),
            "excluded_escape_weight": float(aggregation["excluded_escape_weight"]),
            "absorbed_weight": float(raw.absorbed_weight),
            "terminated_weight": float(raw.terminated_weight),
            "outgoing_surface_weight": float(raw.outgoing_surface_weight),
            "phi_external": phi_external,
            "escaped_event_count": int(len(weights)),
            "excluded_ineligible_event_weight": float(excluded_weight),
            "energy_balance_error": float(raw.energy_balance_error),
            "ray_count": int(raw.launched_ray_count),
            "valid_ray_count": int(len(weights)),
            "reference_geometry_checksum": contract["reference_geometry_checksum"],
            "surface_mapping_checksum": contract["surface_mapping_checksum"],
        },
        "reference_surface_elements": elements,
        "accounting": {
            "event_weight_sum": event_weight_sum,
            "eligible_external_element_weight_sum": phi_external,
            "runtime_escaped_weight": float(raw.escaped_weight),
            "runtime_material_surface_weight": float(raw.outgoing_surface_weight),
            "excluded_escape_weight": float(aggregation["excluded_escape_weight"]),
            "excluded_escape_categories": aggregation["excluded_escape_categories"],
            "event_to_material_surface_abs_error": abs(event_weight_sum - float(raw.outgoing_surface_weight)),
            "surface_to_material_surface_abs_error": abs(phi_external - float(raw.outgoing_surface_weight)),
            "total_escape_reconciliation_abs_error": abs(
                float(raw.escaped_weight)
                - float(raw.outgoing_surface_weight)
                - float(aggregation["excluded_escape_weight"])
            ),
            "passed": True,
            "weight_semantics": "exit-event weights are already integrated transported weights; no geometric measure multiplier",
        },
        "created_at": _now(),
    }
    atomic_write_json(path, _jsonable(payload))
    return payload


def _metrics(payload: Mapping[str, Any], theta: float, partition: Mapping[str, Any], *, baseline: Mapping[str, Any] | None) -> dict[str, Any]:
    elements = payload["reference_surface_elements"]
    weights = {int(element["element_id"]): float(element["exit_weight_sum"]) for element in elements}
    side_ids = set(int(value) for value in partition["side_element_ids"])
    eligible_ids = set(int(value) for value in partition["eligible_external_element_ids"])
    phi_side = float(sum(weights.get(element_id, 0.0) for element_id in side_ids))
    phi_external = float(sum(weights.get(element_id, 0.0) for element_id in eligible_ids))
    launched = float(payload["result"]["launched_weight"])
    if launched <= 0.0:
        raise ValueError("launched weight must be positive")
    eta = phi_side / launched
    f_side = phi_side / phi_external if phi_external > 0.0 else None
    row = {
        "eta_side": eta,
        "f_side": f_side,
        "Phi_side": phi_side,
        "Phi_external": phi_external,
        "E_launched": launched,
        "launched_weight": launched,
        "escaped_weight": float(payload["result"]["escaped_weight"]),
        "runtime_material_surface_weight": float(
            payload["result"].get("runtime_material_surface_weight", payload["result"]["escaped_weight"])
        ),
        "theta_deg": theta,
        "theta": theta,
        "contact_direction_reference": list(partition["contact_direction_reference"]),
        "eligible_external_element_ids": list(partition["eligible_external_element_ids"]),
        "front_element_ids": list(partition["front_element_ids"]),
        "side_element_ids": list(partition["side_element_ids"]),
        "reference_side_measure": partition["side_measure"],
        "reference_front_measure": partition["front_measure"],
        "reference_eligible_external_measure": partition["eligible_external_measure"],
        "side_region_definition": f"eligible external reference/material surface; reference normal partition theta={theta:g} deg",
        "provenance_status": "PASS",
    }
    if baseline is None:
        row.update({
            "eta_side_no_load": eta,
            "f_side_no_load": f_side,
            "Phi_side_no_load": phi_side,
            "Delta_eta_side": 0.0,
            "Delta_f_side": 0.0,
            "G_side": 0.0,
            "low_baseline_flag": phi_side <= RELATIVE_GAIN_MIN_BASELINE,
        })
        return row
    baseline_eta = float(baseline["eta_side"])
    baseline_f = baseline["f_side"]
    baseline_phi = float(baseline["Phi_side"])
    row.update({
        "eta_side_no_load": baseline_eta,
        "f_side_no_load": baseline_f,
        "Phi_side_no_load": baseline_phi,
        "Delta_eta_side": eta - baseline_eta,
        "Delta_f_side": None if f_side is None or baseline_f is None else f_side - float(baseline_f),
        "G_side": None if baseline_phi <= RELATIVE_GAIN_MIN_BASELINE else (phi_side - baseline_phi) / baseline_phi,
        "low_baseline_flag": baseline_phi <= RELATIVE_GAIN_MIN_BASELINE,
    })
    return row


def _state_table(
    trace_payloads: Mapping[tuple[str, str, str], Mapping[str, Any]],
    contact_directions: Mapping[tuple[str, str, str], np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for theta in THETAS_DEG:
        partitions: dict[tuple[str, str, str], dict[str, Any]] = {}
        for morphology in _case_rows():
            morphology_id = str(morphology["morphology_id"])
            for mode in ("2D", "3D"):
                elements = trace_payloads[(morphology_id, mode, "no_load")]["reference_surface_elements"]
                for contact_side in ("left", "right"):
                    partitions[(morphology_id, mode, contact_side)] = _partition(
                        elements,
                        contact_directions[(morphology_id, mode, contact_side)],
                        theta,
                    )
        for morphology in _case_rows():
            morphology_id = str(morphology["morphology_id"])
            for mode in ("2D", "3D"):
                for contact_side in ("left", "right"):
                    partition = partitions[(morphology_id, mode, contact_side)]
                    baseline = _metrics(
                        trace_payloads[(morphology_id, mode, "no_load")],
                        theta,
                        partition,
                        baseline=None,
                    )
                    for load_state in ("no_load", contact_side):
                        metric = baseline if load_state == "no_load" else _metrics(
                            trace_payloads[(morphology_id, mode, load_state)],
                            theta,
                            partition,
                            baseline=baseline,
                        )
                        rows.append({
                            "morphology": morphology_id,
                            "base_id": morphology["base_id"],
                            "arm": morphology["arm"],
                            "morphology_fingerprint": morphology["morphology_fingerprint"],
                            "dimension": mode,
                            "optical_mode": "PLANAR_2D" if mode == "2D" else "FULL_3D",
                            "load_state": load_state,
                            "contact_side": contact_side,
                            **metric,
                        })
    return rows


def _find_row(
    table: Sequence[Mapping[str, Any]],
    morphology: str,
    mode: str,
    load_state: str,
    contact_side: str,
    theta: float = 45.0,
) -> Mapping[str, Any]:
    return next(
        row
        for row in table
        if row["morphology"] == morphology
        and row["dimension"] == mode
        and row["load_state"] == load_state
        and row["contact_side"] == contact_side
        and row["theta_deg"] == theta
    )


def _sign(value: float | None) -> int:
    if value is None or not math.isfinite(float(value)) or abs(float(value)) <= 1.0e-12:
        return 0
    return 1 if float(value) > 0.0 else -1


def _dimension_results(table: Sequence[Mapping[str, Any]], mode: str, theta: float = 45.0) -> list[dict[str, Any]]:
    results = []
    for morphology in _case_rows():
        morphology_id = str(morphology["morphology_id"])
        no_load_left = _find_row(table, morphology_id, mode, "no_load", "left", theta)
        no_load_right = _find_row(table, morphology_id, mode, "no_load", "right", theta)
        left = _find_row(table, morphology_id, mode, "left", "left", theta)
        right = _find_row(table, morphology_id, mode, "right", "right", theta)
        results.append({
            "morphology": morphology_id,
            "morphology_fingerprint": morphology["morphology_fingerprint"],
            "no_load": {
                "left": {key: no_load_left[key] for key in ("eta_side", "f_side", "Phi_side", "Phi_external")},
                "right": {key: no_load_right[key] for key in ("eta_side", "f_side", "Phi_side", "Phi_external")},
            },
            "left": {key: left[key] for key in ("eta_side", "f_side", "Phi_side", "Phi_external", "Delta_eta_side", "Delta_f_side", "G_side")},
            "right": {key: right[key] for key in ("eta_side", "f_side", "Phi_side", "Phi_external", "Delta_eta_side", "Delta_f_side", "G_side")},
            "Delta_eta_mean": 0.5 * (float(left["Delta_eta_side"]) + float(right["Delta_eta_side"])),
            "Delta_eta_asymmetry": abs(float(left["Delta_eta_side"]) - float(right["Delta_eta_side"])),
            "Delta_f_mean": None if left["Delta_f_side"] is None or right["Delta_f_side"] is None else 0.5 * (float(left["Delta_f_side"]) + float(right["Delta_f_side"])),
            "Delta_f_asymmetry": None if left["Delta_f_side"] is None or right["Delta_f_side"] is None else abs(float(left["Delta_f_side"]) - float(right["Delta_f_side"])),
        })
    return results


def _rank_order(results: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    return [str(row["morphology"]) for row in sorted(results, key=lambda row: (-float(row[key]), str(row["morphology"]))) ]


def _correlation(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    if len(x) < 3:
        return {"n": len(x), "spearman_rho": None, "kendall_tau": None, "status": "INSUFFICIENT_N"}
    from scipy.stats import kendalltau, spearmanr
    return {
        "n": len(x),
        "spearman_rho": float(spearmanr(x, y).statistic),
        "kendall_tau": float(kendalltau(x, y).statistic),
        "status": "DESCRIPTIVE_ONLY_N3_PILOT",
    }


def _comparison(table: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    full = _dimension_results(table, "3D")
    planar = _dimension_results(table, "2D")
    by_morph_3d = {row["morphology"]: row for row in full}
    by_morph_2d = {row["morphology"]: row for row in planar}
    per_morph = []
    for morphology in _case_rows():
        key = morphology["morphology_id"]
        a, b = by_morph_2d[key], by_morph_3d[key]
        per_morph.append({
            "morphology": key,
            "left": {
                "Delta_eta_2D": a["left"]["Delta_eta_side"],
                "Delta_eta_3D": b["left"]["Delta_eta_side"],
                "sign_agreement": _sign(a["left"]["Delta_eta_side"]) == _sign(b["left"]["Delta_eta_side"]),
            },
            "right": {
                "Delta_eta_2D": a["right"]["Delta_eta_side"],
                "Delta_eta_3D": b["right"]["Delta_eta_side"],
                "sign_agreement": _sign(a["right"]["Delta_eta_side"]) == _sign(b["right"]["Delta_eta_side"]),
            },
            "mean": {
                "Delta_eta_2D": a["Delta_eta_mean"],
                "Delta_eta_3D": b["Delta_eta_mean"],
                "sign_agreement": _sign(a["Delta_eta_mean"]) == _sign(b["Delta_eta_mean"]),
            },
        })
    eta2 = [float(row["Delta_eta_mean"]) for row in planar]
    eta3 = [float(row["Delta_eta_mean"]) for row in full]
    threshold = {}
    for theta in THETAS_DEG:
        p = _dimension_results(table, "2D", theta)
        f = _dimension_results(table, "3D", theta)
        p_order = _rank_order(p, "Delta_eta_mean")
        f_order = _rank_order(f, "Delta_eta_mean")
        sign_agreement = all(_sign(a["Delta_eta_mean"]) == _sign(b["Delta_eta_mean"]) for a, b in zip(p, f, strict=True))
        threshold[str(int(theta))] = {
            "2D_order": p_order,
            "3D_order": f_order,
            "same_morphology_order": p_order == f_order,
            "all_signs_agree": sign_agreement,
            "2D_results": p,
            "3D_results": f,
        }
    return {
        "theta_nominal_deg": 45.0,
        "FULL_3D": full,
        "PLANAR_2D": planar,
        "per_morphology": per_morph,
        "morphology_order_2D": _rank_order(planar, "Delta_eta_mean"),
        "morphology_order_3D": _rank_order(full, "Delta_eta_mean"),
        "mean_response_correlation": _correlation(eta2, eta3),
        "threshold_sensitivity": threshold,
    }


def _classifications(comparison: Mapping[str, Any]) -> tuple[str, str]:
    full = comparison["FULL_3D"]
    deltas = [float(row["Delta_eta_mean"]) for row in full]
    signs = [_sign(value) for value in deltas]
    nominal_full_order = comparison["morphology_order_3D"]
    nominal_full_signs = signs
    region_sensitive = any(
        _rank_order(comparison["threshold_sensitivity"][str(int(theta))]["3D_results"], "Delta_eta_mean") != nominal_full_order
        or [
            _sign(row["Delta_eta_mean"])
            for row in comparison["threshold_sensitivity"][str(int(theta))]["3D_results"]
        ] != nominal_full_signs
        for theta in (30.0, 60.0)
    )
    if region_sensitive:
        physical = "REGION_DEFINITION_SENSITIVE"
    elif not deltas:
        physical = "INSUFFICIENT_TRANSPORT_CONVERGENCE"
    elif sum(sign > 0 for sign in signs) >= 2 and sum(sign < 0 for sign in signs) == 0:
        physical = "SIDE_REDIRECTION_PRESENT" if max(deltas) - min(deltas) <= 1.0e-6 else "MORPHOLOGY_DEPENDENT_REDIRECTION"
    elif sum(sign > 0 for sign in signs) >= 1 and sum(sign < 0 for sign in signs) >= 1:
        physical = "MORPHOLOGY_DEPENDENT_REDIRECTION"
    else:
        physical = "NO_CLEAR_SIDE_REDIRECTION"
    nominal = comparison["threshold_sensitivity"]["45"]
    changed = any(
        comparison["threshold_sensitivity"][str(int(theta))]["2D_order"] != nominal["2D_order"]
        or comparison["threshold_sensitivity"][str(int(theta))]["3D_order"] != nominal["3D_order"]
        or comparison["threshold_sensitivity"][str(int(theta))]["all_signs_agree"] != nominal["all_signs_agree"]
        for theta in (30.0, 60.0)
    )
    if changed:
        relation = "REGION_DEPENDENT_2D3D_RELATION"
    elif comparison["morphology_order_2D"] == comparison["morphology_order_3D"] and all(item["mean"]["sign_agreement"] for item in comparison["per_morphology"]):
        relation = "SIDE_RESPONSE_TREND_PRESERVED"
    elif all(item["mean"]["sign_agreement"] for item in comparison["per_morphology"]):
        relation = "MECHANISM_PRESERVED_RANKING_NOT_PRESERVED"
    elif any(not item["mean"]["sign_agreement"] for item in comparison["per_morphology"]):
        relation = "SIDE_RESPONSE_DIMENSION_SENSITIVE"
    else:
        relation = "INCONCLUSIVE_2D3D_PILOT"
    return physical, relation


def _write_plots(comparison: Mapping[str, Any], output: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"plots skipped: {type(exc).__name__}: {exc}"]
    names = [row["morphology"] for row in comparison["FULL_3D"]]
    created = []
    for mode, title, filename in (("FULL_3D", "FULL_3D side response", "full3d_eta_side.png"), ("PLANAR_2D", "PLANAR_2D side response", "planar2d_eta_side.png")):
        rows = comparison[mode]
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(8, 4))
        for state, label in (("no_load", "no load"), ("left", "left"), ("right", "right")):
            if state == "no_load":
                values = [
                    0.5 * (row["no_load"]["left"]["eta_side"] + row["no_load"]["right"]["eta_side"])
                    for row in rows
                ]
            else:
                values = [row[state]["eta_side"] for row in rows]
            ax.plot(x, values, "o-", label=label)
        ax.set_xticks(x, names, rotation=25, ha="right")
        ax.set_ylabel("eta_side")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        path = output / filename
        fig.savefig(path, dpi=120)
        plt.close(fig)
        created.append(str(path))
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    width = 0.18
    for offset, (mode, color) in enumerate((("PLANAR_2D", "tab:blue"), ("FULL_3D", "tab:orange"))):
        rows = comparison[mode]
        ax.bar(x + (offset * 2 - 1.5) * width, [row["left"]["Delta_eta_side"] for row in rows], width, color=color, alpha=0.75, label=f"{mode} left")
        ax.bar(x + (offset * 2 - 0.5) * width, [row["right"]["Delta_eta_side"] for row in rows], width, color=color, alpha=0.35, label=f"{mode} right")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel("Delta eta_side")
    ax.set_title("Loaded side response by dimension")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output / "delta_eta_side_2d_vs_3d.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    created.append(str(path))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([row["Delta_eta_mean"] for row in comparison["PLANAR_2D"]], [row["Delta_eta_mean"] for row in comparison["FULL_3D"]])
    for row in comparison["FULL_3D"]:
        ax.annotate(row["morphology"], (comparison["PLANAR_2D"][names.index(row["morphology"])] ["Delta_eta_mean"], row["Delta_eta_mean"]), fontsize=7)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("PLANAR_2D Delta eta_mean")
    ax.set_ylabel("FULL_3D Delta eta_mean")
    ax.set_title("Mean side response: native 2D vs native 3D")
    fig.tight_layout()
    path = output / "delta_eta_mean_2d_vs_3d.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    created.append(str(path))
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode, style in (("PLANAR_2D", "o-"), ("FULL_3D", "s--")):
        for row_index, row in enumerate(comparison["threshold_sensitivity"]["45"][f"{mode.replace('PLANAR_2D', '2D').replace('FULL_3D', '3D')}_results"]):
            values = [comparison["threshold_sensitivity"][str(int(theta))][f"{mode.replace('PLANAR_2D', '2D').replace('FULL_3D', '3D')}_results"][row_index]["Delta_eta_mean"] for theta in THETAS_DEG]
            ax.plot(THETAS_DEG, values, style, label=f"{mode} {row['morphology']}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("front/side threshold (deg)")
    ax.set_ylabel("Delta eta_mean")
    ax.set_title("Reference-region threshold sensitivity")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = output / "theta_region_sensitivity.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    created.append(str(path))
    return created


def run_pilot(output: Path = OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    global TRACE_OUTPUT
    TRACE_OUTPUT = output / "traces"
    morphologies = _case_rows()
    manifest = localized._manifest()
    pilot_manifest = {
        "schema": PILOT_SCHEMA,
        "accounting_contract_revision": 2,
        "pilot_id": PILOT_ID,
        "experiment_fingerprint": manifest["experiment_fingerprint"],
        "production_force_n": localized._production_force(),
        "planned_unique_physical_morphologies": 3,
        "planned_optical_traces": 18,
        "morphologies": [
            {
                "morphology_id": row["morphology_id"],
                "base_id": row["base_id"],
                "arm": row["arm"],
                "morphology_fingerprint": row["morphology_fingerprint"],
                "parameters": row["parameters"],
            }
            for row in morphologies
        ],
        "trace_plan": [
            {"morphology_id": row["morphology_id"], "dimension": mode, "load_state": load_state}
            for row in morphologies
            for mode in ("2D", "3D")
            for load_state in ("no_load", "left", "right")
        ],
        "fea_rerun": False,
        "existing_production_J_modified": False,
        "created_at": _now(),
    }
    atomic_write_json(output / "pilot_manifest.json", _jsonable(pilot_manifest))
    contract_description = {
        "schema": TRACE_SCHEMA,
        "accounting_contract_revision": 2,
        "event_weight_semantics": "Each material-surface escape weight is already an integrated transported weight; aggregation is sum(weight), never weight times edge length or triangle area.",
        "escaped_weight_accounting": "Transport escaped_weight may also include virtual-envelope exits. The artifact records runtime_material_surface_weight and explicitly reconciles the excluded virtual_envelope remainder.",
        "PLANAR_2D": "reference boundary edge element IDs, semantic tags, reference centroid/normal, exit weight sum and count",
        "FULL_3D": "reference lateral surface triangle element IDs, semantic tags, reference centroid/normal, exit weight sum and count",
        "provenance": "experiment, morphology, mechanics, reference geometry, surface mapping, and transport fingerprints are stored per trace",
    }
    atomic_write_json(output / "surface_exit_artifact_contract.json", _jsonable(contract_description))
    region_definition = {
        "eligible_external_2d_tags": sorted(EXTERNAL_2D_TAGS),
        "eligible_external_3d_tag_prefix": EXTERNAL_3D_TAG_PREFIX,
        "always_excluded": ["bonded pad/link interface", "void/cavity boundary", "longitudinal caps", "internal surfaces", "rigid/non-silicone geometry"],
        "contact_direction": "reference outward direction opposite the exact inward normal resultant of the existing localized load footprint; 2D uses line-length/profile weighting and 3D uses triangle-area/profile weighting",
        "thresholds_deg": list(THETAS_DEG),
        "nominal_threshold_deg": 45.0,
        "normal_source": "reference/material normals only; deformed positions and normals are never used",
        "measure_units": {"2D": "boundary length mm", "3D": "surface area mm2"},
    }
    atomic_write_json(output / "surface_region_definition.json", _jsonable(region_definition))

    prepared: dict[tuple[str, str, str], dict[str, Any]] = {}
    contact_directions: dict[tuple[str, str, str], np.ndarray] = {}
    settings_by_morphology_mode: dict[tuple[str, str], Transport3DSettings] = {}
    for morphology in morphologies:
        morphology_id = str(morphology["morphology_id"])
        states: dict[tuple[str, str], dict[str, Any]] = {}
        for mode in ("2D", "3D"):
            for load_state in ("no_load", "left", "right"):
                state = _load_2d_state(morphology, load_state) if mode == "2D" else _load_3d_state(morphology, load_state)
                states[(mode, load_state)] = state
            for contact_side in ("left", "right"):
                if mode == "2D":
                    contact_directions[(morphology_id, mode, contact_side)] = _contact_direction_2d(
                        states[(mode, "no_load")]["reference_mesh"], contact_side
                    )
                else:
                    contact_directions[(morphology_id, mode, contact_side)] = _contact_direction_3d(
                        states[(mode, "no_load")]["elements"], contact_side
                    )
            all_vertices = []
            for load_state in ("no_load", "left", "right"):
                geometry = states[(mode, load_state)]["geometry"]
                for surface in (geometry.silicone, geometry.rigid, geometry.envelope):
                    all_vertices.append(np.asarray(surface.vertices, dtype=float))
            vertices = np.vstack(all_vertices)
            span = max(float(np.ptp(vertices[:, 0])), float(np.ptp(vertices[:, 1])))
            margin = 0.04 * span
            bounds = ((float(vertices[:, 0].min() - margin), float(vertices[:, 0].max() + margin)), (float(vertices[:, 1].min() - margin), float(vertices[:, 1].max() + margin)))
            settings_by_morphology_mode[(morphology_id, mode)] = _settings(mode, bounds)
            for load_state in ("no_load", "left", "right"):
                prepared[(morphology_id, mode, load_state)] = states[(mode, load_state)]

    runtime = create_runtime()
    trace_payloads: dict[tuple[str, str, str], dict[str, Any]] = {}
    for morphology in morphologies:
        morphology_id = str(morphology["morphology_id"])
        for mode in ("2D", "3D"):
            settings = settings_by_morphology_mode[(morphology_id, mode)]
            for load_state in ("no_load", "left", "right"):
                trace_payloads[(morphology_id, mode, load_state)] = _run_trace(
                    morphology,
                    mode,
                    load_state,
                    prepared[(morphology_id, mode, load_state)],
                    settings,
                    runtime,
                )

    table = _state_table(trace_payloads, contact_directions)
    atomic_write_json(output / "side_exit_state_table.json", _jsonable(table))
    comparison = _comparison(table)
    atomic_write_json(output / "side_exit_2d3d_comparison.json", _jsonable(comparison))
    physical_classification, relation_classification = _classifications(comparison)
    plots = _write_plots(comparison, output)
    summary = {
        "schema": "side-exit-flux-summary-v1",
        "pilot_id": PILOT_ID,
        "experiment_fingerprint": manifest["experiment_fingerprint"],
        "label": "PREDEFINED 3-MORPHOLOGY / 18-TRACE PILOT — NOT FULL DESIGN-SPACE STATISTICS",
        "planned_trace_count": 18,
        "actual_trace_count": len(trace_payloads),
        "physical_side_redirection_classification": physical_classification,
        "2d_to_3d_relation_classification": relation_classification,
        "FULL_3D_first": comparison["FULL_3D"],
        "PLANAR_2D_second": comparison["PLANAR_2D"],
        "comparison": comparison,
        "plots": plots,
        "fea_rerun": False,
        "production_J_changed": False,
        "created_at": _now(),
    }
    atomic_write_json(output / "side_exit_summary.json", _jsonable(summary))
    method_notes = f"""# Side-exit 2D/3D pilot

This bounded pilot used exactly three unique physical morphologies and 18 optical traces. Existing localized-load FEA artifacts were reused; FEA was not rerun.

Each material-surface escape event carries an already integrated transported weight. `Phi_side` and `Phi_external` are sums of event weights assigned to reference/material surface elements. No edge-length or triangle-area multiplier is applied. The transport core's total `escaped_weight` can additionally contain virtual-envelope exits; those are explicitly separated from `runtime_material_surface_weight` in each trace accounting record.

External eligible surfaces are `pad_outer_left`, `pad_outer_arc`, and `pad_outer_right` in 2D, and `outer_compliant_*` in 3D. Bond, void, cap, internal, and rigid surfaces are excluded. Front/side is based on reference outward normals and thresholds 30/45/60 degrees. The contact-facing direction is the outward opposite of the exact profile-weighted inward normal resultant used by the localized-load implementation.

The 45-degree FULL_3D result is reported before PLANAR_2D. The pilot has n=3 physical morphologies and is descriptive only; it is not full design-space evidence and does not replace J.
"""
    (output / "method_notes.md").write_text(method_notes, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    summary = run_pilot(args.output)
    print(json.dumps({
        "status": "PASS",
        "pilot_id": summary["pilot_id"],
        "actual_trace_count": summary["actual_trace_count"],
        "physical_side_redirection_classification": summary["physical_side_redirection_classification"],
        "2d_to_3d_relation_classification": summary["2d_to_3d_relation_classification"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
