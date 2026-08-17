"""Force-controlled localized-load 2D/3D mechanics and OptiX trend study.

This validation owns a new load contract.  It reuses the authoritative
24-pair morphology manifest but never reuses explicit-contact mechanics
artifacts as localized-load results.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr

from fem.solid3d import SolidFEASettings, solve_solid_3d
from mesh.fingertip import generate_fingertip_mesh
from mesh.types import mesh_settings_for_level
from mesh.volume3d import generate_volume_mesh
from mesh.volume_types import volume_mesh_settings_for_tier
from model import Fingertip, FingertipParameters, build_fingertip_solid
from optics.transport3d import (
    OptiXTransport,
    UnifiedTransportResult,
    fingerprint_mapping,
    load_full3d_surface_artifact,
    native_field_separability,
    save_case_artifact,
    trace_geometry,
    transport_configuration,
)
from optics.transport3d.geometry import build_transport_geometry
from optics.transport3d.settings import Transport3DSettings
from validation.common.io import atomic_write_json, strict_read_json
from validation.common.provenance import sha256_file
from validation import localized_load_trend as localized
from validation import overnight_24_pair_trend as legacy_optix
from validation.three_d_migration import (
    M4_REFERENCE_MESH_SETTINGS,
    _external_surface_u,
    _native_array_fingerprint,
    _orient_surface_faces,
)


OUTPUT = Path("output/validation/overnight_force_localized_trend")
INTERIM_OUTPUT = OUTPUT / "interim_optix"
MECHANISTIC_BRIDGE_OUTPUT = INTERIM_OUTPUT / "mechanistic_bridge"
PARENT_MANIFEST = Path("output/validation/overnight_24_pair_trend/experiment_manifest.json")
SCHEMA = "overnight-force-localized-trend-v1"
EXPERIMENT_ID = "overnight_force_localized_2d3d_v1"
FORCE_TARGETS_N = (5.0, 10.0, 15.0)
LOWER_CALIBRATION_FORCE_TARGETS_N = (1.0, 2.0, 3.0, 4.0)
RADIUS_MM = 4.0
STEPS = 12
SIDES = {"left": -3.0, "right": 3.0}
MESH_2D_LEVEL = "medium"
MESH_3D_TIER = "search"
RAY_COUNT = 1024
GRID_WIDTH = 48
GRID_HEIGHT = 48
GRID_Z_BINS = 16
PAIR_TIE_TOLERANCE = 1.0e-12


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
    return hashlib.sha256(
        json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _parent_manifest() -> dict[str, Any]:
    payload = strict_read_json(PARENT_MANIFEST)
    if payload.get("schema") != "overnight-24-pair-trend-v1":
        raise RuntimeError("authoritative parent sampling manifest is invalid")
    if len(payload.get("pairs", [])) != 24:
        raise RuntimeError("authoritative parent manifest must contain 24 pairs")
    return payload


def _morphologies() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for pair in _parent_manifest()["pairs"]:
        for arm in ("FIXED", "VARIED"):
            row = dict(pair["arms"][arm])
            row.update({
                "base_id": pair["base_id"],
                "arm": arm,
                "morphology_id": f"{pair['base_id']}__{arm}",
                "anchor": pair.get("anchor"),
            })
            result[row["morphology_id"]] = row
    return result


def _cases() -> list[dict[str, Any]]:
    rows = _morphologies()
    cases = []
    for pair in _parent_manifest()["pairs"]:
        for arm in ("FIXED", "VARIED"):
            morphology = rows[f"{pair['base_id']}__{arm}"]
            for side, x_mm in SIDES.items():
                cases.append({
                    "case_id": f"{morphology['morphology_id']}__{side}",
                    "base_id": pair["base_id"],
                    "arm": arm,
                    "side": side,
                    "center_x_mm": x_mm,
                    "parameters": morphology["parameters"],
                    "morphology_fingerprint": morphology["morphology_fingerprint"],
                    "normalized_coordinate": morphology.get("normalized_coordinate"),
                })
    return cases


def _mesh_contract() -> dict[str, Any]:
    return {
        "2D": {"level": MESH_2D_LEVEL, "settings": asdict(mesh_settings_for_level(MESH_2D_LEVEL))},
        "3D": {"tier": MESH_3D_TIER, "settings": asdict(volume_mesh_settings_for_tier(MESH_3D_TIER))},
    }


def _base_manifest() -> dict[str, Any]:
    parent = _parent_manifest()
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "parent_sampling_fingerprint": parent["precommit_fingerprint"],
        "parent_sampling_manifest": str(PARENT_MANIFEST),
        "base_design_count": 24,
        "arms": ["FIXED", "VARIED"],
        "sides": dict(SIDES),
        "force_targets_n": list(FORCE_TARGETS_N),
        "production_force_n": None,
        "footprint": {
            "radius_mm": RADIUS_MM,
            "profile": "compact_cosine_radial",
            "normalization": "discrete_3d_resultant_force",
            "2D_center_x_mm": [-3.0, 3.0],
            "3D_center_xz_mm": [[-3.0, 0.0], [3.0, 0.0]],
        },
        "load_matching": {
            "3D": "target total resultant force in N, normalized on each discrete loaded surface",
            "2D": "same case pressure amplitude applied to unit-depth line analogue, resultant reported in N/mm",
            "absolute_2d_3d_force_equality": False,
        },
        "mechanics": {
            "material": {"young_modulus_mpa": 0.55, "poisson_ratio": 0.49, "law": "current_hyperelastic"},
            "external_contact": False,
            "steps": STEPS,
            "mesh": _mesh_contract(),
        },
        "optix": {
            "planar_mode": "PLANAR_2D",
            "full_mode": "FULL_3D",
            "extrusion_depth_mm": 11.0,
            "ray_count": RAY_COUNT,
            "native_p3_retains_z": True,
        },
        "created_at": _now(),
    }


def _manifest() -> dict[str, Any]:
    path = OUTPUT / "experiment_manifest.json"
    if not path.exists():
        payload = _base_manifest()
        payload["experiment_fingerprint"] = _fingerprint(payload)
        atomic_write_json(path, payload)
        return payload
    payload = strict_read_json(path)
    fingerprint = payload.get("experiment_fingerprint")
    expected = dict(payload)
    expected.pop("experiment_fingerprint", None)
    if fingerprint != _fingerprint(expected):
        raise RuntimeError("force-localized experiment manifest fingerprint mismatch")
    if payload.get("parent_sampling_fingerprint") != _parent_manifest()["precommit_fingerprint"]:
        raise RuntimeError("force-localized experiment does not use the authoritative sampling manifest")
    return payload


def _case_contract(case: Mapping[str, Any], stage: str, *, force_n: float | None = None, pressure_mpa: float | None = None) -> dict[str, Any]:
    manifest = _manifest()
    return {
        "schema": "force-localized-case-contract-v1",
        "experiment_id": EXPERIMENT_ID,
        "experiment_fingerprint": manifest["experiment_fingerprint"],
        "stage": stage,
        "case_id": case["case_id"],
        "base_id": case["base_id"],
        "arm": case["arm"],
        "side": case["side"],
        "center_x_mm": case["center_x_mm"],
        "parameters": case["parameters"],
        "morphology_fingerprint": case["morphology_fingerprint"],
        "force_target_n": force_n,
        "pressure_mpa": pressure_mpa,
        "footprint_radius_mm": RADIUS_MM,
        "steps": STEPS,
        "external_contact": False,
        "mesh": _mesh_contract(),
    }


def _case_path(stage: str, case_id: str) -> Path:
    return OUTPUT / stage / f"{case_id}.json"


def _force_profile_3d(mesh: Any, side: str, target_force_n: float) -> dict[str, Any]:
    center_x = SIDES[side]
    rows: list[dict[str, Any]] = []
    coefficient = np.zeros(3, dtype=float)
    loaded_area = 0.0
    for tag, triangles in mesh.surface_triangles.items():
        definition = next((item for item in mesh.solid.surfaces if item.name == tag), None)
        if definition is None or definition.kind != "outer_compliant" or definition.material_region != "pad":
            continue
        for triangle in triangles:
            points = np.asarray([[mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm, mesh.nodes[node_id].z_mm] for node_id in triangle.node_ids], dtype=float)
            centroid = points.mean(axis=0)
            distance = math.hypot(float(centroid[0]) - center_x, float(centroid[2]))
            profile = localized.localized_profile(distance, RADIUS_MM)
            if profile <= 0.0:
                continue
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            norm = float(np.linalg.norm(normal))
            if not math.isfinite(norm) or norm <= 1.0e-12:
                raise RuntimeError(f"invalid localized-load triangle {triangle.id}")
            inward = -normal / norm
            area = 0.5 * norm
            coefficient += area * profile * inward
            loaded_area += area
            rows.append({"triangle_id": int(triangle.id), "node_ids": list(triangle.node_ids), "area_mm2": area, "profile_weight": float(profile), "inward_normal": inward.tolist(), "centroid_mm": centroid.tolist()})
    coefficient_magnitude = float(np.linalg.norm(coefficient))
    if not rows or coefficient_magnitude <= 0.0:
        raise RuntimeError("3D localized footprint has zero discrete resultant")
    pressure = float(target_force_n / coefficient_magnitude)
    achieved = float(np.linalg.norm(coefficient * pressure))
    return {
        "target_force_n": float(target_force_n),
        "achieved_discrete_force_n": achieved,
        "pressure_mpa": pressure,
        "peak_pressure_mpa": pressure,
        "mean_effective_pressure_mpa": float(target_force_n / loaded_area),
        "loaded_area_mm2": loaded_area,
        "normalization_error_n": achieved - target_force_n,
        "radius_mm": RADIUS_MM,
        "profile": "compact_cosine_radial",
        "normalization": "discrete_3d_resultant_force",
        "selected_triangle_count": len(rows),
        "selected_triangles": rows,
        "coefficient_resultant_unit_pressure_n": coefficient.tolist(),
    }


def _force_profile_2d(mesh: Any, side: str, pressure_mpa: float, target_force_n: float) -> dict[str, Any]:
    center_x = SIDES[side]
    coefficient = np.zeros(2, dtype=float)
    loaded_length = 0.0
    count = 0
    for edge in mesh.boundary_edges["pad_outer_arc"]:
        first, second = (mesh.nodes[node_id] for node_id in edge.node_ids)
        midpoint = np.asarray(((first.x_mm + second.x_mm) * 0.5, (first.y_mm + second.y_mm) * 0.5))
        dx = second.x_mm - first.x_mm
        dy = second.y_mm - first.y_mm
        length = math.hypot(dx, dy)
        profile = localized.localized_profile(abs(float(midpoint[0]) - center_x), RADIUS_MM)
        if profile <= 0.0 or length <= 0.0:
            continue
        inward = np.asarray((-dy, dx), dtype=float) / length
        coefficient += length * profile * inward
        loaded_length += length
        count += 1
    resultant = coefficient * pressure_mpa
    return {
        "target_force_reference_n": float(target_force_n),
        "pressure_mpa": float(pressure_mpa),
        "achieved_line_resultant_n_per_mm": float(np.linalg.norm(resultant)),
        "line_resultant_vector_n_per_mm": resultant.tolist(),
        "loaded_length_mm": loaded_length,
        "mean_effective_line_pressure_mpa": float(np.linalg.norm(resultant) / loaded_length) if loaded_length else None,
        "normalization": "same_case_3d_pressure_amplitude_unit_depth_analogue",
        "selected_edge_count": count,
        "coefficient_resultant_unit_pressure_n_per_mm": coefficient.tolist(),
    }


def _pressure_for_case(case: Mapping[str, Any], force_n: float) -> float:
    """Compute the case-specific pressure from the authoritative 3D surface."""
    tip = Fingertip(FingertipParameters(**case["parameters"]))
    solid = build_fingertip_solid(tip.geometry)
    mesh = generate_volume_mesh(solid, volume_mesh_settings_for_tier(MESH_3D_TIER))
    return float(_force_profile_3d(mesh, case["side"], force_n)["pressure_mpa"])


def _deformed_3d_metrics(mesh: Any, result: Any, load: Mapping[str, Any]) -> dict[str, Any]:
    """Return cheap, solver-independent deformation and quality diagnostics."""
    reference = np.asarray(result.reference_coordinates_mm, dtype=float)
    deformed = np.asarray(result.deformed_coordinates_mm, dtype=float)
    node_order = tuple(sorted(mesh.nodes))
    node_index = {int(node_id): index for index, node_id in enumerate(node_order)}
    ratios: list[float] = []
    for tetra in mesh.tetrahedra:
        indices = [node_index[int(node_id)] for node_id in tetra.node_ids]
        ref = reference[indices]
        current = deformed[indices]
        ref_det = float(np.linalg.det(np.stack((ref[1] - ref[0], ref[2] - ref[0], ref[3] - ref[0]))))
        current_det = float(np.linalg.det(np.stack((current[1] - current[0], current[2] - current[0], current[3] - current[0]))))
        if abs(ref_det) <= 1.0e-14:
            ratios.append(float("nan"))
        else:
            ratios.append(current_det / ref_det)
    finite_ratios = [value for value in ratios if math.isfinite(value)]
    selected_nodes = {
        int(node_id)
        for item in load.get("selected_triangles", [])
        for node_id in item.get("node_ids", [])
    }
    displacement_norm = np.linalg.norm(deformed - reference, axis=1)
    load_indices = [node_index[node_id] for node_id in selected_nodes if node_id in node_index]
    void_indices = {
        int(node_id)
        for tag, triangles in mesh.surface_triangles.items()
        if "void" in tag.lower()
        for triangle in triangles
        for node_id in triangle.node_ids
    }
    void_reference = reference[[node_index[node_id] for node_id in void_indices if node_id in node_index]] if void_indices else np.empty((0, 3))
    void_deformed = deformed[[node_index[node_id] for node_id in void_indices if node_id in node_index]] if void_indices else np.empty((0, 3))
    void_descriptor = {"node_count": int(len(void_indices))}
    if len(void_reference):
        for name, points in (("reference", void_reference), ("deformed", void_deformed)):
            void_descriptor[name] = {
                "centroid_mm": points.mean(axis=0).tolist(),
                "x_width_mm": float(np.ptp(points[:, 0])),
                "y_height_mm": float(np.ptp(points[:, 1])),
                "z_width_mm": float(np.ptp(points[:, 2])),
            }
        void_descriptor["centroid_displacement_mm"] = (void_deformed.mean(axis=0) - void_reference.mean(axis=0)).tolist()
        void_descriptor["x_width_change_mm"] = float(np.ptp(void_deformed[:, 0]) - np.ptp(void_reference[:, 0]))
        void_descriptor["y_height_change_mm"] = float(np.ptp(void_deformed[:, 1]) - np.ptp(void_reference[:, 1]))
    return {
        "max_displacement_mm": float(displacement_norm.max(initial=0.0)),
        "rms_displacement_mm": float(np.sqrt(np.mean(displacement_norm * displacement_norm))),
        "load_region_displacement_mm": float(displacement_norm[load_indices].max(initial=0.0)) if load_indices else None,
        "minimum_deformed_volume_ratio": min(finite_ratios, default=None),
        "minimum_detF": min(finite_ratios, default=None),
        "inverted_or_invalid_tetrahedra": int(sum(not math.isfinite(value) or value <= 0.0 for value in ratios)),
        "void_deformation": void_descriptor,
        "finite_state": bool(np.all(np.isfinite(deformed)) and all(math.isfinite(value) for value in ratios)),
    }


def _timed_2d(case: Mapping[str, Any], force_n: float, pressure_mpa: float, *, stage: str) -> dict[str, Any]:
    started = time.perf_counter()
    phase_started = started
    parameters = FingertipParameters(**case["parameters"])
    tip = Fingertip(parameters)
    geometry_seconds = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    mesh = generate_fingertip_mesh(tip.geometry, mesh_settings_for_level(MESH_2D_LEVEL))
    meshing_seconds = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    load = _force_profile_2d(mesh, case["side"], pressure_mpa, force_n)
    load_seconds = time.perf_counter() - phase_started
    load_definition = {
        "load_type": "localized_normal_traction_pressure",
        "center_x_mm": case["center_x_mm"],
        "center_z_mm": 0.0,
        "radius_mm": RADIUS_MM,
        "pressure_mpa": pressure_mpa,
        "profile": "compact_cosine_radial",
        "normalization": load["normalization"],
        "target_force_n": force_n,
    }
    phase_started = time.perf_counter()
    payload, state = localized._solve_2d({**case, "morphology_id": case["case_id"]}, mesh, case["side"], pressure_mpa)
    solve_seconds = time.perf_counter() - phase_started
    payload["force_control"] = load
    payload["mesh"] = {"level": MESH_2D_LEVEL, "nodes": len(mesh.nodes), "pad_elements": len(mesh.pad_elements)}
    payload["timing"] = {
        "geometry_construction_seconds": geometry_seconds,
        "meshing_seconds": meshing_seconds,
        "load_construction_seconds": load_seconds,
        "nonlinear_solve_seconds": solve_seconds,
        "solver_reported_timing": payload.get("timing", {}),
        "total_runtime_seconds": time.perf_counter() - started,
    }
    payload["load"] = load_definition
    payload["status"] = "PASS" if payload.get("status") == "PASS" and state is not None else payload.get("status", "NUMERICAL_FAIL")
    contract = _case_contract(case, stage, force_n=force_n, pressure_mpa=pressure_mpa)
    artifact = {**contract, **_jsonable(payload), "artifact_created_at": _now()}
    path = _case_path(stage, case["case_id"])
    if state is not None:
        state_path = path.with_suffix(".npz")
        _atomic_npz(state_path, displacement=state)
        artifact["state_artifact"] = str(state_path)
        artifact["state_sha256"] = sha256_file(state_path)
    serialization_started = time.perf_counter()
    atomic_write_json(path, artifact)
    artifact["timing"]["artifact_serialization_seconds"] = time.perf_counter() - serialization_started
    atomic_write_json(path, artifact)
    return artifact


def _native_localized_state(case: Mapping[str, Any], mesh: Any, result: Any, load_definition: Mapping[str, Any], payload: Mapping[str, Any], output_dir: Path) -> Path:
    if not result.converged or result.deformed_coordinates_mm is None or result.displacement_mm is None:
        raise RuntimeError("cannot export an unconverged localized 3D state")
    node_order = tuple(sorted(mesh.nodes))
    node_ids = np.asarray(node_order, dtype=np.int64)
    reference = np.asarray(result.reference_coordinates_mm, dtype=float)
    deformed = np.asarray(result.deformed_coordinates_mm, dtype=float)
    displacement = np.asarray(result.displacement_mm, dtype=float)
    node_index = {int(node_id): index for index, node_id in enumerate(node_ids)}
    tetrahedra = np.asarray([tetra.node_ids for tetra in mesh.tetrahedra], dtype=np.int64)
    rows: list[tuple[int, str, tuple[int, int, int]]] = []
    for tag, triangles in sorted(mesh.surface_triangles.items()):
        rows.extend((int(triangle.id), str(tag), tuple(int(v) for v in triangle.node_ids)) for triangle in triangles)
    rows.sort(key=lambda value: (value[1], value[0], value[2]))
    faces_raw = np.asarray([row[2] for row in rows], dtype=np.int64)
    faces = _orient_surface_faces(faces_raw, node_ids, reference)
    tags = tuple(row[1] for row in rows)
    indices = np.asarray([[node_index[int(v)] for v in face] for face in faces], dtype=np.int64)
    ref_points = reference[indices]
    def_points = deformed[indices]
    ref_cross = np.cross(ref_points[:, 1] - ref_points[:, 0], ref_points[:, 2] - ref_points[:, 0])
    def_cross = np.cross(def_points[:, 1] - def_points[:, 0], def_points[:, 2] - def_points[:, 0])
    ref_len = np.linalg.norm(ref_cross, axis=1)
    def_len = np.linalg.norm(def_cross, axis=1)
    if np.any(ref_len <= 1.0e-12) or np.any(def_len <= 1.0e-12) or np.any(np.sum(ref_cross * def_cross, axis=1) <= 0.0):
        raise RuntimeError("localized 3D surface is degenerate or orientation-flipped")
    ref_normals = ref_cross / ref_len[:, None]
    def_normals = def_cross / def_len[:, None]
    lateral = [i for i, tag in enumerate(tags) if not tag.startswith("longitudinal_end_")]
    if not lateral:
        raise RuntimeError("localized 3D state has no lateral surface")
    lateral_faces = faces[lateral]
    lateral_tags = tuple(tags[i] for i in lateral)
    silicone_ids = np.unique(lateral_faces.reshape(-1))
    silicone_index = {int(v): i for i, v in enumerate(silicone_ids)}
    silicone_faces = np.asarray([[silicone_index[int(v)] for v in face] for face in lateral_faces], dtype=np.uint32)
    silicone_vertices = np.asarray([deformed[node_index[int(v)]] for v in silicone_ids], dtype=np.float32)
    reference_mesh = mesh
    native_reference = {int(v): reference[node_index[int(v)], :2] for v in silicone_ids}
    boundary_u = _external_surface_u(Fingertip(FingertipParameters(**case["parameters"])).mesh(M4_REFERENCE_MESH_SETTINGS), native_reference)
    u_start, u_end, external = [], [], []
    for face, tag in zip(lateral_faces, lateral_tags):
        values = [boundary_u.get(int(v), 0.0) for v in face]
        u_start.append(float(min(values))); u_end.append(float(max(values))); external.append(tag.startswith("outer_compliant_"))
    tip = Fingertip(FingertipParameters(**case["parameters"]))
    reference_transport = build_transport_geometry(tip, tip.mesh(M4_REFERENCE_MESH_SETTINGS).pad, tip.mesh(M4_REFERENCE_MESH_SETTINGS), depth_mm=11.0)
    state_fp = _fingerprint({"case": case["case_id"], "morphology": case["morphology_fingerprint"], "load": load_definition, "configuration": payload.get("configuration", {})})
    mesh_fp = _native_array_fingerprint(("node_ids", node_ids), ("undeformed_nodes_xyz", reference), ("tetrahedra_node_ids", tetrahedra), ("surface_faces_node_ids", faces), ("surface_semantic_tags", np.frombuffer("\n".join(tags).encode(), dtype=np.uint8)))
    mechanics_configuration = {"configuration": payload.get("configuration", {}), "localized_load": load_definition, "force_control": payload.get("force_control", {})}
    mechanics_fp = _fingerprint(mechanics_configuration)
    metadata = {
        "schema": "native-3d-fea-state-v1",
        "morphology_id": case["case_id"],
        "morphology_fingerprint": case["morphology_fingerprint"],
        "contact_state_fingerprint": state_fp,
        "mechanics_source": "fem.solid3d.solve_solid_3d.localized_load",
        "surface_provenance": "direct result.deformed_coordinates_mm; no 2D deformation reconstruction",
        "surface_scope": "lateral compliant-pad surfaces; longitudinal cell caps remain periodic transport boundaries",
        "surface_semantic_tags": sorted(set(tags)),
        "mesh_fingerprint": mesh_fp,
        "mechanics_config_fingerprint": mechanics_fp,
        "localized_load": load_definition,
        "configuration": payload.get("configuration", {}),
    }
    arrays = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        "node_ids": node_ids, "undeformed_nodes_xyz": reference, "deformed_nodes_xyz": deformed, "displacement_xyz": displacement,
        "tetrahedra_node_ids": tetrahedra, "surface_faces_node_ids": faces, "surface_semantic_tags_json": np.asarray(json.dumps(list(tags))),
        "surface_reference_normals": ref_normals, "surface_deformed_normals": def_normals,
        "silicone_node_ids": silicone_ids, "silicone_vertices": silicone_vertices, "silicone_faces": silicone_faces,
        "silicone_normals": np.asarray(def_normals[lateral], dtype=np.float32), "silicone_external_surface": np.asarray(external, dtype=bool),
        "silicone_u_start": np.asarray(u_start), "silicone_u_end": np.asarray(u_end), "silicone_semantic_tags_json": np.asarray(json.dumps(list(lateral_tags))),
    }
    for prefix, surface in (("rigid", reference_transport.rigid), ("envelope", reference_transport.envelope)):
        arrays[f"{prefix}_vertices"] = surface.vertices; arrays[f"{prefix}_faces"] = surface.faces; arrays[f"{prefix}_normals"] = surface.normals
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = case["case_id"]
    state_path = output_dir / f"{stem}.npz"
    temporary = state_path.with_name(f".{state_path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(state_path)
    state_sha = sha256_file(state_path)
    manifest = {
        "schema": "native-3d-fea-state-v1", "morphology_id": case["case_id"], "morphology_fingerprint": case["morphology_fingerprint"],
        "contact_state_fingerprint": state_fp, "mechanics_source": metadata["mechanics_source"], "mechanics_config_fingerprint": mechanics_fp,
        "mesh_fingerprint": mesh_fp, "tier": MESH_3D_TIER, "contact_location": case["side"], "contact_location_mm": case["center_x_mm"],
        "total_prescribed_travel_mm": None, "indenter_radius_mm": None, "initial_gap_mm": None,
        "localized_load_only": True, "force_target_n": load_definition.get("target_force_n"), "source_position_mm": list(reference_transport.source_position_mm),
        "source_medium": int(reference_transport.source_medium), "native_state_artifact": state_path.name, "native_state_sha256": state_sha,
        "surface_artifact": state_path.name, "surface_sha256": state_sha, "surface_provenance": metadata["surface_provenance"],
        "optical_fixed_surface_source": "authoritative undeformed FingertipMesh carrier/envelope", "created_at_utc": _now(),
    }
    manifest_path = output_dir / f"{stem}.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def _timed_3d(case: Mapping[str, Any], force_n: float, *, stage: str) -> dict[str, Any]:
    started = time.perf_counter()
    phase_started = started
    parameters = FingertipParameters(**case["parameters"])
    tip = Fingertip(parameters)
    geometry_seconds = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    mesh = generate_volume_mesh(build_fingertip_solid(tip.geometry), volume_mesh_settings_for_tier(MESH_3D_TIER))
    meshing_seconds = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    load = _force_profile_3d(mesh, case["side"], force_n)
    load_seconds = time.perf_counter() - phase_started
    load_definition = {"load_type": "localized_normal_surface_pressure", "center_x_mm": case["center_x_mm"], "center_z_mm": 0.0, "radius_mm": RADIUS_MM, "pressure_mpa": load["pressure_mpa"], "target_force_n": force_n, "profile": "compact_cosine_radial", "normalization": "discrete_3d_resultant_force", "orientation": "inward_surface_normal"}
    history: list[dict[str, Any]] = []
    phase_started = time.perf_counter()
    result = solve_solid_3d(mesh, None, SolidFEASettings(mode="production", number_of_steps=STEPS, indentation_mm=1.0, external_contact=False), step_history=history, localized_load=load_definition)
    solve_seconds = time.perf_counter() - phase_started
    payload = {"status": "PASS" if result.converged else "NUMERICAL_FAIL", "dimension": "3D", "case_id": case["case_id"], "side": case["side"], "load": load_definition, "force_control": load, "history": history, "configuration": result.configuration, "reaction_force_n": result.reaction_force_n, "failure_message": result.failure_message, "mesh": {"tier": MESH_3D_TIER, "nodes": len(mesh.nodes), "tetrahedra": len(mesh.tetrahedra), "quality": asdict(mesh.quality)}, "timing": {"geometry_construction_seconds": geometry_seconds, "meshing_seconds": meshing_seconds, "load_construction_seconds": load_seconds, "nonlinear_solve_seconds": solve_seconds, "total_runtime_seconds": time.perf_counter() - started}, "no_external_contact_code": True}
    contract = _case_contract(case, stage, force_n=force_n, pressure_mpa=load["pressure_mpa"])
    path = _case_path(stage, case["case_id"])
    if result.converged and result.displacement_mm is not None:
        payload.update(_deformed_3d_metrics(mesh, result, load))
        phase_started = time.perf_counter()
        native = _native_localized_state(case, mesh, result, load_definition, payload, OUTPUT / stage / "native_states")
        export_seconds = time.perf_counter() - phase_started
        payload["native_manifest"] = str(native)
        payload["native_manifest_sha256"] = sha256_file(native)
        payload["timing"]["native_export_seconds"] = export_seconds
    artifact = {**contract, **_jsonable(payload), "artifact_created_at": _now()}
    serialization_started = time.perf_counter()
    atomic_write_json(path, artifact)
    artifact["timing"]["artifact_serialization_seconds"] = time.perf_counter() - serialization_started
    atomic_write_json(path, artifact)
    return artifact


def _amend_lower_calibration_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    amended = dict(manifest)
    amended["calibration_amendment"] = {
        "reason": "nominal 2D 5 N exhausted 35 Newton iterations at step 6/12",
        "original_5_10_15_n_preserved": True,
        "lower_bracket_force_targets_n": list(LOWER_CALIBRATION_FORCE_TARGETS_N),
    }
    amended["calibration_force_targets_n"] = list(LOWER_CALIBRATION_FORCE_TARGETS_N)
    amended.pop("experiment_fingerprint", None)
    amended["experiment_fingerprint"] = _fingerprint(amended)
    atomic_write_json(OUTPUT / "experiment_manifest.json", _jsonable(amended))
    return _manifest()


def _read_existing_calibration_3d() -> dict[str, Any] | None:
    path = OUTPUT / "calibration_3d" / "base_00_nominal__FIXED__left.json"
    if not path.exists():
        return None
    payload = strict_read_json(path)
    native_path = Path(str(payload.get("native_manifest", "")))
    if payload.get("max_displacement_mm") is None and native_path.exists():
        native = strict_read_json(native_path)
        state_path = Path(str(native.get("native_state_artifact", "")))
        if not state_path.is_absolute():
            state_path = native_path.parent / state_path
        if state_path.exists():
            with np.load(state_path, allow_pickle=False) as archive:
                displacement = np.asarray(archive["displacement_xyz"], dtype=float)
            norms = np.linalg.norm(displacement, axis=1)
            payload["max_displacement_mm"] = float(norms.max(initial=0.0))
            payload["rms_displacement_mm"] = float(np.sqrt(np.mean(norms * norms)))
            atomic_write_json(path, _jsonable(payload))
    return payload


def _calibration() -> dict[str, Any]:
    manifest = _manifest()
    nominal = next(case for case in _cases() if case["case_id"] == "base_00_nominal__FIXED__left")
    results = []
    existing_2d_path = OUTPUT / "calibration_2d" / "base_00_nominal__FIXED__left.json"
    existing_3d = _read_existing_calibration_3d()
    if existing_2d_path.exists() and existing_3d is not None:
        results.append({"target_force_n": 5.0, "2D": strict_read_json(existing_2d_path), "3D": existing_3d})
    else:
        pressure = _pressure_for_case(nominal, 5.0)
        results.append({"target_force_n": 5.0, "2D": _timed_2d(nominal, 5.0, pressure, stage="calibration_2d"), "3D": _timed_3d(nominal, 5.0, stage="calibration_3d")})
    five_is_2d_valid = results[0]["2D"].get("status") == "PASS"
    if five_is_2d_valid:
        manifest = dict(manifest)
        calibration_targets = (10.0, 15.0)
    else:
        manifest = _amend_lower_calibration_manifest(manifest)
        calibration_targets = LOWER_CALIBRATION_FORCE_TARGETS_N
    for force in calibration_targets:
        pressure = _pressure_for_case(nominal, force)
        two_d = _timed_2d(nominal, force, pressure, stage="calibration_2d")
        three_d = _timed_3d(nominal, force, stage="calibration_3d")
        results.append({"target_force_n": force, "2D": two_d, "3D": three_d})
    assessments = []
    for row in results:
        d2, d3 = row["2D"], row["3D"]
        d3u = float(d3.get("max_displacement_mm") or 0.0)
        d3_quality = d3.get("minimum_deformed_volume_ratio")
        assessments.append({"target_force_n": row["target_force_n"], "mechanically_valid": d2.get("status") == "PASS" and d3.get("status") == "PASS" and bool(d3.get("finite_state", False)) and (d3_quality is None or d3_quality > 0.0), "useful_deformation": 0.05 < d3u < 10.0, "preferred_deformation_range": 1.0 <= d3u <= 3.0, "max_displacement_3d_mm": d3u, "pressure_mpa_3d": d3.get("force_control", {}).get("pressure_mpa"), "achieved_force_n": d3.get("force_control", {}).get("achieved_discrete_force_n"), "minimum_deformed_volume_ratio": d3_quality, "void_deformation": d3.get("void_deformation")})
    eligible = [row for row in assessments if row["mechanically_valid"] and row["useful_deformation"]]
    selected = max((row["target_force_n"] for row in eligible), default=None)
    summary = {"schema": "force-localized-calibration-v1", "experiment_fingerprint": _manifest()["experiment_fingerprint"], "precommit_experiment_fingerprint": manifest["experiment_fingerprint"], "original_force_targets_n": list(FORCE_TARGETS_N), "calibration_force_targets_n": [row["target_force_n"] for row in results], "results": results, "assessments": assessments, "selected_production_force_n": selected, "selection_rule": "highest precommitted mechanically valid force with useful non-destructive deformation; no optical criterion", "five_newton_2d_boundary": not five_is_2d_valid, "status": "PASS" if selected is not None else "BLOCKED"}
    atomic_write_json(OUTPUT / "calibration.json", _jsonable(summary))
    if selected is None:
        return summary
    selected_row = next(row for row in results if row["target_force_n"] == selected)
    frozen = dict(manifest)
    frozen["production_force_n"] = selected
    frozen["production_profile"] = selected_row["3D"].get("force_control")
    frozen["calibration_artifact"] = str(OUTPUT / "calibration.json")
    frozen.pop("experiment_fingerprint", None)
    frozen["experiment_fingerprint"] = _fingerprint(frozen)
    atomic_write_json(OUTPUT / "experiment_manifest.json", _jsonable(frozen))
    return summary


def _smoke_cases() -> list[dict[str, Any]]:
    selected_base_ids = {
        "base_00_nominal",
        "base_01_candidate49",
        "base_02_lhs_01",
        "base_23_lhs_22",
    }
    selected_arms = {"FIXED", "VARIED"}
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for case in _cases():
        key = (str(case["base_id"]), str(case["arm"]))
        if case["base_id"] in selected_base_ids and case["arm"] in selected_arms and key not in seen:
            result.append(case)
            seen.add(key)
    return result


def _smoke_stage() -> dict[str, Any]:
    force = _production_force()
    records: list[dict[str, Any]] = []
    pressure_cache: dict[tuple[str, str], float] = {}
    for case in _smoke_cases():
        sides = ("left", "right") if case["base_id"] in {"base_00_nominal", "base_01_candidate49"} else ("left",)
        for side in sides:
            selected = dict(case)
            selected["side"] = side
            selected["center_x_mm"] = SIDES[side]
            selected["case_id"] = f"{case['base_id']}__{case['arm']}__{side}"
            key = (str(case["morphology_fingerprint"]), side)
            try:
                pressure = pressure_cache.setdefault(key, _pressure_for_case(selected, force))
                two_d = _timed_2d(selected, force, pressure, stage="smoke_2d")
                three_d = _timed_3d(selected, force, stage="smoke_3d")
                record = {"case_id": selected["case_id"], "side": side, "2D": two_d, "3D": three_d, "status": "PASS" if two_d.get("status") == "PASS" and three_d.get("status") == "PASS" else "NUMERICAL_FAIL"}
            except Exception as exc:
                record = {"case_id": selected["case_id"], "side": side, "status": "IMPLEMENTATION_FAIL", "failure_reason": f"{type(exc).__name__}: {exc}"}
            records.append(record)
            atomic_write_json(OUTPUT / "smoke_progress.json", _jsonable({"experiment_fingerprint": _manifest()["experiment_fingerprint"], "completed_case_count": len(records), "records": records}))
    counts = {status: sum(row.get("status") == status for row in records) for status in ("PASS", "NUMERICAL_FAIL", "IMPLEMENTATION_FAIL", "RUNTIME_LIMIT")}
    summary = {"schema": "force-localized-smoke-summary-v1", "experiment_fingerprint": _manifest()["experiment_fingerprint"], "production_force_n": force, "planned_cases": len(records), "records": records, "counts": counts, "status": "PASS" if counts["PASS"] == len(records) else "INCOMPLETE"}
    atomic_write_json(OUTPUT / "smoke_summary.json", _jsonable(summary))
    return summary


def _read_case(stage: str, case: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _case_path(stage, case["case_id"])
    if not path.exists():
        return None
    try:
        payload = strict_read_json(path)
        if payload.get("case_id") != case["case_id"] or payload.get("experiment_fingerprint") != _manifest()["experiment_fingerprint"]:
            return None
        if payload.get("status") not in {"PASS", "NUMERICAL_FAIL", "IMPLEMENTATION_FAIL", "RUNTIME_LIMIT"}:
            return None
        if payload.get("status") == "PASS":
            if stage.startswith("fea2d"):
                state = Path(str(payload.get("state_artifact", "")))
                if not state.exists() or payload.get("state_sha256") != sha256_file(state):
                    return None
            if stage.startswith("fea3d"):
                native = Path(str(payload.get("native_manifest", "")))
                if not native.exists() or payload.get("native_manifest_sha256") != sha256_file(native):
                    return None
        return payload
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _production_force() -> float:
    value = _manifest().get("production_force_n")
    if value is None:
        raise RuntimeError("production force is not frozen; run calibration first")
    return float(value)


def _mechanics_stage(stage: str) -> dict[str, Any]:
    force = _production_force()
    records = []
    cases = _cases()
    pressure_cache: dict[tuple[str, str, float], float] = {}
    for index, case in enumerate(cases, start=1):
        existing = _read_case(stage, case)
        if existing is not None:
            record = {**existing, "reused": True}
        else:
            try:
                if stage == "fea2d":
                    cache_key = (str(case["morphology_fingerprint"]), str(case["side"]), float(force))
                    pressure = pressure_cache.setdefault(cache_key, _pressure_for_case(case, force))
                    record = _timed_2d(case, force, pressure, stage="fea2d")
                else:
                    record = _timed_3d(case, force, stage="fea3d")
            except Exception as exc:
                record = {"case_id": case["case_id"], "experiment_fingerprint": _manifest()["experiment_fingerprint"], "status": "IMPLEMENTATION_FAIL", "failure_reason": f"{type(exc).__name__}: {exc}"}
                atomic_write_json(_case_path(stage, case["case_id"]), record)
        records.append(record)
        atomic_write_json(OUTPUT / f"{stage}_progress.json", {"schema": f"{stage}-progress-v1", "experiment_fingerprint": _manifest()["experiment_fingerprint"], "completed_case_count": index, "records": [{"case_id": row.get("case_id"), "status": row.get("status")} for row in records]})
    counts = {status: sum(row.get("status") == status for row in records) for status in ("PASS", "NUMERICAL_FAIL", "IMPLEMENTATION_FAIL", "RUNTIME_LIMIT")}
    stage_status = "PASS" if counts["PASS"] == len(cases) else ("COMPLETE_WITH_FAILURES" if len(records) == len(cases) else "INCOMPLETE")
    summary = {"schema": f"force-localized-{stage}-summary-v1", "experiment_fingerprint": _manifest()["experiment_fingerprint"], "production_force_n": force, "planned_cases": len(cases), "records": records, "counts": counts, "status": stage_status}
    atomic_write_json(OUTPUT / f"{stage}_summary.json", _jsonable(summary))
    return summary


def _optix_settings(mode: str, bounds: tuple[tuple[float, float], tuple[float, float]]) -> Transport3DSettings:
    return Transport3DSettings(mode=mode, ray_count=RAY_COUNT, max_interactions=10, minimum_ray_weight=1.0e-4, maximum_segment_count=max(24000, 24 * RAY_COUNT), maximum_periodic_wraps=32, terminate_on_periodic_wrap_limit=True, terminate_on_no_event=True, extrusion_depth_mm=11.0, internal_grid_width=GRID_WIDTH, internal_grid_height=GRID_HEIGHT, internal_z_bins=GRID_Z_BINS, projected_grid_width=GRID_WIDTH, projected_grid_height=GRID_HEIGHT, x_bounds_mm=bounds[0], y_bounds_mm=bounds[1], retain_projected_segments=mode == "planar", retain_internal_path_field=mode == "full3d")


def _load_2d(record: Mapping[str, Any], case: Mapping[str, Any]) -> tuple[Fingertip, Any, Any]:
    tip = Fingertip(FingertipParameters(**case["parameters"]))
    mesh = generate_fingertip_mesh(tip.geometry, mesh_settings_for_level(MESH_2D_LEVEL))
    with np.load(Path(str(record["state_artifact"])), allow_pickle=False) as archive:
        displacement = np.asarray(archive["displacement"], dtype=float)
    return tip, mesh, mesh.pad.deformed(displacement)


def _load_3d(record: Mapping[str, Any], case: Mapping[str, Any]):
    tip = Fingertip(FingertipParameters(**case["parameters"]))
    artifact = load_full3d_surface_artifact(Path(str(record["native_manifest"])), expected_morphology_fingerprint=case["morphology_fingerprint"], expected_contact_state_fingerprint=str(strict_read_json(Path(str(record["native_manifest"]))).get("contact_state_fingerprint")))
    return tip, artifact


def _repair_localized_native_orientation(record: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    """Repair exporter-only normal metadata using the persisted native state.

    The localized exporter historically computed normals from raw face winding
    after persisting a separately oriented face list.  This changes only the
    derived normal arrays in an existing, exact-fingerprint mechanics artifact;
    it does not rerun FEA or alter coordinates, faces, loads, or mechanics.
    """
    manifest_path = Path(str(record["native_manifest"]))
    manifest = strict_read_json(manifest_path)
    state_path = Path(str(manifest.get("native_state_artifact", manifest.get("surface_artifact", ""))))
    if not state_path.is_absolute():
        state_path = manifest_path.parent / state_path
    with np.load(state_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    node_ids = np.asarray(arrays["node_ids"], dtype=np.int64)
    reference = np.asarray(arrays["undeformed_nodes_xyz"], dtype=float)
    deformed = np.asarray(arrays["deformed_nodes_xyz"], dtype=float)
    faces = np.asarray(arrays["surface_faces_node_ids"], dtype=np.int64)
    node_index = {int(node_id): index for index, node_id in enumerate(node_ids)}
    indices = np.asarray([[node_index[int(value)] for value in face] for face in faces], dtype=np.int64)

    def normals(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = coordinates[indices]
        cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        if np.any(lengths <= 1.0e-12) or not np.all(np.isfinite(cross)):
            raise ValueError("localized 3D surface has a degenerate persisted triangle")
        return cross / lengths[:, None], cross

    reference_normals, reference_cross = normals(reference)
    deformed_normals, deformed_cross = normals(deformed)
    if np.any(np.sum(reference_cross * deformed_cross, axis=1) <= 0.0):
        raise ValueError("localized 3D surface has a true deformation orientation flip")
    arrays["surface_reference_normals"] = reference_normals
    arrays["surface_deformed_normals"] = deformed_normals

    tags = json.loads(str(np.asarray(arrays["surface_semantic_tags_json"]).item()))
    lateral = [index for index, tag in enumerate(tags) if not str(tag).startswith("longitudinal_end_")]
    if not lateral:
        raise ValueError("localized 3D state has no lateral surface")
    arrays["silicone_normals"] = np.asarray(deformed_normals[lateral], dtype=np.float32)
    _atomic_npz(state_path, **arrays)

    state_sha = sha256_file(state_path)
    repair = {
        "type": "metadata_only_normal_repair",
        "reason": "exporter computed normals from raw face winding instead of persisted oriented faces",
        "source": "persisted native coordinates and surface_faces_node_ids",
        "fea_rerun": False,
        "repaired_at": _now(),
    }
    manifest["native_state_sha256"] = state_sha
    manifest["surface_sha256"] = state_sha
    manifest["native_artifact_repair"] = repair
    atomic_write_json(manifest_path, manifest)

    fea_path = _case_path("fea3d", case["case_id"])
    fea_payload = strict_read_json(fea_path)
    fea_payload["native_manifest_sha256"] = sha256_file(manifest_path)
    fea_payload["native_artifact_repair"] = repair
    atomic_write_json(fea_path, fea_payload)
    return fea_payload


def _bounds(left: Any, right: Any, left3: Any, right3: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    values_x, values_y = [], []
    for mesh in (left, right):
        values_x.extend(mesh.coordinates[:, 0].tolist()); values_y.extend(mesh.coordinates[:, 1].tolist())
    for artifact in (left3, right3):
        for surface in (artifact.silicone, artifact.rigid, artifact.envelope):
            values_x.extend(np.asarray(surface.vertices[:, 0]).tolist()); values_y.extend(np.asarray(surface.vertices[:, 1]).tolist())
    span = max(max(values_x) - min(values_x), max(values_y) - min(values_y))
    margin = 0.04 * span
    return (min(values_x) - margin, max(values_x) + margin), (min(values_y) - margin, max(values_y) + margin)


def _material_config(tip: Fingertip, settings: Transport3DSettings) -> dict[str, Any]:
    return transport_configuration(settings, material={"refractive_index_air": tip.optical.refractive_index_air, "refractive_index_silicone": tip.optical.refractive_index_silicone, "absorption_per_mm": tip.optical.absorption_per_mm, "scattering_per_mm": tip.optical.scattering_per_mm})


def _optical_case(case: Mapping[str, Any], record: Mapping[str, Any], mode: str, tip: Fingertip, geometry: Any, settings: Transport3DSettings, runtime: Any, *, artifact_dir: Path | None = None, mechanics_dimension: str | None = None, bridge_only: bool = False) -> tuple[UnifiedTransportResult, dict[str, Any]]:
    contract = {"schema": "force-localized-optix-case-contract-v1", "experiment_fingerprint": _manifest()["experiment_fingerprint"], "case_id": case["case_id"], "morphology_fingerprint": case["morphology_fingerprint"], "optical_mode": mode, "mechanics_source": record.get("state_artifact", record.get("native_manifest")), "force_target_n": _production_force(), "load_profile": {"radius_mm": RADIUS_MM, "profile": "compact_cosine_radial"}, "transport_configuration": _material_config(tip, settings)}
    if mechanics_dimension is not None:
        contract["mechanics_dimension"] = mechanics_dimension
    if bridge_only:
        contract["bridge_only"] = True
    contract_fp = fingerprint_mapping(contract)
    output_dir = OUTPUT / "optix_cases" if artifact_dir is None else artifact_dir
    path = output_dir / f"{case['case_id']}__{mode}__{RAY_COUNT}.json"
    raw_path = path.with_name(path.stem + "__raw.json")
    if path.exists() and raw_path.exists():
        try:
            return localized_optical_load(path, contract), strict_read_json(raw_path)
        except Exception:
            pass
    raw = trace_geometry(tip, geometry, settings=settings, runtime=runtime)
    result = UnifiedTransportResult.from_transport_result(raw, morphology_id=case["case_id"], morphology_fingerprint=case["morphology_fingerprint"], mechanics_source=str(contract["mechanics_source"]), mechanics_dimension=mechanics_dimension or ("2D" if mode == "PLANAR_2D" else "3D"), contact_state={"localized_load_only": True, "side": case["side"], **({"bridge_only": True} if bridge_only else {})}, transport_configuration_fingerprint=fingerprint_mapping(contract["transport_configuration"]))
    save_case_artifact(path, result, contract)
    raw_summary = {**legacy_optix._raw_descriptors(raw, result), "contract": contract, "contract_fingerprint": contract_fp}
    atomic_write_json(raw_path, _jsonable(raw_summary))
    return result, raw_summary


def _load_saved_interim_optix(path: Path, case: Mapping[str, Any], mechanics: Mapping[str, Any], mode: str) -> UnifiedTransportResult:
    """Load one existing interim optical artifact without allowing a rerun."""
    metadata = strict_read_json(path)
    contract = metadata.get("contract")
    expected_source = mechanics.get("state_artifact", mechanics.get("native_manifest"))
    if not isinstance(contract, Mapping) or contract.get("experiment_fingerprint") != _manifest()["experiment_fingerprint"]:
        raise ValueError(f"interim OptiX artifact has an invalid experiment fingerprint: {path}")
    if contract.get("case_id") != case["case_id"] or contract.get("morphology_fingerprint") != case["morphology_fingerprint"]:
        raise ValueError(f"interim OptiX artifact morphology mismatch: {path}")
    if contract.get("optical_mode") != mode or contract.get("mechanics_source") != expected_source:
        raise ValueError(f"interim OptiX artifact contract mismatch: {path}")
    if float(contract.get("force_target_n")) != _production_force():
        raise ValueError(f"interim OptiX artifact force contract mismatch: {path}")
    return localized_optical_load(path, contract)


def _interim_add_bridge_metric(record: Mapping[str, Any], left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Backfill J3_xy from existing FULL_3D fields, never by tracing again."""
    if record.get("J3_xy"):
        return dict(record)
    mechanics = record["mechanics"]
    artifacts = record["optix_artifacts"]
    fl = _load_saved_interim_optix(Path(str(artifacts["P3_left"])), left, mechanics["3d_left"], "FULL_3D")
    fr = _load_saved_interim_optix(Path(str(artifacts["P3_right"])), right, mechanics["3d_right"], "FULL_3D")
    updated = dict(record)
    updated["J3_xy"] = _interim_p3_xy_separability(fl, fr)
    updated["J3_xy_definition"] = "BRIDGE ONLY: P3_xy = sum_z(P3); never used as authoritative J3"
    return updated


def localized_optical_load(path: Path, contract: Mapping[str, Any]) -> UnifiedTransportResult:
    metadata = json.loads(path.read_text())
    if metadata.get("contract") != dict(contract):
        raise ValueError("localized OptiX contract mismatch")
    field_path = Path(str(metadata["field_artifact"]))
    if not field_path.is_absolute():
        field_path = path.parent / field_path.name
    if metadata.get("field_sha256") != sha256_file(field_path):
        raise ValueError("localized OptiX field checksum mismatch")
    with np.load(field_path, allow_pickle=False) as archive:
        field = archive["field"]
        axes = tuple(archive[f"axis_{index}"] for index in range(field.ndim))
    data = metadata["result"]
    return UnifiedTransportResult(morphology_id=data["morphology_id"], morphology_fingerprint=data["morphology_fingerprint"], mechanics_source=data["mechanics_source"], mechanics_dimension=data["mechanics_dimension"], contact_state=data["contact_state"], optical_mode=data["optical_mode"], ray_count=int(data["ray_count"]), transport_configuration_fingerprint=data["transport_configuration_fingerprint"], field=field, field_axes=axes, total_transport=float(data["total_transport"]), launched_weight=float(data["launched_weight"]), escaped_weight=float(data["escaped_weight"]), absorbed_weight=float(data["absorbed_weight"]), terminated_weight=float(data["terminated_weight"]), valid_ray_count=int(data["valid_ray_count"]), terminated_ray_count=int(data["terminated_ray_count"]), energy_balance_error=float(data["energy_balance_error"]), path_diagnostics=data.get("path_diagnostics", {}))


def _optix_stage() -> dict[str, Any]:
    force = _production_force()
    two_d = {row["case_id"]: row for row in strict_read_json(OUTPUT / "fea2d_summary.json")["records"]}
    three_d = {row["case_id"]: row for row in strict_read_json(OUTPUT / "fea3d_summary.json")["records"]}
    runtime = legacy_optix._Runtime.create()
    records = []
    for pair in _parent_manifest()["pairs"]:
        for arm in ("FIXED", "VARIED"):
            left = next(case for case in _cases() if case["base_id"] == pair["base_id"] and case["arm"] == arm and case["side"] == "left")
            right = next(case for case in _cases() if case["base_id"] == pair["base_id"] and case["arm"] == arm and case["side"] == "right")
            l2, r2, l3, r3 = two_d[left["case_id"]], two_d[right["case_id"]], three_d[left["case_id"]], three_d[right["case_id"]]
            if any(row.get("status") != "PASS" for row in (l2, r2, l3, r3)):
                records.append({"case_id": f"{pair['base_id']}__{arm}", "base_id": pair["base_id"], "arm": arm, "status": "EXCLUDED", "reason": "incomplete mechanics pair"})
                continue
            tip_l, full_l, pad_l = _load_2d(l2, left); tip_r, full_r, pad_r = _load_2d(r2, right)
            tip3_l, art_l = _load_3d(l3, left); tip3_r, art_r = _load_3d(r3, right)
            bounds = _bounds(pad_l, pad_r, art_l, art_r)
            psettings = _optix_settings("planar", bounds); fsettings = _optix_settings("full3d", bounds)
            pgeo_l = build_transport_geometry(tip_l, pad_l, full_l, depth_mm=11.0); pgeo_r = build_transport_geometry(tip_r, pad_r, full_r, depth_mm=11.0)
            fgeo_l = art_l.geometry(tip3_l); fgeo_r = art_r.geometry(tip3_r)
            pl, raw_pl = _optical_case(left, l2, "PLANAR_2D", tip_l, pgeo_l, psettings, runtime)
            pr, raw_pr = _optical_case(right, r2, "PLANAR_2D", tip_r, pgeo_r, psettings, runtime)
            fl, raw_fl = _optical_case(left, l3, "FULL_3D", tip3_l, fgeo_l, fsettings, runtime)
            fr, raw_fr = _optical_case(right, r3, "FULL_3D", tip3_r, fgeo_r, fsettings, runtime)
            records.append({"case_id": f"{pair['base_id']}__{arm}", "base_id": pair["base_id"], "arm": arm, "status": "PASS", "parameters": pair["arms"][arm]["parameters"], "normalized_coordinate": pair["arms"][arm].get("normalized_coordinate"), "J2": native_field_separability(pl, pr), "J3": native_field_separability(fl, fr), "raw": {"planar_left": raw_pl, "planar_right": raw_pr, "full_left": raw_fl, "full_right": raw_fr}, "fea": {"2d_left": l2, "2d_right": r2, "3d_left": l3, "3d_right": r3}})
            atomic_write_json(OUTPUT / "optix_progress.json", _jsonable({"experiment_fingerprint": _manifest()["experiment_fingerprint"], "records": records}))
    summary = {"schema": "force-localized-optix-summary-v1", "experiment_fingerprint": _manifest()["experiment_fingerprint"], "production_force_n": force, "planned_pairs": 48, "records": records, "counts": {status: sum(row.get("status") == status for row in records) for status in ("PASS", "EXCLUDED")}, "status": "PASS" if all(row.get("status") == "PASS" for row in records) else "INCOMPLETE"}
    atomic_write_json(OUTPUT / "optix_summary.json", _jsonable(summary))
    return summary


def _interim_p3_xy_descriptors(result: UnifiedTransportResult) -> dict[str, Any]:
    field = np.asarray(result.field, dtype=float)
    if field.ndim != 3:
        raise ValueError("interim P3 bridge requires a native 3D field")
    # Native FULL_3D fields are stored as (x, y, z); sum only the z axis.
    projected = np.sum(field, axis=2)
    axes = result.field_axes
    centers = [0.5 * (axis[:-1] + axis[1:]) for axis in axes]
    mass = float(projected.sum())
    if mass <= 0.0:
        return {"status": "ZERO_FIELD", "field_mass": mass}
    weights = projected / mass
    descriptors: dict[str, Any] = {
        "status": "PASS",
        "field_mass": mass,
        "total_transport": float(result.total_transport),
        "bridge": "P3_xy=sum_z(P3), no extra z-width multiplier",
    }
    for index, name in enumerate(("x", "y")):
        marginal = np.sum(weights, axis=1 - index)
        mean = float(np.sum(marginal * centers[index]) / max(float(marginal.sum()), 1.0e-30))
        variance = float(np.sum(marginal * (centers[index] - mean) ** 2) / max(float(marginal.sum()), 1.0e-30))
        descriptors[f"{name}_centroid_mm"] = mean
        descriptors[f"{name}_spread_mm"] = math.sqrt(max(variance, 0.0))
    return descriptors


def _interim_p3_xy_field(result: UnifiedTransportResult) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Return the non-authoritative P3_xy bridge field and its x/y grid."""
    field = np.asarray(result.field, dtype=float)
    axes = tuple(np.asarray(axis, dtype=float) for axis in result.field_axes)
    if field.ndim != 3 or len(axes) != 3 or field.shape != tuple(len(axis) - 1 for axis in axes):
        raise ValueError("interim P3_xy bridge requires a valid (x, y, z) field")
    return np.sum(field, axis=2), (axes[0], axes[1])


def _interim_p3_xy_separability(first: UnifiedTransportResult, second: UnifiedTransportResult) -> dict[str, Any]:
    """Compare P3_xy fields while keeping native FULL_3D P3 authoritative."""
    left, left_axes = _interim_p3_xy_field(first)
    right, right_axes = _interim_p3_xy_field(second)
    if left.shape != right.shape or any(not np.array_equal(a, b) for a, b in zip(left_axes, right_axes)):
        raise ValueError("P3_xy bridge requires identical x/y grids")
    first_mass = float(np.sum(left))
    second_mass = float(np.sum(right))
    normalized = None
    status = "singular_zero_field"
    if first_mass > 0.0 and second_mass > 0.0:
        normalized = 0.5 * float(np.sum(np.abs(left / first_mass - right / second_mass)))
        status = "valid"
    return {
        "metric": "J3_xy",
        "bridge_only": True,
        "definition": "P3_xy = sum_z(P3)",
        "optical_mode": "FULL_3D_P3_XY_BRIDGE",
        "normalized_redistribution_l1": normalized,
        "normalized_status": status,
        "first_projected_field_mass": first_mass,
        "second_projected_field_mass": second_mass,
        "first_total_transport": float(first.total_transport),
        "second_total_transport": float(second.total_transport),
        "total_transport_difference": float(first.total_transport - second.total_transport),
    }


def _interim_state_path(case_id: str) -> Path:
    return INTERIM_OUTPUT / f"{case_id}.json"


def _interim_valid_record(path: Path) -> dict[str, Any] | None:
    try:
        payload = strict_read_json(path)
        if payload.get("experiment_fingerprint") != _manifest()["experiment_fingerprint"]:
            return None
        if payload.get("status") != "PASS" or not payload.get("J2") or not payload.get("J3"):
            return None
        return payload
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _interim_eligible_cases() -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]]:
    by_morphology: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for case in _cases():
        by_morphology.setdefault((case["base_id"], case["arm"]), {})[case["side"]] = case
    eligible = []
    for (base_id, arm), sides in by_morphology.items():
        left, right = sides["left"], sides["right"]
        l2, r2 = _read_case("fea2d", left), _read_case("fea2d", right)
        l3, r3 = _read_case("fea3d", left), _read_case("fea3d", right)
        if all(row is not None and row.get("status") == "PASS" for row in (l2, r2, l3, r3)):
            eligible.append((left, right, l2, r2, l3, r3))
    return eligible


def _interim_pair_details(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for first, second in itertools.combinations(rows, 2):
        j2_first = float(first["J2"]["normalized_redistribution_l1"])
        j2_second = float(second["J2"]["normalized_redistribution_l1"])
        j3_first = float(first["J3"]["normalized_redistribution_l1"])
        j3_second = float(second["J3"]["normalized_redistribution_l1"])
        delta2, delta3 = j2_first - j2_second, j3_first - j3_second
        tied = abs(delta2) <= PAIR_TIE_TOLERANCE or abs(delta3) <= PAIR_TIE_TOLERANCE
        relation = "tied" if tied else ("concordant" if delta2 * delta3 > 0.0 else "discordant")
        pairs.append({"first": first["case_id"], "second": second["case_id"], "relation": relation, "abs_delta_J2": abs(delta2), "abs_delta_J3": abs(delta3)})
    pairs.sort(key=lambda row: row["abs_delta_J2"] + row["abs_delta_J3"], reverse=True)
    return {
        "strongest_concordant": [row for row in pairs if row["relation"] == "concordant"][:5],
        "strongest_discordant": [row for row in pairs if row["relation"] == "discordant"][:5],
        "counts": {relation: sum(row["relation"] == relation for row in pairs) for relation in ("concordant", "discordant", "tied")},
    }


def _interim_relation(first: float, second: float) -> str:
    if abs(first) <= PAIR_TIE_TOLERANCE or abs(second) <= PAIR_TIE_TOLERANCE:
        return "tied"
    return "concordant" if first * second > 0.0 else "discordant"


def _interim_inversion_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    inversions = []
    for first, second in itertools.combinations(rows, 2):
        j2_delta = float(first["J2"]["normalized_redistribution_l1"]) - float(second["J2"]["normalized_redistribution_l1"])
        bridge_delta = float(first["J3_xy"]["normalized_redistribution_l1"]) - float(second["J3_xy"]["normalized_redistribution_l1"])
        native_delta = float(first["J3"]["normalized_redistribution_l1"]) - float(second["J3"]["normalized_redistribution_l1"])
        bridge_relation = _interim_relation(j2_delta, bridge_delta)
        native_relation = _interim_relation(j2_delta, native_delta)
        if native_relation != "discordant":
            continue
        if bridge_relation == "discordant":
            source = "already_present_in_J3_xy"
        elif bridge_relation == "concordant":
            source = "appears_only_with_native_z_resolved_P3"
        else:
            source = "bridge_tied_or_unresolved"
        inversions.append({
            "first": first["case_id"],
            "second": second["case_id"],
            "J2_first": float(first["J2"]["normalized_redistribution_l1"]),
            "J2_second": float(second["J2"]["normalized_redistribution_l1"]),
            "J3_xy_first": float(first["J3_xy"]["normalized_redistribution_l1"]),
            "J3_xy_second": float(second["J3_xy"]["normalized_redistribution_l1"]),
            "J3_first": float(first["J3"]["normalized_redistribution_l1"]),
            "J3_second": float(second["J3"]["normalized_redistribution_l1"]),
            "bridge_relation": bridge_relation,
            "native_relation": native_relation,
            "inversion_source": source,
            "strength": abs(j2_delta) + abs(native_delta),
        })
    inversions.sort(key=lambda row: row["strength"], reverse=True)
    return {"native_discordant_pair_count": len(inversions), "strongest": inversions[:5]}


def _interim_requested_triplets(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {row["case_id"]: row for row in rows}

    def triplet(case_id: str) -> dict[str, Any] | None:
        row = by_id.get(case_id)
        if row is None:
            return None
        return {
            "case_id": case_id,
            "J2": float(row["J2"]["normalized_redistribution_l1"]),
            "J3_xy_BRIDGE_ONLY": float(row["J3_xy"]["normalized_redistribution_l1"]),
            "J3_native_FULL_3D": float(row["J3"]["normalized_redistribution_l1"]),
        }

    pair_details = _interim_pair_details(rows)
    def pair_triplets(key: str) -> dict[str, Any] | None:
        pair = pair_details[key][0] if pair_details[key] else None
        if pair is None:
            return None
        return {"pair": [pair["first"], pair["second"]], "triplets": [triplet(pair["first"]), triplet(pair["second"])]}

    return {
        "nominal": [triplet("base_00_nominal__FIXED"), triplet("base_00_nominal__VARIED")],
        "candidate49_FIXED": triplet("base_01_candidate49__FIXED"),
        "candidate49_VARIED": triplet("base_01_candidate49__VARIED"),
        "base03_FIXED": triplet("base_03_lhs_02__FIXED"),
        "strongest_current_concordant_case": pair_triplets("strongest_concordant"),
        "strongest_current_discordant_case": pair_triplets("strongest_discordant"),
    }


def _mechanistic_bridge_state_path(state_id: str) -> Path:
    return MECHANISTIC_BRIDGE_OUTPUT / f"{state_id}.json"


def _mechanistic_bridge_valid_record(path: Path) -> dict[str, Any] | None:
    try:
        payload = strict_read_json(path)
        if payload.get("experiment_fingerprint") != _manifest()["experiment_fingerprint"]:
            return None
        if payload.get("status") != "PASS" or not all(payload.get(key) for key in ("A_J2", "B_J2MECH_FULL3D", "C_J3")):
            return None
        return payload
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _mechanistic_bridge_rank_changes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values = {key: {row["case_id"]: float(row[key]["normalized_redistribution_l1"]) for row in rows} for key in ("A_J2", "B_J2MECH_FULL3D", "C_J3")}
    ranks: dict[str, dict[str, int]] = {}
    for key, entries in values.items():
        ordered = sorted(entries, key=lambda case_id: (-entries[case_id], case_id))
        ranks[key] = {case_id: index + 1 for index, case_id in enumerate(ordered)}
    return [{
        "case_id": row["case_id"],
        "rank_A_J2": ranks["A_J2"][row["case_id"]],
        "rank_B_J2MECH_FULL3D": ranks["B_J2MECH_FULL3D"][row["case_id"]],
        "rank_C_J3": ranks["C_J3"][row["case_id"]],
        "rank_change_A_to_B": ranks["B_J2MECH_FULL3D"][row["case_id"]] - ranks["A_J2"][row["case_id"]],
        "rank_change_B_to_C": ranks["C_J3"][row["case_id"]] - ranks["B_J2MECH_FULL3D"][row["case_id"]],
    } for row in rows]


def _mechanistic_bridge_classification(ab: Mapping[str, Any], bc: Mapping[str, Any]) -> str:
    if ab.get("status") != "PASS" or bc.get("status") != "PASS":
        return "INCONCLUSIVE"
    ab_strong = float(ab["spearman_rho"]) >= 0.70 and float(ab["kendall_tau"]) >= 0.50
    bc_strong = float(bc["spearman_rho"]) >= 0.70 and float(bc["kendall_tau"]) >= 0.50
    if ab_strong and not bc_strong:
        return "MECHANICS_DOMINANT"
    if not ab_strong and bc_strong:
        return "OPTICS_DOMINANT"
    if not ab_strong and not bc_strong:
        return "MIXED"
    return "INCONCLUSIVE"


def _mechanistic_bridge_report(rows: Sequence[Mapping[str, Any]], *, completed: bool = False) -> dict[str, Any]:
    rows = list(rows)
    ab = _rank_stats_for_keys(rows, "A_J2", "B_J2MECH_FULL3D", "A_J2", "B_J2MECH_FULL3D")
    bc = _rank_stats_for_keys(rows, "B_J2MECH_FULL3D", "C_J3", "B_J2MECH_FULL3D", "C_J3")
    populations = {}
    for arm in ("FIXED", "VARIED"):
        subset = [row for row in rows if row.get("arm") == arm]
        populations[arm] = {
            "n": len(subset),
            "A_vs_B": _rank_stats_for_keys(subset, "A_J2", "B_J2MECH_FULL3D", "A_J2", "B_J2MECH_FULL3D"),
            "B_vs_C": _rank_stats_for_keys(subset, "B_J2MECH_FULL3D", "C_J3", "B_J2MECH_FULL3D", "C_J3"),
        }
    return {
        "schema": "force-localized-mechanistic-bridge-summary-v1",
        "label": "INTERIM — MECHANISTIC BRIDGE — EARLY DETERMINISTIC SUBSET — NOT FINAL DESIGN-SPACE STATISTICS",
        "experiment_fingerprint": _manifest()["experiment_fingerprint"],
        "eligible_morphology_state_count": len(rows),
        "fixed_count": sum(row.get("arm") == "FIXED" for row in rows),
        "varied_count": sum(row.get("arm") == "VARIED" for row in rows),
        "A_2D_PLANAR_J2_vs_B_2D_EXTRUDED_FULL3D_J2MECH_FULL3D": ab,
        "B_J2MECH_FULL3D_vs_C_NATIVE_3D_J3": bc,
        "fixed_vs_varied": populations,
        "rank_changes": _mechanistic_bridge_rank_changes(rows),
        "classification": _mechanistic_bridge_classification(ab, bc),
        "states": rows,
        "completed": completed,
        "created_at": _now(),
    }


def _mechanistic_bridge_step(records_by_id: Mapping[str, Mapping[str, Any]], runtime: Any) -> list[dict[str, Any]]:
    MECHANISTIC_BRIDGE_OUTPUT.mkdir(parents=True, exist_ok=True)
    bridge_records = {row["case_id"]: row for row in (_mechanistic_bridge_valid_record(path) for path in MECHANISTIC_BRIDGE_OUTPUT.glob("*.json")) if row is not None}
    for left, right, l2, r2, l3, r3 in _interim_eligible_cases():
        state_id = f"{left['base_id']}__{left['arm']}"
        if state_id in bridge_records or state_id not in records_by_id:
            continue
        interim = records_by_id[state_id]
        tip_l, full_l, pad_l = _load_2d(l2, left); tip_r, full_r, pad_r = _load_2d(r2, right)
        try:
            tip3_l, art_l = _load_3d(l3, left)
        except ValueError as exc:
            if "orientation metadata is inconsistent" not in str(exc):
                raise
            l3 = _repair_localized_native_orientation(l3, left)
            tip3_l, art_l = _load_3d(l3, left)
        try:
            tip3_r, art_r = _load_3d(r3, right)
        except ValueError as exc:
            if "orientation metadata is inconsistent" not in str(exc):
                raise
            r3 = _repair_localized_native_orientation(r3, right)
            tip3_r, art_r = _load_3d(r3, right)
        bounds = _bounds(pad_l, pad_r, art_l, art_r)
        fsettings = _optix_settings("full3d", bounds)
        # This is intentionally a non-authoritative bridge: the shared FULL_3D
        # transport kernel receives the exact 2D-deformed periodic extrusion,
        # while the production FULL_3D provenance guard remains unchanged.
        bgeo_l = build_transport_geometry(tip_l, pad_l, full_l, depth_mm=11.0)
        bgeo_r = build_transport_geometry(tip_r, pad_r, full_r, depth_mm=11.0)
        bl, raw_bl = _optical_case(left, l2, "FULL_3D", tip_l, bgeo_l, fsettings, runtime, artifact_dir=MECHANISTIC_BRIDGE_OUTPUT, mechanics_dimension="3D", bridge_only=True)
        br, raw_br = _optical_case(right, r2, "FULL_3D", tip_r, bgeo_r, fsettings, runtime, artifact_dir=MECHANISTIC_BRIDGE_OUTPUT, mechanics_dimension="3D", bridge_only=True)
        record = {
            "schema": "force-localized-mechanistic-bridge-state-v1",
            "status": "PASS",
            "experiment_fingerprint": _manifest()["experiment_fingerprint"],
            "case_id": state_id,
            "base_id": left["base_id"],
            "arm": left["arm"],
            "parameters": left["parameters"],
            "A_J2": interim["J2"],
            "B_J2MECH_FULL3D": native_field_separability(bl, br),
            "B_xy": _interim_p3_xy_separability(bl, br),
            "B_xy_definition": "BRIDGE ONLY: B_xy = sum_z(P3) for the 2D-deformed FULL_3D propagation",
            "C_J3": interim["J3"],
            "raw_B_left": raw_bl,
            "raw_B_right": raw_br,
            "artifacts": {
                "B_left": str(MECHANISTIC_BRIDGE_OUTPUT / f"{left['case_id']}__FULL_3D__{RAY_COUNT}.json"),
                "B_right": str(MECHANISTIC_BRIDGE_OUTPUT / f"{right['case_id']}__FULL_3D__{RAY_COUNT}.json"),
                "A_planar_left": interim["optix_artifacts"]["P2_left"],
                "A_planar_right": interim["optix_artifacts"]["P2_right"],
                "C_native_left": interim["optix_artifacts"]["P3_left"],
                "C_native_right": interim["optix_artifacts"]["P3_right"],
            },
            "created_at": _now(),
        }
        atomic_write_json(_mechanistic_bridge_state_path(state_id), _jsonable(record))
        bridge_records[state_id] = record
    return list(bridge_records.values())


def _interim_extreme_cases(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"high_J2_low_J3": [], "low_J2_high_J3": []}
    j2 = np.asarray([float(row["J2"]["normalized_redistribution_l1"]) for row in rows])
    j3 = np.asarray([float(row["J3"]["normalized_redistribution_l1"]) for row in rows])
    rank2 = np.argsort(np.argsort(-j2)).astype(float)
    rank3 = np.argsort(np.argsort(j3)).astype(float)
    high_low = np.argsort(rank2 + rank3)[:5]
    low_high = np.argsort((len(rows) - 1 - rank2) + (len(rows) - 1 - rank3))[:5]
    def describe(index: int) -> dict[str, Any]:
        row = rows[int(index)]
        return {"case_id": row["case_id"], "J2": row["J2"], "J3": row["J3"], "void_height_mm": row["parameters"].get("void_height")}
    return {"high_J2_low_J3": [describe(index) for index in high_low], "low_J2_high_J3": [describe(index) for index in low_high]}


def _interim_void_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) < 3:
        return {"status": "INSUFFICIENT_EARLY_SUBSET", "n": len(rows)}
    void = np.asarray([float(row["parameters"].get("void_height", math.nan)) for row in rows])
    disagreement = np.asarray([abs(float(row["J2"]["normalized_redistribution_l1"]) - float(row["J3"]["normalized_redistribution_l1"])) for row in rows])
    z_spread = np.asarray([float(row["P3_left"].get("z_spread_mm", math.nan) + row["P3_right"].get("z_spread_mm", math.nan)) * 0.5 for row in rows])
    total = np.asarray([float(row["P3_left"].get("total_transport", math.nan) + row["P3_right"].get("total_transport", math.nan)) * 0.5 for row in rows])
    def association(values: np.ndarray) -> float | None:
        mask = np.isfinite(void) & np.isfinite(values)
        if np.count_nonzero(mask) < 3 or np.ptp(void[mask]) <= PAIR_TIE_TOLERANCE or np.ptp(values[mask]) <= PAIR_TIE_TOLERANCE:
            return None
        return float(spearmanr(void[mask], values[mask]).statistic)
    return {"status": "PRELIMINARY_EXPLORATORY_ONLY", "n": len(rows), "association_void_height_abs_J2_minus_J3": association(disagreement), "association_void_height_z_spread": association(z_spread), "association_void_height_total_transport": association(total)}


def _interim_report(rows: Sequence[Mapping[str, Any]], *, completed: bool = False) -> dict[str, Any]:
    rows = list(rows)
    all_stats = _rank_stats(rows)
    populations = {arm: _rank_stats([row for row in rows if row.get("arm") == arm]) for arm in ("FIXED", "VARIED")}
    bridge_stats = {
        "J2_vs_J3_xy": _rank_stats_for_keys(rows, "J2", "J3_xy", "j2", "j3_xy"),
        "J3_xy_vs_J3": _rank_stats_for_keys(rows, "J3_xy", "J3", "j3_xy", "j3"),
    }
    p3_z = [float(row["P3_left"].get("z_fraction_away_from_central_region", math.nan) + row["P3_right"].get("z_fraction_away_from_central_region", math.nan)) * 0.5 for row in rows]
    report = {
        "schema": "force-localized-interim-optix-trend-v1",
        "label": "INTERIM — EARLY DETERMINISTIC SUBSET — NOT FINAL DESIGN-SPACE STATISTICS",
        "experiment_fingerprint": _manifest()["experiment_fingerprint"],
        "eligible_morphology_state_count": len(rows),
        "fixed_count": sum(row.get("arm") == "FIXED" for row in rows),
        "varied_count": sum(row.get("arm") == "VARIED" for row in rows),
        "rank_stats": all_stats,
        "bridge_rank_stats": bridge_stats,
        "fixed_vs_varied": populations,
        "pair_details": _interim_pair_details(rows),
        "inversion_diagnostics": _interim_inversion_diagnostics(rows),
        "extreme_cases": _interim_extreme_cases(rows),
        "nominal_and_candidate49": [row for row in rows if row.get("base_id") in {"base_00_nominal", "base_01_candidate49"}],
        "requested_triplets": _interim_requested_triplets(rows),
        "z_transport_diagnostic": {"mean_fraction_away_from_center_plane": float(np.nanmean(p3_z)) if p3_z else None, "states_with_finite_z_diagnostic": int(np.count_nonzero(np.isfinite(p3_z))), "interpretation": "bridge P3_xy is retained for diagnosis; no final causal claim is made from this early subset"},
        "void_height_diagnostic": _interim_void_diagnostics(rows),
        "completed": completed,
        "created_at": _now(),
    }
    return report


def _interim_optix() -> dict[str, Any]:
    INTERIM_OUTPUT.mkdir(parents=True, exist_ok=True)
    runtime = legacy_optix._Runtime.create()
    records_by_id = {row["case_id"]: row for row in (_interim_valid_record(path) for path in INTERIM_OUTPUT.glob("*.json")) if row is not None}
    while True:
        for left, right, l2, r2, l3, r3 in _interim_eligible_cases():
            state_id = f"{left['base_id']}__{left['arm']}"
            if state_id in records_by_id:
                updated = _interim_add_bridge_metric(records_by_id[state_id], left, right)
                if updated != records_by_id[state_id]:
                    atomic_write_json(_interim_state_path(state_id), _jsonable(updated))
                    records_by_id[state_id] = updated
                continue
            tip_l, full_l, pad_l = _load_2d(l2, left); tip_r, full_r, pad_r = _load_2d(r2, right)
            try:
                tip3_l, art_l = _load_3d(l3, left)
            except ValueError as exc:
                if "orientation metadata is inconsistent" not in str(exc):
                    raise
                l3 = _repair_localized_native_orientation(l3, left)
                tip3_l, art_l = _load_3d(l3, left)
            try:
                tip3_r, art_r = _load_3d(r3, right)
            except ValueError as exc:
                if "orientation metadata is inconsistent" not in str(exc):
                    raise
                r3 = _repair_localized_native_orientation(r3, right)
                tip3_r, art_r = _load_3d(r3, right)
            bounds = _bounds(pad_l, pad_r, art_l, art_r)
            psettings = _optix_settings("planar", bounds); fsettings = _optix_settings("full3d", bounds)
            pgeo_l = build_transport_geometry(tip_l, pad_l, full_l, depth_mm=11.0); pgeo_r = build_transport_geometry(tip_r, pad_r, full_r, depth_mm=11.0)
            fgeo_l = art_l.geometry(tip3_l); fgeo_r = art_r.geometry(tip3_r)
            pl, raw_pl = _optical_case(left, l2, "PLANAR_2D", tip_l, pgeo_l, psettings, runtime)
            pr, raw_pr = _optical_case(right, r2, "PLANAR_2D", tip_r, pgeo_r, psettings, runtime)
            fl, raw_fl = _optical_case(left, l3, "FULL_3D", tip3_l, fgeo_l, fsettings, runtime)
            fr, raw_fr = _optical_case(right, r3, "FULL_3D", tip3_r, fgeo_r, fsettings, runtime)
            record = {"schema": "force-localized-interim-optix-state-v1", "status": "PASS", "experiment_fingerprint": _manifest()["experiment_fingerprint"], "case_id": state_id, "base_id": left["base_id"], "arm": left["arm"], "parameters": left["parameters"], "normalized_coordinate": left.get("normalized_coordinate"), "J2": native_field_separability(pl, pr), "J3_xy": _interim_p3_xy_separability(fl, fr), "J3_xy_definition": "BRIDGE ONLY: P3_xy = sum_z(P3); never used as authoritative J3", "J3": native_field_separability(fl, fr), "P2_left": raw_pl, "P2_right": raw_pr, "P3_left": raw_fl, "P3_right": raw_fr, "P3_xy_left": _interim_p3_xy_descriptors(fl), "P3_xy_right": _interim_p3_xy_descriptors(fr), "optix_artifacts": {"P2_left": str(OUTPUT / "optix_cases" / f"{left['case_id']}__PLANAR_2D__{RAY_COUNT}.json"), "P2_right": str(OUTPUT / "optix_cases" / f"{right['case_id']}__PLANAR_2D__{RAY_COUNT}.json"), "P3_left": str(OUTPUT / "optix_cases" / f"{left['case_id']}__FULL_3D__{RAY_COUNT}.json"), "P3_right": str(OUTPUT / "optix_cases" / f"{right['case_id']}__FULL_3D__{RAY_COUNT}.json")}, "mechanics": {"2d_left": l2, "2d_right": r2, "3d_left": l3, "3d_right": r3}, "created_at": _now()}
            atomic_write_json(_interim_state_path(state_id), _jsonable(record))
            records_by_id[state_id] = record
        bridge_rows = _mechanistic_bridge_step(records_by_id, runtime)
        progress = strict_read_json(OUTPUT / "fea3d_progress.json") if (OUTPUT / "fea3d_progress.json").exists() else {}
        completed = (OUTPUT / "fea3d_summary.json").exists() and int(progress.get("completed_case_count", 0)) >= len(_cases())
        rows = list(records_by_id.values())
        atomic_write_json(INTERIM_OUTPUT / "interim_trend_summary.json", _jsonable(_interim_report(rows, completed=completed)))
        atomic_write_json(MECHANISTIC_BRIDGE_OUTPUT / "mechanistic_bridge_summary.json", _jsonable(_mechanistic_bridge_report(bridge_rows, completed=completed)))
        if completed:
            return _interim_report(rows, completed=True)
        time.sleep(15.0)


def _rank_stats_for_keys(rows: Sequence[Mapping[str, Any]], first_key: str, second_key: str, first_label: str, second_label: str) -> dict[str, Any]:
    first_values = np.asarray([float(row[first_key]["normalized_redistribution_l1"]) for row in rows], dtype=float)
    second_values = np.asarray([float(row[second_key]["normalized_redistribution_l1"]) for row in rows], dtype=float)
    if len(rows) < 3:
        return {"status": "INCONCLUSIVE", "n": len(rows)}
    concordant = discordant = tied = 0
    for first, second in itertools.combinations(range(len(rows)), 2):
        delta_first = first_values[first] - first_values[second]
        delta_second = second_values[first] - second_values[second]
        if abs(delta_first) <= PAIR_TIE_TOLERANCE or abs(delta_second) <= PAIR_TIE_TOLERANCE:
            tied += 1
        elif delta_first * delta_second > 0.0:
            concordant += 1
        else:
            discordant += 1
    comparable = concordant + discordant
    return {"status": "PASS", "n": len(rows), "spearman_rho": float(spearmanr(first_values, second_values).statistic), "kendall_tau": float(kendalltau(first_values, second_values).statistic), "concordant_pairs": concordant, "discordant_pairs": discordant, "tied_pairs": tied, "total_pairs": len(rows) * (len(rows) - 1) // 2, "pairwise_concordance_fraction": concordant / comparable if comparable else None, f"{first_label}_dynamic_range": float(np.ptp(first_values)), f"{second_label}_dynamic_range": float(np.ptp(second_values))}


def _rank_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _rank_stats_for_keys(rows, "J2", "J3", "j2", "j3")


def _parameter_trends(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    names = tuple(rows[0]["parameters"].keys())
    result = {}
    for name in names:
        p = np.asarray([float(row["parameters"][name]) for row in rows])
        j2 = np.asarray([float(row["J2"]["normalized_redistribution_l1"]) for row in rows])
        j3 = np.asarray([float(row["J3"]["normalized_redistribution_l1"]) for row in rows])
        r2 = float(spearmanr(p, j2).statistic) if np.ptp(p) > PAIR_TIE_TOLERANCE else None
        r3 = float(spearmanr(p, j3).statistic) if np.ptp(p) > PAIR_TIE_TOLERANCE else None
        result[name] = {"spearman_parameter_J2": r2, "spearman_parameter_J3": r3, "direction_agreement": None if r2 is None or r3 is None else bool(r2 * r3 > 0.0), "sign_disagreement": None if r2 is None or r3 is None else bool(r2 * r3 < 0.0)}
    return result


def _analyze() -> dict[str, Any]:
    data = strict_read_json(OUTPUT / "optix_summary.json")
    rows = [row for row in data["records"] if row.get("status") == "PASS"]
    populations = {}
    for arm in ("FIXED", "VARIED"):
        arm_rows = [row for row in rows if row.get("arm") == arm]
        populations[arm] = {"n": len(arm_rows), "rank_stats": _rank_stats(arm_rows), "parameter_trends": _parameter_trends(arm_rows)}
    disagreements = []
    for row in rows:
        delta = abs(float(row["J2"]["normalized_redistribution_l1"]) - float(row["J3"]["normalized_redistribution_l1"]))
        disagreements.append({"case_id": row["case_id"], "absolute_J2_minus_J3": delta, "void_height_mm": row["parameters"].get("void_height"), "J2": row["J2"], "J3": row["J3"]})
    disagreements.sort(key=lambda row: row["absolute_J2_minus_J3"], reverse=True)
    fixed, varied = populations["FIXED"]["rank_stats"], populations["VARIED"]["rank_stats"]
    excluded_fraction = 1.0 - len(rows) / 48.0
    if excluded_fraction > 0.25 or fixed.get("status") != "PASS" or varied.get("status") != "PASS":
        classification = "E_INCONCLUSIVE"
    elif fixed["spearman_rho"] >= 0.70 and varied["spearman_rho"] >= 0.70 and fixed["kendall_tau"] >= 0.50 and varied["kendall_tau"] >= 0.50:
        classification = "A_STRONG_2D_TREND_PRESERVATION"
    elif fixed["spearman_rho"] - varied["spearman_rho"] >= 0.20:
        classification = "B_VOID_HEIGHT_DOMINATED_2D_3D_MISMATCH"
    elif fixed["spearman_rho"] < 0.0 and varied["spearman_rho"] < 0.0:
        classification = "D_BROAD_DIMENSIONAL_FAILURE"
    else:
        classification = "C_PARTIAL_MULTI_PARAMETER_MISMATCH"
    summary = {"schema": "force-localized-analysis-v1", "experiment_fingerprint": _manifest()["experiment_fingerprint"], "planned_pairs": 48, "completed_pairs": len(rows), "excluded_fraction": excluded_fraction, "populations": populations, "fixed_vs_varied": {"delta_spearman_rho": fixed.get("spearman_rho", math.nan) - varied.get("spearman_rho", math.nan), "delta_kendall_tau": fixed.get("kendall_tau", math.nan) - varied.get("kendall_tau", math.nan)}, "strongest_discordant_cases": disagreements[:5], "void_height_findings": {"fixed_values_mm": sorted({float(row["parameters"]["void_height"]) for row in rows if row["arm"] == "FIXED"}), "varied_range_mm": [min((float(row["parameters"]["void_height"]) for row in rows if row["arm"] == "VARIED"), default=math.nan), max((float(row["parameters"]["void_height"]) for row in rows if row["arm"] == "VARIED"), default=math.nan)]}, "primary_analysis_separate_from_exploratory": True, "classification": classification, "advisor": {"status": "UNAVAILABLE"}, "created_at": _now()}
    atomic_write_json(OUTPUT / "analysis_summary.json", _jsonable(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("precommit", "calibration", "smoke", "fea2d", "fea3d", "optix", "interim-optix", "analyze"), required=True)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "precommit":
        result = _manifest()
    elif args.stage == "calibration":
        result = _calibration()
    elif args.stage == "smoke":
        result = _smoke_stage()
    elif args.stage == "fea2d":
        result = _mechanics_stage("fea2d")
    elif args.stage == "fea3d":
        result = _mechanics_stage("fea3d")
    elif args.stage == "optix":
        result = _optix_stage()
    elif args.stage == "interim-optix":
        result = _interim_optix()
    else:
        result = _analyze()
    print(json.dumps(_jsonable({"stage": args.stage, "status": result.get("status", "PASS"), "experiment_fingerprint": result.get("experiment_fingerprint")}), sort_keys=True))
    return 0 if result.get("status", "PASS") not in {"BLOCKED", "INCOMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
