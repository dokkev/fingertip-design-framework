"""Nominal localized-load FEA/VBD correspondence characterization."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from physics import (
    NewtonSession,
    NewtonSettings,
    ParticleLoad,
    prepare_fingertip_mesh,
)
from mesh.volume3d import generate_volume_mesh
from mesh.volume_types import volume_mesh_settings_for_tier
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters
from model.solid import build_fingertip_solid
from validation.common.io import atomic_write_json, strict_read_json

from validation.reference.kratos3d.fea3d_reference import (
    FEA3DReferenceError,
    FEA3DReferenceState,
    load_fea3d_reference,
)


class NewtonCorrespondenceError(ValueError):
    """Raised when the selected FEA state cannot be compared fail-closed."""


REFERENCE_ROOT = Path("output/validation/overnight_force_localized_trend/fea3d")
REFERENCE_CASE_NAME = "base_00_nominal__FIXED__left"
INTERNAL_WALL_TAGS = ("void_left", "void_right")
EXTERNAL_NORMAL_TAGS = ("outer_compliant_arc",)
LOCALIZATION_BINS_MM = (0.0, 4.0, 8.0, 12.0, 16.0, math.inf)
# The current neutral material parameters are intentionally not calibrated in
# this characterization.  The smaller explicit step and additional VBD
# iterations are only the minimal stability settings needed for the selected
# 2 N force-loaded comparison to remain finite.
VBD_CORRESPONDENCE_DT = 1.0 / 50_000.0
VBD_CORRESPONDENCE_ITERATIONS = 20


def _surface_nodes(prepared, tags: tuple[str, ...]) -> np.ndarray:
    triangles = [prepared.surface_triangles[tag] for tag in tags if tag in prepared.surface_triangles]
    if len(triangles) != len(tags):
        missing = sorted(set(tags) - set(prepared.surface_triangles))
        raise NewtonCorrespondenceError(f"required semantic surfaces are missing: {missing}")
    return np.unique(np.concatenate([triangle.reshape(-1) for triangle in triangles])).astype(np.int64)


def _triangles(prepared, tags: tuple[str, ...]) -> np.ndarray:
    triangles = [prepared.surface_triangles[tag] for tag in tags if tag in prepared.surface_triangles]
    if len(triangles) != len(tags):
        missing = sorted(set(tags) - set(prepared.surface_triangles))
        raise NewtonCorrespondenceError(f"required semantic surfaces are missing: {missing}")
    return np.concatenate(triangles, axis=0)


def _triangle_normals(coordinates: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    points = coordinates[triangles]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 1.0e-12):
        raise NewtonCorrespondenceError("comparison surface contains a degenerate triangle")
    return cross / lengths[:, None]


def _angles_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    dots = np.sum(first * second, axis=1)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise NewtonCorrespondenceError("descriptor values must be finite and non-empty")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
    }


def _state_wall_lateral(reference: np.ndarray, displacement: np.ndarray, prepared) -> dict[str, Any]:
    by_surface: dict[str, Any] = {}
    combined: list[np.ndarray] = []
    for tag in INTERNAL_WALL_TAGS:
        indices = _surface_nodes(prepared, (tag,))
        outward_sign = 1.0 if float(np.mean(reference[indices, 0])) > 0.0 else -1.0
        outward = outward_sign * displacement[indices, 0]
        by_surface[tag] = {
            "node_count": int(indices.size),
            "outward_lateral_displacement_mm": _stats(outward),
        }
        combined.append(outward)
    values = np.concatenate(combined)
    return {
        "definition": "signed outward x displacement on semantic void_left/void_right nodes",
        "by_surface": by_surface,
        "combined_outward_lateral_displacement_mm": _stats(values),
    }


def _cavity_width(reference: np.ndarray, deformed: np.ndarray, prepared) -> dict[str, float]:
    left = _surface_nodes(prepared, ("void_left",))
    right = _surface_nodes(prepared, ("void_right",))
    reference_width = float(np.mean(reference[right, 0]) - np.mean(reference[left, 0]))
    deformed_width = float(np.mean(deformed[right, 0]) - np.mean(deformed[left, 0]))
    return {
        "reference_cavity_width_mm": reference_width,
        "deformed_cavity_width_mm": deformed_width,
        "cavity_width_change_mm": deformed_width - reference_width,
    }


def _boundary_rotation(reference: np.ndarray, deformed: np.ndarray, prepared) -> dict[str, Any]:
    triangles = _triangles(prepared, INTERNAL_WALL_TAGS)
    reference_normals = _triangle_normals(reference, triangles)
    deformed_normals = _triangle_normals(deformed, triangles)
    rotation = _angles_deg(reference_normals, deformed_normals)
    return {
        "definition": "angle between reference/deformed normals of semantic internal wall triangles",
        "surface_tags": list(INTERNAL_WALL_TAGS),
        "rotation_deg": _stats(rotation),
    }


def _surface_normal_change(
    reference: np.ndarray,
    fea_deformed: np.ndarray,
    vbd_deformed: np.ndarray,
    prepared,
) -> dict[str, Any]:
    triangles = _triangles(prepared, EXTERNAL_NORMAL_TAGS)
    reference_normals = _triangle_normals(reference, triangles)
    fea_normals = _triangle_normals(fea_deformed, triangles)
    vbd_normals = _triangle_normals(vbd_deformed, triangles)
    fea_change = _angles_deg(reference_normals, fea_normals)
    vbd_change = _angles_deg(reference_normals, vbd_normals)
    model_error = _angles_deg(fea_normals, vbd_normals)
    return {
        "definition": "shared outer_compliant_arc triangle normals; all angles in degrees",
        "surface_tags": list(EXTERNAL_NORMAL_TAGS),
        "fea_change_from_reference_deg": _stats(fea_change),
        "vbd_change_from_reference_deg": _stats(vbd_change),
        "fea_vbd_angular_error_deg": _stats(model_error),
    }


def _localization(
    reference: np.ndarray,
    fea_displacement: np.ndarray,
    vbd_displacement: np.ndarray,
    center_x_mm: float,
    center_z_mm: float,
) -> dict[str, Any]:
    radii = np.hypot(reference[:, 0] - center_x_mm, reference[:, 2] - center_z_mm)
    bins = np.asarray(LOCALIZATION_BINS_MM, dtype=float)
    records: list[dict[str, Any]] = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        selected = (radii >= lower) & (radii < upper)
        if not np.any(selected):
            continue
        fea_magnitude = np.linalg.norm(fea_displacement[selected], axis=1)
        vbd_magnitude = np.linalg.norm(vbd_displacement[selected], axis=1)
        records.append(
            {
                "radius_interval_mm": [float(lower), None if math.isinf(upper) else float(upper)],
                "node_count": int(np.count_nonzero(selected)),
                "fea_mean_displacement_mm": float(np.mean(fea_magnitude)),
                "vbd_mean_displacement_mm": float(np.mean(vbd_magnitude)),
                "absolute_mean_difference_mm": float(abs(np.mean(vbd_magnitude) - np.mean(fea_magnitude))),
            }
        )

    def characteristic(displacement: np.ndarray) -> float | None:
        magnitude = np.linalg.norm(displacement, axis=1)
        total = float(np.sum(magnitude))
        return None if total <= 1.0e-15 else float(np.sum(radii * magnitude) / total)

    fea_characteristic = characteristic(fea_displacement)
    vbd_characteristic = characteristic(vbd_displacement)
    return {
        "definition": "node displacement magnitude binned by reference x-z distance from localized load center",
        "center_mm": [float(center_x_mm), 0.0, float(center_z_mm)],
        "bins": records,
        "fea_characteristic_displacement_radius_mm": fea_characteristic,
        "vbd_characteristic_displacement_radius_mm": vbd_characteristic,
        "characteristic_radius_difference_mm": (
            None
            if fea_characteristic is None or vbd_characteristic is None
            else abs(vbd_characteristic - fea_characteristic)
        ),
    }


def compare_mechanics_states(reference: FEA3DReferenceState, prepared, vbd_result) -> dict[str, Any]:
    reference_coordinates = np.asarray(reference.reference_coordinates_mm, dtype=float)
    fea_deformed = np.asarray(reference.deformed_coordinates_mm, dtype=float)
    fea_displacement = np.asarray(reference.displacement_mm, dtype=float)
    vbd_deformed = np.asarray(vbd_result.deformed_vertices, dtype=float)
    vbd_displacement = np.asarray(vbd_result.displacement, dtype=float)
    if vbd_deformed.shape != fea_deformed.shape:
        raise NewtonCorrespondenceError("FEA and VBD deformed coordinate shapes differ")

    error = vbd_displacement - fea_displacement
    error_norm = np.linalg.norm(error, axis=1)
    fea_norm = np.linalg.norm(fea_displacement, axis=1)
    fea_rms = float(np.sqrt(np.mean(fea_norm * fea_norm)))
    full_field = {
        "displacement_rms_error_mm": float(np.sqrt(np.mean(error_norm * error_norm))),
        "displacement_max_error_mm": float(np.max(error_norm)),
        "relative_displacement_error": None if fea_rms <= 1.0e-12 else float(np.sqrt(np.mean(error_norm * error_norm)) / fea_rms),
        "fea_displacement_rms_mm": fea_rms,
    }
    wall_fea = _state_wall_lateral(reference_coordinates, fea_displacement, prepared)
    wall_vbd = _state_wall_lateral(reference_coordinates, vbd_displacement, prepared)
    cavity_fea = _cavity_width(reference_coordinates, fea_deformed, prepared)
    cavity_vbd = _cavity_width(reference_coordinates, vbd_deformed, prepared)
    rotation_fea = _boundary_rotation(reference_coordinates, fea_deformed, prepared)
    rotation_vbd = _boundary_rotation(reference_coordinates, vbd_deformed, prepared)
    normal_change = _surface_normal_change(
        reference_coordinates,
        fea_deformed,
        vbd_deformed,
        prepared,
    )
    load_metadata = reference.load_metadata.get("load")
    if not isinstance(load_metadata, Mapping):
        raise NewtonCorrespondenceError("selected reference has no localized load metadata")
    localization = _localization(
        reference_coordinates,
        fea_displacement,
        vbd_displacement,
        float(load_metadata["center_x_mm"]),
        float(load_metadata["center_z_mm"]),
    )
    fea_wall_mean = wall_fea["combined_outward_lateral_displacement_mm"]["mean"]
    vbd_wall_mean = wall_vbd["combined_outward_lateral_displacement_mm"]["mean"]
    geometry_relevant = {
        "internal_wall_lateral_displacement": {
            "fea": wall_fea,
            "vbd": wall_vbd,
            "mean_absolute_difference_mm": float(abs(vbd_wall_mean - fea_wall_mean)),
        },
        "cavity_width": {
            "fea": cavity_fea,
            "vbd": cavity_vbd,
            "absolute_delta_width_error_mm": float(
                abs(cavity_vbd["cavity_width_change_mm"] - cavity_fea["cavity_width_change_mm"])
            ),
        },
        "boundary_rotation": {
            "fea": rotation_fea,
            "vbd": rotation_vbd,
            "mean_rotation_difference_deg": float(
                abs(rotation_vbd["rotation_deg"]["mean"] - rotation_fea["rotation_deg"]["mean"])
            ),
        },
        "surface_normal_change": normal_change,
        "deformation_localization": localization,
    }
    return {
        "full_field": full_field,
        "geometry_relevant": geometry_relevant,
        "unsupported_quantities": [
            "stress_field_agreement",
            "reaction_force_equivalence",
            "contact_pressure_or_contact_state",
            "FEA constitutive-law internal variables",
        ],
    }


def verify_exact_mesh_correspondence(volume_mesh, prepared, reference: FEA3DReferenceState) -> dict[str, Any]:
    if reference.source_node_ids is None or reference.tetrahedra_node_ids is None:
        raise NewtonCorrespondenceError("reference lacks explicit node or tetrahedron provenance")
    current_ids = np.asarray(sorted(volume_mesh.nodes), dtype=np.int64)
    current_coordinates = np.asarray(
        [[volume_mesh.nodes[int(node_id)].x_mm, volume_mesh.nodes[int(node_id)].y_mm, volume_mesh.nodes[int(node_id)].z_mm] for node_id in current_ids],
        dtype=float,
    )
    current_tetrahedra = np.asarray([tetrahedron.node_ids for tetrahedron in volume_mesh.tetrahedra], dtype=np.int64)
    source_ids_exact = bool(np.array_equal(current_ids, reference.source_node_ids))
    coordinate_difference = float(np.max(np.abs(current_coordinates - reference.reference_coordinates_mm)))
    coordinate_identity = bool(np.allclose(current_coordinates, reference.reference_coordinates_mm, rtol=0.0, atol=1.0e-10))
    tetrahedra_exact = bool(np.array_equal(current_tetrahedra, reference.tetrahedra_node_ids))
    morphology_exact = bool(volume_mesh.morphology_fingerprint == reference.morphology_fingerprint)
    result = {
        "source_node_ids_exact": source_ids_exact,
        "reference_coordinates_identity": coordinate_identity,
        "reference_coordinate_max_abs_difference_mm": coordinate_difference,
        "tetrahedral_connectivity_exact": tetrahedra_exact,
        "morphology_fingerprint_exact": morphology_exact,
        "node_count": int(current_ids.size),
        "tet_count": int(current_tetrahedra.shape[0]),
    }
    if not all((source_ids_exact, coordinate_identity, tetrahedra_exact, morphology_exact)):
        raise NewtonCorrespondenceError(f"authoritative mesh mismatch: {result}")
    if not np.array_equal(prepared.source_node_ids, current_ids):
        raise NewtonCorrespondenceError("physics adapter changed the authoritative node order")
    return result


def _selected_reference(repo_root: str | Path = ".") -> tuple[dict[str, Any], FEA3DReferenceState, Path]:
    root = Path(repo_root).resolve() / REFERENCE_ROOT
    candidates = []
    for path in sorted(root.glob("*.json")):
        payload = strict_read_json(path)
        if payload.get("schema") != "force-localized-case-contract-v1" or payload.get("status") != "PASS":
            continue
        parameters = payload.get("parameters")
        load = payload.get("load")
        if not isinstance(parameters, Mapping) or not isinstance(load, Mapping):
            continue
        try:
            parsed = FingertipParameters(**parameters)
        except (TypeError, ValueError):
            continue
        if (
            payload.get("side") == "left"
            and payload.get("arm") == "FIXED"
            and float(load.get("target_force_n", -1.0)) == 2.0
            and parsed == FingertipParameters()
            and load.get("load_type") == "localized_normal_surface_pressure"
        ):
            candidates.append((path, payload))
    if len(candidates) != 1:
        raise NewtonCorrespondenceError(
            f"expected one nominal localized reference, found {len(candidates)}"
        )
    case_path, payload = candidates[0]
    raw_manifest = payload.get("native_manifest")
    if not isinstance(raw_manifest, str):
        raise NewtonCorrespondenceError("selected reference has no native manifest")
    manifest = Path(raw_manifest)
    if not manifest.is_absolute():
        manifest = repo_root / manifest
    try:
        reference = load_fea3d_reference(manifest, case_metadata=payload)
    except (FEA3DReferenceError, OSError, ValueError) as exception:
        raise NewtonCorrespondenceError(str(exception)) from exception
    return payload, reference, case_path


def build_localized_particle_load(
    prepared,
    reference: FEA3DReferenceState,
    case_payload: Mapping[str, Any],
) -> tuple[ParticleLoad, dict[str, Any]]:
    force_control = case_payload.get("force_control")
    load_definition = case_payload.get("load")
    if not isinstance(force_control, Mapping) or not isinstance(load_definition, Mapping):
        raise NewtonCorrespondenceError("localized FEA load metadata is incomplete")
    selected = force_control.get("selected_triangles")
    if not isinstance(selected, list) or not selected:
        raise NewtonCorrespondenceError("localized FEA load has no selected triangles")

    source_to_local = {int(source_id): index for index, source_id in enumerate(prepared.source_node_ids)}
    outer_tags = tuple(
        tag
        for tag in prepared.surface_triangles
        if tag in {"outer_compliant_arc", "outer_compliant_left", "outer_compliant_right", "outer_compliant_other"}
    )
    known_triangles = {
        tuple(sorted(int(prepared.source_node_ids[index]) for index in triangle))
        for tag in outer_tags
        for triangle in prepared.surface_triangles[tag]
    }
    coordinates = np.asarray(prepared.tet_mesh.vertices, dtype=float)
    nodal_forces = np.zeros((coordinates.shape[0], 3), dtype=float)
    pressure = float(load_definition["pressure_mpa"])
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise NewtonCorrespondenceError("FEA pressure is not finite and positive")
    loaded_area = 0.0
    reference_resultant = np.zeros(3, dtype=float)
    for row in selected:
        if not isinstance(row, Mapping):
            raise NewtonCorrespondenceError("selected triangle metadata is malformed")
        source_ids = tuple(int(value) for value in row["node_ids"])
        if len(source_ids) != 3 or tuple(sorted(source_ids)) not in known_triangles:
            raise NewtonCorrespondenceError("selected load triangle is not an authoritative outer surface triangle")
        try:
            local = np.asarray([source_to_local[value] for value in source_ids], dtype=np.int64)
        except KeyError as exception:
            raise NewtonCorrespondenceError("selected load triangle references an unknown node") from exception
        points = coordinates[local]
        cross = np.cross(points[1] - points[0], points[2] - points[0])
        norm = float(np.linalg.norm(cross))
        if not math.isfinite(norm) or norm <= 1.0e-12:
            raise NewtonCorrespondenceError("selected load triangle is degenerate")
        area = 0.5 * norm
        centroid = np.mean(points, axis=0)
        inward = -cross / norm
        if not np.allclose(area, float(row["area_mm2"]), rtol=0.0, atol=5.0e-5):
            raise NewtonCorrespondenceError("selected load triangle area differs from FEA artifact")
        if not np.allclose(centroid, np.asarray(row["centroid_mm"], dtype=float), rtol=0.0, atol=5.0e-5):
            raise NewtonCorrespondenceError("selected load triangle centroid differs from FEA artifact")
        if not np.allclose(inward, np.asarray(row["inward_normal"], dtype=float), rtol=0.0, atol=5.0e-6):
            raise NewtonCorrespondenceError("selected load triangle normal differs from FEA artifact")
        profile = float(row["profile_weight"])
        face_force = pressure * profile * area * inward
        reference_resultant += face_force
        loaded_area += area
        for index in local:
            nodal_forces[int(index)] += face_force / 3.0

    selected_indices = np.flatnonzero(np.linalg.norm(nodal_forces, axis=1) > 0.0).astype(np.int32)
    particle_load = ParticleLoad(
        vertex_indices=selected_indices,
        forces_n=nodal_forces[selected_indices],
        load_steps=int(case_payload.get("steps", 1)),
    )
    expected_resultant = float(force_control["achieved_discrete_force_n"])
    construction = {
        "semantic_surface_tags": list(outer_tags),
        "selected_triangle_count": len(selected),
        "unique_loaded_vertex_count": int(selected_indices.size),
        "pressure_mpa": pressure,
        "loaded_area_mm2": loaded_area,
        "face_force_rule": "pressure_mpa [N/mm2] * profile_weight * triangle_area_mm2, equal-lumped /3 to each vertex",
        "fea_resultant_n": reference_resultant.tolist(),
        "fea_resultant_magnitude_n": float(np.linalg.norm(reference_resultant)),
        "artifact_resultant_magnitude_n": expected_resultant,
        "resultant_magnitude_error_n": float(abs(np.linalg.norm(reference_resultant) - expected_resultant)),
        "particle_load_resultant_n": particle_load.resultant_force_n.tolist(),
        "particle_load_resultant_magnitude_n": float(np.linalg.norm(particle_load.resultant_force_n)),
        "load_steps": particle_load.load_steps,
        "center_x_mm": float(load_definition["center_x_mm"]),
        "center_z_mm": float(load_definition["center_z_mm"]),
        "radius_mm": float(load_definition["radius_mm"]),
        "profile": str(load_definition["profile"]),
        "orientation": str(load_definition["orientation"]),
    }
    if construction["resultant_magnitude_error_n"] > 1.0e-5:
        raise NewtonCorrespondenceError("discrete VBD resultant does not match FEA artifact resultant")
    return particle_load, construction


def run_nominal_correspondence(
    *,
    repo_root: str | Path = ".",
    warm_repeats: int = 5,
) -> dict[str, Any]:
    if warm_repeats < 5:
        raise ValueError("warm_repeats must be at least 5")
    root = Path(repo_root).resolve()
    case_payload, reference, case_path = _selected_reference(root)
    parameters = FingertipParameters(**case_payload["parameters"])
    model = FingertipModel(parameters)
    mesh_tier = str(case_payload["mesh"]["tier"])
    volume_mesh = generate_volume_mesh(
        build_fingertip_solid(model),
        volume_mesh_settings_for_tier(mesh_tier),
    )
    prepared = prepare_fingertip_mesh(volume_mesh)
    correspondence = verify_exact_mesh_correspondence(volume_mesh, prepared, reference)
    particle_load, load_construction = build_localized_particle_load(prepared, reference, case_payload)

    settings = NewtonSettings(
        device="cuda:0",
        gravity=0.0,
        dt=VBD_CORRESPONDENCE_DT,
        steps=particle_load.load_steps,
        iterations=VBD_CORRESPONDENCE_ITERATIONS,
        fixed_vertex_indices=prepared.support_vertex_indices,
    )
    session = NewtonSession(prepared.tet_mesh, settings)
    session.solve(particle_load)  # untimed warm-up
    timed_results = []
    timings = []
    for _ in range(warm_repeats):
        result, timing = session.solve_with_timing(particle_load)
        timed_results.append(result)
        timings.append(timing)
    final_result = timed_results[-1]
    baseline = timed_results[0].deformed_vertices
    deterministic_max_error = max(
        float(np.max(np.abs(result.deformed_vertices - baseline))) for result in timed_results[1:]
    )
    solve_times = np.asarray([float(item["per_solve_wall_s"]) for item in timings], dtype=float)
    session_creation = session.session_creation_wall_s
    comparison = compare_mechanics_states(reference, prepared, final_result)
    return {
        "schema": "physics-nominal-fea-vbd-correspondence-v1",
        "scientific_role": "first localized-load correspondence characterization; not a calibrated VBD fidelity claim",
        "fea_rerun": False,
        "optix_run": False,
        "reference": {
            "case_artifact_path": str(case_path),
            "native_manifest_path": str(reference.source_path),
            "native_state_path": reference.provenance["state_path"],
            "primary_schema": case_payload["schema"],
            "native_schema": reference.provenance["artifact_schema"],
            "morphology_fingerprint": reference.morphology_fingerprint,
            "node_count": reference.node_count,
            "tet_count": int(reference.tetrahedra_node_ids.shape[0]) if reference.tetrahedra_node_ids is not None else None,
            "parameters": case_payload["parameters"],
            "mesh": case_payload["mesh"],
            "load": case_payload["load"],
            "force_control": {
                key: value
                for key, value in case_payload["force_control"].items()
                if key != "selected_triangles"
            },
            "reaction_force_n": case_payload.get("reaction_force_n"),
            "fea_timing": case_payload.get("timing"),
            "deformation_provenance": dict(reference.provenance),
        },
        "correspondence": correspondence,
        "load_construction": load_construction,
        "vbd": {
            "settings": asdict(settings),
            "material_parameters": {
                "density": settings.density,
                "k_mu": settings.k_mu,
                "k_lambda": settings.k_lambda,
                "k_damp": settings.k_damp,
            },
            "fixed_support_vertex_count": len(prepared.support_vertex_indices),
            "particle_load_vertex_count": len(particle_load.vertex_indices),
            "model_build_wall_s": float(session.model_build_wall_s),
            "session_creation_wall_s": float(session_creation),
            "warm_up": True,
            "warm_repeats": warm_repeats,
            "warm_solve_median_s": float(np.median(solve_times)),
            "warm_solve_min_s": float(np.min(solve_times)),
            "warm_solve_max_s": float(np.max(solve_times)),
            "solves_per_second": float(1.0 / np.median(solve_times)),
            "per_solve_timings": timings,
            "repeated_result_max_abs_difference_mm": deterministic_max_error,
        },
        "comparison": comparison,
    }


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    correspondence = result["correspondence"]
    full_field = result["comparison"]["full_field"]
    geometry = result["comparison"]["geometry_relevant"]
    vbd = result["vbd"]
    lines = [
        "# Nominal FEA/VBD correspondence characterization",
        "",
        "## NUMERICAL CORRESPONDENCE",
        "",
        f"- Reference: `{result['reference']['case_artifact_path']}`",
        f"- Mesh: `{correspondence['node_count']}` nodes / `{correspondence['tet_count']}` tetrahedra",
        f"- Exact node IDs: `{correspondence['source_node_ids_exact']}`",
        f"- Reference coordinates identity: `{correspondence['reference_coordinates_identity']}` (max `{correspondence['reference_coordinate_max_abs_difference_mm']:.3g}` mm)",
        f"- Exact tetra connectivity: `{correspondence['tetrahedral_connectivity_exact']}`",
        f"- FEA load resultant: `{result['load_construction']['fea_resultant_magnitude_n']:.6g} N`; VBD nodal resultant: `{result['load_construction']['particle_load_resultant_magnitude_n']:.6g} N`",
        f"- Persistent model/session creation: `{vbd['model_build_wall_s']:.6f} s`",
        f"- Warm solve median/min/max: `{vbd['warm_solve_median_s']:.6f}` / `{vbd['warm_solve_min_s']:.6f}` / `{vbd['warm_solve_max_s']} s`",
        f"- Warm solves per second: `{vbd['solves_per_second']:.6g}`",
        f"- Repeated-result max difference: `{vbd['repeated_result_max_abs_difference_mm']:.3g} mm`",
        "",
        "## SCIENTIFIC INTERPRETATION",
        "",
        f"- Full-field displacement RMS/max error: `{full_field['displacement_rms_error_mm']:.6g}` / `{full_field['displacement_max_error_mm']:.6g} mm`",
        f"- Internal wall mean lateral error: `{geometry['internal_wall_lateral_displacement']['mean_absolute_difference_mm']:.6g} mm`",
        f"- Cavity delta-width error: `{geometry['cavity_width']['absolute_delta_width_error_mm']:.6g} mm`",
        f"- Internal boundary mean rotation difference: `{geometry['boundary_rotation']['mean_rotation_difference_deg']:.6g} deg`",
        f"- Outer surface-normal mean/RMS/max angular error: `{geometry['surface_normal_change']['fea_vbd_angular_error_deg']['mean']:.6g}` / `{geometry['surface_normal_change']['fea_vbd_angular_error_deg']['rms']:.6g}` / `{geometry['surface_normal_change']['fea_vbd_angular_error_deg']['maximum']:.6g} deg`",
        "- This is characterization only; stress, reaction-force equivalence, contact state, and calibrated physical fidelity remain unsupported.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/validation/physics/nominal_fea_vbd_correspondence.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("output/validation/physics/nominal_fea_vbd_correspondence.md"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--warm-repeats", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        result = run_nominal_correspondence(repo_root=args.repo_root, warm_repeats=args.warm_repeats)
    except Exception as exception:
        print(f"FAIL: nominal_fea_vbd_correspondence: {exception}")
        return 1
    atomic_write_json(args.output, result)
    _write_report(args.report, result)
    print(
        "PASS: nominal FEA/VBD correspondence "
        f"nodes={result['correspondence']['node_count']} "
        f"tets={result['correspondence']['tet_count']} "
        f"warm_solve_median={result['vbd']['warm_solve_median_s']:.6f}s"
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NewtonCorrespondenceError",
    "VBD_CORRESPONDENCE_DT",
    "VBD_CORRESPONDENCE_ITERATIONS",
    "build_localized_particle_load",
    "compare_mechanics_states",
    "main",
    "run_nominal_correspondence",
    "verify_exact_mesh_correspondence",
]
