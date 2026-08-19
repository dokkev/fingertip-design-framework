"""Matched VBD/FEA full-3D optical trend validation.

This module is deliberately validation-owned.  It loads persisted nonlinear
FEA states, solves the corresponding exact meshes with the already-frozen
mechanics3d session, and sends both deformed surfaces through the one shared
full-3D OptiX transport implementation.  It never runs Kratos FEA.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

from mechanics3d import (
    Mechanics3DSession,
    Mechanics3DSettings,
    prepare_fingertip_mechanics_mesh,
)
from mesh.volume3d import generate_volume_mesh
from mesh.volume_types import volume_mesh_settings_for_tier
from model.fingertip import Fingertip
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters
from model.solid import build_fingertip_solid
from optics.transport3d import (
    OptiXTransport,
    UnifiedTransportResult,
    build_full3d_transport_geometry,
    fingerprint_mapping,
    load_case_artifact,
    load_full3d_surface_artifact,
    native_field_separability,
    save_case_artifact,
    transport_configuration,
)
from optics.transport3d.geometry import TriangleSurface
from optics.transport3d.optix_backend import create_runtime
from optics.transport3d.settings import Transport3DSettings
from validation.common.io import atomic_write_json, strict_read_json
from validation.common.provenance import sha256_file

from .correspondence import (
    VBD_CORRESPONDENCE_DT,
    VBD_CORRESPONDENCE_ITERATIONS,
    build_localized_particle_load,
    compare_mechanics_states,
    verify_exact_mesh_correspondence,
)
from .fea3d_reference import load_fea3d_reference


REFERENCE_ROOT = Path("output/validation/overnight_force_localized_trend/fea3d")
OUTPUT_ROOT = Path("output/validation/mechanics3d")
OPTIX_ROOT = OUTPUT_ROOT / "vbd_fea_optical_optix"
TREND_JSON = OUTPUT_ROOT / "vbd_fea_optical_trend.json"
TREND_MD = OUTPUT_ROOT / "vbd_fea_optical_trend.md"
TREND_CSV = OUTPUT_ROOT / "vbd_fea_optical_ranking.csv"
PROGRESS_JSON = OUTPUT_ROOT / "vbd_fea_optical_trend_progress.json"

SCHEMA = "vbd-fea-optical-trend-v1"
FEA_CASE_SCHEMA = "force-localized-case-contract-v1"
VBD_STEPS = 12
RAY_COUNT = 1024
GRID_WIDTH = 48
GRID_HEIGHT = 48
GRID_Z_BINS = 16
RADIUS_MM = 4.0
PAIR_TIE_TOLERANCE = 1.0e-12


class TrendValidationError(RuntimeError):
    """Raised when the comparison cannot be completed fail-closed."""


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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_path(raw: str | Path, *, relative_to: Path | None = None) -> Path:
    path = Path(raw)
    if path.is_absolute() and path.exists():
        return path.resolve()
    candidates = [path]
    if relative_to is not None:
        candidates.extend((relative_to / path, relative_to / path.name))
    candidates.extend((Path.cwd() / path, Path.cwd() / path.name))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise TrendValidationError(f"persisted artifact does not exist: {raw}")


def _git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "revision": run("rev-parse", "HEAD"),
        "worktree_status": status,
        "worktree_fingerprint": None if status is None else _fingerprint(status),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_case_rows(root: Path = REFERENCE_ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        payload = strict_read_json(path)
        if payload.get("schema") != FEA_CASE_SCHEMA or payload.get("status") != "PASS":
            continue
        required = ("case_id", "base_id", "arm", "side", "parameters", "morphology_fingerprint", "native_manifest")
        if any(field not in payload for field in required):
            raise TrendValidationError(f"FEA case is missing provenance fields: {path}")
        load = payload.get("load")
        force = payload.get("force_control")
        if not isinstance(load, Mapping) or not isinstance(force, Mapping):
            raise TrendValidationError(f"FEA case has incomplete load provenance: {path}")
        if not isinstance(payload["parameters"], Mapping):
            raise TrendValidationError(f"FEA case parameters are malformed: {path}")
        native_manifest = _resolve_path(str(payload["native_manifest"]), relative_to=path.parent)
        native_payload = strict_read_json(native_manifest)
        if native_payload.get("schema") != "native-3d-fea-state-v1":
            raise TrendValidationError(f"FEA native artifact schema is not exact: {native_manifest}")
        target_force = float(load.get("target_force_n", force.get("achieved_discrete_force_n", float("nan"))))
        if not np.isfinite(target_force):
            raise TrendValidationError(f"FEA case target force is not finite: {path}")
        rows.append(
            {
                "case_path": path.resolve(),
                "case_id": str(payload["case_id"]),
                "base_id": str(payload["base_id"]),
                "arm": str(payload["arm"]),
                "side": str(payload["side"]),
                "parameters": dict(payload["parameters"]),
                "morphology_id": f"{payload['base_id']}__{payload['arm']}",
                "morphology_fingerprint": str(payload["morphology_fingerprint"]),
                "native_manifest": native_manifest,
                "native_manifest_sha256": sha256_file(native_manifest),
                "load": dict(load),
                "force_control": dict(force),
                "target_force_n": target_force,
                "radius_mm": float(load.get("radius_mm", float("nan"))),
                "profile": str(load.get("profile", "")),
                "orientation": str(load.get("orientation", "")),
                "center_x_mm": float(load.get("center_x_mm", float("nan"))),
                "center_z_mm": float(load.get("center_z_mm", float("nan"))),
                "case_payload": payload,
            }
        )
    if not rows:
        raise TrendValidationError(f"no successful FEA3D states found under {root}")
    return rows


def _scenario_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["arm"],
        round(float(row["target_force_n"]), 10),
        round(float(row["radius_mm"]), 10),
        row["profile"],
        row["orientation"],
        round(float(row["center_z_mm"]), 10),
    )


def discover_homogeneous_groups(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return all tied largest homogeneous morphology strata.

    Left and right states are paired within each morphology.  A group never
    combines different arm or load/profile strata.
    """
    buckets: dict[tuple[Any, ...], dict[str, dict[str, Mapping[str, Any]]]] = {}
    for row in rows:
        bucket = buckets.setdefault(_scenario_key(row), {})
        morphology = bucket.setdefault(str(row["morphology_id"]), {})
        side = str(row["side"])
        if side in morphology:
            raise TrendValidationError(f"duplicate FEA side state for {row['case_id']}")
        morphology[side] = row
    groups: list[dict[str, Any]] = []
    for key, candidates in buckets.items():
        complete = {morphology: sides for morphology, sides in candidates.items() if set(sides) == {"left", "right"}}
        if len(complete) < 2:
            continue
        groups.append(
            {
                "key": key,
                "scenario_id": (
                    f"{key[0]}__force_{key[1]:g}N__radius_{key[2]:g}mm__"
                    f"{key[3]}__{key[4]}__z_{key[5]:g}mm"
                ),
                "arm": str(key[0]),
                "candidate_count": len(complete),
                "candidates": complete,
            }
        )
    if not groups:
        raise TrendValidationError("no homogeneous multi-morphology FEA scenario exists")
    largest = max(group["candidate_count"] for group in groups)
    return [group for group in groups if group["candidate_count"] == largest]


def _vbd_settings(prepared: Any) -> Mechanics3DSettings:
    return Mechanics3DSettings(
        device="cuda:0",
        gravity=0.0,
        dt=VBD_CORRESPONDENCE_DT,
        steps=VBD_STEPS,
        iterations=VBD_CORRESPONDENCE_ITERATIONS,
        fixed_vertex_indices=prepared.support_vertex_indices,
    )


def _normal_from_surface(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    points = vertices[np.asarray(faces, dtype=np.int64)]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 1.0e-12):
        raise TrendValidationError("VBD optical surface contains a degenerate triangle")
    return cross / lengths[:, None]


def _vbd_geometry(
    tip: Any,
    prepared: Any,
    result: Any,
    fea_artifact: Any,
    *,
    side: str,
    vbd_state_fingerprint: str,
) -> Any:
    source_to_local = {
        int(source_id): index
        for index, source_id in enumerate(np.asarray(prepared.source_node_ids, dtype=np.int64))
    }
    try:
        local = np.asarray(
            [source_to_local[int(source_id)] for source_id in fea_artifact.silicone_node_ids],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise TrendValidationError("VBD surface topology references an unknown source node") from exc
    vertices = np.asarray(result.deformed_vertices, dtype=float)[local]
    silicone = fea_artifact.silicone
    vbd_silicone = TriangleSurface(
        vertices=vertices,
        faces=silicone.faces,
        normals=_normal_from_surface(vertices, silicone.faces),
        external_surface=silicone.external_surface,
        u_start=silicone.u_start,
        u_end=silicone.u_end,
        semantic_tags=silicone.semantic_tags,
        interface_tags=silicone.interface_tags,
    )
    return build_full3d_transport_geometry(
        tip,
        silicone=vbd_silicone,
        # The rigid carrier and periodic envelope are unchanged between the
        # branches.  Only the compliant direct surface is replaced by VBD.
        rigid=fea_artifact.rigid,
        envelope=fea_artifact.envelope,
        source_position_mm=fea_artifact.source_position_mm,
        source_medium=fea_artifact.source_medium,
        metadata={
            "morphology_id": fea_artifact.morphology_id,
            "morphology_fingerprint": fea_artifact.morphology_fingerprint,
            "contact_state_fingerprint": vbd_state_fingerprint,
            "mechanics_source": "mechanics3d.Mechanics3DSession",
            "vbd_side": side,
            "vbd_surface_source": "direct VBD deformed_vertices on FEA silicone topology",
            "full3d_surface_provenance": "actual_deformed_3d_vbd_surface",
        },
    )


def _bounds(geometries: Sequence[Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    points = [
        np.asarray(surface.vertices, dtype=float)
        for geometry in geometries
        for surface in (geometry.silicone, geometry.rigid, geometry.envelope)
    ]
    combined = np.concatenate(points, axis=0)
    spans = np.ptp(combined[:, :2], axis=0)
    margin = 0.04 * float(max(spans))
    return (
        float(np.min(combined[:, 0]) - margin),
        float(np.max(combined[:, 0]) + margin),
    ), (
        float(np.min(combined[:, 1]) - margin),
        float(np.max(combined[:, 1]) + margin),
    )


def _optix_settings(bounds: tuple[tuple[float, float], tuple[float, float]]) -> Transport3DSettings:
    return Transport3DSettings(
        mode="full3d",
        ray_count=RAY_COUNT,
        max_interactions=10,
        minimum_ray_weight=1.0e-4,
        maximum_segment_count=max(24000, 24 * RAY_COUNT),
        maximum_periodic_wraps=32,
        terminate_on_periodic_wrap_limit=True,
        terminate_on_no_event=True,
        extrusion_depth_mm=11.0,
        internal_grid_width=GRID_WIDTH,
        internal_grid_height=GRID_HEIGHT,
        internal_z_bins=GRID_Z_BINS,
        projected_grid_width=GRID_WIDTH,
        projected_grid_height=GRID_HEIGHT,
        x_bounds_mm=bounds[0],
        y_bounds_mm=bounds[1],
        retain_projected_segments=False,
        retain_internal_path_field=True,
    )


def _material(tip: Any) -> dict[str, Any]:
    return {
        "refractive_index_air": tip.optical.refractive_index_air,
        "refractive_index_silicone": tip.optical.refractive_index_silicone,
        "absorption_per_mm": tip.optical.absorption_per_mm,
        "scattering_per_mm": tip.optical.scattering_per_mm,
    }


def artifact_contract_is_exact(metadata: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Check the reusable-artifact contract before attempting field loading."""
    return bool(
        metadata.get("schema") in {
            "unified-optix-transport-case-v3",
            "unified-optix-transport-case-v2",
            "unified-optix-transport-case-v1",
        }
        and metadata.get("contract") == dict(expected)
        and metadata.get("contract_fingerprint") == fingerprint_mapping(dict(expected))
    )


def _trace_or_reuse(
    *,
    path: Path,
    expected_contract: Mapping[str, Any],
    tip: Any,
    geometry: Any,
    settings: Transport3DSettings,
    morphology_id: str,
    morphology_fingerprint: str,
    mechanics_source: str,
    contact_state: Mapping[str, Any],
    transport_config: Mapping[str, Any],
    runtime: Any,
    tracer: OptiXTransport,
    reuse_paths: Sequence[Path] = (),
) -> tuple[UnifiedTransportResult, bool, float, Path]:
    for candidate_path in (path, *reuse_paths):
        if not candidate_path.exists():
            continue
        try:
            metadata = strict_read_json(candidate_path)
            if artifact_contract_is_exact(metadata, expected_contract):
                started = time.perf_counter()
                result = load_case_artifact(candidate_path, expected_contract=expected_contract)
                return result, True, time.perf_counter() - started, candidate_path
        except (OSError, ValueError, TypeError):
            pass
    started = time.perf_counter()
    result = tracer.trace(
        tip,
        geometry,
        settings=settings,
        morphology_id=morphology_id,
        morphology_fingerprint=morphology_fingerprint,
        mechanics_source=mechanics_source,
        mechanics_dimension="3D",
        contact_state=contact_state,
        transport_configuration=transport_config,
        runtime=runtime,
    )
    save_case_artifact(path, result, expected_contract)
    return result, False, time.perf_counter() - started, path


def _prepare_candidate(group: Mapping[str, Any], morphology_id: str) -> dict[str, Any]:
    sides = group["candidates"][morphology_id]
    params = FingertipParameters(**sides["left"]["parameters"])
    tip = Fingertip(params)
    model = FingertipModel(params)
    volume_mesh = generate_volume_mesh(
        build_fingertip_solid(model),
        volume_mesh_settings_for_tier("search"),
    )
    prepared = prepare_fingertip_mechanics_mesh(volume_mesh)
    state: dict[str, Any] = {
        "morphology_id": morphology_id,
        "parameters": dict(sides["left"]["parameters"]),
        "morphology_fingerprint": str(sides["left"]["morphology_fingerprint"]),
        "tip": tip,
        "volume_mesh": volume_mesh,
        "prepared": prepared,
        "sides": {},
    }
    for side in ("left", "right"):
        case = sides[side]
        if case["morphology_fingerprint"] != state["morphology_fingerprint"]:
            raise TrendValidationError(f"left/right morphology fingerprint mismatch: {morphology_id}")
        if dict(case["parameters"]) != state["parameters"]:
            raise TrendValidationError(f"left/right morphology parameters mismatch: {morphology_id}")
        reference = load_fea3d_reference(case["native_manifest"], case_metadata=case["case_payload"])
        correspondence = verify_exact_mesh_correspondence(volume_mesh, prepared, reference)
        native_manifest_payload = strict_read_json(case["native_manifest"])
        contact_fp = native_manifest_payload.get("contact_state_fingerprint")
        if not isinstance(contact_fp, str) or not contact_fp:
            raise TrendValidationError(f"FEA contact-state fingerprint is missing: {case['case_id']}")
        artifact = load_full3d_surface_artifact(
            case["native_manifest"],
            expected_morphology_fingerprint=state["morphology_fingerprint"],
            expected_contact_state_fingerprint=contact_fp,
            repair_derived_normals=True,
        )
        if artifact.mesh_fingerprint != case["case_payload"].get("mesh", {}).get("fingerprint", artifact.mesh_fingerprint):
            # The case contract historically stores the authoritative mesh
            # fingerprint in the native manifest; the loader has already
            # checked that exact value.  Do not reject older case summaries.
            pass
        particle_load, load_construction = build_localized_particle_load(prepared, reference, case["case_payload"])
        state["sides"][side] = {
            "case": case,
            "reference": reference,
            "artifact": artifact,
            "correspondence": correspondence,
            "particle_load": particle_load,
            "load_construction": load_construction,
        }
    return state


def _score_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "raw_l1",
        "raw_l2",
        "normalized_redistribution_l1",
        "first_native_field_mass",
        "second_native_field_mass",
        "first_total_transport",
        "second_total_transport",
        "total_transport_difference",
        "normalized_status",
    )
    return {key: result.get(key) for key in keys}


def _ranked_rows(rows: Sequence[Mapping[str, Any]], *, value_key: str, direction: str) -> list[dict[str, Any]]:
    if direction not in {"maximize", "minimize"}:
        raise ValueError(f"unsupported objective direction: {direction!r}")
    values = np.asarray([float(row[value_key]) for row in rows], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("ranking values must be finite")
    order = sorted(
        range(len(rows)),
        key=lambda index: (
            -values[index] if direction == "maximize" else values[index],
            str(rows[index]["morphology_id"]),
        ),
    )
    result = [dict(rows[index]) for index in order]
    for rank, row in enumerate(result, start=1):
        row[f"rank_{value_key}"] = rank
    return result


def _pairwise_stats(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    concordant = discordant = tied = 0
    for i, j in itertools.combinations(range(len(first)), 2):
        d_first = float(first[i] - first[j])
        d_second = float(second[i] - second[j])
        if abs(d_first) <= PAIR_TIE_TOLERANCE or abs(d_second) <= PAIR_TIE_TOLERANCE:
            tied += 1
        elif d_first * d_second > 0.0:
            concordant += 1
        else:
            discordant += 1
    comparable = concordant + discordant
    return {
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "tied_pairs": tied,
        "total_pairs": len(first) * (len(first) - 1) // 2,
        "pairwise_ordering_agreement": None if comparable == 0 else concordant / comparable,
    }


def rank_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    vbd_key: str = "J_VBD",
    fea_key: str = "J_FEA",
    direction: str = "maximize",
) -> dict[str, Any]:
    """Compute tie-aware selection statistics without changing score values."""
    if len(rows) < 2:
        raise ValueError("at least two morphology rows are required")
    if direction not in {"maximize", "minimize"}:
        raise ValueError(f"unsupported objective direction: {direction!r}")
    vbd = np.asarray([float(row[vbd_key]) for row in rows], dtype=float)
    fea = np.asarray([float(row[fea_key]) for row in rows], dtype=float)
    if not np.all(np.isfinite(vbd)) or not np.all(np.isfinite(fea)):
        raise ValueError("ranking values must be finite")
    vbd_rank = rankdata(-vbd if direction == "maximize" else vbd, method="average")
    fea_rank = rankdata(-fea if direction == "maximize" else fea, method="average")
    vbd_order = sorted(range(len(rows)), key=lambda i: (float(vbd_rank[i]), str(rows[i]["morphology_id"])))
    fea_order = sorted(range(len(rows)), key=lambda i: (float(fea_rank[i]), str(rows[i]["morphology_id"])))
    top_k: dict[str, Any] = {}
    for k in (3, 5):
        if len(rows) < k:
            top_k[f"top_{k}"] = {"status": "not_meaningful", "n": len(rows)}
            continue
        vbd_ids = {str(rows[i]["morphology_id"]) for i in vbd_order[:k]}
        fea_ids = {str(rows[i]["morphology_id"]) for i in fea_order[:k]}
        top_k[f"top_{k}"] = {
            "k": k,
            "vbd_ids": sorted(vbd_ids),
            "fea_ids": sorted(fea_ids),
            "intersection_count": len(vbd_ids & fea_ids),
            "overlap_fraction_of_k": len(vbd_ids & fea_ids) / k,
        }
    pairwise = _pairwise_stats(vbd, fea)
    return {
        "n": len(rows),
        "direction": direction,
        "spearman_rho": float(spearmanr(vbd, fea).statistic),
        "kendall_tau": float(kendalltau(vbd, fea).statistic),
        "vbd_tie_count": int(len(rows) - len(np.unique(vbd))),
        "fea_tie_count": int(len(rows) - len(np.unique(fea))),
        **pairwise,
        "top_k": top_k,
        "vbd_order": [str(rows[i]["morphology_id"]) for i in vbd_order],
        "fea_order": [str(rows[i]["morphology_id"]) for i in fea_order],
        "vbd_ranks": {str(rows[i]["morphology_id"]): float(vbd_rank[i]) for i in range(len(rows))},
        "fea_ranks": {str(rows[i]["morphology_id"]): float(fea_rank[i]) for i in range(len(rows))},
    }


def selection_summary(rows: Sequence[Mapping[str, Any]], *, direction: str = "maximize") -> dict[str, Any]:
    """Return sign-safe VBD-best/FEA-best and regret quantities."""
    stats = rank_statistics(rows, direction=direction)
    vbd_best_id = stats["vbd_order"][0]
    fea_best_id = stats["fea_order"][0]
    by_id = {str(row["morphology_id"]): row for row in rows}
    fea_values = np.asarray([float(row["J_FEA"]) for row in rows], dtype=float)
    selected_fea = float(by_id[vbd_best_id]["J_FEA"])
    best_fea = float(by_id[fea_best_id]["J_FEA"])
    if direction == "maximize":
        regret = best_fea - selected_fea
    elif direction == "minimize":
        regret = selected_fea - best_fea
    else:
        raise ValueError(f"unsupported objective direction: {direction!r}")
    score_range = float(np.max(fea_values) - np.min(fea_values))
    return {
        "vbd_best_morphology": vbd_best_id,
        "fea_best_morphology": fea_best_id,
        "fea_rank_of_vbd_best": stats["fea_ranks"][vbd_best_id],
        "fea_percentile_of_vbd_best": (
            None
            if len(rows) == 1
            else 100.0 * (len(rows) - stats["fea_ranks"][vbd_best_id]) / (len(rows) - 1)
        ),
        "J_FEA_of_vbd_best": selected_fea,
        "J_FEA_best": best_fea,
        "fea_regret": float(regret),
        "normalized_fea_regret": None if score_range <= PAIR_TIE_TOLERANCE else float(regret / score_range),
        "fea_score_range": score_range,
        "statistics": stats,
    }


def _local_neighborhood(rows: Sequence[Mapping[str, Any]], selected_id: str) -> dict[str, Any]:
    if len(rows) < 3:
        return {"status": "insufficient_candidates"}
    names = tuple(sorted(str(name) for name in rows[0]["parameters"]))
    matrix = np.asarray([[float(row["parameters"][name]) for name in names] for row in rows], dtype=float)
    spans = np.ptp(matrix, axis=0)
    spans[spans <= 1.0e-12] = 1.0
    normalized = matrix / spans
    selected_index = next(index for index, row in enumerate(rows) if row["morphology_id"] == selected_id)
    distances = np.linalg.norm(normalized - normalized[selected_index], axis=1)
    order = [index for index in np.argsort(distances) if index != selected_index][: min(3, len(rows) - 1)]
    return {
        "status": "PASS",
        "parameter_names": list(names),
        "selected_morphology": selected_id,
        "neighbors": [
            {
                "morphology_id": str(rows[index]["morphology_id"]),
                "normalized_parameter_distance": float(distances[index]),
                "J_VBD": float(rows[index]["J_VBD"]),
                "J_FEA": float(rows[index]["J_FEA"]),
            }
            for index in order
        ],
    }


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scenario_id", "morphology_id", "morphology_fingerprint", "J_VBD", "J_FEA",
        "rank_J_VBD", "rank_J_FEA", "VBD_score_gap_from_best", "FEA_score_gap_from_best",
        "FEA_left_total_transport", "FEA_right_total_transport",
        "VBD_left_total_transport", "VBD_right_total_transport",
        "FEA_normalized_redistribution", "VBD_normalized_redistribution",
        "mechanics_left_displacement_rms_error_mm", "mechanics_right_displacement_rms_error_mm",
    ]
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _markdown_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# VBD → FEA full-3D optical trend validation",
        "",
        f"- Status: `{result['status']}`",
        f"- Created: `{result['created_at']}`",
        f"- FEA rerun: `{result['provenance']['fea']['rerun']}`",
        "",
        "## Candidate/scenario set",
        "",
        "The largest homogeneous strata are evaluated separately. The established\n"
        "full-3D `J3` scalar is used for selection; production 2D `minimum_auc`\n"
        "requires a 12-trajectory captured-depth study and is not silently\n"
        "substituted into this single localized-load corpus.",
    ]
    for group in result["groups"]:
        lines.extend(
            [
                "",
                f"### `{group['scenario_id']}`",
                "",
                f"- Candidate count: `{group['candidate_count']}`",
                f"- Morphologies: `{', '.join(group['morphology_ids'])}`",
                f"- Spearman rho: `{group['ranking']['spearman_rho']:.6g}`",
                f"- Kendall tau: `{group['ranking']['kendall_tau']:.6g}`",
                f"- Pairwise ordering agreement: `{group['ranking']['pairwise_ordering_agreement']}`",
                f"- VBD-best: `{group['selection']['vbd_best_morphology']}`",
                f"- FEA-best: `{group['selection']['fea_best_morphology']}`",
                f"- FEA rank of VBD-best: `{group['selection']['fea_rank_of_vbd_best']}`",
                f"- FEA percentile of VBD-best: `{group['selection']['fea_percentile_of_vbd_best']}`",
                f"- Raw FEA regret: `{group['selection']['fea_regret']}`",
                f"- Normalized FEA regret: `{group['selection']['normalized_fea_regret']}`",
            ]
        )
        lines.extend(
            [
                "",
                "| Morphology | J_VBD | J_FEA | VBD rank | FEA rank | FEA gap | VBD gap |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in group["rows"]:
            lines.append(
                f"| `{row['morphology_id']}` | {row['J_VBD']:.8g} | {row['J_FEA']:.8g} | "
                f"{row['rank_J_VBD']} | {row['rank_J_FEA']} | {row['FEA_score_gap_from_best']:.8g} | "
                f"{row['VBD_score_gap_from_best']:.8g} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Absolute mechanics differences are diagnostic only. Total outgoing\n"
            "transport and normalized spatial redistribution are reported as\n"
            "separate quantities in the JSON/CSV rows. No post-hoc pass threshold\n"
            "is applied.",
            "",
            "The result is a characterization of whether this limited homogeneous\n"
            "candidate set supports VBD as a design-selection surrogate; it is not\n"
            "a claim that VBD reproduces nonlinear FEA displacement fields.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_comparison(
    *,
    reference_root: Path = REFERENCE_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    """Run the complete saved-FEA/VBD/full-3D-OptiX comparison."""
    started_total = time.perf_counter()
    rows = _load_case_rows(reference_root)
    groups = discover_homogeneous_groups(rows)
    prepared_groups: list[dict[str, Any]] = []
    all_geometries: list[Any] = []
    for group in groups:
        prepared_candidates = []
        for morphology_id in sorted(group["candidates"]):
            state = _prepare_candidate(group, morphology_id)
            settings = _vbd_settings(state["prepared"])
            session = Mechanics3DSession(state["prepared"].tet_mesh, settings)
            for side in ("left", "right"):
                side_state = state["sides"][side]
                reference = side_state["reference"]
                if int(side_state["particle_load"].load_steps) != VBD_STEPS:
                    raise TrendValidationError("VBD load steps are not frozen at 12")
                if not np.allclose(state["prepared"].tet_mesh.vertices, reference.reference_coordinates_mm, atol=1.0e-5, rtol=0.0):
                    raise TrendValidationError("VBD rest mesh does not match persisted FEA reference")
                result, timing = session.solve_with_timing(side_state["particle_load"])
                if not np.allclose(result.rest_vertices, reference.reference_coordinates_mm, atol=1.0e-5, rtol=0.0):
                    raise TrendValidationError("VBD result rest vertices do not preserve exact FEA coordinates")
                vbd_fp = _fingerprint(
                    {
                        "morphology_id": morphology_id,
                        "side": side,
                        "reference_native_manifest_sha256": side_state["case"]["native_manifest_sha256"],
                        "vbd_settings": asdict(settings),
                        "deformed_vertices_sha256": hashlib.sha256(np.asarray(result.deformed_vertices).tobytes()).hexdigest(),
                    }
                )
                geometry = _vbd_geometry(
                    state["tip"], state["prepared"], result, side_state["artifact"],
                    side=side, vbd_state_fingerprint=vbd_fp,
                )
                side_state.update({"vbd_result": result, "vbd_timing": timing, "vbd_fp": vbd_fp, "vbd_geometry": geometry})
                all_geometries.extend((side_state["artifact"].geometry(state["tip"]), geometry))
                side_state["mechanics_diagnostics"] = compare_mechanics_states(reference, state["prepared"], result)
            state["vbd_settings"] = settings
            prepared_candidates.append(state)
        prepared_groups.append({**group, "prepared_candidates": prepared_candidates})

    bounds = _bounds(all_geometries)
    optix_settings = _optix_settings(bounds)
    tracer = OptiXTransport()
    runtime = create_runtime()
    output_root.mkdir(parents=True, exist_ok=True)
    optix_root = output_root / "vbd_fea_optical_optix"
    all_group_records: list[dict[str, Any]] = []
    for group in prepared_groups:
        group_rows: list[dict[str, Any]] = []
        for state in group["prepared_candidates"]:
            optical: dict[str, Any] = {}
            for side in ("left", "right"):
                side_state = state["sides"][side]
                artifact = side_state["artifact"]
                tip = state["tip"]
                fea_geometry = artifact.geometry(tip)
                material = _material(tip)
                source = {
                    "position_mm": list(artifact.source_position_mm),
                    "medium": artifact.source_medium,
                    "model": "existing Fingertip optical source",
                }
                config = transport_configuration(optix_settings, material=material, source=source)
                common_contract = {
                    "schema": "vbd-fea-optical-trend-optix-case-v1",
                    "comparison_schema": SCHEMA,
                    "scenario_id": group["scenario_id"],
                    "morphology_id": state["morphology_id"],
                    "morphology_fingerprint": state["morphology_fingerprint"],
                    "side": side,
                    "optical_mode": "FULL_3D",
                    "transport_configuration": config,
                    "transport_configuration_fingerprint": fingerprint_mapping(config),
                    "source_position_mm": list(artifact.source_position_mm),
                    "source_medium": artifact.source_medium,
                }
                fea_contract = {
                    **common_contract,
                    "branch": "FEA",
                    "mechanics_source": str(artifact.artifact_path),
                    "native_manifest": str(side_state["case"]["native_manifest"]),
                    "native_manifest_sha256": side_state["case"]["native_manifest_sha256"],
                    "contact_state_fingerprint": artifact.contact_state_fingerprint,
                }
                vbd_contract = {
                    **common_contract,
                    "branch": "VBD",
                    "mechanics_source": "mechanics3d.Mechanics3DSession",
                    "native_manifest": str(side_state["case"]["native_manifest"]),
                    "native_manifest_sha256": side_state["case"]["native_manifest_sha256"],
                    "vbd_state_fingerprint": side_state["vbd_fp"],
                    "contact_state_fingerprint": side_state["vbd_fp"],
                    "vbd_settings": asdict(state["vbd_settings"]),
                }
                fea_path = optix_root / f"{state['morphology_id']}__{side}__FEA.json"
                vbd_path = optix_root / f"{state['morphology_id']}__{side}__VBD.json"
                legacy_path = Path("output/validation/overnight_force_localized_trend/optix_cases") / f"{side_state['case']['case_id']}__FULL_3D__{RAY_COUNT}.json"
                fea_result, fea_reused, fea_seconds, fea_used_path = _trace_or_reuse(
                    path=fea_path,
                    expected_contract=fea_contract,
                    tip=tip,
                    geometry=fea_geometry,
                    settings=optix_settings,
                    morphology_id=state["morphology_id"],
                    morphology_fingerprint=state["morphology_fingerprint"],
                    mechanics_source=str(artifact.artifact_path),
                    contact_state={"localized_load_only": True, "side": side, "contact_state_fingerprint": artifact.contact_state_fingerprint},
                    transport_config=config,
                    runtime=runtime,
                    tracer=tracer,
                    reuse_paths=(legacy_path,),
                )
                vbd_result, vbd_reused, vbd_seconds, vbd_used_path = _trace_or_reuse(
                    path=vbd_path,
                    expected_contract=vbd_contract,
                    tip=tip,
                    geometry=side_state["vbd_geometry"],
                    settings=optix_settings,
                    morphology_id=state["morphology_id"],
                    morphology_fingerprint=state["morphology_fingerprint"],
                    mechanics_source="mechanics3d.Mechanics3DSession",
                    contact_state={"localized_load_only": True, "side": side, "contact_state_fingerprint": side_state["vbd_fp"]},
                    transport_config=config,
                    runtime=runtime,
                    tracer=tracer,
                )
                optical[side] = {
                    "FEA": fea_result,
                    "VBD": vbd_result,
                    "FEA_reused": fea_reused,
                    "VBD_reused": vbd_reused,
                    "FEA_optix_seconds": fea_seconds,
                    "VBD_optix_seconds": vbd_seconds,
                    "FEA_artifact": str(fea_used_path),
                    "VBD_artifact": str(vbd_used_path),
                }
            fea_pair = native_field_separability(optical["left"]["FEA"], optical["right"]["FEA"])
            vbd_pair = native_field_separability(optical["left"]["VBD"], optical["right"]["VBD"])
            row = {
                "scenario_id": group["scenario_id"],
                "morphology_id": state["morphology_id"],
                "morphology_fingerprint": state["morphology_fingerprint"],
                "parameters": state["parameters"],
                "J_VBD": vbd_pair["normalized_redistribution_l1"],
                "J_FEA": fea_pair["normalized_redistribution_l1"],
                "VBD_score_definition": "existing full-3D J3 normalized redistribution L1; higher is more separable",
                "FEA_score_definition": "existing full-3D J3 normalized redistribution L1; higher is more separable",
                "VBD_pair": _score_summary(vbd_pair),
                "FEA_pair": _score_summary(fea_pair),
                "optix_provenance": {
                    "FEA_reused_by_side": {side: optical[side]["FEA_reused"] for side in ("left", "right")},
                    "VBD_reused_by_side": {side: optical[side]["VBD_reused"] for side in ("left", "right")},
                    "FEA_artifacts": {side: optical[side]["FEA_artifact"] for side in ("left", "right")},
                    "VBD_artifacts": {side: optical[side]["VBD_artifact"] for side in ("left", "right")},
                },
                "mechanics": {
                    side: state["sides"][side]["mechanics_diagnostics"]
                    for side in ("left", "right")
                },
                "timing": {
                    "vbd": {side: state["sides"][side]["vbd_timing"] for side in ("left", "right")},
                    "optix_seconds": {
                        "FEA": sum(float(optical[side]["FEA_optix_seconds"]) for side in ("left", "right")),
                        "VBD": sum(float(optical[side]["VBD_optix_seconds"]) for side in ("left", "right")),
                    },
                },
            }
            group_rows.append(row)
            atomic_write_json(
                output_root / PROGRESS_JSON.name,
                _jsonable({"schema": SCHEMA + "-progress", "completed_rows": len(all_group_records) + len(group_rows), "updated_at": _now()}),
            )
        ranked = _ranked_rows(group_rows, value_key="J_VBD", direction="maximize")
        # Add both ranks and sign-safe gaps without changing the stored scores.
        vbd_best = max(float(row["J_VBD"]) for row in group_rows)
        fea_best = max(float(row["J_FEA"]) for row in group_rows)
        for row in group_rows:
            row["rank_J_VBD"] = next(index + 1 for index, candidate in enumerate(ranked) if candidate["morphology_id"] == row["morphology_id"])
            row["rank_J_FEA"] = 0
            row["VBD_score_gap_from_best"] = float(vbd_best - float(row["J_VBD"]))
            row["FEA_score_gap_from_best"] = float(fea_best - float(row["J_FEA"]))
            row["FEA_left_total_transport"] = row["FEA_pair"]["first_total_transport"]
            row["FEA_right_total_transport"] = row["FEA_pair"]["second_total_transport"]
            row["VBD_left_total_transport"] = row["VBD_pair"]["first_total_transport"]
            row["VBD_right_total_transport"] = row["VBD_pair"]["second_total_transport"]
            row["FEA_normalized_redistribution"] = row["J_FEA"]
            row["VBD_normalized_redistribution"] = row["J_VBD"]
            row["mechanics_left_displacement_rms_error_mm"] = row["mechanics"]["left"]["full_field"]["displacement_rms_error_mm"]
            row["mechanics_right_displacement_rms_error_mm"] = row["mechanics"]["right"]["full_field"]["displacement_rms_error_mm"]
        ranking = rank_statistics(group_rows, direction="maximize")
        selection = selection_summary(group_rows, direction="maximize")
        for row in group_rows:
            row["rank_J_FEA"] = ranking["fea_ranks"][row["morphology_id"]]
        group_record = {
            "scenario_id": group["scenario_id"],
            "arm": group["arm"],
            "candidate_count": group["candidate_count"],
            "morphology_ids": [row["morphology_id"] for row in group_rows],
            "ranking": ranking,
            "selection": selection,
            "local_neighborhood": _local_neighborhood(group_rows, selection["vbd_best_morphology"]),
            "rows": group_rows,
        }
        all_group_records.append(group_record)

    all_rows = [row for group in all_group_records for row in group["rows"]]
    fea_reused_states = sum(
        int(reused)
        for row in all_rows
        for reused in row["optix_provenance"]["FEA_reused_by_side"].values()
    )
    vbd_reused_states = sum(
        int(reused)
        for row in all_rows
        for reused in row["optix_provenance"]["VBD_reused_by_side"].values()
    )

    provenance = {
        "code": _git_provenance(),
        "environment": {"python": _package_version("python"), "platform": __import__("platform").platform()},
        "fea": {
            "source_root": str(reference_root.resolve()),
            "source_case_count": len(rows),
            "schemas": sorted({"case": FEA_CASE_SCHEMA, "native": "native-3d-fea-state-v1"}),
            "rerun": False,
            "derived_normals": "recomputed in memory from persisted oriented faces; native FEA artifacts were not modified",
            "source_artifacts": [str(row["native_manifest"]) for row in rows],
        },
        "vbd": {
            "solver": "mechanics3d.Mechanics3DSession/Newton SolverVBD",
            "warp_version": _package_version("warp-lang"),
            "newton_version": _package_version("newton"),
            "settings": asdict(prepared_groups[0]["prepared_candidates"][0]["vbd_settings"]),
        },
        "optics": {
            "runtime": "optics.transport3d.OptiXTransport",
            "settings": asdict(optix_settings),
            "configuration_scope": "one frozen full-3D transport configuration and common field bounds for all states",
            "source_configuration": "existing Fingertip optical source carried by each exact morphology; identical FEA/VBD within each state",
            "ray_count": RAY_COUNT,
            "optix_artifact_root": str((output_root / "vbd_fea_optical_optix").resolve()),
            "artifact_summary": {
                "fea_states_reused": fea_reused_states,
                "fea_states_newly_evaluated": 2 * len(all_rows) - fea_reused_states,
                "vbd_states_reused": vbd_reused_states,
                "vbd_states_newly_evaluated": 2 * len(all_rows) - vbd_reused_states,
            },
        },
        "comparison": {
            "objective_metric": "existing full-3D J3 normalized_redistribution_l1",
            "objective_direction": "maximize",
            "production_minimum_auc": "not applicable to this single localized-load pair corpus; not substituted",
            "group_count": len(all_group_records),
            "candidate_count": sum(group["candidate_count"] for group in all_group_records),
        },
    }
    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "created_at": _now(),
        "elapsed_seconds": time.perf_counter() - started_total,
        "provenance": provenance,
        "groups": all_group_records,
        "limitations": [
            "The selected corpus contains one localized 2 N left/right pair per morphology, not the production 12-trajectory minimum_auc study.",
            "The normalized J3 score is an established full-3D redistribution/separability metric; it is not relabelled as minimum_auc.",
            "Mechanics errors are descriptive bridge diagnostics and do not define surrogate acceptance.",
        ],
    }
    json_path = output_root / TREND_JSON.name
    md_path = output_root / TREND_MD.name
    csv_path = output_root / TREND_CSV.name
    atomic_write_json(json_path, _jsonable(result))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_tmp = md_path.with_name(f".{md_path.name}.{time.time_ns()}.tmp")
    md_tmp.write_text(_markdown_report(result), encoding="utf-8")
    md_tmp.replace(md_path)
    _write_csv([row for group in all_group_records for row in group["rows"]], csv_path)
    return result


def _print_summary(result: Mapping[str, Any]) -> None:
    print(f"{result['status']}: {result['schema']}")
    for group in result["groups"]:
        print(
            f"{group['scenario_id']}: n={group['candidate_count']} "
            f"rho={group['ranking']['spearman_rho']:.6g} "
            f"tau={group['ranking']['kendall_tau']:.6g} "
            f"VBD-best={group['selection']['vbd_best_morphology']} "
            f"FEA-rank={group['selection']['fea_rank_of_vbd_best']}"
        )
    print(f"artifacts: {TREND_JSON}, {TREND_MD}, {TREND_CSV}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    try:
        result = run_comparison(reference_root=args.reference_root, output_root=args.output_root)
    except Exception as exc:
        print(f"BLOCK: {type(exc).__name__}: {exc}")
        return 2
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA",
    "TrendValidationError",
    "artifact_contract_is_exact",
    "discover_homogeneous_groups",
    "rank_statistics",
    "run_comparison",
    "selection_summary",
]
