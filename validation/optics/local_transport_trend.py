"""Validate the local 2D-to-intrinsic-3D transport trend near candidate49.

This module deliberately stays camera-independent.  It reuses the frozen
pre-BO evaluator and the established ``J3D-path`` construction from the
11-mm OptiX validator, then performs tie-aware local rank analysis.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import numpy as np
from scipy.stats import kendalltau, qmc, spearmanr

from mesh import mesh_settings_for_level
from model import Fingertip, FingertipParameters, validate_silicone_ligament
from validation.common.io import atomic_write_json, strict_read_json
from validation.optimization.nominal_sweep import (
    FIXED_FLAT_PAD_WIDTH_MM,
    SWEPT_RANGES,
    _run_isolated_design,
)
from optimization.scenarios import ContactScenario
from validation.optics.transport3d_validation import (
    DEPTH_MM,
    FIELD_RESOLUTIONS,
    VALIDATION_MAX_PERIODIC_WRAPS,
    _internal_path_tv,
    _solve_contact,
    _state_trace_3d,
)
from optics.transport3d.optix_backend import create_runtime


OUTPUT = Path("output/validation/optics/local_transport_trend")
BRIDGE_SUMMARY = Path(
    "output/validation/optics/transport3d/internal_bridge_convergence/summary.json"
)
GLOBAL_CANDIDATE_INPUT = Path(
    "output/validation/optimization/pre_bo_nominal_sweep/inputs/candidate_0049.json"
)
GLOBAL_NOMINAL_RESULT = Path(
    "output/validation/optimization/pre_bo_nominal_sweep/child_results/nominal.json"
)
GLOBAL_CANDIDATE_RESULT = Path(
    "output/validation/optimization/pre_bo_nominal_sweep/child_results/candidate_0049.json"
)

LOCAL_SEED = 20260816
LOCAL_SAMPLE_COUNT = 12
POOL_SIZE = 24
LOCAL_HALF_WIDTH = 0.10
SELECTED_RAY_COUNT = 262_144
FIELD_LEVEL = "current"
J2D_TIE_TOLERANCE = 1.0e-10
J3D_PATH_TIE_TOLERANCE = 0.06

TREND_THRESHOLDS = {
    "strong_concordance_fraction": 0.75,
    "moderate_concordance_fraction": 0.55,
    "inverse_concordance_fraction": 0.25,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status() -> str:
    return subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance_files() -> list[Path]:
    return [
        Path("validation/optics/local_transport_trend.py"),
        Path("validation/optimization/nominal_sweep.py"),
        Path("model/fingertip_parameters.py"),
        Path("validation/optics/transport3d_validation.py"),
        GLOBAL_CANDIDATE_INPUT,
        BRIDGE_SUMMARY,
        BRIDGE_SUMMARY.with_name("fields.npz"),
    ]


def _source_provenance() -> dict[str, Any]:
    payload = {
        str(path): _sha256_file(path)
        for path in _provenance_files()
    }
    return payload


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _precommit_fingerprint(payload: Mapping[str, Any]) -> str:
    scientific_contract = {
        "schema_version": payload["schema_version"],
        "authoritative_design_space": {
            key: value
            for key, value in payload["authoritative_design_space"].items()
            if key != "source"
        },
        "candidate49_parameters": payload["candidate49_parameters"],
        "nominal_parameters": payload["nominal_parameters"],
        "local_neighborhood": payload["local_neighborhood"],
        "sampling": payload["sampling"],
        "ordered_sampling_pool": payload["ordered_sampling_pool"],
        "selected_valid_samples": payload["selected_valid_samples"],
        "rejected_samples": payload["rejected_samples"],
        "three_d_plan": payload["three_d_plan"],
        "metric_definitions": payload["metric_definitions"],
        "analysis_plan": payload["analysis_plan"],
        "existing_bridge_artifact": {
            key: value
            for key, value in payload["existing_bridge_artifact"].items()
            if key not in {"summary", "fields"}
        },
    }
    return _canonical_sha256(scientific_contract)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
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


def _candidate_parameters() -> dict[str, Any]:
    payload = strict_read_json(GLOBAL_CANDIDATE_INPUT)
    return {str(key): value for key, value in payload["parameters"].items()}


def _nominal_parameters() -> dict[str, Any]:
    return {str(key): value for key, value in asdict(FingertipParameters()).items()}


def _authoritative_design_space() -> dict[str, Any]:
    candidate = _candidate_parameters()
    variables = [
        {
            "name": name,
            "lower_mm": float(lower),
            "upper_mm": float(upper),
            "candidate49_mm": candidate[name],
            "range_mm": float(upper - lower),
        }
        for name, lower, upper in SWEPT_RANGES
    ]
    return {
        "source": str(Path("validation/optimization/nominal_sweep.py")),
        "fixed_parameters": {"flat_pad_width": FIXED_FLAT_PAD_WIDTH_MM},
        "variables": variables,
        "variable_order": [item["name"] for item in variables],
        "definition": "authoritative pre-BO nominal morphology sweep SWEPT_RANGES",
    }


def _parameters_and_realized_xi(
    requested_xi: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, float], dict[str, bool]]:
    candidate = _candidate_parameters()
    parameters = _nominal_parameters()
    parameters["flat_pad_width"] = FIXED_FLAT_PAD_WIDTH_MM
    realized_xi: dict[str, float] = {}
    clipping_flags: dict[str, bool] = {}
    for name, lower, upper in SWEPT_RANGES:
        requested_value = candidate[name] + float(requested_xi[name]) * (upper - lower)
        realized_value = min(float(upper), max(float(lower), requested_value))
        parameters[name] = realized_value
        realized_xi[name] = float((realized_value - candidate[name]) / (upper - lower))
        clipping_flags[name] = not np.isclose(realized_value, requested_value, rtol=0.0, atol=1.0e-12)
    return parameters, realized_xi, clipping_flags


def _parameters_from_xi(xi: Mapping[str, float]) -> dict[str, Any]:
    parameters, _, _ = _parameters_and_realized_xi(xi)
    return parameters


def _geometry_check(parameters: Mapping[str, float]) -> tuple[bool, str | None]:
    try:
        typed = FingertipParameters(**dict(parameters))
        validate_silicone_ligament(typed)
        Fingertip(typed)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def _duplicate_parameter_groups(samples: list[Mapping[str, Any]]) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for sample in samples:
        parameters = {
            name: sample["parameters"][name]
            for name, _, _ in SWEPT_RANGES
        }
        key = json.dumps(_jsonable(parameters), sort_keys=True, separators=(",", ":"))
        groups.setdefault(key, []).append(str(sample["sample_id"]))
    return [sample_ids for sample_ids in groups.values() if len(sample_ids) > 1]


def _precommit_payload() -> dict[str, Any]:
    design_space = _authoritative_design_space()
    names = design_space["variable_order"]
    sampler = qmc.LatinHypercube(d=len(names), scramble=True, seed=LOCAL_SEED)
    unit_pool = sampler.random(n=POOL_SIZE)
    pool: list[dict[str, Any]] = []
    for index, unit_point in enumerate(unit_pool, start=1):
        xi = {
            name: float(-LOCAL_HALF_WIDTH + 2.0 * LOCAL_HALF_WIDTH * value)
            for name, value in zip(names, unit_point, strict=True)
        }
        parameters, realized_xi, clipping_flags = _parameters_and_realized_xi(xi)
        valid, reason = _geometry_check(parameters)
        pool.append(
            {
                "pool_index": index,
                "sample_id": f"local_{index:03d}",
                "requested_normalized_local_coordinate": xi,
                "normalized_local_coordinate": realized_xi,
                "clipping_flags": clipping_flags,
                "parameters": parameters,
                "geometry_valid": valid,
                "rejection_reason": reason,
            }
        )
    selected = [item for item in pool if item["geometry_valid"]][:LOCAL_SAMPLE_COUNT]
    if len(selected) < LOCAL_SAMPLE_COUNT:
        raise RuntimeError(
            f"pre-generated local pool produced only {len(selected)} valid samples"
        )
    selected_ids = {item["sample_id"] for item in selected}
    rejected = [
        {
            "sample_id": item["sample_id"],
            "pool_index": item["pool_index"],
            "geometry_valid": item["geometry_valid"],
            "requested_normalized_local_coordinate": item["requested_normalized_local_coordinate"],
            "normalized_local_coordinate": item["normalized_local_coordinate"],
            "clipping_flags": item["clipping_flags"],
            "parameters": item["parameters"],
            "rejection_reason": item["rejection_reason"]
            or "valid sample after target count; not selected",
        }
        for item in pool
        if item["sample_id"] not in selected_ids
    ]
    bridge = strict_read_json(BRIDGE_SUMMARY)
    payload = {
        "schema_version": 3,
        "created_at": _now(),
        "git_revision": _git_revision(),
        "authoritative_design_space": design_space,
        "candidate49_parameters": _candidate_parameters(),
        "nominal_parameters": _nominal_parameters(),
        "local_neighborhood": {
            "coordinate_definition": "xi_i=(p_i-p_candidate49_i)/(p_i_max-p_i_min)",
            "half_width": LOCAL_HALF_WIDTH,
            "bounds": [-LOCAL_HALF_WIDTH, LOCAL_HALF_WIDTH],
            "clipping": "clip physical values to original inclusive bounds only",
            "stored_coordinate": "realized xi after clipping",
            "requested_coordinate_field": "requested_normalized_local_coordinate",
        },
        "sampling": {
            "method": "fixed-seed LatinHypercube",
            "seed": LOCAL_SEED,
            "pool_size": POOL_SIZE,
            "target_valid_samples": LOCAL_SAMPLE_COUNT,
            "ordered_pool": True,
            "selection": "first 12 geometry-valid entries; no score-based replacement",
        },
        "ordered_sampling_pool": pool,
        "selected_valid_samples": selected,
        "rejected_samples": rejected,
        "sampling_diagnostics": {
            "selected_sample_count": len(selected),
            "selected_unique_physical_parameter_count": len(selected)
            - sum(len(group) - 1 for group in _duplicate_parameter_groups(selected)),
            "duplicate_physical_parameter_groups": _duplicate_parameter_groups(selected),
            "duplicates_preserved": True,
        },
        "three_d_plan": {
            "ray_count": SELECTED_RAY_COUNT,
            "field_level": FIELD_LEVEL,
            "field_resolution": FIELD_RESOLUTIONS[FIELD_LEVEL],
            "extrusion_depth_mm": DEPTH_MM,
            "maximum_periodic_wraps": VALIDATION_MAX_PERIODIC_WRAPS,
            "minimum_ray_weight": 1.0e-4,
            "mechanics": "medium mesh, 48 FEM steps, three_pairs internal contact",
            "selection_reason": (
                "262144 is the existing persisted J3D-path field fidelity and is reused "
                "for nominal/candidate49; lower tested ray counts preserve ordering but "
                "do not have reusable field artifacts for this sweep"
            ),
        },
        "metric_definitions": {
            "j2d": "existing DesignEvaluator score/minimum separability",
            "j3d_path": "TV between mass-preservingly resampled z-integrated P3_xy fields on common support",
            "j3d_surface": "not evaluated; remains non-primary historical diagnostic",
            "camera_analysis": "out of scope",
        },
        "analysis_plan": {
            "j2d_tie_tolerance": J2D_TIE_TOLERANCE,
            "j3d_path_tie_tolerance": J3D_PATH_TIE_TOLERANCE,
            "minimum_resolved_pair_fraction": 0.5,
            "trend_thresholds": TREND_THRESHOLDS,
            "rank_direction": "higher score is better",
            "statistics": [
                "Spearman",
                "Kendall tau-b",
                "tie-aware pairwise concordance/discordance",
                "candidate49-excluded repeat",
            ],
            "dynamic_range_gate": {
                "plateau": "both J2D and J3D-path range <= 2*tie tolerance",
                "insufficient_resolution": "J3D-path range <= 2*tie tolerance or resolved pair fraction < 0.5",
                "discordance_claim": "resolved under the adopted global tolerance; not per-sample convergence",
            },
        },
        "existing_bridge_artifact": {
            "summary": str(BRIDGE_SUMMARY),
            "fields": str(BRIDGE_SUMMARY.with_name("fields.npz")),
            "ray_convergence_pass": bridge.get("ray_convergence_pass"),
            "existing_final_ordering": bridge.get("final_ordering"),
            "historical_reduced_2d_reference": bridge.get("historical_reduced_2d_reference"),
        },
    }
    payload["provenance"] = {
        "hash_algorithm": "sha256",
        "source_and_reused_artifact_hashes": _source_provenance(),
    }
    payload["precommit_fingerprint"] = _precommit_fingerprint(payload)
    return payload


def _verify_precommit(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 3:
        raise RuntimeError("precommit schema is stale; regenerate it before resuming")
    fingerprint = payload.get("precommit_fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != _precommit_fingerprint(payload):
        raise RuntimeError("precommit fingerprint mismatch; dependent artifacts are stale")
    expected_sources = payload.get("provenance", {}).get("source_and_reused_artifact_hashes", {})
    if expected_sources != _source_provenance():
        raise RuntimeError("precommit source/artifact hashes do not match the current repository")

    if payload.get("candidate49_parameters") != _candidate_parameters():
        raise RuntimeError("precommit candidate49 vector does not match the current candidate input")
    if payload.get("nominal_parameters") != _nominal_parameters():
        raise RuntimeError("precommit nominal vector does not match FingertipParameters defaults")
    current_design_space = _authoritative_design_space()
    stored_design_space = payload.get("authoritative_design_space", {})
    if {
        key: value for key, value in stored_design_space.items() if key != "source"
    } != {
        key: value for key, value in current_design_space.items() if key != "source"
    }:
        raise RuntimeError("precommit design space does not match current authoritative sweep")
    if payload.get("local_neighborhood", {}).get("half_width") != LOCAL_HALF_WIDTH:
        raise RuntimeError("precommit local neighborhood width does not match current contract")
    sampling = payload.get("sampling", {})
    if (
        sampling.get("seed") != LOCAL_SEED
        or sampling.get("pool_size") != POOL_SIZE
        or sampling.get("target_valid_samples") != LOCAL_SAMPLE_COUNT
    ):
        raise RuntimeError("precommit sampling metadata does not match current contract")

    candidate = payload["candidate49_parameters"]
    variable_by_name = {
        item["name"]: item for item in payload["authoritative_design_space"]["variables"]
    }
    pool = payload.get("ordered_sampling_pool", [])
    if len(pool) != POOL_SIZE:
        raise RuntimeError("precommit ordered pool size changed")
    for item in pool:
        requested = item.get("requested_normalized_local_coordinate")
        realized = item.get("normalized_local_coordinate")
        clipping_flags = item.get("clipping_flags")
        parameters = item.get("parameters")
        if not isinstance(requested, Mapping) or not isinstance(realized, Mapping) or not isinstance(clipping_flags, Mapping):
            raise RuntimeError(f"{item.get('sample_id')} lacks requested/realized xi")
        if not isinstance(item.get("geometry_valid"), bool):
            raise RuntimeError(f"{item.get('sample_id')} lacks geometry_valid metadata")
        for name, variable in variable_by_name.items():
            lower = float(variable["lower_mm"])
            upper = float(variable["upper_mm"])
            span = upper - lower
            requested_value = float(candidate[name]) + float(requested[name]) * span
            clipped = min(upper, max(lower, requested_value))
            expected_realized = (clipped - float(candidate[name])) / span
            expected_clipped = not np.isclose(clipped, requested_value, rtol=0.0, atol=1.0e-12)
            if not np.isclose(float(parameters[name]), clipped, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(f"{item['sample_id']} physical parameter does not match requested xi")
            if not np.isclose(float(realized[name]), expected_realized, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(f"{item['sample_id']} realized xi does not match clipped parameter")
            if bool(clipping_flags[name]) != expected_clipped:
                raise RuntimeError(f"{item['sample_id']} clipping flag does not match clipped parameter")


def _verify_result_bundle(payload: Mapping[str, Any], precommit: Mapping[str, Any]) -> None:
    if payload.get("precommit_fingerprint") != precommit.get("precommit_fingerprint"):
        raise RuntimeError("result bundle references a stale precommit")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("result bundle has no provenance")
    if provenance.get("source_and_reused_artifact_hashes") != _source_provenance():
        raise RuntimeError("result bundle source hashes do not match the current repository")
    for record in payload.get("results", []):
        artifact = record.get("source_artifact_id")
        expected_hash = record.get("source_artifact_sha256")
        if artifact is None or expected_hash is None:
            raise RuntimeError(f"{record.get('morphology_id')} has incomplete artifact provenance")
        path = Path(artifact)
        if not path.exists() or _sha256_file(path) != expected_hash:
            raise RuntimeError(f"{record.get('morphology_id')} source artifact is stale or missing")


def write_precommit(output: Path = OUTPUT) -> dict[str, Any]:
    """Create or verify the immutable pre-3D sampling decision artifact."""
    output.mkdir(parents=True, exist_ok=True)
    precommit_path = output / "precommit.json"
    manifest_path = output / "run_manifest.json"
    invalidated: list[dict[str, Any]] = []
    if precommit_path.exists():
        existing = strict_read_json(precommit_path)
        try:
            _verify_precommit(existing)
            payload = existing
        except RuntimeError as exc:
            invalidated.append(
                {
                    "artifact": str(precommit_path),
                    "reason": str(exc),
                    "historical": True,
                }
            )
            payload = _precommit_payload()
            atomic_write_json(precommit_path, payload)
    else:
        payload = _precommit_payload()
        atomic_write_json(precommit_path, payload)
    manifest = {
        "schema_version": 1,
        "stage": "precommit",
        "status": "PRECOMMITTED",
        "created_at": payload["created_at"],
        "updated_at": _now(),
        "precommit_path": str(precommit_path),
        "advisor_status": "pending_pre_run_consultation",
        "reviewer_status": "pending_precommit_review",
        "completed_2d_samples": [],
        "completed_3d_samples": [],
        "reused_artifacts": [],
        "invalidated_artifacts": [],
        "scientific_outcome": None,
        "blockers": [],
        "next_authorized_action": "mandatory_advisor_consultation_then_frozen_2d_stage",
    }
    if manifest_path.exists():
        previous = strict_read_json(manifest_path)
        manifest["advisor_status"] = previous.get("advisor_status", manifest["advisor_status"])
        manifest["reviewer_status"] = previous.get("reviewer_status", manifest["reviewer_status"])
        if previous.get("precommit_fingerprint") == payload["precommit_fingerprint"]:
            for key in (
                "completed_2d_samples",
                "completed_3d_samples",
                "reused_artifacts",
                "artifacts_generated",
                "scientific_outcome",
            ):
                if key in previous:
                    manifest[key] = previous[key]
        for item in previous.get("invalidated_artifacts", []):
            if isinstance(item, Mapping):
                entry = dict(item)
                entry.setdefault("historical", True)
            else:
                entry = {
                    "artifact": "previous_manifest_entry",
                    "reason": str(item),
                    "historical": True,
                }
            invalidated.append(entry)
    manifest["precommit_fingerprint"] = payload["precommit_fingerprint"]
    manifest["invalidated_artifacts"] = invalidated
    atomic_write_json(manifest_path, manifest)
    return payload


def _update_manifest(output: Path, **updates: Any) -> None:
    manifest_path = output / "run_manifest.json"
    if not manifest_path.exists():
        return
    manifest = strict_read_json(manifest_path)
    manifest.update(updates)
    manifest["updated_at"] = _now()
    atomic_write_json(manifest_path, manifest)


def _historical_2d_record(path: Path, morphology: str) -> dict[str, Any]:
    payload = strict_read_json(path)
    evaluation = payload.get("evaluation") or {}
    score = evaluation.get("minimum_separability")
    if payload.get("status") != "success" or score is None:
        raise RuntimeError(f"historical {morphology} 2D artifact is not successful")
    matched = _matched_location_half_mm(evaluation)
    return {
        "morphology_id": morphology,
        "parameters": payload["parameters"],
        "normalized_local_coordinate": None,
        "j2d": float(score),
        "j2d_matched_location_05": matched,
        "j2d_limiting_pair": evaluation.get("limiting_pair"),
        "evaluation": evaluation,
        "geometry_valid": True,
        "status": "success",
        "reused": True,
        "source_artifact_id": str(path),
        "source_artifact_sha256": _sha256_file(path),
    }


def _matched_location_half_mm(evaluation: Mapping[str, Any]) -> float | None:
    """Return the 2D pair used by the established left/right 0.5-mm bridge."""
    for pair in evaluation.get("pairs", []):
        if not isinstance(pair, Mapping) or pair.get("axis") != "location":
            continue
        first = pair.get("first", {})
        second = pair.get("second", {})
        if (
            first.get("indentation_mm") == 0.5
            and second.get("indentation_mm") == 0.5
            and first.get("location_x_mm") == -3.0
            and second.get("location_x_mm") == 3.0
        ):
            value = pair.get("separability")
            return None if value is None else float(value)
    return None


def run_frozen_2d(output: Path = OUTPUT) -> dict[str, Any]:
    """Evaluate all precommitted local samples with the frozen evaluator."""
    precommit = strict_read_json(output / "precommit.json")
    _verify_precommit(precommit)
    results_path = output / "local_2d_results.json"
    if results_path.exists():
        existing = strict_read_json(results_path)
        try:
            _verify_result_bundle(existing, precommit)
            return existing
        except RuntimeError:
            pass
    output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = [
        _historical_2d_record(GLOBAL_NOMINAL_RESULT, "nominal"),
        _historical_2d_record(GLOBAL_CANDIDATE_RESULT, "candidate49"),
    ]
    for sample in precommit["selected_valid_samples"]:
        sample_id = sample["sample_id"]
        parameters = FingertipParameters(**sample["parameters"])
        child_path = output / "2d_child_results" / "child_results" / f"{sample_id}.json"
        reused_child = None
        if child_path.exists():
            candidate_child = strict_read_json(child_path)
            if candidate_child.get("parameters") == sample["parameters"]:
                reused_child = candidate_child
        record: dict[str, Any] = {
            "morphology_id": sample_id,
            "parameters": sample["parameters"],
            "requested_normalized_local_coordinate": sample["requested_normalized_local_coordinate"],
            "normalized_local_coordinate": sample["normalized_local_coordinate"],
            "clipping_flags": sample["clipping_flags"],
            "geometry_valid": True,
            "status": "running",
            "reused": reused_child is not None,
            "source_artifact_id": str(child_path),
        }
        try:
            # Reuse the existing isolated child protocol and exact pre-BO evaluator
            # configuration rather than introducing a second evaluator path.
            child = reused_child or _run_isolated_design(parameters, output / "2d_child_results", sample_id)
            evaluation = child.get("evaluation") or {}
            record.update(
                {
                    "status": child.get("status", "process_failure"),
                    "failure_category": child.get("failure_category"),
                    "failure_message": child.get("failure_message"),
                    "wall_time_seconds": child.get("wall_time_seconds"),
                    "j2d": evaluation.get("minimum_separability"),
                    "j2d_matched_location_05": _matched_location_half_mm(evaluation),
                    "j2d_limiting_pair": evaluation.get("limiting_pair"),
                    "evaluation": evaluation,
                    "source_artifact_sha256": _sha256_file(child_path) if child_path.exists() else None,
                }
            )
        except Exception as exc:
            record.update(
                {
                    "status": "process_failure",
                    "failure_category": "process_failure",
                    "failure_message": f"{type(exc).__name__}: {exc}",
                    "j2d": None,
                }
            )
        results.append(record)
        atomic_write_json(
            output / "local_2d_progress.json",
            {"completed": [item["morphology_id"] for item in results], "results": results},
        )
    successful = [item for item in results if item.get("status") == "success" and item.get("j2d") is not None]
    if len(successful) < 2 + LOCAL_SAMPLE_COUNT:
        raise RuntimeError(f"only {len(successful)} successful 2D records; expected {2 + LOCAL_SAMPLE_COUNT}")
    historical_reference = precommit["existing_bridge_artifact"].get("historical_reduced_2d_reference") or {}
    historical_reproduction = {
        item["morphology_id"]: {
            "expected": historical_reference.get(item["morphology_id"]),
            "observed": item["j2d"],
            "absolute_difference": abs(float(item["j2d"]) - float(historical_reference[item["morphology_id"]]))
            if item["morphology_id"] in historical_reference
            else None,
            "pass": item["morphology_id"] in historical_reference
            and abs(float(item["j2d"]) - float(historical_reference[item["morphology_id"]])) <= J2D_TIE_TOLERANCE,
        }
        for item in results[:2]
    }
    if not all(item["pass"] for item in historical_reproduction.values()):
        raise RuntimeError(f"historical J2D reproduction failed: {historical_reproduction}")
    payload = {
        "schema_version": 1,
        "created_at": _now(),
        "precommit_fingerprint": precommit["precommit_fingerprint"],
        "evaluator_source": str(Path("validation/optimization/nominal_sweep.py")),
        "provenance": {
            "hash_algorithm": "sha256",
            "source_and_reused_artifact_hashes": _source_provenance(),
        },
        "results": results,
        "historical_reproduction": historical_reproduction,
        "status": "COMPLETE",
    }
    atomic_write_json(results_path, payload)
    _update_manifest(
        output,
        stage="2d_complete",
        completed_2d_samples=[item["morphology_id"] for item in results],
        artifacts_generated=[str(results_path), str(output / "local_2d_progress.json")],
        reused_artifacts=[
            {
                "morphology_id": item["morphology_id"],
                "artifact": item["source_artifact_id"],
                "sha256": item["source_artifact_sha256"],
            }
            for item in results
            if item.get("reused") and item.get("source_artifact_sha256")
        ],
        next_authorized_action="independent_reviewer_recheck_then_3d_stage",
    )
    return payload


def _existing_3d_records() -> list[dict[str, Any]]:
    bridge = strict_read_json(BRIDGE_SUMMARY)
    current = bridge["ray_convergence"][str(SELECTED_RAY_COUNT)][FIELD_LEVEL]
    records = []
    for morphology in ("nominal", "candidate49"):
        record = current[morphology]
        records.append(
            {
                "morphology_id": morphology,
                "parameters": _nominal_parameters() if morphology == "nominal" else _candidate_parameters(),
                "normalized_local_coordinate": None,
                "j3d_path": float(record["j3d_path"]),
                "j3d_path_comparison_grid": record["j3d_path_comparison_grid"],
                "geometry_valid": True,
                "status": "success",
                "reused": True,
                "source_artifact_id": str(BRIDGE_SUMMARY.with_name("fields.npz")),
                "source_artifact_sha256": _sha256_file(BRIDGE_SUMMARY.with_name("fields.npz")),
                "ray_count": SELECTED_RAY_COUNT,
                "field_level": FIELD_LEVEL,
                "total_internal_path_mass": None,
                "left_outgoing_surface_weight": None,
                "right_outgoing_surface_weight": None,
            }
        )
    return records


def _local_3d_record(
    sample: Mapping[str, Any],
    output: Path,
    runtime: Any,
) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    parameters = FingertipParameters(**sample["parameters"])
    tip = Fingertip(parameters)
    mesh = tip.mesh(mesh_settings_for_level("medium"))
    cache_root = output / "fea_states"
    left_mesh, left_fem = _solve_contact(
        tip,
        mesh,
        ContactScenario(-3.0, 0.5, 4.0),
        cache_path=cache_root / f"{sample_id}_left_contact.npz",
    )
    right_mesh, right_fem = _solve_contact(
        tip,
        mesh,
        ContactScenario(3.0, 0.5, 4.0),
        cache_path=cache_root / f"{sample_id}_right_contact.npz",
    )
    resolution = FIELD_RESOLUTIONS[FIELD_LEVEL]
    started = time.perf_counter()
    left = _state_trace_3d(
        tip,
        left_mesh,
        mesh,
        runtime,
        mode="full3d",
        ray_count=SELECTED_RAY_COUNT,
        retain_internal_path_field=True,
        field_resolution=resolution,
        minimum_ray_weight=1.0e-4,
    )
    right = _state_trace_3d(
        tip,
        right_mesh,
        mesh,
        runtime,
        mode="full3d",
        ray_count=SELECTED_RAY_COUNT,
        retain_internal_path_field=True,
        field_resolution=resolution,
        minimum_ray_weight=1.0e-4,
    )
    j3d_path, comparison_grid = _internal_path_tv(left, right)
    field_path = output / "fields" / f"{sample_id}.npz"
    field_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        field_path,
        left_p3_xy=left.internal_z_integrated_path_density,
        left_p3_x_edges=left.internal_path_x_edges_mm,
        left_p3_y_edges=left.internal_path_y_edges_mm,
        left_p3_z_edges=left.internal_path_z_edges_mm,
        right_p3_xy=right.internal_z_integrated_path_density,
        right_p3_x_edges=right.internal_path_x_edges_mm,
        right_p3_y_edges=right.internal_path_y_edges_mm,
        right_p3_z_edges=right.internal_path_z_edges_mm,
        left_surface_field=left.outgoing_surface_field,
        right_surface_field=right.outgoing_surface_field,
    )
    return {
        "morphology_id": sample_id,
        "parameters": sample["parameters"],
        "requested_normalized_local_coordinate": sample["requested_normalized_local_coordinate"],
        "normalized_local_coordinate": sample["normalized_local_coordinate"],
        "clipping_flags": sample["clipping_flags"],
        "j3d_path": float(j3d_path),
        "j3d_path_comparison_grid": comparison_grid,
        "geometry_valid": True,
        "status": "success",
        "reused": False,
        "source_artifact_id": str(field_path),
        "source_artifact_sha256": _sha256_file(field_path),
        "ray_count": SELECTED_RAY_COUNT,
        "field_level": FIELD_LEVEL,
        "extrusion_depth_mm": DEPTH_MM,
        "total_internal_path_mass": {
            "left": float(np.sum(left.internal_z_integrated_path_density)),
            "right": float(np.sum(right.internal_z_integrated_path_density)),
        },
        "left_outgoing_surface_weight": float(left.outgoing_surface_weight),
        "right_outgoing_surface_weight": float(right.outgoing_surface_weight),
        "left_escaped_fraction": float(left.escaped_weight / left.launched_weight),
        "right_escaped_fraction": float(right.escaped_weight / right.launched_weight),
        "left_energy_balance_error": float(left.energy_balance_error),
        "right_energy_balance_error": float(right.energy_balance_error),
        "fem": {"left": left_fem, "right": right_fem},
        "trace_wall_time_seconds": time.perf_counter() - started,
    }


def run_intrinsic_3d(output: Path = OUTPUT) -> dict[str, Any]:
    """Run or resume the fixed-fidelity intrinsic local 3D sweep."""
    precommit = strict_read_json(output / "precommit.json")
    _verify_precommit(precommit)
    two_d = strict_read_json(output / "local_2d_results.json")
    if two_d.get("status") != "COMPLETE":
        raise RuntimeError("complete frozen 2D results are required before 3D")
    result_path = output / "local_3d_results.json"
    records = _existing_3d_records()
    completed: dict[str, dict[str, Any]] = {item["morphology_id"]: item for item in records}
    if result_path.exists():
        previous = strict_read_json(result_path)
        try:
            _verify_result_bundle(previous, precommit)
            for item in previous.get("results", []):
                completed[item["morphology_id"]] = item
        except RuntimeError:
            previous_manifest = strict_read_json(output / "run_manifest.json")
            _update_manifest(
                output,
                invalidated_artifacts=[
                    *previous_manifest.get("invalidated_artifacts", []),
                    {
                        "artifact": str(result_path),
                        "reason": "stale or incomplete provenance",
                        "historical": True,
                    },
                ],
            )
    runtime = create_runtime()
    try:
        for sample in precommit["selected_valid_samples"]:
            sample_id = str(sample["sample_id"])
            if sample_id in completed and completed[sample_id].get("status") == "success":
                continue
            try:
                completed[sample_id] = _local_3d_record(sample, output, runtime)
            except Exception as exc:
                completed[sample_id] = {
                    "morphology_id": sample_id,
                    "parameters": sample["parameters"],
                    "normalized_local_coordinate": sample["normalized_local_coordinate"],
                    "requested_normalized_local_coordinate": sample["requested_normalized_local_coordinate"],
                    "clipping_flags": sample["clipping_flags"],
                    "geometry_valid": True,
                    "status": "failure",
                    "reused": False,
                    "failure_message": f"{type(exc).__name__}: {exc}",
                }
            atomic_write_json(
                output / "local_3d_progress.json",
                {
                    "completed": sorted(completed),
                    "results": list(completed.values()),
                },
            )
            try:
                runtime.cp.cuda.Stream.null.synchronize()
                runtime.cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
    finally:
        try:
            runtime.cp.cuda.Stream.null.synchronize()
            runtime.cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    ordered = [completed[item["morphology_id"]] for item in records]
    ordered.extend(completed[item["sample_id"]] for item in precommit["selected_valid_samples"])
    payload = {
        "schema_version": 1,
        "created_at": _now(),
        "precommit_fingerprint": precommit["precommit_fingerprint"],
        "provenance": {
            "hash_algorithm": "sha256",
            "source_and_reused_artifact_hashes": _source_provenance(),
        },
        "results": ordered,
        "status": "COMPLETE" if all(item.get("status") == "success" for item in ordered) else "INCOMPLETE",
    }
    atomic_write_json(result_path, payload)
    _update_manifest(
        output,
        stage="3d_complete" if payload["status"] == "COMPLETE" else "3d_incomplete",
        completed_3d_samples=[item["morphology_id"] for item in ordered if item.get("status") == "success"],
        artifacts_generated=[str(result_path), str(output / "local_3d_progress.json")],
        next_authorized_action="trend_analysis" if payload["status"] == "COMPLETE" else "repair_3d_failures",
    )
    return payload


def _tie_sign(value: float, tolerance: float) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _tie_aware_ranks(values: list[float], tolerance: float) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and abs(values[order[end]] - values[order[position]]) <= tolerance:
            end += 1
        rank = 0.5 * (position + 1 + end)
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def _pairwise_rank_result(
    j2d: list[float],
    j3d: list[float],
    *,
    j2d_tolerance: float,
    j3d_tolerance: float,
) -> dict[str, Any]:
    concordant = discordant = tied = 0
    for first in range(len(j2d)):
        for second in range(first + 1, len(j2d)):
            sign_2d = _tie_sign(j2d[first] - j2d[second], j2d_tolerance)
            sign_3d = _tie_sign(j3d[first] - j3d[second], j3d_tolerance)
            if sign_2d == 0 or sign_3d == 0:
                tied += 1
            elif sign_2d == sign_3d:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant + tied
    ordered = concordant + discordant
    return {
        "concordant_count": concordant,
        "discordant_count": discordant,
        "tied_count": tied,
        "pair_count": total,
        "concordance_fraction": concordant / total if total else None,
        "discordance_fraction": discordant / total if total else None,
        "tie_fraction": tied / total if total else None,
        "non_tied_concordance_fraction": concordant / ordered if ordered else None,
        "tie_tolerances": {"j2d": j2d_tolerance, "j3d_path": j3d_tolerance},
    }


def _candidate_rank_interval(values: list[float], names: list[str], tolerance: float) -> dict[str, Any] | None:
    if "candidate49" not in names:
        return None
    candidate_index = names.index("candidate49")
    candidate = values[candidate_index]
    resolved_better = sum(
        value > candidate + tolerance
        for index, value in enumerate(values)
        if index != candidate_index
    )
    resolved_worse = sum(
        value < candidate - tolerance
        for index, value in enumerate(values)
        if index != candidate_index
    )
    unresolved = len(values) - 1 - resolved_better - resolved_worse
    return {
        "best_possible_rank": 1 + resolved_better,
        "worst_possible_rank": 1 + resolved_better + unresolved,
        "resolved_better_count": resolved_better,
        "resolved_worse_count": resolved_worse,
        "unresolved_count": unresolved,
        "tolerance": tolerance,
    }


def _finite_statistic(result: Any) -> dict[str, float | None] | None:
    if result is None:
        return None
    statistic = float(result.statistic)
    pvalue = float(result.pvalue)
    return {
        "statistic": statistic if np.isfinite(statistic) else None,
        "pvalue": pvalue if np.isfinite(pvalue) else None,
    }


def _trend_for_records(
    records: list[dict[str, Any]],
    *,
    label: str,
    j2d_field: str = "j2d",
) -> dict[str, Any]:
    valid = [item for item in records if item.get("status") == "success" and item.get(j2d_field) is not None and item.get("j3d_path") is not None]
    names = [str(item["morphology_id"]) for item in valid]
    j2d = [float(item[j2d_field]) for item in valid]
    j3d = [float(item["j3d_path"]) for item in valid]
    pairwise = _pairwise_rank_result(
        j2d,
        j3d,
        j2d_tolerance=J2D_TIE_TOLERANCE,
        j3d_tolerance=J3D_PATH_TIE_TOLERANCE,
    )
    rank_2d = _tie_aware_ranks(j2d, J2D_TIE_TOLERANCE)
    rank_3d = _tie_aware_ranks(j3d, J3D_PATH_TIE_TOLERANCE)
    scipy_spearman = spearmanr(j2d, j3d) if len(valid) >= 2 else None
    scipy_kendall = kendalltau(j2d, j3d) if len(valid) >= 2 else None
    tie_adjusted_spearman = (
        spearmanr(rank_2d, rank_3d)
        if len(valid) >= 2 and len(set(rank_2d)) > 1 and len(set(rank_3d)) > 1
        else None
    )
    dynamic_range = {
        "j2d_min": min(j2d) if j2d else None,
        "j2d_max": max(j2d) if j2d else None,
        "j2d_range": max(j2d) - min(j2d) if j2d else None,
        "j3d_path_min": min(j3d) if j3d else None,
        "j3d_path_max": max(j3d) if j3d else None,
        "j3d_path_range": max(j3d) - min(j3d) if j3d else None,
        "j2d_plateau": bool(j2d and max(j2d) - min(j2d) <= 2.0 * J2D_TIE_TOLERANCE),
        "j3d_path_plateau": bool(j3d and max(j3d) - min(j3d) <= 2.0 * J3D_PATH_TIE_TOLERANCE),
    }
    candidate_index = names.index("candidate49") if "candidate49" in names else None
    return {
        "label": label,
        "j2d_field": j2d_field,
        "sample_count": len(valid),
        "morphology_ids": names,
        "scores": {
            name: {"j2d": a, "j3d_path": b, "rank_j2d": c, "rank_j3d_path": d}
            for name, a, b, c, d in zip(names, j2d, j3d, rank_2d, rank_3d, strict=True)
        },
        "spearman": _finite_statistic(scipy_spearman),
        "kendall_tau_b": _finite_statistic(scipy_kendall),
        "tie_adjusted_spearman": _finite_statistic(tie_adjusted_spearman),
        "pairwise_rank": pairwise,
        "dynamic_range": dynamic_range,
        "candidate49": {
            "rank_j2d": rank_2d[candidate_index] if candidate_index is not None else None,
            "rank_j3d_path_display": rank_3d[candidate_index] if candidate_index is not None else None,
            "rank_j3d_path_interval": _candidate_rank_interval(j3d, names, J3D_PATH_TIE_TOLERANCE),
            "sample_count": len(valid),
        },
    }


def analyze_trend(output: Path = OUTPUT) -> dict[str, Any]:
    precommit = strict_read_json(output / "precommit.json")
    two_d = strict_read_json(output / "local_2d_results.json")
    three_d = strict_read_json(output / "local_3d_results.json")
    by_id_2d = {item["morphology_id"]: item for item in two_d["results"]}
    records: list[dict[str, Any]] = []
    for item in three_d["results"]:
        merged = dict(item)
        source_2d = by_id_2d.get(item["morphology_id"])
        if source_2d is not None:
            merged["j2d"] = source_2d.get("j2d")
            merged["j2d_matched_location_05"] = source_2d.get("j2d_matched_location_05")
            merged["j2d_limiting_pair"] = source_2d.get("j2d_limiting_pair")
            merged["j2d_status"] = source_2d.get("status")
        records.append(merged)
    local_records = [item for item in records if item["morphology_id"] != "nominal"]
    full = _trend_for_records(local_records, label="candidate49_and_all_valid_local_samples")
    without_candidate = _trend_for_records(
        [item for item in local_records if item["morphology_id"] != "candidate49"],
        label="candidate49_excluded",
    )
    matched = _trend_for_records(
        local_records,
        label="candidate49_and_local_samples_matched_location_05",
        j2d_field="j2d_matched_location_05",
    )
    valid_local = [item for item in records if item["morphology_id"].startswith("local_") and item.get("status") == "success"]
    resolved_fraction = full["pairwise_rank"]["non_tied_concordance_fraction"]
    pair_fraction = 1.0 - float(full["pairwise_rank"]["tie_fraction"] or 0.0)
    if full["sample_count"] < LOCAL_SAMPLE_COUNT + 1:
        outcome = "T6_INCONCLUSIVE"
    elif full["dynamic_range"]["j2d_plateau"] and full["dynamic_range"]["j3d_path_plateau"]:
        outcome = "T3_ROBUST_LOCAL_PLATEAU"
    elif full["dynamic_range"]["j3d_path_plateau"] or pair_fraction < 0.5:
        outcome = "T6_INCONCLUSIVE"
    elif resolved_fraction is not None and resolved_fraction >= TREND_THRESHOLDS["strong_concordance_fraction"] and without_candidate["pairwise_rank"]["non_tied_concordance_fraction"] >= TREND_THRESHOLDS["moderate_concordance_fraction"] and matched["pairwise_rank"]["non_tied_concordance_fraction"] >= TREND_THRESHOLDS["moderate_concordance_fraction"]:
        outcome = "T1_STRONG_LOCAL_TREND_PRESERVATION"
    elif resolved_fraction is not None and resolved_fraction >= TREND_THRESHOLDS["moderate_concordance_fraction"]:
        outcome = "T2_MODERATE_PARTIAL_PRESERVATION"
    elif without_candidate["pairwise_rank"]["non_tied_concordance_fraction"] is not None and without_candidate["pairwise_rank"]["non_tied_concordance_fraction"] < TREND_THRESHOLDS["inverse_concordance_fraction"]:
        outcome = "T5_LOCAL_TREND_INVERSION_FAILURE"
    elif full["candidate49"]["rank_j2d"] == 1.0 and full["candidate49"].get("rank_j3d_path_interval", {}).get("best_possible_rank") == 1:
        outcome = "T4_CANDIDATE_SPECIFIC_AGREEMENT_ONLY"
    else:
        outcome = "T6_INCONCLUSIVE"
    summary = {
        "schema_version": 1,
        "created_at": _now(),
        "precommit_fingerprint": precommit["precommit_fingerprint"],
        "records": records,
        "valid_local_sample_count": len(valid_local),
        "trend": {"full": full, "candidate49_excluded": without_candidate, "matched_location_05": matched},
        "outcome": outcome,
        "claim": (
            "Within the predefined local morphology neighborhood, the frozen 2D "
            "evaluator preserves the ordering/trend of the matched intrinsic 3D "
            "OptiX J3D-path metric sufficiently for local screening."
            if outcome == "T1_STRONG_LOCAL_TREND_PRESERVATION"
            else "The local evidence does not support a strong 2D-to-intrinsic-3D screening claim; interpretation is limited to the reported neighborhood and outcome category."
        ),
        "limitations": [
            "camera and Fisher/CRLB analyses are out of scope",
            "transport uses deterministic 3D geometric optics on an extruded mechanically deformed 2D cross-section",
            "J3D-path fidelity is fixed to the existing 262144-ray persisted field configuration",
            "pairwise discordance is resolved only under the precommitted global tolerance; no per-sample secondary realization was run",
        ],
    }
    atomic_write_json(output / "trend_summary.json", summary)
    return summary


def _make_figures(output: Path, summary: Mapping[str, Any]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    records = [item for item in summary["records"] if item["morphology_id"] != "nominal" and item.get("j2d") is not None and item.get("j3d_path") is not None]
    names = [item["morphology_id"] for item in records]
    j2d = np.asarray([item["j2d"] for item in records], dtype=float)
    j3d = np.asarray([item["j3d_path"] for item in records], dtype=float)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(j2d[1:], j3d[1:], label="local samples")
    axis.scatter(j2d[0], j3d[0], marker="*", s=120, label="candidate49")
    axis.set_xlabel("J2D")
    axis.set_ylabel("J3D-path")
    axis.legend()
    path = figures / "j2d_vs_j3d_path.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    full_scores = summary["trend"]["full"]["scores"]
    fig, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(names))
    axis.plot(positions, [full_scores[name]["rank_j2d"] for name in names], "o-", label="J2D rank")
    axis.plot(positions, [full_scores[name]["rank_j3d_path"] for name in names], "s-", label="J3D-path rank")
    axis.set_xticks(positions, names, rotation=60, ha="right")
    axis.set_ylabel("rank (1 = highest)")
    axis.legend()
    path = figures / "rank_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    pair = summary["trend"]["full"]["pairwise_rank"]
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.bar(["concordant", "discordant", "tied"], [pair["concordant_count"], pair["discordant_count"], pair["tied_count"]])
    axis.set_ylabel("pair count")
    path = figures / "pairwise_concordance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    map_records = records
    xi = np.asarray(
        [
            [
                0.0
                if item["normalized_local_coordinate"] is None
                else float(item["normalized_local_coordinate"][name])
                for name, _, _ in SWEPT_RANGES
            ]
            for item in map_records
        ]
    )
    fig, axis = plt.subplots(figsize=(9, 5))
    image = axis.imshow(xi.T, aspect="auto", cmap="coolwarm", vmin=-LOCAL_HALF_WIDTH, vmax=LOCAL_HALF_WIDTH)
    axis.set_yticks(np.arange(len(SWEPT_RANGES)), [name for name, _, _ in SWEPT_RANGES])
    axis.set_xticks(np.arange(len(map_records)), [item["morphology_id"] for item in map_records], rotation=60, ha="right")
    axis.set_xlabel("candidate49 and ordered local samples")
    axis.set_title("normalized local parameter coordinates")
    fig.colorbar(image, ax=axis, label="xi")
    path = figures / "local_parameter_map.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    excluded = summary["trend"]["candidate49_excluded"]
    fig, axis = plt.subplots(figsize=(7, 5))
    excluded_names = excluded["morphology_ids"]
    excluded_scores = excluded["scores"]
    axis.scatter([excluded_scores[name]["j2d"] for name in excluded_names], [excluded_scores[name]["j3d_path"] for name in excluded_names])
    axis.set_xlabel("J2D (candidate49 excluded)")
    axis.set_ylabel("J3D-path")
    path = figures / "candidate49_excluded_trend.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def finalize(output: Path = OUTPUT) -> dict[str, Any]:
    summary = analyze_trend(output)
    figure_paths = _make_figures(output, summary)
    summary["figures"] = figure_paths
    summary_path = output / "summary.json"
    manifest_path = output / "run_manifest.json"
    atomic_write_json(summary_path, summary)
    manifest = strict_read_json(manifest_path)
    manifest.update(
        {
            "stage": "analysis_complete",
            "status": "COMPLETE",
            "updated_at": _now(),
            "advisor_status": manifest.get("advisor_status", "consulted"),
            "reviewer_status": "pending_final_review",
            "artifacts_generated": [
                str(output / "precommit.json"),
                str(output / "local_2d_results.json"),
                str(output / "local_3d_results.json"),
                str(output / "trend_summary.json"),
                str(summary_path),
                *figure_paths,
            ],
            "scientific_outcome": summary["outcome"],
            "blockers": [],
            "next_authorized_action": "independent_review",
        }
    )
    atomic_write_json(manifest_path, manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("precommit", "2d", "3d", "analyze", "finalize"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.stage == "precommit":
        write_precommit(args.output)
        print("PRECOMMITTED")
    elif args.stage == "2d":
        run_frozen_2d(args.output)
        print("2D COMPLETE")
    elif args.stage == "3d":
        run_intrinsic_3d(args.output)
        print("3D COMPLETE")
    elif args.stage == "analyze":
        analyze_trend(args.output)
        print("ANALYSIS COMPLETE")
    else:
        summary = finalize(args.output)
        print(json.dumps({"outcome": summary["outcome"], "figures": summary["figures"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
