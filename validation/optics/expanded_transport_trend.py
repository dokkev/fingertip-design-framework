"""Validate the broader intrinsic 2D-to-3D transport trend around candidate49.

This continuation keeps the completed narrow study immutable.  It precommits
two candidate49-centred morphology shells, reuses the validated search-tier
FEA path, and evaluates the frozen 2D score against the established intrinsic
3D ``J3D-path`` metric.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, qmc, spearmanr

from fem.indentation import IndentationSettings, run_indentation_case
from mesh.fingertip import generate_fingertip_mesh
from mesh.indenter import IndenterSettings, build_normal_indenter_fixture_at_x
from model import Fingertip, FingertipParameters, validate_silicone_ligament
from model.fingertip_model import FingertipModel
from optics import trace
from optics.metrics import evaluate as evaluate_optics
from optics.metrics import field_difference
from validation.common.io import atomic_write_json, strict_read_json
from validation.fem.throughput import _mesh_policies
from validation.optimization.nominal_sweep import (
    FIXED_FLAT_PAD_WIDTH_MM,
    SWEPT_RANGES,
)
from validation.optics.transport3d_validation import (
    DEPTH_MM,
    FIELD_RESOLUTIONS,
    VALIDATION_MAX_PERIODIC_WRAPS,
    _internal_path_tv,
    _state_trace_3d,
)
from optics.transport3d.optix_backend import create_runtime


OUTPUT = Path("output/validation/optics/expanded_transport_trend")
PREVIOUS_LOCAL_OUTPUT = Path("output/validation/optics/local_transport_trend")
BRIDGE_SUMMARY = Path(
    "output/validation/optics/transport3d/internal_bridge_convergence/summary.json"
)
CANDIDATE_INPUT = Path(
    "output/validation/optimization/pre_bo_nominal_sweep/inputs/candidate_0049.json"
)
FAST_FEA_SUMMARY = Path("output/validation/fem/throughput/summary.json")

SEED = 20260816
POOL_SIZE_PER_SHELL = 512
SHELLS = {
    "shell_a": {"inner": 0.10, "outer": 0.20, "target": 25},
    "shell_b": {"inner": 0.20, "outer": 0.30, "target": 25},
}
FAST_STEPS = 12
FAST_MESH_POLICY = "coarse_b"
FAST_DIAGNOSTIC_MODE = "minimal"
RAY_COUNT = 262_144
FIELD_LEVEL = "current"
J3D_TIE_TOLERANCE = 0.06
SECONDARY_THRESHOLDS = (0.0, 0.016205, 0.033622, 0.06)
J2D_TIE_TOLERANCE = 1.0e-10
EFFECTIVE_SHELL_SEEDS = {"shell_a": SEED, "shell_b": SEED + 1}

VARIABLE_NAMES = tuple(name for name, _, _ in SWEPT_RANGES)
EXPECTED_BOUNDS = {
    "flat_pad_height": (3.5, 6.5),
    "semielliptical_pad_height": (7.0, 11.0),
    "stem_width": (6.5, 9.0),
    "stem_height": (5.0, 7.5),
    "void_width": (0.5, 2.0),
    "void_height": (0.0, 1.5),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _candidate_parameters() -> dict[str, Any]:
    payload = strict_read_json(CANDIDATE_INPUT)
    return {str(key): value for key, value in payload["parameters"].items()}


def _nominal_parameters() -> dict[str, Any]:
    return {str(key): value for key, value in asdict(FingertipParameters()).items()}


def _authoritative_design_space() -> dict[str, Any]:
    candidate = _candidate_parameters()
    actual = {
        name: (float(lower), float(upper)) for name, lower, upper in SWEPT_RANGES
    }
    expected = {name: tuple(value) for name, value in EXPECTED_BOUNDS.items()}
    if actual != expected:
        raise RuntimeError(
            "authoritative SWEPT_RANGES differs from the source that produced "
            f"candidate49: actual={actual}, expected={expected}"
        )
    if FIXED_FLAT_PAD_WIDTH_MM != 30.0:
        raise RuntimeError("authoritative fixed flat_pad_width is not 30 mm")
    return {
        "source": "validation/optimization/nominal_sweep.py",
        "variable_order": list(VARIABLE_NAMES),
        "variables": [
            {
                "name": name,
                "lower_mm": float(lower),
                "upper_mm": float(upper),
                "range_mm": float(upper - lower),
                "candidate49_mm": float(candidate[name]),
            }
            for name, lower, upper in SWEPT_RANGES
        ],
        "fixed_parameters": {"flat_pad_width": FIXED_FLAT_PAD_WIDTH_MM},
        "definition": "authoritative pre-BO nominal morphology sweep SWEPT_RANGES",
    }


def _morphology_fingerprint(parameters: Mapping[str, Any]) -> str:
    return _sha256_payload({str(key): parameters[key] for key in sorted(parameters)})


def _geometry_check(parameters: Mapping[str, Any]) -> tuple[bool, str | None]:
    try:
        typed = FingertipParameters(**dict(parameters))
        validate_silicone_ligament(typed)
        Fingertip(typed)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def _source_provenance() -> dict[str, str]:
    paths = [
        Path(__file__),
        Path("validation/optimization/nominal_sweep.py"),
        Path("validation/fem/throughput.py"),
        Path("validation/optics/transport3d_validation.py"),
        Path("optics/transport3d/transport.py"),
        Path("optics/transport3d/geometry.py"),
        Path("optics/transport3d/physics.py"),
        Path("optics/transport3d/sampling.py"),
        Path("optics/transport3d/settings.py"),
        Path("optics/transport3d/result.py"),
        Path("optics/transport3d/optix_backend.py"),
        Path("fem/indentation.py"),
        Path("mesh/fingertip.py"),
        Path("mesh/indenter.py"),
        Path("model/fingertip_model.py"),
        Path("optics/transport.py"),
        Path("optics/metrics.py"),
        Path("optics/cross_section/domain.py"),
        Path("optics/cross_section/result.py"),
        Path("optics/cross_section/settings.py"),
        Path("optics/cross_section/transport.py"),
        Path("model/fingertip_parameters.py"),
        CANDIDATE_INPUT,
        FAST_FEA_SUMMARY,
        BRIDGE_SUMMARY,
        BRIDGE_SUMMARY.with_name("fields.npz"),
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"required provenance files are missing: {missing}")
    return {str(path): _sha256_file(path) for path in paths}


def _shell_pool(shell_id: str, spec: Mapping[str, float]) -> list[dict[str, Any]]:
    design = _authoritative_design_space()
    candidate = _candidate_parameters()
    outer = float(spec["outer"])
    inner = float(spec["inner"])
    lower_xi = []
    upper_xi = []
    for variable in design["variables"]:
        span = float(variable["range_mm"])
        candidate_value = float(candidate[variable["name"]])
        lower_xi.append(
            max(-outer, (float(variable["lower_mm"]) - candidate_value) / span)
        )
        upper_xi.append(
            min(outer, (float(variable["upper_mm"]) - candidate_value) / span)
        )
    sampler_seed = EFFECTIVE_SHELL_SEEDS[shell_id]
    sampler = qmc.LatinHypercube(d=len(VARIABLE_NAMES), scramble=True, seed=sampler_seed)
    unit_points = sampler.random(n=POOL_SIZE_PER_SHELL)
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pool_index, unit_point in enumerate(unit_points, start=1):
        xi = {
            name: float(lo + value * (hi - lo))
            for name, value, lo, hi in zip(
                VARIABLE_NAMES, unit_point, lower_xi, upper_xi, strict=True
            )
        }
        radius = max(abs(value) for value in xi.values())
        parameters = _nominal_parameters()
        parameters["flat_pad_width"] = FIXED_FLAT_PAD_WIDTH_MM
        for name, lower, upper in SWEPT_RANGES:
            parameters[name] = float(candidate[name]) + xi[name] * (upper - lower)
        fingerprint = _morphology_fingerprint(parameters)
        in_bounds = all(
            float(lower) - 1.0e-12 <= float(parameters[name]) <= float(upper) + 1.0e-12
            for name, lower, upper in SWEPT_RANGES
        )
        if not in_bounds:
            valid = False
            reason = "original_bound_violation"
        elif not inner < radius <= outer:
            valid = False
            reason = "outside_requested_shell"
        elif fingerprint in seen:
            valid = False
            reason = "duplicate_physical_morphology"
        else:
            valid, reason = _geometry_check(parameters)
        if valid:
            seen.add(fingerprint)
        pool.append(
            {
                "pool_index": pool_index,
                "shell": shell_id,
                "requested_normalized_coordinate": xi,
                "normalized_coordinate": xi,
                "parameters": parameters,
                "morphology_fingerprint": fingerprint,
                "geometry_valid": bool(valid),
                "rejection_reason": reason,
            }
        )
    return pool


def _precommit_payload() -> dict[str, Any]:
    design = _authoritative_design_space()
    candidate = _candidate_parameters()
    pools = {shell_id: _shell_pool(shell_id, spec) for shell_id, spec in SHELLS.items()}
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for shell_id, spec in SHELLS.items():
        valid = [item for item in pools[shell_id] if item["geometry_valid"]]
        target = int(spec["target"])
        if len(valid) < target:
            raise RuntimeError(
                f"{shell_id} pool produced only {len(valid)} valid samples; "
                f"need {target}"
            )
        selected_ids = {item["pool_index"] for item in valid[:target]}
        for index, item in enumerate(valid[:target], start=1):
            selected.append(
                {
                    **item,
                    "sample_id": f"{shell_id}_{index:03d}",
                    "selection_order": index,
                }
            )
        for item in pools[shell_id]:
            if item["pool_index"] in selected_ids:
                continue
            reason = item["rejection_reason"] or "valid_after_target_count"
            rejected.append({**item, "rejection_reason": reason})
    bridge = strict_read_json(BRIDGE_SUMMARY)
    fast = strict_read_json(FAST_FEA_SUMMARY)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": _now(),
        "authoritative_design_space": design,
        "candidate49_parameters": candidate,
        "candidate49_fingerprint": _morphology_fingerprint(candidate),
        "nominal_parameters": _nominal_parameters(),
        "shell_definitions": SHELLS,
        "sampling": {
            "method": "fixed-seed LatinHypercube with deterministic rejection",
            "seed": SEED,
            "effective_seeds": dict(EFFECTIVE_SHELL_SEEDS),
            "pool_size_per_shell": POOL_SIZE_PER_SHELL,
            "selection": "first 25 geometry-valid samples per shell",
            "score_inspection_during_selection": False,
            "clipping": "none; physical values are generated directly inside bound intervals",
            "shell_intervals": {
                shell_id: {
                    "inner_inf_norm_exclusive": float(spec["inner"]),
                    "outer_inf_norm_inclusive": float(spec["outer"]),
                }
                for shell_id, spec in SHELLS.items()
            },
        },
        "ordered_candidate_pools": pools,
        "selected_samples": selected,
        "rejected_candidates": rejected,
        "sampling_diagnostics": {
            "selected_count": len(selected),
            "shell_counts": {
                shell_id: sum(item["shell"] == shell_id for item in selected)
                for shell_id in SHELLS
            },
            "unique_fingerprints": len({item["morphology_fingerprint"] for item in selected}),
        },
        "fea_policy": {
            "tier": "validated_search_fast_path",
            "mesh_policy": FAST_MESH_POLICY,
            "steps": FAST_STEPS,
            "diagnostic_mode": FAST_DIAGNOSTIC_MODE,
            "continuation": "0 -> 0.5 mm snapshot -> 1.0 mm",
            "internal_contact": "three_pairs",
            "indenter_radius_mm": 4.0,
            "locations_x_mm": [-3.0, 3.0],
            "symmetry_reuse": False,
            "source_artifact": str(FAST_FEA_SUMMARY),
            "source_artifact_sha256": _sha256_file(FAST_FEA_SUMMARY),
            "selection": fast.get("recommended_configuration"),
        },
        "optix_policy": {
            "ray_count": RAY_COUNT,
            "field_level": FIELD_LEVEL,
            "extrusion_depth_mm": DEPTH_MM,
            "z_bounds_mm": [-5.5, 5.5],
            "source_z_mm": 0.0,
            "metric": "J3D-path: mass-preserving TV of z-integrated P3_xy",
        },
        "analysis_plan": {
            "primary_j3d_tie_tolerance": J3D_TIE_TOLERANCE,
            "secondary_thresholds": list(SECONDARY_THRESHOLDS),
            "j2d_tie_tolerance": J2D_TIE_TOLERANCE,
            "regional_population": "selected expanded samples; candidate49 fast anchor reported separately and included where labeled",
            "nominal_population": "historical/global anchor only; excluded from regional correlations",
            "relationships": ["J2D_full -> J3D-path", "J2D_matched -> J3D-path"],
            "j2d_full_definition": (
                "the frozen minimum-separability scoring functional evaluated on "
                "the validated coarse_b/12-step fast-FEA tier; it is not the "
                "historical medium/48-step numerical realization"
            ),
            "outcome_rules": {
                "E1": "both shells resolved_fraction>=0.5, rho>=0.5, concordance>=0.75",
                "E2": "combined resolved_fraction>=0.5 and positive shell/combined full-score trend; immediate historical neighborhood remains unresolved",
                "E3": "positive but weaker, discordant, or parameter-direction-dependent trend",
                "E4": "matched-transition trend materially stronger than full evaluator trend",
                "E5": "expanded resolved_fraction remains below 0.5 across shells",
                "E6": "resolved expanded variation exists but full evaluator trend is non-positive",
                "E7": "implementation or numerical validity blocker",
            },
        },
        "historical_evidence": {
            "narrow_output": str(PREVIOUS_LOCAL_OUTPUT),
            "interpretation": "frozen ±10% study remains fine-ranking unresolved at primary tolerance",
            "bridge_summary": str(BRIDGE_SUMMARY),
            "bridge_ray_convergence_pass": bridge.get("ray_convergence_pass"),
        },
    }
    payload["provenance"] = {
        "hash_algorithm": "sha256",
        "source_and_artifact_hashes": _source_provenance(),
    }
    fingerprint_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at", "provenance"}
    }
    payload["precommit_fingerprint"] = _sha256_payload(fingerprint_payload)
    return payload


def _verify_precommit(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise RuntimeError("expanded precommit schema is stale")
    expected_sources = payload.get("provenance", {}).get("source_and_artifact_hashes")
    if expected_sources != _source_provenance():
        raise RuntimeError("expanded precommit source/artifact hashes are stale")
    expected_fingerprint = _sha256_payload(
        {key: value for key, value in payload.items() if key not in {"created_at", "provenance", "precommit_fingerprint"}}
    )
    if payload.get("precommit_fingerprint") != expected_fingerprint:
        raise RuntimeError("expanded precommit fingerprint mismatch")
    if payload.get("candidate49_parameters") != _candidate_parameters():
        raise RuntimeError("candidate49 vector changed since precommit")
    if payload.get("authoritative_design_space") != _authoritative_design_space():
        raise RuntimeError("authoritative design space changed since precommit")
    selected = payload.get("selected_samples", [])
    if len(selected) != 50:
        raise RuntimeError("expanded precommit must contain exactly 50 selected samples")
    fingerprints: set[str] = set()
    for sample in selected:
        params = sample.get("parameters")
        if not isinstance(params, Mapping):
            raise RuntimeError(f"{sample.get('sample_id')} lacks parameters")
        fingerprint = _morphology_fingerprint(params)
        if fingerprint != sample.get("morphology_fingerprint") or fingerprint in fingerprints:
            raise RuntimeError(f"invalid or duplicate fingerprint for {sample.get('sample_id')}")
        fingerprints.add(fingerprint)
        xi = sample.get("normalized_coordinate")
        if not isinstance(xi, Mapping):
            raise RuntimeError(f"{sample.get('sample_id')} lacks normalized coordinate")
        for name, lower, upper in SWEPT_RANGES:
            expected = float(_candidate_parameters()[name]) + float(xi[name]) * (upper - lower)
            if not np.isclose(float(params[name]), expected, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(f"{sample.get('sample_id')} parameter binding mismatch")
            if not float(lower) <= float(params[name]) <= float(upper):
                raise RuntimeError(f"{sample.get('sample_id')} violates original bounds")
        radius = max(abs(float(value)) for value in xi.values())
        spec = SHELLS[str(sample["shell"])]
        if not float(spec["inner"]) < radius <= float(spec["outer"]):
            raise RuntimeError(f"{sample.get('sample_id')} violates shell definition")


def _manifest_path(output: Path) -> Path:
    return output / "run_manifest.json"


def _read_manifest(output: Path) -> dict[str, Any]:
    path = _manifest_path(output)
    return strict_read_json(path) if path.exists() else {}


def _update_manifest(output: Path, **updates: Any) -> None:
    manifest = _read_manifest(output)
    manifest.update(updates)
    manifest["updated_at"] = _now()
    atomic_write_json(_manifest_path(output), manifest)


def write_precommit(output: Path = OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "expanded_precommit.json"
    invalidated: list[dict[str, Any]] = []
    if path.exists():
        existing = strict_read_json(path)
        try:
            _verify_precommit(existing)
            payload = existing
        except RuntimeError as exc:
            raise RuntimeError(
                f"existing expanded precommit is stale and immutable: {exc}; "
                "remove only this generated artifact and restart precommit if "
                "the pre-run experiment has not been authorized"
            ) from exc
    else:
        payload = _precommit_payload()
        atomic_write_json(path, payload)
    previous = _read_manifest(output)
    manifest = {
        "schema_version": 1,
        "stage": "precommit",
        "status": "PRECOMMITTED",
        "created_at": payload["created_at"],
        "updated_at": _now(),
        "precommit_path": str(path),
        "precommit_fingerprint": payload["precommit_fingerprint"],
        "advisor_status": previous.get("advisor_status", "pending_pre_run_consultation"),
        "reviewer_pre_run_status": previous.get("reviewer_pre_run_status", "pending_precommit_review"),
        "reviewer_guardrail_status": previous.get("reviewer_guardrail_status", "pending_guardrail_review"),
        "reviewer_final_status": previous.get("reviewer_final_status"),
        "completed_2d_samples": previous.get("completed_2d_samples", []),
        "completed_fea_samples": previous.get("completed_fea_samples", []),
        "completed_3d_samples": previous.get("completed_3d_samples", []),
        "reused_artifacts": previous.get("reused_artifacts", []),
        "invalidated_artifacts": [*previous.get("invalidated_artifacts", []), *invalidated],
        "scientific_outcome": previous.get("scientific_outcome"),
        "blockers": previous.get("blockers", []),
    }
    atomic_write_json(_manifest_path(output), manifest)
    return payload


def authorize_pre_run(output: Path = OUTPUT, *, advisor_note: str, reviewer_note: str) -> None:
    """Persist the external consultation/gate decision before expensive stages."""
    manifest = _read_manifest(output)
    manifest.update(
        {
            "advisor_status": "consulted",
            "advisor_note": advisor_note,
            "reviewer_pre_run_status": "PASS",
            "reviewer_pre_run_note": reviewer_note,
            "stage": "pre_run_gate_passed",
            "updated_at": _now(),
        }
    )
    atomic_write_json(_manifest_path(output), manifest)


def _require_gate(output: Path, *, require_guardrail: bool = True) -> None:
    manifest = _read_manifest(output)
    if manifest.get("advisor_status") != "consulted":
        raise RuntimeError("advisor consultation is required before expensive stages")
    if manifest.get("reviewer_pre_run_status") != "PASS":
        raise RuntimeError("pre-run reviewer PASS is required before expensive stages")
    if require_guardrail and manifest.get("reviewer_guardrail_status") != "PASS":
        raise RuntimeError("same-reviewer guardrail PASS is required before the 50-sample sweep")


def authorize_guardrail(output: Path = OUTPUT, *, reviewer_note: str) -> None:
    manifest = _read_manifest(output)
    if manifest.get("guardrail_status") != "PASS":
        raise RuntimeError("a computationally passing guardrail is required before reviewer authorization")
    manifest.update(
        {
            "reviewer_guardrail_status": "PASS",
            "reviewer_guardrail_note": reviewer_note,
            "updated_at": _now(),
        }
    )
    atomic_write_json(_manifest_path(output), manifest)


def _sample_record(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(sample["sample_id"]),
        "shell": str(sample["shell"]),
        "normalized_coordinate": sample["normalized_coordinate"],
        "parameters": sample["parameters"],
        "morphology_fingerprint": sample["morphology_fingerprint"],
        "precommit_fingerprint": sample["_precommit_fingerprint"],
        "process_fingerprint": sample["_process_fingerprint"],
    }


def _process_fingerprint(precommit: Mapping[str, Any]) -> str:
    return _sha256_payload(
        {
            "precommit_fingerprint": precommit["precommit_fingerprint"],
            "fea_policy": precommit["fea_policy"],
            "optix_policy": precommit["optix_policy"],
            "source_and_artifact_hashes": precommit["provenance"]["source_and_artifact_hashes"],
        }
    )


def _optix_configuration() -> dict[str, Any]:
    resolution = FIELD_RESOLUTIONS[FIELD_LEVEL]
    return {
        "mode": "full3d",
        "ray_count": RAY_COUNT,
        "field_level": FIELD_LEVEL,
        "minimum_ray_weight": 1.0e-4,
        "maximum_segment_count": max(20000, 24 * RAY_COUNT),
        "maximum_periodic_wraps": VALIDATION_MAX_PERIODIC_WRAPS,
        "terminate_on_periodic_wrap_limit": True,
        "terminate_on_no_event": True,
        "extrusion_depth_mm": DEPTH_MM,
        "z_bounds_mm": [-DEPTH_MM / 2.0, DEPTH_MM / 2.0],
        "source_z_mm": 0.0,
        "internal_grid_width": resolution["x_bins"],
        "internal_grid_height": resolution["y_bins"],
        "internal_z_bins": resolution["z_bins"],
        "retain_internal_path_field": True,
        "contact_states": ["left_0p5", "right_0p5"],
    }


def _execution_sample(sample: Mapping[str, Any], precommit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(sample),
        "_precommit_fingerprint": precommit["precommit_fingerprint"],
        "_process_fingerprint": _process_fingerprint(precommit),
    }


def _fast_policy() -> Any:
    return next(policy for policy in _mesh_policies() if policy.name == FAST_MESH_POLICY)


def _fea_directory(output: Path, sample: Mapping[str, Any]) -> Path:
    return output / "fea" / str(sample["morphology_fingerprint"])


def _fea_paths(output: Path, sample: Mapping[str, Any]) -> tuple[Path, Path]:
    directory = _fea_directory(output, sample)
    return directory / "states.npz", directory / "fea_record.json"


def _load_fea_bundle(output: Path, sample: Mapping[str, Any]) -> dict[str, Any] | None:
    states_path, record_path = _fea_paths(output, sample)
    if not states_path.exists() or not record_path.exists():
        return None
    record = strict_read_json(record_path)
    if record.get("morphology_fingerprint") != sample.get("morphology_fingerprint"):
        return None
    if record.get("parameters") != sample.get("parameters"):
        return None
    if record.get("precommit_fingerprint") != sample.get("_precommit_fingerprint"):
        return None
    if record.get("process_fingerprint") != sample.get("_process_fingerprint"):
        return None
    if record.get("states_sha256") != _sha256_file(states_path):
        return None
    if record.get("status") != "PASS":
        return None
    policy = _fast_policy()
    expected_configuration = {
        "mesh_policy": FAST_MESH_POLICY,
        "mesh_settings": asdict(policy.settings),
        "steps": FAST_STEPS,
        "diagnostic_mode": FAST_DIAGNOSTIC_MODE,
        "continuation": "0 -> 0.5 mm snapshot -> 1.0 mm",
        "internal_contact": "three_pairs",
        "symmetry_reuse": False,
    }
    for key, expected in expected_configuration.items():
        if record.get(key) != expected:
            return None
    expected_scenarios = {
        "left": {"location_x_mm": -3.0, "indentation_mm": 1.0, "radius_mm": 4.0},
        "right": {"location_x_mm": 3.0, "indentation_mm": 1.0, "radius_mm": 4.0},
    }
    for label, expected in expected_scenarios.items():
        scenario = record.get("scenario_records", {}).get(label, {})
        if any(scenario.get(key) != value for key, value in expected.items()):
            return None
    parameters = FingertipParameters(**sample["parameters"])
    mesh = generate_fingertip_mesh(FingertipModel(parameters), policy.settings)
    contract = {
        "pad_node_ids": [int(value) for value in mesh.pad.node_ids],
        "pad_node_order_sha256": _sha256_array(mesh.pad.node_ids),
        "pad_coordinates_sha256": _sha256_array(mesh.pad.coordinates),
        "pad_triangles_sha256": _sha256_array(mesh.pad.triangles),
    }
    if record.get("reference_mesh_contract") != contract:
        return None
    with np.load(states_path, allow_pickle=False) as data:
        states = {key: np.asarray(data[key], dtype=float) for key in data.files}
    expected_keys = {"left_0p5", "left_1p0", "right_0p5", "right_1p0"}
    if set(states) != expected_keys or any(
        value.shape != mesh.pad.coordinates.shape for value in states.values()
    ):
        return None
    return {
        "record": record,
        "states": states,
        "states_path": states_path,
        "record_path": record_path,
        "artifact_sha256": _sha256_file(record_path),
    }


def _run_fast_fea(output: Path, sample: Mapping[str, Any]) -> dict[str, Any]:
    cached = _load_fea_bundle(output, sample)
    if cached is not None:
        return cached
    parameters = FingertipParameters(**sample["parameters"])
    model = FingertipModel(parameters)
    tip = Fingertip(parameters)
    policy = _fast_policy()
    mesh = generate_fingertip_mesh(model, policy.settings)
    states: dict[str, np.ndarray] = {}
    scenario_records: dict[str, Any] = {}
    started = time.perf_counter()
    for location, label in ((-3.0, "left"), (3.0, "right")):
        fixture = build_normal_indenter_fixture_at_x(
            model,
            location,
            IndenterSettings(radius_mm=4.0),
        )
        result, artifacts = run_indentation_case(
            model,
            "medium",
            IndentationSettings(1.0, FAST_STEPS),
            fixture_override=fixture,
            internal_contact_configuration="three_pairs",
            mesh_override=mesh,
            diagnostic_mode=FAST_DIAGNOSTIC_MODE,
        )
        if artifacts is None or result.get("solve_status") != "PASS":
            raise RuntimeError(
                f"fast FEA failed for {sample['sample_id']} {label}: "
                f"{result.get('failure_reason') or result.get('exception')}"
            )
        snapshots = artifacts.snapshots
        for depth, depth_label in ((0.5, "0p5"), (1.0, "1p0")):
            snapshot = snapshots.get(f"{depth:g}")
            if snapshot is None:
                raise RuntimeError(f"fast FEA missing {depth} mm snapshot for {label}")
            displacement = np.asarray(
                [snapshot["displacements"][int(node_id)] for node_id in mesh.pad.node_ids],
                dtype=float,
            )
            if not np.all(np.isfinite(displacement)):
                raise RuntimeError(f"fast FEA produced non-finite displacement for {label}")
            states[f"{label}_{depth_label}"] = displacement
        scenario_records[label] = {
            "location_x_mm": location,
            "indentation_mm": 1.0,
            "radius_mm": 4.0,
            "steps": FAST_STEPS,
            "diagnostic_mode": FAST_DIAGNOSTIC_MODE,
            "status": result.get("status"),
            "solve_status": result.get("solve_status"),
            "completed_increments": result.get("completed_increments"),
            "snapshot_steps": {
                key: int(value["step"]) for key, value in snapshots.items() if key in {"0.5", "1"}
            },
            "reaction_force_n": result.get("final", {}).get("indenter_normal_reaction_n"),
            "timing": result.get("timing", {}),
            "case_acceptance_checks": result.get("case_acceptance_checks", {}),
        }
    states_path, record_path = _fea_paths(output, sample)
    states_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(states_path, **states)
    record = {
        "schema_version": 1,
        "status": "PASS",
        "sample_id": sample["sample_id"],
        "shell": sample.get("shell"),
        "morphology_fingerprint": sample["morphology_fingerprint"],
        "parameters": sample["parameters"],
        "precommit_fingerprint": sample["_precommit_fingerprint"],
        "process_fingerprint": sample["_process_fingerprint"],
        "mesh_policy": FAST_MESH_POLICY,
        "mesh_settings": asdict(policy.settings),
        "steps": FAST_STEPS,
        "diagnostic_mode": FAST_DIAGNOSTIC_MODE,
        "continuation": "0 -> 0.5 mm snapshot -> 1.0 mm",
        "internal_contact": "three_pairs",
        "symmetry_reuse": False,
        "scenario_records": scenario_records,
        "reference_mesh_contract": {
            "pad_node_ids": [int(value) for value in mesh.pad.node_ids],
            "pad_node_order_sha256": _sha256_array(mesh.pad.node_ids),
            "pad_coordinates_sha256": _sha256_array(mesh.pad.coordinates),
            "pad_triangles_sha256": _sha256_array(mesh.pad.triangles),
        },
        "states_sha256": _sha256_file(states_path),
        "states_path": str(states_path),
        "wall_time_seconds": time.perf_counter() - started,
    }
    atomic_write_json(record_path, record)
    return {
        "record": record,
        "states": states,
        "states_path": states_path,
        "record_path": record_path,
        "artifact_sha256": _sha256_file(record_path),
    }


def _meshes_for_bundle(sample: Mapping[str, Any], bundle: Mapping[str, Any]) -> tuple[Fingertip, Any, dict[str, Any]]:
    parameters = FingertipParameters(**sample["parameters"])
    model = FingertipModel(parameters)
    tip = Fingertip(parameters)
    mesh = generate_fingertip_mesh(model, _fast_policy().settings)
    loaded = {
        key: mesh.pad.deformed(value, metadata={"condition": "fast_fea", "morphology_fingerprint": sample["morphology_fingerprint"]})
        for key, value in bundle["states"].items()
    }
    return tip, mesh, loaded


def _evaluate_2d(sample: Mapping[str, Any], bundle: Mapping[str, Any]) -> dict[str, Any]:
    tip, mesh, loaded = _meshes_for_bundle(sample, bundle)
    reference = trace(tip, mesh.pad)
    transports: dict[str, Any] = {}
    scenario_metrics: dict[str, Any] = {}
    for key, state in (
        ("left_0p5", loaded["left_0p5"]),
        ("right_0p5", loaded["right_0p5"]),
        ("left_1p0", loaded["left_1p0"]),
        ("right_1p0", loaded["right_1p0"]),
    ):
        result = trace(tip, state)
        transports[key] = result
        metrics = evaluate_optics(reference, result)
        scenario_metrics[key] = _jsonable(metrics)
    pair_definitions = (
        ("location", "left_0p5", "right_0p5"),
        ("location", "left_1p0", "right_1p0"),
        ("indentation", "left_0p5", "left_1p0"),
        ("indentation", "right_0p5", "right_1p0"),
    )
    pair_values = [
        {
            "axis": axis,
            "first": first,
            "second": second,
            "separability": float(field_difference(transports[first], transports[second])),
        }
        for axis, first, second in pair_definitions
    ]
    return {
        "j2d_full": min(item["separability"] for item in pair_values),
        "j2d_matched": pair_values[0]["separability"],
        "pair_values": pair_values,
        "scenario_metrics": scenario_metrics,
        "fea_artifact_id": str(bundle["record_path"]),
        "fea_artifact_sha256": bundle["artifact_sha256"],
        "morphology_fingerprint": sample["morphology_fingerprint"],
    }


def _save_progress(output: Path, filename: str, records: Mapping[str, Any]) -> None:
    atomic_write_json(output / filename, {"records": list(records.values()), "updated_at": _now()})


def _valid_2d_resume(
    output: Path,
    sample: Mapping[str, Any],
    record: Mapping[str, Any],
    precommit: Mapping[str, Any],
) -> bool:
    if record.get("status") != "PASS":
        return False
    if record.get("precommit_fingerprint") != precommit["precommit_fingerprint"]:
        return False
    if record.get("morphology_fingerprint") != sample["morphology_fingerprint"]:
        return False
    bundle = _load_fea_bundle(output, sample)
    return bool(
        bundle is not None
        and record.get("fea_artifact_sha256") == bundle["artifact_sha256"]
        and record.get("fea_artifact_id") == str(bundle["record_path"])
    )


def _valid_3d_resume(
    output: Path,
    sample: Mapping[str, Any],
    record: Mapping[str, Any],
    precommit: Mapping[str, Any],
) -> bool:
    if record.get("status") != "PASS":
        return False
    if record.get("precommit_fingerprint") != precommit["precommit_fingerprint"]:
        return False
    if record.get("morphology_fingerprint") != sample["morphology_fingerprint"]:
        return False
    bundle = _load_fea_bundle(output, sample)
    optix_path = Path(str(record.get("optix_artifact_id", "")))
    expected_configuration = {
        "ray_count": RAY_COUNT,
        "field_level": FIELD_LEVEL,
        "extrusion_depth_mm": DEPTH_MM,
        "contact_states": ["left_0p5", "right_0p5"],
        "source_z_mm": 0.0,
    }
    return bool(
        bundle is not None
        and record.get("fea_artifact_sha256") == bundle["artifact_sha256"]
        and record.get("fea_artifact_id") == str(bundle["record_path"])
        and optix_path.exists()
        and record.get("optix_artifact_sha256") == _sha256_file(optix_path)
        and record.get("optix_configuration") == expected_configuration
    )


def run_guardrail(output: Path = OUTPUT) -> dict[str, Any]:
    _verify_precommit(strict_read_json(output / "expanded_precommit.json"))
    _require_gate(output, require_guardrail=False)
    precommit = strict_read_json(output / "expanded_precommit.json")
    local_precommit = strict_read_json(PREVIOUS_LOCAL_OUTPUT / "precommit.json")
    local_sample = next(item for item in local_precommit["selected_valid_samples"] if item["sample_id"] == "local_001")
    samples = [
        {
            "sample_id": "nominal_fast_guardrail",
            "shell": "anchor",
            "parameters": precommit["nominal_parameters"],
            "normalized_coordinate": None,
            "morphology_fingerprint": _morphology_fingerprint(precommit["nominal_parameters"]),
        },
        {
            "sample_id": "candidate49_fast_guardrail",
            "shell": "anchor",
            "parameters": precommit["candidate49_parameters"],
            "normalized_coordinate": None,
            "morphology_fingerprint": _morphology_fingerprint(precommit["candidate49_parameters"]),
        },
        {
            "sample_id": "local_001_fast_guardrail",
            "shell": "historical_local",
            "parameters": local_sample["parameters"],
            "normalized_coordinate": local_sample["normalized_local_coordinate"],
            "morphology_fingerprint": _morphology_fingerprint(local_sample["parameters"]),
        },
    ]
    samples = [_execution_sample(sample, precommit) for sample in samples]
    runtime = create_runtime()
    records: dict[str, Any] = {}
    try:
        for sample in samples:
            bundle = _run_fast_fea(output, sample)
            two_d = _evaluate_2d(sample, bundle)
            tip, mesh, loaded = _meshes_for_bundle(sample, bundle)
            left = _state_trace_3d(
                tip, loaded["left_0p5"], mesh, runtime, mode="full3d",
                ray_count=RAY_COUNT, retain_internal_path_field=True,
                field_resolution=FIELD_RESOLUTIONS[FIELD_LEVEL], minimum_ray_weight=1.0e-4,
            )
            right = _state_trace_3d(
                tip, loaded["right_0p5"], mesh, runtime, mode="full3d",
                ray_count=RAY_COUNT, retain_internal_path_field=True,
                field_resolution=FIELD_RESOLUTIONS[FIELD_LEVEL], minimum_ray_weight=1.0e-4,
            )
            j3d, grid = _internal_path_tv(left, right)
            records[sample["sample_id"]] = {
                **_sample_record(sample),
                **two_d,
                "j3d_path": float(j3d),
                "j3d_path_comparison_grid": grid,
                "fea_artifact_id": str(bundle["record_path"]),
                "fea_artifact_sha256": bundle["artifact_sha256"],
                "optix_configuration": _optix_configuration(),
                "validity": "PASS",
            }
            atomic_write_json(
                output / "guardrail_progress.json",
                {
                    "precommit_fingerprint": precommit["precommit_fingerprint"],
                    "process_fingerprint": _process_fingerprint(precommit),
                    "records": list(records.values()),
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
    bridge = strict_read_json(BRIDGE_SUMMARY)
    historical_2d_rows = strict_read_json(PREVIOUS_LOCAL_OUTPUT / "local_2d_results.json")["results"]
    historical_2d = {
        "nominal_fast_guardrail": next(item for item in historical_2d_rows if item["morphology_id"] == "nominal"),
        "candidate49_fast_guardrail": next(item for item in historical_2d_rows if item["morphology_id"] == "candidate49"),
        "local_001_fast_guardrail": next(item for item in historical_2d_rows if item["morphology_id"] == "local_001"),
    }
    historical = {
        "nominal_fast_guardrail": float(bridge["ray_convergence"][str(RAY_COUNT)][FIELD_LEVEL]["nominal"]["j3d_path"]),
        "candidate49_fast_guardrail": float(bridge["ray_convergence"][str(RAY_COUNT)][FIELD_LEVEL]["candidate49"]["j3d_path"]),
        "local_001_fast_guardrail": float(strict_read_json(PREVIOUS_LOCAL_OUTPUT / "local_3d_results.json")["results"][2]["j3d_path"]),
    }
    for key, record in records.items():
        record["historical_j2d_full"] = historical_2d[key]["j2d"]
        record["historical_j2d_matched"] = historical_2d[key]["j2d_matched_location_05"]
        record["fast_minus_historical_j2d_full"] = record["j2d_full"] - historical_2d[key]["j2d"]
        record["fast_minus_historical_j2d_matched"] = record["j2d_matched"] - historical_2d[key]["j2d_matched_location_05"]
        record["historical_high_fidelity_j3d_path"] = historical[key]
        record["j3d_delta_vs_historical"] = record["j3d_path"] - historical[key]
    ordering = {
        "fast_j2d_candidate49_above_nominal": records["candidate49_fast_guardrail"]["j2d_full"] > records["nominal_fast_guardrail"]["j2d_full"],
        "fast_j2d_matched_candidate49_above_nominal": records["candidate49_fast_guardrail"]["j2d_matched"] > records["nominal_fast_guardrail"]["j2d_matched"],
        "historical_j2d_candidate49_above_nominal": historical_2d["candidate49_fast_guardrail"]["j2d"] > historical_2d["nominal_fast_guardrail"]["j2d"],
        "fast_j3d_candidate49_above_nominal": records["candidate49_fast_guardrail"]["j3d_path"] > records["nominal_fast_guardrail"]["j3d_path"],
        "historical_j3d_ordering_candidate49_above_nominal": historical["candidate49_fast_guardrail"] > historical["nominal_fast_guardrail"],
    }
    payload = {
        "schema_version": 1,
        "status": (
            "PASS"
            if len(records) == 3
            and all(
                record.get("validity") == "PASS"
                and np.isfinite(float(record.get("j2d_full")))
                and np.isfinite(float(record.get("j2d_matched")))
                and np.isfinite(float(record.get("j3d_path")))
                for record in records.values()
            )
            else "BLOCKED"
        ),
        "records": list(records.values()),
        "historical_reference_j2d": {
            key: {
                "j2d_full": value["j2d"],
                "j2d_matched": value["j2d_matched_location_05"],
            }
            for key, value in historical_2d.items()
        },
        "historical_reference_j3d": historical,
        "ordering": ordering,
        "interpretation": "bounded fast-FEA/J3D guardrail; no equality threshold imposed",
        "precommit_fingerprint": precommit["precommit_fingerprint"],
        "process_fingerprint": _process_fingerprint(precommit),
    }
    atomic_write_json(output / "fast_fea_j3d_guardrail.json", payload)
    _update_manifest(
        output,
        stage="guardrail_complete",
        guardrail_status=payload["status"],
        reviewer_guardrail_status="pending_guardrail_review",
    )
    return payload


def run_2d(output: Path = OUTPUT) -> dict[str, Any]:
    _verify_precommit(strict_read_json(output / "expanded_precommit.json"))
    _require_gate(output)
    precommit = strict_read_json(output / "expanded_precommit.json")
    records: dict[str, Any] = {}
    progress_path = output / "expanded_2d_progress.json"
    if progress_path.exists():
        existing = strict_read_json(progress_path)
        records.update({str(item["sample_id"]): item for item in existing.get("records", [])})
    for item in precommit["selected_samples"]:
        sample_id = str(item["sample_id"])
        execution_item = _execution_sample(item, precommit)
        if sample_id in records and _valid_2d_resume(output, execution_item, records[sample_id], precommit):
            continue
        bundle = _run_fast_fea(output, execution_item)
        two_d = _evaluate_2d(execution_item, bundle)
        records[sample_id] = {
            **_sample_record(execution_item),
            **two_d,
            "status": "PASS",
            "precommit_fingerprint": precommit["precommit_fingerprint"],
            "fea_tier": precommit["fea_policy"],
            "j2d_full_tier": "validated_fast_fea",
            "historical_j2d": None,
        }
        _save_progress(output, "expanded_2d_progress.json", records)
    payload = {
        "schema_version": 1,
        "status": "COMPLETE" if len(records) == 50 else "INCOMPLETE",
        "precommit_fingerprint": precommit["precommit_fingerprint"],
        "results": [records[item["sample_id"]] for item in precommit["selected_samples"] if item["sample_id"] in records],
    }
    atomic_write_json(output / "expanded_2d_results.json", payload)
    _update_manifest(output, stage="expanded_2d_complete", completed_2d_samples=sorted(records))
    return payload


def run_fea(output: Path = OUTPUT) -> dict[str, Any]:
    _verify_precommit(strict_read_json(output / "expanded_precommit.json"))
    _require_gate(output)
    precommit = strict_read_json(output / "expanded_precommit.json")
    records = []
    for item in precommit["selected_samples"]:
        execution_item = _execution_sample(item, precommit)
        bundle = _load_fea_bundle(output, execution_item)
        if bundle is None:
            bundle = _run_fast_fea(output, execution_item)
        records.append({
            **_sample_record(execution_item),
            "precommit_fingerprint": precommit["precommit_fingerprint"],
            "status": bundle["record"]["status"],
            "fea_artifact_id": str(bundle["record_path"]),
            "fea_artifact_sha256": bundle["artifact_sha256"],
            "states_sha256": bundle["record"]["states_sha256"],
            "fea_record": bundle["record"],
        })
    payload = {
        "schema_version": 1,
        "status": "COMPLETE" if all(item["status"] == "PASS" for item in records) else "INCOMPLETE",
        "precommit_fingerprint": precommit["precommit_fingerprint"],
        "results": records,
    }
    atomic_write_json(output / "expanded_fea_results.json", payload)
    _update_manifest(output, stage="expanded_fea_complete", completed_fea_samples=[item["sample_id"] for item in records if item["status"] == "PASS"])
    return payload


def run_3d(output: Path = OUTPUT) -> dict[str, Any]:
    _verify_precommit(strict_read_json(output / "expanded_precommit.json"))
    _require_gate(output)
    precommit = strict_read_json(output / "expanded_precommit.json")
    two_d = strict_read_json(output / "expanded_2d_results.json")
    if two_d.get("status") != "COMPLETE":
        raise RuntimeError("complete expanded 2D results are required before 3D")
    records: dict[str, Any] = {}
    progress_path = output / "expanded_3d_progress.json"
    if progress_path.exists():
        existing = strict_read_json(progress_path)
        records.update({str(item["sample_id"]): item for item in existing.get("records", [])})
    runtime = create_runtime()
    try:
        for item in precommit["selected_samples"]:
            sample_id = str(item["sample_id"])
            execution_item = _execution_sample(item, precommit)
            if sample_id in records and _valid_3d_resume(output, execution_item, records[sample_id], precommit):
                continue
            bundle = _load_fea_bundle(output, execution_item)
            if bundle is None:
                raise RuntimeError(f"missing valid FEA artifact for {sample_id}")
            tip, mesh, loaded = _meshes_for_bundle(execution_item, bundle)
            left = _state_trace_3d(
                tip, loaded["left_0p5"], mesh, runtime, mode="full3d",
                ray_count=RAY_COUNT, retain_internal_path_field=True,
                field_resolution=FIELD_RESOLUTIONS[FIELD_LEVEL], minimum_ray_weight=1.0e-4,
            )
            right = _state_trace_3d(
                tip, loaded["right_0p5"], mesh, runtime, mode="full3d",
                ray_count=RAY_COUNT, retain_internal_path_field=True,
                field_resolution=FIELD_RESOLUTIONS[FIELD_LEVEL], minimum_ray_weight=1.0e-4,
            )
            j3d, grid = _internal_path_tv(left, right)
            field_path = output / "optix" / f"{sample_id}_{item['morphology_fingerprint']}.npz"
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
            )
            records[sample_id] = {
                **_sample_record(execution_item),
                "j3d_path": float(j3d),
                "j3d_path_comparison_grid": grid,
                "fea_artifact_id": str(bundle["record_path"]),
                "fea_artifact_sha256": bundle["artifact_sha256"],
                "precommit_fingerprint": precommit["precommit_fingerprint"],
                "optix_artifact_id": str(field_path),
                "optix_artifact_sha256": _sha256_file(field_path),
                "optix_configuration": {
                    "ray_count": RAY_COUNT,
                    "field_level": FIELD_LEVEL,
                    "extrusion_depth_mm": DEPTH_MM,
                    "contact_states": ["left_0p5", "right_0p5"],
                    "source_z_mm": 0.0,
                },
                "status": "PASS",
            }
            _save_progress(output, "expanded_3d_progress.json", records)
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
    payload = {
        "schema_version": 1,
        "status": "COMPLETE" if len(records) == 50 else "INCOMPLETE",
        "precommit_fingerprint": precommit["precommit_fingerprint"],
        "results": [records[item["sample_id"]] for item in precommit["selected_samples"] if item["sample_id"] in records],
    }
    atomic_write_json(output / "expanded_3d_results.json", payload)
    _update_manifest(output, stage="expanded_3d_complete", completed_3d_samples=sorted(records))
    return payload


def _pairwise(j2d: list[float], j3d: list[float], threshold: float) -> dict[str, Any]:
    concordant = discordant = tied = j2d_tied = 0
    for first in range(len(j2d)):
        for second in range(first + 1, len(j2d)):
            sign_2d = 0 if abs(j2d[first] - j2d[second]) <= J2D_TIE_TOLERANCE else (1 if j2d[first] > j2d[second] else -1)
            sign_3d = 0 if abs(j3d[first] - j3d[second]) <= threshold else (1 if j3d[first] > j3d[second] else -1)
            if sign_2d == 0:
                j2d_tied += 1
            if sign_3d == 0 or sign_2d == 0:
                tied += 1
            elif sign_2d == sign_3d:
                concordant += 1
            else:
                discordant += 1
    total = len(j2d) * (len(j2d) - 1) // 2
    resolved = concordant + discordant
    return {
        "threshold": threshold,
        "total_pairs": total,
        "concordant": concordant,
        "discordant": discordant,
        "tied": tied,
        "j2d_tied": j2d_tied,
        "resolved_pairs": resolved,
        "resolved_pair_fraction": resolved / total if total else None,
        "concordance_among_resolved": concordant / resolved if resolved else None,
    }


def _correlation(j2d: list[float], j3d: list[float]) -> dict[str, Any]:
    if len(j2d) < 2:
        return {"spearman_rho": None, "spearman_pvalue": None, "kendall_tau_b": None, "kendall_pvalue": None}
    spear = spearmanr(j2d, j3d)
    kendall = kendalltau(j2d, j3d, variant="b")
    return {
        "spearman_rho": float(spear.statistic) if np.isfinite(spear.statistic) else None,
        "spearman_pvalue": float(spear.pvalue) if np.isfinite(spear.pvalue) else None,
        "kendall_tau_b": float(kendall.statistic) if np.isfinite(kendall.statistic) else None,
        "kendall_pvalue": float(kendall.pvalue) if np.isfinite(kendall.pvalue) else None,
    }


def _population(records: list[Mapping[str, Any]], label: str) -> dict[str, Any]:
    j2d_full = [float(item["j2d_full"]) for item in records]
    j2d_matched = [float(item["j2d_matched"]) for item in records]
    j3d = [float(item["j3d_path"]) for item in records]
    return {
        "label": label,
        "sample_count": len(records),
        "morphology_ids": [item["sample_id"] for item in records],
        "correlation": {
            "j2d_full_to_j3d": _correlation(j2d_full, j3d),
            "j2d_matched_to_j3d": _correlation(j2d_matched, j3d),
        },
        "pairwise": {
            "j2d_full_to_j3d": [_pairwise(j2d_full, j3d, threshold) for threshold in SECONDARY_THRESHOLDS],
            "j2d_matched_to_j3d": [_pairwise(j2d_matched, j3d, threshold) for threshold in SECONDARY_THRESHOLDS],
        },
        "dynamic_range": {
            "j2d_full": {"min": min(j2d_full), "max": max(j2d_full), "range": max(j2d_full) - min(j2d_full)},
            "j2d_matched": {"min": min(j2d_matched), "max": max(j2d_matched), "range": max(j2d_matched) - min(j2d_matched)},
            "j3d_path": {"min": min(j3d), "max": max(j3d), "range": max(j3d) - min(j3d)},
        },
    }


def _parameter_diagnostics(records: list[Mapping[str, Any]], candidate: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for name in VARIABLE_NAMES:
        xi = [float(item["normalized_coordinate"][name]) for item in records]
        delta_full = [float(item["j2d_full"]) - float(candidate["j2d_full"]) for item in records]
        delta_matched = [float(item["j2d_matched"]) - float(candidate["j2d_matched"]) for item in records]
        delta_3d = [float(item["j3d_path"]) - float(candidate["j3d_path"]) for item in records]
        diagnostics[name] = {
            "spearman_xi_to_delta_j2d_full": _correlation(xi, delta_full),
            "spearman_xi_to_delta_j2d_matched": _correlation(xi, delta_matched),
            "spearman_xi_to_delta_j3d": _correlation(xi, delta_3d),
        }
    return diagnostics


def _make_figures(output: Path, records: list[Mapping[str, Any]], populations: Mapping[str, Any]) -> list[str]:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    colors = {"shell_a": "tab:blue", "shell_b": "tab:orange", "anchor": "black"}

    def scatter(xkey: str, filename: str, xlabel: str) -> None:
        fig, axis = plt.subplots(figsize=(7, 5))
        for shell in ("shell_a", "shell_b"):
            subset = [item for item in records if item["shell"] == shell]
            axis.scatter([item[xkey] for item in subset], [item["j3d_path"] for item in subset], label=shell, alpha=0.8, color=colors[shell])
        axis.set_xlabel(xlabel)
        axis.set_ylabel("J3D-path")
        axis.legend()
        fig.tight_layout()
        path = figure_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    scatter("j2d_full", "j2d_full_vs_j3d_path.png", "J2D_full")
    scatter("j2d_matched", "j2d_matched_vs_j3d_path.png", "J2D_matched")

    for shell in ("shell_a", "shell_b"):
        subset = [item for item in records if item["shell"] == shell]
        ranks_2d = np.argsort(np.argsort([-item["j2d_full"] for item in subset])) + 1
        ranks_3d = np.argsort(np.argsort([-item["j3d_path"] for item in subset])) + 1
        fig, axis = plt.subplots(figsize=(6, 5))
        axis.scatter(ranks_2d, ranks_3d, color=colors[shell])
        axis.plot([1, len(subset)], [1, len(subset)], "k--", linewidth=1)
        axis.set_xlabel("J2D_full rank")
        axis.set_ylabel("J3D-path rank")
        axis.set_title(f"{shell} rank comparison")
        fig.tight_layout()
        path = figure_dir / f"{shell}_rank_comparison.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    thresholds = list(SECONDARY_THRESHOLDS)
    fig, axis = plt.subplots(figsize=(7, 5))
    for label, population in populations.items():
        pairs = population["pairwise"]["j2d_full_to_j3d"]
        axis.plot(thresholds, [item["resolved_pair_fraction"] for item in pairs], marker="o", label=label)
    axis.set_xlabel("J3D tie threshold")
    axis.set_ylabel("resolved-pair fraction")
    axis.legend()
    fig.tight_layout()
    path = figure_dir / "resolved_pair_fraction_vs_threshold.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, axis = plt.subplots(figsize=(7, 5))
    labels = list(populations)
    x = np.arange(len(labels))
    rho = [populations[label]["correlation"]["j2d_full_to_j3d"]["spearman_rho"] for label in labels]
    concordance = [populations[label]["pairwise"]["j2d_full_to_j3d"][-1]["concordance_among_resolved"] for label in labels]
    axis.plot(x, rho, marker="o", label="Spearman rho")
    axis.plot(x, concordance, marker="s", label="concordance @0.06")
    axis.set_xticks(x, labels, rotation=20)
    axis.set_ylim(-1.05, 1.05)
    axis.legend()
    fig.tight_layout()
    path = figure_dir / "trend_vs_scale.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, axis = plt.subplots(figsize=(8, 5))
    for shell in ("shell_a", "shell_b"):
        subset = [item for item in records if item["shell"] == shell]
        axis.scatter(
            [item["normalized_coordinate"][VARIABLE_NAMES[0]] for item in subset],
            [item["normalized_coordinate"][VARIABLE_NAMES[1]] for item in subset],
            c=[item["j3d_path"] for item in subset], cmap="viridis", label=shell,
        )
    axis.set_xlabel(f"xi: {VARIABLE_NAMES[0]}")
    axis.set_ylabel(f"xi: {VARIABLE_NAMES[1]}")
    axis.set_title("sample coverage (color = J3D-path)")
    axis.legend()
    fig.tight_layout()
    path = figure_dir / "normalized_sample_coverage.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def analyze(output: Path = OUTPUT) -> dict[str, Any]:
    precommit = strict_read_json(output / "expanded_precommit.json")
    _verify_precommit(precommit)
    guardrail = strict_read_json(output / "fast_fea_j3d_guardrail.json")
    two_d = strict_read_json(output / "expanded_2d_results.json")
    three_d = strict_read_json(output / "expanded_3d_results.json")
    if guardrail.get("status") != "PASS" or two_d.get("status") != "COMPLETE" or three_d.get("status") != "COMPLETE":
        raise RuntimeError("guardrail and complete expanded 2D/3D results are required")
    by_id = {item["sample_id"]: dict(item) for item in two_d["results"]}
    by_id.update({item["sample_id"]: {**by_id.get(item["sample_id"], {}), **item} for item in three_d["results"]})
    selected = [by_id[item["sample_id"]] for item in precommit["selected_samples"]]
    candidate_anchor = next(item for item in guardrail["records"] if item["sample_id"] == "candidate49_fast_guardrail")
    anchor = {
        **candidate_anchor,
        "sample_id": "candidate49_fast_anchor",
        "shell": "anchor",
        "normalized_coordinate": {name: 0.0 for name in VARIABLE_NAMES},
    }
    populations: dict[str, Any] = {}
    for shell in ("shell_a", "shell_b"):
        shell_records = [item for item in selected if item["shell"] == shell]
        populations[f"{shell}_samples_only"] = _population(shell_records, f"{shell}_samples_only")
        populations[f"{shell}_with_candidate_anchor"] = _population([anchor, *shell_records], f"{shell}_with_candidate_anchor")
    populations["expanded_samples_only"] = _population(selected, "expanded_samples_only")
    populations["expanded_with_candidate_anchor"] = _population([anchor, *selected], "expanded_with_candidate_anchor")
    candidate_excluded = populations["expanded_samples_only"]
    parameter_diagnostics = _parameter_diagnostics(selected, anchor)
    figures = _make_figures(output, selected, populations)
    immediate = strict_read_json(PREVIOUS_LOCAL_OUTPUT / "summary.json")
    combined = populations["expanded_with_candidate_anchor"]
    regional = populations["expanded_samples_only"]
    shell_a = populations["shell_a_samples_only"]
    shell_b = populations["shell_b_samples_only"]
    primary = regional["pairwise"]["j2d_full_to_j3d"][-1]
    rho = regional["correlation"]["j2d_full_to_j3d"]["spearman_rho"]
    shell_primary = {
        shell: populations[f"{shell}_samples_only"]["pairwise"]["j2d_full_to_j3d"][-1]
        for shell in ("shell_a", "shell_b")
    }
    shell_rho = {
        shell: populations[f"{shell}_samples_only"]["correlation"]["j2d_full_to_j3d"]["spearman_rho"]
        for shell in ("shell_a", "shell_b")
    }
    if primary["resolved_pair_fraction"] < 0.5 and all(
        shell_primary[shell]["resolved_pair_fraction"] < 0.5 for shell in ("shell_a", "shell_b")
    ):
        outcome = "E5_BROAD_PLATEAU"
    elif rho is not None and rho <= 0.0 and primary["resolved_pair_fraction"] >= 0.5:
        outcome = "E6_TREND_FAILURE"
    elif (
        rho is not None
        and rho >= 0.5
        and primary["resolved_pair_fraction"] >= 0.5
        and all(shell_primary[shell]["concordance_among_resolved"] is not None and shell_primary[shell]["concordance_among_resolved"] >= 0.75 for shell in ("shell_a", "shell_b"))
        and all(shell_rho[shell] is not None and shell_rho[shell] >= 0.5 for shell in ("shell_a", "shell_b"))
    ):
        outcome = "E1_STRONG_REGIONAL_PROXY"
    elif (
        rho is not None
        and rho > 0.0
        and primary["resolved_pair_fraction"] >= 0.5
        and all(shell_primary[shell]["resolved_pair_fraction"] >= 0.5 for shell in ("shell_a", "shell_b"))
        and all(shell_rho[shell] is not None and shell_rho[shell] > 0.0 for shell in ("shell_a", "shell_b"))
    ):
        outcome = "E2_SCALE_DEPENDENT_PROXY"
    else:
        matched = regional["correlation"]["j2d_matched_to_j3d"]["spearman_rho"]
        outcome = "E4_MATCHED_TRANSITION_ONLY" if matched is not None and rho is not None and matched > rho + 0.2 else "E3_PARTIAL_PROXY"
    report = {
        "schema_version": 1,
        "status": "COMPLETE",
        "outcome": outcome,
        "precommit_fingerprint": precommit["precommit_fingerprint"],
        "historical_immediate_neighborhood": {
            "summary_path": str(PREVIOUS_LOCAL_OUTPUT / "summary.json"),
            "outcome": immediate.get("outcome"),
            "interpretation": "historical ±10% result remains frozen and is not merged with the fast-FEA population",
        },
        "fast_fea_guardrail": guardrail,
        "candidate49_fast_anchor": anchor,
        "populations": populations,
        "candidate49_excluded_population": candidate_excluded,
        "parameter_direction_diagnostics": parameter_diagnostics,
        "figures": figures,
        "claim": "Within the tested candidate49-centred region and under the validated fast-FEA plus intrinsic 3D OptiX configuration, the frozen 2D evaluator is assessed only as a regional screening proxy; no global, camera, sensor, or full-3D-mechanics claim is made.",
        "limitations": [
            "new expanded samples use the validated coarse_b/12-step fast-FEA tier, while the historical ±10% study used a higher-fidelity tier",
            "candidate49 anchor uses the fast-FEA guardrail for homogeneous regional statistics and historical high-fidelity values remain separate",
            "J3D-path is an intrinsic transport metric on an extruded mechanically deformed 2D cross-section",
            "the 0.06 criterion is primary; lower thresholds are descriptive only",
        ],
    }
    atomic_write_json(output / "expanded_trend_report.json", report)
    _update_manifest(output, stage="analysis_complete", scientific_outcome=outcome, artifacts_generated=figures + [str(output / "expanded_trend_report.json")])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("precommit", "guardrail", "2d", "fea", "3d", "analyze"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.stage == "precommit":
        result = write_precommit(args.output)
    elif args.stage == "guardrail":
        result = run_guardrail(args.output)
    elif args.stage == "2d":
        result = run_2d(args.output)
    elif args.stage == "fea":
        result = run_fea(args.output)
    elif args.stage == "3d":
        result = run_3d(args.output)
    else:
        result = analyze(args.output)
    print(json.dumps({"stage": args.stage, "status": result.get("status"), "outcome": result.get("outcome")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
