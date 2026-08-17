"""Restartable overnight 2D/3D morphology-trend validation.

The experiment is deliberately separate from optimization.  It freezes a
24-point paired design, runs only the validated search/12-step mechanics
tiers, promotes native 3D deformed surfaces to the unified OptiX contract,
and writes one independently checkable artifact for every child case.
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
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
from scipy.stats import kendalltau, spearmanr

from fem.indentation import IndentationSettings, run_indentation_case
from mesh.fingertip import generate_fingertip_mesh
from mesh.indenter import IndenterSettings, build_normal_indenter_fixture_at_x
from mesh.volume3d import generate_volume_mesh
from mesh.volume_types import volume_mesh_settings_for_tier
from model import Fingertip, FingertipParameters, build_fingertip_solid
from model.fingertip_model import FingertipModel
from optics import trace
from optics.transport3d import (
    OptiXTransport,
    Transport3DSettings,
    UnifiedTransportResult,
    fingerprint_mapping,
    load_case_artifact,
    load_full3d_surface_artifact,
    native_field_separability,
    save_case_artifact,
    trace_3d,
    transport_configuration,
)
from optics.transport3d.optix_backend import create_runtime
from validation.common.io import atomic_write_json, strict_read_json
from validation.fem.throughput import _mesh_policies
from validation.optimization.nominal_sweep import (
    FIXED_FLAT_PAD_WIDTH_MM,
    SWEPT_RANGES,
)
from validation.three_d_migration import (
    _export_native_3d_state,
    _m5_deformation_surface_checks,
)


OUTPUT = Path("output/validation/overnight_24_pair_trend")
PRECOMMIT = OUTPUT / "experiment_manifest.json"
STAGE_MANIFEST = OUTPUT / "stage_manifest.json"
SEED = 20260816
BASE_DESIGN_COUNT = 24
LHS_COUNT = 22
STEPS = 12
INDENTATION_MM = 0.5
INITIAL_GAP_MM = 0.2
RADIUS_MM = 4.0
CONTACT_LOCATIONS = {"left": -3.0, "right": 3.0}
RAY_COUNT = 1024
FIDELITY_RAY_COUNTS = (512, 1024, 2048)
GRID_WIDTH = 48
GRID_HEIGHT = 48
GRID_Z_BINS = 16
PAIR_TIE_TOLERANCE = 1.0e-12
SEARCH_2D_POLICY = "coarse_b"
SEARCH_3D_TIER = "search"
CASE_TIMEOUT_SECONDS = {"2d": 900.0, "3d": 3600.0}

CANDIDATE49 = {
    "flat_pad_height": 3.937175708822906,
    "semielliptical_pad_height": 7.309789158403873,
    "stem_width": 7.289858109783381,
    "stem_height": 5.102298432029784,
    "void_width": 0.6931721470318735,
    "void_height": 1.2690955214202404,
}
PARAMETER_NAMES = tuple(name for name, _, _ in SWEPT_RANGES)
BOUNDS = {name: (float(lower), float(upper)) for name, lower, upper in SWEPT_RANGES}


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


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _fingerprint(value: Any) -> str:
    return _sha256_bytes(_canonical(value))


def _implementation_fingerprint() -> str:
    paths = (
        Path(__file__),
        Path("fem/indentation.py"),
        Path("fem/solid3d.py"),
        Path("mesh/volume3d.py"),
        Path("optics/cross_section/transport.py"),
        Path("optics/transport3d/settings.py"),
        Path("optics/transport3d/transport.py"),
        Path("optics/transport3d/unified.py"),
        Path("validation/three_d_migration.py"),
    )
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(str(path).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _atomic_case(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(payload))


def _parameters_payload(parameters: FingertipParameters) -> dict[str, Any]:
    return {str(key): value for key, value in asdict(parameters).items()}


def _morphology_fingerprint(parameters: FingertipParameters) -> str:
    return build_fingertip_solid(Fingertip(parameters).geometry).morphology_fingerprint


def _normalized(parameters: Mapping[str, float]) -> dict[str, float]:
    return {
        name: (float(parameters[name]) - lower) / (upper - lower)
        for name, (lower, upper) in BOUNDS.items()
    }


def _parameters_from_normalized(point: Mapping[str, float]) -> FingertipParameters:
    values = {"flat_pad_width": FIXED_FLAT_PAD_WIDTH_MM}
    for name in PARAMETER_NAMES:
        lower, upper = BOUNDS[name]
        values[name] = lower + float(point[name]) * (upper - lower)
    return FingertipParameters(**values)


def _geometry_valid(parameters: FingertipParameters) -> bool:
    try:
        Fingertip(parameters)
        build_fingertip_solid(Fingertip(parameters).geometry)
    except Exception:
        return False
    return True


def _maximin_lhs() -> list[dict[str, Any]]:
    """Choose one deterministic, geometry-valid maximin LHS candidate.

    The candidate pool and selection score are generated before any mechanics
    or optical result is read.  Anchors participate in the distance score but
    are never replaced by sampled points.
    """
    nominal = _parameters_payload(FingertipParameters())
    candidate = dict(nominal)
    candidate.update(CANDIDATE49)
    anchors = np.asarray(
        [[_normalized(nominal)[name] for name in PARAMETER_NAMES],
         [_normalized(candidate)[name] for name in PARAMETER_NAMES]],
        dtype=float,
    )
    best: tuple[float, int, np.ndarray] | None = None
    for candidate_index in range(256):
        rng = np.random.default_rng(SEED + candidate_index)
        points = np.empty((LHS_COUNT, len(PARAMETER_NAMES)), dtype=float)
        for dimension in range(len(PARAMETER_NAMES)):
            permutation = rng.permutation(LHS_COUNT)
            points[:, dimension] = (permutation + rng.random(LHS_COUNT)) / LHS_COUNT
        distances = np.linalg.norm(
            np.concatenate((anchors, points), axis=0)[:, None, :]
            - np.concatenate((anchors, points), axis=0)[None, :, :],
            axis=2,
        )
        distances[np.diag_indices_from(distances)] = np.inf
        score = float(np.min(distances))
        valid = True
        for row in points:
            if not _geometry_valid(_parameters_from_normalized(dict(zip(PARAMETER_NAMES, row, strict=True)))):
                valid = False
                break
        if not valid:
            continue
        if best is None or score > best[0]:
            best = (score, candidate_index, points)
    if best is None:
        raise RuntimeError("deterministic LHS pool did not produce 22 valid morphologies")
    score, candidate_index, points = best
    samples = []
    for index, row in enumerate(points, start=1):
        normalized = {
            name: float(value)
            for name, value in zip(PARAMETER_NAMES, row, strict=True)
        }
        parameters = _parameters_from_normalized(normalized)
        samples.append(
            {
                "sample_index": index,
                "normalized_coordinate": normalized,
                "parameters": _parameters_payload(parameters),
                "morphology_fingerprint": _morphology_fingerprint(parameters),
            }
        )
    return [{"maximin_score": score, "selected_pool_seed_offset": candidate_index}, *samples]


def _precommit_payload() -> dict[str, Any]:
    nominal = FingertipParameters()
    candidate = FingertipParameters(**{**_parameters_payload(nominal), **CANDIDATE49})
    if _morphology_fingerprint(candidate) != _morphology_fingerprint(
        FingertipParameters(**{**_parameters_payload(nominal), **CANDIDATE49})
    ):
        raise RuntimeError("candidate49 fingerprint construction is not deterministic")
    lhs = _maximin_lhs()
    lhs_meta = lhs[0]
    lhs_samples = lhs[1:]
    bases = [
        {
            "base_id": "base_00_nominal",
            "anchor": "nominal",
            "normalized_coordinate": _normalized(_parameters_payload(nominal)),
            "parameters": _parameters_payload(nominal),
            "morphology_fingerprint": _morphology_fingerprint(nominal),
        },
        {
            "base_id": "base_01_candidate49",
            "anchor": "candidate49",
            "normalized_coordinate": _normalized(_parameters_payload(candidate)),
            "parameters": _parameters_payload(candidate),
            "morphology_fingerprint": _morphology_fingerprint(candidate),
        },
    ]
    bases.extend(
        {
            "base_id": f"base_{index + 1:02d}_lhs_{sample['sample_index']:02d}",
            "anchor": None,
            **sample,
        }
        for index, sample in enumerate(lhs_samples, start=1)
    )
    if len(bases) != BASE_DESIGN_COUNT:
        raise RuntimeError("precommit did not produce exactly 24 base designs")
    nominal_parameters = _parameters_payload(nominal)
    pairs = []
    for base in bases:
        fixed = dict(base["parameters"])
        fixed["void_height"] = nominal_parameters["void_height"]
        varied = dict(base["parameters"])
        fixed_parameters = FingertipParameters(**fixed)
        varied_parameters = FingertipParameters(**varied)
        pairs.append(
            {
                "base_id": base["base_id"],
                "anchor": base.get("anchor"),
                "non_void_parameters": {
                    key: value for key, value in base["parameters"].items() if key != "void_height"
                },
                "arms": {
                    "FIXED": {
                        "parameters": _parameters_payload(fixed_parameters),
                        "normalized_coordinate": _normalized(_parameters_payload(fixed_parameters)),
                        "morphology_fingerprint": _morphology_fingerprint(fixed_parameters),
                    },
                    "VARIED": {
                        "parameters": _parameters_payload(varied_parameters),
                        "normalized_coordinate": _normalized(_parameters_payload(varied_parameters)),
                        "morphology_fingerprint": _morphology_fingerprint(varied_parameters),
                    },
                },
            }
        )
    source_paths = [
        Path(__file__),
        Path("model/fingertip_parameters.py"),
        Path("validation/optimization/nominal_sweep.py"),
        Path("validation/fem/throughput.py"),
        Path("fem/indentation.py"),
        Path("fem/solid3d.py"),
        Path("mesh/volume3d.py"),
        Path("optics/transport3d/settings.py"),
        Path("optics/transport3d/transport.py"),
        Path("optics/transport3d/unified.py"),
    ]
    provenance = {
        str(path): _sha256_file(path)
        for path in source_paths
        if path.is_file()
    }
    payload = {
        "schema": "overnight-24-pair-trend-v1",
        "created_at": _now(),
        "scope": "2D circle FEA -> PLANAR_2D versus 3D sphere FEA -> FULL_3D",
        "m6_m7_m8": "NOT_IN_SCOPE",
        "design_space": {
            "fixed_parameters": {"flat_pad_width": FIXED_FLAT_PAD_WIDTH_MM},
            "variable_order": list(PARAMETER_NAMES),
            "bounds_mm": {name: list(bounds) for name, bounds in BOUNDS.items()},
        },
        "sampling": {
            "base_design_count": BASE_DESIGN_COUNT,
            "anchors": ["nominal", "candidate49"],
            "method": "deterministic maximin Latin hypercube with geometry-valid rejection",
            "seed": SEED,
            "candidate_pool_count": 256,
            "selected_pool_seed_offset": lhs_meta["selected_pool_seed_offset"],
            "selected_maximin_score": lhs_meta["maximin_score"],
            "score_inspection_during_selection": False,
            "candidate49_source": "validation/three_d_migration.py CANDIDATE49; exact solid fingerprint required",
        },
        "pairing": {
            "arms": ["FIXED", "VARIED"],
            "FIXED": "void_height set to FingertipParameters nominal default for every base",
            "VARIED": "same non-void coordinates; base-sampled void_height retained",
        },
        "pairs": pairs,
        "mechanics": {
            "two_d_mesh_policy": SEARCH_2D_POLICY,
            "three_d_mesh_tier": SEARCH_3D_TIER,
            "steps": STEPS,
            "indentation_mm": INDENTATION_MM,
            "initial_gap_mm": INITIAL_GAP_MM,
            "indenter_radius_mm": RADIUS_MM,
            "contact_locations_mm": CONTACT_LOCATIONS,
            "young_modulus_mpa": float(nominal.young_modulus_mpa),
            "poisson_ratio": float(nominal.poisson_ratio),
            "internal_contact": "three_pairs",
            "external_contact": True,
        },
        "optics": {
            "ray_count": RAY_COUNT,
            "fidelity_ray_counts": list(FIDELITY_RAY_COUNTS),
            "grid": {"x_bins": GRID_WIDTH, "y_bins": GRID_HEIGHT, "z_bins": GRID_Z_BINS},
            "extrusion_depth_mm": 11.0,
            "source_z_mm": 0.0,
            "metric": "native unified P2/P3 normalized spatial separability",
            "pair_grid": "union of left/right geometry bounds with fixed 4 percent margin; no result tuning",
        },
        "analysis": {
            "pair_tie_tolerance": PAIR_TIE_TOLERANCE,
            "classification_thresholds": {
                "strong_rho": 0.70,
                "strong_kendall": 0.50,
                "strong_pairwise": 0.70,
                "fixed_improvement_for_void_claim": 0.20,
                "fixed_improvement_for_partial": 0.10,
                "maximum_excluded_fraction": 0.25,
            },
        },
        "provenance": {"source_sha256": provenance},
    }
    payload["precommit_fingerprint"] = _fingerprint(
        {key: value for key, value in payload.items() if key != "created_at"}
    )
    return payload


def _verify_precommit(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "overnight-24-pair-trend-v1":
        raise RuntimeError("overnight precommit schema is stale")
    if payload.get("precommit_fingerprint") != _fingerprint(
        {key: value for key, value in payload.items() if key not in {"created_at", "precommit_fingerprint"}}
    ):
        raise RuntimeError("overnight precommit fingerprint mismatch")
    if len(payload.get("pairs", [])) != BASE_DESIGN_COUNT:
        raise RuntimeError("overnight precommit must contain exactly 24 pairs")
    fingerprints = []
    for pair in payload["pairs"]:
        arms = pair.get("arms", {})
        if set(arms) != {"FIXED", "VARIED"}:
            raise RuntimeError(f"{pair.get('base_id')} does not contain both arms")
        non_void = pair.get("non_void_parameters", {})
        if not non_void:
            raise RuntimeError(f"{pair.get('base_id')} has no paired non-void parameters")
        for arm_name in ("FIXED", "VARIED"):
            parameters = arms[arm_name]["parameters"]
            actual = _morphology_fingerprint(FingertipParameters(**parameters))
            if actual != arms[arm_name]["morphology_fingerprint"]:
                raise RuntimeError(f"{pair.get('base_id')} {arm_name} fingerprint mismatch")
            fingerprints.append(actual)
    candidate_pair = next(pair for pair in payload["pairs"] if pair.get("anchor") == "candidate49")
    candidate_varied = candidate_pair["arms"]["VARIED"]
    if candidate_varied["parameters"] != _parameters_payload(
        FingertipParameters(**{**_parameters_payload(FingertipParameters()), **CANDIDATE49})
    ):
        raise RuntimeError("candidate49 VARIED anchor is not the authoritative vector")
    # The nominal anchor is intentionally identical in both arms.  Duplicate
    # fingerprints across the two arms are therefore expected; duplicates
    # within one arm would indicate a malformed sample manifest.
    for arm in ("FIXED", "VARIED"):
        arm_fingerprints = [pair["arms"][arm]["morphology_fingerprint"] for pair in payload["pairs"]]
        if len(arm_fingerprints) != len(set(arm_fingerprints)):
            raise RuntimeError(f"duplicate morphology fingerprint within {arm} arm")


def _load_precommit() -> dict[str, Any]:
    if not PRECOMMIT.exists():
        raise RuntimeError(f"missing precommit manifest: {PRECOMMIT}; run --stage precommit")
    payload = strict_read_json(PRECOMMIT)
    _verify_precommit(payload)
    return payload


def _stage_update(**updates: Any) -> None:
    current = strict_read_json(STAGE_MANIFEST) if STAGE_MANIFEST.exists() else {}
    current.update(updates)
    current["updated_at"] = _now()
    atomic_write_json(STAGE_MANIFEST, current)


def _pair_cases(precommit: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases = []
    for pair in precommit["pairs"]:
        for arm in ("FIXED", "VARIED"):
            arm_data = pair["arms"][arm]
            for side, x_mm in CONTACT_LOCATIONS.items():
                cases.append(
                    {
                        "case_id": f"{pair['base_id']}__{arm}__{side}",
                        "base_id": pair["base_id"],
                        "arm": arm,
                        "side": side,
                        "contact_x_mm": x_mm,
                        "parameters": arm_data["parameters"],
                        "morphology_fingerprint": arm_data["morphology_fingerprint"],
                    }
                )
    return cases


def _case_contract(case: Mapping[str, Any], stage: str, *, include_implementation: bool = True) -> str:
    contract = {
            "schema": "overnight-child-v1",
            "stage": stage,
            "case_id": case["case_id"],
            "base_id": case["base_id"],
            "arm": case["arm"],
            "side": case["side"],
            "contact_x_mm": case["contact_x_mm"],
            "parameters": case["parameters"],
            "morphology_fingerprint": case["morphology_fingerprint"],
            "mechanics": {
                "two_d_mesh_policy": SEARCH_2D_POLICY,
                "three_d_mesh_tier": SEARCH_3D_TIER,
                "steps": STEPS,
                "indentation_mm": INDENTATION_MM,
                "initial_gap_mm": INITIAL_GAP_MM,
                "radius_mm": RADIUS_MM,
            },
        }
    if include_implementation:
        contract["implementation_fingerprint"] = _implementation_fingerprint()
    return _fingerprint(contract)


def _case_path(stage: str, case: Mapping[str, Any]) -> Path:
    return OUTPUT / stage / f"{case['case_id']}.json"


def _case_log_path(stage: str, case: Mapping[str, Any], stream: str) -> Path:
    return OUTPUT / "logs" / stage / f"{case['case_id']}.{stream}.log"


def _write_failure(stage: str, case: Mapping[str, Any], outcome: str, reason: str, **extra: Any) -> None:
    payload = {
        "schema": "overnight-child-v1",
        "stage": stage,
        "case_id": case["case_id"],
        "base_id": case["base_id"],
        "arm": case["arm"],
        "side": case["side"],
        "parameters": case["parameters"],
        "morphology_fingerprint": case["morphology_fingerprint"],
        "contact_x_mm": case["contact_x_mm"],
        "case_fingerprint": _case_contract(case, stage),
        "outcome": outcome,
        "status": outcome,
        "failure_reason": reason,
        "created_at": _now(),
        **_jsonable(extra),
    }
    _atomic_case(_case_path(stage, case), payload)


def _run_2d_child(case: Mapping[str, Any]) -> int:
    stage = "fea2d"
    started = time.perf_counter()
    path = _case_path(stage, case)
    try:
        parameters = FingertipParameters(**case["parameters"])
        model = FingertipModel(parameters)
        tip = Fingertip(parameters)
        policy = next(item for item in _mesh_policies() if item.name == SEARCH_2D_POLICY)
        mesh = generate_fingertip_mesh(model, policy.settings)
        captured: dict[str, Any] = {}

        def observe(step: Any) -> None:
            if int(step.result_point["step"]) == STEPS:
                captured["displacements"] = {
                    int(node_id): tuple(float(value) for value in displacement)
                    for node_id, displacement in step.displacements.items()
                }

        fixture = build_normal_indenter_fixture_at_x(
            model,
            float(case["contact_x_mm"]),
            IndenterSettings(radius_mm=RADIUS_MM, initial_gap_mm=INITIAL_GAP_MM),
        )
        result, artifacts = run_indentation_case(
            model,
            "medium",
            IndentationSettings(INDENTATION_MM, STEPS),
            fixture_override=fixture,
            internal_contact_configuration="three_pairs",
            mesh_override=mesh,
            converged_step_observer=observe,
            diagnostic_mode="minimal",
        )
        if artifacts is None or result.get("solve_status") != "PASS" or "displacements" not in captured:
            _write_failure(stage, case, "NUMERICAL_FAIL", str(result.get("failure_reason") or result.get("exception") or "2D solve did not converge"), result=result)
            return 1
        displacement = np.asarray(
            [captured["displacements"][int(node_id)] for node_id in mesh.pad.node_ids],
            dtype=float,
        )
        if displacement.shape != mesh.pad.coordinates.shape or not np.all(np.isfinite(displacement)):
            _write_failure(stage, case, "NUMERICAL_FAIL", "2D displacement artifact is invalid", result=result)
            return 1
        state_path = path.with_suffix(".npz")
        _atomic_npz(state_path, displacement=displacement)
        final = result.get("final", {})
        payload = {
            "schema": "overnight-child-v1",
            "stage": stage,
            "case_id": case["case_id"],
            "base_id": case["base_id"],
            "arm": case["arm"],
            "side": case["side"],
            "case_fingerprint": _case_contract(case, stage),
            "outcome": "PASS",
            "status": "PASS",
            "morphology_fingerprint": case["morphology_fingerprint"],
            "parameters": case["parameters"],
            "mesh_policy": SEARCH_2D_POLICY,
            "mesh_settings": asdict(mesh.settings),
            "steps": STEPS,
            "indentation_mm": INDENTATION_MM,
            "initial_gap_mm": INITIAL_GAP_MM,
            "contact_x_mm": case["contact_x_mm"],
            "reaction_force_n": final.get("indenter_normal_reaction_n"),
            "history": result.get("history", []),
            "contact_state": final.get("contact_groups", {}),
            "max_displacement_mm": float(np.linalg.norm(displacement, axis=1).max()),
            "rms_displacement_mm": float(np.sqrt(np.mean(np.sum(displacement * displacement, axis=1)))),
            "state_artifact": str(state_path),
            "state_sha256": _sha256_file(state_path),
            "runtime_seconds": time.perf_counter() - started,
            "configuration": result.get("configuration", {}),
        }
        _atomic_case(path, payload)
        return 0
    except Exception as exc:
        _write_failure(stage, case, "NUMERICAL_FAIL", f"{type(exc).__name__}: {exc}", runtime_seconds=time.perf_counter() - started)
        return 1


def _run_3d_child(case: Mapping[str, Any]) -> int:
    stage = "fea3d"
    started = time.perf_counter()
    path = _case_path(stage, case)
    try:
        parameters = FingertipParameters(**case["parameters"])
        tip = Fingertip(parameters)
        solid = build_fingertip_solid(tip.geometry)
        mesh = generate_volume_mesh(solid, volume_mesh_settings_for_tier(SEARCH_3D_TIER))
        fixture = build_normal_indenter_fixture_at_x(
            tip.geometry,
            float(case["contact_x_mm"]),
            IndenterSettings(radius_mm=RADIUS_MM, initial_gap_mm=INITIAL_GAP_MM),
        )
        from fem.solid3d import SolidFEASettings, solve_solid_3d

        result = solve_solid_3d(
            mesh,
            fixture,
            SolidFEASettings(
                mode="production",
                number_of_steps=STEPS,
                indentation_mm=INDENTATION_MM,
                external_contact=True,
            ),
        )
        surface = _m5_deformation_surface_checks(mesh, result.deformed_coordinates_mm)
        reaction = None if result.reaction_force_n is None else float(result.reaction_force_n)
        checks = {
            "converged": bool(result.converged),
            "finite_reaction": reaction is not None and math.isfinite(reaction),
            "surface_valid": bool(surface["passed"]),
            "finite_displacement": result.displacement_mm is not None and bool(np.all(np.isfinite(result.displacement_mm))),
        }
        if not all(checks.values()):
            _write_failure(stage, case, "NUMERICAL_FAIL", str(result.failure_message or checks), result={"configuration": result.configuration, "contact_state": result.contact_state, "checks": checks, "surface_validity": surface})
            return 1
        payload = {
            "status": "PASS",
            "stage": stage,
            "case_id": case["case_id"],
            "base_id": case["base_id"],
            "arm": case["arm"],
            "side": case["side"],
            "morphology": case["case_id"],
            "contact_location": case["side"],
            "contact_location_mm": case["contact_x_mm"],
            "mesh_tier": SEARCH_3D_TIER,
            "steps": STEPS,
            "morphology_fingerprint": case["morphology_fingerprint"],
            "case_fingerprint": _case_contract(case, stage),
            "configuration": result.configuration,
            "contact_state": result.contact_state,
            "reaction_force_n": reaction,
            "surface_validity": surface,
            "checks": checks,
            "no_fallback_or_nan_suppression": True,
        }
        native_manifest = _export_native_3d_state(
            name=case["case_id"],
            tier=SEARCH_3D_TIER,
            steps=STEPS,
            location_name=case["side"],
            fingertip=tip,
            mesh=mesh,
            fixture=fixture,
            result=result,
            payload=payload,
            output_dir=OUTPUT / "native_3d_states",
            indentation_mm=INDENTATION_MM,
            contact_location_mm=float(case["contact_x_mm"]),
        )
        payload.update(
            {
                "outcome": "PASS",
                "native_manifest": str(native_manifest),
                "native_manifest_sha256": _sha256_file(native_manifest),
                "max_displacement_mm": float(np.linalg.norm(result.displacement_mm, axis=1).max()),
                "rms_displacement_mm": float(np.sqrt(np.mean(np.sum(result.displacement_mm * result.displacement_mm, axis=1)))),
                "runtime_seconds": time.perf_counter() - started,
            }
        )
        _atomic_case(path, _jsonable(payload))
        return 0
    except Exception as exc:
        _write_failure(stage, case, "NUMERICAL_FAIL", f"{type(exc).__name__}: {exc}", runtime_seconds=time.perf_counter() - started)
        return 1


def _child_dispatch(stage: str, case_id: str) -> int:
    precommit = _load_precommit()
    cases = {case["case_id"]: case for case in _pair_cases(precommit)}
    if case_id not in cases:
        raise RuntimeError(f"unknown case id {case_id}")
    return _run_2d_child(cases[case_id]) if stage == "fea2d" else _run_3d_child(cases[case_id])


def _read_case(stage: str, case: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _case_path(stage, case)
    if not path.exists():
        return None
    try:
        payload = strict_read_json(path)
        current_contract = _case_contract(case, stage)
        legacy_contract = _case_contract(case, stage, include_implementation=False)
        stable_pass_reuse = (
            payload.get("outcome") == "PASS"
            and payload.get("morphology_fingerprint") == case["morphology_fingerprint"]
            and payload.get("steps") == STEPS
            and (
                (stage == "fea2d" and payload.get("mesh_policy") == SEARCH_2D_POLICY)
                or (stage == "fea3d" and payload.get("mesh_tier") == SEARCH_3D_TIER)
            )
        )
        legacy_pass_reuse = stable_pass_reuse and (
            payload.get("case_fingerprint") == legacy_contract
            or payload.get("case_fingerprint") != current_contract
        )
        if payload.get("case_fingerprint") != current_contract and not legacy_pass_reuse:
            return None
        if (payload.get("case_id") != case["case_id"] and not legacy_pass_reuse) or payload.get("outcome") not in {
            "PASS", "NUMERICAL_FAIL", "RUNTIME_LIMIT", "ENVIRONMENT_FAIL"
        }:
            return None
        if payload.get("outcome") == "PASS":
            if stage == "fea2d":
                state = Path(str(payload.get("state_artifact", "")))
                if not state.exists() or payload.get("state_sha256") != _sha256_file(state):
                    return None
            if stage == "fea3d":
                manifest = Path(str(payload.get("native_manifest", "")))
                if not manifest.exists() or payload.get("native_manifest_sha256") != _sha256_file(manifest):
                    return None
        return payload
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _run_child(stage: str, case: Mapping[str, Any]) -> dict[str, Any]:
    existing = _read_case(stage, case)
    if existing is not None:
        existing["reused"] = True
        return existing
    log_out = _case_log_path(stage, case, "stdout")
    log_err = _case_log_path(stage, case, "stderr")
    log_out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        with log_out.open("w", encoding="utf-8") as stdout, log_err.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                [sys.executable, "-m", "validation.overnight_24_pair_trend", "--child-stage", stage, "--case-id", case["case_id"]],
                stdout=stdout,
                stderr=stderr,
                timeout=CASE_TIMEOUT_SECONDS["2d" if stage == "fea2d" else "3d"],
                check=False,
            )
        payload = _read_case(stage, case)
        if payload is not None:
            payload["parent_return_code"] = completed.returncode
            payload["parent_wall_time_seconds"] = time.perf_counter() - started
            return payload
        outcome = "ENVIRONMENT_FAIL" if completed.returncode < 0 else "NUMERICAL_FAIL"
        _write_failure(stage, case, outcome, "child exited without a valid atomic result", return_code=completed.returncode, logs={"stdout": str(log_out), "stderr": str(log_err)})
    except subprocess.TimeoutExpired:
        _write_failure(stage, case, "RUNTIME_LIMIT", "configured engineering runtime budget expired; no scientific conclusion", runtime_limit_seconds=CASE_TIMEOUT_SECONDS["2d" if stage == "fea2d" else "3d"], logs={"stdout": str(log_out), "stderr": str(log_err)})
    payload = _read_case(stage, case)
    if payload is None:
        raise RuntimeError(f"failed to persist parent outcome for {case['case_id']}")
    payload["parent_wall_time_seconds"] = time.perf_counter() - started
    return payload


def _run_mechanics_stage(stage: str) -> dict[str, Any]:
    precommit = _load_precommit()
    cases = _pair_cases(precommit)
    records = []
    for index, case in enumerate(cases, start=1):
        record = _run_child(stage, case)
        record = {
            **record,
            "base_id": case["base_id"],
            "arm": case["arm"],
            "side": case["side"],
            "parameters": case["parameters"],
            "morphology_fingerprint": case["morphology_fingerprint"],
        }
        records.append(record)
        _stage_update(
            precommit_fingerprint=precommit["precommit_fingerprint"],
            stage=stage,
            completed_case_count=index,
            case_outcomes={
                row["case_id"]: row.get("outcome")
                for row in records
            },
        )
    summary = {
        "schema": f"overnight-{stage}-v1",
        "precommit_fingerprint": precommit["precommit_fingerprint"],
        "planned_cases": len(cases),
        "records": records,
        "counts": {
            outcome: sum(row.get("outcome") == outcome for row in records)
            for outcome in ("PASS", "NUMERICAL_FAIL", "RUNTIME_LIMIT", "ENVIRONMENT_FAIL")
        },
        "reused_case_count": sum(bool(row.get("reused")) for row in records),
        "created_at": _now(),
    }
    atomic_write_json(OUTPUT / f"{stage}_summary.json", _jsonable(summary))
    return summary


def _material_configuration(tip: Fingertip, settings: Transport3DSettings) -> dict[str, Any]:
    return transport_configuration(
        settings,
        material={
            "refractive_index_air": tip.optical.refractive_index_air,
            "refractive_index_silicone": tip.optical.refractive_index_silicone,
            "absorption_per_mm": tip.optical.absorption_per_mm,
            "scattering_per_mm": tip.optical.scattering_per_mm,
        },
    )


def _pair_bounds(case_left: Mapping[str, Any], case_right: Mapping[str, Any], two_d: Mapping[str, Mapping[str, Any]], three_d: Mapping[str, Mapping[str, Any]]) -> tuple[tuple[float, float], tuple[float, float]]:
    values_x: list[float] = []
    values_y: list[float] = []
    for case in (case_left, case_right):
        parameters = FingertipParameters(**case["parameters"])
        tip = Fingertip(parameters)
        state = two_d[case["side"]]
        mesh = generate_fingertip_mesh(FingertipModel(parameters), next(item for item in _mesh_policies() if item.name == SEARCH_2D_POLICY).settings)
        deformed = mesh.pad.deformed(np.asarray(state["displacement"], dtype=float))
        values_x.extend([float(value) for value in deformed.coordinates[:, 0]])
        values_y.extend([float(value) for value in deformed.coordinates[:, 1]])
        artifact = load_full3d_surface_artifact(
            Path(three_d[case["side"]]["native_manifest"]),
            expected_morphology_fingerprint=case["morphology_fingerprint"],
            expected_contact_state_fingerprint=three_d[case["side"]]["contact_state_fingerprint"],
        )
        for surface in (artifact.silicone, artifact.rigid, artifact.envelope):
            values_x.extend(np.asarray(surface.vertices[:, 0], dtype=float).tolist())
            values_y.extend(np.asarray(surface.vertices[:, 1], dtype=float).tolist())
        bounds_xy = tip.geometry.material_geometry.bounds
        values_x.extend([float(bounds_xy[0]), float(bounds_xy[2])])
        values_y.extend([float(bounds_xy[1]), float(bounds_xy[3])])
    span = max(max(values_x) - min(values_x), max(values_y) - min(values_y))
    margin = 0.04 * span
    return (min(values_x) - margin, max(values_x) + margin), (min(values_y) - margin, max(values_y) + margin)


def _transport_settings(mode: str, bounds: tuple[tuple[float, float], tuple[float, float]], ray_count: int = RAY_COUNT) -> Transport3DSettings:
    return Transport3DSettings(
        mode=mode,  # type: ignore[arg-type]
        ray_count=ray_count,
        max_interactions=10,
        minimum_ray_weight=1.0e-4,
        maximum_segment_count=max(24000, 24 * ray_count),
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
        retain_projected_segments=mode == "planar",
        retain_internal_path_field=mode == "full3d",
    )


def _optical_contract(case: Mapping[str, Any], mode: str, settings: Transport3DSettings, fea_path: Path, contact_fp: str, configuration: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "overnight-optix-case-contract-v1",
        "morphology_id": case["case_id"],
        "morphology_parameters_fingerprint": case["morphology_fingerprint"],
        "mechanics_dimension": "2D" if mode == "PLANAR_2D" else "3D",
        "mechanics_source": str(fea_path),
        "contact_location": case["side"],
        "contact_state_fingerprint": contact_fp,
        "optical_mode": mode,
        "ray_count": settings.ray_count,
        "optical_configuration": _jsonable(configuration),
        "transport_configuration_fingerprint": fingerprint_mapping(dict(configuration)),
        "fea_artifact_sha256": _sha256_file(fea_path),
        "contact_x_mm": case["contact_x_mm"],
        "initial_gap_mm": INITIAL_GAP_MM,
        "total_prescribed_travel_mm": INDENTATION_MM,
        "indenter_radius_mm": RADIUS_MM,
    }


def _field_descriptors(result: UnifiedTransportResult) -> dict[str, Any]:
    field = np.asarray(result.field, dtype=float)
    mass = float(field.sum())
    axes = result.field_axes
    centers = [0.5 * (axis[:-1] + axis[1:]) for axis in axes]
    if mass <= 0.0:
        return {"total_transport": result.total_transport, "field_mass": mass, "status": "ZERO_FIELD"}
    weights = field / mass
    descriptors: dict[str, Any] = {
        "status": "PASS",
        "total_transport": result.total_transport,
        "field_mass": mass,
        "normalized_spatial_redistribution_l1_reference": "field normalized to unit mass",
        "valid_ray_fraction": result.valid_ray_count / result.ray_count,
        "terminated_ray_fraction": result.terminated_ray_count / result.ray_count,
        "energy_balance_error": result.energy_balance_error,
    }
    for index, name in enumerate(("x", "y", "z")[:field.ndim]):
        marginal_axes = tuple(range(field.ndim))
        marginal = np.sum(weights, axis=tuple(axis for axis in marginal_axes if axis != index))
        mean = float(np.sum(marginal * centers[index]) / max(float(marginal.sum()), 1.0e-30))
        variance = float(np.sum(marginal * (centers[index] - mean) ** 2) / max(float(marginal.sum()), 1.0e-30))
        descriptors[f"{name}_centroid_mm"] = mean
        descriptors[f"{name}_spread_mm"] = math.sqrt(max(variance, 0.0))
    if field.ndim == 3:
        z_centers = centers[2]
        z_marginal = np.sum(weights, axis=(0, 1))
        central = np.abs(z_centers) <= 0.25 * 11.0
        descriptors["z_fraction_away_from_central_region"] = float(np.sum(z_marginal[~central]))
        descriptors["z_distribution"] = {
            "edges_mm": [float(value) for value in axes[2]],
            "normalized_mass": [float(value) for value in z_marginal],
        }
    return descriptors


def _raw_descriptors(raw: Any, result: UnifiedTransportResult) -> dict[str, Any]:
    path_lengths = np.asarray(raw.escape_path_lengths_mm, dtype=float)
    interactions = np.asarray(raw.escape_interaction_counts, dtype=int)
    metadata = dict(result.path_diagnostics)
    path_field = metadata.get("internal_path_field", {})
    return {
        **_field_descriptors(result),
        "launched_ray_count": int(raw.launched_ray_count),
        "escaped_ray_count": int(len(raw.escape_weights)),
        "absorbed_weight": float(raw.absorbed_weight),
        "terminated_weight": float(raw.terminated_weight),
        "processed_segment_count": metadata.get("processed_segment_count"),
        "retained_segment_count": metadata.get("retained_segment_count"),
        "path_length_statistics_mm": {
            "count": int(len(path_lengths)),
            "mean": float(path_lengths.mean()) if len(path_lengths) else None,
            "p95": float(np.percentile(path_lengths, 95.0)) if len(path_lengths) else None,
            "max": float(path_lengths.max()) if len(path_lengths) else None,
        },
        "escape_interaction_statistics": {
            "mean": float(interactions.mean()) if len(interactions) else None,
            "max": int(interactions.max()) if len(interactions) else None,
            "histogram": {str(int(value)): int(np.count_nonzero(interactions == value)) for value in np.unique(interactions)},
        },
        "termination_statistics": {
            "periodic_wrap": metadata.get("periodic_wrap_termination"),
            "no_event": metadata.get("no_event_termination"),
            "weight": float(raw.terminated_weight),
        },
        "branching_statistics": {
            "status": "escape-interaction proxy; branch creation counters are not exposed by current neutral result",
            "mean_escape_interactions": float(interactions.mean()) if len(interactions) else None,
        },
        "reflection_refraction_tir_statistics": {
            "status": "UNAVAILABLE",
            "reason": "current neutral transport result exposes escape interactions but not event-type counters",
        },
        "internal_path_metadata": path_field,
    }


def _run_optical_case(
    case: Mapping[str, Any],
    mode: str,
    settings: Transport3DSettings,
    geometry: Any,
    tip: Fingertip,
    fea_path: Path,
    contact_fp: str,
    runtime: Any,
) -> tuple[UnifiedTransportResult, dict[str, Any], bool]:
    configuration = _material_configuration(tip, settings)
    contract = _optical_contract(case, mode, settings, fea_path, contact_fp, configuration)
    output_dir = OUTPUT / "optix_cases"
    path = output_dir / f"{case['case_id']}__{mode}__{settings.ray_count}.json"
    try:
        loaded = load_case_artifact(path, expected_contract=contract)
        raw_summary_path = path.with_name(path.stem + "__raw.json")
        if raw_summary_path.exists():
            return loaded, strict_read_json(raw_summary_path), True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        pass
    raw = trace_3d(
        tip,
        geometry["mesh"],
        reference_mesh=geometry.get("reference_mesh"),
        settings=settings,
        runtime=runtime,
    ) if mode == "PLANAR_2D" else __import__("optics.transport3d.transport", fromlist=["trace_geometry"]).trace_geometry(
        tip, geometry["geometry"], settings=settings, runtime=runtime
    )
    result = UnifiedTransportResult.from_transport_result(
        raw,
        morphology_id=case["case_id"],
        morphology_fingerprint=case["morphology_fingerprint"],
        mechanics_source=str(fea_path),
        mechanics_dimension="2D" if mode == "PLANAR_2D" else "3D",
        contact_state={"contact_state_fingerprint": contact_fp, "contact_location": case["side"]},
        transport_configuration_fingerprint=fingerprint_mapping(configuration),
    )
    save_case_artifact(path, result, contract)
    raw_summary = _raw_descriptors(raw, result)
    raw_summary["contract"] = contract
    atomic_write_json(path.with_name(path.stem + "__raw.json"), _jsonable(raw_summary))
    return result, raw_summary, False


def _load_2d_state(case: Mapping[str, Any], record: Mapping[str, Any]) -> tuple[Fingertip, Any, dict[str, Any]]:
    parameters = FingertipParameters(**case["parameters"])
    tip = Fingertip(parameters)
    policy = next(item for item in _mesh_policies() if item.name == SEARCH_2D_POLICY)
    mesh = generate_fingertip_mesh(FingertipModel(parameters), policy.settings)
    state_path = Path(str(record["state_artifact"]))
    with np.load(state_path, allow_pickle=False) as archive:
        displacement = np.asarray(archive["displacement"], dtype=float)
    if displacement.shape != mesh.pad.coordinates.shape:
        raise RuntimeError(f"2D state topology mismatch for {case['case_id']}")
    return tip, mesh, {"displacement": displacement, "mesh": mesh.pad.deformed(displacement), "reference_mesh": mesh}


def _load_3d_state(case: Mapping[str, Any], record: Mapping[str, Any]) -> tuple[Fingertip, Any]:
    parameters = FingertipParameters(**case["parameters"])
    tip = Fingertip(parameters)
    manifest = Path(str(record["native_manifest"]))
    artifact = load_full3d_surface_artifact(
        manifest,
        expected_morphology_fingerprint=case["morphology_fingerprint"],
        expected_contact_state_fingerprint=str(strict_read_json(manifest)["contact_state_fingerprint"]),
    )
    return tip, artifact


def _run_optix_stage() -> dict[str, Any]:
    precommit = _load_precommit()
    two_d = {row["case_id"]: row for row in strict_read_json(OUTPUT / "fea2d_summary.json")["records"]}
    three_d = {row["case_id"]: row for row in strict_read_json(OUTPUT / "fea3d_summary.json")["records"]}
    runtime = create_runtime()
    records: list[dict[str, Any]] = []
    pair_results: dict[str, Any] = {}
    try:
        for pair in precommit["pairs"]:
            for arm in ("FIXED", "VARIED"):
                left_case = next(case for case in _pair_cases(precommit) if case["base_id"] == pair["base_id"] and case["arm"] == arm and case["side"] == "left")
                right_case = next(case for case in _pair_cases(precommit) if case["base_id"] == pair["base_id"] and case["arm"] == arm and case["side"] == "right")
                left2 = two_d[left_case["case_id"]]
                right2 = two_d[right_case["case_id"]]
                left3 = three_d[left_case["case_id"]]
                right3 = three_d[right_case["case_id"]]
                pair_key = f"{pair['base_id']}__{arm}"
                if left2.get("outcome") != "PASS" or right2.get("outcome") != "PASS" or left3.get("outcome") != "PASS" or right3.get("outcome") != "PASS":
                    pair_results[pair_key] = {"status": "EXCLUDED", "reason": "incomplete mechanics pair"}
                    continue
                tip_left, mesh_left, state_left = _load_2d_state(left_case, left2)
                tip_right, mesh_right, state_right = _load_2d_state(right_case, right2)
                tip3_left, artifact_left = _load_3d_state(left_case, left3)
                tip3_right, artifact_right = _load_3d_state(right_case, right3)
                bounds = _pair_bounds(left_case, right_case, {"left": state_left, "right": state_right}, {"left": left3, "right": right3})
                planar_settings = _transport_settings("planar", bounds)
                full_settings = _transport_settings("full3d", bounds)
                planar_left, raw_planar_left, reused_pl_left = _run_optical_case(left_case, "PLANAR_2D", planar_settings, {"mesh": state_left["mesh"], "reference_mesh": state_left["reference_mesh"]}, tip_left, Path(left2["state_artifact"]), "2d-" + _fingerprint(left2), runtime)
                planar_right, raw_planar_right, reused_pl_right = _run_optical_case(right_case, "PLANAR_2D", planar_settings, {"mesh": state_right["mesh"], "reference_mesh": state_right["reference_mesh"]}, tip_right, Path(right2["state_artifact"]), "2d-" + _fingerprint(right2), runtime)
                full_left, raw_full_left, reused_full_left = _run_optical_case(left_case, "FULL_3D", full_settings, {"geometry": artifact_left.geometry(tip3_left)}, tip3_left, Path(left3["native_manifest"]), str(strict_read_json(Path(left3["native_manifest"]))["contact_state_fingerprint"]), runtime)
                full_right, raw_full_right, reused_full_right = _run_optical_case(right_case, "FULL_3D", full_settings, {"geometry": artifact_right.geometry(tip3_right)}, tip3_right, Path(right3["native_manifest"]), str(strict_read_json(Path(right3["native_manifest"]))["contact_state_fingerprint"]), runtime)
                j2 = native_field_separability(planar_left, planar_right)
                j3 = native_field_separability(full_left, full_right)
                pair_results[pair_key] = {
                    "status": "PASS",
                    "base_id": pair["base_id"],
                    "arm": arm,
                    "parameters": pair["arms"][arm]["parameters"],
                    "normalized_coordinate": pair["arms"][arm]["normalized_coordinate"],
                    "morphology_fingerprint": pair["arms"][arm]["morphology_fingerprint"],
                    "pair_grid": {"x_bounds_mm": list(bounds[0]), "y_bounds_mm": list(bounds[1]), "settings": asdict(planar_settings)},
                    "J2": j2,
                    "J3": j3,
                    "raw": {"planar_left": raw_planar_left, "planar_right": raw_planar_right, "full_left": raw_full_left, "full_right": raw_full_right},
                    "fea": {"2d_left": left2, "2d_right": right2, "3d_left": left3, "3d_right": right3},
                    "reused_optix_case_count": int(reused_pl_left) + int(reused_pl_right) + int(reused_full_left) + int(reused_full_right),
                }
                records.append(pair_results[pair_key])
                atomic_write_json(OUTPUT / "optix_progress.json", _jsonable({"records": list(pair_results.values()), "precommit_fingerprint": precommit["precommit_fingerprint"]}))
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
    summary = {
        "schema": "overnight-optix-v1",
        "precommit_fingerprint": precommit["precommit_fingerprint"],
        "planned_pairs": BASE_DESIGN_COUNT * 2,
        "records": list(pair_results.values()),
        "counts": {
            "PASS": sum(row.get("status") == "PASS" for row in pair_results.values()),
            "EXCLUDED": sum(row.get("status") == "EXCLUDED" for row in pair_results.values()),
        },
        "created_at": _now(),
    }
    atomic_write_json(OUTPUT / "optix_summary.json", _jsonable(summary))
    return summary


def _rank_stats(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    j2 = np.asarray([float(row["J2"]["normalized_redistribution_l1"]) for row in rows], dtype=float)
    j3 = np.asarray([float(row["J3"]["normalized_redistribution_l1"]) for row in rows], dtype=float)
    if len(rows) < 3 or np.ptp(j2) <= PAIR_TIE_TOLERANCE or np.ptp(j3) <= PAIR_TIE_TOLERANCE:
        return {"status": "INCONCLUSIVE", "n": len(rows), "j2_range": float(np.ptp(j2)) if len(j2) else None, "j3_range": float(np.ptp(j3)) if len(j3) else None}
    concordant = discordant = tied = 0
    for first, second in itertools.combinations(range(len(rows)), 2):
        d2 = j2[first] - j2[second]
        d3 = j3[first] - j3[second]
        t2 = abs(d2) <= PAIR_TIE_TOLERANCE
        t3 = abs(d3) <= PAIR_TIE_TOLERANCE
        if t2 or t3:
            tied += 1
        elif d2 * d3 > 0.0:
            concordant += 1
        else:
            discordant += 1
    total_pairs = len(rows) * (len(rows) - 1) // 2
    comparable = concordant + discordant
    return {
        "status": "PASS",
        "n": len(rows),
        "spearman_rho": float(spearmanr(j2, j3).statistic),
        "kendall_tau": float(kendalltau(j2, j3).statistic),
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "tied_pairs": tied,
        "total_pairs": total_pairs,
        "pairwise_concordance_fraction": concordant / comparable if comparable else None,
        "j2_dynamic_range": float(np.ptp(j2)),
        "j3_dynamic_range": float(np.ptp(j3)),
    }


def _parameter_trends(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for name in PARAMETER_NAMES:
        parameter = np.asarray([float(row["parameters"][name]) for row in rows], dtype=float)
        j2 = np.asarray([float(row["J2"]["normalized_redistribution_l1"]) for row in rows], dtype=float)
        j3 = np.asarray([float(row["J3"]["normalized_redistribution_l1"]) for row in rows], dtype=float)
        r2 = float(spearmanr(parameter, j2).statistic) if np.ptp(parameter) > PAIR_TIE_TOLERANCE else None
        r3 = float(spearmanr(parameter, j3).statistic) if np.ptp(parameter) > PAIR_TIE_TOLERANCE else None
        result[name] = {"spearman_parameter_J2": r2, "spearman_parameter_J3": r3, "direction_agreement": None if r2 is None or r3 is None or abs(r2) <= PAIR_TIE_TOLERANCE or abs(r3) <= PAIR_TIE_TOLERANCE else bool(r2 * r3 > 0.0), "sign_disagreement": None if r2 is None or r3 is None else bool(r2 * r3 < 0.0)}
    return result


def _mechanics_trends(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    descriptors: dict[str, Any] = {}
    for dimension in ("2d", "3d"):
        for descriptor in ("reaction_force_n", "max_displacement_mm", "rms_displacement_mm"):
            values = []
            for row in rows:
                side_values = [row["fea"][f"{dimension}_left"].get(descriptor), row["fea"][f"{dimension}_right"].get(descriptor)]
                numeric = [float(value) for value in side_values if value is not None and math.isfinite(float(value))]
                values.append(float(np.mean(numeric)) if numeric else math.nan)
            descriptors[f"{dimension}_{descriptor}"] = values
    associations = {}
    disagreement = np.asarray([abs(float(row["J2"]["normalized_redistribution_l1"]) - float(row["J3"]["normalized_redistribution_l1"])) for row in rows], dtype=float)
    for name, values in descriptors.items():
        array = np.asarray(values, dtype=float)
        valid = np.isfinite(array)
        associations[name] = float(spearmanr(array[valid], disagreement[valid]).statistic) if np.count_nonzero(valid) >= 3 and np.ptp(array[valid]) > PAIR_TIE_TOLERANCE else None
    return {"values_by_pair": descriptors, "association_with_abs_J2_minus_J3": associations}


def _raw_transport_trends(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    names = ("total_transport", "x_centroid_mm", "y_centroid_mm", "x_spread_mm", "y_spread_mm", "z_centroid_mm", "z_spread_mm", "z_fraction_away_from_central_region", "valid_ray_fraction", "terminated_ray_fraction")
    result: dict[str, Any] = {}
    for name in names:
        per_pair = []
        for row in rows:
            full = row["raw"]["full_left"]
            right = row["raw"]["full_right"]
            values = [full.get(name), right.get(name)]
            numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
            per_pair.append(float(np.mean(numeric)) if numeric else None)
        result[name] = {"values_by_pair": per_pair}
    return result


def _fidelity_convergence(precommit: Mapping[str, Any], optix_summary: Mapping[str, Any]) -> dict[str, Any]:
    targets = [row for row in optix_summary["records"] if row.get("status") == "PASS" and row.get("base_id") in {"base_00_nominal", "base_01_candidate49"}]
    if not targets:
        return {"status": "INCONCLUSIVE", "reason": "anchor pair transport records missing"}
    runtime = create_runtime()
    records = []
    try:
        for target in targets:
            case_lookup = {case["case_id"]: case for case in _pair_cases(precommit)}
            left_case = case_lookup[f"{target['base_id']}__{target['arm']}__left"]
            right_case = case_lookup[f"{target['base_id']}__{target['arm']}__right"]
            fea3 = {row["case_id"]: row for row in strict_read_json(OUTPUT / "fea3d_summary.json")["records"]}
            tip_l, art_l = _load_3d_state(left_case, fea3[left_case["case_id"]])
            tip_r, art_r = _load_3d_state(right_case, fea3[right_case["case_id"]])
            bounds = (tuple(target["pair_grid"]["x_bounds_mm"]), tuple(target["pair_grid"]["y_bounds_mm"]))
            values = []
            for ray_count in FIDELITY_RAY_COUNTS:
                settings = _transport_settings("full3d", bounds, ray_count)
                left = trace_3d if False else None
                from optics.transport3d.transport import trace_geometry
                raw_l = trace_geometry(tip_l, art_l.geometry(tip_l), settings=settings, runtime=runtime)
                raw_r = trace_geometry(tip_r, art_r.geometry(tip_r), settings=settings, runtime=runtime)
                unified_l = UnifiedTransportResult.from_transport_result(raw_l, morphology_id=left_case["case_id"], morphology_fingerprint=left_case["morphology_fingerprint"], mechanics_source="fidelity", mechanics_dimension="3D", contact_state={}, transport_configuration_fingerprint="fidelity")
                unified_r = UnifiedTransportResult.from_transport_result(raw_r, morphology_id=right_case["case_id"], morphology_fingerprint=right_case["morphology_fingerprint"], mechanics_source="fidelity", mechanics_dimension="3D", contact_state={}, transport_configuration_fingerprint="fidelity")
                values.append({"ray_count": ray_count, "J3": native_field_separability(unified_l, unified_r)})
            records.append({"base_id": target["base_id"], "arm": target["arm"], "ray_convergence": values})
    finally:
        try:
            runtime.cp.cuda.Stream.null.synchronize()
            runtime.cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    return {"status": "PASS", "records": records, "ray_counts": list(FIDELITY_RAY_COUNTS), "selection_precommitted": True}


def _classify(fixed: Mapping[str, Any], varied: Mapping[str, Any], excluded_fraction: float) -> str:
    if excluded_fraction > 0.25:
        return "E_INCONCLUSIVE"
    thresholds = {"rho": 0.70, "tau": 0.50, "pair": 0.70}
    def strong(stats: Mapping[str, Any]) -> bool:
        return all(float(stats.get(key, math.nan)) >= value for key, value in (("spearman_rho", thresholds["rho"]), ("kendall_tau", thresholds["tau"]), ("pairwise_concordance_fraction", thresholds["pair"])))
    if fixed.get("status") != "PASS" or varied.get("status") != "PASS":
        return "E_INCONCLUSIVE"
    if strong(fixed) and strong(varied):
        return "A_STRONG_REDUCED_MODEL_TREND_PRESERVATION"
    delta = float(fixed.get("spearman_rho", 0.0)) - float(varied.get("spearman_rho", 0.0))
    if delta >= 0.20 and strong(fixed):
        return "B_VOID_HEIGHT_DOMINATED_DIMENSIONAL_MISMATCH"
    if delta >= 0.10:
        return "C_PARTIAL_MULTI_PARAMETER_MISMATCH"
    if not strong(fixed) and not strong(varied):
        return "D_BROAD_DIMENSIONAL_FAILURE"
    return "C_PARTIAL_MULTI_PARAMETER_MISMATCH"


def _analyze() -> dict[str, Any]:
    precommit = _load_precommit()
    optix = strict_read_json(OUTPUT / "optix_summary.json")
    rows = [row for row in optix["records"] if row.get("status") == "PASS"]
    populations = {}
    for arm in ("FIXED", "VARIED"):
        arm_rows = [row for row in rows if row.get("arm") == arm]
        populations[arm] = {
            "rows": arm_rows,
            "rank_stats": _rank_stats(arm_rows),
            "parameter_trends": _parameter_trends(arm_rows),
            "mechanics_trends": _mechanics_trends(arm_rows),
            "raw_transport_trends": _raw_transport_trends(arm_rows),
        }
    excluded_fraction = 1.0 - len(rows) / (BASE_DESIGN_COUNT * 2)
    disagreements = []
    for row in rows:
        delta = abs(float(row["J2"]["normalized_redistribution_l1"]) - float(row["J3"]["normalized_redistribution_l1"]))
        disagreements.append({"base_id": row["base_id"], "arm": row["arm"], "absolute_J2_minus_J3": delta, "void_height_mm": row["parameters"]["void_height"], "parameters": row["parameters"]})
    disagreements.sort(key=lambda item: float(item["absolute_J2_minus_J3"]), reverse=True)
    fixed_stats = populations["FIXED"]["rank_stats"]
    varied_stats = populations["VARIED"]["rank_stats"]
    fidelity = _fidelity_convergence(precommit, optix)
    summary = {
        "schema": "overnight-24-pair-analysis-v1",
        "precommit_fingerprint": precommit["precommit_fingerprint"],
        "planned_pair_count": BASE_DESIGN_COUNT * 2,
        "completed_pair_count": len(rows),
        "excluded_fraction": excluded_fraction,
        "populations": populations,
        "fixed_vs_varied": {
            "delta_spearman_rho": (float(fixed_stats.get("spearman_rho")) - float(varied_stats.get("spearman_rho"))) if fixed_stats.get("spearman_rho") is not None and varied_stats.get("spearman_rho") is not None else None,
            "delta_kendall_tau": (float(fixed_stats.get("kendall_tau")) - float(varied_stats.get("kendall_tau"))) if fixed_stats.get("kendall_tau") is not None and varied_stats.get("kendall_tau") is not None else None,
        },
        "strongest_disagreements": disagreements[:5],
        "void_height_analysis": {
            "fixed_void_height_values_mm": sorted({float(row["parameters"]["void_height"]) for row in rows if row["arm"] == "FIXED"}),
            "varied_void_height_min_mm": min((float(row["parameters"]["void_height"]) for row in rows if row["arm"] == "VARIED"), default=None),
            "varied_void_height_max_mm": max((float(row["parameters"]["void_height"]) for row in rows if row["arm"] == "VARIED"), default=None),
            "varied_void_height_j2_j3_trends": populations["VARIED"]["parameter_trends"].get("void_height"),
        },
        "optix_fidelity_convergence": fidelity,
        "periodic_z_limitation": {
            "status": "RETAINED",
            "reference_planes_mm": [-5.5, 5.5],
            "interpretation": "deformed surfaces are not clipped; each FULL_3D raw record retains actual z extent metadata",
        },
        "primary_analysis_separate_from_exploratory": True,
        "exploratory_analysis": {"strongest_disagreements": disagreements[:5]},
        "classification": _classify(fixed_stats, varied_stats, excluded_fraction),
        "socrates": {"status": "UNAVAILABLE", "reason": "no Socrates subagent is available in this workspace"},
        "created_at": _now(),
    }
    atomic_write_json(OUTPUT / "analysis_summary.json", _jsonable(summary))
    return summary


def _assemble() -> dict[str, Any]:
    """Artifact-only assembly; it never invokes Kratos or OptiX."""
    precommit = _load_precommit()
    result = {"schema": "overnight-artifact-assembly-v1", "precommit_fingerprint": precommit["precommit_fingerprint"], "cases": {}}
    for stage in ("fea2d", "fea3d"):
        for case in _pair_cases(precommit):
            record = _read_case(stage, case)
            result["cases"].setdefault(case["case_id"], {})[stage] = record["outcome"] if record else "NOT_RUN"
    result["created_at"] = _now()
    atomic_write_json(OUTPUT / "artifact_only_assembly.json", _jsonable(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("precommit", "fea2d", "fea3d", "optix", "analyze", "assemble", "all"), default="all")
    parser.add_argument("--child-stage", choices=("fea2d", "fea3d"))
    parser.add_argument("--case-id")
    args = parser.parse_args()
    if args.child_stage:
        if not args.case_id:
            raise SystemExit("--case-id is required for child execution")
        return _child_dispatch(args.child_stage, args.case_id)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "precommit":
        if PRECOMMIT.exists():
            payload = _load_precommit()
        else:
            payload = _precommit_payload()
            atomic_write_json(PRECOMMIT, _jsonable(payload))
        _stage_update(precommit_fingerprint=payload["precommit_fingerprint"], stage="precommit", planned_pair_count=BASE_DESIGN_COUNT * 2)
        print(json.dumps({"stage": "precommit", "status": "PASS", "precommit_fingerprint": payload["precommit_fingerprint"]}, sort_keys=True))
        return 0
    if args.stage == "assemble":
        result = _assemble()
    else:
        if not PRECOMMIT.exists():
            payload = _precommit_payload()
            atomic_write_json(PRECOMMIT, _jsonable(payload))
        if args.stage in ("fea2d", "all"):
            _run_mechanics_stage("fea2d")
        if args.stage in ("fea3d", "all"):
            _run_mechanics_stage("fea3d")
        if args.stage in ("optix", "all"):
            _run_optix_stage()
        if args.stage in ("analyze", "all"):
            result = _analyze()
        else:
            result = strict_read_json(OUTPUT / f"{args.stage}_summary.json") if (OUTPUT / f"{args.stage}_summary.json").exists() else {"status": "PASS"}
    print(json.dumps({"stage": args.stage, "status": result.get("classification", result.get("schema", "PASS"))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
