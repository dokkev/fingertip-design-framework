"""Bounded 24-step recovery study for the matched 2D contact contract.

The 12-step experiment is used only as a recorded comparison baseline.  This
module owns a new solver-policy fingerprint and never treats a 12-step state
artifact as a 24-step result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np

from fem.indentation import IndentationSettings, run_indentation_case
from mesh.fingertip import generate_fingertip_mesh
from mesh.indenter import IndenterSettings, build_normal_indenter_fixture_at_x
from model import Fingertip, FingertipParameters
from model.fingertip_model import FingertipModel
from validation.common.io import atomic_write_json, strict_read_json
from validation.fem.throughput import _mesh_policies
from validation import overnight_24_pair_trend_gap0_travel2 as twelve


OUTPUT = Path("output/validation/overnight_24_pair_trend_gap0_travel2_steps24")
PARENT_MANIFEST = Path("output/validation/overnight_24_pair_trend_gap0_travel2/experiment_manifest.json")
PARENT_SMOKE = Path("output/validation/overnight_24_pair_trend_gap0_travel2/smoke_summary.json")
MANIFEST = OUTPUT / "experiment_manifest.json"
STAGE_MANIFEST = OUTPUT / "stage_manifest.json"
STEPS = 24
BASELINE_STEPS = 12
INDENTATION_MM = 2.0
INITIAL_GAP_MM = 0.0
RADIUS_MM = 4.0
MESH_POLICY = "coarse_b"
CONTACT_LOCATIONS = {"left": -3.0, "right": 3.0}
CASE_TIMEOUT_SECONDS = 1800.0
EXPERIMENT_SCHEMA = "overnight-24-pair-trend-v3-gap0-travel2-steps24"

# Fixed before any 24-step result is read.  It contains nominal, candidate49,
# two previously passing 12-step states, and broad numerical-failure states.
SMOKE_CASE_IDS = (
    "base_00_nominal__FIXED__left",
    "base_00_nominal__FIXED__right",
    "base_00_nominal__VARIED__left",
    "base_00_nominal__VARIED__right",
    "base_01_candidate49__VARIED__left",
    "base_01_candidate49__VARIED__right",
    "base_05_lhs_04__FIXED__left",
    "base_05_lhs_04__FIXED__right",
    "base_07_lhs_06__VARIED__left",
    "base_07_lhs_06__VARIED__right",
    "base_20_lhs_19__VARIED__left",
    "base_20_lhs_19__VARIED__right",
)
CONSISTENCY_THRESHOLDS = {
    "reaction_relative_difference_max": 0.15,
    "max_displacement_relative_difference_max": 0.15,
    "rms_displacement_relative_difference_max": 0.15,
}
RECOVERY_GATE = {
    "minimum_recovery_fraction": 0.50,
    "maximum_regression_count": 0,
    "maximum_implementation_failure_count": 0,
    "stable_pass_consistency_required": True,
}


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


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path("fem/indentation.py"),
        Path("mesh/fingertip.py"),
        Path("mesh/indenter.py"),
        Path("validation/fem/throughput.py"),
    ):
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


def _load_parent() -> dict[str, Any]:
    if not PARENT_MANIFEST.is_file() or not PARENT_SMOKE.is_file():
        raise RuntimeError("the completed 12-step manifest and smoke baseline are required")
    parent = strict_read_json(PARENT_MANIFEST)
    twelve._verify_experiment(parent)
    smoke = strict_read_json(PARENT_SMOKE)
    if smoke.get("experiment_fingerprint") != parent.get("experiment_fingerprint"):
        raise RuntimeError("12-step smoke baseline does not match its experiment manifest")
    return parent


def _experiment_payload(parent: Mapping[str, Any]) -> dict[str, Any]:
    pairs = json.loads(json.dumps(parent["pairs"]))
    source_sha256 = {
        str(path): _sha256_file(Path(path))
        for path in (
            PARENT_MANIFEST,
            PARENT_SMOKE,
            Path(__file__),
            Path("fem/indentation.py"),
            Path("mesh/fingertip.py"),
            Path("mesh/indenter.py"),
        )
        if Path(path).is_file()
    }
    payload: dict[str, Any] = {
        "schema": EXPERIMENT_SCHEMA,
        "created_at": _now(),
        "scope": "bounded 24-step recovery smoke; 3D/reference paused",
        "parent_12_step_manifest": str(PARENT_MANIFEST),
        "parent_12_step_smoke": str(PARENT_SMOKE),
        "parent_sampling_fingerprint": parent["parent_sampling_fingerprint"],
        "parent_pairs_fingerprint": _fingerprint(parent["pairs"]),
        "design_space": parent["design_space"],
        "sampling": {
            "preserved": True,
            "resampling": "none",
            "base_design_count": 24,
            "source_manifest": str(PARENT_MANIFEST),
        },
        "pairing": parent["pairing"],
        "pairs": pairs,
        "mechanics": {
            "two_d_mesh_policy": MESH_POLICY,
            "steps": STEPS,
            "baseline_steps": BASELINE_STEPS,
            "indentation_mm": INDENTATION_MM,
            "initial_gap_mm": INITIAL_GAP_MM,
            "indenter_radius_mm": RADIUS_MM,
            "contact_locations_mm": CONTACT_LOCATIONS,
            "young_modulus_mpa": parent["mechanics"]["young_modulus_mpa"],
            "poisson_ratio": parent["mechanics"]["poisson_ratio"],
            "internal_contact": parent["mechanics"]["internal_contact"],
            "adaptive_stepping": False,
        },
        "smoke_selection": {
            "case_ids": list(SMOKE_CASE_IDS),
            "selection_precommitted": True,
            "selection_rule": (
                "fixed nominal/candidate49 anchors plus broad precommitted cases; "
                "includes both prior PASS and NUMERICAL_FAIL 12-step outcomes"
            ),
            "result_dependent_selection": False,
        },
        "consistency_thresholds": CONSISTENCY_THRESHOLDS,
        "recovery_gate": RECOVERY_GATE,
        "execution_policy": {
            "case_execution": "one fresh child process per case, sequential",
            "case_timeout_seconds": CASE_TIMEOUT_SECONDS,
            "full_96_status": "NOT_STARTED_PENDING_RECOVERY_GATE",
            "reference_fidelity": "NOT_RUN",
            "three_d_status": "NOT_STARTED_BY_USER_POLICY",
            "optix": "NOT_STARTED",
        },
        "provenance": {"source_sha256": source_sha256},
    }
    payload["experiment_fingerprint"] = _fingerprint(
        {key: value for key, value in payload.items() if key != "created_at"}
    )
    return payload


def _verify_experiment(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != EXPERIMENT_SCHEMA:
        raise RuntimeError("24-step experiment schema is stale")
    expected = _fingerprint(
        {
            key: value
            for key, value in payload.items()
            if key not in {"created_at", "experiment_fingerprint"}
        }
    )
    if payload.get("experiment_fingerprint") != expected:
        raise RuntimeError("24-step experiment fingerprint mismatch")
    mechanics = payload.get("mechanics", {})
    expected_values = {
        "steps": STEPS,
        "baseline_steps": BASELINE_STEPS,
        "indentation_mm": INDENTATION_MM,
        "initial_gap_mm": INITIAL_GAP_MM,
        "two_d_mesh_policy": MESH_POLICY,
        "adaptive_stepping": False,
    }
    for key, value in expected_values.items():
        if mechanics.get(key) != value:
            raise RuntimeError(f"24-step contract mismatch for {key}")
    if tuple(payload.get("smoke_selection", {}).get("case_ids", ())) != SMOKE_CASE_IDS:
        raise RuntimeError("24-step smoke selection changed")
    if len(payload.get("pairs", [])) != 24:
        raise RuntimeError("24-step experiment does not preserve 24 pairs")


def _load_experiment() -> dict[str, Any]:
    if not MANIFEST.is_file():
        raise RuntimeError(f"missing 24-step manifest: {MANIFEST}")
    payload = strict_read_json(MANIFEST)
    _verify_experiment(payload)
    return payload


def _case_list(experiment: Mapping[str, Any]) -> list[dict[str, Any]]:
    return twelve._case_list(experiment)


def _case_contract(case: Mapping[str, Any]) -> str:
    return _fingerprint(
        {
            "schema": "overnight-gap0-travel2-steps24-child-v1",
            "case": {
                key: case[key]
                for key in (
                    "case_id",
                    "base_id",
                    "arm",
                    "side",
                    "contact_x_mm",
                    "parameters",
                    "morphology_fingerprint",
                )
            },
            "mechanics": {
                "mesh_policy": MESH_POLICY,
                "steps": STEPS,
                "baseline_steps": BASELINE_STEPS,
                "indentation_mm": INDENTATION_MM,
                "initial_gap_mm": INITIAL_GAP_MM,
                "radius_mm": RADIUS_MM,
            },
            "implementation_fingerprint": _implementation_fingerprint(),
        }
    )


def _case_path(case: Mapping[str, Any]) -> Path:
    return OUTPUT / "fea2d" / f"{case['case_id']}.json"


def _log_path(case: Mapping[str, Any], stream: str) -> Path:
    return OUTPUT / "logs" / "fea2d" / f"{case['case_id']}.{stream}.log"


def _write_case(case: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    atomic_write_json(_case_path(case), _jsonable(payload))


def _failure(case: Mapping[str, Any], outcome: str, reason: str, **extra: Any) -> None:
    _write_case(
        case,
        {
            "schema": "overnight-gap0-travel2-steps24-child-v1",
            "stage": "fea2d_steps24",
            "case_id": case["case_id"],
            "base_id": case["base_id"],
            "arm": case["arm"],
            "side": case["side"],
            "parameters": case["parameters"],
            "morphology_fingerprint": case["morphology_fingerprint"],
            "case_fingerprint": _case_contract(case),
            "outcome": outcome,
            "status": outcome,
            "failure_reason": reason,
            "created_at": _now(),
            **_jsonable(extra),
        },
    )


def _run_child(case: Mapping[str, Any]) -> int:
    started = time.perf_counter()
    try:
        parameters = FingertipParameters(**case["parameters"])
        morphology_start = time.perf_counter()
        model = FingertipModel(parameters)
        tip = Fingertip(parameters)
        morphology_seconds = time.perf_counter() - morphology_start

        mesh_start = time.perf_counter()
        policy = next(item for item in _mesh_policies() if item.name == MESH_POLICY)
        mesh = generate_fingertip_mesh(model, policy.settings)
        mesh_seconds = time.perf_counter() - mesh_start

        fixture_start = time.perf_counter()
        fixture = build_normal_indenter_fixture_at_x(
            model,
            float(case["contact_x_mm"]),
            IndenterSettings(radius_mm=RADIUS_MM, initial_gap_mm=INITIAL_GAP_MM),
        )
        fixture_seconds = time.perf_counter() - fixture_start

        result, artifacts = run_indentation_case(
            model,
            "medium",
            IndentationSettings(INDENTATION_MM, STEPS),
            fixture_override=fixture,
            internal_contact_configuration="three_pairs",
            basal_interface="explicit_contact",
            mesh_override=mesh,
            diagnostic_mode="minimal",
        )
        solve_timing = result.get("timing", {})
        timing_profile = {
            "morphology_construction_wall_clock_seconds": morphology_seconds,
            "mesh_generation_wall_clock_seconds": mesh_seconds,
            "fixture_construction_wall_clock_seconds": fixture_seconds,
            "kratos_setup_wall_clock_seconds": solve_timing.get("setup_wall_clock_seconds"),
            "nonlinear_solver_wall_clock_seconds": solve_timing.get("nonlinear_solve_wall_clock_seconds"),
            "solver_postprocess_wall_clock_seconds": (
                float(solve_timing.get("per_step_postprocess_wall_clock_seconds", 0.0))
                + float(solve_timing.get("final_extraction_wall_clock_seconds", 0.0))
            ),
        }
        timing_profile["child_wall_clock_seconds"] = time.perf_counter() - started
        if artifacts is None or result.get("solve_status") != "PASS":
            _failure(
                case,
                "NUMERICAL_FAIL",
                str(result.get("failure_reason") or "24-step solve did not converge"),
                result=result,
                timing_profile=timing_profile,
            )
            return 1

        raw = artifacts.snapshots[f"{INDENTATION_MM:g}"]["displacements"]
        displacement = np.asarray(
            [raw[int(node_id)] for node_id in mesh.pad.node_ids],
            dtype=float,
        ) if isinstance(raw, Mapping) else np.asarray(raw, dtype=float)
        final = result.get("final", {})
        reaction = final.get("indenter_normal_reaction_n")
        checks = twelve._smoke_checks(mesh, displacement, result, reaction)
        if not checks["passed"]:
            _failure(
                case,
                "NUMERICAL_FAIL",
                "24-step mechanical validity checks failed",
                result=result,
                checks=checks,
                timing_profile=timing_profile,
            )
            return 1

        state_path = _case_path(case).with_suffix(".npz")
        serialization_start = time.perf_counter()
        _atomic_npz(state_path, displacement=displacement)
        payload = {
            "schema": "overnight-gap0-travel2-steps24-child-v1",
            "stage": "fea2d_steps24",
            "case_id": case["case_id"],
            "base_id": case["base_id"],
            "arm": case["arm"],
            "side": case["side"],
            "case_fingerprint": _case_contract(case),
            "outcome": "PASS",
            "status": "PASS",
            "morphology_fingerprint": case["morphology_fingerprint"],
            "parameters": case["parameters"],
            "mesh_policy": MESH_POLICY,
            "mesh_settings": asdict(mesh.settings),
            "steps": STEPS,
            "baseline_steps": BASELINE_STEPS,
            "indentation_mm": INDENTATION_MM,
            "initial_gap_mm": INITIAL_GAP_MM,
            "indenter_radius_mm": RADIUS_MM,
            "contact_x_mm": case["contact_x_mm"],
            "reaction_force_n": reaction,
            "history": result.get("history", []),
            "contact_state": final.get("contact_groups", {}),
            "max_displacement_mm": float(np.linalg.norm(displacement, axis=1).max()),
            "rms_displacement_mm": float(np.sqrt(np.mean(np.sum(displacement * displacement, axis=1)))),
            "checks": checks,
            "state_artifact": str(state_path),
            "state_sha256": _sha256_file(state_path),
            "result_timing": result.get("timing", {}),
            "configuration": result.get("configuration", {}),
        }
        timing_profile["artifact_serialization_wall_clock_seconds"] = time.perf_counter() - serialization_start
        payload["timing_profile"] = timing_profile
        payload["runtime_seconds"] = timing_profile["child_wall_clock_seconds"]
        _write_case(case, payload)
        return 0
    except Exception as exc:
        _failure(
            case,
            "IMPLEMENTATION_FAIL",
            f"{type(exc).__name__}: {exc}",
            runtime_seconds=time.perf_counter() - started,
        )
        return 1


def _child_dispatch(case_id: str) -> int:
    experiment = _load_experiment()
    cases = {case["case_id"]: case for case in _case_list(experiment)}
    if case_id not in cases:
        raise RuntimeError(f"unknown case id {case_id}")
    return _run_child(cases[case_id])


def _read_case(case: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _case_path(case)
    if not path.is_file():
        return None
    try:
        payload = strict_read_json(path)
        if payload.get("case_id") != case["case_id"] or payload.get("parameters") != case["parameters"]:
            return None
        if payload.get("outcome") not in {"PASS", "NUMERICAL_FAIL", "IMPLEMENTATION_FAIL", "RUNTIME_LIMIT"}:
            return None
        result = payload.get("result") or {}
        configuration = payload.get("configuration", result.get("configuration", {}))
        indentation = configuration.get("indentation", {})
        indenter = configuration.get("indenter", {}).get("settings", {})
        expected_mesh_settings = asdict(
            next(item for item in _mesh_policies() if item.name == MESH_POLICY).settings
        )
        stable_contract = (
            indentation.get("indentation_mm") == INDENTATION_MM
            and indentation.get("number_of_steps") == STEPS
            and indenter.get("initial_gap_mm") == INITIAL_GAP_MM
            and indenter.get("radius_mm") == RADIUS_MM
            and configuration.get("mesh_settings") == expected_mesh_settings
            and payload.get("status") == payload.get("outcome")
        )
        if payload.get("outcome") == "PASS":
            stable_contract = stable_contract and (
                payload.get("mesh_policy") == MESH_POLICY
                and payload.get("steps") == STEPS
                and payload.get("indentation_mm") == INDENTATION_MM
                and payload.get("initial_gap_mm") == INITIAL_GAP_MM
            )
        if payload.get("case_fingerprint") != _case_contract(case) and not stable_contract:
            return None
        if payload.get("case_fingerprint") != _case_contract(case):
            payload["case_fingerprint"] = _case_contract(case)
            payload["artifact_rebound_without_solver_rerun"] = True
            atomic_write_json(path, _jsonable(payload))
        if payload.get("outcome") == "PASS":
            state = Path(str(payload.get("state_artifact", "")))
            if not state.is_file() or payload.get("state_sha256") != _sha256_file(state):
                return None
        return payload
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _run_case(case: Mapping[str, Any]) -> dict[str, Any]:
    existing = _read_case(case)
    if existing is not None:
        existing["reused"] = True
        return existing
    log_out = _log_path(case, "stdout")
    log_err = _log_path(case, "stderr")
    log_out.parent.mkdir(parents=True, exist_ok=True)
    parent_start = time.perf_counter()
    try:
        with log_out.open("w", encoding="utf-8") as stdout, log_err.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                [sys.executable, "-m", "validation.overnight_24_pair_trend_gap0_travel2_steps24", "--child-stage", "fea2d", "--case-id", case["case_id"]],
                stdout=stdout,
                stderr=stderr,
                timeout=CASE_TIMEOUT_SECONDS,
                check=False,
            )
        payload = _read_case(case)
        if payload is None:
            _failure(
                case,
                "IMPLEMENTATION_FAIL" if completed.returncode >= 0 else "RUNTIME_LIMIT",
                "child exited without a valid atomic result",
                return_code=completed.returncode,
                logs={"stdout": str(log_out), "stderr": str(log_err)},
            )
        else:
            payload["parent_return_code"] = completed.returncode
            payload["parent_wall_time_seconds"] = time.perf_counter() - parent_start
            child_wall = payload.get("timing_profile", {}).get("child_wall_clock_seconds")
            if child_wall is not None:
                payload["orchestration_wall_time_seconds"] = max(
                    0.0, float(payload["parent_wall_time_seconds"]) - float(child_wall)
                )
            atomic_write_json(_case_path(case), _jsonable(payload))
            return payload
    except subprocess.TimeoutExpired:
        _failure(
            case,
            "RUNTIME_LIMIT",
            "configured engineering runtime budget expired; no scientific conclusion",
            runtime_limit_seconds=CASE_TIMEOUT_SECONDS,
            logs={"stdout": str(log_out), "stderr": str(log_err)},
        )
    payload = _read_case(case)
    if payload is None:
        raise RuntimeError(f"failed to persist 24-step outcome for {case['case_id']}")
    payload["parent_wall_time_seconds"] = time.perf_counter() - parent_start
    return payload


def _stage_update(**updates: Any) -> None:
    current = strict_read_json(STAGE_MANIFEST) if STAGE_MANIFEST.is_file() else {}
    current.update(updates)
    current["updated_at"] = _now()
    atomic_write_json(STAGE_MANIFEST, current)


def _baseline_records() -> dict[str, Mapping[str, Any]]:
    smoke = strict_read_json(PARENT_SMOKE)
    return {record["case_id"]: record for record in smoke["records"]}


def _relative_difference(first: Any, second: Any) -> float | None:
    if first is None or second is None:
        return None
    first_value = float(first)
    second_value = float(second)
    scale = max(abs(first_value), abs(second_value), 1.0e-12)
    return abs(first_value - second_value) / scale


def _consistency(baseline: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    reaction_delta = _relative_difference(
        baseline.get("reaction_force_n"), current.get("reaction_force_n")
    )
    max_delta = _relative_difference(
        baseline.get("max_displacement_mm"), current.get("max_displacement_mm")
    )
    rms_delta = _relative_difference(
        baseline.get("rms_displacement_mm"), current.get("rms_displacement_mm")
    )
    baseline_contact = baseline.get("contact_state", {}).get("external_pad_indenter", {})
    current_contact = current.get("contact_state", {}).get("external_pad_indenter", {})
    contact_match = (
        int(baseline_contact.get("active_condition_count", 0)) > 0
        and int(current_contact.get("active_condition_count", 0)) > 0
    )
    checks = {
        "reaction_within_threshold": reaction_delta is not None and reaction_delta <= CONSISTENCY_THRESHOLDS["reaction_relative_difference_max"],
        "max_displacement_within_threshold": max_delta is not None and max_delta <= CONSISTENCY_THRESHOLDS["max_displacement_relative_difference_max"],
        "rms_displacement_within_threshold": rms_delta is not None and rms_delta <= CONSISTENCY_THRESHOLDS["rms_displacement_relative_difference_max"],
        "external_contact_active_in_both": contact_match,
        "current_checks_pass": bool(current.get("checks", {}).get("passed", False)),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "reaction_relative_difference": reaction_delta,
        "max_displacement_relative_difference": max_delta,
        "rms_displacement_relative_difference": rms_delta,
    }


def _classify(baseline: Mapping[str, Any], current: Mapping[str, Any]) -> str:
    old = baseline.get("outcome")
    new = current.get("outcome")
    if old == "IMPLEMENTATION_FAIL" or new == "IMPLEMENTATION_FAIL":
        return "IMPLEMENTATION_FAIL"
    if old == "NUMERICAL_FAIL" and new == "PASS":
        return "RECOVERED"
    if old == "PASS" and new == "PASS":
        return "STABLE_PASS"
    if old == "NUMERICAL_FAIL" and new == "NUMERICAL_FAIL":
        return "PERSISTENT_NUMERICAL_FAIL"
    if old == "PASS" and new != "PASS":
        return "REGRESSION"
    return "IMPLEMENTATION_FAIL"


def _validity_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    checks = record.get("checks") or {}
    check_flags = checks.get("checks") or {}
    result = record.get("result") or {}
    acceptance = result.get("case_acceptance_checks") or {}
    contact_groups = record.get("contact_state") or (result.get("final") or {}).get("contact_groups") or {}
    external = contact_groups.get("external_pad_indenter") or {}
    mesh = result.get("mesh") or {}
    return {
        "checks_available": bool(check_flags or acceptance),
        "mechanical_checks": check_flags or acceptance,
        "mechanical_checks_passed": checks.get("passed"),
        "valid_deformed_mesh": check_flags.get("valid_deformed_mesh"),
        "active_external_contact": check_flags.get("active_external_contact"),
        "active_external_condition_count": external.get("active_condition_count"),
        "generated_external_condition_count": external.get("generated_condition_count"),
        "penetration_pass": external.get("penetration_pass"),
        "finite_contact_pressure": (
            (external.get("lagrange_multiplier_contact_pressure") or {}).get("finite")
        ),
        "mesh_policy": record.get("mesh_policy") or record.get("mesh_level") or mesh.get("level"),
        "minimum_pad_det_f": result.get("minimum_pad_det_f"),
        "maximum_pad_strain": result.get("maximum_pad_strain"),
        "deformed_mesh_error": checks.get("deformed_mesh_error"),
    }


def _comparison_rows(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    baselines = _baseline_records()
    rows = []
    for current in records:
        baseline = baselines.get(current["case_id"], {})
        classification = _classify(baseline, current)
        row = {
            "case_id": current["case_id"],
            "base_id": current["base_id"],
            "arm": current["arm"],
            "side": current["side"],
            "twelve_step": {
                "outcome": baseline.get("outcome", "NOT_AVAILABLE"),
                "failure_reason": baseline.get("failure_reason"),
                "failure_step": baseline.get("result", {}).get("failure_step"),
                "reaction_force_n": baseline.get("reaction_force_n"),
                "max_displacement_mm": baseline.get("max_displacement_mm"),
                "rms_displacement_mm": baseline.get("rms_displacement_mm"),
                "runtime_seconds": baseline.get("parent_wall_time_seconds") or baseline.get("timing_profile", {}).get("child_wall_clock_seconds"),
            },
            "twenty_four_step": {
                "outcome": current.get("outcome"),
                "failure_reason": current.get("failure_reason"),
                "failure_step": current.get("result", {}).get("failure_step"),
                "reaction_force_n": current.get("reaction_force_n"),
                "max_displacement_mm": current.get("max_displacement_mm"),
                "rms_displacement_mm": current.get("rms_displacement_mm"),
                "runtime_seconds": current.get("parent_wall_time_seconds") or current.get("runtime_seconds"),
            },
            "validity": {
                "twelve_step": _validity_summary(baseline),
                "twenty_four_step": _validity_summary(current),
            },
            "classification": classification,
        }
        if classification == "STABLE_PASS":
            row["final_state_consistency"] = _consistency(baseline, current)
        rows.append(row)
    return rows


def _summary(name: str, records: list[Mapping[str, Any]], *, comparison: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    runtimes = [
        float(record.get("parent_wall_time_seconds") or record.get("runtime_seconds"))
        for record in records
        if record.get("parent_wall_time_seconds") is not None or record.get("runtime_seconds") is not None
    ]
    result: dict[str, Any] = {
        "schema": f"overnight-gap0-travel2-steps24-{name}-v1",
        "experiment_fingerprint": _load_experiment()["experiment_fingerprint"],
        "records": records,
        "planned_cases": len(records),
        "counts": {
            outcome: sum(record.get("outcome") == outcome for record in records)
            for outcome in ("PASS", "NUMERICAL_FAIL", "IMPLEMENTATION_FAIL", "RUNTIME_LIMIT")
        },
        "runtime_seconds": {
            "count": len(runtimes),
            "mean": float(np.mean(runtimes)) if runtimes else None,
            "median": float(np.median(runtimes)) if runtimes else None,
            "minimum": float(np.min(runtimes)) if runtimes else None,
            "maximum": float(np.max(runtimes)) if runtimes else None,
        },
        "created_at": _now(),
    }
    if comparison is not None:
        result["comparison"] = comparison
        category_counts = {
            category: sum(row["classification"] == category for row in comparison)
            for category in (
                "RECOVERED",
                "STABLE_PASS",
                "PERSISTENT_NUMERICAL_FAIL",
                "REGRESSION",
                "IMPLEMENTATION_FAIL",
            )
        }
        baseline_numeric = sum(
            row["twelve_step"]["outcome"] == "NUMERICAL_FAIL" for row in comparison
        )
        recovered = category_counts["RECOVERED"]
        stable_consistency = [
            row.get("final_state_consistency", {}).get("passed", False)
            for row in comparison
            if row["classification"] == "STABLE_PASS"
        ]
        result["recovery"] = {
            "category_counts": category_counts,
            "baseline_numerical_fail_count": baseline_numeric,
            "recovered_count": recovered,
            "recovery_fraction": recovered / baseline_numeric if baseline_numeric else None,
            "stable_pass_consistency_all_pass": all(stable_consistency),
            "gate": {
                "minimum_recovery_fraction": RECOVERY_GATE["minimum_recovery_fraction"],
                "maximum_regression_count": RECOVERY_GATE["maximum_regression_count"],
                "maximum_implementation_failure_count": RECOVERY_GATE["maximum_implementation_failure_count"],
                "passed": (
                    baseline_numeric > 0
                    and recovered / baseline_numeric >= RECOVERY_GATE["minimum_recovery_fraction"]
                    and category_counts["REGRESSION"] <= RECOVERY_GATE["maximum_regression_count"]
                    and category_counts["IMPLEMENTATION_FAIL"] <= RECOVERY_GATE["maximum_implementation_failure_count"]
                    and (
                        not RECOVERY_GATE["stable_pass_consistency_required"]
                        or all(stable_consistency)
                    )
                ),
            },
        }
    atomic_write_json(OUTPUT / f"{name}_summary.json", _jsonable(result))
    return result


def _run_smoke() -> dict[str, Any]:
    experiment = _load_experiment()
    cases_by_id = {case["case_id"]: case for case in _case_list(experiment)}
    cases = [cases_by_id[case_id] for case_id in SMOKE_CASE_IDS]
    records = []
    for index, case in enumerate(cases, start=1):
        record = _run_case(case)
        records.append(record)
        _stage_update(stage="smoke", completed_case_count=index, case_outcomes={row["case_id"]: row.get("outcome") for row in records})
    comparison = _comparison_rows(records)
    summary = _summary("smoke", records, comparison=comparison)
    summary["status"] = (
        "PASS"
        if summary.get("recovery", {}).get("gate", {}).get("passed", False)
        else "BLOCKED_2D_MECHANICS"
    )
    summary["selection"] = {
        "case_ids": list(SMOKE_CASE_IDS),
        "precommitted": True,
        "selection_result_independent": True,
    }
    atomic_write_json(OUTPUT / "smoke_summary.json", _jsonable(summary))
    return summary


def _run_full() -> dict[str, Any]:
    smoke = strict_read_json(OUTPUT / "smoke_summary.json")
    if not smoke.get("recovery", {}).get("gate", {}).get("passed", False):
        raise RuntimeError("24-step full stage is gated by recovery smoke; no 96-case matrix launched")
    experiment = _load_experiment()
    records = []
    for index, case in enumerate(_case_list(experiment), start=1):
        record = _run_case(case)
        records.append(record)
        _stage_update(stage="fea2d_steps24", full_completed_case_count=index, full_case_outcomes={row["case_id"]: row.get("outcome") for row in records})
    summary = _summary("fea2d", records)
    summary["status"] = "PASS" if len(records) == 96 else "INCOMPLETE"
    atomic_write_json(OUTPUT / "fea2d_summary.json", _jsonable(summary))
    return summary


def _assemble() -> dict[str, Any]:
    experiment = _load_experiment()
    result: dict[str, Any] = {
        "schema": "overnight-gap0-travel2-steps24-artifact-assembly-v1",
        "experiment_fingerprint": experiment["experiment_fingerprint"],
        "stages_invoked": ["assemble"],
        "cases": {},
        "created_at": _now(),
    }
    for case in _case_list(experiment):
        record = _read_case(case)
        result["cases"][case["case_id"]] = record.get("outcome") if record else "NOT_RUN"
    atomic_write_json(OUTPUT / "artifact_only_assembly.json", _jsonable(result))
    _stage_update(stage="assemble", assembly_status="PASS")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("precommit", "smoke", "fea2d", "assemble"), default="precommit")
    parser.add_argument("--child-stage", choices=("fea2d",))
    parser.add_argument("--case-id")
    args = parser.parse_args()
    if args.child_stage:
        if not args.case_id:
            raise SystemExit("--case-id is required for child execution")
        return _child_dispatch(args.case_id)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "precommit":
        parent = _load_parent()
        candidate = _experiment_payload(parent)
        if MANIFEST.is_file():
            existing = strict_read_json(MANIFEST)
            _verify_experiment(existing)
            if existing.get("experiment_fingerprint") == candidate.get("experiment_fingerprint"):
                candidate = existing
        atomic_write_json(MANIFEST, _jsonable(candidate))
        _stage_update(stage="precommit", experiment_fingerprint=candidate["experiment_fingerprint"], planned_cases=12)
        print(json.dumps({"stage": "precommit", "status": "PASS", "experiment_fingerprint": candidate["experiment_fingerprint"]}, sort_keys=True))
        return 0
    if args.stage == "smoke":
        result = _run_smoke()
    elif args.stage == "fea2d":
        result = _run_full()
    else:
        result = _assemble()
    print(json.dumps({"stage": args.stage, "status": result.get("status", result.get("recovery", {}).get("gate", {}).get("passed", "PASS"))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
