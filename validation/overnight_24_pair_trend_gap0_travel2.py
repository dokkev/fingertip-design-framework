"""Run the matched-contact 2D stage of the broad morphology trend study.

This module is intentionally a new experiment boundary.  It imports the
already precommitted 24-pair sampling manifest, changes only the physical
contact contract, and does not expose a 3D or reference execution stage.
Generated artifacts are restartable and remain below ``output/``.
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


OUTPUT = Path("output/validation/overnight_24_pair_trend_gap0_travel2")
PARENT_MANIFEST = Path("output/validation/overnight_24_pair_trend/experiment_manifest.json")
MANIFEST = OUTPUT / "experiment_manifest.json"
STAGE_MANIFEST = OUTPUT / "stage_manifest.json"
STEPS = 12
INDENTATION_MM = 2.0
INITIAL_GAP_MM = 0.0
RADIUS_MM = 4.0
CONTACT_LOCATIONS = {"left": -3.0, "right": 3.0}
MESH_POLICY = "coarse_b"
PLANNED_3D_TIER = "search"
CASE_TIMEOUT_SECONDS = 1800.0
SMOKE_SCHEMA = "overnight-gap0-travel2-smoke-v1"
EXPERIMENT_SCHEMA = "overnight-24-pair-trend-v2-gap0-travel2"


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
    paths = (
        Path(__file__),
        Path("fem/indentation.py"),
        Path("mesh/fingertip.py"),
        Path("mesh/indenter.py"),
        Path("validation/fem/throughput.py"),
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


def _parameters_fingerprint(parameters: Mapping[str, Any]) -> str:
    return _fingerprint(dict(parameters))


def _validate_parent_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "overnight-24-pair-trend-v1":
        raise RuntimeError("parent manifest is not the authoritative v1 24-pair manifest")
    expected = _fingerprint(
        {
            key: value
            for key, value in payload.items()
            if key not in {"created_at", "precommit_fingerprint"}
        }
    )
    if payload.get("precommit_fingerprint") != expected:
        raise RuntimeError("parent manifest fingerprint does not validate")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 24:
        raise RuntimeError("parent manifest does not contain exactly 24 pairs")
    for pair in pairs:
        if set(pair.get("arms", {})) != {"FIXED", "VARIED"}:
            raise RuntimeError(f"{pair.get('base_id')} does not contain both arms")
        for arm in ("FIXED", "VARIED"):
            data = pair["arms"][arm]
            parameters = data.get("parameters")
            if not isinstance(parameters, Mapping):
                raise RuntimeError(f"{pair.get('base_id')} {arm} has no parameters")
            actual = _parameters_fingerprint(parameters)
            if data.get("parameters_fingerprint") not in (None, actual):
                raise RuntimeError(f"{pair.get('base_id')} {arm} parameter fingerprint mismatch")


def _load_parent_manifest() -> dict[str, Any]:
    if not PARENT_MANIFEST.is_file():
        raise RuntimeError(f"missing authoritative sampling manifest: {PARENT_MANIFEST}")
    payload = strict_read_json(PARENT_MANIFEST)
    _validate_parent_manifest(payload)
    return payload


def _experiment_payload(parent: Mapping[str, Any]) -> dict[str, Any]:
    pairs = json.loads(json.dumps(parent["pairs"]))
    source_sha256 = {
        str(path): _sha256_file(Path(path))
        for path in (
            PARENT_MANIFEST,
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
        "scope": "matched 2D circle FEA stage; 3D matrix paused",
        "historical_positive_gap_manifest": str(PARENT_MANIFEST),
        "parent_sampling_fingerprint": parent["precommit_fingerprint"],
        "parent_pairs_fingerprint": _fingerprint(parent["pairs"]),
        "design_space": parent["design_space"],
        "sampling": {
            "preserved": True,
            "base_design_count": 24,
            "anchors": ["nominal", "candidate49"],
            "resampling": "none",
            "source_manifest": str(PARENT_MANIFEST),
        },
        "pairing": parent["pairing"],
        "pairs": pairs,
        "mechanics": {
            "two_d_mesh_policy": MESH_POLICY,
            "three_d_mesh_tier": PLANNED_3D_TIER,
            "three_d_status": "NOT_STARTED_BY_USER_POLICY",
            "steps": STEPS,
            "indentation_mm": INDENTATION_MM,
            "initial_gap_mm": INITIAL_GAP_MM,
            "indenter_radius_mm": RADIUS_MM,
            "contact_locations_mm": CONTACT_LOCATIONS,
            "young_modulus_mpa": float(parent["mechanics"]["young_modulus_mpa"]),
            "poisson_ratio": float(parent["mechanics"]["poisson_ratio"]),
            "internal_contact": parent["mechanics"]["internal_contact"],
            "external_contact": True,
        },
        "execution_policy": {
            "case_execution": "one fresh child process per case, sequential",
            "parallelism": "not used",
            "case_timeout_seconds": CASE_TIMEOUT_SECONDS,
            "reference_fidelity": "NOT_RUN",
            "optix": "NOT_STARTED",
            "artifact_only_assembly": True,
        },
        "smoke_policy": {
            "selection": "nominal, candidate49, and deterministic broad/extreme existing samples",
            "selection_result_independent": True,
            "required_checks": [
                "left_right_convergence",
                "finite_reaction",
                "finite_displacement",
                "valid_deformed_mesh",
                "active_external_contact",
                "meaningful_internal_void_deformation",
            ],
            "full_stage_gate": "all selected smoke cases PASS and all required checks PASS",
        },
        "provenance": {"source_sha256": source_sha256},
    }
    payload["experiment_fingerprint"] = _fingerprint(
        {key: value for key, value in payload.items() if key != "created_at"}
    )
    return payload


def _verify_experiment(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != EXPERIMENT_SCHEMA:
        raise RuntimeError("gap0/travel2 experiment schema is stale")
    expected = _fingerprint(
        {key: value for key, value in payload.items() if key not in {"created_at", "experiment_fingerprint"}}
    )
    if payload.get("experiment_fingerprint") != expected:
        raise RuntimeError("gap0/travel2 experiment fingerprint mismatch")
    mechanics = payload.get("mechanics", {})
    if mechanics.get("initial_gap_mm") != INITIAL_GAP_MM:
        raise RuntimeError("experiment does not encode initial gap 0.0 mm")
    if mechanics.get("indentation_mm") != INDENTATION_MM:
        raise RuntimeError("experiment does not encode 2.0 mm travel")
    if mechanics.get("steps") != STEPS or mechanics.get("two_d_mesh_policy") != MESH_POLICY:
        raise RuntimeError("experiment does not encode coarse_b/12-step 2D mechanics")
    if len(payload.get("pairs", [])) != 24:
        raise RuntimeError("experiment does not preserve 24 pairs")


def _load_experiment() -> dict[str, Any]:
    if not MANIFEST.is_file():
        raise RuntimeError(f"missing experiment manifest: {MANIFEST}; run --stage precommit")
    payload = strict_read_json(MANIFEST)
    _verify_experiment(payload)
    return payload


def _case_list(experiment: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for pair in experiment["pairs"]:
        for arm in ("FIXED", "VARIED"):
            arm_data = pair["arms"][arm]
            for side, x_mm in CONTACT_LOCATIONS.items():
                cases.append(
                    {
                        "case_id": f"{pair['base_id']}__{arm}__{side}",
                        "base_id": pair["base_id"],
                        "anchor": pair.get("anchor"),
                        "arm": arm,
                        "side": side,
                        "contact_x_mm": x_mm,
                        "parameters": arm_data["parameters"],
                        "morphology_fingerprint": arm_data["morphology_fingerprint"],
                    }
                )
    return cases


def _case_contract(case: Mapping[str, Any]) -> str:
    return _fingerprint(
        {
            "schema": "overnight-gap0-travel2-child-v1",
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
                "two_d_mesh_policy": MESH_POLICY,
                "steps": STEPS,
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
            "schema": "overnight-gap0-travel2-child-v1",
            "stage": "fea2d",
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


def _internal_deformation(mesh: Any, displacement: np.ndarray) -> dict[str, Any]:
    indices: set[int] = set()
    tag_counts: dict[str, int] = {}
    for tag in ("pad_cutout_left", "pad_cutout_right", "pad_cutout_bottom"):
        tag_indices = np.asarray(mesh.pad.boundary_node_indices_for(tag), dtype=int)
        tag_counts[tag] = int(tag_indices.size)
        indices.update(int(index) for index in tag_indices)
    if not indices:
        return {
            "finite": False,
            "meaningful": False,
            "node_count": 0,
            "rms_displacement_mm": None,
            "max_displacement_mm": None,
            "tag_node_counts": tag_counts,
        }
    values = np.linalg.norm(displacement[sorted(indices)], axis=1)
    rms = float(np.sqrt(np.mean(values * values)))
    maximum = float(np.max(values))
    return {
        "finite": bool(np.all(np.isfinite(values))),
        "meaningful": math.isfinite(rms) and rms > 1.0e-8,
        "node_count": len(indices),
        "rms_displacement_mm": rms,
        "max_displacement_mm": maximum,
        "tag_node_counts": tag_counts,
    }


def _smoke_checks(mesh: Any, displacement: np.ndarray, result: Mapping[str, Any], reaction: Any) -> dict[str, Any]:
    try:
        mesh.pad.deformed(displacement)
        valid_deformed_mesh = True
        deformation_error = None
    except Exception as exc:
        valid_deformed_mesh = False
        deformation_error = f"{type(exc).__name__}: {exc}"
    contact = result.get("final", {}).get("contact_groups", {})
    external = contact.get("external_pad_indenter", {})
    active_count = int(external.get("active_condition_count", 0))
    internal = _internal_deformation(mesh, displacement)
    finite_reaction = reaction is not None and math.isfinite(float(reaction))
    checks = {
        "finite_reaction": finite_reaction,
        "positive_reaction": finite_reaction and float(reaction) > 0.0,
        "finite_displacement": bool(np.all(np.isfinite(displacement))),
        "valid_deformed_mesh": valid_deformed_mesh,
        "active_external_contact": active_count > 0,
        "meaningful_internal_void_deformation": bool(internal["meaningful"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "external_active_condition_count": active_count,
        "internal_void_deformation": internal,
        "deformed_mesh_error": deformation_error,
    }


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
            mesh_override=mesh,
            diagnostic_mode="minimal",
        )
        solve_timing = result.get("timing", {})
        if artifacts is None or result.get("solve_status") != "PASS":
            _failure(
                case,
                "NUMERICAL_FAIL",
                str(result.get("failure_reason") or result.get("exception") or "2D solve did not converge"),
                result=result,
                timing_profile={
                    "morphology_construction_wall_clock_seconds": morphology_seconds,
                    "mesh_generation_wall_clock_seconds": mesh_seconds,
                    "fixture_construction_wall_clock_seconds": fixture_seconds,
                    "kratos_model_initialization_and_contact_setup_wall_clock_seconds": solve_timing.get("setup_wall_clock_seconds"),
                    "nonlinear_solver_wall_clock_seconds": solve_timing.get("nonlinear_solve_wall_clock_seconds"),
                    "solver_postprocess_wall_clock_seconds": (
                        float(solve_timing.get("per_step_postprocess_wall_clock_seconds", 0.0))
                        + float(solve_timing.get("final_extraction_wall_clock_seconds", 0.0))
                    ),
                    "child_wall_clock_seconds": time.perf_counter() - started,
                },
            )
            return 1

        displacement = np.asarray(artifacts.snapshots[str(INDENTATION_MM)]["displacements"], dtype=float)
        if displacement.shape != mesh.pad.coordinates.shape:
            _failure(case, "NUMERICAL_FAIL", "2D displacement artifact has the wrong shape", result=result)
            return 1
        final = result.get("final", {})
        reaction = final.get("indenter_normal_reaction_n")
        checks = _smoke_checks(mesh, displacement, result, reaction)
        if not checks["passed"]:
            _failure(
                case,
                "NUMERICAL_FAIL",
                "2D mechanical validity checks failed",
                result=result,
                checks=checks,
            )
            return 1

        state_path = _case_path(case).with_suffix(".npz")
        serialization_start = time.perf_counter()
        _atomic_npz(state_path, displacement=displacement)
        payload = {
            "schema": "overnight-gap0-travel2-child-v1",
            "stage": "fea2d",
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
        payload["timing_profile"] = {
            "morphology_construction_wall_clock_seconds": morphology_seconds,
            "mesh_generation_wall_clock_seconds": mesh_seconds,
            "fixture_construction_wall_clock_seconds": fixture_seconds,
            "kratos_model_initialization_and_contact_setup_wall_clock_seconds": solve_timing.get("setup_wall_clock_seconds"),
            "nonlinear_solver_wall_clock_seconds": solve_timing.get("nonlinear_solve_wall_clock_seconds"),
            "solver_postprocess_wall_clock_seconds": (
                float(solve_timing.get("per_step_postprocess_wall_clock_seconds", 0.0))
                + float(solve_timing.get("final_extraction_wall_clock_seconds", 0.0))
            ),
            "artifact_serialization_wall_clock_seconds": None,
            "child_wall_clock_seconds": None,
        }
        serialization_seconds = time.perf_counter() - serialization_start
        payload["timing_profile"]["artifact_serialization_wall_clock_seconds"] = serialization_seconds
        payload["timing_profile"]["child_wall_clock_seconds"] = time.perf_counter() - started
        payload["runtime_seconds"] = payload["timing_profile"]["child_wall_clock_seconds"]
        _write_case(case, payload)
        return 0
    except Exception as exc:
        _failure(
            case,
            "NUMERICAL_FAIL",
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
        if payload.get("case_fingerprint") != _case_contract(case):
            return None
        if payload.get("case_id") != case["case_id"]:
            return None
        if payload.get("outcome") not in {"PASS", "NUMERICAL_FAIL", "RUNTIME_LIMIT", "ENVIRONMENT_FAIL"}:
            return None
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
                [sys.executable, "-m", "validation.overnight_24_pair_trend_gap0_travel2", "--child-stage", "fea2d", "--case-id", case["case_id"]],
                stdout=stdout,
                stderr=stderr,
                timeout=CASE_TIMEOUT_SECONDS,
                check=False,
            )
        payload = _read_case(case)
        if payload is None:
            _failure(
                case,
                "ENVIRONMENT_FAIL" if completed.returncode < 0 else "IMPLEMENTATION_FAIL",
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
        raise RuntimeError(f"failed to persist parent outcome for {case['case_id']}")
    payload["parent_wall_time_seconds"] = time.perf_counter() - parent_start
    return payload


def _stage_update(**updates: Any) -> None:
    current = strict_read_json(STAGE_MANIFEST) if STAGE_MANIFEST.is_file() else {}
    current.update(updates)
    current["updated_at"] = _now()
    atomic_write_json(STAGE_MANIFEST, current)


def _write_summary(name: str, records: list[Mapping[str, Any]], *, status: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    counts = {
        outcome: sum(record.get("outcome") == outcome for record in records)
        for outcome in ("PASS", "NUMERICAL_FAIL", "RUNTIME_LIMIT", "ENVIRONMENT_FAIL", "IMPLEMENTATION_FAIL")
    }
    runtimes = [
        float(record["parent_wall_time_seconds"])
        for record in records
        if record.get("parent_wall_time_seconds") is not None
    ]
    summary: dict[str, Any] = {
        "schema": f"overnight-gap0-travel2-{name}-v1",
        "experiment_fingerprint": _load_experiment()["experiment_fingerprint"],
        "status": status,
        "planned_cases": len(records),
        "records": records,
        "counts": counts,
        "runtime_seconds": {
            "count": len(runtimes),
            "mean": float(np.mean(runtimes)) if runtimes else None,
            "median": float(np.median(runtimes)) if runtimes else None,
            "minimum": float(np.min(runtimes)) if runtimes else None,
            "maximum": float(np.max(runtimes)) if runtimes else None,
        },
        "created_at": _now(),
    }
    if extra:
        summary.update(_jsonable(extra))
    atomic_write_json(OUTPUT / f"{name}_summary.json", _jsonable(summary))
    return summary


def _select_smoke_cases(experiment: Mapping[str, Any]) -> list[dict[str, Any]]:
    pairs = list(experiment["pairs"])
    selected: list[str] = ["base_00_nominal", "base_01_candidate49"]
    remaining = [pair for pair in pairs if pair["base_id"] not in selected]

    def coordinate_score(pair: Mapping[str, Any]) -> float:
        values = pair["arms"]["VARIED"]["normalized_coordinate"].values()
        return float(sum(float(value) for value in values))

    def add(base_id: str) -> None:
        if base_id not in selected:
            selected.append(base_id)

    add(min(remaining, key=coordinate_score)["base_id"])
    add(max(remaining, key=coordinate_score)["base_id"])
    add(min(remaining, key=lambda pair: float(pair["arms"]["VARIED"]["parameters"]["void_height"]))["base_id"])
    add(max(remaining, key=lambda pair: float(pair["arms"]["VARIED"]["parameters"]["void_height"]))["base_id"])
    selected_pairs = [pair for pair in pairs if pair["base_id"] in selected]
    selected_pairs.sort(key=lambda pair: pairs.index(pair))
    selected_cases = [
        case
        for case in _case_list({"pairs": selected_pairs})
    ]
    return selected_cases


def _check_left_right(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["base_id"], record["arm"]), {})[record["side"]] = record
    rows = []
    for key, sides in sorted(grouped.items()):
        left = sides.get("left")
        right = sides.get("right")
        rows.append(
            {
                "base_id": key[0],
                "arm": key[1],
                "left_outcome": left.get("outcome") if left else "NOT_RUN",
                "right_outcome": right.get("outcome") if right else "NOT_RUN",
                "both_pass": bool(left and right and left.get("outcome") == "PASS" and right.get("outcome") == "PASS"),
            }
        )
    return {
        "rows": rows,
        "complete_pair_count": sum(row["both_pass"] for row in rows),
        "pair_count": len(rows),
    }


def _run_profile() -> dict[str, Any]:
    experiment = _load_experiment()
    case = next(
        case for case in _case_list(experiment)
        if case["base_id"] == "base_00_nominal" and case["arm"] == "FIXED" and case["side"] == "left"
    )
    record = _run_case(case)
    profile = {
        "schema": "overnight-gap0-travel2-profile-v1",
        "experiment_fingerprint": experiment["experiment_fingerprint"],
        "case_id": case["case_id"],
        "record": record,
        "path_contract": {
            "mesh_policy": MESH_POLICY,
            "steps": STEPS,
            "reference_mesh_loaded": False,
            "diagnostic_mode": "minimal",
            "visualization_export": False,
            "repeated_solve": False,
            "continuation": "single production run with fresh solver state",
            "symmetry_or_legacy_validation": False,
        },
        "created_at": _now(),
    }
    atomic_write_json(OUTPUT / "nominal_timing_profile.json", _jsonable(profile))
    _stage_update(profile_status=record.get("outcome"), profile_case_id=case["case_id"])
    return profile


def _run_smoke() -> dict[str, Any]:
    experiment = _load_experiment()
    cases = _select_smoke_cases(experiment)
    records = []
    for index, case in enumerate(cases, start=1):
        record = _run_case(case)
        records.append(record)
        _stage_update(smoke_completed_case_count=index, smoke_case_outcomes={row["case_id"]: row.get("outcome") for row in records})
    left_right = _check_left_right(records)
    status = "PASS" if all(record.get("outcome") == "PASS" for record in records) and all(
        row["both_pass"] for row in left_right["rows"]
    ) else "SMOKE_BLOCKED"
    return _write_summary(
        "smoke",
        records,
        status=status,
        extra={
            "selected_base_ids": sorted({case["base_id"] for case in cases}),
            "selection_method": "precommitted manifest only; nominal/candidate anchors plus coordinate/void extremes",
            "left_right_convergence": left_right,
        },
    )


def _run_full_2d() -> dict[str, Any]:
    experiment = _load_experiment()
    smoke_path = OUTPUT / "smoke_summary.json"
    if not smoke_path.is_file():
        raise RuntimeError("2D full stage requires a completed smoke stage")
    smoke = strict_read_json(smoke_path)
    if smoke.get("status") != "PASS":
        raise RuntimeError("2D full stage is gated by smoke PASS; no full matrix launched")
    cases = _case_list(experiment)
    records = []
    for index, case in enumerate(cases, start=1):
        record = _run_case(case)
        records.append(record)
        _stage_update(full_2d_completed_case_count=index, full_2d_case_outcomes={row["case_id"]: row.get("outcome") for row in records})
    left_right = _check_left_right(records)
    status = "PASS" if len(records) == 96 else "INCOMPLETE"
    return _write_summary("fea2d", records, status=status, extra={"left_right_convergence": left_right})


def _assemble() -> dict[str, Any]:
    experiment = _load_experiment()
    cases = _case_list(experiment)
    result: dict[str, Any] = {
        "schema": "overnight-gap0-travel2-artifact-only-assembly-v1",
        "experiment_fingerprint": experiment["experiment_fingerprint"],
        "cases": {},
        "stages_invoked": [],
        "created_at": _now(),
    }
    for case in cases:
        record = _read_case(case)
        result["cases"][case["case_id"]] = record.get("outcome") if record else "NOT_RUN"
    atomic_write_json(OUTPUT / "artifact_only_assembly.json", _jsonable(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("precommit", "profile", "smoke", "fea2d", "assemble"), default="precommit")
    parser.add_argument("--child-stage", choices=("fea2d",))
    parser.add_argument("--case-id")
    args = parser.parse_args()
    if args.child_stage:
        if not args.case_id:
            raise SystemExit("--case-id is required for child execution")
        return _child_dispatch(args.case_id)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "precommit":
        parent = _load_parent_manifest()
        if MANIFEST.is_file():
            experiment = _load_experiment()
        else:
            experiment = _experiment_payload(parent)
            atomic_write_json(MANIFEST, _jsonable(experiment))
        _stage_update(experiment_fingerprint=experiment["experiment_fingerprint"], stage="precommit", planned_2d_cases=96)
        print(json.dumps({"stage": "precommit", "status": "PASS", "experiment_fingerprint": experiment["experiment_fingerprint"]}, sort_keys=True))
        return 0
    if args.stage == "profile":
        result = _run_profile()
    elif args.stage == "smoke":
        result = _run_smoke()
    elif args.stage == "fea2d":
        result = _run_full_2d()
    else:
        result = _assemble()
    print(json.dumps({"stage": args.stage, "status": result.get("status", "PASS")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
