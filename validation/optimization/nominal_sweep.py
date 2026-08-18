"""Run the resumable pre-BO nominal morphology sweep."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

from scipy.stats import qmc

from model import (
    FingertipParameters,
    silicone_ligament_measures,
    validate_silicone_ligament,
)
from optimization import (
    PRODUCTION_FIXED_FLAT_PAD_WIDTH_MM,
    PRODUCTION_SEARCH_BOUNDS,
    ScenarioGrid,
    create_production_study,
)
from optimization.evaluator import DesignEvaluation, DesignEvaluator
from validation.common.io import atomic_write_json, strict_read_json
from validation.common.runner import run_isolated
from validation.optimization.dry_run import _evaluation_to_dict


DEFAULT_OUTPUT = Path("output/validation/optimization/pre_bo_nominal_sweep")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOBOL_SEED = 20260815
SAMPLE_COUNT = 64
SOBOL_EXPONENT = 6
DESIGN_TIMEOUT_SECONDS = 1800.0
FIXED_FLAT_PAD_WIDTH_MM = PRODUCTION_FIXED_FLAT_PAD_WIDTH_MM
SWEPT_RANGES = PRODUCTION_SEARCH_BOUNDS
OBJECTIVE_FIELDS = (
    "score",
    "minimum_auc",
    "mean_auc",
    "median_auc",
    "minimum_raw_contact_metric",
    "mean_raw_contact_metric",
)


class SweepConfigurationError(ValueError):
    """Raised when an existing checkpoint does not match this sweep."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scenario_configuration(grid: ScenarioGrid) -> dict[str, Any]:
    return {
        "locations_x_mm": list(grid.locations_x_mm),
        "indenter_radii_mm": list(grid.indenter_radii_mm),
        "indenter_diameters_mm": [
            2.0 * radius for radius in grid.indenter_radii_mm
        ],
        "captured_depths_mm": list(grid.captured_depths_mm),
        "trajectory_count": grid.trajectory_count,
        "captured_state_count": grid.captured_state_count,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _create_evaluator() -> DesignEvaluator:
    return create_production_study().create_evaluator()


def _configuration() -> dict[str, Any]:
    evaluator = _create_evaluator()
    return _jsonable({
        "fixed_flat_pad_width_mm": FIXED_FLAT_PAD_WIDTH_MM,
        "swept_ranges_mm": {
            name: {"lower": lower, "upper": upper}
            for name, lower, upper in SWEPT_RANGES
        },
        "swept_parameter_order": [name for name, _, _ in SWEPT_RANGES],
        "sobol": {
            "scrambled": True,
            "seed": SOBOL_SEED,
            "sample_count": SAMPLE_COUNT,
            "dimension": len(SWEPT_RANGES),
            "draw": "random_base2(m=6)",
        },
        "scenario_grid": _scenario_configuration(evaluator.scenario_grid),
        "mesh_settings": asdict(evaluator.mesh_settings),
        "trace_settings": asdict(evaluator.trace_settings),
        "optical_mode": evaluator.trace_settings.mode,
        "indenter_optics": asdict(evaluator.indenter_optics),
        "fem_steps": evaluator.fem_steps,
        "internal_contact": evaluator.internal_contact,
        "basal_interface": evaluator.basal_interface,
        "timeout_seconds": DESIGN_TIMEOUT_SECONDS,
        "protocol": "pre-BO nominal morphology exploratory sweep",
    })


def sobol_proposals() -> list[list[float]]:
    """Return the deterministic normalized Sobol proposal set."""
    sampler = qmc.Sobol(
        d=len(SWEPT_RANGES),
        scramble=True,
        seed=SOBOL_SEED,
    )
    points = sampler.random_base2(m=SOBOL_EXPONENT)
    return [[float(value) for value in point] for point in points]


def _decode_parameters(normalized_point: list[float]) -> FingertipParameters:
    return FingertipParameters(**_decoded_parameter_values(normalized_point))


def _decoded_parameter_values(normalized_point: list[float]) -> dict[str, float]:
    if len(normalized_point) != len(SWEPT_RANGES):
        raise ValueError("normalized Sobol point has the wrong dimension")
    updates: dict[str, float] = {
        "flat_pad_width": FIXED_FLAT_PAD_WIDTH_MM,
    }
    for normalized, (name, lower, upper) in zip(
        normalized_point,
        SWEPT_RANGES,
        strict=True,
    ):
        updates[name] = lower + normalized * (upper - lower)
    updates["flat_pad_width"] = FIXED_FLAT_PAD_WIDTH_MM
    updates["semielliptical_pad_height"] = 14.0 - updates["flat_pad_height"]
    updates["void_height"] = 0.0
    return updates


def _ligament_fields(
    parameters: FingertipParameters | Mapping[str, float],
) -> dict[str, float]:
    measures = silicone_ligament_measures(parameters)
    return {
        "side_ligament_mm": measures.side_ligament_mm,
        "ellipse_depth_at_cutout_mm": measures.ellipse_depth_at_cutout_mm,
        "distal_ligament_mm": measures.distal_ligament_mm,
        "minimum_silicone_ligament_mm": measures.minimum_silicone_ligament_mm,
    }


def _strip_failed_objectives(evaluation: dict[str, Any]) -> dict[str, Any]:
    if evaluation.get("status") == "success":
        return evaluation
    for field in OBJECTIVE_FIELDS:
        evaluation.pop(field, None)
    return evaluation


def _evaluation_record(
    evaluation: DesignEvaluation,
    *,
    wall_time_seconds: float,
    total_states: int,
) -> dict[str, Any]:
    return _strip_failed_objectives(
        _evaluation_to_dict(
            evaluation,
            wall_time_seconds=wall_time_seconds,
            total_states=total_states,
        )
    )


def _child_arguments(input_path: Path, result_path: Path) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "validation.optimization.nominal_sweep",
        "--_child",
        "--_input",
        str(input_path),
        "--_result",
        str(result_path),
    ]


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONFAULTHANDLER": "1",
        }
    )
    return environment


def _run_child(arguments: argparse.Namespace) -> int:
    if arguments.input_path is None or arguments.result_path is None:
        raise ValueError("child input and result paths are required")
    payload = strict_read_json(arguments.input_path)
    parameters = FingertipParameters(**payload["parameters"])
    evaluator = _create_evaluator()
    start = time.perf_counter()
    try:
        evaluation = evaluator.evaluate(parameters)
    except Exception as exc:
        result = {
            "status": "unexpected_failure",
            "failure_category": "process_failure",
            "failure_message": f"{type(exc).__name__}: {exc}",
            "wall_time_seconds": time.perf_counter() - start,
            "parameters": asdict(parameters),
            "evaluation": None,
        }
        atomic_write_json(arguments.result_path, result)
        return 1

    elapsed = time.perf_counter() - start
    result = {
        "status": evaluation.status,
        "failure_category": _failure_category(evaluation.status),
        "failure_message": evaluation.failure_message,
        "wall_time_seconds": elapsed,
        "parameters": asdict(parameters),
        "evaluation": _evaluation_record(
            evaluation,
            wall_time_seconds=elapsed,
            total_states=evaluator.scenario_grid.captured_state_count,
        ),
    }
    atomic_write_json(arguments.result_path, result)
    return 0


def _failure_category(status: str) -> str | None:
    if status == "success":
        return None
    if status == "invalid_design":
        return "geometry_rejected"
    if status in {"mesh_failure", "fea_failure"}:
        return "fem_failure"
    if status == "optics_failure":
        return "optics_failure"
    return "process_failure"


def _is_environment_blocker(record: Mapping[str, Any]) -> bool:
    message = str(record.get("failure_message") or "").lower()
    return (
        record.get("status") == "fea_failure"
        and "kratos" in message
        and any(token in message for token in ("depend", "unavailable", "import"))
    )


def _run_isolated_design(
    parameters: FingertipParameters,
    output: Path,
    identifier: str,
) -> dict[str, Any]:
    input_path = output / "inputs" / f"{identifier}.json"
    result_path = output / "child_results" / f"{identifier}.json"
    log_path = output / "child_logs" / f"{identifier}.log"
    atomic_write_json(input_path, {"parameters": asdict(parameters)})
    start = time.perf_counter()
    completed = run_isolated(
        _child_arguments(input_path, result_path),
        cwd=REPOSITORY_ROOT,
        environment=_child_environment(),
        output_path=log_path,
        timeout_seconds=DESIGN_TIMEOUT_SECONDS,
        disable_core_dumps=True,
    )
    wall_time_seconds = time.perf_counter() - start
    record: dict[str, Any] = {
        "status": "timeout" if completed.timed_out else "process_failure",
        "failure_category": "timeout" if completed.timed_out else "process_failure",
        "failure_message": (
            f"child process exceeded {DESIGN_TIMEOUT_SECONDS:g} seconds"
            if completed.timed_out
            else "child process exited without a usable result"
        ),
        "wall_time_seconds": wall_time_seconds,
        "parameters": asdict(parameters),
        "child_exit_status": completed.return_code,
        "child_result_path": str(result_path),
        "child_log_path": str(log_path),
    }
    if completed.timed_out:
        return record
    if completed.return_code != 0:
        if result_path.is_file():
            record["child_result"] = strict_read_json(result_path)
            record["failure_message"] = (
                f"child process exited with status {completed.return_code}"
            )
        return record
    if not result_path.is_file():
        return record

    child_result = strict_read_json(result_path)
    record.update(
        {
            "status": child_result.get("status", "process_failure"),
            "failure_category": child_result.get(
                "failure_category", "process_failure"
            ),
            "failure_message": child_result.get("failure_message"),
            "evaluation": child_result.get("evaluation"),
        }
    )
    return record


def _base_record(
    index: int,
    normalized_point: list[float],
) -> dict[str, Any]:
    return {
        "candidate_index": index,
        "normalized_sobol_point": normalized_point,
        "parameters": None,
        "geometry_admissible": False,
        "side_ligament_mm": None,
        "ellipse_depth_at_cutout_mm": None,
        "distal_ligament_mm": None,
        "minimum_silicone_ligament_mm": None,
        "status": "pending",
        "failure_category": None,
        "failure_message": None,
        "wall_time_seconds": 0.0,
        "child_exit_status": None,
    }


def _evaluate_candidate(
    index: int,
    normalized_point: list[float],
    output: Path,
) -> dict[str, Any]:
    start = time.perf_counter()
    record = _base_record(index, normalized_point)
    try:
        parameter_values = _decoded_parameter_values(normalized_point)
        record["parameters"] = parameter_values
        try:
            record.update(_ligament_fields(parameter_values))
        except (ArithmeticError, KeyError, TypeError, ValueError):
            pass
        parameters = FingertipParameters(**parameter_values)
        validate_silicone_ligament(parameters)
    except Exception as exc:
        record.update(
            {
                "status": "geometry_rejected",
                "failure_category": "geometry_rejected",
                "failure_message": f"{type(exc).__name__}: {exc}",
                "wall_time_seconds": time.perf_counter() - start,
            }
        )
        return record

    record["geometry_admissible"] = True
    child = _run_isolated_design(parameters, output, f"candidate_{index:04d}")
    record.update(child)
    record["wall_time_seconds"] = time.perf_counter() - start
    return record


def _initial_state(
    configuration: Mapping[str, Any],
    proposals: list[list[float]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "RUNNING",
        "created_at": _now(),
        "updated_at": _now(),
        "configuration": dict(configuration),
        "normalized_sobol_points": proposals,
        "nominal_result": None,
        "candidates": [],
    }


def _load_state(
    output: Path,
    configuration: Mapping[str, Any],
    proposals: list[list[float]],
) -> dict[str, Any]:
    state_path = output / "checkpoint.json"
    if not state_path.is_file():
        state = _initial_state(configuration, proposals)
        atomic_write_json(state_path, state)
        return state
    state = strict_read_json(state_path)
    if state.get("configuration") != dict(configuration):
        raise SweepConfigurationError(
            "checkpoint configuration differs from the current sweep configuration"
        )
    if state.get("normalized_sobol_points") != proposals:
        raise SweepConfigurationError(
            "checkpoint Sobol proposals differ from the deterministic proposal set"
        )
    return state


def _save_state(output: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    atomic_write_json(output / "checkpoint.json", state)


def _upsert_candidate(state: dict[str, Any], record: Mapping[str, Any]) -> None:
    index = int(record["candidate_index"])
    candidates = [
        existing
        for existing in state.get("candidates", [])
        if int(existing.get("candidate_index", -1)) != index
    ]
    candidates.append(dict(record))
    candidates.sort(key=lambda item: int(item["candidate_index"]))
    state["candidates"] = candidates


def _record_counts(candidates: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "success": 0,
        "geometry_rejected": 0,
        "fem_failure": 0,
        "optics_failure": 0,
        "timeout": 0,
        "process_failure": 0,
    }
    for candidate in candidates:
        category = str(candidate.get("failure_category") or candidate.get("status"))
        if category not in counts:
            category = "process_failure"
        counts[category] += 1
    return counts


def _format(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"


def _print_progress(
    record: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    nominal_objective: float | None,
) -> None:
    evaluation = record.get("evaluation") or {}
    limiting = evaluation.get("limiting_trajectory")
    minimum = evaluation.get("minimum_auc")
    delta = (
        float(minimum) - nominal_objective
        if minimum is not None and nominal_objective is not None
        else None
    )
    print(
        f"candidate {int(record['candidate_index']):02d}/{SAMPLE_COUNT} "
        f"status={record.get('status')} "
        f"wall_s={_format(record.get('wall_time_seconds'))} "
        f"min_ligament_mm={_format(record.get('minimum_silicone_ligament_mm'))} "
        f"min_auc={_format(minimum)} "
        f"delta={_format(delta)} "
        f"limiting={limiting or 'n/a'}"
        + (
            f" failure={record['failure_message']}"
            if record.get("failure_message")
            else ""
    ), flush=True)
    counts = _record_counts(candidates)
    print(
        "  counts "
        + " ".join(f"{name}={counts[name]}" for name in counts),
        flush=True,
    )


def _nominal_objective(nominal_result: Mapping[str, Any]) -> float | None:
    evaluation = nominal_result.get("evaluation") or {}
    value = evaluation.get("minimum_auc")
    return None if value is None else float(value)


def _summary(
    state: Mapping[str, Any],
    *,
    status: str,
    output: Path,
) -> dict[str, Any]:
    candidates = list(state.get("candidates", []))
    counts = _record_counts(candidates)
    successful = [
        candidate
        for candidate in candidates
        if candidate.get("status") == "success"
        and isinstance(candidate.get("evaluation"), Mapping)
    ]
    objectives = sorted(
        float(candidate["evaluation"]["minimum_auc"])
        for candidate in successful
    )
    nominal_result = state.get("nominal_result") or {}
    nominal_objective = _nominal_objective(nominal_result)
    better_than_nominal = (
        sum(
            float(candidate["evaluation"]["minimum_auc"])
            > nominal_objective
            for candidate in successful
        )
        if nominal_objective is not None
        else None
    )
    limiting_axis_counts: dict[str, int] = {}
    for candidate in successful:
        limiting = candidate["evaluation"].get("limiting_trajectory")
        if limiting is not None:
            key = str(limiting)
            limiting_axis_counts[key] = limiting_axis_counts.get(key, 0) + 1
    top_candidates = sorted(
        successful,
        key=lambda candidate: float(
            candidate["evaluation"]["minimum_auc"]
        ),
        reverse=True,
    )[:10]
    top_table = [
        {
            "candidate_index": candidate["candidate_index"],
            "minimum_auc": candidate["evaluation"]["minimum_auc"],
            "mean_auc": candidate["evaluation"]["mean_auc"],
            "limiting_trajectory": candidate["evaluation"].get(
                "limiting_trajectory"
            ),
            "parameters": candidate["parameters"],
            "side_ligament_mm": candidate["side_ligament_mm"],
            "distal_ligament_mm": candidate["distal_ligament_mm"],
            "minimum_silicone_ligament_mm": candidate[
                "minimum_silicone_ligament_mm"
            ],
        }
        for candidate in top_candidates
    ]
    wall_times = [
        float(candidate.get("wall_time_seconds", 0.0)) for candidate in candidates
    ]
    if nominal_result:
        wall_times.append(float(nominal_result.get("wall_time_seconds", 0.0)))
    return {
        "status": status,
        "created_at": state.get("created_at"),
        "completed_at": _now() if status == "COMPLETE" else None,
        "configuration": state["configuration"],
        "total_proposals": SAMPLE_COUNT,
        "successful_designs": counts["success"],
        "geometry_rejected_designs": counts["geometry_rejected"],
        "fem_failures": counts["fem_failure"],
        "optics_failures": counts["optics_failure"],
        "timeouts": counts["timeout"],
        "process_failures": counts["process_failure"],
        "total_wall_time_seconds": sum(wall_times),
        "nominal_result": nominal_result,
        "nominal_minimum_auc": nominal_objective,
        "objective_distribution": {
            "minimum": objectives[0] if objectives else None,
            "maximum": objectives[-1] if objectives else None,
            "median": (
                objectives[len(objectives) // 2]
                if objectives and len(objectives) % 2
                else (
                    0.5
                    * (objectives[len(objectives) // 2 - 1] + objectives[len(objectives) // 2])
                    if objectives
                    else None
                )
            ),
        },
        "successful_designs_better_than_nominal": better_than_nominal,
        "top_10_successful_candidates": top_table,
        "limiting_axis_counts": limiting_axis_counts,
        "raw_checkpoint_path": str(output / "checkpoint.json"),
    }


def _print_nominal(result: Mapping[str, Any]) -> None:
    evaluation = result.get("evaluation") or {}
    print(
        "nominal "
        f"status={result.get('status')} "
        f"wall_s={_format(result.get('wall_time_seconds'))} "
        f"minimum_auc={_format(evaluation.get('minimum_auc'))}"
        + (
            f" failure={result['failure_message']}"
            if result.get("failure_message")
            else ""
        ),
        flush=True,
    )


def _run_parent(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    configuration = _configuration()
    proposals = sobol_proposals()
    state = _load_state(output, configuration, proposals)

    if state.get("nominal_result") is None:
        nominal_parameters = FingertipParameters()
        nominal = _run_isolated_design(nominal_parameters, output, "nominal")
        nominal.update(
            {
                "parameters": asdict(nominal_parameters),
                "geometry_admissible": True,
                **_ligament_fields(nominal_parameters),
            }
        )
        state["nominal_result"] = nominal
        _save_state(output, state)
    nominal = state["nominal_result"]
    _print_nominal(nominal)

    if _is_environment_blocker(nominal):
        state["status"] = "BLOCKED"
        _save_state(output, state)
        summary = _summary(state, status="BLOCKED", output=output)
        atomic_write_json(output / "summary.json", summary)
        print(f"sweep blocked by environment: {nominal.get('failure_message')}")
        return 2

    completed = {
        int(candidate["candidate_index"]): candidate
        for candidate in state.get("candidates", [])
    }
    nominal_objective = _nominal_objective(nominal)
    for offset, normalized_point in enumerate(proposals):
        index = offset + 1
        if index in completed:
            _print_progress(
                completed[index], list(completed.values()), nominal_objective
            )
            continue
        record = _evaluate_candidate(index, normalized_point, output)
        _upsert_candidate(state, record)
        _save_state(output, state)
        completed[index] = record
        _print_progress(record, list(completed.values()), nominal_objective)

    state["status"] = "COMPLETE"
    _save_state(output, state)
    summary = _summary(state, status="COMPLETE", output=output)
    atomic_write_json(output / "summary.json", summary)
    print(f"summary: {output / 'summary.json'}", flush=True)
    return 0


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_input", dest="input_path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_result", dest="result_path", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    if arguments._child:
        return _run_child(arguments)
    try:
        return _run_parent(arguments.output_directory.expanduser().resolve())
    except SweepConfigurationError as exc:
        print(f"sweep configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
