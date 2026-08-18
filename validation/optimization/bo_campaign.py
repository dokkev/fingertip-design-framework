"""Run and persist the first fixed-budget production Ax campaign."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from optics import IndenterOptics
from optimization import PRODUCTION_SEARCH_BOUNDS, create_production_study
from optimization.ax_adapter import (
    AxRunResult,
    AxSettings,
    AxTrialRecord,
    run_ax_optimization,
)
from optimization.study import OptimizationStudy
from validation.common.io import atomic_write_json
from validation.optimization.dry_run import _evaluation_to_dict


DEFAULT_OUTPUT = Path("output/validation/optimization/production_bo_20260818")
CAMPAIGN_SEED = 20260818
INITIALIZATION_TRIALS = 8
SEARCH_TRIALS = 18
TOTAL_ATTEMPTS = 1 + INITIALIZATION_TRIALS + SEARCH_TRIALS
NEAR_BOUND_FRACTION = 0.05


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _scenario_configuration(study: OptimizationStudy) -> dict[str, Any]:
    grid = study.scenario_grid
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


def _configuration(
    study: OptimizationStudy,
    settings: AxSettings,
) -> dict[str, Any]:
    grid = study.scenario_grid
    active_bounds = {
        name: {"lower": lower, "upper": upper}
        for name, lower, upper in PRODUCTION_SEARCH_BOUNDS
    }
    if not grid.is_production_protocol:
        raise RuntimeError(
            "production ScenarioGrid does not match the campaign protocol"
        )
    if tuple(active_bounds) != tuple(
        variable.name for variable in study.design_space.active_variables
    ):
        raise RuntimeError(
            "production active-variable order differs from frozen bounds"
        )
    if study.trace_settings.mode != "planar":
        raise RuntimeError("production campaign requires PLANAR_2D")
    if study.indenter_optics != IndenterOptics("absorber"):
        raise RuntimeError("production campaign requires absorber indenter optics")
    if study.fem_steps != 48:
        raise RuntimeError("production campaign requires 48 FEM steps")
    if (
        study.internal_contact != "sides_separate"
        or study.basal_interface != "bonded"
    ):
        raise RuntimeError("production campaign requires bonded+sides_separate")
    return _jsonable(
        {
            "settings": {
                "initialization_trials": settings.initialization_trials,
                "search_trials": settings.search_trials,
                "seed": settings.seed,
            },
            "expected_attempts": {
                "nominal": 1,
                "initialization": settings.initialization_trials,
                "search": settings.search_trials,
                "total": TOTAL_ATTEMPTS,
            },
            "production_search_bounds_mm": active_bounds,
            "scenario_grid": _scenario_configuration(study),
            "mesh_settings": asdict(study.mesh_settings),
            "trace_settings": asdict(study.trace_settings),
            "optical_mode": study.trace_settings.mode,
            "indenter_optics": asdict(study.indenter_optics),
            "fem_steps": study.fem_steps,
            "internal_contact": study.internal_contact,
            "basal_interface": study.basal_interface,
            "objective": "minimum_auc",
            "protocol": (
                "one nominal, 8 Sobol initialization, "
                "18 Ax search attempts"
            ),
        }
    )


def _limiting_auc(evaluation: Mapping[str, Any]) -> float | None:
    limiting = evaluation.get("limiting_trajectory")
    if limiting is None:
        return None
    for trajectory in evaluation.get("trajectories", []):
        if (
            trajectory.get("diameter_mm") == limiting.get("diameter_mm")
            and trajectory.get("location_x_mm")
            == limiting.get("location_x_mm")
        ):
            return float(trajectory["auc"])
    return None


def _raw_minimum(evaluation: Mapping[str, Any]) -> dict[str, Any] | None:
    state = evaluation.get("minimum_raw_contact_state")
    if state is None:
        return None
    return {
        "diameter_mm": 2.0 * float(state["indenter_radius_mm"]),
        "location_x_mm": state["location_x_mm"],
        "depth_mm": state["indentation_mm"],
        "J_contact": evaluation.get("minimum_raw_contact_metric"),
    }


def _record_payload(
    record: AxTrialRecord,
    *,
    total_states: int,
) -> dict[str, Any]:
    active_parameters = {
        name: float(value) for name, value in record.parameters.items()
    }
    morphology = dict(active_parameters)
    morphology["semielliptical_pad_height"] = (
        14.0 - morphology["flat_pad_height"]
    )
    evaluation_payload = None
    if record.evaluation is not None:
        evaluation_payload = _evaluation_to_dict(
            record.evaluation,
            wall_time_seconds=record.wall_time_seconds or 0.0,
            total_states=total_states,
        )
    status = (
        record.evaluation.status
        if record.evaluation is not None
        else "invalid_design"
    )
    limiting = (
        None
        if evaluation_payload is None
        else evaluation_payload.get("limiting_trajectory")
    )
    raw_minimum = (
        None
        if evaluation_payload is None
        else _raw_minimum(evaluation_payload)
    )
    return {
        "trial_index": record.trial_index,
        "phase": record.phase,
        "parameters": active_parameters,
        "morphology": morphology,
        "status": status,
        "failure_message": record.failure_message,
        "wall_time_seconds": record.wall_time_seconds,
        "minimum_auc": (
            None
            if evaluation_payload is None
            else evaluation_payload.get("minimum_auc")
        ),
        "mean_auc": (
            None
            if evaluation_payload is None
            else evaluation_payload.get("mean_auc")
        ),
        "median_auc": (
            None
            if evaluation_payload is None
            else evaluation_payload.get("median_auc")
        ),
        "limiting_trajectory": limiting,
        "limiting_auc": (
            None
            if evaluation_payload is None
            else _limiting_auc(evaluation_payload)
        ),
        "minimum_raw_contact_state": raw_minimum,
        "evaluation": evaluation_payload,
    }


def _print_progress(record: Mapping[str, Any], completed: int) -> None:
    value = record.get("minimum_auc")
    value_text = "—" if value is None else f"{float(value):.12g}"
    wall = record.get("wall_time_seconds") or 0.0
    print(
        f"[{completed:02d}/{TOTAL_ATTEMPTS}] {record['phase']} "
        f"trial={record['trial_index']} status={record['status']} "
        f"minimum_auc={value_text} wall={wall:.2f}s",
        flush=True,
    )


def _initial_state(configuration: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "RUNNING",
        "created_at": _now(),
        "updated_at": _now(),
        "configuration": dict(configuration),
        "records": [],
        "completed_attempts": 0,
    }


def _persist_checkpoint(
    output: Path,
    state: dict[str, Any],
    client: Any,
    records: Sequence[AxTrialRecord],
) -> None:
    payloads = [
        _record_payload(record, total_states=48)
        for record in records
    ]
    state["records"] = payloads
    state["completed_attempts"] = len(payloads)
    state["updated_at"] = _now()
    atomic_write_json(output / "ax_client.json", client._to_json_snapshot())
    atomic_write_json(output / "checkpoint.json", state)
    _print_progress(payloads[-1], len(payloads))


def _successful_records(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        record
        for record in records
        if record.get("status") == "success"
        and record.get("minimum_auc") is not None
    ]


def _near_bound_names(parameters: Mapping[str, float]) -> list[str]:
    names: list[str] = []
    for name, lower, upper in PRODUCTION_SEARCH_BOUNDS:
        span = upper - lower
        value = float(parameters[name])
        if (
            value - lower <= NEAR_BOUND_FRACTION * span
            or upper - value <= NEAR_BOUND_FRACTION * span
        ):
            names.append(name)
    return names


def _write_summary(
    output: Path,
    state: Mapping[str, Any],
    *,
    total_wall_time_seconds: float,
) -> None:
    records = list(state["records"])
    successful = _successful_records(records)
    nominal = next(record for record in records if record["phase"] == "nominal")
    best = (
        max(successful, key=lambda record: float(record["minimum_auc"]))
        if successful
        else None
    )
    top = sorted(
        successful,
        key=lambda record: float(record["minimum_auc"]),
        reverse=True,
    )[:5]
    failures = [record for record in records if record["status"] != "success"]
    nominal_auc = nominal.get("minimum_auc")
    best_auc = None if best is None else best.get("minimum_auc")
    absolute_improvement = (
        None
        if best_auc is None or nominal_auc is None
        else best_auc - nominal_auc
    )
    relative_improvement = (
        None
        if absolute_improvement is None or nominal_auc in (None, 0)
        else absolute_improvement / nominal_auc
    )

    cumulative: list[float | None] = []
    current: float | None = None
    for record in records:
        value = record.get("minimum_auc")
        if value is not None and (current is None or float(value) > current):
            current = float(value)
        cumulative.append(current)

    search_records = [
        record for record in records if record["phase"] == "search"
    ]
    late_records = search_records[-5:]
    prior_late_best = max(
        (
            float(record["minimum_auc"])
            for record in search_records[:-5]
            if record.get("minimum_auc") is not None
        ),
        default=None,
    )
    late_improved = any(
        record.get("minimum_auc") is not None
        and (
            prior_late_best is None
            or float(record["minimum_auc"]) > prior_late_best
        )
        for record in late_records
    )

    limiting_counts: dict[str, int] = {}
    for record in successful:
        limiting = record.get("limiting_trajectory")
        if limiting is None:
            continue
        key = (
            f"D={limiting['diameter_mm']} mm, "
            f"x={limiting['location_x_mm']} mm"
        )
        limiting_counts[key] = limiting_counts.get(key, 0) + 1

    lines = [
        "# Production Bayesian optimization campaign summary",
        "",
        f"- Status: {state['status']}",
        f"- Total wall time: {total_wall_time_seconds:.6f} s",
        f"- Seed: {state['configuration']['settings']['seed']}",
        "- Budget: nominal 1, initialization 8, search 18, total 27 attempts",
        f"- Successful / failed: {len(successful)} / {len(failures)}",
        "- Protocol: diameters 6/10/14/20 mm, locations 0/1.5/3 mm, depths 0.5/1/1.5/2 mm, medium mesh, 48 steps, bonded+sides_separate, PLANAR_2D absorber",
        "",
        "## Objective results",
        "",
        f"- Nominal minimum_auc: {nominal_auc}",
        f"- Best minimum_auc: {best_auc}",
        f"- Absolute improvement over nominal: {absolute_improvement}",
        f"- Relative improvement over nominal: {relative_improvement}",
        f"- Best phase: {None if best is None else best['phase']}",
        f"- Best morphology: {None if best is None else best['morphology']}",
        "",
        "## Top five successful morphologies",
        "",
        "| Rank | Trial | Phase | minimum_auc | flat_pad_height | stem_width | stem_height | void_width |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, record in enumerate(top, start=1):
        parameters = record["morphology"]
        lines.append(
            f"| {rank} | {record['trial_index']} | {record['phase']} | "
            f"{record['minimum_auc']:.12g} | "
            f"{parameters['flat_pad_height']:.8g} | "
            f"{parameters['stem_width']:.8g} | "
            f"{parameters['stem_height']:.8g} | "
            f"{parameters['void_width']:.8g} |"
        )
    lines.extend(
        [
            "",
            f"Near-bound convention: a variable is near a bound when within {NEAR_BOUND_FRACTION:.0%} of its search span.",
            f"Best near-bound variables: {[] if best is None else _near_bound_names(best['parameters'])}.",
            f"Top-five near-bound variables: {sorted({name for record in top for name in _near_bound_names(record['parameters'])})}.",
            "",
            "## Trial progression",
            "",
            "| Trial | Phase | Status | minimum_auc | cumulative best |",
            "|---:|---|---|---:|---:|",
        ]
    )
    for record, best_so_far in zip(records, cumulative, strict=True):
        value = record.get("minimum_auc")
        value_text = "—" if value is None else f"{float(value):.12g}"
        best_text = "—" if best_so_far is None else f"{best_so_far:.12g}"
        lines.append(
            f"| {record['trial_index']} | {record['phase']} | "
            f"{record['status']} | {value_text} | {best_text} |"
        )
    lines.extend(
        [
            "",
            f"Final five search trials improved the cumulative best: {late_improved}.",
            f"Campaign-local plateau assessment: {not late_improved} (no global convergence claim).",
            "",
            "## Failure summary",
            "",
        ]
    )
    if failures:
        lines.extend(
            [
                "| Trial | Phase | Status | Parameters | Failure message |",
                "|---:|---|---|---|---|",
            ]
        )
        for record in failures:
            lines.append(
                f"| {record['trial_index']} | {record['phase']} | "
                f"{record['status']} | {record['parameters']} | "
                f"{record.get('failure_message') or '—'} |"
            )
    else:
        lines.append("No candidate failed.")
    lines.extend(
        [
            "",
            "## Limiting trajectory distribution",
            "",
        ]
    )
    if limiting_counts:
        for key, count in sorted(limiting_counts.items()):
            lines.append(f"- {key}: {count} successful morphologies")
    else:
        lines.append("No successful morphology was available.")
    lines.extend(
        [
            "",
            "The full per-trial evaluator results, including trajectory AUCs, raw "
            "J_contact states, mechanics diagnostics, and optical diagnostics, "
            "are preserved in checkpoint.json. The Ax client state after the "
            "last persisted trial is in ax_client.json.",
            "",
            "No optional figures were generated because no existing campaign "
            "plotting path was present; the numeric progression table is the "
            "lightweight fallback.",
        ]
    )
    (output / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_campaign(output: Path) -> AxRunResult:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to start a second campaign in {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    settings = AxSettings(
        initialization_trials=INITIALIZATION_TRIALS,
        search_trials=SEARCH_TRIALS,
        seed=CAMPAIGN_SEED,
    )
    study = create_production_study()
    configuration = _configuration(study, settings)
    state = _initial_state(configuration)
    atomic_write_json(output / "checkpoint.json", state)
    started = time.perf_counter()

    def persist(client: Any, records: tuple[AxTrialRecord, ...]) -> None:
        _persist_checkpoint(output, state, client, records)

    try:
        result = run_ax_optimization(study, settings, on_record=persist)
    except Exception as exc:
        state["status"] = "ERROR"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["total_wall_time_seconds"] = time.perf_counter() - started
        state["updated_at"] = _now()
        atomic_write_json(output / "checkpoint.json", state)
        raise

    total_wall_time_seconds = time.perf_counter() - started
    state["status"] = "COMPLETE"
    state["completed_at"] = _now()
    state["total_wall_time_seconds"] = total_wall_time_seconds
    state["updated_at"] = _now()
    atomic_write_json(output / "checkpoint.json", state)
    _write_summary(
        output,
        state,
        total_wall_time_seconds=total_wall_time_seconds,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    run_campaign(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
