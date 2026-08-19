"""Bounded LUMO 3D BO pilot and post-search validation."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from optics.optix.smoke import run_production_optix_smoke
from optimization.ax_adapter import (
    AxSettings,
    CONTACT_STATE_SEPARATION_OBJECTIVE_NAME,
    run_ax_optimization,
)
from optimization.evaluation_registry import EvaluationRegistry
from validation.mechanics3d.multi_location_sphere_contact import (
    SEARCH_MAX_LOAD_INCREMENT_MM,
    SEARCH_SPHERE_SUBDIVISIONS,
    SEARCH_VBD_ITERATIONS,
    VALIDATION_MAX_LOAD_INCREMENT_MM,
    VALIDATION_VBD_ITERATIONS,
)
from validation.optimization.lumo3d_bo_smoke import _record_payload
from validation.optimization.lumo3d_evaluator import (
    LUMO3D_EVALUATION_CONTRACT,
    LUMO3D_EVALUATION_CONTRACT_ID,
    LUMO3D_OBSERVATION_LEVEL,
    Lumo3DEvaluator,
    create_lumo3d_study,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _write_trials(output: Path, records: list[dict[str, Any]]) -> None:
    _write_json(output / "bo_trials.json", records)
    with (output / "bo_trials.csv").open("w", newline="") as stream:
        fields = (
            "trial_index",
            "phase",
            "status",
            "objective_value",
            "wall_time_seconds",
            "registry_key",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})


def _parameter_map(record: Any) -> dict[str, float]:
    return {name: float(value) for name, value in record.parameters.items()}


def _evaluation_summary(evaluation: Any) -> dict[str, Any]:
    return {
        "status": evaluation.status,
        "objective_value": evaluation.objective_value,
        "diagnostics": dict(evaluation.diagnostics),
        "pairwise_distance_matrix": evaluation.pairwise_distance_matrix,
        "mechanics_diagnostics": evaluation.mechanics_diagnostics,
        "optical_diagnostics": evaluation.optical_diagnostics,
        "failure_message": evaluation.failure_message,
    }


def run_lumo3d_bo_pilot(output_dir: str | Path) -> dict[str, Any]:
    """Run the bounded 4-Sobol/6-MBM pilot and validate its provisional best."""
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty pilot directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    configuration = {
        "schema": "lumo3d-bo-pilot-v1",
        "objective_name": CONTACT_STATE_SEPARATION_OBJECTIVE_NAME,
        "objective_direction": "maximize",
        "contract_id": LUMO3D_EVALUATION_CONTRACT_ID,
        "contract": LUMO3D_EVALUATION_CONTRACT,
        "ax": {
            "initialization_trials": 4,
            "search_trials": 6,
            "seed": 20260819,
            "max_consecutive_known_proposals": 8,
        },
        "observation_level": LUMO3D_OBSERVATION_LEVEL,
        "search_mechanics": {
            "sphere_subdivisions": SEARCH_SPHERE_SUBDIVISIONS,
            "max_load_increment_mm": SEARCH_MAX_LOAD_INCREMENT_MM,
            "vbd_iterations": SEARCH_VBD_ITERATIONS,
        },
        "validation_mechanics": {
            "sphere_subdivisions": SEARCH_SPHERE_SUBDIVISIONS,
            "max_load_increment_mm": VALIDATION_MAX_LOAD_INCREMENT_MM,
            "vbd_iterations": VALIDATION_VBD_ITERATIONS,
        },
    }
    _write_json(output / "config.json", configuration)
    (output / "README.md").write_text(
        "# LUMO 3D BO pilot\n\n"
        "Bounded pilot: nominal + 4 Sobol initializations + 6 MBM evaluations.\n"
        "Objective: maximize minimum pairwise normalized native FULL_3D field separation.\n"
        "Observation level: FULL_3D native internal transport redistribution proxy.\n"
        "Object-interface optics and camera observability are not included.\n"
    )
    state: dict[str, Any] = {
        "schema": configuration["schema"],
        "status": "INITIALIZING",
        "configuration": configuration,
        "records": [],
        "created_at": _now(),
    }
    _write_json(output / "checkpoint.json", state)
    try:
        preflight = run_production_optix_smoke()
        state["optix_preflight"] = {"status": "PASS", "evidence": preflight.to_dict()}
        _write_json(output / "preflight.json", state["optix_preflight"])
        _write_json(output / "checkpoint.json", state)

        study = create_lumo3d_study(output / "artifacts", mechanics_mode="search")
        registry = EvaluationRegistry(output / "registry.json")
        settings = AxSettings(
            initialization_trials=4,
            search_trials=6,
            seed=20260819,
            objective_name=CONTACT_STATE_SEPARATION_OBJECTIVE_NAME,
        )
        records_by_trial: dict[int, dict[str, Any]] = {}

        def persist(client: Any, records: tuple[Any, ...]) -> None:
            for record in records:
                records_by_trial[record.trial_index] = _record_payload(record)
            state["records"] = [records_by_trial[index] for index in sorted(records_by_trial)]
            state["updated_at"] = _now()
            _write_trials(output, state["records"])
            _write_json(output / "ax_client.json", client._to_json_snapshot())
            _write_json(output / "checkpoint.json", state)

        result = run_ax_optimization(
            study,
            settings,
            on_record=persist,
            evaluation_registry=registry,
            evaluation_contract_id=LUMO3D_EVALUATION_CONTRACT_ID,
            campaign_id=output.name,
            result_artifact_path=str((output / "checkpoint.json").resolve()),
            max_consecutive_known_proposals=8,
        )
        state["status"] = result.status
        state["records"] = [records_by_trial[index] for index in sorted(records_by_trial)]
        state["ax_proposal_count"] = result.ax_proposal_count
        state["new_evaluation_count"] = result.new_evaluation_count
        state["duplicate_proposal_count"] = result.duplicate_proposal_count
        state["unique_success_count"] = result.unique_success_count
        state["unique_failure_count"] = result.unique_failure_count
        state["completed_at"] = _now()
        state["total_wall_time_seconds"] = time.perf_counter() - started
        _write_trials(output, state["records"])
        _write_json(output / "checkpoint.json", state)

        nominal_record = next(record for record in result.records if record.phase == "nominal")
        successful = [record for record in result.records if record.status == "success"]
        if not successful:
            raise RuntimeError("pilot produced no successful morphology")
        best_record = result.best_record
        if best_record is None:
            raise RuntimeError("pilot produced no best successful morphology")

        validation_root = output / "validation"
        nominal_search = Lumo3DEvaluator(validation_root / "nominal_search", mechanics_mode="search").evaluate(
            study.design_space.decode(_parameter_map(nominal_record))
        )
        best_search = Lumo3DEvaluator(validation_root / "best_search", mechanics_mode="search").evaluate(
            study.design_space.decode(_parameter_map(best_record))
        )
        nominal_validation = Lumo3DEvaluator(
            validation_root / "nominal_validation",
            mechanics_mode="validation",
        ).evaluate(study.design_space.decode(_parameter_map(nominal_record)))
        best_validation = Lumo3DEvaluator(
            validation_root / "best_validation",
            mechanics_mode="validation",
        ).evaluate(study.design_space.decode(_parameter_map(best_record)))
        validation = {
            "nominal_parameters": _parameter_map(nominal_record),
            "best_parameters": _parameter_map(best_record),
            "search": {
                "nominal": _evaluation_summary(nominal_search),
                "best": _evaluation_summary(best_search),
            },
            "validation": {
                "nominal": _evaluation_summary(nominal_validation),
                "best": _evaluation_summary(best_validation),
            },
        }
        _write_json(output / "validation.json", validation)
        search_order = best_search.objective_value > nominal_search.objective_value
        validation_order = best_validation.objective_value > nominal_validation.objective_value
        if search_order != validation_order:
            ordering = "NUMERICALLY_UNRESOLVED"
        elif validation_order:
            ordering = "BEST_ABOVE_NOMINAL"
        else:
            ordering = "NO_IMPROVEMENT"
        pilot_status = {
            "BEST_ABOVE_NOMINAL": "PASS",
            "NUMERICALLY_UNRESOLVED": "NUMERICALLY_UNRESOLVED",
            "NO_IMPROVEMENT": "NO_IMPROVEMENT",
        }[ordering]
        summary = {
            "status": pilot_status,
            "pilot_status": result.status,
            "objective_name": result.objective_name,
            "objective_direction": "maximize",
            "successful_evaluations": result.unique_success_count,
            "failed_evaluations": result.unique_failure_count,
            "sobol_successes": sum(record.status == "success" and record.phase == "initialization" for record in result.records),
            "mbm_successes": sum(record.status == "success" and record.phase == "search" for record in result.records),
            "best_trial_index": best_record.trial_index,
            "best_objective": best_record.evaluation.objective_value,
            "nominal_objective": nominal_record.evaluation.objective_value,
            "validation_ordering": ordering,
            "checkpoint": str(output / "checkpoint.json"),
            "ax_snapshot": str(output / "ax_client.json"),
            "registry": str(output / "registry.json"),
            "validation": str(output / "validation.json"),
            "observation_level": LUMO3D_OBSERVATION_LEVEL,
            "total_wall_time_seconds": state["total_wall_time_seconds"],
        }
        _write_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        state["status"] = "ERROR"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["total_wall_time_seconds"] = time.perf_counter() - started
        _write_json(output / "checkpoint.json", state)
        raise


__all__ = ["run_lumo3d_bo_pilot"]
