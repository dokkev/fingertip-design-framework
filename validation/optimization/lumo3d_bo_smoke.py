"""Bounded real Ax -> Newton -> FULL_3D OptiX smoke for LUMO."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from ax.api.client import Client
from optics.optix.smoke import run_production_optix_smoke
from optimization.ax_adapter import (
    AxSettings,
    CONTACT_STATE_SEPARATION_OBJECTIVE_NAME,
    run_ax_optimization,
)
from optimization.evaluation_registry import EvaluationRegistry
from validation.optimization.lumo3d_evaluator import (
    LUMO3D_EVALUATION_CONTRACT_ID,
    LUMO3D_OBSERVATION_LEVEL,
    create_lumo3d_study,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _record_payload(record: Any) -> dict[str, Any]:
    evaluation = record.evaluation
    return {
        "trial_index": record.trial_index,
        "phase": record.phase,
        "status": record.status,
        "parameters": dict(record.parameters),
        "registry_key": record.registry_key,
        "wall_time_seconds": record.wall_time_seconds,
        "failure_message": record.failure_message,
        "objective_value": (
            None if evaluation is None else getattr(evaluation, "objective_value", None)
        ),
        "diagnostics": None if evaluation is None else dict(evaluation.diagnostics),
        "artifact_paths": (
            None
            if evaluation is None
            else [
                item.get("artifact")
                for item in getattr(evaluation, "optical_diagnostics", ())
                if isinstance(item, dict)
            ]
        ),
    }


def _verify_ax_snapshot(path: Path, *, expected_trial_count: int) -> dict[str, Any]:
    """Reload the persisted Ax snapshot and verify the named objective contract."""
    restored = Client.load_from_json_file(filepath=str(path))
    objective = restored._experiment.optimization_config.objective
    objective_text = str(objective)
    if CONTACT_STATE_SEPARATION_OBJECTIVE_NAME not in objective_text:
        raise RuntimeError(f"Ax snapshot objective mismatch: {objective_text}")
    trial_count = len(restored._experiment.trials)
    if trial_count != expected_trial_count:
        raise RuntimeError(
            f"Ax snapshot trial count mismatch: expected {expected_trial_count}, got {trial_count}"
        )
    return {
        "status": "PASS",
        "trial_count": trial_count,
        "objective": objective_text,
    }


def run_lumo3d_bo_smoke(output_dir: str | Path) -> dict[str, Any]:
    """Run nominal plus one real Ax/Sobol candidate and persist all evidence."""
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty smoke directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    state: dict[str, Any] = {
        "schema": "lumo3d-real-bo-smoke-v1",
        "status": "INITIALIZING",
        "observation_level": LUMO3D_OBSERVATION_LEVEL,
        "objective_name": CONTACT_STATE_SEPARATION_OBJECTIVE_NAME,
        "objective_direction": "maximize",
        "contract_id": LUMO3D_EVALUATION_CONTRACT_ID,
        "records": [],
        "created_at": _now(),
    }
    _write_json(output / "checkpoint.json", state)

    try:
        preflight = run_production_optix_smoke()
        state["optix_preflight"] = {"status": "PASS", "evidence": preflight.to_dict()}
        _write_json(output / "preflight.json", state["optix_preflight"])
        _write_json(output / "checkpoint.json", state)

        study = create_lumo3d_study(output / "artifacts")
        registry = EvaluationRegistry(output / "registry.json")
        settings = AxSettings(
            initialization_trials=1,
            search_trials=0,
            seed=20260819,
            objective_name=CONTACT_STATE_SEPARATION_OBJECTIVE_NAME,
        )

        def persist(client: Any, records: tuple[Any, ...]) -> None:
            state["records"] = [_record_payload(record) for record in records]
            state["updated_at"] = _now()
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
        )
        state["status"] = result.status
        state["records"] = [_record_payload(record) for record in result.records]
        state["ax_proposal_count"] = result.ax_proposal_count
        state["new_evaluation_count"] = result.new_evaluation_count
        state["unique_success_count"] = result.unique_success_count
        state["unique_failure_count"] = result.unique_failure_count
        state["completed_at"] = _now()
        state["total_wall_time_seconds"] = time.perf_counter() - started
        state["ax_snapshot_roundtrip"] = _verify_ax_snapshot(
            output / "ax_client.json",
            expected_trial_count=len(result.records),
        )
        _write_json(output / "checkpoint.json", state)

        if result.unique_success_count != 2 or len(result.records) != 2:
            raise RuntimeError(
                "real LUMO BO smoke requires nominal plus one successful Ax candidate"
            )
        evaluations = [record.evaluation for record in result.records]
        objective_values = [float(evaluation.objective_value) for evaluation in evaluations]
        if any(not (value == value and abs(value) < float("inf")) for value in objective_values):
            raise RuntimeError("real LUMO BO smoke produced a non-finite objective")
        if abs(objective_values[1] - objective_values[0]) <= 5.0e-4:
            raise RuntimeError(
                "real LUMO BO smoke candidate is not distinguishable from nominal "
                "at the measured repeatability scale"
            )
        summary = {
            "status": "PASS",
            "preflight_status": "PASS",
            "objective_name": result.objective_name,
            "objective_direction": "maximize",
            "phases": [record.phase for record in result.records],
            "statuses": [record.status for record in result.records],
            "objective_values": objective_values,
            "nominal_candidate_difference": objective_values[1] - objective_values[0],
            "locations": [0.25, 0.5, 0.75],
            "ax_proposal_count": result.ax_proposal_count,
            "observation_level": LUMO3D_OBSERVATION_LEVEL,
            "artifact_directory": str(output / "artifacts"),
            "registry": str(output / "registry.json"),
            "checkpoint": str(output / "checkpoint.json"),
            "ax_snapshot": str(output / "ax_client.json"),
            "ax_snapshot_roundtrip": state["ax_snapshot_roundtrip"],
            "total_wall_time_seconds": state["total_wall_time_seconds"],
        }
        _write_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        state["status"] = "ERROR"
        state["failure_category"] = "infrastructure_failure" if "preflight" not in state else "evaluation_failure"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["total_wall_time_seconds"] = time.perf_counter() - started
        _write_json(output / "checkpoint.json", state)
        raise


__all__ = ["run_lumo3d_bo_smoke"]
