"""Cheap real-Ax contract check for the LUMO 3D objective boundary."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from dataclasses import replace

from lumo.optimization.adapters.ax import (
    AxSettings,
    run_ax_optimization,
)
from lumo.optimization.evaluation_registry import EvaluationRegistry
from lumo.optimization.design_space import (
    PRODUCTION_NOMINAL_VOID_HEIGHT_MM,
    PRODUCTION_SEARCH_BOUNDS,
)
from lumo.optimization.objectives import ObjectiveIdentifier
from lumo.finger import Fingertip, FingertipParameters
from lumo.optimization.evaluator import create_lumo3d_trajectory_study
from lumo.simulation import LUMO3D_OBSERVATION_LEVEL


CONTACT_STATE_SEPARATION_OBJECTIVE = ObjectiveIdentifier(
    "contact_state_separation", 1
)


@dataclass(frozen=True)
class _SyntheticStudy:
    design_space: Any
    evaluator: "_SyntheticEvaluator"

    def create_evaluator(self) -> "_SyntheticEvaluator":
        return self.evaluator


class _SyntheticEvaluator:
    """Deterministic scalar-only stand-in; it never calls scientific backends."""

    def __init__(self) -> None:
        self.calls: list[dict[str, float]] = []

    def evaluate(self, parameters):
        values = {
            name: float(getattr(parameters, name))
            for name in (
                "flat_pad_height",
                "semielliptical_pad_height",
                "stem_width",
                "stem_height",
                "void_width",
                "void_height",
            )
        }
        self.calls.append(values)
        score = sum(values.values()) / len(values)
        return _SyntheticEvaluation(
            status="success",
            objective_value=score,
            diagnostics={
                "objective_name": CONTACT_STATE_SEPARATION_OBJECTIVE.serialized_name,
                "observation_level": LUMO3D_OBSERVATION_LEVEL,
            },
        )


@dataclass(frozen=True)
class _SyntheticEvaluation:
    status: str
    objective_value: float
    diagnostics: dict[str, Any]

    @property
    def score(self) -> float:
        return self.objective_value


def run_lumo3d_ax_smoke(output_dir: str | Path) -> dict[str, Any]:
    """Run nominal + Sobol + MBM using the installed Ax 1.3.1 Client."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    wiring = create_lumo3d_trajectory_study(output)
    contract_id = wiring.evaluation_contract_id
    evaluator = _SyntheticEvaluator()
    study = _SyntheticStudy(wiring.design_space, evaluator)
    registry_path = output / "registry.json"
    suffix = 0
    while registry_path.exists():
        suffix += 1
        registry_path = output / f"registry.rerun-{suffix}.json"
    registry = EvaluationRegistry(registry_path)
    result = run_ax_optimization(
        study,
        AxSettings(
            initialization_trials=1,
            search_trials=1,
            seed=20260819,
            objective=CONTACT_STATE_SEPARATION_OBJECTIVE,
        ),
        evaluation_registry=registry,
        evaluation_contract_id=contract_id,
        campaign_id="lumo3d-ax-smoke",
        result_artifact_path=str(output / "checkpoint.json"),
    )
    phases = [record.phase for record in result.records]
    statuses = [record.status for record in result.records]
    if phases != ["nominal", "initialization", "search"]:
        raise RuntimeError(f"unexpected Ax generation phases: {phases!r}")
    if statuses != ["success", "success", "success"]:
        raise RuntimeError(f"unexpected Ax smoke statuses: {statuses!r}")
    if result.objective_name != CONTACT_STATE_SEPARATION_OBJECTIVE.serialized_name:
        raise RuntimeError("Ax smoke objective name did not survive orchestration")
    if result.best_record is None or result.best_record.evaluation is None:
        raise RuntimeError("Ax smoke did not retain a successful best record")
    if len(evaluator.calls) != 3:
        raise RuntimeError("synthetic Ax smoke did not evaluate exactly three records")
    stored = registry.records_for_contract(contract_id)
    if len(stored) != 3 or any(record.objective_value is None for record in stored):
        raise RuntimeError("Ax smoke registry did not persist objective_value")
    summary = {
        "status": "PASS",
        "objective_name": result.objective_name,
        "objective_direction": "maximize",
        "phases": phases,
        "statuses": statuses,
        "ax_proposal_count": result.ax_proposal_count,
        "new_evaluation_count": result.new_evaluation_count,
        "objective_values": [
            float(record.evaluation.objective_value)
            for record in result.records
            if record.evaluation is not None
        ],
        "registry_objective_values": [float(record.objective_value) for record in stored],
        "registry_path": str(registry_path),
        "evaluator_call_count": len(evaluator.calls),
        "fe_backend_invoked": False,
        "optix_backend_invoked": False,
        "observation_level": LUMO3D_OBSERVATION_LEVEL,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def run_lumo3d_geometry_sensitivity(output_path: str | Path) -> dict[str, Any]:
    """Persist reproducible one-at-a-time morphology fingerprint evidence."""
    nominal = FingertipParameters(void_height=PRODUCTION_NOMINAL_VOID_HEIGHT_MM)
    base_parameters = {
        name: float(getattr(nominal, name))
        for spec in PRODUCTION_SEARCH_BOUNDS
        for name in (spec.name.value,)
    }
    base_fingerprint = Fingertip(nominal).solid().morphology_fingerprint
    variables: dict[str, dict[str, Any]] = {}
    for spec in PRODUCTION_SEARCH_BOUNDS:
        name, lower, upper = spec.name.value, spec.lower, spec.upper
        value = float(lower if getattr(nominal, name) != lower else upper)
        candidate = replace(nominal, **{name: value})
        fingerprint = Fingertip(candidate).solid().morphology_fingerprint
        variables[name] = {
            "candidate_parameters": {
                field: float(getattr(candidate, field))
                for field in (spec.name.value for spec in PRODUCTION_SEARCH_BOUNDS)
            },
            "perturbed_variable": name,
            "perturbed_value": value,
            "fingerprint": fingerprint,
            "fingerprint_changed": fingerprint != base_fingerprint,
        }
    summary = {
        "status": "PASS" if all(item["fingerprint_changed"] for item in variables.values()) else "FAIL",
        "producer": "validation.optimization.lumo3d_ax_smoke.run_lumo3d_geometry_sensitivity",
        "base_parameters": base_parameters,
        "base_fingerprint": base_fingerprint,
        "variables": variables,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


__all__ = ["run_lumo3d_ax_smoke", "run_lumo3d_geometry_sensitivity"]
