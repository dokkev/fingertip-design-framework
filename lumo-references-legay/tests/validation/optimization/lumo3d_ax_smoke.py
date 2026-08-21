"""Cheap real-Ax contract check for the LUMO 3D objective boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

from lumo.optimization.adapters.ax import (
    AxSettings,
    ax_client_snapshot,
    run_ax_optimization,
)
from lumo.optimization.evaluation_registry import EvaluationRegistry
from lumo.optimization.design_space import (
    DesignSpace,
    DesignVariable,
    PRODUCTION_NOMINAL_VOID_HEIGHT_MM,
    PRODUCTION_SEARCH_BOUNDS,
)
from lumo.optimization.objectives import (
    ObjectiveIdentifier,
    TRAJECTORY_SEPARATION_OBJECTIVE,
)
from lumo.finger import Fingertip, FingertipParameters
from lumo.simulation import LUMO3D_OBSERVATION_LEVEL


OBJECTIVE = TRAJECTORY_SEPARATION_OBJECTIVE


class _SyntheticEvaluator:
    """Deterministic scalar-only stand-in; it never calls scientific backends."""

    def __init__(self) -> None:
        self.calls: list[dict[str, float]] = []

    @property
    def objective_identifier(self) -> ObjectiveIdentifier:
        return OBJECTIVE

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
            objective=_SyntheticObjective(
                objective=OBJECTIVE,
                objective_value=score,
            ),
            diagnostics={
                "objective_name": OBJECTIVE.serialized_name,
                "observation_level": LUMO3D_OBSERVATION_LEVEL,
            },
        )


@dataclass(frozen=True)
class _SyntheticObjective:
    objective: ObjectiveIdentifier
    objective_value: float


@dataclass(frozen=True)
class _SyntheticEvaluation:
    status: str
    objective_value: float
    objective: _SyntheticObjective
    diagnostics: dict[str, Any]

    @property
    def score(self) -> float:
        return self.objective_value


def run_lumo3d_ax_smoke(output_dir: str | Path) -> dict[str, Any]:
    """Run nominal + Sobol + MBM using the installed Ax 1.3.1 Client."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    nominal = FingertipParameters(void_height=PRODUCTION_NOMINAL_VOID_HEIGHT_MM)
    design_space = DesignSpace(
        nominal,
        tuple(
            DesignVariable(spec.name, True, spec.lower, spec.upper)
            for spec in PRODUCTION_SEARCH_BOUNDS
        ),
    )
    contract_id = (
        "lumo3d-ax-smoke-v1:"
        f"{OBJECTIVE.serialized_name}:"
        f"{design_space.parameterization_version}"
    )
    evaluator = _SyntheticEvaluator()
    registry_path = output / "registry.json"
    suffix = 0
    while registry_path.exists():
        suffix += 1
        registry_path = output / f"registry.rerun-{suffix}.json"
    registry = EvaluationRegistry(registry_path)
    latest_client: list[Any] = []

    def persist(client: Any, _records: tuple[Any, ...]) -> None:
        latest_client[:] = [client]

    result = run_ax_optimization(
        design_space,
        evaluator,
        AxSettings(
            initialization_trials=1,
            search_trials=1,
            seed=20260819,
            objective=OBJECTIVE,
        ),
        evaluation_registry=registry,
        evaluation_contract_id=contract_id,
        campaign_id="lumo3d-ax-smoke",
        result_artifact_path=str(output / "checkpoint.json"),
        on_record=persist,
    )
    phases = [record.phase for record in result.records]
    statuses = [record.status for record in result.records]
    if phases[:2] != ["nominal", "initialization"] or phases[-1] != "search":
        raise RuntimeError(f"unexpected Ax generation phases: {phases!r}")
    if statuses[0] != "success" or statuses[1] != "success" or statuses[-1] != "success":
        raise RuntimeError(f"unexpected Ax smoke statuses: {statuses!r}")
    if result.feasible_proposal_count != 2:
        raise RuntimeError("Ax smoke did not count only feasible proposals")
    for record in result.records:
        if record.phase == "nominal" or record.feasibility_rejection:
            continue
        design_space.validate_physical_parameters(
            design_space.decode(record.parameters)
        )
    if result.objective_name != OBJECTIVE.serialized_name:
        raise RuntimeError("Ax smoke objective name did not survive orchestration")
    if result.best_record is None or result.best_record.evaluation is None:
        raise RuntimeError("Ax smoke did not retain a successful best record")
    if len(evaluator.calls) != 3:
        raise RuntimeError("synthetic Ax smoke did not evaluate exactly three records")
    stored = registry.records_for_contract(contract_id)
    if len(stored) != 3 or any(record.objective_value is None for record in stored):
        raise RuntimeError("Ax smoke registry did not persist objective_value")
    if any(record.objective != OBJECTIVE for record in stored):
        raise RuntimeError("Ax smoke registry objective identity drifted")
    summary = {
        "status": "PASS",
        "objective_name": result.objective_name,
        "objective_direction": "maximize",
        "phases": phases,
        "statuses": statuses,
        "ax_proposal_count": result.ax_proposal_count,
        "termination_reason": result.termination_reason.value,
        "feasible_proposal_count": result.feasible_proposal_count,
        "feasibility_rejection_count": result.feasibility_rejection_count,
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
        "parameterization_version": design_space.parameterization_version,
    }
    if latest_client:
        (output / "ax_client.json").write_text(
            json.dumps(
                ax_client_snapshot(latest_client[0], design_space),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
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
