"""Thin Ax 1.3.1 search-backend boundary for morphology studies."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
import json
import math
from pathlib import Path
import tempfile
import time
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping

from ax.api.client import Client
from ax.api.configs import RangeParameterConfig

from lumo.mesh.volume.mesh import VolumeMeshDependencyError
from lumo.physics import PhysicsDependencyError
from lumo.ray_tracing.optical_mechanics import Transport3DDependencyError
from lumo.optimization.design_space import (
    DesignSpace,
    DesignSpaceFeasibilityError,
)
from lumo.optimization.evaluation_registry import (
    EvaluationRegistry,
    EvaluationRegistryRecord,
    evaluation_key,
)
from lumo.optimization.objectives import (
    ObjectiveIdentifier,
    TRAJECTORY_SEPARATION_OBJECTIVE,
)

AxTrialPhase = Literal["nominal", "initialization", "search"]
AxRunStatus = Literal[
    "COMPLETE",
    "nominal_evaluation_failed",
    "optimizer_stalled_on_known_evaluations",
    "proposal_budget_exhausted",
    "evaluation_budget_exhausted",
    "feasible_generation_exhausted",
]


class AxTerminationReason(StrEnum):
    """Closed set of reasons for leaving one synchronous Ax campaign."""

    REQUESTED_BUDGET_REACHED = "requested_budget_reached"
    NOMINAL_FAILED = "nominal_failed"
    PROPOSAL_BUDGET_EXHAUSTED = "proposal_budget_exhausted"
    EVALUATION_BUDGET_EXHAUSTED = "evaluation_budget_exhausted"
    OPTIMIZER_STALLED = "optimizer_stalled"
    FEASIBLE_GENERATION_EXHAUSTED = "feasible_generation_exhausted"


_TERMINATION_REASON_BY_STATUS: Mapping[AxRunStatus, AxTerminationReason] = {
    "COMPLETE": AxTerminationReason.REQUESTED_BUDGET_REACHED,
    "nominal_evaluation_failed": AxTerminationReason.NOMINAL_FAILED,
    "optimizer_stalled_on_known_evaluations": AxTerminationReason.OPTIMIZER_STALLED,
    "proposal_budget_exhausted": AxTerminationReason.PROPOSAL_BUDGET_EXHAUSTED,
    "evaluation_budget_exhausted": (
        AxTerminationReason.EVALUATION_BUDGET_EXHAUSTED
    ),
    "feasible_generation_exhausted": (
        AxTerminationReason.FEASIBLE_GENERATION_EXHAUSTED
    ),
}


class InfrastructureFailureKind(StrEnum):
    """Typed campaign-level failures that must not become candidate failures."""

    OPTIX_RUNTIME_INITIALIZATION = "optix-runtime-initialization"
    GMSH_RUNTIME_INITIALIZATION = "gmsh-runtime-initialization"
    PHYSICS_RUNTIME_INITIALIZATION = "newton-warp-runtime-initialization"


class CampaignInfrastructureError(RuntimeError):
    """Abort a campaign without attributing an environment failure to a design."""

    def __init__(self, message: str, *, kind: InfrastructureFailureKind) -> None:
        super().__init__(message)
        if not isinstance(kind, InfrastructureFailureKind):
            raise TypeError("kind must be an InfrastructureFailureKind")
        self.kind = kind

    @property
    def signature(self) -> str:
        """Return the stable string only at reporting/framework boundaries."""

        return self.kind.value


@dataclass(frozen=True)
class AxSettings:
    """Explicit Ax run budgets and initialization seed."""

    initialization_trials: int
    search_trials: int
    seed: int
    objective: ObjectiveIdentifier = TRAJECTORY_SEPARATION_OBJECTIVE
    max_consecutive_known_proposals: int = 20

    def __post_init__(self) -> None:
        for name, value in (
            ("initialization_trials", self.initialization_trials),
            ("search_trials", self.search_trials),
            ("seed", self.seed),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer, not bool")
        if self.initialization_trials < 1:
            raise ValueError("initialization_trials must be at least 1")
        if self.search_trials < 0:
            raise ValueError("search_trials must be nonnegative")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not isinstance(self.objective, ObjectiveIdentifier):
            raise TypeError("objective must be an ObjectiveIdentifier")
        if (
            not isinstance(self.max_consecutive_known_proposals, int)
            or isinstance(self.max_consecutive_known_proposals, bool)
            or self.max_consecutive_known_proposals < 1
        ):
            raise ValueError("max_consecutive_known_proposals must be positive")

    @property
    def objective_name(self) -> str:
        """Return the Ax string derived at the framework boundary."""

        return self.objective.serialized_name


@dataclass(frozen=True)
class AxTrialRecord:
    """One nominal, initialization, or search trial."""

    trial_index: int
    phase: AxTrialPhase
    parameters: Mapping[str, float]
    evaluation: object | None
    failure_message: str | None
    wall_time_seconds: float | None = None
    registry_key: str | None = None
    duplicate_of_trial_index: int | None = None
    duplicate_of_campaign_id: str | None = None
    duplicate_of_artifact_path: str | None = None
    feasibility_rejection: bool = False
    feasibility_constraint: str | None = None
    reused_evaluation: bool = False
    reused_evaluation_status: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(dict(self.parameters)),
        )

    @property
    def status(self) -> str:
        """Return the persisted scientific or optimizer-control status."""
        if self.duplicate_of_trial_index is not None:
            return "duplicate_skipped"
        if self.feasibility_rejection:
            return "feasibility_rejected"
        if self.evaluation is None:
            return "invalid_design"
        return self.evaluation.status


@dataclass(frozen=True)
class AxPendingTrial:
    """The exact candidate whose evaluator call may be interrupted."""

    trial_index: int
    phase: AxTrialPhase
    latent_parameters: Mapping[str, float]
    physical_parameters: Mapping[str, float] | None
    registry_key: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.trial_index, bool) or self.trial_index < 0:
            raise ValueError("pending trial_index must be nonnegative")
        object.__setattr__(
            self,
            "latent_parameters",
            MappingProxyType(
                {str(name): float(value) for name, value in self.latent_parameters.items()}
            ),
        )
        if self.physical_parameters is not None:
            object.__setattr__(
                self,
                "physical_parameters",
                MappingProxyType(
                    {
                        str(name): float(value)
                        for name, value in self.physical_parameters.items()
                    }
                ),
            )


@dataclass(frozen=True)
class AxCheckpointEvent:
    """Durable checkpoint hook emitted around one evaluator boundary."""

    phase: Literal["pre_evaluation", "post_evaluation"]
    client: Client
    records: tuple[AxTrialRecord, ...]
    pending_trial: AxPendingTrial | None
    historical_success_count: int
    historical_failure_count: int
    termination_reason: str | None = None


@dataclass(frozen=True)
class _RestoredEvaluation:
    """Lightweight evaluation view used to resume Ax accounting."""

    status: str
    objective_value: float | None
    failure_message: str | None = None
    result_artifact_path: str | None = None
    failure_scenario: str | None = None
    report: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AxResumeState:
    """Restored Ax state and audit records from one completed checkpoint."""

    client: Client
    records: tuple[AxTrialRecord, ...]
    pending_trial: AxPendingTrial | None
    historical_success_count: int = 0
    historical_failure_count: int = 0


def ax_trial_record_from_payload(payload: Mapping[str, Any]) -> AxTrialRecord:
    """Restore adapter accounting without pretending to restore physics data."""
    status = str(payload.get("status", "invalid_design"))
    feasibility_rejection = bool(payload.get("feasibility_rejection", False))
    duplicate = status == "duplicate_skipped" or payload.get(
        "duplicate_of_trial_index"
    ) is not None
    evaluation: object | None
    if feasibility_rejection or duplicate:
        evaluation = None
    else:
        evaluation = _RestoredEvaluation(
            status=status,
            objective_value=(
                None
                if payload.get("objective") is None
                else float(payload["objective"])
            ),
            failure_message=payload.get("failure_message"),
            result_artifact_path=payload.get("result_artifact_path"),
            failure_scenario=payload.get("failure_scenario"),
            report=(
                dict(payload["failure_diagnostics"])
                if isinstance(payload.get("failure_diagnostics"), Mapping)
                else {}
            ),
        )
    duplicate_trial = payload.get("duplicate_of_trial_index")
    return AxTrialRecord(
        trial_index=int(payload["trial_index"]),
        phase=payload["phase"],
        parameters={
            str(name): float(value)
            for name, value in payload.get(
                "latent_parameters", payload.get("parameters", {})
            ).items()
        },
        evaluation=evaluation,
        failure_message=payload.get("failure_message"),
        wall_time_seconds=payload.get("wall_time_seconds"),
        registry_key=payload.get("registry_key"),
        duplicate_of_trial_index=(
            None if duplicate_trial is None else int(duplicate_trial)
        ),
        duplicate_of_campaign_id=payload.get("duplicate_of_campaign_id"),
        duplicate_of_artifact_path=payload.get("duplicate_of_artifact_path"),
        feasibility_rejection=feasibility_rejection,
        feasibility_constraint=payload.get("feasibility_constraint"),
        reused_evaluation=bool(payload.get("reused_evaluation", False)),
        reused_evaluation_status=payload.get("reused_evaluation_status"),
    )


@dataclass(frozen=True)
class AxRunResult:
    """Observed trial records from one Ax-backed study run."""

    records: tuple[AxTrialRecord, ...]
    status: AxRunStatus = "COMPLETE"
    consecutive_known_proposals: int = 0
    historical_success_count: int = 0
    historical_failure_count: int = 0
    objective: ObjectiveIdentifier = TRAJECTORY_SEPARATION_OBJECTIVE
    generation_attempt_count: int = 0
    feasibility_rejection_counts: Mapping[str, int] = field(default_factory=dict)
    last_feasibility_rejection: str | None = None
    termination_reason: AxTerminationReason = (
        AxTerminationReason.REQUESTED_BUDGET_REACHED
    )
    pending_trial: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        if not isinstance(self.objective, ObjectiveIdentifier):
            raise TypeError("objective must be an ObjectiveIdentifier")
        if not isinstance(self.termination_reason, AxTerminationReason):
            try:
                object.__setattr__(
                    self,
                    "termination_reason",
                    AxTerminationReason(self.termination_reason),
                )
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "termination_reason must be an AxTerminationReason"
                ) from exc
        expected_termination_reason = _TERMINATION_REASON_BY_STATUS.get(self.status)
        if expected_termination_reason is None:
            raise ValueError(f"unsupported Ax run status: {self.status!r}")
        if self.termination_reason != expected_termination_reason:
            raise ValueError(
                "Ax run status and termination reason disagree: "
                f"status={self.status!r}, "
                f"termination_reason={self.termination_reason.value!r}"
            )
        object.__setattr__(
            self,
            "feasibility_rejection_counts",
            MappingProxyType(
                {
                    str(name): int(count)
                    for name, count in self.feasibility_rejection_counts.items()
                }
            ),
        )

    @property
    def objective_name(self) -> str:
        return self.objective.serialized_name

    @property
    def nominal_successful(self) -> bool:
        return any(
            record.phase == "nominal"
            and (
                record.status == "success"
                or record.reused_evaluation_status == "success"
            )
            for record in self.records
        )

    @property
    def ax_proposal_count(self) -> int:
        """Return all Ax generation attempts, including abandoned rejects."""
        return sum(record.phase != "nominal" for record in self.records)

    @property
    def proposal_count(self) -> int:
        """Return the number of Ax-generated attempts."""
        return self.ax_proposal_count

    @property
    def feasible_proposal_count(self) -> int:
        """Return generated proposals that reached evaluation."""
        return sum(
            record.phase != "nominal"
            and not record.feasibility_rejection
            and record.duplicate_of_trial_index is None
            for record in self.records
        )

    @property
    def successful_initialization_count(self) -> int:
        return sum(
            record.phase == "initialization" and record.status == "success"
            for record in self.records
        )

    @property
    def successful_search_count(self) -> int:
        return sum(
            record.phase == "search" and record.status == "success"
            for record in self.records
        )

    @property
    def successful_generated_count(self) -> int:
        return (
            self.successful_initialization_count
            + self.successful_search_count
        )

    @property
    def failure_count_by_status(self) -> Mapping[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            if record.status in {
                "success",
                "duplicate_skipped",
                "feasibility_rejected",
            }:
                continue
            counts[record.status] = counts.get(record.status, 0) + 1
        return MappingProxyType(counts)

    @property
    def reused_evaluation_count(self) -> int:
        return sum(record.reused_evaluation for record in self.records)

    @property
    def feasibility_rejection_count(self) -> int:
        return sum(
            record.phase != "nominal" and record.feasibility_rejection
            for record in self.records
        )

    @property
    def duplicate_proposal_count(self) -> int:
        return sum(
            record.phase != "nominal" and record.status == "duplicate_skipped"
            for record in self.records
        )

    @property
    def new_evaluation_count(self) -> int:
        return sum(
            record.status not in ("duplicate_skipped", "feasibility_rejected")
            and not record.reused_evaluation
            for record in self.records
        )

    @property
    def unique_success_count(self) -> int:
        return sum(record.status == "success" for record in self.records)

    @property
    def unique_failure_count(self) -> int:
        return sum(
            record.status
            not in ("success", "duplicate_skipped", "feasibility_rejected")
            for record in self.records
        )

    @property
    def best_record(self) -> AxTrialRecord | None:
        """Return the first successful record with the largest observed objective."""
        best: AxTrialRecord | None = None
        best_value: float | None = None
        for record in self.records:
            evaluation = record.evaluation
            if evaluation is None or evaluation.status != "success":  # type: ignore[attr-defined]
                continue
            value = _evaluation_objective_value(evaluation, self.objective_name)
            if value is None:
                continue
            if best is None or best_value is None or value > best_value:
                best = record
                best_value = value
        return best


def create_ax_client(design_space: DesignSpace, settings: AxSettings) -> Client:
    """Configure one Ax client from active design variables and run settings."""
    if not isinstance(design_space, DesignSpace):
        raise TypeError("design_space must be DesignSpace")
    if not isinstance(settings, AxSettings):
        raise TypeError("settings must be AxSettings")

    parameters = [
        RangeParameterConfig(
            name=variable.name,
            bounds=(variable.lower, variable.upper),
            parameter_type="float",
        )
        for variable in design_space.search_variables
    ]
    client = Client(random_seed=settings.seed)
    client.configure_experiment(
        parameters=parameters,
        # Physical constraints are enforced by the latent parameterization.
        # Sending their physical names to Ax would be both redundant and
        # incorrect because Ax only knows the normalized latent variables.
        parameter_constraints=(),
    )
    client.configure_optimization(objective=settings.objective_name)
    client.configure_generation_strategy(
        initialization_budget=settings.initialization_trials,
        initialization_random_seed=settings.seed,
        initialize_with_center=False,
        use_existing_trials_for_initialization=False,
        allow_exceeding_initialization_budget=False,
    )
    return client


def ax_client_snapshot(client: Client, design_space: DesignSpace) -> dict[str, object]:
    """Return a public-API Ax snapshot for validation/reporting callers.

    The production checkpoint path writes directly with
    ``Client.save_to_json_file``.  This compatibility-shaped helper exists for
    lightweight validation reports that still need an in-memory JSON payload;
    it deliberately obtains that payload through the same public API.
    """
    if not isinstance(design_space, DesignSpace):
        raise TypeError("design_space must be DesignSpace")
    with tempfile.TemporaryDirectory(prefix="lumo-ax-snapshot-") as directory:
        path = Path(directory) / "ax_client.json"
        save = getattr(client, "save_to_json_file", None)
        if not callable(save):
            raise TypeError("client must expose public save_to_json_file()")
        save(str(path))
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema": "lumo-ax-client-snapshot-v1",
        "parameterization_version": design_space.parameterization_version,
        "design_space": design_space.to_dict(),
        "ax": snapshot,
    }


def _failure_message(evaluation: object) -> str:
    return evaluation.failure_message or (
        f"DesignEvaluator returned status {evaluation.status!r}"
    )


def _evaluation_objective_value(
    evaluation: object,
    objective_name: str,
) -> float | None:
    """Read the canonical maximize-oriented scalar at the Ax boundary."""
    value = getattr(evaluation, "objective_value", None)
    if value is None:
        value = getattr(evaluation, "score", None)
    if value is None:
        return None
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(
            f"evaluation objective {objective_name!r} must be finite, "
            f"received {resolved!r}"
        )
    return resolved


def validate_successful_evaluation_objective(
    evaluation: object,
    expected: ObjectiveIdentifier,
) -> float:
    """Return the scalar after validating the complete objective identity."""
    if not isinstance(expected, ObjectiveIdentifier):
        raise TypeError("expected must be an ObjectiveIdentifier")
    if getattr(evaluation, "status", None) != "success":
        raise ValueError("objective validation requires a successful evaluation")
    nested = getattr(evaluation, "objective", None)
    if nested is None:
        raise ValueError("successful evaluation must provide a nested objective result")
    nested_identifier = getattr(nested, "objective", None)
    if not isinstance(nested_identifier, ObjectiveIdentifier):
        raise ValueError(
            "successful evaluation nested objective must provide a typed identifier"
        )
    if nested_identifier != expected:
        raise ValueError(
            "successful evaluation nested objective does not match the Ax objective"
        )
    nested_value = getattr(nested, "objective_value", None)
    top_level_value = getattr(evaluation, "objective_value", None)
    if nested_value is None or top_level_value is None:
        raise ValueError(
            "successful evaluation must provide nested and top-level objective values"
        )
    nested_resolved = _evaluation_objective_value(
        nested,
        expected.serialized_name,
    )
    top_level_resolved = _evaluation_objective_value(
        evaluation,
        expected.serialized_name,
    )
    if nested_resolved is None or top_level_resolved is None:
        raise ValueError("successful evaluation objective values must be present")
    if nested_resolved != top_level_resolved:
        raise ValueError(
            "successful evaluation objective value disagrees with nested objective"
        )
    return top_level_resolved


def _validate_registry_objective(
    record: EvaluationRegistryRecord,
    expected: ObjectiveIdentifier,
) -> None:
    if record.objective != expected:
        raise ValueError(
            "registry objective does not match the evaluator objective identifier: "
            f"record={record.objective.serialized_name!r}, "
            f"expected={expected.serialized_name!r}, key={record.key!r}"
        )


def _mark_failed(client: Client, trial_index: int, message: str) -> None:
    client.mark_trial_failed(trial_index=trial_index, failed_reason=message)


def _duplicate_record(
    client: Client,
    registry: EvaluationRegistry,
    known: EvaluationRegistryRecord,
    *,
    trial_index: int,
    phase: AxTrialPhase,
    parameters: Mapping[str, float],
    campaign_id: str,
    expected_objective: ObjectiveIdentifier,
) -> AxTrialRecord:
    """Abandon a known proposal without attaching a second observation."""
    _validate_registry_objective(known, expected_objective)
    client.mark_trial_abandoned(trial_index=trial_index)
    registry.note_duplicate(
        known,
        trial_index=trial_index,
        campaign_id=campaign_id,
    )
    message = (
        "duplicate morphology skipped; original evaluation is "
        f"campaign={known.first_campaign_id!r}, "
        f"trial={known.first_trial_index}"
    )
    return AxTrialRecord(
        trial_index=trial_index,
        phase=phase,
        parameters=dict(parameters),
        evaluation=None,
        failure_message=message,
        registry_key=known.key,
        duplicate_of_trial_index=known.first_trial_index,
        duplicate_of_campaign_id=known.first_campaign_id,
        duplicate_of_artifact_path=known.result_artifact_path,
        reused_evaluation=True,
        reused_evaluation_status=known.status,
    )


def _failure_scenario(evaluation: object | None) -> str | None:
    if evaluation is None:
        return None
    candidate = getattr(evaluation, "failure_scenario", None)
    return candidate if isinstance(candidate, str) else None


def _bootstrap_historical_registry(
    client: Client,
    registry: EvaluationRegistry,
    design_space: DesignSpace,
    contract_id: str,
    objective: ObjectiveIdentifier,
) -> tuple[int, int]:
    """Seed one fresh Ax experiment with each same-contract result once."""
    success_count = 0
    failure_count = 0
    for position, record in enumerate(registry.records_for_contract(contract_id)):
        _validate_registry_objective(record, objective)
        latent = design_space.encode(
            design_space.from_physical_values(record.morphology)
        )
        trial_index = client.attach_trial(
            parameters=latent,
            arm_name=f"historical-registry-{position}",
        )
        if record.status == "success":
            value = record.objective_value
            if value is None:
                raise ValueError(
                    f"historical success has no {objective.serialized_name}: {record.key}"
                )
            client.complete_trial(
                trial_index=trial_index,
                raw_data={objective.serialized_name: (value, 0.0)},
            )
            success_count += 1
        else:
            client.mark_trial_abandoned(trial_index=trial_index)
            failure_count += 1
    return success_count, failure_count


def _feasibility_rejection_record(
    client: Client,
    trial_index: int,
    phase: AxTrialPhase,
    candidate: Mapping[str, float],
    rejection: DesignSpaceFeasibilityError,
    *,
    mark_abandoned: bool = True,
) -> AxTrialRecord:
    """Abandon a latent proposal without recording it as a morphology result."""
    if mark_abandoned:
        client.mark_trial_abandoned(trial_index=trial_index)
    return AxTrialRecord(
        trial_index=trial_index,
        phase=phase,
        parameters=dict(candidate),
        evaluation=None,
        failure_message=f"{rejection.constraint}: {rejection}",
        feasibility_rejection=True,
        feasibility_constraint=rejection.constraint,
    )


def _replayed_registry_record(
    client: Client,
    pending: AxPendingTrial,
    known: EvaluationRegistryRecord,
) -> AxTrialRecord:
    """Apply a registry outcome to the restored pending Ax trial."""
    if known.status == "success":
        if known.objective_value is None:
            raise ValueError(f"successful registry record has no objective: {known.key}")
        client.complete_trial(
            trial_index=pending.trial_index,
            raw_data={known.objective.serialized_name: (known.objective_value, 0.0)},
        )
    else:
        client.mark_trial_failed(
            trial_index=pending.trial_index,
            failed_reason=known.failure_message or known.status,
        )
    evaluation = _RestoredEvaluation(
        status=known.status,
        objective_value=known.objective_value,
        failure_message=known.failure_message,
        result_artifact_path=known.result_artifact_path,
        failure_scenario=known.failure_scenario,
    )
    return AxTrialRecord(
        trial_index=pending.trial_index,
        phase=pending.phase,
        parameters=pending.latent_parameters,
        evaluation=evaluation,
        failure_message=known.failure_message,
        registry_key=known.key,
        reused_evaluation=True,
        reused_evaluation_status=known.status,
    )


def _evaluate_trial(
    client: Client,
    evaluator: object,
    trial_index: int,
    phase: AxTrialPhase,
    candidate: Mapping[str, float],
    objective_name: str,
    expected_objective: ObjectiveIdentifier,
    decoded: object,
) -> AxTrialRecord:
    """Evaluate one already-decoded feasible morphology."""
    parameters = dict(candidate)
    try:
        evaluation = evaluator.evaluate(decoded)  # type: ignore[attr-defined]
        if evaluation.status != "success":
            message = _failure_message(evaluation)
            _mark_failed(client, trial_index, message)
            return AxTrialRecord(
                trial_index=trial_index,
                phase=phase,
                parameters=parameters,
                evaluation=evaluation,
                failure_message=message,
            )

        value = validate_successful_evaluation_objective(
            evaluation,
            expected_objective,
        )
        client.complete_trial(
            trial_index=trial_index,
            raw_data={objective_name: (value, 0.0)},
        )
        return AxTrialRecord(
            trial_index=trial_index,
            phase=phase,
            parameters=parameters,
            evaluation=evaluation,
            failure_message=None,
        )
    except CampaignInfrastructureError:
        raise
    except Exception as exc:
        if isinstance(exc, Transport3DDependencyError):
            client.mark_trial_abandoned(trial_index=trial_index)
            raise CampaignInfrastructureError(
                f"{type(exc).__name__}: {exc}",
                kind=InfrastructureFailureKind.OPTIX_RUNTIME_INITIALIZATION,
            ) from exc
        if isinstance(exc, VolumeMeshDependencyError):
            client.mark_trial_abandoned(trial_index=trial_index)
            raise CampaignInfrastructureError(
                f"{type(exc).__name__}: {exc}",
                kind=InfrastructureFailureKind.GMSH_RUNTIME_INITIALIZATION,
            ) from exc
        if isinstance(exc, PhysicsDependencyError):
            client.mark_trial_abandoned(trial_index=trial_index)
            raise CampaignInfrastructureError(
                f"{type(exc).__name__}: {exc}",
                kind=InfrastructureFailureKind.PHYSICS_RUNTIME_INITIALIZATION,
            ) from exc
        client.mark_trial_abandoned(trial_index=trial_index)
        raise


def _next_candidate(
    client: Client,
    attempt: int,
) -> tuple[int, Mapping[str, float], AxTrialPhase]:
    """Request one candidate and derive its phase from Ax's generation node."""
    generated = client.get_next_trials(max_trials=1)
    if len(generated) != 1:
        raise RuntimeError(
            f"Ax returned {len(generated)} candidates for proposal {attempt}; "
            "expected exactly one"
        )
    trial_index, parameters = next(iter(generated.items()))
    experiment = getattr(client, "_experiment", None)
    trial = None if experiment is None else experiment.trials.get(trial_index)
    generator_run = None if trial is None else trial.generator_run
    node_name = (
        None
        if generator_run is None
        else getattr(generator_run, "_generation_node_name", None)
    )
    if node_name is None:
        node_name = getattr(client, "last_generation_node_name", None)
    if node_name == "Sobol":
        phase: AxTrialPhase = "initialization"
    elif node_name == "MBM":
        phase = "search"
    else:
        raise RuntimeError(
            f"Ax proposal {attempt} has unsupported generation node {node_name!r}; "
            "expected Sobol or MBM"
        )
    return trial_index, parameters, phase


def _ax_ready_for_search(client: Client) -> bool:
    """Return whether Ax has completed Sobol and is ready for MBM."""
    generation_strategy = getattr(client, "_generation_strategy", None)
    if generation_strategy is None:
        return False
    if getattr(generation_strategy, "current_node_name", None) == "MBM":
        return True
    current_node = getattr(generation_strategy, "current_node", None)
    if current_node is None:
        return False
    should_transition, next_node = current_node.should_transition_to_next_node(
        raise_data_required_error=False
    )
    return should_transition and next_node == "MBM"


def run_ax_optimization(
    design_space: DesignSpace,
    evaluator: object,
    settings: AxSettings,
    *,
    on_record: Callable[[Client, tuple[AxTrialRecord, ...]], None] | None = None,
    evaluation_registry: EvaluationRegistry | None = None,
    evaluation_contract_id: str | None = None,
    campaign_id: str | None = None,
    result_artifact_path: str | None = None,
    max_consecutive_known_proposals: int | None = None,
    max_proposals: int | None = None,
    max_evaluations: int | None = None,
    max_feasibility_resamples: int = 100,
    on_checkpoint: Callable[[AxCheckpointEvent], None] | None = None,
    resume_state: AxResumeState | None = None,
    producer_source: Mapping[str, Any] | None = None,
) -> AxRunResult:
    """Evaluate nominal, initialization, and search attempts in order.

    ``on_record`` is an optional observation hook for validation runners that
    need durable per-trial provenance. It is called only after Ax has marked a
    trial complete, failed, or abandoned as a known duplicate. The separate
    ``on_checkpoint`` hook is emitted before and after evaluator calls so a
    campaign runner can persist Ax state and the exact pending candidate.
    """
    if not isinstance(design_space, DesignSpace):
        raise TypeError("design_space must be DesignSpace")
    if not callable(getattr(evaluator, "evaluate", None)):
        raise TypeError("evaluator must provide evaluate()")
    if not isinstance(settings, AxSettings):
        raise TypeError("settings must be AxSettings")
    evaluator_objective = getattr(evaluator, "objective_identifier", None)
    if not isinstance(evaluator_objective, ObjectiveIdentifier):
        raise TypeError(
            "evaluator must provide a typed objective_identifier"
        )
    if evaluator_objective != settings.objective:
        raise ValueError(
            "Ax objective does not match the evaluator objective identifier"
        )
    registry_objective = evaluator_objective
    if max_consecutive_known_proposals is None:
        max_consecutive_known_proposals = settings.max_consecutive_known_proposals
    if evaluation_registry is not None:
        if not evaluation_contract_id:
            raise ValueError(
                "evaluation_contract_id is required with an evaluation registry"
            )
        if not campaign_id:
            raise ValueError("campaign_id is required with an evaluation registry")
        if (
            not isinstance(max_consecutive_known_proposals, int)
            or isinstance(max_consecutive_known_proposals, bool)
            or max_consecutive_known_proposals < 1
        ):
            raise ValueError("max_consecutive_known_proposals must be positive")
    if max_proposals is not None and (
        not isinstance(max_proposals, int)
        or isinstance(max_proposals, bool)
        or max_proposals < 1
    ):
        raise ValueError("max_proposals must be a positive integer or None")
    if max_evaluations is not None and (
        not isinstance(max_evaluations, int)
        or isinstance(max_evaluations, bool)
        or max_evaluations < 1
    ):
        raise ValueError("max_evaluations must be a positive integer or None")
    if (
        not isinstance(max_feasibility_resamples, int)
        or isinstance(max_feasibility_resamples, bool)
        or max_feasibility_resamples < 0
    ):
        raise ValueError("max_feasibility_resamples must be a non-negative integer")

    if resume_state is not None and not isinstance(resume_state, AxResumeState):
        raise TypeError("resume_state must be an AxResumeState or None")
    if producer_source is not None and not isinstance(producer_source, Mapping):
        raise TypeError("producer_source must be an object or None")
    if resume_state is None:
        client = create_ax_client(design_space, settings)
        historical_success_count = 0
        historical_failure_count = 0
        if evaluation_registry is not None:
            historical_success_count, historical_failure_count = (
                _bootstrap_historical_registry(
                    client,
                    evaluation_registry,
                    design_space,
                    evaluation_contract_id,  # type: ignore[arg-type]
                    registry_objective,
                )
            )
        records: list[AxTrialRecord] = []
        pending_resume = None
    else:
        client = resume_state.client
        historical_success_count = resume_state.historical_success_count
        historical_failure_count = resume_state.historical_failure_count
        records = list(resume_state.records)
        pending_resume = resume_state.pending_trial
    last_feasibility_rejection: str | None = None

    def evaluation_count() -> int:
        """Count actual evaluator invocations, including the nominal baseline."""

        return sum(
            record.status not in ("duplicate_skipped", "feasibility_rejected")
            and not record.reused_evaluation
            for record in records
        )

    def emit_checkpoint(
        phase: Literal["pre_evaluation", "post_evaluation"],
        pending: AxPendingTrial | None,
        *,
        termination_reason: str | None = None,
    ) -> None:
        if on_checkpoint is None:
            return
        on_checkpoint(
            AxCheckpointEvent(
                phase=phase,
                client=client,
                records=tuple(records),
                pending_trial=pending,
                historical_success_count=historical_success_count,
                historical_failure_count=historical_failure_count,
                termination_reason=termination_reason,
            )
        )

    def result(
        *,
        status: AxRunStatus = "COMPLETE",
        consecutive_known_proposals: int = 0,
        finalize: bool = False,
    ) -> AxRunResult:
        termination_reason = _TERMINATION_REASON_BY_STATUS[status]
        observed = AxRunResult(
            records=tuple(records),
            status=status,
            consecutive_known_proposals=consecutive_known_proposals,
            historical_success_count=historical_success_count,
            historical_failure_count=historical_failure_count,
            objective=settings.objective,
            generation_attempt_count=sum(
                record.phase != "nominal" for record in records
            ),
            feasibility_rejection_counts={
                constraint: sum(
                    record.feasibility_rejection
                    and record.feasibility_constraint == constraint
                    for record in records
                )
                for constraint in {
                    record.feasibility_constraint
                    for record in records
                    if record.feasibility_rejection
                }
            },
            last_feasibility_rejection=last_feasibility_rejection,
            termination_reason=termination_reason,
        )
        if finalize:
            emit_checkpoint(
                "post_evaluation",
                None,
                termination_reason=termination_reason.value,
            )
        return observed

    def pending_for(
        trial_index: int,
        phase: AxTrialPhase,
        candidate: Mapping[str, float],
        physical_parameters: Mapping[str, float] | None,
    ) -> AxPendingTrial:
        registry_key = None
        if evaluation_registry is not None and physical_parameters is not None:
            registry_key = evaluation_key(
                evaluation_contract_id,  # type: ignore[arg-type]
                physical_parameters,
            )
        return AxPendingTrial(
            trial_index=trial_index,
            phase=phase,
            latent_parameters=candidate,
            physical_parameters=physical_parameters,
            registry_key=registry_key,
        )

    def evaluate_and_record(
        trial_index: int,
        phase: AxTrialPhase,
        candidate: Mapping[str, float],
        decoded: object,
        *,
        emit_pre: bool = True,
    ) -> bool:
        parameters = dict(candidate)
        physical_parameters = design_space.physical_values(decoded)  # type: ignore[arg-type]
        pending = pending_for(trial_index, phase, parameters, physical_parameters)
        if emit_pre:
            emit_checkpoint("pre_evaluation", pending)
        if evaluation_registry is not None:
            known = evaluation_registry.lookup(
                evaluation_contract_id,  # type: ignore[arg-type]
                physical_parameters,
            )
            if known is not None:
                record = _duplicate_record(
                    client,
                    evaluation_registry,
                    known,
                    trial_index=trial_index,
                    phase=phase,
                    parameters=parameters,
                    campaign_id=campaign_id,  # type: ignore[arg-type]
                    expected_objective=registry_objective,
                )
                records.append(record)
                if on_record is not None:
                    on_record(client, tuple(records))
                emit_checkpoint("post_evaluation", None)
                return False

        started = time.perf_counter()
        try:
            record = _evaluate_trial(
                client,
                evaluator,
                trial_index,
                phase,
                candidate,
                settings.objective_name,
                registry_objective,
                decoded,
            )
        except CampaignInfrastructureError:
            emit_checkpoint(
                "post_evaluation",
                None,
                termination_reason="infrastructure_failure",
            )
            raise
        except Exception:
            emit_checkpoint(
                "post_evaluation",
                None,
                termination_reason="unexpected_failure",
            )
            raise
        record = replace(
            record,
            wall_time_seconds=time.perf_counter() - started,
        )
        records.append(record)
        if evaluation_registry is None and on_record is not None:
            on_record(client, tuple(records))

        if evaluation_registry is not None:
            evaluation = record.evaluation
            registry_record = evaluation_registry.register(
                evaluation_contract_id,  # type: ignore[arg-type]
                physical_parameters,
                status="invalid_design" if evaluation is None else evaluation.status,
                first_trial_index=record.trial_index,
                first_campaign_id=campaign_id,  # type: ignore[arg-type]
                result_artifact_path=(
                    result_artifact_path
                    if evaluation is None
                    else getattr(
                        evaluation,
                        "result_artifact_path",
                        result_artifact_path,
                    )
                ),
                objective=registry_objective,
                objective_value=(
                    None
                    if evaluation is None
                    else _evaluation_objective_value(
                        evaluation,
                        settings.objective_name,
                    )
                ),
                failure_category=(
                    "invalid_design"
                    if evaluation is None
                    else None
                    if evaluation.status == "success"
                    else evaluation.status
                ),
                failure_message=record.failure_message,
                failure_scenario=_failure_scenario(evaluation),
                evaluation_wall_time_seconds=record.wall_time_seconds,
                producer_source=producer_source,
            )
            record = replace(record, registry_key=registry_record.key)
            records[-1] = record
            if on_record is not None:
                on_record(client, tuple(records))
        emit_checkpoint("post_evaluation", None)
        return True

    def reconcile_pending(pending: AxPendingTrial) -> None:
        """Finish the exact pre-evaluation trial before asking Ax for more."""
        if pending.physical_parameters is None:
            try:
                decoded = design_space.decode(pending.latent_parameters)
            except DesignSpaceFeasibilityError as exc:
                record = _feasibility_rejection_record(
                    client,
                    pending.trial_index,
                    pending.phase,
                    pending.latent_parameters,
                    exc,
                )
                records.append(record)
                if on_record is not None:
                    on_record(client, tuple(records))
                emit_checkpoint("post_evaluation", None)
                return
            physical_parameters = design_space.physical_values(decoded)
        else:
            decoded = design_space.from_physical_values(pending.physical_parameters)
            physical_parameters = design_space.physical_values(decoded)
        if (
            pending.physical_parameters is not None
            and dict(physical_parameters) != dict(pending.physical_parameters)
        ):
            raise ValueError("pending checkpoint physical morphology failed round-trip")
        known = None
        if evaluation_registry is not None:
            known = evaluation_registry.lookup(
                evaluation_contract_id,  # type: ignore[arg-type]
                physical_parameters,
            )
        if known is not None:
            _validate_registry_objective(known, registry_objective)
            record = _replayed_registry_record(client, pending, known)
            records.append(record)
            if on_record is not None:
                on_record(client, tuple(records))
            emit_checkpoint("post_evaluation", None)
            return
        evaluate_and_record(
            pending.trial_index,
            pending.phase,
            pending.latent_parameters,
            decoded,
            emit_pre=False,
        )

    if pending_resume is not None:
        reconcile_pending(pending_resume)

    if resume_state is None:
        nominal_values = design_space.encode(design_space.nominal_parameters)
        # Leave the arm name to Ax. A historical bootstrap may already contain
        # this exact morphology under its historical arm name; forcing "nominal"
        # would make Ax reject the duplicate arm before the registry can abandon it.
        nominal_trial = client.attach_trial(parameters=nominal_values)
        evaluate_and_record(
            nominal_trial,
            "nominal",
            nominal_values,
            design_space.nominal_parameters,
        )

    nominal_success = any(
        record.phase == "nominal"
        and (
            record.status == "success"
            or record.reused_evaluation_status == "success"
        )
        for record in records
    )
    if not nominal_success:
        return result(status="nominal_evaluation_failed", finalize=True)

    if (
        sum(record.feasibility_rejection for record in records)
        > max_feasibility_resamples
    ):
        return result(status="feasible_generation_exhausted", finalize=True)

    # Nominal is a manually attached baseline, not an Ax proposal. A cached
    # nominal therefore does not contribute to the generated-proposal stall
    # counter or duplicate-proposal budget.
    feasible_proposal_count = sum(
        record.phase != "nominal"
        and not record.feasibility_rejection
        and record.duplicate_of_trial_index is None
        for record in records
    )
    successful_search_evaluations = sum(
        record.phase == "search"
        and record.status == "success"
        for record in records
    )
    consecutive_known = 0
    for record in reversed(records):
        if record.phase == "nominal":
            break
        if record.status != "duplicate_skipped":
            break
        consecutive_known += 1

    # Resume may start from either PRE_EVALUATION (reconciled above) or a
    # completed POST_EVALUATION/final checkpoint.  Re-apply the target test
    # before any hard-cap test or proposal request so an already complete
    # campaign remains complete and cannot generate an extra trial.
    if (
        (
            settings.search_trials == 0
            and _ax_ready_for_search(client)
            and feasible_proposal_count > 0
        )
        or (
            settings.search_trials > 0
            and successful_search_evaluations >= settings.search_trials
        )
    ):
        return result(
            consecutive_known_proposals=consecutive_known,
            finalize=True,
        )

    # Ax owns the Sobol -> MBM transition. Failed/abandoned Sobol trials do
    # not satisfy Ax's initialization criterion, so keep requesting candidates
    # while Ax reports Sobol. Only NEW candidates generated by MBM consume the
    # requested search budget.
    while True:
        if max_evaluations is not None and evaluation_count() >= max_evaluations:
            return result(status="evaluation_budget_exhausted", finalize=True)
        generated_proposal_count = sum(
            record.phase != "nominal" for record in records
        )
        if max_proposals is not None and generated_proposal_count >= max_proposals:
            return result(status="proposal_budget_exhausted", finalize=True)
        proposal_count = generated_proposal_count + 1
        trial_index, candidate, phase = _next_candidate(
            client,
            proposal_count,
        )
        emit_checkpoint(
            "pre_evaluation",
            pending_for(trial_index, phase, candidate, None),
        )
        try:
            decoded = design_space.decode(candidate)
        except DesignSpaceFeasibilityError as exc:
            record = _feasibility_rejection_record(
                client,
                trial_index,
                phase,
                candidate,
                exc,
            )
            records.append(record)
            last_feasibility_rejection = record.failure_message
            if on_record is not None:
                on_record(client, tuple(records))
            emit_checkpoint("post_evaluation", None)
            if (
                sum(record.feasibility_rejection for record in records)
                > max_feasibility_resamples
            ):
                return result(
                    status="feasible_generation_exhausted",
                    finalize=True,
                )
            continue

        is_new = evaluate_and_record(trial_index, phase, candidate, decoded)
        if not is_new:
            consecutive_known += 1
            if (
                evaluation_registry is not None
                and consecutive_known >= max_consecutive_known_proposals
            ):
                return result(
                    status="optimizer_stalled_on_known_evaluations",
                    consecutive_known_proposals=consecutive_known,
                    finalize=True,
                )
            continue

        consecutive_known = 0
        feasible_proposal_count += 1
        if (
            settings.search_trials == 0
            and phase == "initialization"
            and _ax_ready_for_search(client)
        ):
            break
        if phase == "search":
            if records[-1].status == "success":
                successful_search_evaluations += 1
            if successful_search_evaluations >= settings.search_trials:
                break
        # A Sobol evaluation, including a failure, does not consume the MBM
        # search budget. The next Ax request observes the real node transition.

    return result(
        consecutive_known_proposals=consecutive_known,
        finalize=True,
    )


__all__ = [
    "CampaignInfrastructureError",
    "InfrastructureFailureKind",
    "AxTerminationReason",
    "AxRunStatus",
    "AxRunResult",
    "AxSettings",
    "AxCheckpointEvent",
    "AxPendingTrial",
    "AxResumeState",
    "AxTrialPhase",
    "AxTrialRecord",
    "ax_trial_record_from_payload",
    "ax_client_snapshot",
    "create_ax_client",
    "run_ax_optimization",
    "validate_successful_evaluation_objective",
]
