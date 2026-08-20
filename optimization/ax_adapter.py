"""Thin Ax 1.3.1 search-backend boundary for morphology studies."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from types import MappingProxyType
from typing import Callable, Literal, Mapping

from ax.api.client import Client
from ax.api.configs import RangeParameterConfig

from model import InvalidFingertipParameters
from optics.transport3d import Transport3DDependencyError
from optimization.design_space import (
    DesignSpace,
    PRODUCTION_LINEAR_PARAMETER_CONSTRAINTS,
)
from optimization.evaluation_registry import (
    EvaluationRegistry,
    EvaluationRegistryRecord,
)

AX_OBJECTIVE_NAME = "minimum_auc"
CONTACT_STATE_SEPARATION_OBJECTIVE_NAME = "contact_state_separation"
MAX_CONSECUTIVE_KNOWN_PROPOSALS = 20
OPTIX_RUNTIME_FAILURE_SIGNATURE = "optix-runtime-initialization"
AxTrialPhase = Literal["nominal", "initialization", "search"]


class CampaignInfrastructureError(RuntimeError):
    """Abort a campaign without attributing an environment failure to a design."""

    def __init__(self, message: str, *, signature: str) -> None:
        super().__init__(message)
        self.signature = signature


@dataclass(frozen=True)
class AxSettings:
    """Explicit Ax run budgets and initialization seed."""

    initialization_trials: int
    search_trials: int
    seed: int
    objective_name: str = AX_OBJECTIVE_NAME

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
        if not isinstance(self.objective_name, str) or not self.objective_name:
            raise ValueError("objective_name must be a non-empty string")


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
        if self.evaluation is None:
            return "invalid_design"
        return self.evaluation.status


@dataclass(frozen=True)
class AxRunResult:
    """Observed trial records from one Ax-backed study run."""

    records: tuple[AxTrialRecord, ...]
    status: Literal[
        "COMPLETE",
        "optimizer_stalled_on_known_evaluations",
        "proposal_budget_exhausted",
    ] = "COMPLETE"
    consecutive_known_proposals: int = 0
    historical_success_count: int = 0
    historical_failure_count: int = 0
    objective_name: str = AX_OBJECTIVE_NAME

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))

    @property
    def ax_proposal_count(self) -> int:
        """Return only trials created by ``get_next_trials``."""
        return sum(record.phase != "nominal" for record in self.records)

    @property
    def duplicate_proposal_count(self) -> int:
        return sum(
            record.phase != "nominal" and record.status == "duplicate_skipped"
            for record in self.records
        )

    @property
    def new_evaluation_count(self) -> int:
        return sum(record.status != "duplicate_skipped" for record in self.records)

    @property
    def unique_success_count(self) -> int:
        return sum(record.status == "success" for record in self.records)

    @property
    def unique_failure_count(self) -> int:
        return sum(
            record.status not in ("success", "duplicate_skipped")
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


def create_ax_client(study: object, settings: AxSettings) -> Client:
    """Configure one Ax client from active design variables and run settings."""
    if not hasattr(study, "design_space"):
        raise TypeError("study must provide a design_space")
    if not isinstance(settings, AxSettings):
        raise TypeError("settings must be AxSettings")

    parameters = [
        RangeParameterConfig(
            name=variable.name,
            bounds=(variable.lower, variable.upper),
            parameter_type="float",
        )
        for variable in study.design_space.active_variables
    ]
    client = Client(random_seed=settings.seed)
    client.configure_experiment(
        parameters=parameters,
        parameter_constraints=PRODUCTION_LINEAR_PARAMETER_CONSTRAINTS,
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


def _failure_message(evaluation: object) -> str:
    return evaluation.failure_message or (
        f"DesignEvaluator returned status {evaluation.status!r}"
    )


def _evaluation_objective_value(
    evaluation: object,
    objective_name: str,
) -> float | None:
    """Read the named maximize-oriented scalar without 3D/minimum_auc aliases."""
    if objective_name == AX_OBJECTIVE_NAME:
        value = getattr(evaluation, "minimum_auc", None)
    else:
        value = getattr(evaluation, "objective_value", None)
        if value is None:
            value = getattr(evaluation, "score", None)
    return None if value is None else float(value)


def _mark_failed(client: Client, trial_index: int, message: str) -> None:
    client.mark_trial_failed(trial_index=trial_index, failed_reason=message)


def _attempt_mark_failed(client: Client, trial_index: int, message: str) -> None:
    """Preserve an unexpected exception if Ax failure reporting also fails."""
    try:
        _mark_failed(client, trial_index, message)
    except Exception:
        pass


def _duplicate_record(
    client: Client,
    registry: EvaluationRegistry,
    known: EvaluationRegistryRecord,
    *,
    trial_index: int,
    phase: AxTrialPhase,
    parameters: Mapping[str, float],
    campaign_id: str,
) -> AxTrialRecord:
    """Abandon a known proposal without attaching a second observation."""
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
    )


def _failure_scenario(evaluation: object | None) -> str | None:
    if evaluation is None:
        return None
    candidate = evaluation.diagnostics.get("failure_scenario")
    return candidate if isinstance(candidate, str) else None


def _bootstrap_historical_registry(
    client: Client,
    registry: EvaluationRegistry,
    contract_id: str,
    objective_name: str,
) -> tuple[int, int]:
    """Seed one fresh Ax experiment with each same-contract result once."""
    success_count = 0
    failure_count = 0
    for position, record in enumerate(registry.records_for_contract(contract_id)):
        trial_index = client.attach_trial(
            parameters=dict(record.morphology),
            arm_name=f"historical-registry-{position}",
        )
        if record.status == "success":
            value = (
                record.minimum_auc
                if objective_name == AX_OBJECTIVE_NAME
                else record.objective_value
            )
            if value is None:
                raise ValueError(
                    f"historical success has no {objective_name}: {record.key}"
                )
            client.complete_trial(
                trial_index=trial_index,
                raw_data={objective_name: (value, 0.0)},
            )
            success_count += 1
        else:
            client.mark_trial_abandoned(trial_index=trial_index)
            failure_count += 1
    return success_count, failure_count


def _evaluate_trial(
    client: Client,
    evaluator: object,
    design_space: DesignSpace,
    trial_index: int,
    phase: AxTrialPhase,
    candidate: Mapping[str, float],
    objective_name: str,
) -> AxTrialRecord:
    """Decode, evaluate, and report one already-created Ax trial."""
    parameters = dict(candidate)
    try:
        try:
            decoded = design_space.decode(parameters)
        except InvalidFingertipParameters as exc:
            message = f"InvalidFingertipParameters: {exc}"
            _mark_failed(client, trial_index, message)
            return AxTrialRecord(
                trial_index=trial_index,
                phase=phase,
                parameters=parameters,
                evaluation=None,
                failure_message=message,
            )

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

        value = _evaluation_objective_value(evaluation, objective_name)
        if value is None:
            raise ValueError(
                f"successful evaluation has no {objective_name}"
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
            raise CampaignInfrastructureError(
                f"{type(exc).__name__}: {exc}",
                signature=OPTIX_RUNTIME_FAILURE_SIGNATURE,
            ) from exc
        _attempt_mark_failed(client, trial_index, f"{type(exc).__name__}: {exc}")
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
    study: object,
    settings: AxSettings,
    *,
    on_record: Callable[[Client, tuple[AxTrialRecord, ...]], None] | None = None,
    evaluation_registry: EvaluationRegistry | None = None,
    evaluation_contract_id: str | None = None,
    campaign_id: str | None = None,
    result_artifact_path: str | None = None,
    max_consecutive_known_proposals: int = MAX_CONSECUTIVE_KNOWN_PROPOSALS,
    max_proposals: int | None = None,
) -> AxRunResult:
    """Evaluate nominal, initialization, and search attempts in order.

    ``on_record`` is an optional observation hook for validation runners that
    need durable per-trial provenance. It is called only after Ax has marked a
    trial complete, failed, or abandoned as a known duplicate.
    """
    if not hasattr(study, "design_space") or not hasattr(study, "create_evaluator"):
        raise TypeError("study must provide design_space and create_evaluator()")
    if not isinstance(settings, AxSettings):
        raise TypeError("settings must be AxSettings")
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

    client = create_ax_client(study, settings)
    historical_success_count = 0
    historical_failure_count = 0
    if evaluation_registry is not None:
        historical_success_count, historical_failure_count = (
            _bootstrap_historical_registry(
                client,
                evaluation_registry,
                evaluation_contract_id,  # type: ignore[arg-type]
                settings.objective_name,
            )
        )
    evaluator = study.create_evaluator()
    records: list[AxTrialRecord] = []

    def result(
        *,
        status: Literal[
            "COMPLETE",
            "optimizer_stalled_on_known_evaluations",
            "proposal_budget_exhausted",
        ] = "COMPLETE",
        consecutive_known_proposals: int = 0,
    ) -> AxRunResult:
        return AxRunResult(
            records=tuple(records),
            status=status,
            consecutive_known_proposals=consecutive_known_proposals,
            historical_success_count=historical_success_count,
            historical_failure_count=historical_failure_count,
            objective_name=settings.objective_name,
        )

    def evaluate_and_record(
        trial_index: int,
        phase: AxTrialPhase,
        candidate: Mapping[str, float],
    ) -> bool:
        parameters = dict(candidate)
        if evaluation_registry is not None:
            known = evaluation_registry.lookup(
                evaluation_contract_id,  # type: ignore[arg-type]
                parameters,
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
                )
                records.append(record)
                if on_record is not None:
                    on_record(client, tuple(records))
                return False

        started = time.perf_counter()
        record = _evaluate_trial(
            client,
            evaluator,
            study.design_space,
            trial_index,
            phase,
            candidate,
            settings.objective_name,
        )
        record = replace(
            record,
            wall_time_seconds=time.perf_counter() - started,
        )
        records.append(record)
        if on_record is not None:
            on_record(client, tuple(records))

        if evaluation_registry is not None:
            evaluation = record.evaluation
            registry_record = evaluation_registry.register(
                evaluation_contract_id,  # type: ignore[arg-type]
                parameters,
                status="invalid_design" if evaluation is None else evaluation.status,
                first_trial_index=record.trial_index,
                first_campaign_id=campaign_id,  # type: ignore[arg-type]
                result_artifact_path=result_artifact_path,
                minimum_auc=(
                    None
                    if evaluation is None or settings.objective_name != AX_OBJECTIVE_NAME
                    else getattr(evaluation, "minimum_auc", None)
                ),
                objective_value=(
                    None
                    if evaluation is None or settings.objective_name == AX_OBJECTIVE_NAME
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
            )
            record = replace(record, registry_key=registry_record.key)
            records[-1] = record
            if on_record is not None:
                on_record(client, tuple(records))
        return True

    nominal_values = {
        variable.name: getattr(
            study.design_space.nominal_parameters,
            variable.name,
        )
        for variable in study.design_space.active_variables
    }
    # Leave the arm name to Ax. A historical bootstrap may already contain
    # this exact morphology under its historical arm name; forcing "nominal"
    # would make Ax reject the duplicate arm before the registry can abandon it.
    nominal_trial = client.attach_trial(parameters=nominal_values)
    nominal_is_new = evaluate_and_record(
        nominal_trial,
        "nominal",
        nominal_values,
    )

    # Nominal is a manually attached baseline, not an Ax proposal. A cached
    # nominal therefore does not contribute to the generated-proposal stall
    # counter or duplicate-proposal budget.
    del nominal_is_new
    consecutive_known = 0
    proposal_count = 0

    # Ax owns the Sobol -> MBM transition. Failed/abandoned Sobol trials do
    # not satisfy Ax's initialization criterion, so keep requesting candidates
    # while Ax reports Sobol. Only NEW candidates generated by MBM consume the
    # requested search budget.
    search_evaluations = 0
    while True:
        if max_proposals is not None and proposal_count >= max_proposals:
            return result(status="proposal_budget_exhausted")
        proposal_count += 1
        trial_index, candidate, phase = _next_candidate(
            client,
            proposal_count,
        )
        is_new = evaluate_and_record(trial_index, phase, candidate)
        if not is_new:
            consecutive_known += 1
            if (
                evaluation_registry is not None
                and consecutive_known >= max_consecutive_known_proposals
            ):
                return result(
                    status="optimizer_stalled_on_known_evaluations",
                    consecutive_known_proposals=consecutive_known,
                )
            continue

        consecutive_known = 0
        if (
            settings.search_trials == 0
            and phase == "initialization"
            and _ax_ready_for_search(client)
        ):
            break
        if phase == "search":
            search_evaluations += 1
            if search_evaluations >= settings.search_trials:
                break
        # A Sobol evaluation, including a failure, does not consume the MBM
        # search budget. The next Ax request observes the real node transition.

    return result(
        consecutive_known_proposals=consecutive_known,
    )


__all__ = [
    "AX_OBJECTIVE_NAME",
    "CONTACT_STATE_SEPARATION_OBJECTIVE_NAME",
    "CampaignInfrastructureError",
    "MAX_CONSECUTIVE_KNOWN_PROPOSALS",
    "OPTIX_RUNTIME_FAILURE_SIGNATURE",
    "AxRunResult",
    "AxSettings",
    "AxTrialPhase",
    "AxTrialRecord",
    "create_ax_client",
    "run_ax_optimization",
]
