"""Thin Ax 1.3.1 search-backend boundary for morphology studies."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

from ax.api.client import Client
from ax.api.configs import RangeParameterConfig

from model import InvalidFingertipParameters
from optimization.design_space import DesignSpace
from optimization.evaluator import DesignEvaluation
from optimization.study import OptimizationStudy


AX_OBJECTIVE_NAME = "minimum_auc"
AxTrialPhase = Literal["nominal", "initialization", "search"]


@dataclass(frozen=True)
class AxSettings:
    """Explicit Ax run budgets and initialization seed."""

    initialization_trials: int
    search_trials: int
    seed: int

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


@dataclass(frozen=True)
class AxTrialRecord:
    """One nominal, initialization, or search trial."""

    trial_index: int
    phase: AxTrialPhase
    parameters: Mapping[str, float]
    evaluation: DesignEvaluation | None
    failure_message: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(dict(self.parameters)),
        )


@dataclass(frozen=True)
class AxRunResult:
    """Observed trial records from one Ax-backed study run."""

    records: tuple[AxTrialRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))

    @property
    def best_record(self) -> AxTrialRecord | None:
        """Return the first successful record with the largest observed objective."""
        best: AxTrialRecord | None = None
        best_value: float | None = None
        for record in self.records:
            evaluation = record.evaluation
            if (
                evaluation is None
                or evaluation.status != "success"
                or evaluation.minimum_auc is None
            ):
                continue
            value = evaluation.minimum_auc
            if best is None or best_value is None or value > best_value:
                best = record
                best_value = value
        return best


def create_ax_client(study: OptimizationStudy, settings: AxSettings) -> Client:
    """Configure one Ax client from active design variables and run settings."""
    if not isinstance(study, OptimizationStudy):
        raise TypeError("study must be an OptimizationStudy")
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
    client.configure_experiment(parameters=parameters)
    client.configure_optimization(objective=AX_OBJECTIVE_NAME)
    client.configure_generation_strategy(
        initialization_budget=settings.initialization_trials,
        initialization_random_seed=settings.seed,
        initialize_with_center=False,
        use_existing_trials_for_initialization=False,
        allow_exceeding_initialization_budget=False,
    )
    return client


def _failure_message(evaluation: DesignEvaluation) -> str:
    return evaluation.failure_message or (
        f"DesignEvaluator returned status {evaluation.status!r}"
    )


def _mark_failed(client: Client, trial_index: int, message: str) -> None:
    client.mark_trial_failed(trial_index=trial_index, failed_reason=message)


def _attempt_mark_failed(client: Client, trial_index: int, message: str) -> None:
    """Preserve an unexpected exception if Ax failure reporting also fails."""
    try:
        _mark_failed(client, trial_index, message)
    except Exception:
        pass


def _evaluate_trial(
    client: Client,
    evaluator: object,
    design_space: DesignSpace,
    trial_index: int,
    phase: AxTrialPhase,
    candidate: Mapping[str, float],
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

        value = evaluation.minimum_auc
        if value is None:
            raise ValueError(
                "successful DesignEvaluation has no minimum_auc"
            )
        client.complete_trial(
            trial_index=trial_index,
            raw_data={AX_OBJECTIVE_NAME: (value, 0.0)},
        )
        return AxTrialRecord(
            trial_index=trial_index,
            phase=phase,
            parameters=parameters,
            evaluation=evaluation,
            failure_message=None,
        )
    except Exception as exc:
        _attempt_mark_failed(client, trial_index, f"{type(exc).__name__}: {exc}")
        raise


def _next_candidate(
    client: Client,
    phase: AxTrialPhase,
    attempt: int,
) -> tuple[int, Mapping[str, float]]:
    """Request exactly one candidate through the public Ax client API."""
    generated = client.get_next_trials(max_trials=1)
    if len(generated) != 1:
        raise RuntimeError(
            f"Ax returned {len(generated)} candidates for {phase} trial {attempt}; "
            "expected exactly one"
        )
    return next(iter(generated.items()))


def run_ax_optimization(
    study: OptimizationStudy,
    settings: AxSettings,
) -> AxRunResult:
    """Evaluate the nominal trial, initialization attempts, then search attempts."""
    if not isinstance(study, OptimizationStudy):
        raise TypeError("study must be an OptimizationStudy")
    if not isinstance(settings, AxSettings):
        raise TypeError("settings must be AxSettings")

    client = create_ax_client(study, settings)
    evaluator = study.create_evaluator()
    records: list[AxTrialRecord] = []

    nominal_values = {
        variable.name: getattr(
            study.design_space.nominal_parameters,
            variable.name,
        )
        for variable in study.design_space.active_variables
    }
    nominal_trial = client.attach_trial(
        parameters=nominal_values,
        arm_name="nominal",
    )
    records.append(
        _evaluate_trial(
            client,
            evaluator,
            study.design_space,
            nominal_trial,
            "nominal",
            nominal_values,
        )
    )

    for attempt in range(1, settings.initialization_trials + 1):
        trial_index, candidate = _next_candidate(client, "initialization", attempt)
        records.append(
            _evaluate_trial(
                client,
                evaluator,
                study.design_space,
                trial_index,
                "initialization",
                candidate,
            )
        )

    for attempt in range(1, settings.search_trials + 1):
        trial_index, candidate = _next_candidate(client, "search", attempt)
        records.append(
            _evaluate_trial(
                client,
                evaluator,
                study.design_space,
                trial_index,
                "search",
                candidate,
            )
        )

    return AxRunResult(records=tuple(records))


__all__ = [
    "AX_OBJECTIVE_NAME",
    "AxRunResult",
    "AxSettings",
    "AxTrialPhase",
    "AxTrialRecord",
    "create_ax_client",
    "run_ax_optimization",
]
