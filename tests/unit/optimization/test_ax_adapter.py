"""Current Ax orchestration, registry, and budget contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pytest

import optimization.ax_adapter as ax_adapter
from optimization.ax_adapter import AxSettings, run_ax_optimization
from optimization.evaluation_registry import EvaluationRegistry
from optimization.evaluator import DesignEvaluation
from optimization.study import (
    PRODUCTION_EVALUATION_CONTRACT_ID,
    OptimizationStudy,
    create_production_study,
)


NOMINAL = {
    "flat_pad_height": 5.0,
    "stem_width": 7.6,
    "stem_height": 6.0,
    "void_width": 1.0,
}
KNOWN_FAILURE = {
    "flat_pad_height": 5.1,
    "stem_width": 7.7,
    "stem_height": 6.1,
    "void_width": 1.1,
}
NEW_INITIALIZATION = {
    "flat_pad_height": 5.2,
    "stem_width": 7.8,
    "stem_height": 6.2,
    "void_width": 1.2,
}
NEW_SEARCH = {
    "flat_pad_height": 5.3,
    "stem_width": 7.9,
    "stem_height": 6.3,
    "void_width": 1.3,
}


def _evaluation(status: str, value: float | None = None) -> DesignEvaluation:
    success = status == "success"
    metric = value if success else None
    return DesignEvaluation(
        status=status,  # type: ignore[arg-type]
        score=metric,
        minimum_auc=metric,
        mean_auc=metric,
        median_auc=metric,
        minimum_raw_contact_metric=metric,
        mean_raw_contact_metric=metric,
        limiting_trajectory=None,
        limiting_diameter_mm=None,
        limiting_location_x_mm=None,
        minimum_raw_contact_state=None,
        minimum_raw_contact_depth_mm=None,
        trajectories=(),
        states=(),
        diagnostics={},
        failure_message=None if success else "synthetic deterministic failure",
    )


class _CountingEvaluator:
    def __init__(self, evaluations: list[DesignEvaluation]) -> None:
        self._evaluations = iter(evaluations)
        self.calls: list[dict[str, float]] = []

    def evaluate(self, parameters) -> DesignEvaluation:
        self.calls.append(
            {
                name: float(getattr(parameters, name))
                for name in NOMINAL
            }
        )
        return next(self._evaluations)


@dataclass
class _Trial:
    parameters: dict[str, float]
    status: str = "CANDIDATE"
    raw_data: object | None = None


class _ClientDouble:
    def __init__(self, candidates: list[Mapping[str, float]]) -> None:
        self._candidates = iter(candidates)
        self.trials: dict[int, _Trial] = {}

    def attach_trial(self, parameters, arm_name=None) -> int:
        trial_index = len(self.trials)
        self.trials[trial_index] = _Trial(dict(parameters))
        return trial_index

    def get_next_trials(self, max_trials: int):
        assert max_trials == 1
        trial_index = self.attach_trial(next(self._candidates))
        return {trial_index: self.trials[trial_index].parameters}

    def complete_trial(self, trial_index: int, raw_data) -> None:
        self.trials[trial_index].status = "COMPLETED"
        self.trials[trial_index].raw_data = raw_data

    def mark_trial_failed(self, trial_index: int, failed_reason=None) -> None:
        self.trials[trial_index].status = "FAILED"

    def mark_trial_abandoned(self, trial_index: int) -> None:
        self.trials[trial_index].status = "ABANDONED"


def _register(
    registry: EvaluationRegistry,
    parameters: Mapping[str, float],
    *,
    status: str,
    trial_index: int,
    minimum_auc: float | None,
) -> None:
    registry.register(
        PRODUCTION_EVALUATION_CONTRACT_ID,
        parameters,
        status=status,
        first_trial_index=trial_index,
        first_campaign_id="historical",
        result_artifact_path="output/historical/checkpoint.json",
        minimum_auc=minimum_auc,
        failure_category=None if status == "success" else status,
        failure_message=None if status == "success" else "historical failure",
        failure_scenario=None,
        evaluation_wall_time_seconds=1.0,
    )


def _run_with_double(
    monkeypatch,
    tmp_path,
    *,
    client: _ClientDouble,
    evaluator: _CountingEvaluator,
    initialization_trials: int = 1,
    search_trials: int = 1,
    max_known: int = 20,
):
    study = create_production_study()
    registry = EvaluationRegistry(tmp_path / "registry.json")
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: client)
    monkeypatch.setattr(
        OptimizationStudy,
        "create_evaluator",
        lambda self: evaluator,
    )
    result = run_ax_optimization(
        study,
        AxSettings(initialization_trials, search_trials, seed=17),
        evaluation_registry=registry,
        evaluation_contract_id=PRODUCTION_EVALUATION_CONTRACT_ID,
        campaign_id="current",
        result_artifact_path="output/current/checkpoint.json",
        max_consecutive_known_proposals=max_known,
    )
    return registry, result


def test_history_bootstrap_duplicate_abandon_and_unique_evaluation_budgets(
    monkeypatch,
    tmp_path,
) -> None:
    registry = EvaluationRegistry(tmp_path / "registry.json")
    _register(registry, NOMINAL, status="success", trial_index=0, minimum_auc=0.4)
    _register(
        registry,
        KNOWN_FAILURE,
        status="optics_failure",
        trial_index=1,
        minimum_auc=None,
    )
    client = _ClientDouble(
        [KNOWN_FAILURE, NEW_INITIALIZATION, NOMINAL, NEW_SEARCH]
    )
    evaluator = _CountingEvaluator(
        [_evaluation("success", 0.5), _evaluation("fea_failure")]
    )
    study = create_production_study()
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: client)
    monkeypatch.setattr(
        OptimizationStudy,
        "create_evaluator",
        lambda self: evaluator,
    )

    result = run_ax_optimization(
        study,
        AxSettings(1, 1, seed=3),
        evaluation_registry=registry,
        evaluation_contract_id=PRODUCTION_EVALUATION_CONTRACT_ID,
        campaign_id="current",
        result_artifact_path="output/current/checkpoint.json",
    )

    assert [record.status for record in result.records] == [
        "duplicate_skipped",
        "duplicate_skipped",
        "success",
        "duplicate_skipped",
        "fea_failure",
    ]
    assert evaluator.calls == [NEW_INITIALIZATION, NEW_SEARCH]
    assert result.historical_success_count == 1
    assert result.historical_failure_count == 1
    assert result.ax_proposal_count == 5
    assert result.duplicate_proposal_count == 3
    assert result.new_evaluation_count == 2
    assert result.unique_success_count == 1
    assert result.unique_failure_count == 1
    assert [trial.status for trial in client.trials.values()].count("COMPLETED") == 2
    assert [trial.status for trial in client.trials.values()].count("FAILED") == 1
    assert [trial.status for trial in client.trials.values()].count("ABANDONED") == 4
    assert registry.lookup(
        PRODUCTION_EVALUATION_CONTRACT_ID, NEW_INITIALIZATION
    ).status == "success"
    assert registry.lookup(
        PRODUCTION_EVALUATION_CONTRACT_ID, NEW_SEARCH
    ).status == "fea_failure"


def test_known_proposals_do_not_consume_budget_and_trigger_stall_guard(
    monkeypatch,
    tmp_path,
) -> None:
    registry = EvaluationRegistry(tmp_path / "registry.json")
    _register(registry, NOMINAL, status="success", trial_index=0, minimum_auc=0.4)
    _register(
        registry,
        KNOWN_FAILURE,
        status="fea_failure",
        trial_index=1,
        minimum_auc=None,
    )
    client = _ClientDouble([KNOWN_FAILURE, NOMINAL, KNOWN_FAILURE])
    evaluator = _CountingEvaluator([])
    study = create_production_study()
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: client)
    monkeypatch.setattr(
        OptimizationStudy,
        "create_evaluator",
        lambda self: evaluator,
    )

    result = run_ax_optimization(
        study,
        AxSettings(1, 0, seed=3),
        evaluation_registry=registry,
        evaluation_contract_id=PRODUCTION_EVALUATION_CONTRACT_ID,
        campaign_id="current",
        result_artifact_path="output/current/checkpoint.json",
        max_consecutive_known_proposals=3,
    )

    assert result.status == "optimizer_stalled_on_known_evaluations"
    assert result.consecutive_known_proposals == 3
    assert result.new_evaluation_count == 0
    assert result.duplicate_proposal_count == 3
    assert evaluator.calls == []
    assert all(
        client.trials[record.trial_index].status == "ABANDONED"
        for record in result.records
    )


def test_real_ax_accepts_historical_nominal_then_abandons_nominal_duplicate(
    monkeypatch,
    tmp_path,
) -> None:
    study = create_production_study()
    settings = AxSettings(1, 0, seed=23)
    registry = EvaluationRegistry(tmp_path / "registry.json")
    _register(registry, NOMINAL, status="success", trial_index=0, minimum_auc=0.4)
    real_client = ax_adapter.create_ax_client(study, settings)

    class _ForcedClient:
        def __getattr__(self, name):
            return getattr(real_client, name)

        def get_next_trials(self, max_trials: int):
            return real_client.get_next_trials(
                max_trials=max_trials,
                fixed_parameters=NEW_INITIALIZATION,
            )

    evaluator = _CountingEvaluator([_evaluation("success", 0.5)])
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: _ForcedClient())
    monkeypatch.setattr(
        OptimizationStudy,
        "create_evaluator",
        lambda self: evaluator,
    )

    result = run_ax_optimization(
        study,
        settings,
        evaluation_registry=registry,
        evaluation_contract_id=PRODUCTION_EVALUATION_CONTRACT_ID,
        campaign_id="historical-nominal-integration",
        result_artifact_path=None,
    )

    assert [record.status for record in result.records] == [
        "duplicate_skipped",
        "success",
    ]
    assert evaluator.calls == [NEW_INITIALIZATION]
    assert result.historical_success_count == 1
    nominal = result.records[0]
    assert real_client._experiment.trials[nominal.trial_index].status.name == (
        "ABANDONED"
    )


def test_real_ax_failed_point_can_be_reproposed_then_registry_abandons_it(
    monkeypatch,
    tmp_path,
) -> None:
    """Exercise the Ax 1.3.1 Client while forcing the incident A -> A -> B."""
    study = create_production_study()
    settings = AxSettings(1, 1, seed=29)
    real_client = ax_adapter.create_ax_client(study, settings)
    forced = iter((KNOWN_FAILURE, KNOWN_FAILURE, NEW_SEARCH))

    class _ForcedClient:
        def __getattr__(self, name):
            return getattr(real_client, name)

        def get_next_trials(self, max_trials: int):
            return real_client.get_next_trials(
                max_trials=max_trials,
                fixed_parameters=next(forced),
            )

    evaluator = _CountingEvaluator(
        [
            _evaluation("success", 0.4),
            _evaluation("optics_failure"),
            _evaluation("success", 0.6),
        ]
    )
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: _ForcedClient())
    monkeypatch.setattr(
        OptimizationStudy,
        "create_evaluator",
        lambda self: evaluator,
    )

    result = run_ax_optimization(
        study,
        settings,
        evaluation_registry=EvaluationRegistry(tmp_path / "registry.json"),
        evaluation_contract_id=PRODUCTION_EVALUATION_CONTRACT_ID,
        campaign_id="ax-1.3.1-integration",
        result_artifact_path=None,
    )

    assert [record.status for record in result.records] == [
        "success",
        "optics_failure",
        "duplicate_skipped",
        "success",
    ]
    assert evaluator.calls == [NOMINAL, KNOWN_FAILURE, NEW_SEARCH]
    failed, duplicate = result.records[1:3]
    assert real_client._experiment.trials[failed.trial_index].status.name == "FAILED"
    assert (
        real_client._experiment.trials[duplicate.trial_index].status.name
        == "ABANDONED"
    )
    assert result.new_evaluation_count == 3
    assert result.duplicate_proposal_count == 1
