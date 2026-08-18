"""Focused Ax failure-continuation contract tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import optimization.ax_adapter as ax_adapter
from model import FingertipParameters
from optimization import DesignEvaluation, OptimizationStudy, create_production_study


def _success(value: float) -> DesignEvaluation:
    return DesignEvaluation(
        status="success",
        score=value,
        minimum_auc=value,
        mean_auc=value,
        median_auc=value,
        minimum_raw_contact_metric=value,
        mean_raw_contact_metric=value,
        limiting_trajectory=None,
        limiting_diameter_mm=None,
        limiting_location_x_mm=None,
        minimum_raw_contact_state=None,
        minimum_raw_contact_depth_mm=None,
        trajectories=(),
        states=(),
        failure_message=None,
    )


def _failure(message: str) -> DesignEvaluation:
    return DesignEvaluation(
        status="optics_failure",
        score=None,
        minimum_auc=None,
        mean_auc=None,
        median_auc=None,
        minimum_raw_contact_metric=None,
        mean_raw_contact_metric=None,
        limiting_trajectory=None,
        limiting_diameter_mm=None,
        limiting_location_x_mm=None,
        minimum_raw_contact_state=None,
        minimum_raw_contact_depth_mm=None,
        trajectories=(),
        states=(),
        failure_message=message,
    )


class _SyntheticEvaluator:
    def __init__(self, results: list[DesignEvaluation]) -> None:
        self.results = iter(results)

    def evaluate(self, _parameters: FingertipParameters) -> DesignEvaluation:
        return next(self.results)


@dataclass
class _TrialState:
    parameters: dict[str, float]
    status: str


class _ClientDouble:
    def __init__(self, candidates: list[dict[str, float]]) -> None:
        self.candidates = iter(candidates)
        self.trials: dict[int, _TrialState] = {}
        self.completed: list[int] = []
        self.failed: list[int] = []

    def attach_trial(self, *, parameters, arm_name=None):
        del arm_name
        index = len(self.trials)
        self.trials[index] = _TrialState(dict(parameters), "RUNNING")
        return index

    def get_next_trials(self, *, max_trials):
        assert max_trials == 1
        index = len(self.trials)
        parameters = next(self.candidates)
        self.trials[index] = _TrialState(dict(parameters), "RUNNING")
        return {index: parameters}

    def complete_trial(self, *, trial_index, raw_data):
        del raw_data
        self.trials[trial_index].status = "COMPLETED"
        self.completed.append(trial_index)

    def mark_trial_failed(self, *, trial_index, failed_reason):
        del failed_reason
        self.trials[trial_index].status = "FAILED"
        self.failed.append(trial_index)


def test_failure_is_terminal_and_checkpoint_hook_advances_without_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = create_production_study()
    client = _ClientDouble(
        [
            {
                "flat_pad_height": 4.2,
                "stem_width": 7.1,
                "stem_height": 5.4,
                "void_width": 0.8,
            },
            {
                "flat_pad_height": 5.8,
                "stem_width": 8.4,
                "stem_height": 6.7,
                "void_width": 1.6,
            },
        ]
    )
    evaluator = _SyntheticEvaluator(
        [_success(0.5), _failure("failed optical state"), _success(0.7)]
    )
    checkpoints: list[tuple[ax_adapter.AxTrialRecord, ...]] = []
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda _study, _settings: client)
    monkeypatch.setattr(
        OptimizationStudy,
        "create_evaluator",
        lambda _study: evaluator,
    )

    result = ax_adapter.run_ax_optimization(
        study,
        ax_adapter.AxSettings(initialization_trials=2, search_trials=0, seed=2),
        on_record=lambda _client, records: checkpoints.append(records),
    )

    assert [record.trial_index for record in result.records] == [0, 1, 2]
    assert client.trials[1].status == "FAILED"
    assert client.trials[2].status == "COMPLETED"
    assert client.failed == [1]
    assert client.completed == [0, 2]
    assert client.trials[1].parameters != client.trials[2].parameters
    assert [len(records) for records in checkpoints] == [1, 2, 3]
    assert [record.trial_index for record in checkpoints[-1]] == [0, 1, 2]
    assert sum(record.trial_index == 1 for record in checkpoints[-1]) == 1
    assert checkpoints[1][1].failure_message == "failed optical state"
