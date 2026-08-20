"""Ax translation tests for the current morphology-only search boundary."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import optimization.ax_adapter as ax_adapter
from model import FingertipParameters
from optimization.ax_adapter import AxSettings, create_ax_client, run_ax_optimization
from optimization.design_space import DesignSpace, DesignVariable, PRODUCTION_SEARCH_BOUNDS


def _space() -> DesignSpace:
    return DesignSpace(
        FingertipParameters(void_height=0.25),
        tuple(
            DesignVariable(spec.name, True, spec.lower, spec.upper)
            for spec in PRODUCTION_SEARCH_BOUNDS
        ),
    )


@dataclass(frozen=True)
class _Evaluation:
    status: str
    objective_value: float
    diagnostics: dict[str, object]
    failure_message: str | None = None


class _Evaluator:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def evaluate(self, parameters) -> _Evaluation:
        self.calls.append(parameters)
        value = sum(float(getattr(parameters, name)) for name in (
            "flat_pad_height",
            "semielliptical_pad_height",
            "stem_width",
            "stem_height",
            "void_width",
            "void_height",
        ))
        return _Evaluation("success", value, {})


@dataclass
class _Trial:
    parameters: dict[str, float]
    status: str = "CANDIDATE"
    raw_data: object | None = None


class _ClientDouble:
    def __init__(self, candidates: list[dict[str, float]]) -> None:
        self.candidates = iter(candidates)
        self.trials: dict[int, _Trial] = {}
        self.last_generation_node_name = "MBM"

    def attach_trial(self, parameters, arm_name=None) -> int:
        index = len(self.trials)
        self.trials[index] = _Trial(dict(parameters))
        return index

    def get_next_trials(self, max_trials: int):
        assert max_trials == 1
        index = self.attach_trial(next(self.candidates))
        return {index: self.trials[index].parameters}

    def complete_trial(self, trial_index: int, raw_data) -> None:
        self.trials[trial_index].status = "COMPLETED"
        self.trials[trial_index].raw_data = raw_data

    def mark_trial_failed(self, trial_index: int, failed_reason=None) -> None:
        self.trials[trial_index].status = "FAILED"

    def mark_trial_abandoned(self, trial_index: int) -> None:
        self.trials[trial_index].status = "ABANDONED"


def _candidate(value: float) -> dict[str, float]:
    return {
        "flat_pad_height": value,
        "semielliptical_pad_height": 9.0,
        "stem_width": 7.6,
        "stem_height": 6.0,
        "void_width": 1.0,
        "void_height": 0.25,
    }


def test_create_ax_client_translates_all_six_active_morphology_variables() -> None:
    client = create_ax_client(
        SimpleNamespace(design_space=_space()),
        AxSettings(initialization_trials=1, search_trials=1, seed=7),
    )
    assert set(client._experiment.parameters.keys()) == {
        "flat_pad_height",
        "semielliptical_pad_height",
        "stem_width",
        "stem_height",
        "void_width",
        "void_height",
    }


def test_run_ax_optimization_evaluates_morphology_without_mechanics_or_optics(
    monkeypatch,
) -> None:
    client = _ClientDouble([_candidate(5.5)])
    evaluator = _Evaluator()
    study = SimpleNamespace(design_space=_space(), create_evaluator=lambda: evaluator)
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: client)

    result = run_ax_optimization(
        study,
        AxSettings(
            initialization_trials=1,
            search_trials=1,
            seed=7,
            objective_name="contact_state_separation",
        ),
    )

    assert result.status == "COMPLETE"
    assert result.ax_proposal_count == 1
    assert result.unique_success_count == 2  # manually attached nominal + proposal
    assert len(evaluator.calls) == 2
    assert all(record.status == "success" for record in result.records)
