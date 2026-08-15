"""Focused public-behavior tests for the thin Ax search boundary."""

from __future__ import annotations

from dataclasses import replace
import subprocess
import sys
from unittest.mock import Mock

import pytest

import optimization.ax_adapter as ax_adapter
from mesh import mesh_settings_for_level
from model import FingertipParameters, LED, OpticalMaterial
from optics import TraceSettings
from optimization import (
    DesignEvaluation,
    DesignSpace,
    DesignVariable,
    OPTIMIZABLE_PARAMETER_NAMES,
    OptimizationStudy,
    ScenarioGrid,
)


def _design_space(
    *,
    active: tuple[str, ...] = ("stem_width",),
    lower: dict[str, float] | None = None,
    upper: dict[str, float] | None = None,
    nominal_parameters: FingertipParameters | None = None,
) -> DesignSpace:
    nominal_parameters = nominal_parameters or FingertipParameters()
    lower = lower or {}
    upper = upper or {}
    variables = tuple(
        DesignVariable(
            name=name,
            optimize=name in active,
            lower=lower.get(
                name,
                0.0
                if name == "void_width" and name in active
                else getattr(nominal_parameters, name) - 0.5
                if name in active
                else getattr(nominal_parameters, name),
            ),
            upper=upper.get(
                name,
                1.0
                if name == "void_width" and name in active
                else getattr(nominal_parameters, name) + 0.5
                if name in active
                else getattr(nominal_parameters, name),
            ),
        )
        for name in OPTIMIZABLE_PARAMETER_NAMES
    )
    return DesignSpace(nominal_parameters=nominal_parameters, variables=variables)


def _study(
    *,
    active: tuple[str, ...] = ("stem_width",),
    lower: dict[str, float] | None = None,
    upper: dict[str, float] | None = None,
    nominal_parameters: FingertipParameters | None = None,
) -> OptimizationStudy:
    return OptimizationStudy(
        design_space=_design_space(
            active=active,
            lower=lower,
            upper=upper,
            nominal_parameters=nominal_parameters,
        ),
        scenario_grid=ScenarioGrid(
            locations_x_mm=(0.0, 1.0),
            indentations_mm=(0.5,),
            indenter_radii_mm=(2.0,),
        ),
        mesh_settings=mesh_settings_for_level("medium"),
        trace_settings=TraceSettings(
            ray_count=3,
            grid_width=16,
            grid_height=16,
            maximum_segment_count=32,
        ),
        led=LED(),
        optical=OpticalMaterial(),
    )


def _success(value: float) -> DesignEvaluation:
    return DesignEvaluation(
        status="success",
        score=value,
        minimum_separability=value,
        mean_separability=value,
        median_separability=value,
        minimum_location_separability=value,
        minimum_indentation_separability=None,
        minimum_radius_separability=None,
        minimum_reference_field_difference=value,
        limiting_pair=None,
        scenarios=(),
        pairs=(),
        failure_message=None,
    )


def _failure(status: str, message: str) -> DesignEvaluation:
    return DesignEvaluation(
        status=status,  # type: ignore[arg-type]
        score=None,
        minimum_separability=None,
        mean_separability=None,
        median_separability=None,
        minimum_location_separability=None,
        minimum_indentation_separability=None,
        minimum_radius_separability=None,
        minimum_reference_field_difference=None,
        limiting_pair=None,
        scenarios=(),
        pairs=(),
        failure_message=message,
    )


class _SyntheticEvaluator:
    def __init__(self, evaluations: list[DesignEvaluation | Exception]) -> None:
        self.evaluations = iter(evaluations)
        self.parameters: list[FingertipParameters] = []

    def evaluate(self, parameters: FingertipParameters) -> DesignEvaluation:
        self.parameters.append(parameters)
        result = next(self.evaluations)
        if isinstance(result, Exception):
            raise result
        return result


class _FakeClient:
    """Small public Client double for configuration and handoff tests."""

    def __init__(self, *args, **kwargs) -> None:
        self.configured_parameters = None
        self.objective = None
        self.strategy_options = None
        self.trials: dict[int, dict[str, float]] = {}
        self.next_candidates: list[dict[str, float]] = []
        self.completed: list[tuple[int, object]] = []
        self.failed: list[tuple[int, str]] = []

    def configure_experiment(self, *, parameters):
        self.configured_parameters = parameters

    def configure_optimization(self, *, objective):
        self.objective = objective

    def configure_generation_strategy(self, **kwargs):
        self.strategy_options = kwargs

    def attach_trial(self, *, parameters, arm_name=None):
        del arm_name
        index = len(self.trials)
        self.trials[index] = dict(parameters)
        return index

    def get_next_trials(self, *, max_trials):
        assert max_trials == 1
        index = len(self.trials)
        candidate = self.next_candidates.pop(0)
        self.trials[index] = candidate
        return {index: candidate}

    def complete_trial(self, *, trial_index, raw_data):
        self.completed.append((trial_index, raw_data))

    def mark_trial_failed(self, *, trial_index, failed_reason):
        self.failed.append((trial_index, failed_reason))


def _fake_run_client(
    monkeypatch: pytest.MonkeyPatch,
    candidates: list[dict[str, float]],
) -> _FakeClient:
    client = _FakeClient()
    client.next_candidates = list(candidates)
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda study, settings: client)
    return client


def test_ax_settings_uses_explicit_initialization_and_search_budgets() -> None:
    assert ax_adapter.AxSettings(initialization_trials=1, search_trials=0, seed=0)
    assert ax_adapter.AxSettings(initialization_trials=2, search_trials=3, seed=4)
    for kwargs in (
        {"initialization_trials": 0, "search_trials": 0, "seed": 0},
        {"initialization_trials": 1, "search_trials": -1, "seed": 0},
        {"initialization_trials": 1, "search_trials": 0, "seed": -1},
    ):
        with pytest.raises(ValueError):
            ax_adapter.AxSettings(**kwargs)
    for kwargs in (
        {"initialization_trials": True, "search_trials": 0, "seed": 0},
        {"initialization_trials": 1, "search_trials": False, "seed": 0},
        {"initialization_trials": 1, "search_trials": 0, "seed": True},
    ):
        with pytest.raises(TypeError):
            ax_adapter.AxSettings(**kwargs)


def test_create_client_maps_only_active_variables_and_configures_public_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    monkeypatch.setattr(ax_adapter, "Client", lambda **kwargs: client)
    study = _study(
        active=("void_width", "stem_width"),
        lower={"stem_width": 7.0, "void_width": 0.0},
        upper={"stem_width": 8.0, "void_width": 1.0},
    )
    settings = ax_adapter.AxSettings(initialization_trials=2, search_trials=1, seed=3)

    result = ax_adapter.create_ax_client(study, settings)

    assert result is client
    assert {parameter.name for parameter in client.configured_parameters} == {
        "stem_width",
        "void_width",
    }
    bounds = {
        parameter.name: parameter.bounds for parameter in client.configured_parameters
    }
    assert bounds == {"stem_width": (7.0, 8.0), "void_width": (0.0, 1.0)}
    assert client.objective == ax_adapter.AX_OBJECTIVE_NAME
    assert client.strategy_options == {
        "initialization_budget": 2,
        "initialization_random_seed": 3,
        "initialize_with_center": False,
        "use_existing_trials_for_initialization": False,
        "allow_exceeding_initialization_budget": False,
    }


def test_runner_uses_nominal_then_exact_initialization_and_search_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _study(
        active=("void_width", "stem_width"),
        lower={"stem_width": 7.0, "void_width": 0.0},
        upper={"stem_width": 8.0, "void_width": 1.0},
    )
    client = _fake_run_client(
        monkeypatch,
        [
            {"stem_width": 7.2, "void_width": 0.2},
            {"stem_width": 7.4, "void_width": 0.4},
            {"stem_width": 7.8, "void_width": 0.8},
        ],
    )
    evaluator = _SyntheticEvaluator(
        [_success(0.1), _success(0.2), _success(0.3), _success(0.4)]
    )
    monkeypatch.setattr(OptimizationStudy, "create_evaluator", lambda _study: evaluator)

    result = ax_adapter.run_ax_optimization(
        study,
        ax_adapter.AxSettings(initialization_trials=2, search_trials=1, seed=5),
    )

    assert [record.phase for record in result.records] == [
        "nominal",
        "initialization",
        "initialization",
        "search",
    ]
    assert evaluator.parameters[0] == study.design_space.nominal_parameters
    assert set(result.records[0].parameters) == {"stem_width", "void_width"}
    assert len(client.trials) == 4
    assert all(len(parameters) == 2 for parameters in client.trials.values())


def test_successful_zero_objective_reports_sem_zero() -> None:
    client = _FakeClient()
    client.next_candidates = [{"stem_width": 7.2}]
    study = _study()
    evaluator = _SyntheticEvaluator([_success(0.0), _success(0.0)])
    nominal_trial = client.attach_trial(parameters={"stem_width": 7.6})
    initialization = client.get_next_trials(max_trials=1)

    ax_adapter._evaluate_trial(
        client,
        evaluator,
        study.design_space,
        nominal_trial,
        "nominal",
        {"stem_width": 7.6},
    )
    ax_adapter._evaluate_trial(
        client,
        evaluator,
        study.design_space,
        next(iter(initialization)),
        "initialization",
        next(iter(initialization.values())),
    )

    assert len(client.completed) == 2
    assert all(
        raw_data == {ax_adapter.AX_OBJECTIVE_NAME: (0.0, 0.0)}
        for _, raw_data in client.completed
    )


@pytest.mark.parametrize("status", ["invalid_design", "fea_failure", "optics_failure"])
def test_expected_evaluator_failure_has_no_objective(
    status: str,
) -> None:
    client = _FakeClient()
    study = _study()
    evaluator = _SyntheticEvaluator([_failure(status, "synthetic failure")])
    trial = client.attach_trial(parameters={"stem_width": 7.6})

    record = ax_adapter._evaluate_trial(
        client,
        evaluator,
        study.design_space,
        trial,
        "initialization",
        {"stem_width": 7.6},
    )

    assert record.evaluation is not None
    assert record.failure_message == "synthetic failure"
    assert client.failed and not client.completed


def test_decode_failure_has_no_objective_and_does_not_call_evaluator() -> None:
    study = _study(
        active=("flat_pad_width", "stem_width"),
        lower={"flat_pad_width": 15.0, "stem_width": 7.6},
        upper={"flat_pad_width": 20.0, "stem_width": 9.0},
        nominal_parameters=FingertipParameters(flat_pad_width=20.0),
    )
    client = _FakeClient()
    trial = client.attach_trial(
        parameters={"flat_pad_width": 15.0, "stem_width": 9.0}
    )
    evaluator = Mock()

    record = ax_adapter._evaluate_trial(
        client,
        evaluator,
        study.design_space,
        trial,
        "initialization",
        {"flat_pad_width": 15.0, "stem_width": 9.0},
    )

    assert record.evaluation is None
    assert client.failed and not client.completed
    evaluator.evaluate.assert_not_called()


def test_unexpected_exception_is_reraised_after_failure_reporting() -> None:
    study = _study()
    client = _FakeClient()
    trial = client.attach_trial(parameters={"stem_width": 7.6})
    bug = RuntimeError("synthetic bug")
    evaluator = _SyntheticEvaluator([bug])

    with pytest.raises(RuntimeError, match="synthetic bug") as raised:
        ax_adapter._evaluate_trial(
            client,
            evaluator,
            study.design_space,
            trial,
            "initialization",
            {"stem_width": 7.6},
        )
    assert raised.value is bug
    assert client.failed and not client.completed


def test_failed_attempts_are_not_replaced_and_search_zero_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _study()
    client = _fake_run_client(monkeypatch, [{"stem_width": 7.2}, {"stem_width": 7.4}])
    evaluator = _SyntheticEvaluator(
        [_success(0.5), _failure("invalid_design", "failed init"), _success(0.7)]
    )
    monkeypatch.setattr(OptimizationStudy, "create_evaluator", lambda _study: evaluator)

    result = ax_adapter.run_ax_optimization(
        study,
        ax_adapter.AxSettings(initialization_trials=2, search_trials=0, seed=2),
    )

    assert len(result.records) == 3
    assert len(client.trials) == 3
    assert len(client.failed) == 1
    assert len(client.completed) == 2


def test_best_record_is_first_observed_success_and_none_without_success() -> None:
    nominal = ax_adapter.AxTrialRecord(0, "nominal", {"x": 0.0}, _success(0.4), None)
    failed = ax_adapter.AxTrialRecord(1, "initialization", {"x": 0.1}, _failure("fea_failure", "x"), "x")
    initialization = ax_adapter.AxTrialRecord(2, "initialization", {"x": 0.2}, _success(0.8), None)
    tie = ax_adapter.AxTrialRecord(3, "search", {"x": 0.3}, _success(0.8), None)
    assert ax_adapter.AxRunResult((nominal, failed, initialization, tie)).best_record is initialization
    assert ax_adapter.AxRunResult((failed,)).best_record is None
    assert ax_adapter.AxRunResult((replace(nominal, evaluation=_success(0.0)),)).best_record is not None


def test_real_ax_public_client_integration() -> None:
    """Confirm public Client setup and observable nominal/init/search behavior."""
    study = _study()
    evaluator = _SyntheticEvaluator(
        [_success(0.1), _success(0.2), _success(0.3), _success(0.4)]
    )
    original = OptimizationStudy.create_evaluator
    try:
        OptimizationStudy.create_evaluator = lambda _study: evaluator  # type: ignore[method-assign]
        result = ax_adapter.run_ax_optimization(
            study,
            ax_adapter.AxSettings(initialization_trials=2, search_trials=1, seed=17),
        )
    finally:
        OptimizationStudy.create_evaluator = original

    assert [record.phase for record in result.records] == [
        "nominal",
        "initialization",
        "initialization",
        "search",
    ]
    assert len(evaluator.parameters) == 4


def test_core_import_does_not_eagerly_import_ax() -> None:
    code = "import sys; import optimization; assert not any(name == 'ax' or name.startswith('ax.') for name in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)
