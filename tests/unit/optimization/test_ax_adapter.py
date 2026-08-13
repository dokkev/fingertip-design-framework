"""Focused Ax 1.3.1 adapter tests with synthetic evaluations only."""

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
) -> DesignSpace:
    baseline = FingertipParameters()
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
                else getattr(baseline, name) - 0.5
                if name in active
                else getattr(baseline, name),
            ),
            upper=upper.get(
                name,
                1.0
                if name == "void_width" and name in active
                else getattr(baseline, name) + 0.5
                if name in active
                else getattr(baseline, name),
            ),
        )
        for name in OPTIMIZABLE_PARAMETER_NAMES
    )
    return DesignSpace(baseline=baseline, variables=variables)


def _study(
    *,
    active: tuple[str, ...] = ("stem_width",),
    lower: dict[str, float] | None = None,
    upper: dict[str, float] | None = None,
) -> OptimizationStudy:
    return OptimizationStudy(
        design_space=_design_space(active=active, lower=lower, upper=upper),
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
        self.parameters = []

    def evaluate(self, parameters: FingertipParameters) -> DesignEvaluation:
        self.parameters.append(parameters)
        result = next(self.evaluations)
        if isinstance(result, Exception):
            raise result
        return result


def _client_for_run(
    monkeypatch: pytest.MonkeyPatch,
    study: OptimizationStudy,
    settings: ax_adapter.AxSettings,
) -> tuple[object, object]:
    client, strategy = ax_adapter._client_and_strategy(study, settings)
    client.complete_trial = Mock(wraps=client.complete_trial)  # type: ignore[method-assign]
    client.mark_trial_failed = Mock(wraps=client.mark_trial_failed)  # type: ignore[method-assign]
    monkeypatch.setattr(
        ax_adapter,
        "_client_and_strategy",
        lambda _study, _settings: (client, strategy),
    )
    return client, strategy


def test_ax_settings_validation() -> None:
    assert ax_adapter.AxSettings(sobol_trials=1, bo_trials=0, seed=0)
    assert ax_adapter.AxSettings(sobol_trials=2, bo_trials=3, seed=4)
    for kwargs in (
        {"sobol_trials": 0, "bo_trials": 0, "seed": 0},
        {"sobol_trials": 1, "bo_trials": -1, "seed": 0},
        {"sobol_trials": 1, "bo_trials": 0, "seed": -1},
    ):
        with pytest.raises(ValueError):
            ax_adapter.AxSettings(**kwargs)
    for kwargs in (
        {"sobol_trials": True, "bo_trials": 0, "seed": 0},
        {"sobol_trials": 1, "bo_trials": False, "seed": 0},
        {"sobol_trials": 1, "bo_trials": 0, "seed": True},
    ):
        with pytest.raises(TypeError):
            ax_adapter.AxSettings(**kwargs)


def test_search_space_contains_only_active_variables_and_exact_bounds() -> None:
    study = _study(
        active=("void_width", "stem_width"),
        lower={"stem_width": 7.0, "void_width": 0.0},
        upper={"stem_width": 8.0, "void_width": 1.0},
    )
    client = ax_adapter.create_ax_client(
        study,
        ax_adapter.AxSettings(sobol_trials=1, bo_trials=0, seed=3),
    )
    parameters = client._experiment.search_space.parameters  # read-only Ax surface
    assert set(parameters) == {"stem_width", "void_width"}
    assert parameters["stem_width"].lower == 7.0
    assert parameters["stem_width"].upper == 8.0
    assert parameters["void_width"].lower == 0.0
    assert parameters["void_width"].upper == 1.0
    assert "flat_pad_width" not in parameters
    objective = client._experiment.optimization_config.objective
    assert objective.metric_names == [ax_adapter.AX_OBJECTIVE_NAME]
    assert not client._experiment.optimization_config.objective.minimize


def test_generation_strategy_is_explicit_sobol_then_botorch() -> None:
    client = ax_adapter.create_ax_client(
        _study(),
        ax_adapter.AxSettings(sobol_trials=2, bo_trials=1, seed=11),
    )
    strategy = client._generation_strategy
    assert [node.name for node in strategy._nodes] == ["Sobol", "BoTorch"]
    assert strategy._nodes[0].generator_specs[0].generator_key == "Sobol"
    assert strategy._nodes[0].generator_specs[0].generator_kwargs["seed"] == 11
    bo_spec = strategy._nodes[1].generator_specs[0]
    assert bo_spec.generator_key == "BoTorch"
    assert bo_spec.generator_kwargs["botorch_acqf_class"] is (
        ax_adapter.qLogExpectedImprovement
    )
    surrogate = bo_spec.generator_kwargs["surrogate_spec"]
    model_config = surrogate.model_configs[0]
    assert model_config.botorch_model_class is ax_adapter.SingleTaskGP
    assert model_config.covar_module_class is ax_adapter.MaternKernel
    assert model_config.covar_module_options == {"nu": 2.5}
    assert all(node.name != "CenterOfSearchSpace" for node in strategy._nodes)


def test_runner_attaches_baseline_first_and_excludes_it_from_sobol_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _study(
        active=("void_width", "stem_width"),
        lower={"stem_width": 7.0, "void_width": 0.0},
        upper={"stem_width": 8.0, "void_width": 1.0},
    )
    settings = ax_adapter.AxSettings(sobol_trials=2, bo_trials=0, seed=5)
    evaluator = _SyntheticEvaluator([_success(0.1), _success(0.2), _success(0.3)])
    monkeypatch.setattr(OptimizationStudy, "create_evaluator", lambda _study: evaluator)
    client, strategy = _client_for_run(monkeypatch, study, settings)

    result = ax_adapter.run_ax_optimization(study, settings)

    assert [record.phase for record in result.records] == [
        "baseline",
        "sobol",
        "sobol",
    ]
    assert len(evaluator.parameters) == 3
    assert evaluator.parameters[0] == study.design_space.baseline
    assert set(result.records[0].parameters) == {"stem_width", "void_width"}
    assert len(client._experiment.trials) == 3
    assert strategy.current_node_name == "Sobol"


def test_successful_zero_objective_is_completed_with_zero_sem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _study()
    settings = ax_adapter.AxSettings(sobol_trials=1, bo_trials=0, seed=6)
    evaluator = _SyntheticEvaluator([_success(0.0), _success(0.0)])
    monkeypatch.setattr(OptimizationStudy, "create_evaluator", lambda _study: evaluator)
    client, _ = _client_for_run(monkeypatch, study, settings)

    result = ax_adapter.run_ax_optimization(study, settings)

    assert all(record.evaluation is not None for record in result.records)
    assert all(record.evaluation.status == "success" for record in result.records)
    assert client.complete_trial.call_count == 2
    for call in client.complete_trial.call_args_list:
        assert call.kwargs["raw_data"] == {
            ax_adapter.AX_OBJECTIVE_NAME: (0.0, 0.0)
        }
    assert result.best_record is result.records[0]


@pytest.mark.parametrize("status", ["invalid_design", "fea_failure", "optics_failure"])
def test_evaluator_failures_are_failed_without_objective_data(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    study = _study()
    settings = ax_adapter.AxSettings(sobol_trials=1, bo_trials=0, seed=7)
    evaluation = _failure(status, f"{status} synthetic failure")
    evaluator = _SyntheticEvaluator([_success(0.5), evaluation])
    monkeypatch.setattr(OptimizationStudy, "create_evaluator", lambda _study: evaluator)
    client, _ = _client_for_run(monkeypatch, study, settings)

    result = ax_adapter.run_ax_optimization(study, settings)

    failed = result.records[-1]
    assert failed.evaluation is evaluation
    assert failed.failure_message == evaluation.failure_message
    assert client.mark_trial_failed.call_count == 1
    data = client._experiment.lookup_data().df
    assert set(data.trial_index) == {result.records[0].trial_index}


def test_decode_physical_failure_marks_trial_failed_without_evaluator_call() -> None:
    study = _study(
        active=("flat_pad_width", "stem_width"),
        lower={"flat_pad_width": 15.0, "stem_width": 7.6},
        upper={"flat_pad_width": 20.0, "stem_width": 9.0},
    )
    client = ax_adapter.create_ax_client(
        study,
        ax_adapter.AxSettings(sobol_trials=1, bo_trials=0, seed=8),
    )
    trial_index = client.attach_trial(
        {"flat_pad_width": 15.0, "stem_width": 9.0}
    )
    evaluator = Mock()

    record = ax_adapter._evaluate_trial(
        client,
        evaluator,
        study.design_space,
        trial_index,
        "sobol",
        {"flat_pad_width": 15.0, "stem_width": 9.0},
    )

    assert record.evaluation is None
    assert "InvalidFingertipParameters" in (record.failure_message or "")
    evaluator.evaluate.assert_not_called()
    assert client._experiment.trials[trial_index].status.name == "FAILED"
    assert client._experiment.lookup_data().df.empty


def test_unexpected_evaluator_exception_is_failed_then_reraised() -> None:
    study = _study()
    client = ax_adapter.create_ax_client(
        study,
        ax_adapter.AxSettings(sobol_trials=1, bo_trials=0, seed=9),
    )
    trial_index = client.attach_trial({"stem_width": 7.6})
    bug = RuntimeError("synthetic bug")
    evaluator = _SyntheticEvaluator([bug])

    with pytest.raises(RuntimeError, match="synthetic bug") as raised:
        ax_adapter._evaluate_trial(
            client,
            evaluator,
            study.design_space,
            trial_index,
            "sobol",
            {"stem_width": 7.6},
        )
    assert raised.value is bug
    assert client._experiment.trials[trial_index].status.name == "FAILED"


def test_unexpected_decode_exception_is_failed_then_reraised() -> None:
    study = _study()
    client = ax_adapter.create_ax_client(
        study,
        ax_adapter.AxSettings(sobol_trials=1, bo_trials=0, seed=9),
    )
    trial_index = client.attach_trial({"stem_width": 7.6})
    bug = RuntimeError("unexpected decode bug")
    design_space = Mock(spec=DesignSpace)
    design_space.decode.side_effect = bug

    with pytest.raises(RuntimeError, match="unexpected decode bug") as raised:
        ax_adapter._evaluate_trial(
            client,
            Mock(),
            design_space,
            trial_index,
            "sobol",
            {"stem_width": 7.6},
        )
    assert raised.value is bug
    assert client._experiment.trials[trial_index].status.name == "FAILED"


def test_failed_sobol_proposal_consumes_budget_without_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _study()
    settings = ax_adapter.AxSettings(sobol_trials=2, bo_trials=1, seed=10)
    evaluator = _SyntheticEvaluator(
        [
            _success(0.5),
            _failure("invalid_design", "one failed Sobol proposal"),
            _success(0.7),
            _success(0.8),
        ]
    )
    monkeypatch.setattr(OptimizationStudy, "create_evaluator", lambda _study: evaluator)
    client, strategy = _client_for_run(monkeypatch, study, settings)

    result = ax_adapter.run_ax_optimization(study, settings)

    assert len(result.records) == 4
    assert len(evaluator.parameters) == 4
    assert [record.phase for record in result.records] == [
        "baseline",
        "sobol",
        "sobol",
        "bo",
    ]
    assert strategy.current_node_name == "BoTorch"


def test_bo_phase_is_generated_after_exact_sobol_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _study()
    settings = ax_adapter.AxSettings(sobol_trials=2, bo_trials=1, seed=12)
    evaluator = _SyntheticEvaluator([_success(0.1), _success(0.2), _success(0.3), _success(0.4)])
    monkeypatch.setattr(OptimizationStudy, "create_evaluator", lambda _study: evaluator)
    client, strategy = _client_for_run(monkeypatch, study, settings)

    result = ax_adapter.run_ax_optimization(study, settings)

    assert [record.phase for record in result.records] == [
        "baseline",
        "sobol",
        "sobol",
        "bo",
    ]
    assert strategy.current_node_name == "BoTorch"
    assert client.complete_trial.call_count == 4


def test_best_record_uses_observed_successes_and_first_tie() -> None:
    baseline = ax_adapter.AxTrialRecord(0, "baseline", {"x": 0.0}, _success(0.4), None)
    failed = ax_adapter.AxTrialRecord(1, "sobol", {"x": 0.1}, _failure("fea_failure", "x"), "x")
    sobol = ax_adapter.AxTrialRecord(2, "sobol", {"x": 0.2}, _success(0.8), None)
    tie = ax_adapter.AxTrialRecord(3, "bo", {"x": 0.3}, _success(0.8), None)
    assert ax_adapter.AxRunResult((baseline, failed, sobol, tie)).best_record is sobol
    assert ax_adapter.AxRunResult((failed,)).best_record is None
    assert ax_adapter.AxRunResult((replace(baseline, evaluation=_success(0.0)),)).best_record is not None


def test_core_import_does_not_eagerly_import_ax() -> None:
    code = "import sys; import optimization; assert not any(name == 'ax' or name.startswith('ax.') for name in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)
