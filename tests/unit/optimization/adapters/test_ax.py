"""Ax translation tests for the current morphology-only search boundary."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import lumo.optimization.adapters.ax as ax_adapter
from lumo.mesh.volume.mesh import VolumeMeshDependencyError
from lumo.finger import FingertipParameters
from lumo.optimization.adapters.ax import (
    AxRunResult,
    AxSettings,
    AxTerminationReason,
    CampaignInfrastructureError,
    create_ax_client,
    run_ax_optimization,
)
from lumo.optimization.design_space import (
    DesignSpace,
    DesignVariable,
    LATENT_PARAMETER_NAMES,
    PRODUCTION_SEARCH_BOUNDS,
)
from lumo.optimization.evaluation_registry import EvaluationRegistry
from lumo.optimization.objectives import ObjectiveIdentifier
from lumo.ray_tracing.optical_mechanics import Transport3DDependencyError
from lumo.physics import PhysicsDependencyError


TEST_OBJECTIVE = ObjectiveIdentifier("contact_state_separation", 1)


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
    objective_value: float | None
    diagnostics: dict[str, object]
    failure_message: str | None = None
    result_artifact_path: str | None = None

    @property
    def objective(self):
        if self.objective_value is None:
            return None
        return SimpleNamespace(
            objective=TEST_OBJECTIVE,
            objective_value=self.objective_value,
        )


class _Evaluator:
    objective_identifier = TEST_OBJECTIVE

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
    parameters = FingertipParameters(flat_pad_height=value, void_height=0.25)
    return _space().encode(parameters)


def test_create_ax_client_translates_all_six_active_morphology_variables() -> None:
    client = create_ax_client(
        _space(),
        AxSettings(initialization_trials=1, search_trials=1, seed=7),
    )
    assert set(client._experiment.parameters.keys()) == set(LATENT_PARAMETER_NAMES)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_ax_rejects_non_finite_objective_values(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        ax_adapter._evaluation_objective_value(
            SimpleNamespace(objective_value=value),
            "trajectory_objective",
        )


def test_ax_run_result_rejects_contradictory_status_and_termination() -> None:
    with pytest.raises(ValueError, match="status and termination reason disagree"):
        AxRunResult(
            records=(),
            status="nominal_evaluation_failed",
            objective=TEST_OBJECTIVE,
            termination_reason=AxTerminationReason.REQUESTED_BUDGET_REACHED,
        )


def test_run_ax_optimization_evaluates_morphology_without_mechanics_or_optics(
    monkeypatch,
) -> None:
    client = _ClientDouble([_candidate(5.5)])
    evaluator = _Evaluator()
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: client)

    result = run_ax_optimization(
        _space(),
        evaluator,
        AxSettings(
            initialization_trials=1,
            search_trials=1,
            seed=7,
            objective=TEST_OBJECTIVE,
        ),
    )

    assert result.status == "COMPLETE"
    assert result.termination_reason == AxTerminationReason.REQUESTED_BUDGET_REACHED
    assert result.ax_proposal_count == 1
    assert result.successful_initialization_count == 0
    assert result.successful_search_count == 1
    assert result.successful_generated_count == 1
    assert result.nominal_successful is True
    assert result.failure_count_by_status == {}
    assert result.unique_success_count == 2  # manually attached nominal + proposal
    assert len(evaluator.calls) == 2


def test_objective_identity_mismatch_fails_before_ax_generation(monkeypatch) -> None:
    class _MismatchedEvaluator(_Evaluator):
        @property
        def objective_identifier(self):
            return ObjectiveIdentifier("other_objective", 1)

    create_called = False

    def create_client(*_args):
        nonlocal create_called
        create_called = True
        return _ClientDouble([])

    monkeypatch.setattr(ax_adapter, "create_ax_client", create_client)
    with pytest.raises(ValueError, match="does not match"):
        run_ax_optimization(
            _space(),
            _MismatchedEvaluator(),
            AxSettings(
                initialization_trials=1,
                search_trials=1,
                seed=7,
                objective=TEST_OBJECTIVE,
            ),
        )
    assert create_called is False


def test_missing_objective_identity_fails_before_ax_generation(monkeypatch) -> None:
    class _UnidentifiedEvaluator:
        def evaluate(self, _parameters):
            raise AssertionError("evaluation must not start")

    create_called = False

    def create_client(*_args):
        nonlocal create_called
        create_called = True
        return _ClientDouble([])

    monkeypatch.setattr(ax_adapter, "create_ax_client", create_client)
    with pytest.raises(TypeError, match="typed objective_identifier"):
        run_ax_optimization(
            _space(),
            _UnidentifiedEvaluator(),
            AxSettings(
                initialization_trials=1,
                search_trials=1,
                seed=7,
                objective=TEST_OBJECTIVE,
            ),
        )
    assert create_called is False


def test_nested_objective_identity_mismatch_is_not_registered(monkeypatch, tmp_path) -> None:
    class _NestedEvaluator:
        objective_identifier = TEST_OBJECTIVE

        def evaluate(self, _parameters):
            return SimpleNamespace(
                status="success",
                objective_value=1.0,
                objective=SimpleNamespace(
                    objective=ObjectiveIdentifier("wrong_objective", 1),
                    objective_value=1.0,
                ),
                failure_message=None,
            )

    client = _ClientDouble([_candidate(5.5)])
    registry = EvaluationRegistry(tmp_path / "registry.json")
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: client)

    with pytest.raises(ValueError, match="nested objective"):
        run_ax_optimization(
            _space(),
            _NestedEvaluator(),
            AxSettings(
                initialization_trials=1,
                search_trials=1,
                seed=7,
                objective=TEST_OBJECTIVE,
            ),
            evaluation_registry=registry,
            evaluation_contract_id="objective-mismatch-v1",
            campaign_id="objective-mismatch-campaign",
        )
    assert registry.records_for_contract("objective-mismatch-v1") == ()
    assert client.trials[0].status == "ABANDONED"


@pytest.mark.parametrize(
    "nested_objective",
    (
        None,
        SimpleNamespace(objective=TEST_OBJECTIVE, objective_value=2.0),
    ),
)
def test_missing_or_inconsistent_nested_objective_is_not_registered(
    monkeypatch,
    tmp_path,
    nested_objective,
) -> None:
    class _NestedEvaluator:
        objective_identifier = TEST_OBJECTIVE

        def evaluate(self, _parameters):
            return SimpleNamespace(
                status="success",
                objective_value=1.0,
                objective=nested_objective,
                failure_message=None,
            )

    client = _ClientDouble([_candidate(5.5)])
    registry = EvaluationRegistry(tmp_path / "registry.json")
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: client)

    with pytest.raises(ValueError, match="nested objective|disagrees"):
        run_ax_optimization(
            _space(),
            _NestedEvaluator(),
            AxSettings(
                initialization_trials=1,
                search_trials=1,
                seed=7,
                objective=TEST_OBJECTIVE,
            ),
            evaluation_registry=registry,
            evaluation_contract_id="nested-objective-invalid-v1",
            campaign_id="nested-objective-invalid-campaign",
        )
    assert registry.records_for_contract("nested-objective-invalid-v1") == ()
    assert client.trials[0].status == "ABANDONED"


def test_cached_nominal_success_is_accepted_without_reevaluation(
    monkeypatch,
    tmp_path,
) -> None:
    client = _ClientDouble([_candidate(5.5)])
    evaluator = _Evaluator()
    registry = EvaluationRegistry(tmp_path / "registry.json")
    space = _space()
    nominal = space.physical_values(space.nominal_parameters)
    registry.register(
        "cached-nominal-v1",
        nominal,
        status="success",
        first_trial_index=0,
        first_campaign_id="previous-campaign",
        result_artifact_path=None,
        objective=TEST_OBJECTIVE,
        objective_value=0.0,
        failure_category=None,
        failure_message=None,
        failure_scenario=None,
        evaluation_wall_time_seconds=0.0,
    )
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: client)

    result = run_ax_optimization(
        space,
        evaluator,
        AxSettings(
            initialization_trials=1,
            search_trials=1,
            seed=7,
            objective=TEST_OBJECTIVE,
        ),
        evaluation_registry=registry,
        evaluation_contract_id="cached-nominal-v1",
        campaign_id="new-campaign",
    )

    assert result.nominal_successful is True
    assert result.reused_evaluation_count == 1
    assert result.records[0].status == "duplicate_skipped"
    assert result.records[0].reused_evaluation_status == "success"
    assert len(evaluator.calls) == 1
    assert result.records[1].status == "success"


def test_registry_objective_mismatch_fails_before_reuse(monkeypatch, tmp_path) -> None:
    client = _ClientDouble([])
    registry = EvaluationRegistry(tmp_path / "registry.json")
    space = _space()
    registry.register(
        "registry-objective-mismatch-v1",
        space.physical_values(space.nominal_parameters),
        status="success",
        first_trial_index=0,
        first_campaign_id="previous-campaign",
        result_artifact_path=None,
        objective=ObjectiveIdentifier("wrong_objective", 1),
        objective_value=0.0,
        failure_category=None,
        failure_message=None,
        failure_scenario=None,
        evaluation_wall_time_seconds=0.0,
    )
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: client)

    with pytest.raises(ValueError, match="registry objective"):
        run_ax_optimization(
            space,
            _Evaluator(),
            AxSettings(
                initialization_trials=1,
                search_trials=1,
                seed=7,
                objective=TEST_OBJECTIVE,
            ),
            evaluation_registry=registry,
            evaluation_contract_id="registry-objective-mismatch-v1",
            campaign_id="new-campaign",
        )
    assert client.trials == {}


def test_nominal_candidate_failure_stops_before_ax_proposals(monkeypatch) -> None:
    class _NominalFailureEvaluator:
        objective_identifier = TEST_OBJECTIVE

        def evaluate(self, _parameters):
            return _Evaluation(
                status="mechanics_failure",
                objective_value=None,
                diagnostics={},
                failure_message="nominal mechanics failed",
            )

    client = _ClientDouble([_candidate(5.5)])
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: client)

    result = run_ax_optimization(
        _space(),
        _NominalFailureEvaluator(),
        AxSettings(
            initialization_trials=1,
            search_trials=1,
            seed=7,
            objective=TEST_OBJECTIVE,
        ),
    )

    assert result.status == "nominal_evaluation_failed"
    assert result.termination_reason == AxTerminationReason.NOMINAL_FAILED
    assert result.nominal_successful is False
    assert result.ax_proposal_count == 0
    assert len(client.trials) == 1


def test_infeasible_latent_proposal_is_abandoned_and_resampled(monkeypatch) -> None:
    invalid = {name: 0.0 for name in LATENT_PARAMETER_NAMES}
    invalid["latent_cutout_width"] = 1.1
    client = _ClientDouble([invalid, _candidate(5.5)])
    evaluator = _Evaluator()
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: client)

    result = run_ax_optimization(
        _space(),
        evaluator,
        AxSettings(
            initialization_trials=1,
            search_trials=1,
            seed=7,
            objective=TEST_OBJECTIVE,
        ),
        max_feasibility_resamples=1,
    )

    assert result.status == "COMPLETE"
    assert result.ax_proposal_count == 2
    assert result.feasible_proposal_count == 1
    assert result.feasibility_rejection_count == 1
    assert result.feasibility_rejection_counts["latent_bounds"] == 1
    assert result.records[1].status == "feasibility_rejected"
    assert result.records[1].feasibility_constraint == "latent_bounds"
    assert client.trials[1].status == "ABANDONED"
    assert len(evaluator.calls) == 2


def test_feasibility_generation_exhaustion_does_not_reach_evaluator(
    monkeypatch,
    tmp_path,
) -> None:
    invalid = {name: 0.0 for name in LATENT_PARAMETER_NAMES}
    invalid["latent_cutout_width"] = 1.1
    client = _ClientDouble([invalid])
    evaluator = _Evaluator()
    registry = EvaluationRegistry(tmp_path / "registry.json")
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: client)

    result = run_ax_optimization(
        _space(),
        evaluator,
        AxSettings(
            initialization_trials=1,
            search_trials=1,
            seed=7,
            objective=TEST_OBJECTIVE,
        ),
        evaluation_registry=registry,
        evaluation_contract_id="feasibility-exhaustion-v1",
        campaign_id="feasibility-exhaustion-campaign",
        max_feasibility_resamples=0,
    )

    assert result.status == "feasible_generation_exhausted"
    assert result.feasibility_rejection_count == 1
    assert len(evaluator.calls) == 1  # nominal only
    assert len(registry.records_for_contract("feasibility-exhaustion-v1")) == 1
    assert client.trials[1].status == "ABANDONED"


class _PhysicsDependencyEvaluator:
    objective_identifier = TEST_OBJECTIVE

    def evaluate(self, parameters):
        raise PhysicsDependencyError("Warp runtime unavailable")


class _CandidateFailureEvaluator:
    objective_identifier = TEST_OBJECTIVE

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, parameters):
        self.calls += 1
        if self.calls == 1:
            return _Evaluation("success", 0.0, {})
        return _Evaluation(
            status="mechanics_failure",
            objective_value=None,
            diagnostics={"failure_scenario": "candidate_contact"},
            failure_message="CandidateContactError: candidate contact is impossible",
        )


def test_shared_physics_dependency_abandons_ax_trial_and_aborts_campaign(monkeypatch) -> None:
    client = _ClientDouble([_candidate(5.5)])
    evaluator = _PhysicsDependencyEvaluator()
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: client)

    with pytest.raises(CampaignInfrastructureError) as raised:
        run_ax_optimization(
            _space(),
            evaluator,
            AxSettings(
                initialization_trials=1,
                search_trials=1,
                seed=7,
                objective=TEST_OBJECTIVE,
            ),
        )

    assert raised.value.signature == "newton-warp-runtime-initialization"
    assert client.trials
    assert all(trial.status == "ABANDONED" for trial in client.trials.values())


def test_candidate_failure_is_registered_in_real_evaluation_registry(monkeypatch, tmp_path) -> None:
    client = _ClientDouble([_candidate(5.5)])
    evaluator = _CandidateFailureEvaluator()
    registry = EvaluationRegistry(tmp_path / "registry.json")
    registered_counts: list[int] = []
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: client)

    result = run_ax_optimization(
        _space(),
        evaluator,
        AxSettings(
            initialization_trials=1,
            search_trials=1,
            seed=7,
            objective=TEST_OBJECTIVE,
        ),
        evaluation_registry=registry,
        evaluation_contract_id="candidate-contact-test-v1",
        campaign_id="candidate-contact-campaign",
        on_record=lambda _client, _records: registered_counts.append(
            len(registry.records_for_contract("candidate-contact-test-v1"))
        ),
        max_evaluations=2,
    )

    assert result.status == "evaluation_budget_exhausted"
    assert result.termination_reason == AxTerminationReason.EVALUATION_BUDGET_EXHAUSTED
    assert [record.status for record in result.records] == [
        "success",
        "mechanics_failure",
    ]
    stored = registry.records_for_contract("candidate-contact-test-v1")
    assert len(stored) == 2
    assert [record.status for record in stored] == [
        "success",
        "mechanics_failure",
    ]
    assert registered_counts == [1, 2]
    assert client.trials[0].status == "COMPLETED"
    assert client.trials[1].status == "FAILED"
    assert all(trial.status != "ABANDONED" for trial in client.trials.values())


def test_failed_search_evaluation_does_not_reduce_success_target(monkeypatch) -> None:
    client = _ClientDouble([_candidate(5.5), _candidate(6.0)])

    class _FailThenSucceed(_Evaluator):
        def evaluate(self, parameters):
            self.calls.append(parameters)
            if len(self.calls) == 2:
                return _Evaluation(
                    "mechanics_failure",
                    None,
                    {"failure_scenario": "candidate_mechanics_state"},
                    failure_message="candidate failed",
                )
            return _Evaluation("success", float(parameters.flat_pad_height), {})

    evaluator = _FailThenSucceed()
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: client)

    result = run_ax_optimization(
        _space(),
        evaluator,
        AxSettings(
            initialization_trials=1,
            search_trials=1,
            seed=7,
            objective=TEST_OBJECTIVE,
        ),
        max_evaluations=3,
        max_proposals=2,
    )

    assert result.status == "COMPLETE"
    assert result.successful_search_count == 1
    assert [record.status for record in result.records] == [
        "success",
        "mechanics_failure",
        "success",
    ]


def test_proposal_cap_counts_feasibility_rejections(monkeypatch) -> None:
    invalid = {name: 0.0 for name in LATENT_PARAMETER_NAMES}
    invalid["latent_cutout_width"] = 1.1
    client = _ClientDouble([invalid, _candidate(5.5)])
    evaluator = _Evaluator()
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: client)

    result = run_ax_optimization(
        _space(),
        evaluator,
        AxSettings(
            initialization_trials=1,
            search_trials=1,
            seed=7,
            objective=TEST_OBJECTIVE,
        ),
        max_proposals=1,
        max_feasibility_resamples=10,
    )

    assert result.status == "proposal_budget_exhausted"
    assert result.ax_proposal_count == 1
    assert result.feasibility_rejection_count == 1
    assert len(evaluator.calls) == 1


def test_registry_uses_each_candidate_evaluation_artifact_path(
    monkeypatch,
    tmp_path,
) -> None:
    client = _ClientDouble([_candidate(5.5)])

    class _ArtifactEvaluator:
        objective_identifier = TEST_OBJECTIVE

        def evaluate(self, parameters):
            artifact = tmp_path / f"candidate_{parameters.flat_pad_height:g}.json"
            return _Evaluation(
                status="success",
                objective_value=parameters.flat_pad_height,
                diagnostics={},
                result_artifact_path=str(artifact),
            )

    evaluator = _ArtifactEvaluator()
    registry = EvaluationRegistry(tmp_path / "registry.json")
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: client)

    run_ax_optimization(
        _space(),
        evaluator,
        AxSettings(
            initialization_trials=1,
            search_trials=1,
            seed=7,
            objective=TEST_OBJECTIVE,
        ),
        evaluation_registry=registry,
        evaluation_contract_id="candidate-artifact-test-v1",
        campaign_id="candidate-artifact-campaign",
        result_artifact_path=str(tmp_path / "shared_trials.json"),
    )

    stored = registry.records_for_contract("candidate-artifact-test-v1")
    assert {record.result_artifact_path for record in stored} == {
        str(tmp_path / "candidate_5.json"),
        str(tmp_path / "candidate_5.5.json"),
    }


@pytest.mark.parametrize(
    ("exception", "signature"),
    (
        (
            VolumeMeshDependencyError("gmsh unavailable"),
            "gmsh-runtime-initialization",
        ),
        (PhysicsDependencyError("Warp runtime unavailable"), "newton-warp-runtime-initialization"),
        (Transport3DDependencyError("OptiX unavailable"), "optix-runtime-initialization"),
    ),
)
def test_infrastructure_failure_is_not_registered_in_real_evaluation_registry(
    monkeypatch,
    tmp_path,
    exception: Exception,
    signature: str,
) -> None:
    client = _ClientDouble([_candidate(5.5)])

    class _InfrastructureFailureEvaluator:
        objective_identifier = TEST_OBJECTIVE

        def evaluate(self, parameters):
            raise exception

    evaluator = _InfrastructureFailureEvaluator()
    registry = EvaluationRegistry(tmp_path / "registry.json")
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_: client)

    with pytest.raises(CampaignInfrastructureError) as raised:
        run_ax_optimization(
            _space(),
            evaluator,
            AxSettings(
                initialization_trials=1,
                search_trials=1,
                seed=7,
                objective=TEST_OBJECTIVE,
            ),
            evaluation_registry=registry,
            evaluation_contract_id="infrastructure-test-v1",
            campaign_id="infrastructure-campaign",
        )

    assert raised.value.signature == signature
    assert registry.records_for_contract("infrastructure-test-v1") == ()
    assert all(trial.status == "ABANDONED" for trial in client.trials.values())
