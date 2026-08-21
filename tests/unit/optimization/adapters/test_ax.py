"""Ax translation tests for the current morphology-only search boundary."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import lumo.optimization.adapters.ax as ax_adapter
from lumo.mesh.volume.mesh import VolumeMeshDependencyError
from lumo.finger import FingertipParameters
from lumo.optimization.adapters.ax import (
    AxSettings,
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
    assert result.ax_proposal_count == 1
    assert result.unique_success_count == 2  # manually attached nominal + proposal
    assert len(evaluator.calls) == 2
    assert all(record.status == "success" for record in result.records)


def test_infeasible_latent_proposal_is_abandoned_and_resampled(monkeypatch) -> None:
    invalid = {name: 0.0 for name in LATENT_PARAMETER_NAMES}
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
    assert result.feasibility_rejection_counts["minimum_silicone_thickness"] == 1
    assert result.records[1].status == "feasibility_rejected"
    assert result.records[1].feasibility_constraint == "minimum_silicone_thickness"
    assert client.trials[1].status == "ABANDONED"
    assert len(evaluator.calls) == 2


def test_feasibility_generation_exhaustion_does_not_reach_evaluator(
    monkeypatch,
    tmp_path,
) -> None:
    invalid = {name: 0.0 for name in LATENT_PARAMETER_NAMES}
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
    def evaluate(self, parameters):
        raise PhysicsDependencyError("Warp runtime unavailable")


class _CandidateFailureEvaluator:
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
    )

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


def test_registry_uses_each_candidate_evaluation_artifact_path(
    monkeypatch,
    tmp_path,
) -> None:
    client = _ClientDouble([_candidate(5.5)])

    class _ArtifactEvaluator:
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
