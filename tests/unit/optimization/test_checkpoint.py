"""Low-cost checkpoint and resume contracts for the Ax boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import lumo.optimization.adapters.ax as ax_adapter
from lumo.finger import FingertipParameters
from lumo.optimization.adapters.ax import (
    AxResumeState,
    AxSettings,
    create_ax_client,
    run_ax_optimization,
)
from lumo.optimization.checkpoint import (
    CampaignCheckpointStore,
    CheckpointError,
)
from lumo.optimization.design_space import (
    DesignSpace,
    DesignVariable,
    PRODUCTION_SEARCH_BOUNDS,
)
from lumo.optimization.evaluation_registry import EvaluationRegistry
from lumo.optimization.objectives import TRAJECTORY_SEPARATION_OBJECTIVE


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
    objective: object
    failure_message: str | None = None


class _Evaluator:
    objective_identifier = TRAJECTORY_SEPARATION_OBJECTIVE

    def __init__(self) -> None:
        self.calls: list[object] = []

    def evaluate(self, parameters) -> _Evaluation:
        self.calls.append(parameters)
        value = float(parameters.flat_pad_height)
        return _Evaluation(
            status="success",
            objective_value=value,
            objective=SimpleNamespace(
                objective=TRAJECTORY_SEPARATION_OBJECTIVE,
                objective_value=value,
            ),
        )


@dataclass
class _Trial:
    parameters: dict[str, float]
    status: str = "CANDIDATE"


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

    def mark_trial_failed(self, trial_index: int, failed_reason=None) -> None:
        self.trials[trial_index].status = "FAILED"

    def mark_trial_abandoned(self, trial_index: int) -> None:
        self.trials[trial_index].status = "ABANDONED"


def _candidate(space: DesignSpace, flat_pad_height: float) -> dict[str, float]:
    return space.encode(
        FingertipParameters(
            flat_pad_height=flat_pad_height,
            void_height=0.25,
        )
    )


def _state(phase: str, sequence: int = 1) -> dict[str, object]:
    return {
        "campaign_id": "test-campaign",
        "evaluation_contract_id": "test-contract",
        "objective_identifier": {"name": "objective", "version": 1},
        "design_space": {},
        "parameterization_version": "test-v1",
        "ax_package_version": "1.3.1",
        "seed": 7,
        "budget": {},
        "counts": {},
        "pending_trial_index": None,
        "pending_latent_parameters": None,
        "pending_physical_parameters": None,
        "registry_key": None,
        "source": {"git_commit": "test"},
        "resume_contract": {"contract": "test"},
        "phase": phase,
        "sequence": sequence,
    }


def test_checkpoint_store_writes_immutable_versions_and_rejects_corrupt_pointer(
    tmp_path: Path,
) -> None:
    class _SavableClient:
        def save_to_json_file(self, filepath: str) -> None:
            Path(filepath).write_text('{"ax": "state"}\n', encoding="utf-8")

    store = CampaignCheckpointStore(tmp_path / "campaign")
    with store.writer_lock():
        first = store.write(
            ax_client=_SavableClient(),
            trials=[],
            state=_state("post_evaluation"),
        )
        second = store.write(
            ax_client=_SavableClient(),
            trials=[{"trial_index": 1}],
            state=_state("pre_evaluation"),
        )
    assert first.name == "000001"
    assert second.name == "000002"
    loaded = store.load_latest()
    assert loaded.directory == second
    assert list(loaded.trials) == [{"trial_index": 1}]
    assert not list((tmp_path / "campaign" / "checkpoints").glob(".*-incomplete"))

    (tmp_path / "campaign" / "checkpoints" / ".000003-incomplete").mkdir()
    assert store.load_latest().directory == second

    (tmp_path / "campaign" / "checkpoint.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CheckpointError, match="pointer schema"):
        store.load_latest()


def test_checkpoint_uses_ax_public_save_and_load_api(tmp_path: Path) -> None:
    space = _space()
    client = create_ax_client(
        space,
        AxSettings(
            initialization_trials=1,
            search_trials=1,
            seed=7,
            objective=TRAJECTORY_SEPARATION_OBJECTIVE,
        ),
    )
    store = CampaignCheckpointStore(tmp_path / "campaign")
    with store.writer_lock():
        store.write(
            ax_client=client,
            trials=[],
            state=_state("post_evaluation"),
        )
    restored = store.load_ax_client(store.load_latest())
    assert restored._experiment.parameters.keys() == client._experiment.parameters.keys()


def test_resume_reconciles_pending_candidate_before_requesting_next_proposal(
    monkeypatch,
) -> None:
    space = _space()
    candidates = [_candidate(space, 5.5), _candidate(space, 6.5)]
    uninterrupted_client = _ClientDouble(candidates.copy())
    monkeypatch.setattr(
        ax_adapter,
        "create_ax_client",
        lambda *_args: uninterrupted_client,
    )
    settings = AxSettings(
        initialization_trials=1,
        search_trials=2,
        seed=7,
        objective=TRAJECTORY_SEPARATION_OBJECTIVE,
    )
    uninterrupted = run_ax_optimization(
        space,
        _Evaluator(),
        settings,
        max_proposals=2,
    )

    interrupted_client = _ClientDouble(candidates.copy())
    monkeypatch.setattr(
        ax_adapter,
        "create_ax_client",
        lambda *_args: interrupted_client,
    )
    captured = []

    def interrupt(event) -> None:
        captured.append(event)
        if (
            event.phase == "pre_evaluation"
            and event.pending_trial is not None
            and event.pending_trial.phase != "nominal"
        ):
            raise RuntimeError("simulated process interruption")

    with pytest.raises(RuntimeError, match="simulated process interruption"):
        run_ax_optimization(
            space,
            _Evaluator(),
            settings,
            max_proposals=2,
            on_checkpoint=interrupt,
        )
    pending_event = next(
        event
        for event in captured
        if event.pending_trial is not None and event.pending_trial.phase != "nominal"
    )
    resumed = run_ax_optimization(
        space,
        _Evaluator(),
        settings,
        max_proposals=2,
        resume_state=AxResumeState(
            client=pending_event.client,
            records=pending_event.records,
            pending_trial=pending_event.pending_trial,
        ),
    )

    uninterrupted_parameters = [
        dict(record.parameters) for record in uninterrupted.records
    ]
    resumed_parameters = [dict(record.parameters) for record in resumed.records]
    assert resumed_parameters == uninterrupted_parameters
    assert resumed.ax_proposal_count == uninterrupted.ax_proposal_count == 2


def test_resume_replays_registry_hit_without_reevaluating_pending_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    space = _space()
    candidates = [_candidate(space, 5.5), _candidate(space, 6.5)]
    client = _ClientDouble(candidates.copy())
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_args: client)
    settings = AxSettings(
        initialization_trials=1,
        search_trials=2,
        seed=7,
        objective=TRAJECTORY_SEPARATION_OBJECTIVE,
    )
    captured = []

    def interrupt(event) -> None:
        captured.append(event)
        if (
            event.phase == "pre_evaluation"
            and event.pending_trial is not None
            and event.pending_trial.phase != "nominal"
        ):
            raise RuntimeError("simulated process interruption")

    with pytest.raises(RuntimeError):
        run_ax_optimization(
            space,
            _Evaluator(),
            settings,
            max_proposals=2,
            on_checkpoint=interrupt,
        )
    pending_event = next(
        event
        for event in captured
        if event.pending_trial is not None and event.pending_trial.phase != "nominal"
    )
    pending = pending_event.pending_trial
    assert pending is not None
    physical = space.physical_values(space.decode(pending.latent_parameters))
    registry = EvaluationRegistry(tmp_path / "registry.json")
    registry.register(
        "resume-contract",
        physical,
        status="success",
        first_trial_index=99,
        first_campaign_id="previous-campaign",
        result_artifact_path=None,
        objective=TRAJECTORY_SEPARATION_OBJECTIVE,
        objective_value=5.5,
        failure_category=None,
        failure_message=None,
        failure_scenario=None,
        evaluation_wall_time_seconds=0.0,
    )
    evaluator = _Evaluator()
    resumed = run_ax_optimization(
        space,
        evaluator,
        settings,
        max_proposals=2,
        evaluation_registry=registry,
        evaluation_contract_id="resume-contract",
        campaign_id="resumed-campaign",
        resume_state=AxResumeState(
            client=pending_event.client,
            records=pending_event.records,
            pending_trial=pending,
        ),
    )

    assert resumed.records[1].reused_evaluation is True
    assert resumed.records[1].reused_evaluation_status == "success"
    assert len(evaluator.calls) == 1
    assert resumed.ax_proposal_count == 2
    assert resumed.status == "COMPLETE"
    assert resumed.successful_search_count == 2
    assert resumed.new_evaluation_count == 2  # nominal plus one non-reused search


def test_resume_from_completed_post_checkpoint_does_not_generate_extra_trial(
    monkeypatch,
) -> None:
    space = _space()
    candidates = [_candidate(space, 5.5), _candidate(space, 6.5)]
    client = _ClientDouble(candidates.copy())
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_args: client)
    settings = AxSettings(
        initialization_trials=1,
        search_trials=1,
        seed=7,
        objective=TRAJECTORY_SEPARATION_OBJECTIVE,
    )
    checkpoints = []
    completed = run_ax_optimization(
        space,
        _Evaluator(),
        settings,
        max_proposals=2,
        on_checkpoint=checkpoints.append,
    )
    final = checkpoints[-1]
    assert final.phase == "post_evaluation"
    assert final.pending_trial is None
    assert completed.status == "COMPLETE"
    assert completed.ax_proposal_count == 1

    evaluator = _Evaluator()
    resumed = run_ax_optimization(
        space,
        evaluator,
        settings,
        max_proposals=2,
        resume_state=AxResumeState(
            client=final.client,
            records=final.records,
            pending_trial=None,
        ),
    )

    assert evaluator.calls == []
    assert resumed.status == "COMPLETE"
    assert resumed.ax_proposal_count == 1
    assert [dict(record.parameters) for record in resumed.records] == [
        dict(record.parameters) for record in completed.records
    ]


def test_resume_abandons_pending_trial_when_decode_is_infeasible(monkeypatch) -> None:
    space = _space()
    invalid = {variable.name: 0.5 for variable in space.search_variables}
    invalid["latent_cutout_width"] = 1.1
    client = _ClientDouble([invalid])
    monkeypatch.setattr(ax_adapter, "create_ax_client", lambda *_args: client)
    settings = AxSettings(
        initialization_trials=1,
        search_trials=1,
        seed=7,
        objective=TRAJECTORY_SEPARATION_OBJECTIVE,
    )
    captured = []

    def interrupt(event) -> None:
        captured.append(event)
        if event.pending_trial is not None and event.pending_trial.phase != "nominal":
            raise RuntimeError("simulated process interruption")

    with pytest.raises(RuntimeError, match="simulated process interruption"):
        run_ax_optimization(
            space,
            _Evaluator(),
            settings,
            max_proposals=1,
            max_feasibility_resamples=0,
            on_checkpoint=interrupt,
        )
    pending_event = next(
        event
        for event in captured
        if event.pending_trial is not None and event.pending_trial.phase != "nominal"
    )
    pending = pending_event.pending_trial
    assert pending is not None

    resumed = run_ax_optimization(
        space,
        _Evaluator(),
        settings,
        max_proposals=1,
        max_feasibility_resamples=0,
        resume_state=AxResumeState(
            client=pending_event.client,
            records=pending_event.records,
            pending_trial=pending,
        ),
    )

    assert resumed.status == "feasible_generation_exhausted"
    assert resumed.records[-1].status == "feasibility_rejected"
    assert client.trials[pending.trial_index].status == "ABANDONED"
