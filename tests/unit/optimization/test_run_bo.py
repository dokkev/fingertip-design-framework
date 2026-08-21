from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import scripts.optimization.run_bo as run_bo
from lumo.optimization.adapters.ax import AxTerminationReason
from lumo.optimization.checkpoint import CampaignCheckpointStore
from lumo.optimization.objectives import TRAJECTORY_SEPARATION_OBJECTIVE


def test_user_config_contains_the_production_search_controls() -> None:
    execution = run_bo.load_lumo_execution_config(run_bo.DEFAULT_EXECUTION_CONFIG)
    budget = run_bo.CampaignBudget(6, 2, 12, 11)
    payload = run_bo._user_config_payload(execution=execution, budget=budget)

    assert {
        "flat_pad_height",
        "semielliptical_pad_height",
        "stem_width",
        "stem_height",
        "void_width",
        "void_height",
    }.issubset(payload["nominal_parameters"])
    assert payload["campaign_mode"] == "production"
    assert payload["trajectory_protocol"] == run_bo.USER_PROTOCOL.to_dict()
    assert payload["search_bounds"] == [
        bound.to_dict() for bound in run_bo.USER_SEARCH_BOUNDS
    ]
    assert payload["objective"]["config"] == asdict(run_bo.USER_OBJECTIVE)
    assert "success targets are independent" in payload["ax"]["budget_semantics"]
    assert payload["trajectory_protocol"]["contact_locations_u"] == [
        0.25,
        0.5,
        0.75,
    ]
    assert payload["ax"]["initialization_success_target"] == 6
    assert payload["ax"]["search_success_target"] == 2
    assert payload["ax"]["maximum_evaluations"] == 12
    assert payload["ax"]["maximum_proposals"] == 11
    assert payload["execution_config"]["source"]["sha256"]
    assert payload["ax"]["max_feasibility_resamples"] == (
        run_bo.MAX_FEASIBILITY_RESAMPLES
    )
    assert payload["seed"] == run_bo.SEED
    assert run_bo._user_config_payload(
        execution=execution,
        budget=budget,
        campaign_mode="smoke",
        seed=run_bo.SMOKE_SEED,
    )["seed"] == run_bo.SMOKE_SEED
    assert run_bo.SMOKE_SEED != run_bo.SEED
    json.dumps(payload, allow_nan=False)


def test_controlled_campaign_failure_returns_nonzero_cli_status(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        run_bo,
        "run_campaign",
        lambda *_args, **_kwargs: {"status": "FAILED"},
    )

    assert (
        run_bo.main(
            ["--smoke", "--output", str(tmp_path / "failed")]
        )
        == 3
    )
    assert json.loads(capsys.readouterr().out)["status"] == "FAILED"


def test_configuration_or_persistence_failure_returns_infrastructure_status(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    def fail_campaign(*_args, **_kwargs):
        raise FileExistsError("output already exists")

    monkeypatch.setattr(run_bo, "run_campaign", fail_campaign)

    assert (
        run_bo.main(
            [
                "--search-successes", "1",
                "--max-evaluations", "8",
                "--max-proposals", "7",
                "--output", str(tmp_path / "existing"),
            ]
        )
        == 2
    )
    assert "CAMPAIGN_ABORTED" in capsys.readouterr().err


def test_campaign_acceptance_requires_a_generated_success() -> None:
    result = SimpleNamespace(
        nominal_successful=True,
        successful_initialization_count=1,
        successful_search_count=0,
        termination_reason=AxTerminationReason.REQUESTED_BUDGET_REACHED,
    )
    assert run_bo._campaign_acceptance(
        result,
        budget=run_bo.CampaignBudget(1, 1, 3, 2),
    ) == (
        "FAILED",
        ["search_success_target_not_reached"],
    )


def test_campaign_budget_separates_targets_from_hard_caps() -> None:
    budget = run_bo.CampaignBudget(6, 4, 16, 20)
    assert budget.to_dict() == {
        "initialization_success_target": 6,
        "search_success_target": 4,
        "maximum_evaluations": 16,
        "maximum_proposals": 20,
    }
    with pytest.raises(ValueError, match="nominal plus all success targets"):
        run_bo.CampaignBudget(6, 4, 10, 20)
    with pytest.raises(ValueError, match="all generated success targets"):
        run_bo.CampaignBudget(6, 4, 11, 9)


def test_production_rejects_zero_search_success_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one successful MBM"):
        run_bo.run_campaign(
            tmp_path / "campaign",
            budget=run_bo.CampaignBudget(6, 0, 7, 6),
        )


def test_production_requires_at_least_six_successful_sobol_observations(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="at least 6 successful Sobol"):
        run_bo.run_campaign(
            tmp_path / "campaign",
            budget=run_bo.CampaignBudget(1, 1, 3, 2),
        )


def test_production_cli_rejects_reduced_sobol_target() -> None:
    with pytest.raises(SystemExit):
        run_bo.main(
            [
                "--initialization-successes", "1",
                "--search-successes", "1",
                "--max-evaluations", "3",
                "--max-proposals", "2",
            ]
        )


def test_production_cli_rejects_zero_search_success_target() -> None:
    with pytest.raises(SystemExit):
        run_bo.main(
            [
                "--search-successes",
                "0",
                "--max-evaluations",
                "7",
                "--max-proposals",
                "6",
            ]
        )


def test_source_provenance_hashes_untracked_contents_and_dirty_policy(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    tracked = repository / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=repository, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
            "commit", "-qm", "initial",
        ],
        cwd=repository,
        check=True,
    )
    clean = run_bo._source_provenance(repository)
    assert clean["git_dirty"] is False

    tracked.write_text("value = 2\n", encoding="utf-8")
    untracked = repository / "new_config.yaml"
    untracked.write_text("value: first\n", encoding="utf-8")
    first = run_bo._source_provenance(repository)
    untracked.write_text("value: second\n", encoding="utf-8")
    second = run_bo._source_provenance(repository)

    assert first["tracked_diff_sha256"] == second["tracked_diff_sha256"]
    assert first["untracked_content_sha256"] != second["untracked_content_sha256"]
    assert first["source_id"] != second["source_id"]
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        run_bo._enforce_source_policy(first, smoke=False, allow_dirty=False)
    run_bo._enforce_source_policy(first, smoke=False, allow_dirty=True)
    run_bo._enforce_source_policy(first, smoke=True, allow_dirty=False)


def test_checkpoint_evaluation_count_excludes_registry_replay() -> None:
    records = (
        SimpleNamespace(status="success", reused_evaluation=False),
        SimpleNamespace(status="success", reused_evaluation=True),
        SimpleNamespace(status="mechanics_failure", reused_evaluation=False),
        SimpleNamespace(status="duplicate_skipped", reused_evaluation=True),
        SimpleNamespace(status="feasibility_rejected", reused_evaluation=False),
    )

    assert run_bo._actual_evaluation_count(records) == 2


def test_source_provenance_excludes_mutable_campaign_outputs(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    tracked = repository / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=repository, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
            "commit", "-qm", "initial",
        ],
        cwd=repository,
        check=True,
    )
    output = repository / "campaign"
    output.mkdir()
    artifact = output / "summary.json"
    artifact.write_text('{"status": "first"}\n', encoding="utf-8")
    first = run_bo._source_provenance(
        repository,
        excluded_paths=(output,),
    )
    artifact.write_text('{"status": "second"}\n', encoding="utf-8")
    second = run_bo._source_provenance(
        repository,
        excluded_paths=(output,),
    )

    assert first["source_id"] == second["source_id"]
    assert first["git_dirty"] is False


def test_git_tracked_registry_is_rejected_before_it_can_break_resume(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    registry = repository / "registry.json"
    registry.write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "add", "registry.json"], cwd=repository, check=True)

    with pytest.raises(ValueError, match="must not be a Git-tracked file"):
        run_bo._reject_git_tracked_mutable_path(registry, repository)

    registry_lock = repository / ".registry.json.lock"
    registry_lock.write_text("pid=1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", ".registry.json.lock"],
        cwd=repository,
        check=True,
    )
    with pytest.raises(ValueError, match="must not be a Git-tracked file"):
        run_bo._reject_git_tracked_mutable_path(registry_lock, repository)

    untracked = repository / "untracked-registry.json"
    run_bo._reject_git_tracked_mutable_path(untracked, repository)


def test_resume_rejects_immutable_checkpoint_directory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "campaign" / "checkpoints" / "000002"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(run_bo.CheckpointError, match="does not accept"):
        run_bo._resume_root(checkpoint)

def test_record_payload_does_not_hide_non_rejected_decode_failure() -> None:
    design_space = run_bo._design_space(
        run_bo.USER_PARAMETERS,
        run_bo.USER_SEARCH_BOUNDS,
    )
    invalid = {
        variable.name: 0.0 for variable in design_space.search_variables
    }
    invalid["latent_cutout_width"] = 1.1
    record = SimpleNamespace(
        evaluation=None,
        feasibility_rejection=False,
        parameters=invalid,
    )

    with pytest.raises(ValueError, match="latent bounds"):
        run_bo._record_payload(record, design_space)


def test_preflight_only_reports_external_failure_without_starting_campaign(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    requested_output = tmp_path / "custom-output"
    seen_outputs: list[object] = []

    monkeypatch.setattr(
        run_bo,
        "_preflight_payload",
        lambda output, **_kwargs: seen_outputs.append(output)
        or {
            "schema": "test",
            "status": "FAIL_EXTERNAL_PREREQUISITE",
            "failed_checks": ["optix_smoke"],
            "checks": {},
        },
    )

    assert run_bo.main(["--preflight", "--output", str(requested_output)]) == 2
    assert seen_outputs == [requested_output]
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "FAIL_EXTERNAL_PREREQUISITE"


def test_preflight_success_is_a_read_only_exit(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        run_bo,
        "_preflight_payload",
        lambda _output, **_kwargs: {
            "schema": "test",
            "status": "PASS",
            "failed_checks": [],
            "checks": {},
        },
    )

    assert run_bo.main(["--preflight"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "PASS"


def test_preflight_rejects_unavailable_source_before_external_checks(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        run_bo,
        "_source_provenance",
        lambda **_kwargs: {"status": "unavailable", "source_id": None},
    )
    monkeypatch.setattr(
        run_bo,
        "_preflight_payload",
        lambda *_args, **_kwargs: pytest.fail("external preflight was started"),
    )

    assert run_bo.main(["--preflight"]) == 2
    assert "PREFLIGHT_ABORTED" in capsys.readouterr().err


def test_preflight_builds_configuration_under_requested_output_root(
    monkeypatch,
    tmp_path,
) -> None:
    requested_output = tmp_path / "custom-output"
    requested_output.mkdir()
    marker = requested_output / "existing.txt"
    marker.write_text("keep", encoding="utf-8")
    evaluator_roots: list[object] = []

    monkeypatch.setattr(
        run_bo.importlib,
        "import_module",
        lambda _name: object(),
    )

    class _Evaluator:
        def __init__(self, root, **_kwargs):
            evaluator_roots.append(root)

    monkeypatch.setattr(run_bo, "Lumo3DTrajectoryEvaluator", _Evaluator)
    monkeypatch.setattr(
        run_bo,
        "_run_optix_smoke",
        lambda _device: SimpleNamespace(
            to_dict=lambda: {"hit": True, "miss": True}
        ),
    )
    monkeypatch.setattr(
        run_bo,
        "_run_newton_smoke",
        lambda _device: {"finite_result": True},
    )

    payload = run_bo._preflight_payload(
        requested_output,
        execution=run_bo.load_lumo_execution_config(
            run_bo.DEFAULT_EXECUTION_CONFIG
        ),
    )

    assert payload["status"] == "PASS"
    assert payload["source"]["source_id"]
    assert evaluator_roots == [requested_output / "preflight-artifacts"]
    assert requested_output != run_bo.DEFAULT_OUTPUT
    assert marker.read_text(encoding="utf-8") == "keep"


def test_campaign_resume_restores_public_ax_checkpoint_after_pre_evaluation(
    monkeypatch,
    tmp_path,
) -> None:
    class _Evaluation:
        status = "success"
        objective_value = 1.0
        failure_message = None
        result_artifact_path = None
        failure_scenario = None
        objective = SimpleNamespace(
            objective=TRAJECTORY_SEPARATION_OBJECTIVE,
            objective_value=1.0,
        )

    class _Evaluator:
        objective_identifier = TRAJECTORY_SEPARATION_OBJECTIVE
        evaluation_contract_id = "checkpoint-test-contract"

        def __init__(self, *_args, **_kwargs) -> None:
            self.calls = 0

        def evaluate(self, _parameters):
            self.calls += 1
            return _Evaluation()

    monkeypatch.setattr(
        run_bo,
        "_preflight_payload",
        lambda _output, **_kwargs: {
            "schema": "test",
            "status": "PASS",
            "failed_checks": [],
            "checks": {},
        },
    )
    monkeypatch.setattr(run_bo, "Lumo3DTrajectoryEvaluator", _Evaluator)
    output = tmp_path / "campaign"
    budget = run_bo.CampaignBudget(1, 0, 2, 1)
    original_write = CampaignCheckpointStore.write

    def interrupt_after_persisted_pre_checkpoint(self, *, ax_client, trials, state):
        path = original_write(
            self,
            ax_client=ax_client,
            trials=trials,
            state=state,
        )
        if state["phase"] == "pre_evaluation" and state["pending_phase"] != "nominal":
            raise RuntimeError("simulated process interruption")
        return path

    monkeypatch.setattr(
        CampaignCheckpointStore,
        "write",
        interrupt_after_persisted_pre_checkpoint,
    )
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        run_bo.run_campaign(output, budget=budget, smoke=True)
    assert (output / "checkpoint.json").is_file()
    assert (output / "checkpoints" / "000003" / "state.json").is_file()

    monkeypatch.setattr(CampaignCheckpointStore, "write", original_write)
    summary = run_bo.run_campaign(
        output,
        budget=budget,
        smoke=True,
        resume=output,
    )
    assert summary["resumed"] is True
    assert summary["status"] == "PASS"
    latest = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert latest["sequence"] >= 5
