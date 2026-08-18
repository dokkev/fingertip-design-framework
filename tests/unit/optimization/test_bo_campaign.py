"""Cheap campaign persistence and reporting-policy contracts."""

from __future__ import annotations

import pytest

from validation.common.io import atomic_write_json
from validation.optimization import bo_campaign
from validation.optimization.bo_campaign import (
    _configuration,
    _import_historical_checkpoint,
    _plateau_assessment,
    _write_summary,
)
from validation.optimization.registry_cleanup import (
    ABORTED_OPTIX_HEADER_FAILURE_SIGNATURE,
    cleanup_aborted_infrastructure_records,
)
from optics.optix.smoke import ProductionOptixSmokeError
from optimization.ax_adapter import (
    AxRunResult,
    AxSettings,
    AxTrialRecord,
    CampaignInfrastructureError,
)
from optimization.evaluation_registry import EvaluationRegistry
from optimization.study import (
    PRODUCTION_EVALUATION_CONTRACT_ID,
    create_production_study,
)


NOMINAL_PARAMETERS = {
    "flat_pad_height": 5.0,
    "stem_width": 7.6,
    "stem_height": 6.0,
    "void_width": 1.0,
}


def _record(
    index: int,
    *,
    status: str = "success",
    value: float | None = 0.5,
    flat_pad_height: float | None = None,
) -> dict[str, object]:
    parameters = {
        "flat_pad_height": (
            5.0 + 0.01 * index
            if flat_pad_height is None
            else flat_pad_height
        ),
        "stem_width": 7.6,
        "stem_height": 6.0,
        "void_width": 1.0,
    }
    return {
        "trial_index": index,
        "phase": "search",
        "parameters": parameters,
        "status": status,
        "minimum_auc": value,
        "failure_message": None if status == "success" else "synthetic failure",
        "failure_scenario": None,
        "wall_time_seconds": 1.0,
        "registry_key": None,
    }


def test_plateau_requires_five_unique_successful_search_evaluations() -> None:
    records = [_record(index, value=0.5) for index in range(4)]
    records.extend(
        [
            _record(10, status="fea_failure", value=None),
            _record(11, status="duplicate_skipped", value=None),
            _record(12, status="optics_failure", value=None),
        ]
    )
    assert _plateau_assessment(records) == "insufficient_data"


def test_plateau_uses_only_unique_successes_and_detects_improvement() -> None:
    plateau = [_record(index, value=0.5) for index in range(5)]
    improved = [
        _record(index, value=value)
        for index, value in enumerate((0.5, 0.5, 0.5, 0.5, 0.6))
    ]
    duplicate = dict(plateau[-1])
    duplicate["trial_index"] = 30
    duplicate["registry_key"] = "same-exact-key"
    plateau[-1]["registry_key"] = "same-exact-key"

    assert _plateau_assessment([*plateau, duplicate]) == "plateau"
    assert _plateau_assessment(improved) == "improved"


def test_historical_checkpoint_imports_each_exact_result_once(tmp_path) -> None:
    configuration = _configuration(
        create_production_study(),
        AxSettings(initialization_trials=1, search_trials=1, seed=5),
    )
    first = _record(2, value=0.55)
    repeated_first = dict(first)
    repeated_first["trial_index"] = 10
    failed = _record(3, status="optics_failure", value=None)
    checkpoint = tmp_path / "old-campaign" / "checkpoint.json"
    atomic_write_json(
        checkpoint,
        {
            "configuration": configuration,
            "records": [first, repeated_first, failed],
        },
    )
    registry_path = tmp_path / "registry.json"
    registry = EvaluationRegistry(registry_path)

    assert _import_historical_checkpoint(
        registry,
        checkpoint,
        expected_configuration=configuration,
    ) == 2
    assert _import_historical_checkpoint(
        registry,
        checkpoint,
        expected_configuration=configuration,
    ) == 0

    reloaded = EvaluationRegistry(registry_path)
    records = reloaded.records_for_contract(PRODUCTION_EVALUATION_CONTRACT_ID)
    assert len(records) == 2
    assert {record.status for record in records} == {"success", "optics_failure"}
    assert sum(record.minimum_auc is not None for record in records) == 1


def test_summary_restores_known_nominal_and_overall_best(tmp_path) -> None:
    study = create_production_study()
    configuration = _configuration(
        study,
        AxSettings(initialization_trials=1, search_trials=1, seed=5),
    )
    registry = EvaluationRegistry(tmp_path / "registry.json")
    nominal_parameters = {
        "flat_pad_height": 5.0,
        "stem_width": 7.6,
        "stem_height": 6.0,
        "void_width": 1.0,
    }
    historical_best_parameters = {
        **nominal_parameters,
        "flat_pad_height": 5.4,
    }
    for parameters, value, trial in (
        (nominal_parameters, 0.4, 0),
        (historical_best_parameters, 0.9, 8),
    ):
        registry.register(
            PRODUCTION_EVALUATION_CONTRACT_ID,
            parameters,
            status="success",
            first_trial_index=trial,
            first_campaign_id="old-campaign",
            result_artifact_path="output/old-campaign/checkpoint.json",
            minimum_auc=value,
            failure_category=None,
            failure_message=None,
            failure_scenario=None,
            evaluation_wall_time_seconds=1.0,
        )

    nominal = _record(0, status="duplicate_skipped", value=None)
    nominal["phase"] = "nominal"
    nominal["parameters"] = nominal_parameters
    current_best = _record(1, value=0.6)
    state = {
        "status": "COMPLETE",
        "configuration": configuration,
        "records": [nominal, current_best],
        "ax_proposal_count": 1,
        "new_evaluation_count": 1,
        "duplicate_proposal_count": 0,
        "unique_success_count": 1,
        "unique_failure_count": 0,
    }

    _write_summary(
        tmp_path,
        state,
        total_wall_time_seconds=2.0,
        evaluation_registry=registry,
    )
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Nominal baseline minimum_auc: 0.4" in summary
    assert "Campaign new best minimum_auc: 0.6" in summary
    assert "Overall known best minimum_auc: 0.9" in summary
    assert "Overall known best source: ('old-campaign', 8)" in summary


def _failed_preflight(message: str) -> dict[str, object]:
    return {
        "status": "FAIL",
        "failure_category": "infrastructure_failure",
        "failure_signature": "optix-runtime-initialization",
        "error": message,
    }


@pytest.mark.parametrize(
    ("message", "stage"),
    [
        ("Could not find a valid OptiX include directory", "optix_header_resolution"),
        ("NVRTC compilation failed", "nvrtc_compile"),
    ],
)
def test_optix_preflight_classifies_runtime_initialization_failure(
    monkeypatch,
    message: str,
    stage: str,
) -> None:
    def fail_runtime():
        raise ProductionOptixSmokeError(stage, message)

    monkeypatch.setattr(bo_campaign, "run_production_optix_smoke", fail_runtime)

    result = bo_campaign._optix_preflight()

    assert result["status"] == "FAIL"
    assert result["failure_category"] == "infrastructure_failure"
    assert result["failure_signature"] == "optix-runtime-initialization"
    assert result["failure_stage"] == stage
    assert message in result["error"]


@pytest.mark.parametrize(
    "message",
    [
        "Transport3DDependencyError: Could not find a valid OptiX include directory",
        "Transport3DDependencyError: OptiX runtime setup failed: NVRTC compilation failed",
    ],
)
def test_campaign_preflight_fails_before_ax_evaluator_or_registry(
    monkeypatch,
    tmp_path,
    message: str,
) -> None:
    events: list[str] = []
    output = tmp_path / "campaign"
    registry_path = tmp_path / "registry.json"

    monkeypatch.setattr(
        bo_campaign,
        "_optix_preflight",
        lambda: _failed_preflight(message),
    )

    def unexpected_ax(*args, **kwargs):
        events.append("ax")
        raise AssertionError("Ax must not be created after preflight failure")

    monkeypatch.setattr(bo_campaign, "run_ax_optimization", unexpected_ax)

    with pytest.raises(CampaignInfrastructureError):
        bo_campaign.run_campaign(
            output,
            registry_path=registry_path,
            historical_checkpoints=(),
        )

    assert events == []
    checkpoint = bo_campaign.strict_read_json(output / "checkpoint.json")
    assert checkpoint["status"] == "ERROR"
    assert checkpoint["failure_category"] == "infrastructure_failure"
    assert checkpoint["records"] == []
    assert not (output / "ax_client.json").exists()
    assert not registry_path.exists()


def test_successful_preflight_reaches_ax_orchestration(monkeypatch, tmp_path) -> None:
    events: list[str] = []
    output = tmp_path / "campaign"

    def pass_preflight() -> dict[str, object]:
        events.append("preflight")
        return {
            "status": "PASS",
            "failure_category": None,
            "failure_signature": None,
            "error": None,
            "runtime_metadata": {"synthetic": True},
        }

    record = AxTrialRecord(
        trial_index=0,
        phase="nominal",
        parameters=NOMINAL_PARAMETERS,
        evaluation=None,
        failure_message="synthetic orchestration stub",
    )

    class _SnapshotClient:
        def _to_json_snapshot(self):
            return {"synthetic": True}

    def fake_ax(*args, **kwargs):
        events.append("ax")
        kwargs["on_record"](_SnapshotClient(), (record,))
        return AxRunResult(records=(record,))

    monkeypatch.setattr(bo_campaign, "_optix_preflight", pass_preflight)
    monkeypatch.setattr(bo_campaign, "run_ax_optimization", fake_ax)

    result = bo_campaign.run_campaign(
        output,
        registry_path=tmp_path / "registry.json",
        historical_checkpoints=(),
    )

    assert events == ["preflight", "ax"]
    assert result.records == (record,)
    checkpoint = bo_campaign.strict_read_json(output / "checkpoint.json")
    assert checkpoint["status"] == "COMPLETE"
    assert checkpoint["optix_preflight"]["status"] == "PASS"


def test_cleanup_removes_only_checkpoint_proven_infrastructure_records(tmp_path) -> None:
    registry_path = tmp_path / "registry.json"
    checkpoint_path = tmp_path / "aborted" / "checkpoint.json"
    backup_path = tmp_path / "registry.before-cleanup.json"
    registry = EvaluationRegistry(registry_path)

    contaminated = {
        **NOMINAL_PARAMETERS,
        "flat_pad_height": 5.1,
    }
    historical_success = {
        **NOMINAL_PARAMETERS,
        "flat_pad_height": 5.2,
    }
    historical_trace_failure = {
        **NOMINAL_PARAMETERS,
        "flat_pad_height": 5.3,
    }

    registry.register(
        PRODUCTION_EVALUATION_CONTRACT_ID,
        contaminated,
        status="optics_failure",
        first_trial_index=101,
        first_campaign_id="production_bo_20260818_02",
        result_artifact_path=str(checkpoint_path.resolve()),
        minimum_auc=None,
        failure_category="optics_failure",
        failure_message=(
            "Transport3DDependencyError: "
            f"{ABORTED_OPTIX_HEADER_FAILURE_SIGNATURE}"
        ),
        failure_scenario=None,
        evaluation_wall_time_seconds=1.5,
    )
    registry.register(
        PRODUCTION_EVALUATION_CONTRACT_ID,
        historical_success,
        status="success",
        first_trial_index=2,
        first_campaign_id="production_bo_20260818",
        result_artifact_path="output/old/checkpoint.json",
        minimum_auc=0.7,
        failure_category=None,
        failure_message=None,
        failure_scenario=None,
        evaluation_wall_time_seconds=10.0,
    )
    registry.register(
        PRODUCTION_EVALUATION_CONTRACT_ID,
        historical_trace_failure,
        status="optics_failure",
        first_trial_index=3,
        first_campaign_id="production_bo_20260818",
        result_artifact_path="output/old/checkpoint.json",
        minimum_auc=None,
        failure_category="optics_failure",
        failure_message="Transport3DTraceError: active branch has no physical hit",
        failure_scenario="candidate-specific transport",
        evaluation_wall_time_seconds=11.0,
    )
    contaminated_key = registry.lookup(
        PRODUCTION_EVALUATION_CONTRACT_ID,
        contaminated,
    ).key

    atomic_write_json(
        checkpoint_path,
        {
            "records": [
                {
                    "registry_key": contaminated_key,
                    "phase": "initialization",
                    "status": "optics_failure",
                    "failure_message": (
                        "Transport3DDependencyError: "
                        f"{ABORTED_OPTIX_HEADER_FAILURE_SIGNATURE}"
                    ),
                    "evaluation": {
                        "fem_trajectories_attempted": 0,
                        "captured_states_attempted": 0,
                    },
                }
            ]
        },
    )

    report = cleanup_aborted_infrastructure_records(
        registry_path,
        checkpoint_path,
        campaign_id="production_bo_20260818_02",
        backup_path=backup_path,
    )

    assert report.before_count == 3
    assert report.removed_count == 1
    assert report.after_count == 2
    assert report.retained_statuses == ("optics_failure", "success")
    assert report.retained_campaign_ids == (
        "production_bo_20260818",
    )
    assert backup_path.exists()
    reloaded = EvaluationRegistry(registry_path)
    assert reloaded.lookup(PRODUCTION_EVALUATION_CONTRACT_ID, contaminated) is None
    assert reloaded.lookup(PRODUCTION_EVALUATION_CONTRACT_ID, historical_success)
    assert reloaded.lookup(PRODUCTION_EVALUATION_CONTRACT_ID, historical_trace_failure)
