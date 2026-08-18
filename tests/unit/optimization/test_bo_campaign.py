"""Cheap campaign persistence and reporting-policy contracts."""

from __future__ import annotations

from validation.common.io import atomic_write_json
from validation.optimization.bo_campaign import (
    _configuration,
    _import_historical_checkpoint,
    _plateau_assessment,
)
from optimization.ax_adapter import AxSettings
from optimization.evaluation_registry import EvaluationRegistry
from optimization.study import (
    PRODUCTION_EVALUATION_CONTRACT_ID,
    create_production_study,
)


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
