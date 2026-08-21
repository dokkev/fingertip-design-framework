"""Exact-key and persistence contracts for production evaluation reuse."""

from __future__ import annotations

import json

import pytest

from lumo.optimization.evaluation_registry import (
    EvaluationRegistry,
    EvaluationRegistryLockError,
    canonical_morphology,
    evaluation_key,
    evaluation_registry_writer_lock,
)
from lumo.optimization.objectives import ObjectiveIdentifier


CONTRACT = "production-contract-a"
MORPHOLOGY = {
    "flat_pad_height": 5.0,
    "semielliptical_pad_height": 9.0,
    "stem_width": 7.6,
    "stem_height": 6.0,
    "void_width": 1.0,
    "void_height": 0.25,
}
OBJECTIVE = ObjectiveIdentifier("trajectory_separation", 1)


def _register(
    registry: EvaluationRegistry,
    parameters: dict[str, float],
    *,
    status: str,
    trial_index: int,
    objective_value: float | None = None,
):
    return registry.register(
        CONTRACT,
        parameters,
        status=status,
        first_trial_index=trial_index,
        first_campaign_id="campaign-a",
        result_artifact_path="output/campaign-a/checkpoint.json",
        objective=OBJECTIVE,
        failure_category=None if status == "success" else status,
        failure_message=None if status == "success" else "synthetic failure",
        failure_scenario=None,
        evaluation_wall_time_seconds=3.5,
        objective_value=objective_value,
        producer_source={"source_id": "source-a", "git_commit": "abc"},
    )


def test_registry_persists_success_failure_and_duplicate_provenance(tmp_path) -> None:
    path = tmp_path / "evaluation_registry.json"
    registry = EvaluationRegistry(path)
    success = _register(
        registry,
        MORPHOLOGY,
        status="success",
        trial_index=2,
        objective_value=0.42,
    )
    failed_morphology = {**MORPHOLOGY, "stem_height": 6.25}
    failed = _register(
        registry,
        failed_morphology,
        status="optics_failure",
        trial_index=7,
    )
    registry.note_duplicate(success, trial_index=11, campaign_id="campaign-b")

    reloaded = EvaluationRegistry(path)
    reloaded_success = reloaded.lookup(CONTRACT, MORPHOLOGY)
    reloaded_failure = reloaded.lookup(CONTRACT, failed_morphology)
    assert reloaded_success is not None
    assert reloaded_success.status == "success"
    assert reloaded_success.objective == OBJECTIVE
    assert reloaded_success.objective_value == 0.42
    assert reloaded_success.producer_source_id == "source-a"
    assert reloaded_success.duplicate_count == 1
    assert reloaded_success.last_duplicate_trial_index == 11
    assert reloaded_failure is not None
    assert reloaded_failure.status == "optics_failure"
    assert reloaded_failure.objective_value is None
    assert {record.key for record in reloaded.records_for_contract(CONTRACT)} == {
        success.key,
        failed.key,
    }
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 4
    assert reloaded.source_audit(CONTRACT, "source-a") == {
        "same_source": 2,
        "different_source": 0,
        "unknown_source": 0,
    }
    assert reloaded.source_audit(CONTRACT, "source-b") == {
        "same_source": 0,
        "different_source": 2,
        "unknown_source": 0,
    }


def test_exact_key_uses_lossless_float_hex_without_nearby_deduplication(
    tmp_path,
) -> None:
    first = {**MORPHOLOGY, "void_width": 1.000000000000001}
    nearby = {**MORPHOLOGY, "void_width": 1.000000000000002}
    registry = EvaluationRegistry(tmp_path / "registry.json")
    _register(
        registry,
        first,
        status="mechanics_failure",
        trial_index=0,
    )

    assert canonical_morphology(first)["void_width"] == first["void_width"].hex()
    assert evaluation_key(CONTRACT, first) != evaluation_key(CONTRACT, nearby)
    assert registry.lookup(CONTRACT, first) is not None
    assert registry.lookup(CONTRACT, nearby) is None
    assert registry.lookup("production-contract-b", first) is None


def test_registry_never_overwrites_an_existing_exact_result(tmp_path) -> None:
    registry = EvaluationRegistry(tmp_path / "registry.json")
    _register(
        registry,
        MORPHOLOGY,
        status="success",
        trial_index=1,
        objective_value=0.5,
    )
    with pytest.raises(KeyError, match="already registered"):
        _register(
            registry,
            MORPHOLOGY,
            status="mechanics_failure",
            trial_index=2,
        )


def test_registry_persists_negative_objective_value(tmp_path) -> None:
    path = tmp_path / "registry.json"
    registry = EvaluationRegistry(path)
    record = _register(
        registry,
        MORPHOLOGY,
        status="success",
        trial_index=3,
        objective_value=-0.1,
    )

    assert record.objective_value == -0.1
    reloaded = EvaluationRegistry(path).lookup(CONTRACT, MORPHOLOGY)
    assert reloaded is not None
    assert reloaded.objective_value == -0.1


@pytest.mark.parametrize(
    ("status", "objective_value", "message"),
    [
        ("success", None, "requires objective_value"),
        ("mechanics_failure", 0.2, "must not carry objective_value"),
        ("unknown", None, "unsupported registry status"),
    ],
)
def test_registry_validates_scientific_status_payloads(
    tmp_path,
    status,
    objective_value,
    message,
) -> None:
    registry = EvaluationRegistry(tmp_path / "registry.json")
    with pytest.raises(ValueError, match=message):
        _register(
            registry,
            MORPHOLOGY,
            status=status,
            trial_index=0,
            objective_value=objective_value,
        )


def test_registry_writer_lock_rejects_a_concurrent_campaign(tmp_path) -> None:
    path = tmp_path / "shared-registry.json"

    with evaluation_registry_writer_lock(path):
        with pytest.raises(EvaluationRegistryLockError, match="already locked"):
            with evaluation_registry_writer_lock(path):
                pytest.fail("a second campaign acquired the shared registry")
