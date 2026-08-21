"""Mechanics identity fields that protect scientific cache semantics."""

from lumo.mechanics_contract import (
    MechanicsContract,
    PRESCRIBED_POSE_ERROR_METRIC_VERSION,
)
from lumo.physics.contracts import VBDDeterminismMode


def test_pose_metric_version_participates_in_mechanics_identity() -> None:
    contract = MechanicsContract()

    assert contract.to_dict()[
        "prescribed_pose_error_metric_version"
    ] == PRESCRIBED_POSE_ERROR_METRIC_VERSION
    assert contract.to_dict()["deterministic_mode"] == "run_to_run"
    assert contract.deterministic_mode is VBDDeterminismMode.RUN_TO_RUN
