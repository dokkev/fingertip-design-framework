"""Mechanics identity fields that protect scientific cache semantics."""

from lumo.mechanics_contract import (
    MechanicsContract,
    PRESCRIBED_POSE_ERROR_METRIC_VERSION,
)


def test_pose_metric_version_participates_in_mechanics_identity() -> None:
    assert MechanicsContract().to_dict()[
        "prescribed_pose_error_metric_version"
    ] == PRESCRIBED_POSE_ERROR_METRIC_VERSION
