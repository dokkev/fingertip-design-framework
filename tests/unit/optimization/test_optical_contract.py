from __future__ import annotations

from types import SimpleNamespace

import pytest

from lumo.optimization.optical_contract import (
    DEFAULT_OPTICAL_NUMERICAL_ACCEPTANCE,
    OpticalNumericalAcceptanceContract,
    summarize_optical_failure_diagnostics,
)


def _result(**overrides: object) -> SimpleNamespace:
    values = {
        "launched_weight": 1.0,
        "segment_budget_termination_count": 0,
        "segment_budget_termination_weight": 0.0,
        "processed_sample_count": 10,
        "clipped_sample_count": 0,
        "represented_weighted_path_length_mm": 2.0,
        "clipped_weighted_path_length_mm": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_optical_numerical_contract_accepts_clean_state() -> None:
    assessment = DEFAULT_OPTICAL_NUMERICAL_ACCEPTANCE.assess(_result())

    assert assessment.accepted is True
    assert assessment.segment_budget_termination_fraction == pytest.approx(0.0)
    assert assessment.clipped_weighted_path_length_mm == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    (
        ({"segment_budget_termination_count": 1}, "segment_budget_termination"),
        ({"clipped_sample_count": 1}, "path_field_clipping"),
        ({"clipped_weighted_path_length_mm": 1.0e-12}, "path_field_clipping"),
    ),
)
def test_default_optical_numerical_contract_rejects_hard_failures(
    overrides: dict[str, object],
    reason: str,
) -> None:
    assessment = DEFAULT_OPTICAL_NUMERICAL_ACCEPTANCE.assess(_result(**overrides))

    assert assessment.accepted is False
    assert reason in assessment.failure_reasons


def test_objective_pathology_cannot_be_disabled_in_production_contract() -> None:
    rejected = DEFAULT_OPTICAL_NUMERICAL_ACCEPTANCE.assess(
        _result(), objective_pathology=True
    )

    assert rejected.accepted is False
    assert "objective_pathology" in rejected.failure_reasons
    with pytest.raises(ValueError, match="always rejects"):
        OpticalNumericalAcceptanceContract(reject_objective_pathology=False)


def test_nonzero_segment_budget_weight_is_rejected_even_when_count_is_zero() -> None:
    assessment = DEFAULT_OPTICAL_NUMERICAL_ACCEPTANCE.assess(
        _result(segment_budget_termination_weight=1.0e-12)
    )

    assert assessment.accepted is False
    assert assessment.failure_reasons == ("segment_budget_termination",)


def test_candidate_summary_aggregates_termination_and_clipping() -> None:
    summary = DEFAULT_OPTICAL_NUMERICAL_ACCEPTANCE.summarize(
        (
            _result(),
            _result(
                segment_budget_termination_count=1,
                segment_budget_termination_weight=0.25,
                clipped_sample_count=2,
                clipped_weighted_path_length_mm=0.5,
            ),
        )
    )

    assert summary["state_count"] == 2
    assert summary["failure_count"] == 1
    assert summary["segment_budget_termination_count"] == 1
    assert summary["clipped_sample_count"] == 2
    assert summary["clipped_weighted_path_length_mm"] == pytest.approx(0.5)


def test_campaign_summary_aggregates_bounded_optical_failure_evidence() -> None:
    summary = summarize_optical_failure_diagnostics(
        (
            {
                "status": "optics_failure",
                "failure_scenario": "numerical_acceptance",
                "failure_diagnostics": {
                    "optical_numerical_summary": {
                        "failure_count": 2,
                        "failure_reasons": [
                            "path_field_clipping",
                            "segment_budget_termination",
                        ],
                        "segment_budget_termination_count": 3,
                        "segment_budget_termination_weight": 0.25,
                        "clipped_sample_count": 4,
                        "represented_weighted_path_length_mm": 5.0,
                        "clipped_weighted_path_length_mm": 0.5,
                    }
                },
            },
            {
                "status": "optics_failure",
                "failure_scenario": "candidate_optics_geometry",
                "failure_diagnostics": {
                    "cause_type": "Transport3DCandidateGeometryError"
                },
            },
        )
    )

    assert summary["optics_failure_candidate_count"] == 2
    assert summary["path_field_clipping_candidate_count"] == 1
    assert summary["segment_budget_termination_candidate_count"] == 1
    assert summary["optical_failure_state_count"] == 2
    assert summary["segment_budget_termination_count"] == 3
    assert summary["clipped_sample_count"] == 4
    assert summary["cause_type_counts"] == {
        "Transport3DCandidateGeometryError": 1
    }
