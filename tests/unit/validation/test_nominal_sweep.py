"""Focused bookkeeping tests for the pre-BO morphology sweep."""

from __future__ import annotations

import pytest

from validation.common.io import strict_read_json
from validation.optimization.nominal_sweep import (
    SAMPLE_COUNT,
    SWEPT_RANGES,
    _configuration,
    _decode_parameters,
    _initial_state,
    _load_state,
    _save_state,
    _upsert_candidate,
    sobol_proposals,
)


def test_sobol_proposals_are_exactly_64_and_reproducible() -> None:
    first = sobol_proposals()
    second = sobol_proposals()

    assert len(first) == SAMPLE_COUNT == 64
    assert all(len(point) == len(SWEPT_RANGES) == 6 for point in first)
    assert first == second
    assert all(0.0 <= value < 1.0 for point in first for value in point)


def test_decoding_uses_fixed_width_and_exact_sweep_ranges() -> None:
    parameters = _decode_parameters([0.0] * len(SWEPT_RANGES))
    assert parameters.flat_pad_width == 30.0
    for name, lower, _ in SWEPT_RANGES:
        assert getattr(parameters, name) == lower

    parameters = _decode_parameters([1.0] * len(SWEPT_RANGES))
    for name, _, upper in SWEPT_RANGES:
        assert getattr(parameters, name) == upper


def test_checkpoint_resume_keeps_proposals_and_replaces_one_index(tmp_path) -> None:
    configuration = _configuration()
    proposals = sobol_proposals()
    state = _load_state(tmp_path, configuration, proposals)
    record = {
        "candidate_index": 1,
        "status": "geometry_rejected",
        "failure_category": "geometry_rejected",
    }
    _upsert_candidate(state, record)
    _save_state(tmp_path, state)

    resumed = _load_state(tmp_path, configuration, proposals)
    assert resumed["normalized_sobol_points"] == proposals
    assert resumed["candidates"] == [record]

    replacement = dict(record, status="success", failure_category=None)
    _upsert_candidate(resumed, replacement)
    _save_state(tmp_path, resumed)
    final = strict_read_json(tmp_path / "checkpoint.json")
    assert final["candidates"] == [replacement]


def test_checkpoint_rejects_a_different_proposal_set(tmp_path) -> None:
    configuration = _configuration()
    proposals = sobol_proposals()
    _load_state(tmp_path, configuration, proposals)
    with pytest.raises(ValueError, match="Sobol proposals differ"):
        _load_state(tmp_path, configuration, [list(proposals[0])] + proposals[2:])
