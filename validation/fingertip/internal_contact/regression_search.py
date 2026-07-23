"""Phase 4I-F projection, search-pair, and crosspoint contracts."""

from __future__ import annotations

import json

import pytest

from fem.indentation_analysis import set_indenter_travel
from fem.internal_contact_diagnostic import (
    FIRST_STEP_TRAVEL_MM,
    _build_context,
    _contact_condition_records,
)
from fem.kratos_adapter import _import_kratos
from fem.search_crosspoint_audit import (
    CAUSAL_VARIANTS,
    endpoint_pair_records,
    endpoint_snapshot,
    line2_projection_local_domain,
    unavailable_case_records,
)
from fem.right_side_audit import _endpoint_id


@pytest.mark.parametrize(
    ("point", "inside", "local_coordinate"),
    (
        ((0.0, 0.0), True, -1.0),
        ((0.5, 0.0), True, 0.0),
        ((1.0, 0.0), True, 1.0),
        ((2.0, 0.0), False, 3.0),
    ),
)
def test_line2_projection_uses_actual_local_domain(
    point: tuple[float, float],
    inside: bool,
    local_coordinate: float,
) -> None:
    result = line2_projection_local_domain(
        point, (0.0, 0.0), (1.0, 0.0)
    )
    assert result["inside_local_domain"] is inside
    assert result["local_coordinate"] == pytest.approx(local_coordinate)
    assert result["local_domain"] == [-1.0, 1.0]


@pytest.mark.parametrize("scale", (0.1, 1.0, 7.5))
def test_out_of_domain_detection_is_mesh_scale_and_id_independent(
    scale: float,
) -> None:
    result = line2_projection_local_domain(
        (2.0 * scale, 8.0),
        (0.0, 8.0),
        (scale, 8.0),
    )
    assert result["segment_fraction"] == pytest.approx(2.0)
    assert result["local_coordinate"] == pytest.approx(3.0)
    assert not result["inside_local_domain"]


def test_diagnostic_variants_do_not_change_production_configuration() -> None:
    assert CAUSAL_VARIANTS["F00"].side == "right"
    assert not CAUSAL_VARIANTS["F00"].force_invalid_pair_inactive
    assert CAUSAL_VARIANTS["F02"].force_invalid_pair_inactive
    assert unavailable_case_records()["F01"][
        "production_configuration_modified"
    ] is False


@pytest.fixture(scope="module")
def right_search_context():
    context = _build_context("medium", "right_only")
    solver = context.analysis._GetSolver()
    context.analysis.time = solver.AdvanceInTime(context.analysis.time)
    set_indenter_travel(
        context.model_part,
        context.indenter_topology.node_ids,
        context.fixture,
        FIRST_STEP_TRAVEL_MM,
    )
    context.analysis.ApplyBoundaryConditions()
    context.analysis.ChangeMaterialProperties()
    solver.InitializeSolutionStep()
    solver.Predict()
    try:
        yield context
    finally:
        context.analysis.FinalizeSolutionStep()
        context.analysis.Finalize()


def test_runtime_search_identifies_invalid_and_preserves_valid_endpoint_pair(
    right_search_context,
) -> None:
    endpoint_id = _endpoint_id(
        right_search_context.model_part, "right", slave=True
    )
    pairs = endpoint_pair_records(right_search_context, 1, endpoint_id)
    assert len(pairs) == 2
    assert sum(pair["valid_endpoint_pair"] for pair in pairs) == 1
    assert sum(pair["out_of_domain_extra_pair"] for pair in pairs) == 1
    invalid = next(
        pair for pair in pairs if pair["out_of_domain_extra_pair"]
    )
    assert invalid["endpoint_projection"]["local_coordinate"] == pytest.approx(
        3.0
    )
    assert invalid["exact_overlap"]["overlap_length_mm"] < 1.0e-12


def test_endpoint_snapshot_is_json_serializable_and_identifies_crosspoint(
    right_search_context,
) -> None:
    snapshot = endpoint_snapshot(
        right_search_context,
        "right",
        "test_snapshot",
        None,
    )
    json.dumps(snapshot, allow_nan=False)
    assert snapshot["crosspoint"]["pad_internal_contact_boundary"]
    assert snapshot["crosspoint"]["pad_bond_boundary"]
    assert snapshot["crosspoint"]["dirichlet_boundary"]
    assert snapshot["crosspoint"]["xy_free_primal_dof_count"] == 0
    valid_only = snapshot["local_lm_assembly"][
        "valid_pairs_only_before_dirichlet"
    ]
    assert valid_only["row_norm_free_columns"] < 1.0e-12


def test_runtime_pairs_remain_pure_and_have_unique_generated_ids(
    right_search_context,
) -> None:
    records, summary = _contact_condition_records(right_search_context)
    identifiers = [
        record["generated_condition_id"] for record in records
    ]
    assert summary["all_generated_conditions_pair_pure"]
    assert len(identifiers) == len(set(identifiers))


def test_f02_flag_mutation_preserves_condition_container(
    right_search_context,
) -> None:
    KM, _, _, _ = _import_kratos()
    endpoint_id = _endpoint_id(
        right_search_context.model_part, "right", slave=True
    )
    before = endpoint_pair_records(right_search_context, 1, endpoint_id)
    invalid_id = next(
        record["generated_condition_id"]
        for record in before
        if record["out_of_domain_extra_pair"]
    )
    computing = right_search_context.model[
        "Structure.ComputingContact.ComputingContactSub1"
    ]
    count_before = computing.NumberOfConditions()
    condition = computing.Conditions[invalid_id]
    original_active = bool(condition.Is(KM.ACTIVE))
    condition.Set(KM.ACTIVE, False)
    try:
        after = endpoint_pair_records(
            right_search_context, 1, endpoint_id
        )
        assert computing.NumberOfConditions() == count_before
        invalid = next(
            record
            for record in after
            if record["generated_condition_id"] == invalid_id
        )
        assert not invalid["condition_active"]
        assert any(record["valid_endpoint_pair"] for record in after)
    finally:
        condition.Set(KM.ACTIVE, original_active)
