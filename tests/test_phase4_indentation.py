"""Optional Kratos integration checks for the Phase 4I runtime contract."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("KratosMultiphysics")

from fem.indentation_analysis import (
    IndentationSettings,
    inspect_indentation_runtime_contract,
    run_indentation_case,
)
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


@pytest.fixture(scope="module")
def runtime_preflight():
    return inspect_indentation_runtime_contract(
        FingertipModel(FingertipParameters()),
        "medium",
        IndentationSettings(0.25, 48),
    )


def test_external_contact_initialization_and_node_separation(
    runtime_preflight,
) -> None:
    assert runtime_preflight["status"] == "PASS"
    assert runtime_preflight["pad_indenter_node_ids_disjoint"]
    assert runtime_preflight["strategy_check"] == 0
    external = runtime_preflight["runtime_contact_contract"]["groups"][
        "external_pad_indenter"
    ]
    assert external["slave"] == "PadOuterArc"
    assert external["master"] == "IndenterContactArc"
    assert external["slave_node_flags"]["SLAVE"] == external["slave_node_count"]
    assert external["master_node_flags"]["MASTER"] == external["master_node_count"]


def test_internal_three_pair_runtime_contract_is_preserved(
    runtime_preflight,
) -> None:
    groups = runtime_preflight["runtime_contact_contract"]["groups"]
    for name in ("internal_left", "internal_right", "internal_bottom"):
        assert all(groups[name]["checks"].values())
        assert groups[name]["contact_submodelpart_condition_ids"]


@pytest.fixture(scope="module")
def separated_internal_gap_small_solve():
    # This diagnostic isolates external ALM solve wiring from the default
    # zero-clearance internal-contact rank deficiency found by the Phase 4I
    # trial.  It is not the accepted Phase 4I physical baseline.
    result, _ = run_indentation_case(
        FingertipModel(
            FingertipParameters(void_width=0.2, void_height=0.2)
        ),
        "medium",
        IndentationSettings(0.01, 1),
    )
    return result


def test_small_external_indentation_solves_with_finite_fields(
    separated_internal_gap_small_solve,
) -> None:
    result = separated_internal_gap_small_solve
    assert result["solve_status"] == "PASS", result.get("exception")
    point = result["history"][-1]
    assert point["solver_converged"]
    assert point["active_set_converged"]
    assert point["finite_fields"]
    assert point["contact_groups"]["external_pad_indenter"][
        "active_condition_count"
    ] > 0
    assert math.isfinite(point["indenter_normal_reaction_n"])
    assert abs(point["indenter_signed_reaction_along_loading_n"]) > 0.0
    assert point["pad_strain_det_f"]["det_f"]["min"] > 0.0
