"""Phase 4I-G rule and Phase 4J checkpoint contracts."""

from __future__ import annotations

import json

import pytest

from fem.crosspoint_multiplier_treatment import (
    CrosspointRuleInput,
    contact_coupled_free_primal_dof_count,
    fully_prescribed_contact_crosspoint,
)
from fem.no_void_baseline import (
    append_or_replace_case,
    atomic_write_json,
    completed_case_result,
    no_void_geometry_contract,
    reaction_work_proxy,
)
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


@pytest.mark.parametrize("node_id", (2, 5, 101, 987654))
def test_crosspoint_rule_is_node_id_and_side_independent(node_id: int) -> None:
    record = CrosspointRuleInput(
        node_id=node_id,
        displacement_x_fixed=True,
        displacement_y_fixed=True,
        incident_active_contact_condition_count=1,
    )
    assert fully_prescribed_contact_crosspoint(record)
    assert contact_coupled_free_primal_dof_count(True, True) == 0


def test_crosspoint_requires_active_contact_and_full_fixity() -> None:
    assert not fully_prescribed_contact_crosspoint(
        CrosspointRuleInput(1, True, True, 0)
    )
    assert not fully_prescribed_contact_crosspoint(
        CrosspointRuleInput(1, True, False, 1)
    )
    assert contact_coupled_free_primal_dof_count(True, False) == 1
    assert contact_coupled_free_primal_dof_count(False, False) == 2


def test_default_model_is_explicit_no_void_reference() -> None:
    contract = no_void_geometry_contract(
        FingertipModel(FingertipParameters())
    )
    assert contract["pass"]
    assert contract["void_classification"] == "zero_clearance_fit"
    assert contract["void_geometry_is_none"]
    assert contract["internal_contact_configuration"] == "none"


def test_atomic_checkpoint_and_resume_validation(tmp_path) -> None:
    path = tmp_path / "result.json"
    value = {
        "phase": "4J",
        "case_name": "J0",
        "status": "PASS",
        "finite": True,
        "history": [{"step": 1}],
    }
    atomic_write_json(path, value)
    (tmp_path / "history.csv").write_text(
        "step,status\n1,PASS\n", encoding="utf-8"
    )
    assert json.loads(path.read_text(encoding="utf-8")) == value
    assert completed_case_result(path, "J0") == value
    assert completed_case_result(path, "J1") is None
    assert not list(tmp_path.glob(".*.tmp"))


def test_failed_case_does_not_discard_later_queue_checkpoint() -> None:
    state: dict[str, object] = {"phase": "4J", "cases": []}
    append_or_replace_case(
        state, {"case_name": "J0", "status": "FAIL"}
    )
    append_or_replace_case(
        state, {"case_name": "J1", "status": "PASS"}
    )
    assert [
        record["status"] for record in state["cases"]  # type: ignore[index]
    ] == ["FAIL", "PASS"]


def test_reaction_work_proxy_is_not_labeled_strain_energy() -> None:
    result = reaction_work_proxy(
        [
            {
                "achieved_indentation_mm": 1.0,
                "indenter_normal_reaction_n": 2.0,
            },
            {
                "achieved_indentation_mm": 2.0,
                "indenter_normal_reaction_n": 4.0,
            },
        ]
    )
    assert result["value_n_mm"] == pytest.approx(4.0)
    assert "not reported as" in result["interpretation"]
