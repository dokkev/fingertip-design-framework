"""Kratos integration contracts for Phase 4I-D."""

from __future__ import annotations

import pytest

from fem.fingertip_mesher import generate_fingertip_mesh
from fem.indentation_analysis import (
    IndentationSettings,
    inspect_indentation_runtime_contract,
)
from fem.internal_contact_configuration import (
    PAD_U_AGGREGATE,
    PAD_U_SEGMENTS,
    STEM_U_AGGREGATE,
    STEM_U_SEGMENTS,
    create_continuous_u_submodel_parts,
    u_corner_node_ids,
)
from fem.internal_contact_diagnostic import (
    assemble_first_step_diagnostics,
    run_first_step_case,
)
from fem.kratos_adapter import _import_kratos, populate_kratos_model_part
from fem.mesh_types import mesh_settings_for_level
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


@pytest.fixture(scope="module")
def continuous_u_membership():
    KM, _, _, _ = _import_kratos()
    fingertip_model = FingertipModel(FingertipParameters())
    mesh = generate_fingertip_mesh(
        fingertip_model, mesh_settings_for_level("medium")
    )
    model = KM.Model()
    model_part = model.CreateModelPart("Structure")
    contract_before = populate_kratos_model_part(model_part, mesh)
    root_condition_count = model_part.NumberOfConditions()
    contract = create_continuous_u_submodel_parts(model_part)
    return (
        model_part,
        contract_before,
        contract,
        root_condition_count,
    )


def test_continuous_u_reuses_semantic_membership_without_root_duplicates(
    continuous_u_membership,
) -> None:
    model_part, _, contract, root_condition_count = continuous_u_membership
    assert model_part.NumberOfConditions() == root_condition_count
    assert all(contract["checks"].values())
    for aggregate_name, semantic_names in (
        (PAD_U_AGGREGATE, PAD_U_SEGMENTS),
        (STEM_U_AGGREGATE, STEM_U_SEGMENTS),
    ):
        aggregate = model_part.GetSubModelPart(aggregate_name)
        expected_nodes = {
            node.Id
            for name in semantic_names
            for node in model_part.GetSubModelPart(name).Nodes
        }
        expected_conditions = {
            condition.Id
            for name in semantic_names
            for condition in model_part.GetSubModelPart(name).Conditions
        }
        assert {node.Id for node in aggregate.Nodes} == expected_nodes
        assert {
            condition.Id for condition in aggregate.Conditions
        } == expected_conditions
        for name in semantic_names:
            assert model_part.HasSubModelPart(name)


def test_continuous_u_corner_node_ids_are_unique(
    continuous_u_membership,
) -> None:
    model_part, _, _, _ = continuous_u_membership
    corner_ids = u_corner_node_ids(model_part)
    assert len(corner_ids) == 4
    assert len(set(corner_ids.values())) == 4


@pytest.fixture(scope="module")
def initialized_contracts():
    model = FingertipModel(FingertipParameters())
    settings = IndentationSettings(0.25 / 48.0, 1)
    return {
        configuration: inspect_indentation_runtime_contract(
            model,
            "medium",
            settings,
            internal_contact_configuration=configuration,
        )
        for configuration in (
            "none",
            "bottom_only",
            "three_pairs",
            "continuous_u",
        )
    }


@pytest.mark.parametrize(
    ("configuration", "expected_groups"),
    (
        ("none", {"external_pad_indenter"}),
        (
            "bottom_only",
            {"external_pad_indenter", "internal_bottom"},
        ),
        (
            "three_pairs",
            {
                "external_pad_indenter",
                "internal_left",
                "internal_right",
                "internal_bottom",
            },
        ),
        (
            "continuous_u",
            {"external_pad_indenter", "internal_u"},
        ),
    ),
)
def test_runtime_pair_initialization(
    initialized_contracts,
    configuration: str,
    expected_groups: set[str],
) -> None:
    result = initialized_contracts[configuration]
    assert result["status"] == "PASS"
    groups = result["runtime_contact_contract"]["groups"]
    assert set(groups) == expected_groups
    assert result["runtime_contact_contract"]["all_group_contracts_pass"]


def test_continuous_u_runtime_roles_and_reuse(
    initialized_contracts,
) -> None:
    result = initialized_contracts["continuous_u"]
    internal = result["runtime_contact_contract"]["groups"]["internal_u"]
    assert internal["slave"] == PAD_U_AGGREGATE
    assert internal["master"] == STEM_U_AGGREGATE
    assert internal["checks"]["slave_runtime_role"]
    assert internal["checks"]["master_runtime_role"]
    aggregate = result["continuous_u_aggregate_contract"]
    assert aggregate["checks"]["root_condition_count_unchanged"]
    assert aggregate["checks"]["pad_has_no_duplicate_connectivity"]
    assert aggregate["checks"]["stem_has_no_duplicate_connectivity"]


@pytest.fixture(scope="module")
def continuous_u_assembly():
    return assemble_first_step_diagnostics("E")


def test_continuous_u_computing_contact_pair_purity(
    continuous_u_assembly,
) -> None:
    result, dof_rows, contact_records = continuous_u_assembly
    assert result["status"] == "PASS", result.get("exception")
    assert result["contact_pair_purity"][
        "all_generated_conditions_pair_pure"
    ]
    assert result["contact_pair_purity"][
        "generated_condition_ids_unique_across_processes"
    ]
    assert {record["contact_group"] for record in contact_records} == {
        "external_pad_indenter",
        "internal_u",
    }
    assert dof_rows
    assembled_equations = [
        row["equation_id"] for row in dof_rows if row["assembled"]
    ]
    assert len(assembled_equations) == len(set(assembled_equations))


def test_unused_internal_conditions_are_excluded_from_external_only_case() -> None:
    result, _, _ = assemble_first_step_diagnostics("A")
    unused = result["runtime_contact_contract"][
        "unused_internal_surfaces"
    ]
    assert result["runtime_contact_contract"][
        "all_unused_internal_conditions_excluded"
    ]
    assert all(record["condition_excluded"] for record in unused.values())


def test_case_a_first_step_converges() -> None:
    result = run_first_step_case("A")
    assert result["status"] == "PASS", result["acceptance_checks"]
    assert result["solver_converged"]
    assert result["reaction_n"] > 0.0
    assert result["det_f_min"] > 0.0
