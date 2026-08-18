"""Unit contracts for Phase 4I-D configuration and sparse diagnostics."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from case.fea2d import FEA2D
from fem.indentation import (
    InvalidIndentationSettings,
    _resolve_bonded_bottom,
    run_indentation_case,
)
from fem.kratos_settings import (
    build_project_parameters_data,
    build_indentation_project_parameters_data,
    indentation_contact_groups,
    validate_internal_contact_configuration,
)
from fem.solve import solve
from validation.fingertip.internal_contact.sparse import analyze_sparse_system


@pytest.mark.parametrize(
    ("configuration", "pair_count"),
    (
        ("none", 1),
        ("bottom_only", 2),
        ("sides_separate", 3),
        ("three_pairs", 4),
        ("continuous_u", 2),
    ),
)
def test_contact_configuration_expected_pair_count(
    configuration: str, pair_count: int
) -> None:
    groups = indentation_contact_groups(configuration)
    assert len(groups) == pair_count
    assert groups[0] == (
        "external_pad_indenter",
        "PadOuterArc",
        "IndenterContactArc",
    )
    data = build_indentation_project_parameters_data(1, configuration)
    process = data["processes"]["contact_process_list"][0]["Parameters"]
    assert len(process["contact_model_part"]) == pair_count
    assert len(process["assume_master_slave"]) == pair_count


def test_invalid_contact_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported internal contact"):
        validate_internal_contact_configuration("invented_pair")


def test_production_defaults_use_bonded_bottom_and_side_contacts() -> None:
    expected = (
        ("external_pad_indenter", "PadOuterArc", "IndenterContactArc"),
        ("internal_left", "PadCutoutLeft", "StemLeft"),
        ("internal_right", "PadCutoutRight", "StemRight"),
    )
    assert indentation_contact_groups() == expected
    data = build_indentation_project_parameters_data(1)
    contact_model_part = data["processes"]["contact_process_list"][0]["Parameters"][
        "contact_model_part"
    ]
    assert len(contact_model_part) == 3
    assert all("PadCutoutBottom" not in pair for pair in contact_model_part.values())
    initialization_data = build_project_parameters_data()
    initialization_pairs = initialization_data["processes"][
        "contact_process_list"
    ][0]["Parameters"]["contact_model_part"]
    assert len(initialization_pairs) == 3
    assert all("PadCutoutBottom" not in pair for pair in initialization_pairs.values())
    assert inspect.signature(FEA2D).parameters["internal_contact"].default == (
        "sides_separate"
    )
    assert inspect.signature(solve).parameters["internal_contact"].default == (
        "sides_separate"
    )
    assert inspect.signature(run_indentation_case).parameters[
        "internal_contact_configuration"
    ].default == "sides_separate"


def test_bonded_bottom_requires_zero_height_and_is_not_a_diagnostic_contact() -> None:
    zero_gap_mesh = SimpleNamespace(parameters=SimpleNamespace(void_height=0.0))
    finite_gap_mesh = SimpleNamespace(parameters=SimpleNamespace(void_height=0.25))

    assert _resolve_bonded_bottom(zero_gap_mesh, "sides_separate")
    assert not _resolve_bonded_bottom(zero_gap_mesh, "three_pairs")
    assert not _resolve_bonded_bottom(finite_gap_mesh, "three_pairs")
    with pytest.raises(InvalidIndentationSettings, match="void_height=0.0"):
        _resolve_bonded_bottom(finite_gap_mesh, "sides_separate")


def test_synthetic_zero_row_maps_to_node_and_dof() -> None:
    matrix = sparse.csr_matrix(
        np.asarray(
            [
                [2.0, -1.0, 0.0],
                [-1.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
    )
    equation_map = {
        0: {"node_id": 1, "variable": "DISPLACEMENT_X"},
        1: {"node_id": 1, "variable": "DISPLACEMENT_Y"},
        2: {
            "node_id": 9,
            "variable": "LAGRANGE_MULTIPLIER_CONTACT_PRESSURE",
        },
    }
    result = analyze_sparse_system(
        matrix, [0.0, 1.0, 0.0], equation_map
    )
    assert result["exact_zero_row_count"] == 1
    assert result["near_zero_row_count"] == 1
    assert result["near_zero_rows"][0]["equation_id"] == 2
    assert result["near_zero_rows"][0]["node_id"] == 9
    assert (
        result["near_zero_rows"][0]["variable"]
        == "LAGRANGE_MULTIPLIER_CONTACT_PRESSURE"
    )
    assert not result["sparse_factorization"]["succeeded"]
