"""Unit contracts for Phase 4I-D configuration and sparse diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from fem.kratos_settings import (
    build_indentation_project_parameters_data,
    indentation_contact_groups,
    validate_internal_contact_configuration,
)
from fem.sparse_diagnostics import analyze_sparse_system


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
