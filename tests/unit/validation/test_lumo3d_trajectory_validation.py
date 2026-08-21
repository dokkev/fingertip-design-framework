from validation.optimization.lumo3d_trajectory_validation import (
    _exact_mechanics_arrays,
    _trajectory_hard_checks,
)

import numpy as np


def test_current_trajectory_gate_excludes_historical_cross_contract_result() -> None:
    assert _trajectory_hard_checks(
        direct_path_equivalence_pass=True,
        domain_check_pass=True,
        no_objective_pathology=True,
    )


def test_current_trajectory_gate_requires_same_contract_direct_path() -> None:
    assert not _trajectory_hard_checks(
        direct_path_equivalence_pass=False,
        domain_check_pass=True,
        no_objective_pathology=True,
    )


def test_same_contract_direct_path_requires_bit_exact_arrays() -> None:
    arrays = {
        "rest_vertices_mm": np.zeros((2, 3), dtype=np.float32),
        "deformed_vertices_mm": np.ones((2, 3), dtype=np.float32),
        "tetrahedra": np.asarray([[0, 1, 1, 0]], dtype=np.int32),
        "source_node_ids": np.asarray([0, 1], dtype=np.int64),
        "carrier_contact_vertex_indices": np.asarray([1], dtype=np.int64),
        "carrier_contact_source_node_ids": np.asarray([1], dtype=np.int64),
        "surface_outer": np.asarray([[0, 1, 1]], dtype=np.int32),
    }
    assert _exact_mechanics_arrays(arrays, arrays)["pass"]

    changed = {name: np.array(value, copy=True) for name, value in arrays.items()}
    changed["deformed_vertices_mm"][0, 0] = np.nextafter(
        changed["deformed_vertices_mm"][0, 0],
        np.float32(2.0),
    )
    result = _exact_mechanics_arrays(arrays, changed)
    assert not result["pass"]
    assert not result["arrays"]["deformed_vertices_mm"]

    missing_contact = {
        name: value
        for name, value in arrays.items()
        if name != "carrier_contact_source_node_ids"
    }
    missing = _exact_mechanics_arrays(arrays, missing_contact)
    assert not missing["array_key_sets_match"]
    assert not missing["pass"]
