from __future__ import annotations

from validation.physics.vbd_fea_optical_trend import (
    artifact_contract_is_exact,
    rank_statistics,
    selection_summary,
)
from optics.transport3d import fingerprint_mapping


def _rows() -> list[dict[str, object]]:
    return [
        {"morphology_id": "a", "J_VBD": 0.90, "J_FEA": 0.80},
        {"morphology_id": "b", "J_VBD": 0.70, "J_FEA": 0.95},
        {"morphology_id": "c", "J_VBD": 0.50, "J_FEA": 0.60},
        {"morphology_id": "d", "J_VBD": 0.30, "J_FEA": 0.20},
        {"morphology_id": "e", "J_VBD": 0.10, "J_FEA": 0.10},
        {"morphology_id": "f", "J_VBD": 0.05, "J_FEA": 0.15},
        {"morphology_id": "g", "J_VBD": 0.01, "J_FEA": 0.05},
    ]


def test_rank_statistics_uses_maximize_direction_and_meaningful_top_k() -> None:
    result = rank_statistics(_rows(), direction="maximize")

    assert result["n"] == 7
    assert result["vbd_order"][0] == "a"
    assert result["fea_order"][0] == "b"
    assert result["top_k"]["top_3"]["intersection_count"] == 3
    assert result["top_k"]["top_5"]["k"] == 5
    assert result["pairwise_ordering_agreement"] == 19 / 21


def test_selection_summary_is_sign_safe_for_minimize_objective() -> None:
    rows = [
        {"morphology_id": "a", "J_VBD": 1.0, "J_FEA": 3.0},
        {"morphology_id": "b", "J_VBD": 2.0, "J_FEA": 1.0},
        {"morphology_id": "c", "J_VBD": 3.0, "J_FEA": 2.0},
    ]

    result = selection_summary(rows, direction="minimize")

    assert result["vbd_best_morphology"] == "a"
    assert result["fea_best_morphology"] == "b"
    assert result["fea_rank_of_vbd_best"] == 3.0
    assert result["fea_regret"] == 2.0
    assert result["normalized_fea_regret"] == 1.0


def test_reuse_contract_requires_exact_fingerprint_and_payload() -> None:
    contract = {"schema": "comparison-v1", "branch": "FEA", "ray_count": 1024}
    metadata = {
        "schema": "unified-optix-transport-case-v3",
        "contract": contract,
        "contract_fingerprint": fingerprint_mapping(contract),
    }

    assert artifact_contract_is_exact(metadata, contract)
    assert not artifact_contract_is_exact(
        {**metadata, "contract_fingerprint": "wrong"}, contract
    )
    assert not artifact_contract_is_exact(
        {**metadata, "contract": {**contract, "ray_count": 2048}}, contract
    )
