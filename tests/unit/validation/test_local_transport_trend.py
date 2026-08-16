"""Focused contracts for the local intrinsic 2D-to-3D analysis."""

from __future__ import annotations

from validation.optics.local_transport_trend import (
    J2D_TIE_TOLERANCE,
    J3D_PATH_TIE_TOLERANCE,
    LOCAL_HALF_WIDTH,
    POOL_SIZE,
    _pairwise_rank_result,
    _precommit_payload,
    _tie_aware_ranks,
    _verify_precommit,
)


def test_precommit_sampling_is_reproducible_and_bounded() -> None:
    first = _precommit_payload()
    second = _precommit_payload()
    first_pool = [
        (item["sample_id"], item["normalized_local_coordinate"])
        for item in first["ordered_sampling_pool"]
    ]
    second_pool = [
        (item["sample_id"], item["normalized_local_coordinate"])
        for item in second["ordered_sampling_pool"]
    ]
    assert first_pool == second_pool
    assert len(first["ordered_sampling_pool"]) == POOL_SIZE
    assert len(first["selected_valid_samples"]) == 12
    assert all(
        all(-LOCAL_HALF_WIDTH <= value <= LOCAL_HALF_WIDTH for value in sample["normalized_local_coordinate"].values())
        for sample in first["selected_valid_samples"]
    )
    _verify_precommit(first)
    assert all("geometry_valid" in item for item in first["rejected_samples"])


def test_clipped_samples_preserve_requested_and_realized_coordinates() -> None:
    payload = _precommit_payload()
    sample = next(item for item in payload["ordered_sampling_pool"] if item["sample_id"] == "local_003")
    assert sample["requested_normalized_local_coordinate"]["stem_height"] < -0.04
    assert abs(sample["normalized_local_coordinate"]["stem_height"] + 0.04091937281191349) < 1.0e-12
    assert abs(sample["parameters"]["stem_height"] - 5.0) < 1.0e-12


def test_tie_aware_ranks_and_pairwise_counts_surface_ties() -> None:
    ranks = _tie_aware_ranks([3.0, 3.0 + 0.5 * J2D_TIE_TOLERANCE, 1.0], J2D_TIE_TOLERANCE)
    assert ranks[0] == ranks[1]
    result = _pairwise_rank_result(
        [3.0, 3.0 + 0.5 * J2D_TIE_TOLERANCE, 1.0],
        [2.0, 2.0 + 0.5 * J3D_PATH_TIE_TOLERANCE, 1.0],
        j2d_tolerance=J2D_TIE_TOLERANCE,
        j3d_tolerance=J3D_PATH_TIE_TOLERANCE,
    )
    assert result["tied_count"] == 1
    assert result["concordant_count"] == 2
    assert result["discordant_count"] == 0
