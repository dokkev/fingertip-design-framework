"""Regression tests for the Phase 4K-Viz metadata and coordinate contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from visualization.adapters.phase4k import (
    canonicalize_array,
    common_eta_profiles,
    descriptor_verified_mask,
    display_zeta_for_side,
    load_codtm_dataset,
    mirror_metrics,
    profile_comparison_metrics,
    profile_segments,
    zeta_for_side,
)
from visualization.transforms import (
    CODTMVisualizationError,
    location_distance_matrix,
    select_indentation,
    shape_distance_matrix,
    signature_norm,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_INPUT = REPOSITORY_ROOT / "output" / "phase4_mechanical_transfer_map"


@pytest.fixture(scope="module")
def dataset():
    loaded, audit = load_codtm_dataset(CANONICAL_INPUT)
    assert audit["status"] == "PASS"
    return loaded


def test_metadata_axes_are_parsed_from_canonical_schema(dataset) -> None:
    assert dataset.canonical_field("u_normal").shape == (8, 48, 2, 41)
    assert dataset.canonical_field("u_xy").shape == (8, 48, 2, 41, 2)
    assert set(dataset.side_order) == {"left", "right"}


def test_named_axis_canonicalization_is_independent_of_axis_order() -> None:
    canonical = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    permuted = np.transpose(canonical, (2, 0, 1))
    restored = canonicalize_array(
        permuted,
        ("sample", "case", "side"),
        ("case", "side", "sample"),
    )
    assert np.array_equal(restored, canonical)


def test_side_order_does_not_change_distance() -> None:
    eta = np.stack([np.linspace(0.0, 1.0, 5)] * 2)
    signatures = np.asarray(
        [
            [np.linspace(0.0, 1.0, 5), np.linspace(1.0, 2.0, 5)],
            [np.linspace(0.1, 1.1, 5), np.linspace(0.8, 1.8, 5)],
        ]
    )
    expected = location_distance_matrix(signatures, eta)
    assert np.allclose(
        location_distance_matrix(signatures[:, ::-1], eta[::-1]), expected
    )


def test_eta_order_does_not_change_integrated_metric() -> None:
    eta = np.stack([np.linspace(0.0, 1.0, 7)] * 2)
    field = np.asarray(
        [[eta[0] ** 2, 2.0 * eta[1]], [eta[0], eta[1] ** 3]]
    )
    expected = location_distance_matrix(field, eta)
    assert np.allclose(
        location_distance_matrix(field[..., ::-1], eta[..., ::-1]), expected
    )


def test_signed_zeta_mapping_matches_contract() -> None:
    eta = np.asarray([0.0, 0.5, 1.0])
    assert np.array_equal(zeta_for_side("right", eta), [-1.0, -0.5, 0.0])
    assert np.array_equal(zeta_for_side("left", eta), [1.0, 0.5, 0.0])


def test_display_zeta_keeps_two_eta_one_endpoints_distinct() -> None:
    right = display_zeta_for_side("right", [1.0])
    left = display_zeta_for_side("left", [1.0])
    assert right[0] < 0.0 < left[0]
    assert left[0] - right[0] == pytest.approx(0.08)


def test_profile_segments_never_create_a_center_connection() -> None:
    segments = profile_segments(
        {"right": [1.0, 2.0], "left": [3.0, 4.0]},
        {"right": [0.0, 1.0], "left": [0.0, 1.0]},
    )
    assert len(segments) == 2
    assert [segment[0] for segment in segments] == ["right", "left"]
    assert not np.shares_memory(segments[0][2], segments[1][2])


def test_exact_indentation_selection() -> None:
    result = select_indentation(
        [0.25, 0.50, 0.75],
        np.asarray([[1.0], [2.0], [3.0]]),
        [True, True, True],
        0.50,
    )
    assert result.exact
    assert result.lower_step_index == result.upper_step_index == 1
    assert result.values == pytest.approx([2.0])


def test_bracketed_interpolation_reproduces_affine_field() -> None:
    delta = np.asarray([0.0, 1.0, 2.0])
    values = np.stack([2.0 * delta + 3.0, -4.0 * delta + 1.0], axis=1)
    result = select_indentation(delta, values, [True, True, True], 1.5)
    assert not result.exact
    assert result.interpolation_weight == pytest.approx(0.5)
    assert result.values == pytest.approx([6.0, -5.0])


def test_indentation_extrapolation_is_rejected() -> None:
    with pytest.raises(CODTMVisualizationError, match="extrapolation"):
        select_indentation([0.5, 1.0], np.ones((2, 1)), [True, True], 1.5)


def test_interpolation_across_invalid_step_is_rejected() -> None:
    with pytest.raises(CODTMVisualizationError, match="invalid step"):
        select_indentation(
            [0.5, 1.0, 1.5], np.ones((3, 1)), [True, False, True], 1.25
        )


def test_distance_matrix_is_symmetric() -> None:
    eta = np.stack([np.linspace(0.0, 1.0, 11)] * 2)
    fields = np.arange(3 * 2 * 11, dtype=float).reshape(3, 2, 11)
    matrix = location_distance_matrix(fields, eta)
    assert np.array_equal(matrix, matrix.T)


def test_distance_diagonal_is_exactly_zero() -> None:
    eta = np.stack([np.linspace(0.0, 1.0, 4)] * 2)
    fields = np.arange(2 * 2 * 4, dtype=float).reshape(2, 2, 4)
    assert np.array_equal(np.diag(location_distance_matrix(fields, eta)), [0.0, 0.0])


def test_side_integrals_are_separate_and_do_not_include_gap() -> None:
    eta = np.stack([np.linspace(0.0, 1.0, 101)] * 2)
    signature = np.ones((2, 101))
    # Each unit-length side contributes one; the center display gap contributes zero.
    assert signature_norm(signature, eta) == pytest.approx(np.sqrt(2.0))


def test_shape_distance_has_zero_norm_guard() -> None:
    eta = np.stack([np.linspace(0.0, 1.0, 5)] * 2)
    with pytest.raises(CODTMVisualizationError, match="zero norm"):
        shape_distance_matrix(np.zeros((2, 2, 5)), eta)


def test_mirror_mapping_is_exact_for_synthetic_pair() -> None:
    eta = np.stack([np.linspace(0.0, 1.0, 9)] * 2)
    original = np.stack([1.0 + eta[0], 3.0 - eta[1]])
    partner = np.stack([original[1], original[0]])
    result = mirror_metrics(original, partner, eta, ("left", "right"))
    assert result["absolute_l2_mm"] == pytest.approx(0.0)
    assert result["relative_l2"] == pytest.approx(0.0)
    assert result["max_abs_mm"] == pytest.approx(0.0)


def test_common_eta_mapping_reproduces_affine_profile() -> None:
    source_eta = np.stack([np.linspace(0.0, 1.0, 5)] * 2)
    target_eta = np.stack([np.linspace(0.0, 1.0, 9)] * 2)
    values = np.stack([2.0 * source_eta[0] + 1.0, -source_eta[1] + 4.0])
    mapped = common_eta_profiles(values, source_eta, target_eta)
    assert mapped[0] == pytest.approx(2.0 * target_eta[0] + 1.0)
    assert mapped[1] == pytest.approx(-target_eta[1] + 4.0)


def test_descriptor_mask_is_distinct_from_displacement_validity(dataset) -> None:
    descriptor = descriptor_verified_mask(dataset)
    displacement = np.asarray(dataset.arrays["valid_mask"], dtype=bool)
    assert displacement.all()
    assert not descriptor.all()
    assert descriptor.shape == displacement.shape



def test_medium_fine_profile_metrics_match_reported_scale(dataset) -> None:
    medium = dataset.select_case_field("medium_xi_0p20", "u_normal", 1.5).values
    fine = dataset.select_case_field("fine_xi_0p20", "u_normal", 1.5).values
    metrics = profile_comparison_metrics(medium, fine)
    assert metrics["relative_l2"] == pytest.approx(0.00635, rel=0.02)
    assert metrics["shape_correlation"] > 0.99996
