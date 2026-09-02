"""Focused tests for reusable dense and LED contact observers."""

import numpy as np
import pytest

from experiments.localization import (
    DenseProfileConfig,
    build_dense_template_model,
    estimate_affine_position_from_centroid,
    estimate_dense_template_position,
    fit_affine_position_from_centroid,
    load_dense_template_model,
    response_centroid,
    save_dense_template_model,
)


def test_exact_dense_template_returns_its_labelled_position() -> None:
    profiles = np.array(
        (
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    model = build_dense_template_model(
        profiles,
        np.array((0.0, 5.0, 10.0)),
        canonical_shape=(4, 8),
    )

    estimate = estimate_dense_template_position(profiles[1], model)

    assert estimate.position_mm == 5.0
    assert estimate.matched_index == 1


def test_shifted_response_centroids_produce_ordered_affine_positions() -> None:
    first = np.array((0.0, 4.0, 1.0, 0.0, 0.0))
    middle = np.array((0.0, 0.0, 1.0, 4.0, 0.0))
    last = np.array((0.0, 0.0, 0.0, 1.0, 4.0))
    centroids = np.array(
        [response_centroid(profile) for profile in (first, middle, last)]
    )
    model = fit_affine_position_from_centroid(
        centroids,
        np.array((0.0, 10.0, 20.0)),
    )
    baseline = np.zeros(5)

    estimates = [
        estimate_affine_position_from_centroid(profile, baseline, model)
        for profile in (first, middle, last)
    ]
    assert np.all(np.diff(estimates) > 0.0)
    assert np.allclose(estimates, (0.0, 10.0, 20.0), atol=2.0)


def test_baseline_dependent_centroid_fails_without_valid_response() -> None:
    model = fit_affine_position_from_centroid(
        np.array((0.2, 0.8)),
        np.array((0.0, 10.0)),
    )

    with pytest.raises(ValueError, match="no robust positive response"):
        estimate_affine_position_from_centroid(
            np.ones(8),
            np.ones(8),
            model,
        )


def test_dense_template_npz_round_trip_uses_explicit_metadata(tmp_path) -> None:
    config = DenseProfileConfig(
        mode="mean_red",
        transverse_reduction="mean",
        top_fraction=0.2,
    )
    model = build_dense_template_model(
        np.array(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        np.array((2.0, 7.0)),
        canonical_shape=(3, 12),
        feature_config=config,
    )
    path = tmp_path / "dense_model.npz"

    save_dense_template_model(path, model)
    loaded = load_dense_template_model(path)

    assert np.array_equal(loaded.positions_mm, model.positions_mm)
    assert np.array_equal(loaded.templates, model.templates)
    assert loaded.canonical_shape == model.canonical_shape
    assert loaded.feature_config == config
    assert loaded.normalization == model.normalization


def test_response_centroid_ignores_low_response_background() -> None:
    response = np.full(11, 2.0)
    response[7:9] = (12.0, 8.0)

    centroid = response_centroid(response)

    assert centroid > 0.65
