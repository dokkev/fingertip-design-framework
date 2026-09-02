"""Focused tests for canonical dense optical features."""

import cv2
import numpy as np

from experiments.localization import (
    DenseProfileConfig,
    extract_dense_profile,
    extract_dense_response_profile,
    mean_center_l2,
)


def test_dense_top10_red_matches_direct_numpy_reference() -> None:
    rng = np.random.default_rng(12)
    image = rng.integers(0, 256, size=(17, 20, 3), dtype=np.uint8)
    config = DenseProfileConfig(
        mode="top10_red",
        transverse_start_fraction=0.2,
        transverse_stop_fraction=0.8,
        top_fraction=0.10,
        longitudinal_smoothing_sigma_px=0.0,
    )
    selected = image[:, 4:16, 0].astype(np.float64)
    count = int(np.ceil(0.10 * selected.shape[1]))
    expected = np.mean(np.sort(selected, axis=1)[:, -count:], axis=1)

    assert np.array_equal(extract_dense_profile(image, config), expected)


def test_mean_center_l2_is_invariant_to_offset_and_positive_scale() -> None:
    profile = np.array((1.0, 4.0, 2.0, 8.0, 3.0))

    assert np.allclose(
        mean_center_l2(profile),
        mean_center_l2(7.5 * profile + 31.0),
    )


def test_highpass_and_gradient_profiles_are_finite_and_keep_height() -> None:
    image = np.zeros((31, 23, 3), dtype=np.uint8)
    cv2.circle(image, (11, 16), 6, (240, 80, 40), -1)

    for mode in ("abs_highpass_red", "red_gradient"):
        profile = extract_dense_profile(
            image,
            DenseProfileConfig(mode=mode, longitudinal_smoothing_sigma_px=0.0),
        )
        assert profile.shape == (31,)
        assert np.all(np.isfinite(profile))
        assert np.max(profile) > 0.0


def test_unloaded_relative_highpass_is_zero_only_without_image_change() -> None:
    unloaded = np.full((25, 21, 3), 80, dtype=np.uint8)
    loaded = unloaded.copy()
    cv2.circle(loaded, (10, 15), 4, (180, 80, 80), -1)
    config = DenseProfileConfig(
        mode="abs_highpass_red",
        longitudinal_smoothing_sigma_px=0.0,
    )

    unchanged = extract_dense_response_profile(unloaded, unloaded, config)
    response = extract_dense_response_profile(loaded, unloaded, config)

    assert np.array_equal(unchanged, np.zeros(25))
    assert np.all(np.isfinite(response))
    assert np.max(response) > 0.0


def test_unloaded_relative_highpass_supports_verified_transverse_mean() -> None:
    rng = np.random.default_rng(19)
    unloaded = rng.integers(0, 160, size=(21, 20, 3), dtype=np.uint8)
    loaded = rng.integers(0, 256, size=(21, 20, 3), dtype=np.uint8)
    config = DenseProfileConfig(
        mode="abs_highpass_red",
        transverse_start_fraction=0.0,
        transverse_stop_fraction=0.95,
        transverse_reduction="mean",
        longitudinal_smoothing_sigma_px=0.0,
    )
    difference = (
        loaded[:, :, 0].astype(np.float32)
        - unloaded[:, :, 0].astype(np.float32)
    )
    smooth = cv2.GaussianBlur(
        difference,
        (0, 0),
        config.highpass_sigma_px,
        config.highpass_sigma_px,
    )
    expected = np.mean(np.abs(difference[:, :19] - smooth[:, :19]), axis=1)

    assert np.allclose(
        extract_dense_response_profile(loaded, unloaded, config),
        expected,
    )


def test_red_gradient_uses_two_dimensional_magnitude() -> None:
    image = np.zeros((19, 23, 3), dtype=np.uint8)
    image[:, 12:, 0] = 200

    profile = extract_dense_profile(
        image,
        DenseProfileConfig(mode="red_gradient"),
    )

    assert np.all(profile > 0.0)
