"""Focused tests for image-only LED/contact localization."""

import cv2
import numpy as np

from experiments.localization import (
    brightest_red_features,
    constrain_led_array_motion,
    detect_led_array,
    estimate_contact_position,
    track_led_array,
    unloaded_baseline_statistics,
)


def _synthetic_led_image() -> np.ndarray:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    for y_coordinate in (150, 170, 190, 210, 230):
        cv2.circle(image, (390, y_coordinate), 4, (255, 80, 50), -1)
    return image


def test_detects_ordered_five_led_array_and_measures_features() -> None:
    image = _synthetic_led_image()
    geometry = detect_led_array(np.repeat(image[None, ...], 7, axis=0))
    features = brightest_red_features(image, geometry)

    assert geometry.landmarks_xy_px.shape == (5, 2)
    assert np.all(np.diff(geometry.landmarks_xy_px[:, 1]) > 0.0)
    assert np.isclose(geometry.median_spacing_px, 20.0, atol=1.5)
    assert features.shape == (5,)
    assert np.all(features > 0.0)


def test_local_roi_feature_matches_full_frame_mask_definition() -> None:
    geometry = detect_led_array(_synthetic_led_image())
    image = np.random.default_rng(17).integers(
        0,
        256,
        size=(480, 640, 3),
        dtype=np.uint8,
    )
    expected = []
    red = image[:, :, 0].astype(np.float64)
    for polygon in geometry.roi_polygons_xy_px:
        mask = np.zeros(red.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
        pixels = red[mask > 0]
        count = max(1, int(np.ceil(0.10 * pixels.size)))
        expected.append(
            float(np.mean(np.partition(pixels, pixels.size - count)[-count:]))
        )

    actual = brightest_red_features(image, geometry)

    assert np.array_equal(actual, expected)


def test_component_fallback_handles_oblique_led_spacing() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    for x_coordinate, y_coordinate in zip(
        (390, 391, 393, 396, 400),
        (150, 168, 191, 220, 266),
        strict=True,
    ):
        cv2.circle(image, (x_coordinate, y_coordinate), 5, (255, 80, 50), -1)

    geometry = detect_led_array(np.repeat(image[None, ...], 7, axis=0))

    assert np.all(np.diff(geometry.landmarks_xy_px[:, 1]) > 0.0)
    assert np.allclose(
        geometry.landmarks_xy_px[:, 1],
        (150, 168, 191, 220, 266),
        atol=2.0,
    )


def test_contact_position_is_positive_response_weighted_centroid() -> None:
    estimate = estimate_contact_position(
        features=np.array((10.0, 12.0, 18.0, 14.0, 10.0)),
        unloaded_baseline=np.full(5, 10.0),
        unloaded_noise_sigma=np.ones(5),
        led_positions_mm=np.array((-22.0, -11.0, 0.0, 11.0, 22.0)),
    )

    assert estimate.contact_detected
    assert estimate.predicted_led_index == 2
    assert np.isclose(estimate.position_mm, 22.0 / 14.0)
    assert np.isclose(estimate.top_two_margin, 4.0)


def test_rigid_led_motion_rejects_one_corrupted_candidate() -> None:
    previous = np.column_stack((np.full(5, 100.0), np.arange(5) * 20.0 + 100.0))
    angle = np.deg2rad(4.0)
    scale = 1.03
    rotation = scale * np.array(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    expected = previous @ rotation.T + np.array((7.0, -3.0))
    candidates = expected.copy()
    candidates[2] += np.array((35.0, -28.0))

    constrained = constrain_led_array_motion(
        previous,
        candidates,
        np.ones(5, dtype=bool),
        median_spacing_px=20.0,
    )

    assert np.allclose(constrained, expected, atol=1.0e-6)
    assert np.linalg.norm(constrained[2] - candidates[2]) > 20.0
    assert np.all(np.diff(constrained[:, 1]) > 0.0)
    assert np.allclose(
        np.linalg.norm(np.diff(constrained, axis=0), axis=1),
        scale * 20.0,
    )


def test_rigid_led_motion_requires_four_valid_correspondences() -> None:
    landmarks = np.column_stack((np.full(5, 100.0), np.arange(5) * 20.0 + 100.0))

    with np.testing.assert_raises_regex(RuntimeError, "at least four"):
        constrain_led_array_motion(
            landmarks,
            landmarks,
            np.array((True, True, True, False, False)),
            median_spacing_px=20.0,
        )


def test_track_led_array_survives_one_failed_backward_status(monkeypatch) -> None:
    image = _synthetic_led_image()
    geometry = detect_led_array(image)
    previous_points = geometry.landmarks_xy_px.astype(np.float32).reshape(-1, 1, 2)
    expected_points = previous_points + np.asarray((3.0, -2.0), dtype=np.float32)
    call_count = 0

    def fake_lk(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return expected_points.copy(), np.ones((5, 1), dtype=np.uint8), None
        status = np.ones((5, 1), dtype=np.uint8)
        status[2] = 0
        return previous_points.copy(), status, None

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_lk)

    tracked = track_led_array(image, image, geometry)

    assert call_count == 2
    assert np.allclose(
        tracked.landmarks_xy_px,
        expected_points.reshape(-1, 2),
        atol=1.0e-6,
    )


def test_unloaded_baseline_uses_robust_per_led_statistics() -> None:
    samples = np.tile(np.array((40.0, 50.0, 60.0, 70.0, 80.0)), (30, 1))
    samples[4] += 100.0

    baseline, noise_sigma = unloaded_baseline_statistics(samples)

    assert np.array_equal(baseline, (40.0, 50.0, 60.0, 70.0, 80.0))
    assert np.all(np.isfinite(noise_sigma))
    assert np.all(noise_sigma > 0.0)


def test_contact_gate_rejects_unloaded_noise() -> None:
    estimate = estimate_contact_position(
        features=np.array((10.2, 10.4, 10.1, 10.3, 10.0)),
        unloaded_baseline=np.full(5, 10.0),
        unloaded_noise_sigma=np.ones(5),
        led_positions_mm=np.array((-22.0, -11.0, 0.0, 11.0, 22.0)),
    )

    assert not estimate.contact_detected
    assert estimate.position_mm is None


def test_contact_gate_accepts_clear_response() -> None:
    estimate = estimate_contact_position(
        features=np.array((10.0, 10.0, 15.0, 10.0, 10.0)),
        unloaded_baseline=np.full(5, 10.0),
        unloaded_noise_sigma=np.ones(5),
        led_positions_mm=np.array((-22.0, -11.0, 0.0, 11.0, 22.0)),
    )

    assert estimate.contact_detected
    assert estimate.predicted_led_index == 2


def test_contact_gate_keeps_equal_neighbor_response_active() -> None:
    estimate = estimate_contact_position(
        features=np.array((10.0, 15.0, 15.0, 10.0, 10.0)),
        unloaded_baseline=np.full(5, 10.0),
        unloaded_noise_sigma=np.ones(5),
        led_positions_mm=np.array((-22.0, -11.0, 0.0, 11.0, 22.0)),
    )

    assert estimate.contact_detected
    assert np.isclose(estimate.top_two_margin, 0.0)
    assert np.isclose(estimate.position_mm, -5.5)
