"""Focused tests for image-only LED/contact localization."""

import cv2
import numpy as np

from lumo.localization import (
    brightest_red_features,
    detect_led_array,
    estimate_contact_position,
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
        led_positions_mm=np.array((-22.0, -11.0, 0.0, 11.0, 22.0)),
    )

    assert estimate.predicted_led_index == 2
    assert np.isclose(estimate.position_mm, 22.0 / 14.0)
    assert np.isclose(estimate.top_two_margin, 4.0)
