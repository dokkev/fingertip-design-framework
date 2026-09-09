import cv2
import numpy as np
import pytest

import algorithm.led_localization as led_localization
from algorithm.fingertip_segmentation import segment_fingertip
from algorithm.led_localization import (
    LED_POSITIONS_MM,
    LedGeometry,
    detect_leds,
    estimate_led_similarity_transform,
    reanchor_leds,
    track_leds,
)


def _geometry(points: np.ndarray) -> LedGeometry:
    return LedGeometry(
        landmarks_xy_px=points,
        positions_mm=LED_POSITIONS_MM,
        median_spacing_px=float(np.median(np.linalg.norm(np.diff(points, axis=0), axis=1))),
    )


def _apply(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.column_stack((points, np.ones(len(points)))) @ transform.T


def _synthetic_led_image() -> np.ndarray:
    rgb = np.full((240, 240, 3), 10, dtype=np.uint8)
    cv2.rectangle(rgb, (70, 20), (150, 220), (20, 180, 190), -1)
    for y_coordinate in (55, 92, 128, 165, 202):
        cv2.circle(rgb, (100, y_coordinate), 5, (255, 220, 80), -1)
    return rgb


def test_detect_leds_returns_ordered_regular_five_point_array() -> None:
    rgb = _synthetic_led_image()
    detected = detect_leds(rgb, segment_fingertip(rgb))

    assert detected.landmarks_xy_px.shape == (5, 2)
    np.testing.assert_allclose(
        detected.landmarks_xy_px[:, 1],
        [55, 92, 128, 165, 202],
        atol=3,
    )
    assert np.all(np.diff(detected.landmarks_xy_px[:, 1]) > 0.0)
    np.testing.assert_allclose(detected.positions_mm, [0, 11, 22, 33, 44])


@pytest.mark.parametrize(
    "angle_deg,scale,translation",
    [
        (0.0, 1.0, (8.0, -6.0)),
        (12.0, 1.0, (3.0, 5.0)),
        (-8.0, 1.08, (-4.0, 7.0)),
    ],
)
def test_similarity_transform_recovers_rigid_motion(
    angle_deg: float,
    scale: float,
    translation: tuple[float, float],
) -> None:
    reference_points = np.column_stack((np.full(5, 80.0), np.arange(5) * 30.0 + 40.0))
    angle = np.deg2rad(angle_deg)
    expected = np.asarray(
        [
            [scale * np.cos(angle), -scale * np.sin(angle), translation[0]],
            [scale * np.sin(angle), scale * np.cos(angle), translation[1]],
        ]
    )
    current_points = _apply(reference_points, expected)

    estimated = estimate_led_similarity_transform(
        _geometry(reference_points),
        _geometry(current_points),
    )

    np.testing.assert_allclose(_apply(reference_points, estimated), current_points, atol=1.0e-4)


def test_track_leds_rejects_one_bad_correspondence_with_ransac(monkeypatch) -> None:
    previous_points = np.column_stack((np.full(5, 80.0), np.arange(5) * 30.0 + 40.0))
    transform = np.asarray([[1.0, 0.0, 4.0], [0.0, 1.0, 3.0]])
    candidates = _apply(previous_points, transform)
    candidates[2] += (45.0, -38.0)
    calls = 0

    def fake_lk(previous, current, points, unused, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return candidates.astype(np.float32)[:, None, :], np.ones((5, 1), np.uint8), None
        return previous_points.astype(np.float32)[:, None, :], np.ones((5, 1), np.uint8), None

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_lk)
    image = np.zeros((240, 240, 3), dtype=np.uint8)

    tracked = track_leds(image, image, _geometry(previous_points))

    np.testing.assert_allclose(tracked.landmarks_xy_px, _apply(previous_points, transform), atol=1.0e-4)
    array_axis = tracked.landmarks_xy_px[-1] - tracked.landmarks_xy_px[0]
    assert np.all(np.diff(tracked.landmarks_xy_px, axis=0) @ array_axis > 0.0)


def test_track_leds_rejects_fewer_than_four_correspondences(monkeypatch) -> None:
    previous_points = np.column_stack((np.full(5, 80.0), np.arange(5) * 30.0 + 40.0))

    def fake_lk(previous, current, points, unused, **kwargs):
        return points.copy(), np.asarray([[1], [1], [1], [0], [0]], np.uint8), None

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_lk)
    image = np.zeros((240, 240, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="fewer than four"):
        track_leds(image, image, _geometry(previous_points))


def test_reanchor_rejects_large_absolute_detector_jump(monkeypatch) -> None:
    previous_points = np.column_stack((np.full(5, 80.0), np.arange(5) * 30.0 + 40.0))
    previous = _geometry(previous_points)
    rgb = _synthetic_led_image()
    fingertip = segment_fingertip(rgb)
    monkeypatch.setattr(
        led_localization,
        "detect_leds",
        lambda image, region: _geometry(previous_points + (20.0, 0.0)),
    )

    with pytest.raises(RuntimeError, match="correction is too large"):
        reanchor_leds(rgb, fingertip, previous)
