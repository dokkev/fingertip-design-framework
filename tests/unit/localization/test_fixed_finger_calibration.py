"""Tests for the fixed-experiment projective geometry calibration."""

from types import SimpleNamespace

import cv2
import numpy as np

from experiments.localization import (
    FixedFingerCalibration,
    load_fixed_finger_calibration,
    project_longitudinal_positions,
    save_fixed_finger_calibration,
)
from experiments.localization.fixed_finger_calibration import (
    _five_led_window_score,
    _physical_led_fractions,
    _refine_led_array_fractions,
)
import experiments.localization.fixed_finger_calibration as fixed_calibration


def _project(homography: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points_h = np.column_stack((points_xy, np.ones(len(points_xy))))
    projected_h = points_h @ homography.T
    return projected_h[:, :2] / projected_h[:, 2, None]


def _line_through(first_xy: np.ndarray, second_xy: np.ndarray) -> np.ndarray:
    return np.cross(
        np.asarray((*first_xy, 1.0)),
        np.asarray((*second_xy, 1.0)),
    )


def _projective_fixture(
    homography: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dorsal_ends = _project(homography, np.asarray(((0.0, 0.0), (0.0, 1.0))))
    palmar_ends = _project(homography, np.asarray(((1.0, 0.0), (1.0, 1.0))))
    led_ends = _project(homography, np.asarray(((0.35, 0.0), (0.35, 1.0))))
    distal_line = _line_through(dorsal_ends[0], palmar_ends[0])
    proximal_line = _line_through(dorsal_ends[1], palmar_ends[1])
    led_line = _line_through(led_ends[0], led_ends[1])
    longitudinal_vanishing_point = homography @ np.asarray((0.0, 1.0, 0.0))
    return led_line, distal_line, proximal_line, longitudinal_vanishing_point


def test_projective_led_positions_match_planar_homography() -> None:
    homography = np.asarray(
        (
            (220.0, 30.0, 120.0),
            (15.0, 300.0, 80.0),
            (0.15, 0.35, 1.0),
        )
    )
    led_line, distal, proximal, vanishing = _projective_fixture(homography)
    fractions = np.asarray((0.1, 0.3, 0.5, 0.7, 0.9))

    actual = project_longitudinal_positions(
        led_line,
        distal,
        proximal,
        vanishing,
        fractions,
    )
    expected = _project(
        homography,
        np.column_stack((np.full(5, 0.35), fractions)),
    )

    assert np.allclose(actual, expected, atol=1.0e-10)
    assert np.all(np.diff(actual[:, 1]) > 0.0)


def test_calibration_recovers_synthetic_projective_led_array(monkeypatch) -> None:
    homography = np.asarray(
        (
            (210.0, 35.0, 120.0),
            (20.0, 420.0, 70.0),
            (0.10, 0.22, 1.0),
        )
    )
    image = np.zeros((520, 420, 3), dtype=np.uint8)
    world_corners = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    image_corners = _project(homography, world_corners)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(image_corners).astype(np.int32), 1)
    image[mask.astype(bool)] = (15, 180, 190)
    physical_fractions = _physical_led_fractions()
    expected_centers = _project(
        homography,
        np.column_stack((np.full(5, 0.28), physical_fractions)),
    )
    for center in expected_centers:
        cv2.circle(
            image,
            tuple(np.rint(center).astype(np.int32)),
            5,
            (255, 255, 255),
            -1,
        )
    monkeypatch.setattr(
        fixed_calibration,
        "segment_fingertip",
        lambda _: SimpleNamespace(final_mask=mask.astype(bool)),
    )

    calibration = fixed_calibration.calibrate_fixed_finger(image)

    errors_px = np.linalg.norm(
        calibration.led_centers_xy_px - expected_centers,
        axis=1,
    )
    assert np.allclose(calibration.led_longitudinal_fractions, physical_fractions)
    np.testing.assert_allclose(
        calibration.led_line_alpha,
        0.28,
        atol=0.02,
    )
    assert np.max(errors_px) < 7.0


def test_affine_camera_reduces_to_equal_image_spacing() -> None:
    homography = np.asarray(
        (
            (220.0, 30.0, 120.0),
            (15.0, 300.0, 80.0),
            (0.0, 0.0, 1.0),
        )
    )
    led_line, distal, proximal, vanishing = _projective_fixture(homography)
    fractions = np.asarray((0.1, 0.3, 0.5, 0.7, 0.9))

    projected = project_longitudinal_positions(
        led_line,
        distal,
        proximal,
        vanishing,
        fractions,
    )

    assert np.allclose(np.diff(projected, axis=0), np.diff(projected, axis=0)[0])


def test_hardware_led_fractions_include_the_distal_end_cap() -> None:
    expected = np.asarray((10.5, 21.5, 32.5, 43.5, 54.5)) / 60.0

    assert np.allclose(_physical_led_fractions(), expected)
    assert np.allclose(np.diff(_physical_led_fractions()), 11.0 / 60.0)


def test_led_line_score_uses_only_the_five_expected_windows() -> None:
    samples = np.linspace(0.0, 1.0, 1001)
    predicted = _physical_led_fractions()
    contrast = np.zeros_like(samples)
    for center in predicted:
        contrast[np.argmin(np.abs(samples - center))] = 2.0
    contrast[0] = 1000.0

    score = _five_led_window_score(samples, contrast, predicted)

    assert score == 10.0


def test_led_refinement_preserves_one_rigid_equal_pitch_array() -> None:
    samples = np.linspace(0.0, 1.0, 1001)
    predicted = np.linspace(0.1, 0.9, 5)
    contrast = np.zeros_like(samples)
    for center in predicted[:-1] + 0.02:
        contrast[np.argmin(np.abs(samples - center))] = 1.0
    contrast[np.argmin(np.abs(samples - (predicted[-1] + 0.04)))] = 1.0

    refined = _refine_led_array_fractions(predicted, samples, contrast)

    assert np.allclose(refined, predicted + 0.02)
    assert np.allclose(np.diff(refined), 0.2)


def test_led_refinement_keeps_geometry_when_any_window_is_ambiguous() -> None:
    samples = np.linspace(0.0, 1.0, 1001)
    predicted = np.linspace(0.1, 0.9, 5)
    contrast = np.zeros_like(samples)
    for center in predicted + 0.01:
        contrast[np.argmin(np.abs(samples - center))] = 1.0
    contrast[np.argmin(np.abs(samples - (predicted[-1] + 0.04)))] = 0.9

    refined = _refine_led_array_fractions(predicted, samples, contrast)

    assert np.array_equal(refined, predicted)


def test_fixed_calibration_npz_round_trip_has_no_image_payload(tmp_path) -> None:
    height, width = 20, 30
    map_x, map_y = np.meshgrid(
        np.linspace(5.0, 25.0, 4, dtype=np.float32),
        np.linspace(2.0, 18.0, 6, dtype=np.float32),
    )
    calibration = FixedFingerCalibration(
        image_shape=(height, width),
        dorsal_line=np.asarray((1.0, 0.0, -5.0)),
        palmar_line=np.asarray((1.0, 0.0, -25.0)),
        led_line=np.asarray((1.0, 0.0, -9.0)),
        vanishing_point_h=np.asarray((0.0, 1.0, 0.0)),
        distal_longitudinal_limit=np.asarray((0.0, 1.0, -2.0)),
        proximal_longitudinal_limit=np.asarray((0.0, 1.0, -18.0)),
        led_centers_xy_px=np.column_stack(
            (np.full(5, 9.0), np.linspace(3.6, 16.4, 5))
        ),
        led_longitudinal_fractions=np.linspace(0.1, 0.9, 5),
        canonical_map_x=map_x,
        canonical_map_y=map_y,
        reference_mask=np.ones((height, width), dtype=bool),
        distal_orientation="minimum_longitudinal",
        led_line_alpha=0.2,
        led_line_score=4.0,
    )
    path = tmp_path / "fixed_calibration.npz"

    save_fixed_finger_calibration(path, calibration)
    loaded = load_fixed_finger_calibration(path)

    assert loaded.image_shape == calibration.image_shape
    assert loaded.distal_orientation == calibration.distal_orientation
    for field in (
        "dorsal_line",
        "palmar_line",
        "led_line",
        "vanishing_point_h",
        "distal_longitudinal_limit",
        "proximal_longitudinal_limit",
        "led_centers_xy_px",
        "led_longitudinal_fractions",
        "canonical_map_x",
        "canonical_map_y",
        "reference_mask",
    ):
        assert np.array_equal(getattr(loaded, field), getattr(calibration, field))
    with np.load(path, allow_pickle=False) as data:
        assert "unloaded_rgb" not in data.files
        assert "raw_image" not in data.files
