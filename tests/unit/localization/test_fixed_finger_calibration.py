"""Tests for the fixed-experiment projective geometry calibration."""

import numpy as np

from experiments.localization import (
    FixedFingerCalibration,
    load_fixed_finger_calibration,
    project_longitudinal_positions,
    save_fixed_finger_calibration,
)


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
