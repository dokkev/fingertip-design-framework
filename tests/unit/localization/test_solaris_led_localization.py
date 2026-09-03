"""Tests for the Solaris rigid periodic-array localizer."""

from types import SimpleNamespace

import cv2
import numpy as np

import experiments.localization.solaris_led_localization as solaris
from lumo.fingertip.layout import LED_CENTERS_Y_MM, TOTAL_Y_BOUNDS_MM


def _project(homography: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points_h = np.column_stack((points_xy, np.ones(len(points_xy))))
    projected_h = points_h @ homography.T
    return projected_h[:, :2] / projected_h[:, 2, None]


def _line_through(first_xy: np.ndarray, second_xy: np.ndarray) -> np.ndarray:
    return np.cross(np.asarray((*first_xy, 1.0)), np.asarray((*second_xy, 1.0)))


def _projective_fixture(
    homography: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x_coordinate = 0.28
    distal = _project(homography, np.asarray(((0.0, 0.0), (1.0, 0.0))))
    led = _project(homography, np.asarray(((x_coordinate, 0.0), (x_coordinate, 1.0))))
    led_line = _line_through(led[0], led[1])
    distal_limit = _line_through(distal[0], distal[1])
    vanishing = homography @ np.asarray((0.0, 1.0, 0.0))
    vanishing /= np.linalg.norm(vanishing[:2])

    distal_h = homography @ np.asarray((x_coordinate, 0.0, 1.0))
    direction_h = homography @ np.asarray((0.0, 1.0, 0.0))
    derivative = (
        direction_h[:2] * distal_h[2] - distal_h[:2] * direction_h[2]
    ) / distal_h[2] ** 2
    distal_point = distal_h[:2] / distal_h[2]
    projective_direction = vanishing[:2] - vanishing[2] * distal_point
    if np.dot(projective_direction, derivative) < 0.0:
        vanishing *= -1.0
    return led_line, distal_limit, vanishing, float(np.linalg.norm(derivative))


def _synthetic_solaris(
    proximal_end_y: int,
    terminal_y: int,
    terminal_intensity: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = 720, 360
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (80, 40), (300, proximal_end_y), 1, -1)
    image[mask.astype(bool)] = (20, 150, 170)

    scale_px_per_mm = 1.8
    for distance_mm in solaris.solaris_physical_led_layout():
        center = (145, int(round(40 + scale_px_per_mm * distance_mm)))
        cv2.circle(image, center, 5, (255, 255, 255), -1)
    cv2.rectangle(
        image,
        (135, terminal_y),
        (155, min(terminal_y + 30, proximal_end_y - 1)),
        (terminal_intensity, terminal_intensity, terminal_intensity),
        -1,
    )
    return image, mask.astype(bool)


def test_physical_layout_enforces_11_mm_pitch_and_44_mm_array_span() -> None:
    positions_mm = solaris.solaris_physical_led_layout()
    distal_y_mm = max(TOTAL_Y_BOUNDS_MM)
    expected = distal_y_mm - np.sort(np.asarray(LED_CENTERS_Y_MM))[::-1]

    assert np.array_equal(positions_mm, expected)
    assert np.allclose(np.diff(positions_mm), 11.0)
    assert positions_mm[-1] - positions_mm[0] == 44.0


def test_distal_projective_scale_matches_oblique_planar_homography() -> None:
    homography = np.asarray(
        (
            (210.0, 35.0, 120.0),
            (20.0, 420.0, 70.0),
            (0.10, 0.22, 1.0),
        )
    )
    led_line, distal_limit, vanishing, initial_scale = _projective_fixture(homography)
    distances_mm = solaris.solaris_physical_led_layout()

    actual = solaris._project_from_distal(
        led_line,
        distal_limit,
        vanishing,
        distances_mm,
        initial_scale,
    )
    expected = _project(
        homography,
        np.column_stack((np.full(5, 0.28), distances_mm)),
    )

    np.testing.assert_allclose(actual, expected, atol=1.0e-10)
    spacings = np.linalg.norm(np.diff(actual, axis=0), axis=1)
    assert not np.allclose(spacings, spacings[0])


def test_distal_projective_scale_reduces_to_affine_equal_spacing() -> None:
    homography = np.asarray(
        (
            (210.0, 35.0, 120.0),
            (20.0, 420.0, 70.0),
            (0.0, 0.0, 1.0),
        )
    )
    led_line, distal_limit, vanishing, initial_scale = _projective_fixture(homography)
    projected = solaris._project_from_distal(
        led_line,
        distal_limit,
        vanishing,
        solaris.solaris_physical_led_layout(),
        initial_scale,
    )

    spacings = np.diff(projected, axis=0)
    np.testing.assert_allclose(spacings, np.tile(spacings[0], (4, 1)))


def test_full_periodic_pattern_recovers_rigid_array(monkeypatch) -> None:
    image, mask = _synthetic_solaris(680, 620, 30)
    monkeypatch.setattr(
        solaris,
        "segment_fingertip",
        lambda _: SimpleNamespace(final_mask=mask),
    )

    result = solaris.localize_solaris_leds(image)
    expected = np.column_stack(
        (
            np.full(5, 145.0),
            40.0 + 1.8 * solaris.solaris_physical_led_layout(),
        )
    )

    np.testing.assert_allclose(result.led_centers_xy_px, expected, atol=2.0)
    assert result.led_center_responses.shape == (5,)
    assert result.inter_led_responses.shape == (4,)
    assert result.line_score == np.mean(result.led_center_responses) - np.mean(
        result.inter_led_responses
    )


def test_terminal_light_beyond_led5_cannot_move_array(monkeypatch) -> None:
    weak_image, weak_mask = _synthetic_solaris(700, 600, 20)
    bright_image, bright_mask = _synthetic_solaris(700, 600, 255)

    masks = iter((weak_mask, bright_mask))
    monkeypatch.setattr(
        solaris,
        "segment_fingertip",
        lambda _: SimpleNamespace(final_mask=next(masks)),
    )
    weak_result = solaris.localize_solaris_leds(weak_image)
    bright_result = solaris.localize_solaris_leds(bright_image)

    np.testing.assert_allclose(
        weak_result.led_centers_xy_px,
        bright_result.led_centers_xy_px,
        rtol=0.0,
        atol=2.0e-5,
    )
    assert weak_result.led_line_alpha == bright_result.led_line_alpha
    assert np.isclose(
        weak_result.longitudinal_scale_px_per_mm,
        bright_result.longitudinal_scale_px_per_mm,
        rtol=0.0,
        atol=2.0e-7,
    )
    assert np.isclose(
        weak_result.line_score,
        bright_result.line_score,
        rtol=0.0,
        atol=5.0e-4,
    )


def test_proximal_endpoint_does_not_set_array_scale(monkeypatch) -> None:
    short_image, short_mask = _synthetic_solaris(660, 600, 20)
    long_image, long_mask = _synthetic_solaris(700, 600, 20)
    masks = iter((short_mask, long_mask))
    monkeypatch.setattr(
        solaris,
        "segment_fingertip",
        lambda _: SimpleNamespace(final_mask=next(masks)),
    )

    short_result = solaris.localize_solaris_leds(short_image)
    long_result = solaris.localize_solaris_leds(long_image)

    np.testing.assert_allclose(
        short_result.led_centers_xy_px,
        long_result.led_centers_xy_px,
        rtol=0.0,
        atol=3.0e-5,
    )


def test_leds_three_to_five_disambiguate_periodic_hypotheses() -> None:
    positions_mm = solaris.solaris_physical_led_layout()
    distances_mm = np.linspace(positions_mm[0], positions_mm[-1], 401)
    midpoint_mm = 0.5 * (positions_mm[:-1] + positions_mm[1:])
    contrast = np.zeros((2, len(distances_mm)), dtype=np.float64)
    half_window_mm = 0.8
    for center_mm in positions_mm[:2]:
        contrast[:, np.abs(distances_mm - center_mm) <= half_window_mm] = 4.0
    for center_mm in positions_mm[2:]:
        contrast[0, np.abs(distances_mm - center_mm) <= half_window_mm] = 4.0
    for center_mm in midpoint_mm[2:]:
        contrast[1, np.abs(distances_mm - center_mm) <= half_window_mm] = 4.0

    center, midpoint, valid = solaris._local_periodic_responses(
        distances_mm,
        contrast,
        np.ones_like(contrast, dtype=bool),
        positions_mm,
        half_window_mm,
    )
    scores = np.mean(center, axis=1) - np.mean(midpoint, axis=1)

    np.testing.assert_allclose(center[0, :2], center[1, :2])
    assert np.all(valid)
    assert scores[0] > scores[1]
