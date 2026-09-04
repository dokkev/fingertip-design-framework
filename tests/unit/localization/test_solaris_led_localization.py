"""Focused tests for the fixed-setup Solaris five-lobe localizer."""

from types import SimpleNamespace

import cv2
import numpy as np

import experiments.localization.solaris_led_localization as solaris


def _synthetic_solaris(
    *,
    led_side: str = "left",
    terminal_intensity: int = 255,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = 240, 220
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    mask[20:221, 40:181] = True
    image[mask] = (25, 150, 170)
    peak_rows = np.asarray((55, 85, 115, 145, 175), dtype=np.float64)
    x = 72 if led_side == "left" else 148
    for row in peak_rows.astype(int):
        cv2.circle(image, (x, row), 7, (230, 230, 230), -1)
    cv2.circle(image, (x, 208), 8, (terminal_intensity,) * 3, -1)
    return image, mask, np.column_stack((np.full(5, x), peak_rows))


def _patch_segmentation(monkeypatch, mask: np.ndarray) -> None:
    monkeypatch.setattr(
        solaris,
        "segment_fingertip",
        lambda _: SimpleNamespace(final_mask=mask),
    )


def test_localizer_returns_five_ordered_centers(monkeypatch) -> None:
    image, mask, expected = _synthetic_solaris()
    _patch_segmentation(monkeypatch, mask)

    result = solaris.localize_solaris_leds(image)

    assert result.led_centers_xy_px.shape == (5, 2)
    assert np.all(np.diff(result.peak_rows_px) > 0.0)
    np.testing.assert_allclose(result.led_centers_xy_px, expected, atol=2.0)


def test_first_regular_sequence_precedes_enormous_terminal_peak() -> None:
    rows = np.arange(201, dtype=np.float64)
    profile = np.full(len(rows), 20.0)
    expected = np.asarray((35, 65, 95, 125, 155), dtype=np.float64)
    for row in expected:
        profile += 35.0 * np.exp(-0.5 * ((rows - row) / 3.0) ** 2)
    profile += 235.0 * np.exp(-0.5 * ((rows - 188.0) / 3.0) ** 2)

    _, selected, _, _ = solaris._regular_five_peak_sequence(rows, profile)

    np.testing.assert_allclose(selected, expected, atol=1.0)


def test_terminal_brightness_does_not_move_selected_leds(monkeypatch) -> None:
    weak, mask, _ = _synthetic_solaris(terminal_intensity=80)
    bright, _, _ = _synthetic_solaris(terminal_intensity=255)
    _patch_segmentation(monkeypatch, mask)

    weak_result = solaris.localize_solaris_leds(weak)
    bright_result = solaris.localize_solaris_leds(bright)

    np.testing.assert_allclose(
        weak_result.peak_rows_px,
        bright_result.peak_rows_px,
        atol=1.0,
    )
    np.testing.assert_allclose(
        weak_result.led_centers_xy_px,
        bright_result.led_centers_xy_px,
        atol=1.0,
    )


def test_side_with_regular_five_lobes_is_selected(monkeypatch) -> None:
    image, mask, _ = _synthetic_solaris(led_side="right")
    for row in (47, 102, 159):
        cv2.circle(image, (72, row), 7, (230, 230, 230), -1)
    _patch_segmentation(monkeypatch, mask)

    result = solaris.localize_solaris_leds(image)

    assert result.selected_side == "right"


def test_irregular_silhouette_does_not_move_right_lobes_to_left_side() -> None:
    height, width = 240, 220
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    mask[20:221, 30:46] = True
    mask[20:221, 90:191] = True
    mask[20:25, 30:191] = True
    mask[216:221, 30:191] = True
    image[mask] = (25, 150, 170)

    peak_rows = (55, 85, 115, 145, 175)
    for row in peak_rows:
        cv2.circle(image, (170, row), 7, (230, 230, 230), -1)
        cv2.circle(image, (125, row), 7, (255, 255, 255), -1)

    result = solaris.localize_solaris_leds(image, reference_mask=mask)

    assert result.selected_side == "right"


def test_brightest_fraction_centroid_recovers_local_white_lobe() -> None:
    image, mask, expected = _synthetic_solaris()

    center = solaris._brightest_fraction_centroid(
        image[:, :, 0].astype(np.float64),
        mask,
        "left",
        expected[2, 1],
        half_height=8,
    )

    np.testing.assert_allclose(center, expected[2], atol=1.0)


def test_temporal_median_removes_moving_bright_occluder() -> None:
    base, _, _ = _synthetic_solaris(terminal_intensity=80)
    frames = np.repeat(base[None, ...], 5, axis=0)
    for index, x_start in enumerate((5, 35, 65, 95, 125)):
        frames[index, 90:110, x_start : x_start + 15] = 255

    median = solaris.temporal_median_rgb(frames)

    np.testing.assert_array_equal(median, base)
