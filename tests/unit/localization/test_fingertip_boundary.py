"""Focused tests for side-view fingertip-boundary detection."""

import cv2
import numpy as np

from experiments.localization import detect_fingertip_boundary


def _synthetic_side_view(
    *,
    internal_streaks: bool = False,
    dorsal_distractor: bool = False,
) -> np.ndarray:
    height, width = 480, 640
    image = np.full((height, width, 3), 210, dtype=np.uint8)
    rows = np.arange(70, 411)
    dorsal_x = np.rint(175.0 + 0.045 * (rows - 70)).astype(np.int32)
    palmar_x = np.rint(388.0 + 0.00018 * (rows - 240) ** 2).astype(np.int32)
    for row, left, right in zip(rows, dorsal_x, palmar_x, strict=True):
        image[row, left:right] = (35, 185, 200)

    for x_coordinate in (455, 485, 515, 545, 575):
        cv2.rectangle(
            image,
            (x_coordinate, 105),
            (x_coordinate + 9, 163),
            (25, 235, 245),
            -1,
        )

    if internal_streaks:
        for x_coordinate in (270, 303, 337):
            cv2.rectangle(
                image,
                (x_coordinate, 85),
                (x_coordinate + 5, 395),
                (15, 255, 255),
                -1,
            )
    if dorsal_distractor:
        cv2.rectangle(image, (320, 90), (334, 390), (215, 215, 215), -1)
    return image


def _expected_dorsal_x(rows: np.ndarray) -> np.ndarray:
    return 175.0 + 0.045 * (rows - 70.0)


def _expected_palmar_x(rows: np.ndarray) -> np.ndarray:
    return 388.0 + 0.00018 * (rows - 240.0) ** 2


def test_dorsal_boundary_selects_long_pad_transition_over_short_fins() -> None:
    region = detect_fingertip_boundary(
        _synthetic_side_view(dorsal_distractor=True)
    )
    rows = region.dorsal_boundary_xy_px[:, 1]
    expected = _expected_dorsal_x(rows)

    assert np.median(np.abs(region.dorsal_boundary_xy_px[:, 0] - expected)) < 3.0
    assert np.max(region.dorsal_boundary_xy_px[:, 0]) < 250.0


def test_dorsal_gradient_threshold_scales_to_high_resolution() -> None:
    image = _synthetic_side_view()
    high_resolution = cv2.resize(
        image,
        (1280, 960),
        interpolation=cv2.INTER_LINEAR,
    )

    region = detect_fingertip_boundary(high_resolution)
    rows = region.dorsal_boundary_xy_px[:, 1] / 2.0
    detected_x = region.dorsal_boundary_xy_px[:, 0] / 2.0

    assert np.median(np.abs(detected_x - _expected_dorsal_x(rows))) < 3.0


def test_palmar_dynamic_programming_ignores_stronger_internal_streaks() -> None:
    region = detect_fingertip_boundary(_synthetic_side_view(internal_streaks=True))
    rows = region.palmar_boundary_xy_px[:, 1]
    expected = _expected_palmar_x(rows)

    assert np.median(np.abs(region.palmar_boundary_xy_px[:, 0] - expected)) < 8.0
    assert np.median(region.palmar_boundary_xy_px[:, 0]) > 370.0


def test_search_mask_stays_between_boundaries_and_excludes_far_structure() -> None:
    region = detect_fingertip_boundary(_synthetic_side_view(internal_streaks=True))

    for row_index in np.linspace(
        0,
        len(region.dorsal_boundary_xy_px) - 1,
        7,
        dtype=int,
    ):
        dorsal_x, y_coordinate = region.dorsal_boundary_xy_px[row_index]
        palmar_x = region.palmar_boundary_xy_px[row_index, 0]
        active_x = np.flatnonzero(region.search_mask[int(y_coordinate)])
        assert active_x.size
        assert active_x[0] > dorsal_x
        assert active_x[-1] < palmar_x
        assert not region.search_mask[int(y_coordinate), 500]
