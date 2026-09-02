"""Focused tests for paired-LSD side-view fingertip boundaries."""

import cv2
import numpy as np
import pytest

from experiments.localization import detect_fingertip_boundary
from experiments.localization.fingertip_boundary import _LineFit, _paired_boundaries


def _synthetic_side_view(
    *,
    fragmented_dorsal: bool = False,
    wrong_polarity_distractor: bool = False,
    internal_streaks_and_fins: bool = False,
) -> np.ndarray:
    height, width = 480, 640
    image = np.full((height, width, 3), 45, dtype=np.uint8)
    rows = np.arange(70, 411)
    dorsal_x = np.rint(175.0 + 0.045 * (rows - 70)).astype(np.int32)
    palmar_x = np.rint(382.0 + 0.015 * (rows - 70)).astype(np.int32)
    for row, left, right in zip(rows, dorsal_x, palmar_x, strict=True):
        image[row, :left] = (155, 155, 155)
        image[row, left:right] = (25, 230, 245)

    if fragmented_dorsal:
        for y_start, y_stop in ((137, 162), (247, 273), (348, 369)):
            for row in range(y_start, y_stop):
                left = int(round(175.0 + 0.045 * (row - 70)))
                image[row, left - 18 : left + 18] = (225, 225, 225)

    if wrong_polarity_distractor:
        image[85:396, :88] = (25, 230, 245)

    if internal_streaks_and_fins:
        for x_coordinate in (260, 298, 334):
            cv2.rectangle(
                image,
                (x_coordinate, 95),
                (x_coordinate + 5, 390),
                (245, 245, 245),
                -1,
            )
        for x_coordinate in (490, 525, 560, 595):
            cv2.rectangle(
                image,
                (x_coordinate, 105),
                (x_coordinate + 10, 180),
                (25, 230, 245),
                -1,
            )
    return image


def _expected_dorsal_x(rows: np.ndarray) -> np.ndarray:
    return 175.0 + 0.045 * (rows - 70.0)


def _expected_palmar_x(rows: np.ndarray) -> np.ndarray:
    return 382.0 + 0.015 * (rows - 70.0)


def test_fragmented_dorsal_segments_are_combined_into_one_boundary() -> None:
    region = detect_fingertip_boundary(
        _synthetic_side_view(fragmented_dorsal=True)
    )
    rows = region.dorsal_boundary_xy_px[:, 1]

    assert np.median(
        np.abs(region.dorsal_boundary_xy_px[:, 0] - _expected_dorsal_x(rows))
    ) < 3.0
    assert region.core_y_span[1] - region.core_y_span[0] > 250


def test_dorsal_selection_rejects_a_long_wrong_polarity_line() -> None:
    region = detect_fingertip_boundary(
        _synthetic_side_view(wrong_polarity_distractor=True)
    )
    rows = region.dorsal_boundary_xy_px[:, 1]

    assert np.median(
        np.abs(region.dorsal_boundary_xy_px[:, 0] - _expected_dorsal_x(rows))
    ) < 3.0
    assert np.median(region.dorsal_boundary_xy_px[:, 0]) > 150.0


def test_palmar_pair_excludes_internal_streaks_and_far_fins() -> None:
    region = detect_fingertip_boundary(
        _synthetic_side_view(internal_streaks_and_fins=True)
    )
    rows = region.palmar_boundary_xy_px[:, 1]

    assert np.median(
        np.abs(region.palmar_boundary_xy_px[:, 0] - _expected_palmar_x(rows))
    ) < 5.0
    assert np.median(region.palmar_boundary_xy_px[:, 0]) > 370.0
    assert not np.any(region.search_mask[:, 490:])


def test_paired_width_sanity_rejects_crossing_and_implausible_pairs() -> None:
    dorsal = _LineFit(vx=0.0, vy=1.0, x0=180.0, y0=200.0)
    support = (80.0, 400.0)

    reasonable = _LineFit(vx=0.03, vy=1.0, x0=385.0, y0=200.0)
    _, _, mask, _, width = _paired_boundaries(
        (480, 640),
        dorsal,
        support,
        reasonable,
        support,
    )
    assert np.any(mask)
    assert 190.0 < width < 220.0

    crossing = _LineFit(vx=-0.8, vy=0.6, x0=190.0, y0=200.0)
    with pytest.raises(RuntimeError, match="cross"):
        _paired_boundaries(
            (480, 640),
            dorsal,
            support,
            crossing,
            support,
        )

    too_narrow = _LineFit(vx=0.0, vy=1.0, x0=210.0, y0=200.0)
    with pytest.raises(RuntimeError, match="implausible"):
        _paired_boundaries(
            (480, 640),
            dorsal,
            support,
            too_narrow,
            support,
        )


def test_search_mask_remains_strictly_between_fitted_boundaries() -> None:
    region = detect_fingertip_boundary(
        _synthetic_side_view(internal_streaks_and_fins=True)
    )
    for index in np.linspace(
        0,
        len(region.dorsal_boundary_xy_px) - 1,
        7,
        dtype=int,
    ):
        dorsal_x, y_coordinate = region.dorsal_boundary_xy_px[index]
        palmar_x = region.palmar_boundary_xy_px[index, 0]
        active_x = np.flatnonzero(region.search_mask[int(y_coordinate)])
        assert active_x.size
        assert active_x[0] > dorsal_x
        assert active_x[-1] < palmar_x
