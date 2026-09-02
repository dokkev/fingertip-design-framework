"""Focused tests for paired-LSD side-view fingertip boundaries."""

import cv2
import numpy as np
import pytest

from experiments.localization import detect_fingertip_boundary
from experiments.localization.fingertip_segmentation import (
    _LineFit,
    _detect_paired_lsd_prior,
    _paired_boundaries,
    segment_fingertip,
)


def _synthetic_side_view(
    *,
    fragmented_dorsal: bool = False,
    wrong_polarity_distractor: bool = False,
    internal_streaks_and_fins: bool = False,
    saturated_center: bool = False,
    horizontal_bridge: bool = False,
    brighter_distant_object: bool = False,
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
    if saturated_center:
        cv2.ellipse(image, (288, 235), (48, 82), 0, 0, 360, (250, 250, 250), -1)
    if horizontal_bridge:
        cv2.rectangle(image, (380, 218), (570, 224), (25, 240, 250), -1)
        cv2.rectangle(image, (535, 115), (620, 350), (25, 250, 255), -1)
    if brighter_distant_object:
        cv2.circle(image, (555, 240), 78, (10, 255, 255), -1)
    return image


def _expected_dorsal_x(rows: np.ndarray) -> np.ndarray:
    return 175.0 + 0.045 * (rows - 70.0)


def _expected_palmar_x(rows: np.ndarray) -> np.ndarray:
    return 382.0 + 0.015 * (rows - 70.0)


def test_fragmented_dorsal_segments_are_combined_into_one_boundary() -> None:
    region = _detect_paired_lsd_prior(
        _synthetic_side_view(fragmented_dorsal=True)
    ).region
    rows = region.dorsal_boundary_xy_px[:, 1]

    assert np.median(
        np.abs(region.dorsal_boundary_xy_px[:, 0] - _expected_dorsal_x(rows))
    ) < 3.0
    assert region.core_y_span[1] - region.core_y_span[0] > 250


def test_dorsal_selection_rejects_a_long_wrong_polarity_line() -> None:
    region = _detect_paired_lsd_prior(
        _synthetic_side_view(wrong_polarity_distractor=True)
    ).region
    rows = region.dorsal_boundary_xy_px[:, 1]

    assert np.median(
        np.abs(region.dorsal_boundary_xy_px[:, 0] - _expected_dorsal_x(rows))
    ) < 3.0
    assert np.median(region.dorsal_boundary_xy_px[:, 0]) > 150.0


def test_palmar_pair_excludes_internal_streaks_and_far_fins() -> None:
    region = _detect_paired_lsd_prior(
        _synthetic_side_view(internal_streaks_and_fins=True)
    ).region
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
    core_margin = round(0.10 * (region.core_y_span[1] - region.core_y_span[0]))
    for y_coordinate in np.linspace(
        region.core_y_span[0] + core_margin,
        region.core_y_span[1] - core_margin - 1,
        7,
        dtype=int,
    ):
        index = int(y_coordinate - region.dorsal_boundary_xy_px[0, 1])
        dorsal_x = region.dorsal_boundary_xy_px[index, 0]
        palmar_x = region.palmar_boundary_xy_px[index, 0]
        active_x = np.flatnonzero(region.search_mask[int(y_coordinate)])
        assert active_x.size
        assert active_x[0] > dorsal_x
        assert active_x[-1] < palmar_x


def test_high_resolution_geometry_is_mapped_back_to_camera_coordinates() -> None:
    image = _synthetic_side_view(internal_streaks_and_fins=True)
    native = segment_fingertip(image)
    high_resolution = segment_fingertip(
        cv2.resize(image, (1280, 960), interpolation=cv2.INTER_NEAREST)
    )

    expected_mask = cv2.resize(
        native.final_mask.astype(np.uint8),
        (1280, 960),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    assert np.array_equal(high_resolution.final_mask, expected_mask)
    assert high_resolution.region.search_mask.shape == (960, 1280)
    assert high_resolution.geometry_scale == pytest.approx(0.5)
    assert high_resolution.region.estimated_pad_width_px == pytest.approx(
        2.0 * native.region.estimated_pad_width_px,
        abs=2.0,
    )


def test_saturated_luminous_interior_is_filled() -> None:
    diagnostics = segment_fingertip(
        _synthetic_side_view(saturated_center=True)
    )

    assert np.all(diagnostics.final_mask[190:280, 255:320])


def test_thin_bridge_does_not_extend_the_final_fingertip() -> None:
    diagnostics = segment_fingertip(
        _synthetic_side_view(horizontal_bridge=True)
    )

    assert np.max(diagnostics.contour_xy_px[:, 0]) < 470.0
    assert not np.any(diagnostics.final_mask[:, 535:])


def test_brighter_distant_object_is_not_selected() -> None:
    diagnostics = segment_fingertip(
        _synthetic_side_view(brighter_distant_object=True)
    )
    rows, columns = np.nonzero(diagnostics.final_mask)

    assert float(np.mean(columns)) < 400.0
    assert not np.any(diagnostics.final_mask[:, 500:])


def test_regularized_contour_is_closed_finite_and_has_no_extreme_spike() -> None:
    diagnostics = segment_fingertip(
        _synthetic_side_view(
            saturated_center=True,
            horizontal_bridge=True,
        )
    )
    contour = diagnostics.contour_xy_px
    closed_segments = np.diff(np.vstack((contour, contour[0])), axis=0)

    assert contour.shape == (256, 2)
    assert np.all(np.isfinite(contour))
    assert np.any(diagnostics.final_mask)
    assert np.max(np.linalg.norm(closed_segments, axis=1)) < (
        0.20 * diagnostics.region.estimated_pad_width_px
    )
