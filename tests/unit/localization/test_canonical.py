"""Focused tests for finger-relative canonical image coordinates."""

import numpy as np

from experiments.localization import (
    CanonicalFingerConfig,
    FingertipBoundaryRegion,
    build_canonical_finger_map,
    similarity_from_landmarks,
    transform_canonical_map,
    warp_to_canonical,
)


def _region(
    *,
    slope: float = 0.0,
    core_y_span: tuple[int, int] = (10, 90),
) -> FingertipBoundaryRegion:
    rows = np.arange(10, 90, dtype=np.float64)
    left = 20.0 + slope * (rows - 10.0)
    right = left + 60.0
    mask = np.zeros((100, 140), dtype=bool)
    for row, start, stop in zip(rows.astype(int), left, right, strict=True):
        mask[row, int(np.ceil(start)) : int(np.floor(stop)) + 1] = True
    return FingertipBoundaryRegion(
        dorsal_boundary_xy_px=np.column_stack((left, rows)),
        palmar_boundary_xy_px=np.column_stack((right, rows)),
        search_mask=mask,
        core_y_span=core_y_span,
        estimated_pad_width_px=60.0,
    )


def _finger_texture(region: FingertipBoundaryRegion) -> np.ndarray:
    image = np.zeros((*region.search_mask.shape, 3), dtype=np.uint8)
    rows = region.dorsal_boundary_xy_px[:, 1].astype(int)
    left = region.dorsal_boundary_xy_px[:, 0]
    right = region.palmar_boundary_xy_px[:, 0]
    for index, row in enumerate(rows):
        columns = np.arange(image.shape[1])
        u = np.clip((columns - left[index]) / (right[index] - left[index]), 0.0, 1.0)
        v = (row - rows[0]) / (rows[-1] - rows[0])
        image[row, :, 0] = np.rint(255.0 * v).astype(np.uint8)
        image[row, :, 1] = np.rint(255.0 * u).astype(np.uint8)
    return image


def test_straight_finger_warp_preserves_longitudinal_and_transverse_order() -> None:
    region = _region()
    canonical_map = build_canonical_finger_map(
        region,
        CanonicalFingerConfig(
            output_height=16,
            output_width=12,
            transverse_inset_fraction=0.0,
        ),
    )
    warped = warp_to_canonical(_finger_texture(region), canonical_map)

    assert np.all(np.diff(warped[:, warped.shape[1] // 2, 0].astype(int)) >= 0)
    assert np.all(np.diff(warped[warped.shape[0] // 2, :, 1].astype(int)) >= 0)
    assert warped[-1, -1, 0] > warped[0, 0, 0]
    assert warped[-1, -1, 1] > warped[-1, 0, 1]


def test_slanted_finger_texture_maps_to_same_canonical_coordinates() -> None:
    straight = _region()
    slanted = _region(slope=0.35)
    straight_warp = warp_to_canonical(
        _finger_texture(straight),
        build_canonical_finger_map(
            straight,
            CanonicalFingerConfig(output_height=32, output_width=24),
        ),
    )
    slanted_warp = warp_to_canonical(
        _finger_texture(slanted),
        build_canonical_finger_map(
            slanted,
            CanonicalFingerConfig(output_height=32, output_width=24),
        ),
    )

    assert np.mean(np.abs(straight_warp.astype(float) - slanted_warp.astype(float))) < 1.5


def test_similarity_transform_moves_reference_canonical_map() -> None:
    region = _region()
    reference_map = build_canonical_finger_map(
        region,
        CanonicalFingerConfig(output_height=8, output_width=6),
    )
    reference = np.array(((30.0, 20.0), (30.0, 40.0), (30.0, 60.0)))
    angle = np.deg2rad(12.0)
    expected_linear = 1.08 * np.array(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    expected_translation = np.array((7.0, -4.0))
    current = reference @ expected_linear.T + expected_translation

    transform = similarity_from_landmarks(reference, current)
    moved = transform_canonical_map(reference_map, transform)
    expected_x = (
        expected_linear[0, 0] * reference_map.map_x
        + expected_linear[0, 1] * reference_map.map_y
        + expected_translation[0]
    )
    expected_y = (
        expected_linear[1, 0] * reference_map.map_x
        + expected_linear[1, 1] * reference_map.map_y
        + expected_translation[1]
    )

    assert np.allclose(transform[:, :2], expected_linear)
    assert np.allclose(transform[:, 2], expected_translation)
    assert np.allclose(moved.map_x, expected_x)
    assert np.allclose(moved.map_y, expected_y)


def test_full_silhouette_span_keeps_distal_and_proximal_response_rows() -> None:
    region = _region(core_y_span=(20, 75))
    image = np.zeros((*region.search_mask.shape, 3), dtype=np.uint8)
    image[14, 20:81, 0] = 255
    image[86, 20:81, 0] = 255

    full = warp_to_canonical(
        image,
        build_canonical_finger_map(
            region,
            CanonicalFingerConfig(
                output_height=80,
                output_width=16,
                transverse_inset_fraction=0.0,
            ),
        ),
    )
    core = warp_to_canonical(
        image,
        build_canonical_finger_map(
            region,
            CanonicalFingerConfig(
                output_height=56,
                output_width=16,
                transverse_inset_fraction=0.0,
                longitudinal_span="core",
            ),
        ),
    )

    assert np.max(full[:8, :, 0]) == 255
    assert np.max(full[-8:, :, 0]) == 255
    assert np.max(core[:, :, 0]) == 0
