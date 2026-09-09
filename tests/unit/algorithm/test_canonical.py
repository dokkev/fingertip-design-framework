import cv2
import numpy as np

from algorithm.canonical import (
    CanonicalFingerMap,
    build_canonical_map,
    transform_canonical_map,
)
from algorithm.fingertip_segmentation import segment_fingertip


def test_known_similarity_transform_moves_sampling_map() -> None:
    source = CanonicalFingerMap(
        map_x=np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
        map_y=np.asarray([[5.0, 5.0], [7.0, 7.0]], dtype=np.float32),
    )
    angle = np.deg2rad(20.0)
    scale = 1.08
    transform = np.asarray(
        [
            [scale * np.cos(angle), -scale * np.sin(angle), 12.0],
            [scale * np.sin(angle), scale * np.cos(angle), -4.0],
        ]
    )

    moved = transform_canonical_map(source, transform)

    homogeneous = np.column_stack(
        (source.map_x.ravel(), source.map_y.ravel(), np.ones(source.map_x.size))
    )
    expected = homogeneous @ transform.T
    np.testing.assert_allclose(moved.map_x.ravel(), expected[:, 0], atol=1.0e-6)
    np.testing.assert_allclose(moved.map_y.ravel(), expected[:, 1], atol=1.0e-6)


def test_canonical_map_follows_oblique_fingertip_axis() -> None:
    rgb = np.full((300, 300, 3), 10, dtype=np.uint8)
    box = cv2.boxPoints(((150.0, 150.0), (70.0, 220.0), -35.0)).astype(np.int32)
    cv2.fillConvexPoly(rgb, box, (20, 180, 190))

    canonical = build_canonical_map(segment_fingertip(rgb))
    centerline = np.column_stack(
        (
            np.mean(canonical.map_x, axis=1),
            np.mean(canonical.map_y, axis=1),
        )
    )

    assert centerline[-1, 1] > centerline[0, 1]
    assert np.linalg.norm(centerline[-1] - centerline[0]) > 150.0
