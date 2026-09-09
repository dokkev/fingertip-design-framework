import cv2
import numpy as np
import pytest

from algorithm.fingertip_segmentation import segment_fingertip


def _synthetic_finger() -> np.ndarray:
    rgb = np.full((240, 240, 3), 10, dtype=np.uint8)
    cv2.rectangle(rgb, (70, 35), (150, 205), (20, 180, 190), -1)
    cv2.circle(rgb, (110, 35), 35, (20, 180, 190), -1)
    cv2.circle(rgb, (110, 205), 35, (20, 180, 190), -1)
    cv2.rectangle(rgb, (175, 55), (225, 190), (240, 240, 240), -1)
    return rgb


def test_segment_fingertip_selects_cyan_finger_not_white_reflector() -> None:
    result = segment_fingertip(_synthetic_finger())

    assert result.mask[100, 110]
    assert not result.mask[100, 200]
    assert np.all(result.search_mask <= result.mask)
    assert result.dorsal_boundary_xy_px.shape == result.palmar_boundary_xy_px.shape
    assert result.estimated_width_px > 60.0


def test_segment_fingertip_rejects_larger_dim_cyan_fixture() -> None:
    rgb = _synthetic_finger()
    cv2.rectangle(rgb, (175, 25), (225, 225), (18, 70, 75), -1)

    result = segment_fingertip(rgb)

    assert result.mask[100, 110]
    assert not result.mask[100, 200]


@pytest.mark.parametrize(
    "rgb",
    [
        np.zeros((20, 20), dtype=np.uint8),
        np.zeros((20, 20, 3), dtype=np.uint8),
        np.zeros((64, 64, 3), dtype=np.float32),
    ],
)
def test_segment_fingertip_rejects_malformed_rgb(rgb: np.ndarray) -> None:
    with pytest.raises(ValueError):
        segment_fingertip(rgb)
