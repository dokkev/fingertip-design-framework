"""Focused tests for RealSense color-frame acquisition."""

import numpy as np

from experiments.hardware import RealSenseColorCamera


class _FakeColorFrame:
    def __init__(self, bgr: np.ndarray) -> None:
        self._bgr = bgr

    def get_data(self) -> np.ndarray:
        return self._bgr

    def get_timestamp(self) -> float:
        return 12.5

    def get_frame_number(self) -> int:
        return 7


class _FakeFrameset:
    def __init__(self, color_frame: _FakeColorFrame) -> None:
        self._color_frame = color_frame

    def get_color_frame(self) -> _FakeColorFrame:
        return self._color_frame


class _FakePipeline:
    def __init__(self, color_frame: _FakeColorFrame) -> None:
        self._frameset = _FakeFrameset(color_frame)

    def wait_for_frames(self, timeout_ms: int) -> _FakeFrameset:
        assert timeout_ms == 123
        return self._frameset


def test_read_converts_camera_bgr_to_owned_rgb() -> None:
    bgr = np.array([[[10, 20, 200]]], dtype=np.uint8)
    camera = RealSenseColorCamera(width=1, height=1)
    camera._pipeline = _FakePipeline(_FakeColorFrame(bgr))

    frame = camera.read(timeout_ms=123)
    bgr.fill(0)

    np.testing.assert_array_equal(
        frame.rgb,
        np.array([[[200, 20, 10]]], dtype=np.uint8),
    )
    assert not frame.rgb.flags.writeable
    assert frame.timestamp_ms == 12.5
    assert frame.frame_number == 7
