"""Focused tests for RealSense color-frame acquisition."""

from types import SimpleNamespace
import sys

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


class _FakeColorSensor:
    def __init__(self) -> None:
        self.values = {
            "enable_auto_exposure": 1.0,
            "exposure": 166.0,
            "gain": 64.0,
            "enable_auto_white_balance": 1.0,
            "white_balance": 4600.0,
        }
        self.ranges = {
            "enable_auto_exposure": SimpleNamespace(
                min=0.0, max=1.0, step=1.0
            ),
            "exposure": SimpleNamespace(min=1.0, max=10000.0, step=1.0),
            "gain": SimpleNamespace(min=0.0, max=128.0, step=1.0),
            "enable_auto_white_balance": SimpleNamespace(
                min=0.0, max=1.0, step=1.0
            ),
            "white_balance": SimpleNamespace(
                min=2800.0, max=6500.0, step=10.0
            ),
        }

    def supports(self, option: str) -> bool:
        return option in self.values

    def get_option_range(self, option: str) -> SimpleNamespace:
        return self.ranges[option]

    def set_option(self, option: str, value: float) -> None:
        self.values[option] = float(value)

    def get_option(self, option: str) -> float:
        return self.values[option]


class _FakeDevice:
    def __init__(self, sensor: _FakeColorSensor) -> None:
        self.sensor = sensor

    def query_sensors(self) -> list[_FakeColorSensor]:
        return [self.sensor]


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


def test_manual_photometric_controls_are_disabled_set_and_read_back(
    monkeypatch,
) -> None:
    options = SimpleNamespace(
        enable_auto_exposure="enable_auto_exposure",
        exposure="exposure",
        gain="gain",
        enable_auto_white_balance="enable_auto_white_balance",
        white_balance="white_balance",
    )
    monkeypatch.setitem(
        sys.modules,
        "pyrealsense2",
        SimpleNamespace(option=options),
    )
    sensor = _FakeColorSensor()
    camera = RealSenseColorCamera(width=1, height=1)
    camera._pipeline = object()
    camera._device = _FakeDevice(sensor)

    camera.set_manual_photometric_controls(
        exposure_us=120.0,
        gain=16.0,
        white_balance_k=4500.0,
    )

    assert sensor.values["enable_auto_exposure"] == 0.0
    assert sensor.values["enable_auto_white_balance"] == 0.0
    assert camera.exposure_us == 120.0
    assert camera.gain == 16.0
    assert camera.white_balance_k == 4500.0
