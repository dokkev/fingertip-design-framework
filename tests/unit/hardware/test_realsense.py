"""Focused tests for RealSense photometric-control locking."""

from types import SimpleNamespace

import pytest

from experiments.hardware import RealSenseColorCamera


class _FakeColorSensor:
    def __init__(self) -> None:
        self.values = {
            "enable_auto_exposure": 1.0,
            "exposure": 166.0,
            "gain": 64.0,
            "enable_auto_white_balance": 1.0,
            "white_balance": 4600.0,
        }
        self.calls: list[tuple[str, str, float | None]] = []

    def supports(self, option: str) -> bool:
        return option in self.values

    def get_option(self, option: str) -> float:
        self.calls.append(("get", option, None))
        return self.values[option]

    def set_option(self, option: str, value: float) -> None:
        self.calls.append(("set", option, value))
        self.values[option] = value


def test_photometric_lock_retains_values_before_disabling_auto_controls() -> None:
    option = SimpleNamespace(
        enable_auto_exposure="enable_auto_exposure",
        exposure="exposure",
        gain="gain",
        enable_auto_white_balance="enable_auto_white_balance",
        white_balance="white_balance",
    )
    sensor = _FakeColorSensor()
    camera = RealSenseColorCamera()
    camera._pipeline = object()
    camera._rs = SimpleNamespace(option=option)
    camera._color_sensor = sensor

    locked = camera.lock_color_photometric_controls()

    assert locked == {
        "exposure": 166.0,
        "gain": 64.0,
        "white_balance": 4600.0,
    }
    assert sensor.values["enable_auto_exposure"] == 0.0
    assert sensor.values["enable_auto_white_balance"] == 0.0
    assert sensor.calls.index(("get", "exposure", None)) < sensor.calls.index(
        ("set", "enable_auto_exposure", 0.0)
    )
    assert sensor.calls.index(("get", "white_balance", None)) < sensor.calls.index(
        ("set", "enable_auto_white_balance", 0.0)
    )


def test_photometric_lock_fails_explicitly_without_exposure_control() -> None:
    option = SimpleNamespace(
        enable_auto_exposure="enable_auto_exposure",
        exposure="exposure",
        gain="gain",
        enable_auto_white_balance="enable_auto_white_balance",
        white_balance="white_balance",
    )
    sensor = _FakeColorSensor()
    del sensor.values["enable_auto_exposure"]
    camera = RealSenseColorCamera()
    camera._pipeline = object()
    camera._rs = SimpleNamespace(option=option)
    camera._color_sensor = sensor

    with pytest.raises(RuntimeError, match="cannot freeze exposure"):
        camera.lock_color_photometric_controls()
