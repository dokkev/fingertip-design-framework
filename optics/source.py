"""Compatibility wrappers for sensor-owned LED source coordinates."""

from __future__ import annotations

from model.fingertip_model import FingertipModel
from model.fingertip_sensor_model import FingertipSensorModel


def _sensor(value: FingertipSensorModel | FingertipModel) -> FingertipSensorModel:
    if isinstance(value, FingertipSensorModel):
        return value
    return FingertipSensorModel.from_geometry(value)


def led_source_position_2d(
    sensor_model: FingertipSensorModel | FingertipModel,
) -> tuple[float, float]:
    """Delegate to the physical sensor model's source position."""
    return _sensor(sensor_model).led_source_position_2d


def led_source_position_3d(
    sensor_model: FingertipSensorModel | FingertipModel,
) -> tuple[float, float, float]:
    """Delegate to the physical sensor model's three-dimensional source."""
    return _sensor(sensor_model).led_source_position_3d
