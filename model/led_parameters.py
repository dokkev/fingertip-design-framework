"""Physical LED package and idealized emission parameters."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


class InvalidLEDParameters(ValueError):
    """Raised when LED package or emission properties are invalid."""


@dataclass(frozen=True)
class LEDParameters:
    """Physical LED package and idealized emission parameters."""

    width_mm: float = 4.0
    height_mm: float = 2.0
    relative_radiant_power: float = 1.0
    emission_half_angle_deg: float = 80.0
    emission_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        """Validate physical package and provisional emission properties."""
        scalars = {
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "relative_radiant_power": self.relative_radiant_power,
            "emission_half_angle_deg": self.emission_half_angle_deg,
        }
        for name, value in scalars.items():
            if not isfinite(value):
                raise InvalidLEDParameters(f"{name} must be finite")
        if self.width_mm <= 0.0 or self.height_mm <= 0.0:
            raise InvalidLEDParameters(
                "width_mm and height_mm must be greater than zero"
            )
        if self.relative_radiant_power < 0.0:
            raise InvalidLEDParameters(
                "relative_radiant_power must be nonnegative"
            )
        if not 0.0 < self.emission_half_angle_deg < 90.0:
            raise InvalidLEDParameters(
                "emission_half_angle_deg must be between 0 and 90 degrees"
            )
        if len(self.emission_rgb) != 3:
            raise InvalidLEDParameters("emission_rgb must contain three components")
        if any(not isfinite(component) for component in self.emission_rgb):
            raise InvalidLEDParameters("emission_rgb components must be finite")
        if any(component < 0.0 for component in self.emission_rgb):
            raise InvalidLEDParameters(
                "emission_rgb components must be nonnegative"
            )
        if not any(component > 0.0 for component in self.emission_rgb):
            raise InvalidLEDParameters(
                "at least one emission_rgb component must be positive"
            )
