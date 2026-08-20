"""Physical LED and bulk optical-material properties."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class LED:
    """Physical package dimensions and idealized emission properties."""

    width_mm: float = 4.0
    height_mm: float = 2.0
    relative_radiant_power: float = 1.0
    emission_half_angle_deg: float = 80.0

    def __post_init__(self) -> None:
        _require_finite("width_mm", self.width_mm)
        _require_finite("height_mm", self.height_mm)
        _require_finite("relative_radiant_power", self.relative_radiant_power)
        _require_finite("emission_half_angle_deg", self.emission_half_angle_deg)
        if self.width_mm <= 0.0 or self.height_mm <= 0.0:
            raise ValueError("LED width and height must be greater than zero")
        if self.relative_radiant_power < 0.0:
            raise ValueError("relative_radiant_power must be nonnegative")
        if not 0.0 < self.emission_half_angle_deg < 90.0:
            raise ValueError(
                "emission_half_angle_deg must be between 0 and 90 degrees"
            )


@dataclass(frozen=True)
class OpticalMaterial:
    """Optical properties consumed by the current FULL_3D transport."""

    refractive_index_air: float = 1.0
    refractive_index_silicone: float = 1.41
    absorption_per_mm: float = 0.02

    def __post_init__(self) -> None:
        _require_finite("refractive_index_air", self.refractive_index_air)
        _require_finite(
            "refractive_index_silicone", self.refractive_index_silicone
        )
        _require_finite("absorption_per_mm", self.absorption_per_mm)
        if self.refractive_index_air <= 0.0:
            raise ValueError("refractive_index_air must be greater than zero")
        if self.refractive_index_silicone <= 0.0:
            raise ValueError(
                "refractive_index_silicone must be greater than zero"
            )
        if self.absorption_per_mm < 0.0:
            raise ValueError("absorption_per_mm must be nonnegative")

__all__ = ["LED", "OpticalMaterial"]
