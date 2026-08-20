"""Physical LED and bulk optical-material properties."""

from __future__ import annotations

from dataclasses import dataclass
from util import require_finite, require_nonnegative, require_positive


@dataclass(frozen=True)
class LED:
    """Physical package dimensions and idealized emission properties."""

    width_mm: float = 4.0
    height_mm: float = 2.0
    relative_radiant_power: float = 1.0
    emission_half_angle_deg: float = 80.0
    emission_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        require_finite("width_mm", self.width_mm)
        require_finite("height_mm", self.height_mm)
        require_finite("relative_radiant_power", self.relative_radiant_power)
        require_finite("emission_half_angle_deg", self.emission_half_angle_deg)
        if self.width_mm <= 0.0 or self.height_mm <= 0.0:
            raise ValueError("LED width and height must be greater than zero")
        require_nonnegative("relative_radiant_power", self.relative_radiant_power)
        if not 0.0 < self.emission_half_angle_deg < 90.0:
            raise ValueError(
                "emission_half_angle_deg must be between 0 and 90 degrees"
            )
        if len(self.emission_rgb) != 3:
            raise ValueError("emission_rgb must contain three components")
        require_finite("emission_rgb[0]", self.emission_rgb[0])
        require_finite("emission_rgb[1]", self.emission_rgb[1])
        require_finite("emission_rgb[2]", self.emission_rgb[2])
        if any(value < 0.0 for value in self.emission_rgb):
            raise ValueError("emission_rgb must be finite and nonnegative")
        if not any(value > 0.0 for value in self.emission_rgb):
            raise ValueError("at least one emission_rgb component must be positive")


@dataclass(frozen=True)
class OpticalMaterial:
    """Bulk optical properties of the fingertip pad and surrounding air."""

    refractive_index_air: float = 1.0
    refractive_index_silicone: float = 1.41
    absorption_per_mm: float = 0.02
    scattering_per_mm: float = 0.23
    anisotropy_g: float = 0.0

    def __post_init__(self) -> None:
        require_finite("refractive_index_air", self.refractive_index_air)
        require_finite(
            "refractive_index_silicone", self.refractive_index_silicone
        )
        require_finite("absorption_per_mm", self.absorption_per_mm)
        require_finite("scattering_per_mm", self.scattering_per_mm)
        require_finite("anisotropy_g", self.anisotropy_g)
        require_positive("refractive_index_air", self.refractive_index_air)
        require_positive(
            "refractive_index_silicone", self.refractive_index_silicone
        )
        if self.absorption_per_mm < 0.0 or self.scattering_per_mm < 0.0:
            raise ValueError(
                "absorption_per_mm and scattering_per_mm must be nonnegative"
            )
        if not -1.0 < self.anisotropy_g < 1.0:
            raise ValueError("anisotropy_g must lie strictly between -1 and 1")

    @property
    def extinction_per_mm(self) -> float:
        return self.absorption_per_mm + self.scattering_per_mm

    @property
    def single_scattering_albedo(self) -> float:
        extinction = self.extinction_per_mm
        return 0.0 if extinction == 0.0 else self.scattering_per_mm / extinction

__all__ = ["LED", "OpticalMaterial"]
