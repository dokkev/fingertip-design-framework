"""Physical optical-material metadata for the fingertip sensor."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


class InvalidOpticalMaterialParameters(ValueError):
    """Raised when optical-material properties are invalid."""


@dataclass(frozen=True)
class OpticalMaterialParameters:
    """Provisional bulk optical properties for the fingertip materials."""

    refractive_index_air: float = 1.0
    refractive_index_silicone: float = 1.41
    absorption_per_mm: float = 0.02
    scattering_per_mm: float = 0.23
    anisotropy_g: float = 0.0

    def __post_init__(self) -> None:
        """Validate qualitative bulk optical properties."""
        values = {
            "refractive_index_air": self.refractive_index_air,
            "refractive_index_silicone": self.refractive_index_silicone,
            "absorption_per_mm": self.absorption_per_mm,
            "scattering_per_mm": self.scattering_per_mm,
            "anisotropy_g": self.anisotropy_g,
        }
        for name, value in values.items():
            if not isfinite(value):
                raise InvalidOpticalMaterialParameters(f"{name} must be finite")
        if self.refractive_index_air <= 0.0:
            raise InvalidOpticalMaterialParameters(
                "refractive_index_air must be greater than zero"
            )
        if self.refractive_index_silicone <= 0.0:
            raise InvalidOpticalMaterialParameters(
                "refractive_index_silicone must be greater than zero"
            )
        if self.absorption_per_mm < 0.0 or self.scattering_per_mm < 0.0:
            raise InvalidOpticalMaterialParameters(
                "absorption_per_mm and scattering_per_mm must be nonnegative"
            )
        if not -1.0 < self.anisotropy_g < 1.0:
            raise InvalidOpticalMaterialParameters(
                "anisotropy_g must lie strictly between -1 and 1"
            )

    @property
    def extinction_per_mm(self) -> float:
        """Return total absorption-plus-scattering extinction per millimeter."""
        return self.absorption_per_mm + self.scattering_per_mm

    @property
    def single_scattering_albedo(self) -> float:
        """Return the scattering fraction of total extinction."""
        extinction = self.extinction_per_mm
        if extinction == 0.0:
            return 0.0
        return self.scattering_per_mm / extinction
