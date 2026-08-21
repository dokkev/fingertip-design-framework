"""Optical material and LED source parameters for the LUMO fingertip."""

from __future__ import annotations

from dataclasses import dataclass

from lumo2.util.scalar_validation import require_nonnegative, require_positive

from .geometric_param import InvalidFingertipParameters


@dataclass(frozen=True)
class OpticalParameters:
    """Bulk optical properties of the compliant silicone pad."""

    refractive_index: float = 1.41
    absorption_per_mm: float = 0.02

    def __post_init__(self) -> None:
        error_type = InvalidFingertipParameters

        require_positive(
            "refractive_index",
            self.refractive_index,
            error_type=error_type,
        )
        require_nonnegative(
            "absorption_per_mm",
            self.absorption_per_mm,
            error_type=error_type,
        )


@dataclass(frozen=True)
class LEDParameters:
    """Physical and optical parameters of the embedded LED."""

    width_mm: float = 4.0
    height_mm: float = 2.0

    relative_radiant_power: float = 1.0
    emission_half_angle_deg: float = 80.0

    def __post_init__(self) -> None:
        error_type = InvalidFingertipParameters

        require_positive(
            "width_mm",
            self.width_mm,
            error_type=error_type,
        )
        require_positive(
            "height_mm",
            self.height_mm,
            error_type=error_type,
        )
        require_positive(
            "relative_radiant_power",
            self.relative_radiant_power,
            error_type=error_type,
        )

        if not 0.0 < self.emission_half_angle_deg < 90.0:
            raise error_type(
                "emission_half_angle_deg must be between 0 and 90 degrees"
            )


__all__ = ["LEDParameters", "OpticalParameters"]
