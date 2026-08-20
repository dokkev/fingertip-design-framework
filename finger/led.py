"""Physical LED source properties."""

from __future__ import annotations

from dataclasses import dataclass

from util.validation import require_nonnegative, require_positive


@dataclass(frozen=True)
class LED:
    """Physical package dimensions and idealized emission properties."""

    width_mm: float = 4.0
    height_mm: float = 2.0
    relative_radiant_power: float = 1.0
    emission_half_angle_deg: float = 80.0

    def __post_init__(self) -> None:
        require_positive("width_mm", self.width_mm)
        require_positive("height_mm", self.height_mm)
        require_nonnegative("relative_radiant_power", self.relative_radiant_power)
        if not 0.0 < self.emission_half_angle_deg < 90.0:
            raise ValueError(
                "emission_half_angle_deg must be between 0 and 90 degrees"
            )

__all__ = ["LED"]
