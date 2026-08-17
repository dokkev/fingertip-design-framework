"""Neutral physical contact state shared by case and optimization layers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True, order=True)
class ContactState:
    """One prescribed location, indentation, and indenter radius."""

    location_x_mm: float
    indentation_mm: float
    indenter_radius_mm: float

    def __post_init__(self) -> None:
        values = (
            self.location_x_mm,
            self.indentation_mm,
            self.indenter_radius_mm,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("contact-state values must be finite real numbers")
        indentation = float(self.indentation_mm)
        radius = float(self.indenter_radius_mm)
        if indentation <= 0.0:
            raise ValueError("indentation_mm must be positive")
        if radius <= 0.0:
            raise ValueError("indenter_radius_mm must be positive")
        object.__setattr__(self, "location_x_mm", float(self.location_x_mm))
        object.__setattr__(self, "indentation_mm", indentation)
        object.__setattr__(self, "indenter_radius_mm", radius)


__all__ = ["ContactState"]
