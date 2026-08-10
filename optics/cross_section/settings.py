"""Numerical settings for deterministic 2D optical transport."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class TraceSettings:
    """Discretization, termination, and geometric-tolerance settings."""

    ray_count: int = 161
    max_interactions: int = 10
    minimum_ray_weight: float = 1.0e-4
    maximum_segment_count: int = 20000
    grid_width: int = 240
    grid_height: int = 240
    source_epsilon_mm: float = 1.0e-5
    intersection_epsilon_mm: float = 1.0e-6

    def __post_init__(self) -> None:
        """Reject intrinsically invalid trace settings immediately."""
        self.validate(geometry_tolerance_mm=0.0)

    def validate(self, *, geometry_tolerance_mm: float) -> None:
        """Validate numerical settings against a domain length tolerance."""
        scalars = {
            "minimum_ray_weight": self.minimum_ray_weight,
            "source_epsilon_mm": self.source_epsilon_mm,
            "intersection_epsilon_mm": self.intersection_epsilon_mm,
            "geometry_tolerance_mm": geometry_tolerance_mm,
        }
        for name, value in scalars.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.minimum_ray_weight <= 0.0:
            raise ValueError("minimum_ray_weight must be greater than zero")
        if geometry_tolerance_mm < 0.0:
            raise ValueError("geometry_tolerance_mm must be nonnegative")
        if self.source_epsilon_mm <= geometry_tolerance_mm:
            raise ValueError(
                "source_epsilon_mm must exceed the geometry tolerance"
            )
        if self.intersection_epsilon_mm <= geometry_tolerance_mm:
            raise ValueError(
                "intersection_epsilon_mm must exceed the geometry tolerance"
            )

        integer_minimums = {
            "ray_count": (self.ray_count, 3),
            "max_interactions": (self.max_interactions, 1),
            "maximum_segment_count": (self.maximum_segment_count, 1),
            "grid_width": (self.grid_width, 16),
            "grid_height": (self.grid_height, 16),
        }
        for name, (value, minimum) in integer_minimums.items():
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer of at least {minimum}")
