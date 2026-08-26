"""Derived carrier-silicone bonding interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


Point2D = tuple[float, float]


@dataclass(frozen=True)
class BondingInterface:
    """Left and right polylines receiving the perfect kinematic bond."""

    left: tuple[Point2D, ...]
    right: tuple[Point2D, ...]

    def __post_init__(self) -> None:
        for name in ("left", "right"):
            polyline = tuple(
                (float(x_mm), float(z_mm)) for x_mm, z_mm in getattr(self, name)
            )
            if len(polyline) < 2:
                raise ValueError(f"bonding interface {name} needs at least two points")
            if any(not isfinite(value) for point in polyline for value in point):
                raise ValueError(f"bonding interface {name} must contain finite points")
            object.__setattr__(self, name, polyline)

__all__ = ["BondingInterface"]
