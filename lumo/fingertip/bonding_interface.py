"""Derived carrier-silicone bonding interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Self

from shapely import LineString, line_merge


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

    def clipped_to(self, geometry_boundary: BondingInterface) -> Self:
        """Clip this interface to the actual carrier-silicone boundary."""
        return type(self)(
            left=_clip_polyline(
                "left",
                self.left,
                geometry_boundary.left,
            ),
            right=_clip_polyline(
                "right",
                self.right,
                geometry_boundary.right,
            ),
        )


def _clip_polyline(
    name: str,
    requested: tuple[Point2D, ...],
    geometry_boundary: tuple[Point2D, ...],
) -> tuple[Point2D, ...]:
    requested_line = LineString(requested)
    clipped = line_merge(
        requested_line.intersection(LineString(geometry_boundary))
    )

    if requested_line.equals(clipped):
        return requested

    print(
        f"[WARNING] bonding interface {name} extends beyond the actual "
        "carrier-silicone boundary and was clipped"
    )

    if clipped.geom_type != "LineString" or clipped.is_empty:
        raise ValueError(
            f"bonding interface {name} has no single connected segment "
            "inside the actual carrier-silicone boundary"
        )

    return tuple(
        (float(x_mm), float(z_mm))
        for x_mm, z_mm in clipped.coords
    )


__all__ = ["BondingInterface"]
