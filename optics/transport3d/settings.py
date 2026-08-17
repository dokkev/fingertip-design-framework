"""Numerical settings for deterministic camera-independent 3D transport."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal


Transport3DMode = Literal["planar", "full3d"]


@dataclass(frozen=True)
class Transport3DSettings:
    """Discretization, termination, geometry, and field settings.

    The defaults mirror the reduced tracer where the two models share a
    convention.  The fixed 11 mm cell is a periodic numerical cell; its caps
    are never optical escape surfaces.
    """

    mode: Transport3DMode = "full3d"
    ray_count: int = 4096
    max_interactions: int = 10
    minimum_ray_weight: float = 1.0e-4
    maximum_segment_count: int = 20000
    maximum_periodic_wraps: int = 32
    extrusion_depth_mm: float = 11.0
    surface_u_bins: int = 128
    surface_z_bins: int = 64
    projected_grid_width: int = 240
    projected_grid_height: int = 240
    internal_grid_width: int = 240
    internal_grid_height: int = 240
    internal_z_bins: int = 32
    x_bounds_mm: tuple[float, float] | None = None
    y_bounds_mm: tuple[float, float] | None = None
    source_epsilon_mm: float = 1.0e-5
    intersection_epsilon_mm: float = 1.0e-6
    energy_balance_tolerance: float = 1.0e-5
    retain_projected_segments: bool = False
    retain_internal_path_field: bool = False
    terminate_on_periodic_wrap_limit: bool = False
    terminate_on_no_event: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.mode not in ("planar", "full3d"):
            raise ValueError("mode must be 'planar' or 'full3d'")
        integer_minimums = {
            "ray_count": (self.ray_count, 3),
            "max_interactions": (self.max_interactions, 1),
            "maximum_segment_count": (self.maximum_segment_count, 1),
            "maximum_periodic_wraps": (self.maximum_periodic_wraps, 1),
            "surface_u_bins": (self.surface_u_bins, 1),
            "surface_z_bins": (self.surface_z_bins, 1),
            "projected_grid_width": (self.projected_grid_width, 16),
            "projected_grid_height": (self.projected_grid_height, 16),
            "internal_grid_width": (self.internal_grid_width, 16),
            "internal_grid_height": (self.internal_grid_height, 16),
            "internal_z_bins": (self.internal_z_bins, 1),
        }
        for name, (value, minimum) in integer_minimums.items():
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        positive = {
            "minimum_ray_weight": self.minimum_ray_weight,
            "extrusion_depth_mm": self.extrusion_depth_mm,
            "source_epsilon_mm": self.source_epsilon_mm,
            "intersection_epsilon_mm": self.intersection_epsilon_mm,
            "energy_balance_tolerance": self.energy_balance_tolerance,
        }
        if any(not isfinite(value) or value <= 0.0 for value in positive.values()):
            raise ValueError("3D transport positive settings must be finite and positive")
        for name, bounds in (("x_bounds_mm", self.x_bounds_mm), ("y_bounds_mm", self.y_bounds_mm)):
            if bounds is None:
                continue
            if len(bounds) != 2 or any(not isfinite(float(value)) for value in bounds):
                raise ValueError(f"{name} must contain two finite values")
            if not float(bounds[1]) > float(bounds[0]):
                raise ValueError(f"{name} must be strictly increasing")
