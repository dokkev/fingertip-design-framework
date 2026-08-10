"""Neutral results produced by the cross-sectional optical tracer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

OpticalMedium = Literal["air", "silicone"]


@dataclass(frozen=True)
class RaySegment2D:
    """One retained straight segment of a deterministic ray branch."""

    start_mm: tuple[float, float]
    end_mm: tuple[float, float]
    medium: OpticalMedium
    start_weight: float
    end_weight: float
    primary_ray_index: int
    interaction_index: int


@dataclass(frozen=True)
class CrossSectionTransportResult:
    """Raw weighted paths and a regular-grid path-length accumulation."""

    source_position_mm: tuple[float, float]
    x_edges_mm: np.ndarray
    y_edges_mm: np.ndarray
    weighted_path_density: np.ndarray
    optical_mask: np.ndarray
    segments: tuple[RaySegment2D, ...]
    launched_ray_count: int
    launched_weight: float
    escaped_weight: float
    absorbed_weight: float
    terminated_weight: float

    def __post_init__(self) -> None:
        """Own immutable copies of the public result arrays."""
        x_edges = np.array(self.x_edges_mm, dtype=float, copy=True)
        y_edges = np.array(self.y_edges_mm, dtype=float, copy=True)
        density = np.array(self.weighted_path_density, dtype=float, copy=True)
        mask = np.array(self.optical_mask, dtype=bool, copy=True)
        if density.shape != mask.shape:
            raise ValueError("weighted_path_density and optical_mask shapes differ")
        if len(x_edges) != density.shape[1] + 1:
            raise ValueError("x_edges_mm does not match the density width")
        if len(y_edges) != density.shape[0] + 1:
            raise ValueError("y_edges_mm does not match the density height")
        weights = np.asarray(
            [
                self.launched_weight,
                self.escaped_weight,
                self.absorbed_weight,
                self.terminated_weight,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("transport weights must be finite and nonnegative")
        for array in (x_edges, y_edges, density, mask):
            array.setflags(write=False)
        object.__setattr__(self, "x_edges_mm", x_edges)
        object.__setattr__(self, "y_edges_mm", y_edges)
        object.__setattr__(self, "weighted_path_density", density)
        object.__setattr__(self, "optical_mask", mask)

    @property
    def normalized_path_density(self) -> np.ndarray:
        """Return a display normalization without changing the raw field."""
        maximum = float(np.max(self.weighted_path_density))
        if maximum <= 0.0:
            return np.zeros_like(self.weighted_path_density)
        return self.weighted_path_density / maximum
