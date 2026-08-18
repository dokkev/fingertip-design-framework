"""Shared physical-axis policy for visualization draw and plot layers."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes


def bounds_from_points(
    *point_sets: np.ndarray,
    padding: float = 0.05,
    minimum_span: float = 2.0,
) -> tuple[float, float, float, float]:
    """Return one padded x/y bound for already-computed display geometry."""
    if not 0.0 <= padding:
        raise ValueError("padding must be nonnegative")
    arrays = [np.asarray(points, dtype=float) for points in point_sets if len(points)]
    if not arrays:
        raise ValueError("at least one point set is required")
    if any(array.ndim != 2 or array.shape[1] != 2 for array in arrays):
        raise ValueError("point sets must have shape (N, 2)")
    points = np.vstack(arrays)
    if not np.all(np.isfinite(points)):
        raise ValueError("display points must be finite")
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    span = max(float(np.max(maximum - minimum)), float(minimum_span))
    margin = padding * span
    return (
        float(minimum[0] - margin),
        float(maximum[0] + margin),
        float(minimum[1] - margin),
        float(maximum[1] + margin),
    )


def bounds_from_geometries(
    *geometries: Any,
    padding: float = 0.05,
    minimum_span: float = 2.0,
) -> tuple[float, float, float, float]:
    """Return shared bounds from Shapely-like objects exposing ``.bounds``."""
    points = []
    for geometry in geometries:
        if geometry is None or geometry.is_empty:
            continue
        min_x, min_y, max_x, max_y = geometry.bounds
        points.append(
            np.asarray(
                [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]],
                dtype=float,
            )
        )
    return bounds_from_points(*points, padding=padding, minimum_span=minimum_span)


def apply_physical_axes(
    ax: Axes,
    bounds: tuple[float, float, float, float],
    *,
    show_axes: bool = True,
    xlabel: str = "x [mm]",
    ylabel: str = "y [mm]",
) -> None:
    """Apply shared limits/aspect/labels after all draw layers are complete."""
    x_min, x_max, y_min, y_max = bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    if show_axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.axis("off")


__all__ = ["apply_physical_axes", "bounds_from_geometries", "bounds_from_points"]
