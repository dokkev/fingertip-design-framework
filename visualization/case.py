"""Matplotlib view for one complete explicit-contact fingertip case."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from shapely.geometry import LineString, MultiLineString

from visualization.geometry import plot_fingertip
from visualization.mesh import plot_displacement


def _line_parts(geometry: Any) -> tuple[Any, ...]:
    if geometry is None:
        return ()
    if isinstance(geometry, LineString):
        return (geometry,)
    if isinstance(geometry, MultiLineString):
        return tuple(geometry.geoms)
    raise TypeError("contact_patch must be a LineString or MultiLineString")


def _plot_pose(ax: Any, pose: Any) -> None:
    carrier_x, carrier_y = pose.carrier_geometry.exterior.xy
    ax.fill(
        carrier_x,
        carrier_y,
        facecolor="#B8C2CC",
        edgecolor="#495057",
        alpha=0.35,
        label="Exact indenter pose",
        zorder=6,
    )
    for index, line in enumerate(_line_parts(pose.contact_patch)):
        patch_x, patch_y = line.xy
        ax.plot(
            patch_x,
            patch_y,
            color="#D62728",
            linewidth=3.0,
            label="Mechanical contact patch" if index == 0 else "_nolegend_",
            zorder=8,
        )


def _set_case_limits(ax: Any, case: Any) -> None:
    if case.fea.result is None:
        raise RuntimeError("case FEA result is unavailable")
    deformed = np.asarray(case.fea.result.deformed_mesh.coordinates, dtype=float)
    carrier = np.asarray(
        case.fea.result.indenter_pose.carrier_geometry.exterior.coords,
        dtype=float,
    )
    coordinates = np.vstack((deformed, carrier))
    minimum = np.min(coordinates, axis=0)
    maximum = np.max(coordinates, axis=0)
    padding = 0.08 * max(float(np.max(maximum - minimum)), 1.0)
    ax.set_xlim(minimum[0] - padding, maximum[0] + padding)
    ax.set_ylim(minimum[1] - padding, maximum[1] + padding)


def _plot_optical_field(ax: Any, case: Any) -> None:
    if case.raytracing.summary is None or case.raytracing.raw is None:
        raise RuntimeError("case optical result is unavailable")
    field = np.asarray(case.raytracing.summary.field, dtype=float)
    x_edges, y_edges = case.raytracing.summary.field_axes
    scale = max(float(np.max(field)), 1.0e-15)
    image = ax.pcolormesh(
        x_edges,
        y_edges,
        field.T,
        shading="flat",
        cmap="magma",
        vmin=0.0,
        vmax=scale,
    )
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label(
        "P2 weighted path density"
    )

    positions = np.asarray(case.raytracing.raw.escape_positions_mm, dtype=float)
    directions = np.asarray(case.raytracing.raw.escape_directions, dtype=float)
    if len(positions):
        selected = np.linspace(
            0,
            len(positions) - 1,
            min(48, len(positions)),
            dtype=int,
        )
        ax.scatter(
            positions[selected, 0],
            positions[selected, 1],
            s=12.0,
            color="#F8F9FA",
            edgecolors="#212529",
            linewidths=0.3,
            label="OptiX exits",
            zorder=5,
        )
        ax.quiver(
            positions[selected, 0],
            positions[selected, 1],
            directions[selected, 0],
            directions[selected, 1],
            angles="xy",
            scale_units="xy",
            scale=3.0,
            color="#F8F9FA",
            width=0.0018,
            zorder=6,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("PLANAR_2D OptiX")
    ax.legend(loc="upper center", fontsize=8)


def plot_case(case: Any) -> Figure:
    """Plot mechanics, exact indenter pose, contact patch, and native P2 data.

    The helper intentionally consumes the existing ``FingertipCase`` shape
    without importing the case package, preserving visualization's neutral
    dependency boundary.
    """
    required = ("fingertip", "fea", "raytracing")
    if any(not hasattr(case, name) for name in required):
        raise TypeError("case must expose the FingertipCase visualization contract")

    if case.fea.result is None or case.fea.result.indenter_pose is None:
        raise RuntimeError("case FEA result is unavailable")
    tip = case.fingertip
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), constrained_layout=True)
    plot_fingertip(
        tip,
        ax=axes[0],
        show_led=False,
        show_light_source=False,
        show_legend=False,
        title="Reference geometry + deformed FEA mesh",
    )
    plot_displacement(
        case.fea.result.mesh,
        case.fea.result.displacement,
        ax=axes[0],
        show_magnitude=False,
        show_vectors=False,
        show_rigid_structure=False,
        title="Reference geometry + deformed FEA mesh",
    )
    _plot_pose(axes[0], case.fea.result.indenter_pose)
    _set_case_limits(axes[0], case)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(loc="upper center", fontsize=8)

    _plot_optical_field(axes[1], case)
    figure.suptitle("Nominal fingertip: explicit contact FEA → PLANAR_2D OptiX")
    return figure


__all__ = ["plot_case"]
