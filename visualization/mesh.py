"""Small Matplotlib helpers for neutral two-dimensional pad meshes."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.tri import Triangulation

from mesh import FingertipMesh, PadMesh


def _pad_view(mesh: Any) -> Any:
    """Return the neutral pad view without generating or inspecting a mesh."""
    if isinstance(mesh, FingertipMesh):
        return mesh.pad
    if isinstance(mesh, PadMesh):
        return mesh
    if not all(hasattr(mesh, name) for name in ("coordinates", "triangles")):
        raise TypeError("mesh must be a PadMesh or FingertipMesh")
    return mesh


def _mesh_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    pad = _pad_view(mesh)
    coordinates = np.asarray(pad.coordinates, dtype=float)
    triangles = np.asarray(pad.triangles, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("mesh.coordinates must have shape (N, 2)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("mesh.triangles must have shape (M, 3)")
    if np.any(triangles < 0) or np.any(triangles >= len(coordinates)):
        raise ValueError("mesh.triangles contains an invalid node index")
    return coordinates, triangles


def _set_limits(ax: Axes, coordinates: np.ndarray) -> None:
    minimum = np.min(coordinates, axis=0)
    maximum = np.max(coordinates, axis=0)
    padding = 0.04 * max(float(np.max(maximum - minimum)), 1.0)
    ax.set_xlim(minimum[0] - padding, maximum[0] + padding)
    ax.set_ylim(minimum[1] - padding, maximum[1] + padding)


def plot_mesh(
    mesh: PadMesh | FingertipMesh,
    *,
    ax: Axes | None = None,
    show_nodes: bool = False,
    title: str | None = None,
) -> Axes:
    """Plot T3 connectivity and semantic boundary edges for one pad mesh."""
    coordinates, triangles = _mesh_arrays(mesh)
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.triplot(
        coordinates[:, 0],
        coordinates[:, 1],
        triangles,
        color="#56616A",
        linewidth=0.45,
        label="T3 mesh",
    )
    if show_nodes:
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=10.0,
            color="#263238",
            label="Nodes",
            zorder=3,
        )
    pad = _pad_view(mesh)
    for tag in getattr(pad, "semantic_boundary_tags", ()):
        edges = np.asarray(pad.boundary_edges_for(tag), dtype=np.int64)
        if len(edges):
            for index, edge in enumerate(edges):
                points = coordinates[edge]
                ax.plot(
                    points[:, 0],
                    points[:, 1],
                    color="#D95F02",
                    linewidth=1.2,
                    label=(
                        "Semantic boundary"
                        if index == 0 and tag == pad.semantic_boundary_tags[0]
                        else "_nolegend_"
                    ),
                    zorder=2,
                )
    _set_limits(ax, coordinates)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title or "Pad T3 mesh")
    return ax


def _validate_displacement(
    coordinates: np.ndarray,
    displacement: np.ndarray,
) -> np.ndarray:
    values = np.asarray(displacement, dtype=float)
    if values.shape != coordinates.shape:
        raise ValueError(
            "displacement must have the same shape as mesh.coordinates (N, 2)"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("displacement must contain finite values")
    return values


def _validate_point(value: Any, *, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite 2-vector")
    return point


def _arrow_indices(
    coordinates: np.ndarray,
    magnitudes: np.ndarray,
    *,
    minimum: float,
    maximum: int,
) -> np.ndarray:
    candidates = np.flatnonzero(magnitudes >= minimum)
    if len(candidates) <= maximum:
        return candidates
    order = np.lexsort((candidates, coordinates[candidates, 1], coordinates[candidates, 0]))
    ordered = candidates[order]
    selected = np.linspace(0, len(ordered) - 1, maximum, dtype=int)
    return np.sort(ordered[selected])


def plot_displacement(
    mesh: PadMesh | FingertipMesh,
    displacement: np.ndarray,
    *,
    ax: Axes | None = None,
    show_magnitude: bool = True,
    show_vectors: bool = True,
    deformation_scale: float = 1.0,
    arrow_scale: float = 1.0,
    arrow_minimum_mm: float = 0.0,
    maximum_arrows: int = 80,
    normalization_max: float | None = None,
    contact_point: Any | None = None,
    indentation_direction: Any | None = None,
    title: str | None = None,
) -> Axes:
    """Plot nodal displacement magnitude and physical displacement vectors."""
    coordinates, triangles = _mesh_arrays(mesh)
    values = _validate_displacement(coordinates, displacement)
    for name, value in (
        ("deformation_scale", deformation_scale),
        ("arrow_scale", arrow_scale),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(arrow_minimum_mm) or arrow_minimum_mm < 0.0:
        raise ValueError("arrow_minimum_mm must be finite and nonnegative")
    if arrow_scale <= 0.0:
        raise ValueError("arrow_scale must be positive")
    if not isinstance(maximum_arrows, (int, np.integer)) or maximum_arrows < 1:
        raise ValueError("maximum_arrows must be a positive integer")
    if normalization_max is not None and (
        not np.isfinite(normalization_max) or normalization_max <= 0.0
    ):
        raise ValueError("normalization_max must be finite and positive")
    if contact_point is not None:
        contact = _validate_point(contact_point, name="contact_point")
    else:
        contact = None
    if indentation_direction is not None:
        direction = _validate_point(
            indentation_direction, name="indentation_direction"
        )
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            raise ValueError("indentation_direction must be nonzero")
        direction = direction / norm
    else:
        direction = None

    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 6.0))
    deformed = coordinates + deformation_scale * values
    magnitude = np.linalg.norm(values, axis=1)
    triangulation = Triangulation(
        deformed[:, 0], deformed[:, 1], triangles
    )
    if show_magnitude:
        scale = (
            float(np.max(magnitude))
            if normalization_max is None
            else float(normalization_max)
        )
        scale = max(scale, 1.0e-15)
        image = ax.tripcolor(
            triangulation,
            magnitude,
            shading="gouraud",
            cmap="viridis",
            vmin=0.0,
            vmax=scale,
        )
        colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Displacement magnitude |u| [mm]")
    ax.triplot(
        triangulation,
        color="#56616A",
        linewidth=0.4,
        alpha=0.45,
        label="Deformed T3 mesh",
    )
    if show_vectors:
        selected = _arrow_indices(
            deformed,
            magnitude,
            minimum=arrow_minimum_mm,
            maximum=int(maximum_arrows),
        )
        if len(selected):
            ax.quiver(
                deformed[selected, 0],
                deformed[selected, 1],
                arrow_scale * values[selected, 0],
                arrow_scale * values[selected, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="#111111",
                width=0.0025,
                label="Displacement u",
                zorder=5,
            )
    if contact is not None:
        ax.scatter(
            [contact[0]],
            [contact[1]],
            marker="x",
            color="#C9473D",
            s=48.0,
            label="Contact point",
            zorder=7,
        )
        if direction is not None:
            length = max(float(np.ptp(deformed, axis=0).max()), 1.0) * 0.12
            ax.quiver(
                [contact[0]],
                [contact[1]],
                [length * direction[0]],
                [length * direction[1]],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="#C9473D",
                width=0.003,
                label="Indentation direction",
                zorder=7,
            )
    _set_limits(ax, deformed)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title or "Nodal displacement")
    return ax


__all__ = ["plot_displacement", "plot_mesh"]
