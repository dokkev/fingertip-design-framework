"""Public mesh plot wrapper over the shared mechanics renderer."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from mesh import FingertipMesh, PadMesh
from visualization._axes import apply_physical_axes, bounds_from_points
from visualization.mechanics import _mesh_arrays, draw_mesh


def plot_mesh(
    mesh: PadMesh | FingertipMesh,
    *,
    ax: Axes | None = None,
    show_nodes: bool = False,
    title: str | None = None,
) -> Axes:
    """Plot T3 connectivity and semantic boundary edges for one pad mesh."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 6.0))
    draw_mesh(ax, mesh, show_nodes=show_nodes)
    coordinates, _ = _mesh_arrays(mesh)
    apply_physical_axes(ax, bounds_from_points(coordinates))
    ax.set_title(title or "Pad T3 mesh")
    return ax


__all__ = ["plot_mesh"]
