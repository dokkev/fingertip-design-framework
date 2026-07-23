"""Matplotlib visualization for the parameterized LIT Hand pad."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MatplotlibPath
from shapely.geometry import MultiPolygon, Polygon

from model.fingertip_model import BoundarySegment, FingertipModel, PolygonalGeometry

PAD_COLOR = "#9ED7E5"
PAD_EDGE = "#287D91"
RIGID_COLOR = "#747B84"
RIGID_EDGE = "#2D3339"
STEM_COLOR = "#626A74"
VOID_COLOR = "#F7B4AE"
VOID_EDGE = "#C9473D"
PAD_CONTACT_COLOR = "#D95F02"
STEM_CONTACT_COLOR = "#6A3D9A"


def plot_fingertip(
    model: FingertipModel,
    *,
    ax: Axes | None = None,
    show_void: bool = True,
    show_interface: bool = True,
    show_contact_boundaries: bool = True,
    show_symmetry_axis: bool = False,
    show_dimensions: bool = False,
    show_axes: bool = True,
    show_legend: bool = True,
    title: str | None = None,
) -> Axes:
    """Plot the compliant pad, rigid link/stem, clearance, and interface."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 5.0))

    _add_polygonal_patches(
        ax,
        model.pad_material_geometry,
        facecolor=PAD_COLOR,
        edgecolor=PAD_EDGE,
        linewidth=1.8,
        label="Silicone pad",
        zorder=1,
    )
    _add_polygonal_patches(
        ax,
        model.link_plate_geometry,
        facecolor=RIGID_COLOR,
        edgecolor=RIGID_EDGE,
        linewidth=1.5,
        label="Rigid link",
        zorder=5,
    )
    _add_polygonal_patches(
        ax,
        model.stem_geometry,
        facecolor=STEM_COLOR,
        edgecolor=RIGID_EDGE,
        linewidth=1.5,
        label="Rigid stem",
        zorder=6,
    )

    if show_void and model.void_geometry is not None:
        _add_polygonal_patches(
            ax,
            model.void_geometry,
            facecolor=VOID_COLOR,
            edgecolor=VOID_EDGE,
            linewidth=1.3,
            linestyle="--",
            hatch="///",
            label="Void",
            zorder=3,
        )

    if show_interface:
        for index, segment in enumerate(model.pad_link_interface.geoms):
            interface_x, interface_y = segment.xy
            ax.plot(
                interface_x,
                interface_y,
                color="black",
                linestyle="-",
                linewidth=4.2,
                label="Bonded interface" if index == 0 else None,
                zorder=8,
            )

    if show_contact_boundaries:
        pad_contact_boundaries = (
            model.boundaries.pad_cutout_left,
            model.boundaries.pad_cutout_right,
            model.boundaries.pad_cutout_bottom,
        )
        stem_contact_boundaries = (
            model.boundaries.stem_left,
            model.boundaries.stem_right,
            model.boundaries.stem_bottom,
        )
        _plot_boundary_segments(
            ax,
            pad_contact_boundaries,
            color=PAD_CONTACT_COLOR,
            linestyle="--",
            linewidth=3.0,
            label="Pad contact boundary",
            zorder=9,
        )
        _plot_boundary_segments(
            ax,
            stem_contact_boundaries,
            color=STEM_CONTACT_COLOR,
            linestyle=":",
            linewidth=1.8,
            label="Stem contact boundary",
            zorder=10,
        )

    if show_symmetry_axis:
        symmetry_x, symmetry_y = model.symmetry_axis.xy
        ax.plot(
            symmetry_x,
            symmetry_y,
            color="#6C757D",
            linestyle=":",
            linewidth=1.2,
            label="Symmetry axis",
            zorder=2,
        )

    if show_dimensions:
        _draw_clearance_dimensions(ax, model)

    _set_padded_limits(ax, model)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title or "Parameterized LIT Hand pad")
    if show_axes:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
    else:
        ax.axis("off")
    if show_legend:
        ax.legend(loc="best", fontsize=8)
    return ax


def save_fingertip_figure(
    model: FingertipModel,
    output_path: str | Path,
    *,
    dpi: int = 200,
    **plot_kwargs: object,
) -> Path:
    """Plot a model, save it to ``output_path``, and return the resolved path."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.0, 5.0))
    plot_fingertip(model, ax=axis, **plot_kwargs)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _draw_clearance_dimensions(ax: Axes, model: FingertipModel) -> None:
    parameters = model.parameters
    dimension_color = "#A62F28"
    text_color = "#8E2923"

    if parameters.void_width > 0.0:
        dimension_y = -0.45 * parameters.stem_height
        stem_edge = parameters.stem_width / 2.0
        cutout_edge = parameters.cutout_half_width
        ax.annotate(
            "",
            xy=(cutout_edge, dimension_y),
            xytext=(stem_edge, dimension_y),
            arrowprops={"arrowstyle": "<->", "color": dimension_color, "lw": 1.2},
            zorder=9,
        )
        ax.text(
            (stem_edge + cutout_edge) / 2.0,
            dimension_y + 0.45,
            r"$w_v$",
            color=text_color,
            ha="center",
            va="bottom",
            fontsize=10,
            zorder=9,
        )

    if parameters.void_height > 0.0:
        stem_bottom = -parameters.stem_height
        cutout_bottom = -parameters.cutout_depth
        ax.annotate(
            "",
            xy=(0.0, cutout_bottom),
            xytext=(0.0, stem_bottom),
            arrowprops={"arrowstyle": "<->", "color": dimension_color, "lw": 1.2},
            zorder=9,
        )
        ax.text(
            0.45,
            (stem_bottom + cutout_bottom) / 2.0,
            r"$h_v$",
            color=text_color,
            ha="left",
            va="center",
            fontsize=10,
            zorder=9,
        )


def _add_polygonal_patches(
    ax: Axes,
    geometry: PolygonalGeometry,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float,
    label: str,
    zorder: int,
    linestyle: str = "-",
    hatch: str | None = None,
) -> None:
    for index, polygon in enumerate(_iter_polygons(geometry)):
        ax.add_patch(
            PathPatch(
                _polygon_to_path(polygon),
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                linestyle=linestyle,
                hatch=hatch,
                label=label if index == 0 else None,
                zorder=zorder,
            )
        )


def _plot_boundary_segments(
    ax: Axes,
    segments: tuple[BoundarySegment, ...],
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    label: str,
    zorder: int,
) -> None:
    for index, segment in enumerate(segments):
        boundary_x, boundary_y = segment.geometry.xy
        ax.plot(
            boundary_x,
            boundary_y,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label if index == 0 else None,
            zorder=zorder,
        )


def _iter_polygons(geometry: PolygonalGeometry) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        return (geometry,)
    if isinstance(geometry, MultiPolygon):
        return geometry.geoms
    return ()


def _polygon_to_path(polygon: Polygon) -> MatplotlibPath:
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    for ring in (polygon.exterior, *polygon.interiors):
        ring_vertices = [(float(x), float(y)) for x, y in ring.coords]
        vertices.extend(ring_vertices)
        codes.extend(
            [MatplotlibPath.MOVETO]
            + [MatplotlibPath.LINETO] * (len(ring_vertices) - 2)
            + [MatplotlibPath.CLOSEPOLY]
        )
    return MatplotlibPath(np.asarray(vertices, dtype=float), codes)


def _set_padded_limits(ax: Axes, model: FingertipModel) -> None:
    min_x, min_y, max_x, max_y = model.raw_material_geometry.bounds
    width = max_x - min_x
    height = max_y - min_y
    padding = 0.08 * max(width, height, 1.0)
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
