"""Matplotlib visualization for the parameterized LIT Hand pad."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MatplotlibPath
from shapely.geometry import MultiPolygon, Polygon

from model import Fingertip
from model.fingertip_model import BoundarySegment, FingertipModel, PolygonalGeometry

PAD_COLOR = "#C7E8D2"
PAD_EDGE = "#4E9270"
ALUMINUM_COLOR = "#D9DCDF"
ALUMINUM_EDGE = "#7B8288"
VOID_COLOR = "#F7B4AE"
VOID_EDGE = "#C9473D"
PAD_CONTACT_COLOR = "#D95F02"
STEM_CONTACT_COLOR = "#6A3D9A"
LED_COLOR = "#F6C453"
LED_EDGE = "#9A6700"
LIGHT_SOURCE_COLOR = "#E63946"
LIGHT_SOURCE_EDGE = "#4A1018"


def plot_fingertip(
    tip: Fingertip | FingertipModel,
    *,
    ax: Axes | None = None,
    show_void: bool = True,
    show_led: bool = True,
    show_light_source: bool = True,
    show_interface: bool = True,
    show_contact_boundaries: bool = True,
    show_symmetry_axis: bool = False,
    show_axes: bool = True,
    show_legend: bool = True,
    title: str | None = None,
) -> Axes:
    """Plot the pad, rigid link/stem, LED overlay, clearance, and interface."""
    physical = tip if isinstance(tip, Fingertip) else Fingertip(tip.parameters)
    model = physical.geometry
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
        facecolor=ALUMINUM_COLOR,
        edgecolor=ALUMINUM_EDGE,
        linewidth=1.5,
        label="Rigid link / stem",
        zorder=5,
    )
    _add_polygonal_patches(
        ax,
        model.stem_geometry,
        facecolor=ALUMINUM_COLOR,
        edgecolor=ALUMINUM_EDGE,
        linewidth=1.5,
        label="_nolegend_",
        zorder=6,
    )

    if show_led:
        _add_led_overlay(ax, physical)

    if show_light_source:
        _add_light_source_overlay(ax, physical)

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
        _plot_bonded_interface_arrows(ax, model)

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

    _set_padded_limits(ax, model)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title or "Parameterized LIT Hand pad")
    if show_axes:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
    else:
        ax.axis("off")
    if show_legend:
        ax.legend(
            loc="upper center",
            fontsize=8,
            ncol=2,
            handler_map={
                FancyArrowPatch: HandlerPatch(
                    patch_func=_legend_bidirectional_arrow
                )
            },
        )
    return ax


def _add_led_overlay(
    ax: Axes,
    tip: Fingertip,
) -> Rectangle:
    """Draw the fingertip-owned LED package with visualization-only styling."""
    x_min, y_min, x_max, y_max = tip.led_package_geometry.bounds
    led = Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        facecolor=LED_COLOR,
        edgecolor=LED_EDGE,
        linewidth=1.2,
        label="LED",
        zorder=7,
    )
    ax.add_patch(led)
    return led


def _add_light_source_overlay(
    ax: Axes,
    tip: Fingertip,
) -> None:
    """Mark the ideal optical source at the LED's lower emitting edge."""
    x, y = tip.led_source
    ax.scatter(
        [x],
        [y],
        s=42.0,
        color=LIGHT_SOURCE_COLOR,
        edgecolors=LIGHT_SOURCE_EDGE,
        linewidths=0.8,
        label="Light source",
        zorder=8,
    )


def save_fingertip_figure(
    tip: Fingertip | FingertipModel,
    output_path: str | Path,
    *,
    dpi: int = 200,
    **plot_kwargs: object,
) -> Path:
    """Plot a model, save it to ``output_path``, and return the resolved path."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.0, 5.0))
    plot_fingertip(tip, ax=axis, **plot_kwargs)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _plot_bonded_interface_arrows(ax: Axes, model: FingertipModel) -> None:
    """Mark bonded spans with closely spaced red bidirectional arrows."""
    is_first_arrow = True
    for segment in model.pad_link_interface.geoms:
        arrow_count = max(1, int(round(segment.length / 1.8)))
        arrow_distances = np.linspace(
            segment.length / (arrow_count + 1),
            segment.length * arrow_count / (arrow_count + 1),
            arrow_count,
        )
        for distance in arrow_distances:
            center = segment.interpolate(float(distance))
            arrow = FancyArrowPatch(
                (center.x, center.y + 1.0),
                (center.x, center.y - 1.0),
                arrowstyle="<->",
                mutation_scale=12.0,
                color="#C9473D",
                linewidth=1.5,
                label="Bonded interface" if is_first_arrow else "_nolegend_",
                zorder=8,
            )
            ax.add_patch(arrow)
            is_first_arrow = False


def _legend_bidirectional_arrow(
    legend: object,
    orig_handle: FancyArrowPatch,
    xdescent: float,
    ydescent: float,
    width: float,
    height: float,
    fontsize: float,
) -> FancyArrowPatch:
    """Draw a bidirectional arrow instead of the default legend rectangle."""
    del legend
    center_y = ydescent + height / 2.0
    return FancyArrowPatch(
        (xdescent, center_y),
        (xdescent + width, center_y),
        arrowstyle="<->",
        mutation_scale=fontsize,
        color=orig_handle.get_edgecolor(),
        linewidth=orig_handle.get_linewidth(),
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
    base_span = max(width, height, 2.0)
    padding = 0.08 * base_span
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
