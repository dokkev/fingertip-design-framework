"""Shared display layers for the public fingertip geometry facade."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from model import Fingertip
from visualization._axes import apply_physical_axes, bounds_from_geometries
from visualization._plotting import add_polygonal_patches
from visualization._style import STYLE


def draw_outline(
    ax: Axes,
    geometry: Any,
    *,
    label: str,
    color: str = STYLE.silicone_edge,
    linewidth: float = 1.0,
    zorder: int = 6,
) -> None:
    """Draw a clean outline for a Shapely-like polygonal geometry."""
    polygons = geometry.geoms if hasattr(geometry, "geoms") else (geometry,)
    for index, polygon in enumerate(polygons):
        x, y = polygon.exterior.xy
        ax.plot(
            x,
            y,
            color=color,
            linewidth=linewidth,
            label=label if index == 0 else "_nolegend_",
            zorder=zorder,
        )


def draw_pad(ax: Axes, tip: Fingertip, *, label: str = "Silicone pad") -> None:
    """Draw the silicone material geometry without changing axis state."""
    add_polygonal_patches(
        ax,
        tip.geometry.pad_material_geometry,
        facecolor=STYLE.silicone_face,
        edgecolor=STYLE.silicone_edge,
        linewidth=1.8,
        label=label,
        zorder=1,
    )


def draw_rigid_structure(
    ax: Axes,
    tip: Fingertip,
    *,
    label: str = "Rigid link / stem",
    zorder: int = 5,
) -> None:
    """Draw the rigid link plate and stem."""
    model = tip.geometry
    add_polygonal_patches(
        ax,
        model.link_plate_geometry,
        facecolor=STYLE.rigid_face,
        edgecolor=STYLE.rigid_edge,
        linewidth=1.5,
        label=label,
        zorder=zorder,
    )
    add_polygonal_patches(
        ax,
        model.stem_geometry,
        facecolor=STYLE.rigid_face,
        edgecolor=STYLE.rigid_edge,
        linewidth=1.5,
        label="_nolegend_",
        zorder=zorder + 1,
    )


def draw_void(ax: Axes, tip: Fingertip) -> None:
    """Draw the optional internal void."""
    geometry = tip.geometry.void_geometry
    if geometry is None:
        return
    add_polygonal_patches(
        ax,
        geometry,
        facecolor=STYLE.void_face,
        edgecolor=STYLE.void_edge,
        linewidth=1.3,
        linestyle="--",
        hatch="///",
        label="Void",
        zorder=3,
    )


def draw_led(ax: Axes, tip: Fingertip) -> Rectangle:
    """Draw the LED package and return its patch."""
    x_min, y_min, _, _ = tip.led_package_geometry.bounds
    led = Rectangle(
        (x_min, y_min),
        tip.led.width_mm,
        tip.led.height_mm,
        facecolor=STYLE.led_face,
        edgecolor=STYLE.led_edge,
        linewidth=1.2,
        label="LED",
        zorder=7,
    )
    ax.add_patch(led)
    return led


def draw_light_source(ax: Axes, tip: Fingertip) -> None:
    """Draw the LED emission point."""
    x, y = tip.led_source
    ax.scatter(
        [x],
        [y],
        s=42.0,
        color=STYLE.source_face,
        edgecolors=STYLE.source_edge,
        linewidths=0.8,
        label="Light source",
        zorder=8,
    )


def draw_bonded_interfaces(ax: Axes, tip: Fingertip) -> None:
    """Highlight bonded pad/link interfaces as display-only overlays."""
    segments = list(tip.geometry.pad_link_interface.geoms)
    if tip.parameters.void_height == 0.0:
        segments.append(tip.geometry.boundaries.stem_bottom.geometry)
    for index, segment in enumerate(segments):
        overlay = segment.buffer(0.22, cap_style=2, join_style=2)
        add_polygonal_patches(
            ax,
            overlay,
            facecolor=STYLE.bonded_interface_face,
            edgecolor=STYLE.bonded_interface_edge,
            linewidth=0.8,
            label="Bonded interface" if index == 0 else "_nolegend_",
            zorder=10,
            alpha=0.62,
        )


def draw_contact_boundaries(ax: Axes, tip: Fingertip) -> None:
    """Draw pad cutout boundaries used by the mechanical contact geometry."""
    segments = (
        tip.geometry.boundaries.pad_cutout_left,
        tip.geometry.boundaries.pad_cutout_right,
        tip.geometry.boundaries.pad_cutout_bottom,
    )
    for index, segment in enumerate(segments):
        boundary_x, boundary_y = segment.geometry.xy
        ax.plot(
            boundary_x,
            boundary_y,
            color=STYLE.contact_edge,
            linestyle="--",
            linewidth=3.0,
            label="Pad contact boundary" if index == 0 else "_nolegend_",
            zorder=9,
        )


def draw_fingertip(
    ax: Axes,
    tip: Fingertip,
    *,
    show_void: bool = True,
    show_led: bool = True,
    show_light_source: bool = True,
    show_interface: bool = True,
    show_contact_boundaries: bool = True,
    show_symmetry_axis: bool = False,
) -> None:
    """Compose fingertip layers without selecting limits or labels."""
    if not isinstance(tip, Fingertip):
        raise TypeError("tip must be a Fingertip")
    draw_pad(ax, tip)
    draw_rigid_structure(ax, tip)
    if show_led:
        draw_led(ax, tip)
    if show_light_source:
        draw_light_source(ax, tip)
    if show_void:
        draw_void(ax, tip)
    if show_interface:
        draw_bonded_interfaces(ax, tip)
    if show_contact_boundaries:
        draw_contact_boundaries(ax, tip)
    if show_symmetry_axis:
        symmetry_x, symmetry_y = tip.geometry.symmetry_axis.xy
        ax.plot(
            symmetry_x,
            symmetry_y,
            color=STYLE.mesh_edge,
            linestyle=":",
            linewidth=1.2,
            label="Symmetry axis",
            zorder=2,
        )


def plot_fingertip(
    tip: Fingertip,
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
    """Plot one :class:`model.Fingertip` using the shared geometry layers."""
    if not isinstance(tip, Fingertip):
        raise TypeError("tip must be a Fingertip")
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 5.0))
    draw_fingertip(
        ax,
        tip,
        show_void=show_void,
        show_led=show_led,
        show_light_source=show_light_source,
        show_interface=show_interface,
        show_contact_boundaries=show_contact_boundaries,
        show_symmetry_axis=show_symmetry_axis,
    )
    bounds = bounds_from_geometries(tip.geometry.raw_material_geometry)
    apply_physical_axes(ax, bounds, show_axes=show_axes)
    ax.set_title(title or "Parameterized LIT Hand pad")
    if show_legend:
        ax.legend(loc="upper center", fontsize=8, ncol=2)
    return ax


__all__ = ["plot_fingertip"]
