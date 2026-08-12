"""Matplotlib visualization for the public fingertip geometry facade."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from model import Fingertip
from model.fingertip_model import BoundarySegment
from visualization._plotting import (
    ALUMINUM_COLOR,
    ALUMINUM_EDGE,
    BONDED_INTERFACE_COLOR,
    BONDED_INTERFACE_EDGE,
    LED_COLOR,
    LED_EDGE,
    LIGHT_SOURCE_COLOR,
    LIGHT_SOURCE_EDGE,
    PAD_COLOR,
    PAD_CONTACT_COLOR,
    PAD_EDGE,
    VOID_COLOR,
    VOID_EDGE,
    add_polygonal_patches,
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
    """Plot one :class:`model.Fingertip` and its display-only overlays."""
    if not isinstance(tip, Fingertip):
        raise TypeError("tip must be a Fingertip")
    model = tip.geometry
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 5.0))

    add_polygonal_patches(
        ax,
        model.pad_material_geometry,
        facecolor=PAD_COLOR,
        edgecolor=PAD_EDGE,
        linewidth=1.8,
        label="Silicone pad",
        zorder=1,
    )
    add_polygonal_patches(
        ax,
        model.link_plate_geometry,
        facecolor=ALUMINUM_COLOR,
        edgecolor=ALUMINUM_EDGE,
        linewidth=1.5,
        label="Rigid link / stem",
        zorder=5,
    )
    add_polygonal_patches(
        ax,
        model.stem_geometry,
        facecolor=ALUMINUM_COLOR,
        edgecolor=ALUMINUM_EDGE,
        linewidth=1.5,
        label="_nolegend_",
        zorder=6,
    )

    if show_led:
        _add_led_overlay(ax, tip)
    if show_light_source:
        _add_light_source_overlay(ax, tip)
    if show_void and model.void_geometry is not None:
        add_polygonal_patches(
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
        _plot_bonded_interface(ax, model)
    if show_contact_boundaries:
        _plot_boundary_segments(
            ax,
            (
                model.boundaries.pad_cutout_left,
                model.boundaries.pad_cutout_right,
                model.boundaries.pad_cutout_bottom,
            ),
            color=PAD_CONTACT_COLOR,
            linestyle="--",
            linewidth=3.0,
            label="Pad contact boundary",
            zorder=9,
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
        )
    return ax


def _add_led_overlay(ax: Axes, tip: Fingertip) -> Rectangle:
    x_min, y_min, x_max, y_max = tip.led_package_geometry.bounds
    led = Rectangle(
        (x_min, y_min),
        tip.led.width_mm,
        tip.led.height_mm,
        facecolor=LED_COLOR,
        edgecolor=LED_EDGE,
        linewidth=1.2,
        label="LED",
        zorder=7,
    )
    ax.add_patch(led)
    return led


def _add_light_source_overlay(ax: Axes, tip: Fingertip) -> None:
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


def _plot_bonded_interface(ax: Axes, model: Any) -> None:
    """Highlight the two L-shaped bonded surfaces as translucent overlays.

    The analytic interface is a line in this 2D model. A small display-only
    buffer makes that surface visible without changing the physical geometry.
    """
    display_half_width_mm = 0.22
    for index, segment in enumerate(model.pad_link_interface.geoms):
        overlay = segment.buffer(
            display_half_width_mm,
            cap_style=2,
            join_style=2,
        )
        add_polygonal_patches(
            ax,
            overlay,
            facecolor=BONDED_INTERFACE_COLOR,
            edgecolor=BONDED_INTERFACE_EDGE,
            linewidth=0.8,
            label="Bonded interface" if index == 0 else "_nolegend_",
            zorder=8,
            alpha=0.62,
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


def _set_padded_limits(ax: Axes, model: Any) -> None:
    min_x, min_y, max_x, max_y = model.raw_material_geometry.bounds
    width = max_x - min_x
    height = max_y - min_y
    base_span = max(width, height, 2.0)
    padding = 0.08 * base_span
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)


__all__ = ["plot_fingertip"]
