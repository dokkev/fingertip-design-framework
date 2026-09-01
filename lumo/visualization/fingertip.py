"""Axes-owned schematic of the parametric LUMO fingertip cross-section."""

from __future__ import annotations

from math import pi

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from lumo.fingertip import Fingertip

from .style import DEFAULT_STYLE, PublicationStyle


_SCHEMATIC_LED_GAP_MM = 2.2


def _silicone_boundary(fingertip: Fingertip) -> np.ndarray:
    silicone = fingertip.silicone
    angles = np.linspace(0.0, pi, 257)
    ellipse = np.column_stack(
        (
            silicone.ellipse_radius_x_mm * np.cos(angles),
            silicone.ellipse_center_z_mm
            - silicone.ellipse_radius_z_mm * np.sin(angles),
        )
    )
    return np.vstack(
        (
            silicone.void_left,
            silicone.void_bottom[1:],
            silicone.void_right[1:],
            fingertip.bonding_interface.right[::-1][1:],
            silicone.outer_right[:1],
            ellipse[1:],
            silicone.outer_left[:1],
            fingertip.bonding_interface.left[::-1][1:],
        )
    )


def _label(symbol: str, value_mm: float, show_values: bool) -> str:
    if show_values:
        return rf"${symbol}={value_mm:g}\,\mathrm{{mm}}$"
    return rf"${symbol}$"


def _dimension(
    axes: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    *,
    color: str,
    label_background: bool = True,
    label_offset: tuple[float, float] = (0.0, 0.0),
    outside_arrows: bool = False,
    style: PublicationStyle,
) -> tuple[Artist, ...]:
    arrows: list[Artist] = []
    if outside_arrows:
        direction = np.asarray(end, dtype=np.float64) - np.asarray(
            start,
            dtype=np.float64,
        )
        direction /= np.linalg.norm(direction)
        outside_length = 1.4
        arrow_endpoints = (
            (np.asarray(start) - outside_length * direction, start),
            (np.asarray(end) + outside_length * direction, end),
        )
        arrow_style = "->"
    else:
        arrow_endpoints = ((start, end),)
        arrow_style = "<->"

    for arrow_start, arrow_end in arrow_endpoints:
        arrow = FancyArrowPatch(
            arrow_start,
            arrow_end,
            arrowstyle=arrow_style,
            mutation_scale=7.0,
            linewidth=style.line_width_pt,
            color=color,
            shrinkA=0.0,
            shrinkB=0.0,
            clip_on=False,
            zorder=7,
        )
        axes.add_patch(arrow)
        arrows.append(arrow)
    midpoint = (0.5 * (start[0] + end[0]), 0.5 * (start[1] + end[1]))
    text = axes.text(
        midpoint[0] + label_offset[0],
        midpoint[1] + label_offset[1],
        label,
        color=color,
        fontsize=style.axis_label_font_size_pt,
        ha="center",
        va="center",
        bbox=(
            {
                "boxstyle": "square,pad=0.05",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.95,
            }
            if label_background
            else None
        ),
        zorder=8,
    )
    return *arrows, text


def _extension(
    axes: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    style: PublicationStyle,
) -> Artist:
    (line,) = axes.plot(
        (start[0], end[0]),
        (start[1], end[1]),
        color=color,
        linewidth=style.grid_width_pt,
        linestyle=(0, (3.0, 3.0)),
        alpha=0.45,
        clip_on=False,
    )
    return line


def plot_fingertip_parameterization(
    axes: Axes,
    fingertip: Fingertip,
    *,
    show_values: bool = False,
    show_legend: bool = True,
    show_fixed_dimensions: bool = True,
    style: PublicationStyle = DEFAULT_STYLE,
) -> tuple[Artist, ...]:
    """Plot the current local X-Z fingertip morphology and its parameters.

    The cross-section passes through an LED station. The LED-to-silicone gap is
    exaggerated for visibility; the physical fingertip geometry is unchanged.
    """

    if not isinstance(fingertip, Fingertip):
        raise TypeError("fingertip must be a Fingertip")

    geometry = fingertip.parameters.geometry
    led = fingertip.parameters.led
    silicone = fingertip.silicone
    neutral = style.colors.neutral
    carrier_color = style.colors.carrier
    optical_color = style.colors.optical
    mechanical_color = style.colors.mechanical
    optimization_color = style.colors.optimization
    artists: list[Artist] = []

    silicone_patch = Polygon(
        _silicone_boundary(fingertip),
        closed=True,
        facecolor=style.colors.silicone,
        edgecolor=neutral,
        linewidth=style.spine_width_pt,
        zorder=1,
    )
    axes.add_patch(silicone_patch)
    artists.append(silicone_patch)

    carrier_section = np.asarray(fingertip.carrier.cross_section, dtype=np.float64)
    stem_bottom_z_mm = carrier_section[:, 1].min()
    carrier_section[
        np.isclose(carrier_section[:, 1], stem_bottom_z_mm),
        1,
    ] += _SCHEMATIC_LED_GAP_MM
    carrier_patch = Polygon(
        carrier_section,
        closed=True,
        facecolor=carrier_color,
        edgecolor=carrier_color,
        linewidth=style.spine_width_pt,
        zorder=2,
    )
    axes.add_patch(carrier_patch)
    artists.append(carrier_patch)

    for interface in (
        fingertip.bonding_interface.left,
        fingertip.bonding_interface.right,
    ):
        interface_points = np.asarray(interface)
        (bond_line,) = axes.plot(
            interface_points[:, 0],
            interface_points[:, 1],
            color=mechanical_color,
            linewidth=style.line_width_pt,
            linestyle="--",
            zorder=3,
        )
        artists.append(bond_line)

    source_z_mm = stem_bottom_z_mm + _SCHEMATIC_LED_GAP_MM
    led_patch = Rectangle(
        (-0.5 * led.width_mm, source_z_mm),
        led.width_mm,
        led.height_mm,
        facecolor=optical_color,
        edgecolor=optical_color,
        linewidth=style.spine_width_pt,
        zorder=4,
    )
    axes.add_patch(led_patch)
    artists.append(led_patch)
    half_width = silicone.half_width_mm
    bond_inner = silicone.bond_right_inner_x_mm
    cavity_right = silicone.cavity_right_x_mm
    stem_half_width = 0.5 * geometry.stem_width_mm
    ellipse_bottom = silicone.ellipse_center_z_mm - silicone.ellipse_radius_z_mm

    dimension_color = carrier_color
    extension_color = neutral

    if show_fixed_dimensions:
        for x_mm in (-half_width, half_width):
            artists.append(
                _extension(
                    axes,
                    (x_mm, geometry.link_thickness_mm),
                    (x_mm, 13.2),
                    color=extension_color,
                    style=style,
                )
            )
        artists.extend(
            _dimension(
                axes,
                (-half_width, 12.8),
                (half_width, 12.8),
                _label(r"w_l", geometry.flat_pad_width_mm, show_values),
                color=dimension_color,
                style=style,
            )
        )

        for x_mm in (-half_width, -bond_inner):
            artists.append(
                _extension(
                    axes,
                    (x_mm, geometry.bond_extension_height_mm),
                    (x_mm, 11.5),
                    color=extension_color,
                    style=style,
                )
            )
        artists.extend(
            _dimension(
                axes,
                (-half_width, 11.0),
                (-bond_inner, 11.0),
                _label(
                    r"w_{\mathrm{bf}}",
                    geometry.bond_extension_width_mm,
                    show_values,
                ),
                color=dimension_color,
                style=style,
            )
        )

    left_dimension_x = -18.2
    left_extension_points = (
        (-half_width, 0.0),
        (-half_width, silicone.ellipse_center_z_mm),
        (0.0, ellipse_bottom),
    )
    if show_fixed_dimensions:
        left_extension_points = (
            (-half_width, geometry.bond_extension_height_mm),
            *left_extension_points,
        )
    for start_x_mm, z_mm in left_extension_points:
        artists.append(
            _extension(
                axes,
                (start_x_mm, z_mm),
                (left_dimension_x - 0.4, z_mm),
                color=extension_color,
                style=style,
            )
        )
    left_dimensions = [
        (
            silicone.ellipse_center_z_mm,
            0.0,
            _label(r"h_{\mathrm{fp}}", geometry.flat_pad_height_mm, show_values),
            optimization_color,
        ),
        (
            ellipse_bottom,
            silicone.ellipse_center_z_mm,
            _label(
                r"h_{\mathrm{ep}}",
                geometry.semiellipse_height_mm,
                show_values,
            ),
            optimization_color,
        ),
    ]
    if show_fixed_dimensions:
        left_dimensions.insert(
            0,
            (
                0.0,
                geometry.bond_extension_height_mm,
                _label(
                    r"h_{\mathrm{bf}}",
                    geometry.bond_extension_height_mm,
                    show_values,
                ),
                dimension_color,
            ),
        )
    for start_z, end_z, label, label_color in left_dimensions:
        artists.extend(
            _dimension(
                axes,
                (left_dimension_x, start_z),
                (left_dimension_x, end_z),
                label,
                color=label_color,
                style=style,
            )
        )

    right_dimension_x = 18.2
    right_extension_heights = [0.0, source_z_mm]
    if show_fixed_dimensions:
        right_extension_heights.insert(0, geometry.link_thickness_mm)
    for z_mm in right_extension_heights:
        artists.append(
            _extension(
                axes,
                (stem_half_width, z_mm),
                (right_dimension_x + 0.4, z_mm),
                color=extension_color,
                style=style,
            )
        )
    if show_fixed_dimensions:
        artists.extend(
            _dimension(
                axes,
                (right_dimension_x, 0.0),
                (right_dimension_x, geometry.link_thickness_mm),
                _label(r"h_l", geometry.link_thickness_mm, show_values),
                color=dimension_color,
                style=style,
            )
        )
    artists.extend(
        _dimension(
            axes,
            (right_dimension_x, source_z_mm),
            (right_dimension_x, 0.0),
            _label(r"h_s", geometry.stem_height_mm, show_values),
            color=optimization_color,
            style=style,
        )
    )

    stem_dimension_z = silicone.cavity_bottom_z_mm - 1.6
    for x_mm in (-stem_half_width, stem_half_width):
        artists.append(
            _extension(
                axes,
                (x_mm, source_z_mm),
                (x_mm, stem_dimension_z - 0.3),
                color=extension_color,
                style=style,
            )
        )
    artists.extend(
        _dimension(
            axes,
            (-stem_half_width, stem_dimension_z),
            (stem_half_width, stem_dimension_z),
            _label(r"w_s", geometry.stem_width_mm, show_values),
            color=optimization_color,
            label_background=False,
            label_offset=(0.0, -0.75),
            style=style,
        )
    )

    artists.extend(
        _dimension(
            axes,
            (stem_half_width, -2.3),
            (cavity_right, -2.3),
            _label(r"w_v", geometry.void_width_mm, show_values),
            color=optimization_color,
            label_background=False,
            label_offset=(0.0, 0.75),
            outside_arrows=True,
            style=style,
        )
    )

    if show_fixed_dimensions:
        artists.extend(
            _dimension(
                axes,
                (cavity_right + 1.2, silicone.cavity_bottom_z_mm),
                (cavity_right + 1.2, source_z_mm),
                r"$h_v$",
                color=dimension_color,
                label_background=False,
                label_offset=(0.9, 0.0),
                outside_arrows=True,
                style=style,
            )
        )

        ellipse_angles = np.linspace(0.0, pi, 4097)
        ellipse = np.column_stack(
            (
                silicone.ellipse_radius_x_mm * np.cos(ellipse_angles),
                silicone.ellipse_center_z_mm
                - silicone.ellipse_radius_z_mm * np.sin(ellipse_angles),
            )
        )
        corner = np.asarray(silicone.void_bottom[0])
        nearest = ellipse[np.argmin(np.linalg.norm(ellipse - corner, axis=1))]
        artists.extend(
            _dimension(
                axes,
                tuple(corner),
                tuple(nearest),
                _label(
                    r"t_{\min}",
                    silicone.minimum_silicone_thickness_mm,
                    show_values,
                ),
                color=mechanical_color,
                label_background=False,
                label_offset=(-0.5, 1.1),
                style=style,
            )
        )

    x_limits = (-22.5, 22.5)
    y_limits = (ellipse_bottom - 1.2, 14.5)
    axes.set_xlim(*x_limits)
    axes.set_ylim(*y_limits)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel(r"Lateral coordinate, $X$ [mm]")
    axes.set_ylabel(r"Vertical coordinate, $Z$ [mm]")
    axes.xaxis.set_major_locator(MultipleLocator(5.0))
    axes.yaxis.set_major_locator(MultipleLocator(5.0))
    axes.xaxis.set_minor_locator(MultipleLocator(1.0))
    axes.yaxis.set_minor_locator(MultipleLocator(1.0))
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.tick_params(
        which="major",
        width=1.2 * style.tick_width_pt,
        length=1.6 * style.tick_length_pt,
        labelsize=style.tick_font_size_pt,
        direction="out",
    )
    axes.tick_params(
        which="minor",
        width=0.5 * style.tick_width_pt,
        length=0.5 * style.tick_length_pt,
        direction="out",
    )
    if show_legend:
        legend = axes.legend(
            handles=(
                Patch(
                    facecolor=carrier_color,
                    edgecolor=carrier_color,
                    label="Carrier",
                ),
                Patch(
                    facecolor=style.colors.silicone,
                    edgecolor=neutral,
                    label="Pad",
                ),
                Patch(
                    facecolor=style.colors.silicone,
                    edgecolor=optical_color,
                    label="LED",
                ),
                Line2D(
                    (),
                    (),
                    color=mechanical_color,
                    linestyle="--",
                    linewidth=style.line_width_pt,
                    label="Bonding surface",
                ),
            ),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.86),
            ncol=2,
            columnspacing=1.2,
            handlelength=1.8,
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor=style.colors.grid,
            fontsize=style.legend_font_size_pt,
        )
        artists.append(legend)
    return tuple(artists)


__all__ = ["plot_fingertip_parameterization"]
