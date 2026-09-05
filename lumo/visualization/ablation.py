"""Reusable panels for the carrier and lateral-void ablation figure."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PathCollection
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Affine2D

from .style import DEFAULT_STYLE, PublicationStyle


def _outer_pad() -> np.ndarray:
    angles = np.linspace(np.pi, 2.0 * np.pi, 80)
    lower = np.column_stack(
        (4.8 * np.cos(angles), -0.4 + 3.4 * np.sin(angles))
    )
    return np.vstack(
        (
            (-4.8, 3.4),
            (-4.8, -0.4),
            lower[1:-1],
            (4.8, -0.4),
            (4.8, 3.4),
        )
    )


def _draw_structure(
    axes: plt.Axes,
    center_x: float,
    center_y: float,
    *,
    carrier: bool,
    stem_half_width: float,
    void_half_width: float,
    void_height: float,
    scale: float,
    style: PublicationStyle,
) -> None:
    transform = Affine2D().scale(scale).translate(center_x, center_y) + axes.transData
    axes.add_patch(
        Polygon(
            _outer_pad(),
            closed=True,
            facecolor=style.colors.silicone,
            edgecolor="#858585",
            linewidth=0.6,
            zorder=1,
            transform=transform,
        )
    )
    if not carrier:
        return

    if void_half_width > 0.0:
        axes.add_patch(
            Rectangle(
                (-void_half_width, -1.25 - void_height),
                2.0 * void_half_width,
                2.3 + void_height,
                facecolor="white",
                edgecolor="#7560A8",
                linewidth=0.35,
                zorder=2,
                transform=transform,
            )
        )
    carrier_outline = np.asarray(
        (
            (-4.8, 2.75),
            (-3.1, 2.75),
            (-3.1, 0.75),
            (-stem_half_width, 0.75),
            (-stem_half_width, -1.25),
            (stem_half_width, -1.25),
            (stem_half_width, 0.75),
            (3.1, 0.75),
            (3.1, 2.75),
            (4.8, 2.75),
            (4.8, 3.4),
            (-4.8, 3.4),
        ),
        dtype=np.float64,
    )
    axes.add_patch(
        Polygon(
            carrier_outline,
            closed=True,
            facecolor=style.colors.carrier,
            edgecolor="#303438",
            linewidth=0.6,
            zorder=3,
            transform=transform,
        )
    )


def plot_structural_ablation_schematic(
    axes: plt.Axes,
    *,
    sample_count: int,
    style: PublicationStyle = DEFAULT_STYLE,
) -> None:
    """Draw the three paired structural counterfactuals left to right."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    centers_x = (-5.6, 0.0, 5.6)
    scale = 0.52
    _draw_structure(
        axes,
        centers_x[0],
        0.0,
        carrier=False,
        stem_half_width=1.35,
        void_half_width=0.0,
        void_height=0.0,
        scale=scale,
        style=style,
    )
    _draw_structure(
        axes,
        centers_x[1],
        0.0,
        carrier=True,
        stem_half_width=1.35,
        void_half_width=0.0,
        void_height=0.0,
        scale=scale,
        style=style,
    )
    _draw_structure(
        axes,
        centers_x[2],
        0.0,
        carrier=True,
        stem_half_width=1.00,
        void_half_width=2.45,
        void_height=0.75,
        scale=scale,
        style=style,
    )

    names = ("Soft-only", "No-void carrier", "LUMO")
    for center_x, name in zip(centers_x, names, strict=True):
        axes.text(center_x, -2.25, name, ha="center", va="top", fontsize=6.3)

    transitions = (
        (-3.05, -2.62, "+ carrier", style.colors.mechanical),
        (2.62, 3.05, "+ void", "#7560A8"),
    )
    for start, end, label, color in transitions:
        axes.annotate(
            "",
            xy=(end, 0.0),
            xytext=(start, 0.0),
            arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "#555555"},
        )
        axes.text(
            0.5 * (start + end),
            0.70,
            label,
            ha="center",
            va="center",
            color=color,
            fontsize=5.7,
        )

    axes.set_xlim(-8.55, 8.55)
    axes.set_ylim(-3.15, 3.15)
    axes.set_aspect("equal", adjustable="box")
    axes.axis("off")


def _style_cartesian_axes(axes: plt.Axes, style: PublicationStyle) -> None:
    for spine in axes.spines.values():
        spine.set_visible(True)
        spine.set_color("#5F5F5F")
        spine.set_linewidth(0.8)
    axes.tick_params(direction="out", pad=1.5)
    axes.grid(False)


def plot_carrier_identity_comparison(
    axes: plt.Axes,
    soft_contact: Sequence[float],
    carrier_contact: Sequence[float],
    *,
    style: PublicationStyle = DEFAULT_STYLE,
) -> None:
    """Plot paired soft-only and carrier contact objectives."""

    soft = np.asarray(soft_contact, dtype=np.float64)
    carrier = np.asarray(carrier_contact, dtype=np.float64)
    if soft.shape != carrier.shape or soft.ndim != 1:
        raise ValueError("carrier identity arrays must be matching one-dimensional data")
    if soft.size == 0 or not np.all(np.isfinite((soft, carrier))):
        raise ValueError("carrier identity arrays must contain finite samples")

    lower = float(min(np.min(soft), np.min(carrier)))
    upper = float(max(np.max(soft), np.max(carrier)))
    padding = max(0.035 * (upper - lower), 0.005)
    limits = (lower - padding, upper + padding)
    axes.plot(limits, limits, color="#909090", linewidth=0.8, linestyle="--", zorder=1)
    axes.scatter(
        soft,
        carrier,
        s=20,
        facecolor=style.colors.mechanical,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.93,
        zorder=2,
    )
    axes.set_xlim(limits)
    axes.set_ylim(limits)
    axes.set_aspect("equal", adjustable="box")
    axes.set_anchor("N")
    axes.set_xlabel(r"Soft-only $J_{contact}$", fontsize=6.4, labelpad=1.0)
    axes.set_ylabel(r"No-void $J_{contact}$", fontsize=6.4, labelpad=1.0)
    axes.tick_params(labelsize=5.8)

    delta = carrier - soft
    median = float(np.median(delta))
    axes.text(
        0.97,
        0.04,
        (
            f"{int(np.count_nonzero(delta > 0.0))} / {delta.size} improved\n"
            rf"median $\Delta J={median:+.3f}$"
        ),
        transform=axes.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.6,
        linespacing=1.10,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
    )
    _style_cartesian_axes(axes, style)


def plot_void_coupled_response(
    axes: plt.Axes,
    delta_contact: Sequence[float],
    delta_d_1n: Sequence[float],
    lumo_void_width_mm: Sequence[float],
    *,
    colormap: Colormap,
    normalization: Normalize | None = None,
    style: PublicationStyle = DEFAULT_STYLE,
) -> ScalarMappable:
    """Plot paired lateral-void effects and return the colorbar mappable."""

    mechanics = np.asarray(delta_contact, dtype=np.float64)
    optical = np.asarray(delta_d_1n, dtype=np.float64)
    void_width = np.asarray(lumo_void_width_mm, dtype=np.float64)
    if mechanics.shape != optical.shape or mechanics.shape != void_width.shape:
        raise ValueError("void-effect arrays must have matching shapes")
    if mechanics.ndim != 1 or mechanics.size == 0:
        raise ValueError("void-effect arrays must be non-empty one-dimensional data")
    if not np.all(np.isfinite((mechanics, optical, void_width))):
        raise ValueError("void-effect arrays must contain finite samples")

    zero = np.isclose(void_width, 0.0, atol=1.0e-12)
    finite = ~zero
    if not np.any(finite):
        raise ValueError("at least one finite-void sample is required")
    if normalization is None:
        normalization = Normalize(
            vmin=float(np.min(void_width[finite])),
            vmax=float(np.max(void_width[finite])),
        )
    mechanics_display = mechanics * 1.0e4
    optical_display = optical * 1.0e5
    axes.axhline(0.0, color="#909090", linewidth=0.75, linestyle="--", zorder=1)
    axes.axvline(0.0, color="#909090", linewidth=0.75, linestyle="--", zorder=1)
    axes.scatter(
        mechanics_display[finite],
        optical_display[finite],
        c=void_width[finite],
        cmap=colormap,
        norm=normalization,
        s=22,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.93,
        zorder=3,
    )
    axes.scatter(
        mechanics_display[zero],
        optical_display[zero],
        marker="x",
        s=30,
        color="#787878",
        linewidth=1.0,
        zorder=4,
    )
    axes.set_xlabel(
        r"$\Delta J_{contact}$ [$\times 10^{-4}$]",
        fontsize=6.4,
        labelpad=1.0,
    )
    axes.set_ylabel(
        r"$\Delta D(1\,\mathrm{N})$ [$\times 10^{-5}$]",
        fontsize=6.4,
        labelpad=1.0,
    )
    axes.tick_params(labelsize=5.8)

    median_mechanics = float(np.median(mechanics[finite]))
    median_optical = float(np.median(optical[finite]))
    axes.text(
        0.97,
        0.97,
        (
            f"finite void, $n={np.count_nonzero(finite)}$\n"
            rf"median $\Delta J={median_mechanics * 1.0e4:+.1f}$"
            "\n"
            rf"median $\Delta D={median_optical * 1.0e5:+.1f}$"
        ),
        transform=axes.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        linespacing=1.15,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
    )
    axes.text(
        0.97,
        0.06,
        rf"$\times\;w_v=0$, $n={np.count_nonzero(zero)}$",
        transform=axes.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.6,
        color="#666666",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
    )
    x_padding = 0.05 * max(float(np.ptp(mechanics_display)), 10.0)
    y_padding = 0.12 * max(float(np.ptp(optical_display)), 1.0)
    axes.set_xlim(
        float(np.min(mechanics_display)) - x_padding,
        float(np.max(mechanics_display)) + x_padding,
    )
    axes.set_ylim(
        float(np.min(optical_display)) - y_padding,
        float(np.max(optical_display)) + 3.0 * y_padding,
    )
    _style_cartesian_axes(axes, style)
    return ScalarMappable(norm=normalization, cmap=colormap)


def plot_pareto_small_multiple(
    axes: plt.Axes,
    j_contact: Sequence[float],
    j_obs: Sequence[float],
    void_width_mm: Sequence[float],
    pareto_mask: Sequence[bool],
    balanced_index: int,
    *,
    colormap: Colormap,
    normalization: Normalize,
    style: PublicationStyle = DEFAULT_STYLE,
) -> PathCollection:
    """Plot one evaluated empirical Pareto set on an existing axes."""

    contact = np.asarray(j_contact, dtype=np.float64)
    observation = np.asarray(j_obs, dtype=np.float64)
    void_width = np.asarray(void_width_mm, dtype=np.float64)
    pareto = np.asarray(pareto_mask, dtype=np.bool_)
    if not (
        contact.shape == observation.shape == void_width.shape == pareto.shape
        and contact.ndim == 1
        and contact.size > 0
    ):
        raise ValueError("Pareto arrays must be matching non-empty vectors")
    if not np.all(np.isfinite((contact, observation, void_width))):
        raise ValueError("Pareto arrays must contain finite values")
    if not np.any(pareto):
        raise ValueError("Pareto mask must select at least one design")
    if not 0 <= balanced_index < contact.size:
        raise IndexError("balanced_index is outside the evaluated design array")
    if not pareto[balanced_index]:
        raise ValueError("balanced design must be empirically Pareto optimal")

    observation_display = observation * 1.0e3
    axes.scatter(
        contact,
        observation_display,
        s=7,
        facecolor="#C8C8C8",
        edgecolor="none",
        alpha=0.62,
        zorder=1,
    )
    order = np.flatnonzero(pareto)[np.argsort(contact[pareto])]
    axes.plot(
        contact[order],
        observation_display[order],
        color="#777777",
        linewidth=0.6,
        zorder=2,
    )
    pareto_points = axes.scatter(
        contact[pareto],
        observation_display[pareto],
        c=void_width[pareto],
        cmap=colormap,
        norm=normalization,
        s=19,
        edgecolor="#3F3F3F",
        linewidth=0.35,
        alpha=0.94,
        zorder=3,
    )
    axes.scatter(
        contact[balanced_index],
        observation_display[balanced_index],
        marker="*",
        s=55,
        facecolor=style.colors.mechanical,
        edgecolor="#2F2F2F",
        linewidth=0.45,
        zorder=4,
    )
    axes.xaxis.set_major_locator(MaxNLocator(nbins=3))
    axes.yaxis.set_major_locator(MaxNLocator(nbins=3))
    axes.margins(x=0.06, y=0.10)
    _style_cartesian_axes(axes, style)
    return pareto_points
