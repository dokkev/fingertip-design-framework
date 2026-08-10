"""Matplotlib rendering for self-contained optical transport results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from optics import RaySegment, TransportResult
from visualization.geometry import (
    ALUMINUM_COLOR,
    ALUMINUM_EDGE,
    LED_COLOR,
    LED_EDGE,
    LIGHT_SOURCE_COLOR,
    LIGHT_SOURCE_EDGE,
    PAD_EDGE,
    VOID_COLOR,
    VOID_EDGE,
    _add_polygonal_patches,
)

RAY_DISPLAY_WEIGHT_THRESHOLD = 1.0e-4
AIR_RAY_COLOR = "#8BE9FD"
SILICONE_RAY_COLOR = "#FFF3B0"


def _selected_segments(
    segments: tuple[RaySegment, ...],
    maximum_display_segments: int,
) -> tuple[RaySegment, ...]:
    if maximum_display_segments < 1:
        raise ValueError("maximum_display_segments must be at least one")
    candidates = tuple(
        segment
        for segment in segments
        if max(segment.start_weight, segment.end_weight)
        >= RAY_DISPLAY_WEIGHT_THRESHOLD
    )
    if len(candidates) <= maximum_display_segments:
        return candidates
    indices = np.linspace(
        0,
        len(candidates) - 1,
        maximum_display_segments,
        dtype=int,
    )
    return tuple(candidates[index] for index in indices)


def plot_transport(
    result: TransportResult,
    *,
    ax: Axes | None = None,
    normalization_max: float | None = None,
    show_rays: bool = True,
    maximum_display_segments: int = 600,
    title: str = "Qualitative light transport",
) -> Axes:
    """Plot a normalized path-density proxy using result-owned geometry."""
    if not isinstance(result, TransportResult):
        raise TypeError("result must be a TransportResult")
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 7.0))

    if normalization_max is None:
        normalization_scale = float(np.max(result.density))
    else:
        normalization_scale = float(normalization_max)
        if not np.isfinite(normalization_scale) or normalization_scale <= 0.0:
            raise ValueError("normalization_max must be finite and positive")
    display_density = (
        np.zeros_like(result.density)
        if normalization_scale <= 0.0
        else np.clip(result.density / normalization_scale, 0.0, 1.0)
    )
    heatmap = ax.pcolormesh(
        result.x_edges,
        result.y_edges,
        np.ma.masked_where(~result.optical_mask, display_density),
        shading="flat",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        zorder=0,
    )

    if not result.air_region.is_empty:
        _add_polygonal_patches(
            ax,
            result.air_region,
            facecolor=VOID_COLOR,
            edgecolor=VOID_EDGE,
            linewidth=1.1,
            linestyle="--",
            label="Internal air",
            zorder=1,
        )
    _add_polygonal_patches(
        ax,
        result.silicone_region,
        facecolor=(0.0, 0.0, 0.0, 0.0),
        edgecolor=PAD_EDGE,
        linewidth=1.8,
        label="Silicone pad",
        zorder=3,
    )
    _add_polygonal_patches(
        ax,
        result.rigid_region,
        facecolor=ALUMINUM_COLOR,
        edgecolor=ALUMINUM_EDGE,
        linewidth=1.5,
        label="Rigid link / stem",
        zorder=5,
    )
    led_min_x, led_min_y, led_max_x, led_max_y = result.led_region.bounds
    ax.add_patch(
        Rectangle(
            (led_min_x, led_min_y),
            led_max_x - led_min_x,
            led_max_y - led_min_y,
            facecolor=LED_COLOR,
            edgecolor=LED_EDGE,
            linewidth=1.2,
            label="LED",
            zorder=7,
        )
    )
    ax.scatter(
        [result.source[0]],
        [result.source[1]],
        s=42.0,
        color=LIGHT_SOURCE_COLOR,
        edgecolors=LIGHT_SOURCE_EDGE,
        linewidths=0.8,
        label="Light source",
        zorder=8,
    )

    if show_rays:
        labeled_media: set[str] = set()
        for segment in _selected_segments(
            result.segments,
            maximum_display_segments,
        ):
            color = AIR_RAY_COLOR if segment.medium == "air" else SILICONE_RAY_COLOR
            label = (
                f"{segment.medium.capitalize()} ray"
                if segment.medium not in labeled_media
                else "_nolegend_"
            )
            ax.plot(
                [segment.start[0], segment.end[0]],
                [segment.start[1], segment.end[1]],
                color=color,
                linewidth=0.55,
                alpha=0.62,
                label=label,
                zorder=4,
            )
            labeled_media.add(segment.medium)

    colorbar = ax.figure.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Normalized weighted ray-path density")
    min_x, min_y, max_x, max_y = result.outer_envelope.bounds
    span = max(max_x - min_x, max_y - min_y, 2.0)
    padding = 0.08 * span
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title)
    ax.legend(loc="upper center", fontsize=8, ncol=2)
    return ax


__all__ = ["plot_transport"]
