"""Matplotlib rendering for neutral 2D optical-transport results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from model.fingertip_sensor_model import FingertipSensorModel
from optics.cross_section.domain import CrossSectionOpticalDomain
from optics.cross_section.result import CrossSectionTransportResult, RaySegment2D
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
    segments: tuple[RaySegment2D, ...],
    maximum_display_segments: int,
) -> tuple[RaySegment2D, ...]:
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


def plot_cross_section_transport(
    sensor_model: FingertipSensorModel,
    domain: CrossSectionOpticalDomain,
    result: CrossSectionTransportResult,
    *,
    ax: Axes | None = None,
    normalization_max: float | None = None,
    show_rays: bool = True,
    maximum_display_segments: int = 600,
    title: str = "2D light transport",
) -> Axes:
    """Plot a transport result against its matching analytic or loaded domain."""
    if not np.allclose(
        result.source_position_mm,
        sensor_model.led_source_position_2d,
        rtol=0.0,
        atol=sensor_model.geometry.parameters.geometry_tolerance,
    ):
        raise ValueError("transport and sensor LED source positions do not match")
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 7.0))

    if normalization_max is None:
        normalization_scale = float(np.max(result.weighted_path_density))
    else:
        normalization_scale = float(normalization_max)
        if not np.isfinite(normalization_scale) or normalization_scale <= 0.0:
            raise ValueError("normalization_max must be finite and positive")
    display_density = (
        np.zeros_like(result.weighted_path_density)
        if normalization_scale <= 0.0
        else np.clip(result.weighted_path_density / normalization_scale, 0.0, 1.0)
    )
    masked_density = np.ma.masked_where(~result.optical_mask, display_density)
    heatmap = ax.pcolormesh(
        result.x_edges_mm,
        result.y_edges_mm,
        masked_density,
        shading="flat",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        zorder=0,
    )

    internal_air = domain.accessible_region.difference(domain.silicone_region)
    if not internal_air.is_empty:
        _add_polygonal_patches(
            ax,
            internal_air,
            facecolor=VOID_COLOR,
            edgecolor=VOID_EDGE,
            linewidth=1.1,
            linestyle="--",
            label="Internal air",
            zorder=1,
        )
    _add_polygonal_patches(
        ax,
        domain.silicone_region,
        facecolor=(0.0, 0.0, 0.0, 0.0),
        edgecolor=PAD_EDGE,
        linewidth=1.8,
        label="Silicone pad",
        zorder=3,
    )
    _add_polygonal_patches(
        ax,
        domain.rigid_region,
        facecolor=ALUMINUM_COLOR,
        edgecolor=ALUMINUM_EDGE,
        linewidth=1.5,
        label="Rigid link / stem",
        zorder=5,
    )
    led_min_x, led_min_y, led_max_x, led_max_y = (
        sensor_model.led_package_geometry.bounds
    )
    from matplotlib.patches import Rectangle

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
    source_x, source_y = sensor_model.led_source_position_2d
    ax.scatter(
        [source_x],
        [source_y],
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
                [segment.start_mm[0], segment.end_mm[0]],
                [segment.start_mm[1], segment.end_mm[1]],
                color=color,
                linewidth=0.55,
                alpha=0.62,
                label=label,
                zorder=4,
            )
            labeled_media.add(segment.medium)

    colorbar = ax.figure.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Normalized weighted ray-path density")
    min_x, min_y, max_x, max_y = domain.outer_envelope.bounds
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


def save_cross_section_transport_figure(
    sensor_model: FingertipSensorModel,
    domain: CrossSectionOpticalDomain,
    result: CrossSectionTransportResult,
    output_path: str | Path,
    *,
    dpi: int = 200,
    normalization_max: float | None = None,
    show_rays: bool = True,
    maximum_display_segments: int = 600,
    title: str = "2D light transport",
) -> Path:
    """Render a 2D transport result, save its PNG, and return its path."""
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.0, 7.0))
    plot_cross_section_transport(
        sensor_model,
        domain,
        result,
        ax=axis,
        normalization_max=normalization_max,
        show_rays=show_rays,
        maximum_display_segments=maximum_display_segments,
        title=title,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output
