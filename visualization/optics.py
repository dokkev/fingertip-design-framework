"""Shared optical field and optional ray/path display layers."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle

from optics import RaySegment, TransportResult
from optics.transport3d import Transport3DResult
from visualization._axes import apply_physical_axes, bounds_from_geometries
from visualization._plotting import add_polygonal_patches
from visualization._style import OPTICS_CMAP, STYLE


OPTICAL_DISPLAY_GAMMA = 0.45
OPTICAL_UPPER_PERCENTILE = 99.5
OPTICAL_SMOOTHING_RADIUS_CELLS = 1
OPTICAL_DISPLAY_FLOOR_FRACTION = 1.0e-4
MAX_REPRESENTATIVE_PATHS = 100
MAX_DEBUG_PATHS = 600


def _optical_grid(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the result-owned y/x optical grid and physical mask."""
    if isinstance(result, TransportResult):
        x_edges = np.asarray(result.x_edges, dtype=float)
        y_edges = np.asarray(result.y_edges, dtype=float)
        field = np.asarray(result.density, dtype=float)
        mask = np.asarray(result.optical_mask, dtype=bool)
    elif isinstance(result, Transport3DResult) or hasattr(result, "projected_weighted_path_density"):
        if (
            getattr(result, "projected_x_edges_mm", None) is None
            or getattr(result, "projected_y_edges_mm", None) is None
            or getattr(result, "projected_weighted_path_density", None) is None
        ):
            raise ValueError(
                "PLANAR_2D result must retain its projected field and optical mask"
            )
        x_edges = np.asarray(result.projected_x_edges_mm, dtype=float)
        y_edges = np.asarray(result.projected_y_edges_mm, dtype=float)
        field = np.asarray(result.projected_weighted_path_density, dtype=float)
        supplied_mask = getattr(result, "projected_optical_mask", None)
        if supplied_mask is None:
            supplied_mask = getattr(result, "optical_mask", None)
        mask = np.ones_like(field, dtype=bool) if supplied_mask is None else np.asarray(supplied_mask, dtype=bool)
    else:
        raise TypeError("result must be a TransportResult or Transport3DResult")
    if field.shape != (len(y_edges) - 1, len(x_edges) - 1):
        raise ValueError("optical field shape does not match its axes")
    if mask.shape != field.shape:
        raise ValueError("optical mask shape does not match its field")
    if (
        not np.all(np.isfinite(x_edges))
        or not np.all(np.isfinite(y_edges))
        or not np.all(np.isfinite(field))
        or np.any(field < 0.0)
        or np.any(np.diff(x_edges) <= 0.0)
        or np.any(np.diff(y_edges) <= 0.0)
    ):
        raise ValueError("optical result contains invalid grid data")
    return x_edges, y_edges, field, mask


def _smooth_display_field(
    field: np.ndarray,
    domain_mask: np.ndarray,
    *,
    radius_cells: int,
) -> np.ndarray:
    """Smooth a copied raster for display only; never mutate the raw field."""
    if radius_cells not in (0, 1):
        raise ValueError("optical display smoothing is limited to zero or one cell")
    source = np.where(domain_mask, np.asarray(field, dtype=float), 0.0)
    if radius_cells == 0:
        return source.copy()
    kernel_1d = np.asarray([1.0, 2.0, 1.0], dtype=float)
    kernel = np.outer(kernel_1d, kernel_1d)
    padded_source = np.pad(source, 1, mode="constant")
    padded_domain = np.pad(domain_mask.astype(float), 1, mode="constant")
    smoothed = np.zeros_like(source)
    weights = np.zeros_like(source)
    for row in range(3):
        for column in range(3):
            weight = kernel[row, column]
            source_slice = padded_source[
                row : row + source.shape[0], column : column + source.shape[1]
            ]
            domain_slice = padded_domain[
                row : row + source.shape[0], column : column + source.shape[1]
            ]
            smoothed += weight * source_slice
            weights += weight * domain_slice
    return np.divide(smoothed, weights, out=np.zeros_like(source), where=weights > 0.0)


def shared_optical_normalization(
    results: tuple[Any, ...],
) -> tuple[PowerNorm, Any]:
    """Build one robust PowerNorm from all positive in-domain raw fields."""
    if not results:
        raise ValueError("at least one optical result is required")
    positive = []
    for result in results:
        _, _, field, mask = _optical_grid(result)
        values = field[mask & np.isfinite(field) & (field > 0.0)]
        if len(values):
            positive.append(values)
    if not positive:
        raise ValueError("optical results contain no positive in-domain transport")
    vmax = float(np.percentile(np.concatenate(positive), OPTICAL_UPPER_PERCENTILE))
    if not np.isfinite(vmax) or vmax <= 0.0:
        raise ValueError("optical results have no finite positive display scale")
    return (
        PowerNorm(gamma=OPTICAL_DISPLAY_GAMMA, vmin=0.0, vmax=vmax, clip=True),
        plt.get_cmap(OPTICS_CMAP).with_extremes(bad=STYLE.masked_cell),
    )


def display_optical_field(
    result: Any,
    *,
    norm: PowerNorm,
    smoothing_radius_cells: int = OPTICAL_SMOOTHING_RADIUS_CELLS,
    floor_fraction: float = OPTICAL_DISPLAY_FLOOR_FRACTION,
) -> np.ma.MaskedArray:
    """Return a masked display copy, with raw arrays left untouched.

    The floor and one-cell raster filter suppress sampling traces and reduce
    bin aliasing for figures only. Evaluation and optimization consume the
    result-owned field directly.
    """
    _, _, field, mask = _optical_grid(result)
    if not np.isfinite(floor_fraction) or floor_fraction < 0.0:
        raise ValueError("floor_fraction must be finite and nonnegative")
    smoothed = _smooth_display_field(
        field,
        mask,
        radius_cells=smoothing_radius_cells,
    )
    threshold = float(norm.vmax) * floor_fraction
    suppressed = (~mask) | (~np.isfinite(field)) | (~np.isfinite(smoothed))
    suppressed |= field <= threshold
    return np.ma.masked_where(suppressed, smoothed)


def draw_optical_field(
    ax: Axes,
    result: Any,
    *,
    norm: PowerNorm,
    cmap: Any,
    smoothing_radius_cells: int = OPTICAL_SMOOTHING_RADIUS_CELLS,
    floor_fraction: float = OPTICAL_DISPLAY_FLOOR_FRACTION,
) -> Any:
    """Draw only the scalar optical field; no geometry, rays, or colorbar."""
    x_edges, y_edges, _, _ = _optical_grid(result)
    image = ax.pcolormesh(
        x_edges,
        y_edges,
        display_optical_field(
            result,
            norm=norm,
            smoothing_radius_cells=smoothing_radius_cells,
            floor_fraction=floor_fraction,
        ),
        shading="flat",
        cmap=cmap,
        norm=norm,
        rasterized=True,
        zorder=0,
    )
    return image


def _legacy_segments(result: TransportResult) -> tuple[RaySegment, ...]:
    return tuple(result.segments)


def _production_segments(result: Transport3DResult) -> tuple[tuple[np.ndarray, np.ndarray, int, float, float, int], ...]:
    values = (
        result.retained_segment_starts_mm,
        result.retained_segment_ends_mm,
        result.retained_segment_media,
        result.retained_segment_start_weights,
        result.retained_segment_end_weights,
        result.retained_segment_primary_ray_indices,
    )
    if any(value is None for value in values):
        return ()
    starts, ends, media, start_weights, end_weights, primary = values
    return tuple(
        (
            np.asarray(starts[index], dtype=float),
            np.asarray(ends[index], dtype=float),
            int(media[index]),
            float(start_weights[index]),
            float(end_weights[index]),
            int(primary[index]),
        )
        for index in range(len(starts))
    )


def draw_ray_paths(
    ax: Axes,
    result: Any,
    *,
    maximum_display_paths: int = MAX_REPRESENTATIVE_PATHS,
    weight_floor_fraction: float = OPTICAL_DISPLAY_FLOOR_FRACTION,
) -> int:
    """Draw deterministic, bounded representative paths for debug figures."""
    if maximum_display_paths < 1:
        raise ValueError("maximum_display_paths must be positive")
    if weight_floor_fraction < 0.0 or not np.isfinite(weight_floor_fraction):
        raise ValueError("weight_floor_fraction must be finite and nonnegative")
    records = []
    if isinstance(result, TransportResult):
        for segment in _legacy_segments(result):
            records.append(
                (
                    np.asarray(segment.start, dtype=float),
                    np.asarray(segment.end, dtype=float),
                    1 if segment.medium == "silicone" else 0,
                    float(segment.start_weight),
                    float(segment.end_weight),
                    int(segment.ray_index),
                )
            )
    elif isinstance(result, Transport3DResult):
        records = list(_production_segments(result))
    else:
        raise TypeError("result must be a TransportResult or Transport3DResult")
    if not records:
        return 0
    maximum_weight = max(max(record[3], record[4]) for record in records)
    floor = maximum_weight * weight_floor_fraction
    records = [record for record in records if max(record[3], record[4]) >= floor]
    primary_ids = np.asarray(sorted({record[5] for record in records}), dtype=np.int64)
    if len(primary_ids) > maximum_display_paths:
        selected = np.linspace(0, len(primary_ids) - 1, maximum_display_paths, dtype=int)
        primary_ids = primary_ids[selected]
    selected_ids = set(int(value) for value in primary_ids)
    labels: set[int] = set()
    for start, end, medium, _, _, primary in records:
        if primary not in selected_ids:
            continue
        color = STYLE.silicone_ray if medium else STYLE.air_ray
        label = medium if medium not in labels else "_nolegend_"
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=0.55,
            alpha=0.62,
            label=("Silicone ray" if medium else "Air ray") if label != "_nolegend_" else label,
            zorder=4,
        )
        labels.add(medium)
    return len(selected_ids)


def draw_legacy_optical_geometry(ax: Axes, result: TransportResult) -> None:
    """Draw the optional physical context carried by the legacy result."""
    if not result.air_region.is_empty:
        add_polygonal_patches(
            ax,
            result.air_region,
            facecolor=STYLE.void_face,
            edgecolor=STYLE.void_edge,
            linewidth=1.1,
            linestyle="--",
            label="Internal air",
            zorder=1,
        )
    add_polygonal_patches(
        ax,
        result.silicone_region,
        facecolor=(0.0, 0.0, 0.0, 0.0),
        edgecolor=STYLE.silicone_edge,
        linewidth=1.8,
        label="Silicone pad",
        zorder=3,
    )
    add_polygonal_patches(
        ax,
        result.rigid_region,
        facecolor=STYLE.rigid_face,
        edgecolor=STYLE.rigid_edge,
        linewidth=1.5,
        label="Rigid link / stem",
        zorder=5,
    )
    min_x, min_y, max_x, max_y = result.led_region.bounds
    ax.add_patch(
        Rectangle(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            facecolor=STYLE.led_face,
            edgecolor=STYLE.led_edge,
            linewidth=1.2,
            label="LED",
            zorder=7,
        )
    )
    ax.scatter(
        [result.source[0]],
        [result.source[1]],
        s=42.0,
        color=STYLE.source_face,
        edgecolors=STYLE.source_edge,
        linewidths=0.8,
        label="Light source",
        zorder=8,
    )


def plot_transport(
    result: TransportResult | Transport3DResult,
    *,
    ax: Axes | None = None,
    norm: PowerNorm | None = None,
    normalization_max: float | None = None,
    show_rays: bool = False,
    debug: bool = False,
    maximum_display_paths: int = MAX_REPRESENTATIVE_PATHS,
    title: str = "Optical path-density field",
) -> Axes:
    """Plot an optical field; enable ``debug`` for bounded ray paths/exits."""
    if not isinstance(result, (TransportResult, Transport3DResult)):
        if not hasattr(result, "projected_weighted_path_density"):
            raise TypeError("result must be a TransportResult or Transport3DResult")
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 7.0))
    if norm is None:
        if normalization_max is not None:
            if not np.isfinite(normalization_max) or normalization_max <= 0.0:
                raise ValueError("normalization_max must be finite and positive")
            norm = PowerNorm(gamma=OPTICAL_DISPLAY_GAMMA, vmin=0.0, vmax=normalization_max, clip=True)
        else:
            norm, _ = shared_optical_normalization((result,))
    cmap = plt.get_cmap(OPTICS_CMAP).with_extremes(bad=STYLE.masked_cell)
    image = draw_optical_field(ax, result, norm=norm, cmap=cmap)
    if isinstance(result, TransportResult):
        draw_legacy_optical_geometry(ax, result)
        bounds = bounds_from_geometries(result.outer_envelope)
    else:
        x_edges, y_edges, _, _ = _optical_grid(result)
        bounds = (
            float(x_edges[0]),
            float(x_edges[-1]),
            float(y_edges[0]),
            float(y_edges[-1]),
        )
    if show_rays or debug:
        draw_ray_paths(
            ax,
            result,
            maximum_display_paths=(MAX_DEBUG_PATHS if debug else maximum_display_paths),
        )
        if debug and isinstance(result, Transport3DResult):
            positions = np.asarray(result.escape_positions_mm, dtype=float)
            weights = np.asarray(result.escape_weights, dtype=float)
            if len(positions):
                selected = np.argsort(weights, kind="stable")[::-1][:maximum_display_paths]
                ax.scatter(
                    positions[selected, 0],
                    positions[selected, 1],
                    s=8.0,
                    color=STYLE.debug_overlay,
                    alpha=0.35,
                    linewidths=0.0,
                    label="OptiX exits",
                    zorder=5,
                )
    apply_physical_axes(ax, bounds)
    ax.set_title(title)
    if debug:
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label(
            "Weighted optical path density"
        )
        ax.legend(loc="upper center", fontsize=8, ncol=2)
    return ax


__all__ = [
    "draw_optical_field",
    "draw_ray_paths",
    "display_optical_field",
    "plot_transport",
    "shared_optical_normalization",
]
