"""Shared optical field and optional ray/path display layers."""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm, to_rgb
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
MAX_REPRESENTATIVE_PATHS = 240
MAX_DEBUG_PATHS = 600
RAY_PATH_LINEWIDTH = 0.35
# Presentation-only controls; these do not represent transport attenuation.
RAY_PATH_FADE_FRACTION = 0.4
RAY_PATH_SUBDIVISIONS = 8
# Fixed display-space layers approximate a narrow Gaussian transverse glow.
# Values are (linewidth multiplier, absolute alpha); they are not a physical
# scattering width or an irradiance model.
RAY_GLOW_LAYERS = (
    (6.0, 0.006),
    (4.5, 0.010),
    (3.2, 0.016),
    (2.2, 0.025),
    (1.4, 0.040),
    (1.0, 0.070),
)
RAY_CENTERLINE_WIDTH_MULTIPLIER = 0.70
RAY_CENTERLINE_ALPHA = 0.070


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
    alpha: float = 1.0,
) -> Any:
    """Draw only the scalar optical field; no geometry, rays, or colorbar."""
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be finite and lie in (0, 1]")
    ax.set_facecolor(STYLE.optical_background)
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
        alpha=alpha,
        zorder=0,
    )
    return image


def _legacy_segments(result: TransportResult) -> tuple[RaySegment, ...]:
    return tuple(result.segments)


def _production_segments(result: Any) -> tuple[tuple[np.ndarray, np.ndarray, int, float, float, int], ...]:
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


def _ray_path_records(result: Any) -> list[tuple[np.ndarray, np.ndarray, int, float, float, int]]:
    """Return display-only segment records without changing the transport result."""
    records: list[tuple[np.ndarray, np.ndarray, int, float, float, int]] = []
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
    elif isinstance(result, Transport3DResult) or hasattr(
        result, "retained_segment_starts_mm"
    ):
        records.extend(_production_segments(result))
    else:
        raise TypeError("result must be a TransportResult or retained-segment transport result")
    return records


def _eligible_primary_ray_ids(
    records: Sequence[tuple[np.ndarray, np.ndarray, int, float, float, int]],
    *,
    weight_floor_fraction: float,
) -> set[int]:
    if not records:
        return set()
    maximum_weight = max(max(record[3], record[4]) for record in records)
    floor = maximum_weight * weight_floor_fraction
    return {
        record[5]
        for record in records
        if max(record[3], record[4]) >= floor
    }


def _sample_primary_ray_ids(
    primary_ids: Sequence[int],
    *,
    maximum_display_paths: int,
) -> tuple[int, ...]:
    """Select evenly spaced sorted ray IDs for a deterministic display sample."""
    ordered = np.asarray(sorted(set(int(value) for value in primary_ids)), dtype=np.int64)
    if len(ordered) > maximum_display_paths:
        ordered = ordered[
            np.linspace(0, len(ordered) - 1, maximum_display_paths, dtype=int)
        ]
    return tuple(int(value) for value in ordered)


def shared_ray_sample_ids(
    results: Sequence[Any],
    *,
    maximum_display_paths: int = MAX_REPRESENTATIVE_PATHS,
    weight_floor_fraction: float = OPTICAL_DISPLAY_FLOOR_FRACTION,
) -> tuple[int, ...]:
    """Return one deterministic primary-ray sample shared by multiple views."""
    if not results:
        raise ValueError("at least one transport result is required")
    if maximum_display_paths < 1:
        raise ValueError("maximum_display_paths must be positive")
    if weight_floor_fraction < 0.0 or not np.isfinite(weight_floor_fraction):
        raise ValueError("weight_floor_fraction must be finite and nonnegative")
    eligible = [
        _eligible_primary_ray_ids(
            _ray_path_records(result),
            weight_floor_fraction=weight_floor_fraction,
        )
        for result in results
    ]
    shared = set.intersection(*eligible) if eligible else set()
    return _sample_primary_ray_ids(
        tuple(shared),
        maximum_display_paths=maximum_display_paths,
    )


def draw_ray_paths(
    ax: Axes,
    result: Any,
    *,
    maximum_display_paths: int = MAX_REPRESENTATIVE_PATHS,
    weight_floor_fraction: float = OPTICAL_DISPLAY_FLOOR_FRACTION,
    selected_primary_ray_indices: Sequence[int] | None = None,
) -> int:
    """Draw deterministic ray centerlines with a display-only soft glow.

    Retained transport segments are grouped by primary ray, subdivided for
    display, and emitted as layered ``LineCollection`` entries with
    per-segment RGBA. The layers are a narrow transverse presentation effect;
    the centerline remains the actual retained trajectory. Alpha is
    intentionally independent of transport weight: every selected ray remains
    visible and receives only a mild presentation-only distance fade from the
    source. This is not an attenuation or irradiance model and never changes
    the result-owned transport arrays.
    """
    if maximum_display_paths < 1:
        raise ValueError("maximum_display_paths must be positive")
    if weight_floor_fraction < 0.0 or not np.isfinite(weight_floor_fraction):
        raise ValueError("weight_floor_fraction must be finite and nonnegative")
    records = _ray_path_records(result)
    if not records:
        return 0
    eligible_ids = _eligible_primary_ray_ids(
        records,
        weight_floor_fraction=weight_floor_fraction,
    )
    if selected_primary_ray_indices is None:
        selected_ids = set(
            _sample_primary_ray_ids(
                tuple(eligible_ids),
                maximum_display_paths=maximum_display_paths,
            )
        )
    else:
        selected_ids = {
            int(value) for value in selected_primary_ray_indices
        } & eligible_ids
    selected_records = [
        record for record in records if record[5] in selected_ids
    ]
    records_by_primary: dict[
        int, list[tuple[np.ndarray, np.ndarray, int, float, float, int]]
    ] = {}
    for record in selected_records:
        records_by_primary.setdefault(record[5], []).append(record)

    display_segments: list[np.ndarray] = []
    display_path_fades: list[float] = []
    for ray_records in records_by_primary.values():
        lengths = np.asarray(
            [
                np.linalg.norm(record[1][:2] - record[0][:2])
                for record in ray_records
            ],
            dtype=float,
        )
        total_length = float(np.sum(lengths))
        if not np.isfinite(total_length) or total_length <= 0.0:
            continue
        distance_at_start = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
        for record, length, distance in zip(
            ray_records, lengths, distance_at_start, strict=True
        ):
            if length <= 0.0:
                continue
            start, end, medium, _, _, _ = record
            fractions = np.linspace(0.0, 1.0, RAY_PATH_SUBDIVISIONS + 1)
            start_xy = np.asarray(start[:2], dtype=float)
            end_xy = np.asarray(end[:2], dtype=float)
            for fraction_start, fraction_end in zip(
                fractions[:-1], fractions[1:], strict=True
            ):
                small_start = start_xy + fraction_start * (end_xy - start_xy)
                small_end = start_xy + fraction_end * (end_xy - start_xy)
                midpoint_s = (
                    float(distance)
                    + 0.5 * float(length) * (fraction_start + fraction_end)
                ) / total_length
                path_fade = 1.0 - RAY_PATH_FADE_FRACTION * midpoint_s
                display_segments.append(
                    np.asarray([small_start, small_end], dtype=float)
                )
                display_path_fades.append(path_fade)

    if not display_segments:
        return len(selected_ids)
    segments = np.asarray(display_segments, dtype=float)
    path_fades = np.asarray(display_path_fades, dtype=float)
    base_colors = np.ones((len(segments), 4), dtype=float)
    base_colors[:, :3] = to_rgb(STYLE.silicone_ray)
    for layer_index, (linewidth_multiplier, layer_alpha) in enumerate(
        RAY_GLOW_LAYERS
    ):
        layer_colors = base_colors.copy()
        layer_colors[:, 3] = layer_alpha * path_fades
        ax.add_collection(
            LineCollection(
                segments,
                colors=layer_colors,
                linewidths=RAY_PATH_LINEWIDTH * linewidth_multiplier,
                label="Ray glow" if layer_index == 0 else "_nolegend_",
                zorder=3.0 + 0.12 * layer_index,
                rasterized=True,
            )
        )
    centerline_colors = base_colors.copy()
    centerline_colors[:, 3] = RAY_CENTERLINE_ALPHA * path_fades
    ax.add_collection(
        LineCollection(
            segments,
            colors=centerline_colors,
            linewidths=RAY_PATH_LINEWIDTH * RAY_CENTERLINE_WIDTH_MULTIPLIER,
            label="Ray centerline",
            zorder=4.0,
            rasterized=True,
        )
    )
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
    "shared_ray_sample_ids",
    "shared_optical_normalization",
]
