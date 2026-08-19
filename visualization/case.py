"""Paper comparison composer for precomputed mechanics and optical results."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import numpy as np
from shapely.geometry import LineString

from visualization._axes import apply_physical_axes, bounds_from_geometries
from visualization._style import MECHANICS_CMAP, STYLE
from visualization.geometry import (
    draw_led,
    draw_light_source,
    draw_outline,
    draw_rigid_structure,
)
from visualization.mechanics import (
    _pad_view,
    draw_contact_patch,
    draw_fea,
    draw_pad_outline,
)
from model import Fingertip, FingertipParameters
from visualization.optics import (
    MAX_DEBUG_PATHS,
    MAX_REPRESENTATIVE_PATHS,
    draw_ray_paths,
    shared_ray_sample_ids,
)


def _stress_values(reference_mesh: Any, stress_by_element_id: dict[int, float]) -> np.ndarray:
    pad = _pad_view(reference_mesh)
    if hasattr(reference_mesh, "pad_elements"):
        element_ids = [int(element.id) for element in reference_mesh.pad_elements]
    else:
        element_ids = list(range(len(pad.triangles)))
    values = np.asarray([stress_by_element_id[element_id] for element_id in element_ids], dtype=float)
    if values.shape != (len(pad.triangles),) or not np.all(np.isfinite(values)):
        raise ValueError("FEA stress must contain one finite value per pad element")
    if np.any(values < 0.0):
        raise ValueError("FEA von Mises stress must be nonnegative")
    return values


def fingertip_plot_limits(parameters: FingertipParameters) -> tuple[float, float, float, float]:
    """Return shared display limits derived only from fingertip geometry."""
    if not isinstance(parameters, FingertipParameters):
        raise TypeError("parameters must be FingertipParameters")
    tip = Fingertip(parameters)
    return bounds_from_geometries(
        tip.geometry.raw_material_geometry,
        padding=0.04,
        minimum_span=2.0,
    )


def _outgoing_flux_profile(
    raw: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw binned lateral outgoing flux without display smoothing."""
    profile_method = getattr(raw, "lateral_outgoing_profiles", None)
    if not callable(profile_method):
        raise TypeError(
            "optical result must expose lateral_outgoing_profiles()"
        )
    edges, left, right = profile_method()
    edges = np.asarray(edges, dtype=float)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if (
        edges.ndim != 1
        or left.shape != (len(edges) - 1,)
        or right.shape != left.shape
        or len(edges) < 2
        or not np.all(np.isfinite(edges))
        or not np.all(np.diff(edges) > 0.0)
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or np.any(left < 0.0)
        or np.any(right < 0.0)
    ):
        raise ValueError("lateral outgoing profile is invalid")
    return edges, left, right


def _ordered_boundary_line(
    edges: Any,
    coordinates: Any,
    reference_line: LineString,
) -> LineString:
    """Reconstruct one loaded boundary chain in reference-line orientation."""
    edge_array = np.asarray(edges, dtype=np.int64)
    points = np.asarray(coordinates, dtype=float)
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("boundary edges must have shape (N, 2)")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("boundary coordinates must have shape (N, 2)")
    if np.any(edge_array < 0) or np.any(edge_array >= len(points)):
        raise ValueError("boundary edge index is out of range")
    adjacency: dict[int, set[int]] = {}
    for first, second in edge_array:
        first_index = int(first)
        second_index = int(second)
        adjacency.setdefault(first_index, set()).add(second_index)
        adjacency.setdefault(second_index, set()).add(first_index)
    if not adjacency:
        raise ValueError("boundary chain is empty")
    reference_start = np.asarray(reference_line.coords[0], dtype=float)
    start = min(
        adjacency,
        key=lambda index: float(np.linalg.norm(points[index] - reference_start)),
    )
    chain = [start]
    previous: int | None = None
    current = start
    while True:
        candidates = sorted(
            neighbour
            for neighbour in adjacency[current]
            if neighbour != previous and neighbour not in chain
        )
        if not candidates:
            break
        next_index = candidates[0]
        chain.append(next_index)
        previous, current = current, next_index
    if len(chain) != len(adjacency):
        raise ValueError("boundary chain is disconnected or contains a loop")
    return LineString(points[np.asarray(chain, dtype=np.int64)])


def _optical_boundary_line(case: Any, tag: str, *, loaded: bool) -> LineString:
    """Return an analytic or deformed outer boundary for display overlays."""
    reference_line = getattr(case.fingertip.geometry.boundaries, tag).geometry
    if not loaded:
        return reference_line
    result = case.fea.result
    try:
        pad = _pad_view(result.deformed_mesh)
        return _ordered_boundary_line(
            pad.boundary_edges_for(tag),
            pad.coordinates,
            reference_line,
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        # Lightweight visualization fixtures may not carry semantic mesh tags.
        # The production deformed pad does, so this fallback is display-only.
        return reference_line


def _optical_boundary_intervals(case: Any) -> dict[str, tuple[float, float]]:
    boundaries = case.fingertip.geometry.boundaries
    lines = {
        tag: getattr(boundaries, tag).geometry
        for tag in ("pad_outer_left", "pad_outer_arc", "pad_outer_right")
    }
    total = float(sum(line.length for line in lines.values()))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("fingertip optical boundary has no positive length")
    left_end = float(lines["pad_outer_left"].length / total)
    arc_end = float(
        (lines["pad_outer_left"].length + lines["pad_outer_arc"].length) / total
    )
    return {
        "pad_outer_left": (0.0, left_end),
        "pad_outer_right": (arc_end, 1.0),
    }


def _draw_outgoing_flux_boundaries(
    ax: Any,
    case: Any,
    raw: Any,
    *,
    loaded: bool,
    flux_norm: Normalize,
    flux_cmap: Any,
) -> None:
    """Map raw side profiles back onto the corresponding physical boundaries."""
    edges, left, right = _outgoing_flux_profile(raw)
    intervals = _optical_boundary_intervals(case)
    for tag, profile, linestyle in (
        ("pad_outer_left", left, "solid"),
        ("pad_outer_right", right, "dashed"),
    ):
        line = _optical_boundary_line(case, tag, loaded=loaded)
        start_u, end_u = intervals[tag]
        if end_u <= start_u or line.length <= 0.0:
            continue
        segments: list[np.ndarray] = []
        values: list[float] = []
        for index, value in enumerate(profile):
            overlap_start = max(float(edges[index]), start_u)
            overlap_end = min(float(edges[index + 1]), end_u)
            if overlap_end <= overlap_start or value <= 0.0:
                continue
            start_fraction = (overlap_start - start_u) / (end_u - start_u)
            end_fraction = (overlap_end - start_u) / (end_u - start_u)
            segments.append(
                np.asarray(
                    [
                        line.interpolate(start_fraction * line.length).coords[0],
                        line.interpolate(end_fraction * line.length).coords[0],
                    ],
                    dtype=float,
                )
            )
            values.append(float(value))
        if not segments:
            continue
        value_array = np.asarray(values, dtype=float)
        normalized = np.asarray(flux_norm(value_array), dtype=float)
        collection = LineCollection(
            np.asarray(segments, dtype=float),
            colors=flux_cmap(normalized),
            linewidths=0.9 + 2.5 * normalized,
            linestyles=linestyle,
            label=f"Outgoing optical flux — {tag.removeprefix('pad_outer_')}",
            zorder=9,
        )
        ax.add_collection(collection)


def _plot_fea_panel(
    ax: Any,
    case: Any,
    *,
    loaded: bool,
    stress_norm: Normalize,
    stress_cmap: Any,
    unloaded_pose: Any | None,
) -> None:
    result = case.fea.result
    if result is None or result.reference_mesh is None:
        raise RuntimeError("case FEA result and reference mesh are required")
    reference_mesh = result.reference_mesh
    stress = (
        _stress_values(reference_mesh, dict(result.element_von_mises_stress_mpa))
        if loaded
        else np.zeros(len(_pad_view(reference_mesh).triangles), dtype=float)
    )
    draw_fea(
        ax,
        reference_mesh,
        stress,
        field_name="von_mises",
        deformed_mesh=result.deformed_mesh if loaded else None,
        norm=stress_norm,
        cmap=stress_cmap,
        show_rigid_structure=True,
        pose=result.indenter_pose if loaded else None,
        show_contact_patch=loaded,
    )
    ax.set_title("FEA — loaded" if loaded else "FEA — unloaded reference (zero stress)")


def _plot_optical_panel(
    ax: Any,
    case: Any,
    raw: Any,
    *,
    loaded: bool,
    debug: bool,
    selected_primary_ray_indices: tuple[int, ...],
) -> None:
    result = case.fea.result
    if result is None or result.reference_mesh is None:
        raise RuntimeError("case FEA reference mesh is unavailable")
    ax.set_facecolor(STYLE.optical_background)
    # Keep the solid carrier behind the actual ray paths and other overlays.
    draw_rigid_structure(ax, case.fingertip, zorder=1)
    draw_ray_paths(
        ax,
        raw,
        maximum_display_paths=MAX_DEBUG_PATHS if debug else MAX_REPRESENTATIVE_PATHS,
        selected_primary_ray_indices=selected_primary_ray_indices,
    )
    if loaded:
        draw_pad_outline(ax, result.deformed_mesh, label="Loaded fingertip outline")
        draw_contact_patch(
            ax,
            result.indenter_pose,
            label="Loaded contact boundary",
            linewidth=2.0,
        )
    else:
        draw_outline(
            ax,
            case.fingertip.geometry.outer_pad_geometry,
            label="Unloaded fingertip outline",
        )
    draw_led(ax, case.fingertip)
    draw_light_source(ax, case.fingertip)
    if debug:
        if hasattr(raw, "escape_positions_mm") and hasattr(raw, "escape_weights"):
            positions = np.asarray(raw.escape_positions_mm, dtype=float)
            weights = np.asarray(raw.escape_weights, dtype=float)
            if len(positions):
                selected = np.argsort(weights, kind="stable")[::-1][:100]
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
                directions = getattr(raw, "escape_directions", None)
                if directions is not None:
                    directions = np.asarray(directions, dtype=float)
                    ax.quiver(
                        positions[selected, 0],
                        positions[selected, 1],
                        directions[selected, 0],
                        directions[selected, 1],
                        angles="xy",
                        scale_units="xy",
                        scale=3.0,
                        color=STYLE.debug_overlay,
                        width=0.0018,
                        zorder=6,
                    )
    state = "loaded" if loaded else "unloaded"
    title = f"PLANAR_2D OptiX — {state} (sampled ray paths)"
    if debug:
        title += " + sampled rays"
    ax.set_title(title)


def plot_case_comparison(
    case: Any,
    unloaded_optics: Any,
    *,
    unloaded_pose: Any | None = None,
    show_exits: bool = False,
    title: str | None = None,
) -> Figure:
    """Compose FEA and sampled-ray optical comparison panels.

    The paper optical panels use only actual retained transport path segments
    with a display-only green glow. ``show_exits=True`` adds supplementary
    exit overlays without changing the ray sample or transport result.
    """
    required = ("fingertip", "fea", "raytracing")
    if any(not hasattr(case, name) for name in required):
        raise TypeError("case must expose the FingertipCase visualization contract")
    if case.fea.result is None or case.fea.result.indenter_pose is None:
        raise RuntimeError("loaded FEA result and indenter pose are required")
    if case.fea.result.reference_mesh is None or case.raytracing.raw is None:
        raise RuntimeError("completed FEA and optical results are required")
    loaded_optics = case.raytracing.raw
    stress_mapping = case.fea.result.element_von_mises_stress_mpa
    if stress_mapping is None:
        raise RuntimeError("loaded FEA result has no von Mises stress field")
    stress_values = _stress_values(case.fea.result.reference_mesh, dict(stress_mapping))
    stress_vmax = float(np.max(stress_values))
    if not np.isfinite(stress_vmax) or stress_vmax <= 0.0:
        raise ValueError("loaded FEA stress maximum must be finite and positive")
    stress_norm = Normalize(vmin=0.0, vmax=stress_vmax)
    stress_cmap = plt.get_cmap(MECHANICS_CMAP)
    selected_primary_ray_indices = shared_ray_sample_ids(
        (unloaded_optics, loaded_optics),
        maximum_display_paths=(MAX_DEBUG_PATHS if show_exits else MAX_REPRESENTATIVE_PATHS),
    )

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 10.0), constrained_layout=True, squeeze=False)
    _plot_fea_panel(axes[0, 0], case, loaded=False, stress_norm=stress_norm, stress_cmap=stress_cmap, unloaded_pose=unloaded_pose)
    _plot_fea_panel(axes[0, 1], case, loaded=True, stress_norm=stress_norm, stress_cmap=stress_cmap, unloaded_pose=None)
    _plot_optical_panel(
        axes[1, 0],
        case,
        unloaded_optics,
        loaded=False,
        debug=show_exits,
        selected_primary_ray_indices=selected_primary_ray_indices,
    )
    _plot_optical_panel(
        axes[1, 1],
        case,
        loaded_optics,
        loaded=True,
        debug=show_exits,
        selected_primary_ray_indices=selected_primary_ray_indices,
    )
    bounds = fingertip_plot_limits(case.fingertip.parameters)
    for axis in axes.flat:
        apply_physical_axes(axis, bounds)

    stress_mappable = ScalarMappable(norm=stress_norm, cmap=stress_cmap)
    stress_mappable.set_array(np.asarray([], dtype=float))
    figure.colorbar(stress_mappable, ax=axes[0, :].tolist(), fraction=0.046, pad=0.04).set_label("Cauchy von Mises stress [MPa]")
    figure.suptitle(title or "Fingertip unloaded vs loaded comparison")
    return figure


__all__ = ["fingertip_plot_limits", "plot_case_comparison"]
