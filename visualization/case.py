"""Paper comparison composer for precomputed mechanics and optical results."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.figure import Figure
import numpy as np

from visualization._axes import apply_physical_axes, bounds_from_points
from visualization._style import MECHANICS_CMAP, STYLE
from visualization.geometry import draw_light_source, draw_outline
from visualization.mechanics import (
    _mesh_arrays,
    _pad_view,
    draw_contact_patch,
    draw_fea,
    draw_pad_outline,
)
from visualization.optics import (
    _optical_grid,
    _smooth_display_field,
    draw_optical_field,
    draw_ray_paths,
    shared_optical_normalization,
)


OPTICAL_DISPLAY_GAMMA = 0.45
OPTICAL_UPPER_PERCENTILE = 99.5
OPTICAL_SMOOTHING_RADIUS_CELLS = 1


def _optical_field(raw: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw y/x field data without making a display copy."""
    x_edges, y_edges, field, _ = _optical_grid(raw)
    return x_edges, y_edges, field


def _shared_optical_norm(raw_results: tuple[Any, ...], *_args: Any, **_kwargs: Any) -> tuple[PowerNorm, Any]:
    """Compatibility-local name for the common result-owned normalization."""
    return shared_optical_normalization(raw_results)


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
        pose=result.indenter_pose if loaded else unloaded_pose,
        show_contact_patch=loaded,
    )
    ax.set_title("FEA — loaded" if loaded else "FEA — unloaded reference (zero stress)")


def _plot_optical_panel(
    ax: Any,
    case: Any,
    raw: Any,
    *,
    loaded: bool,
    optical_norm: PowerNorm,
    optical_cmap: Any,
    debug: bool,
) -> None:
    draw_optical_field(ax, raw, norm=optical_norm, cmap=optical_cmap)
    result = case.fea.result
    if result is None or result.reference_mesh is None:
        raise RuntimeError("case FEA reference mesh is unavailable")
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
    draw_light_source(ax, case.fingertip)
    if debug:
        if hasattr(raw, "segments") or hasattr(raw, "retained_segment_starts_mm"):
            draw_ray_paths(ax, raw, maximum_display_paths=600)
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
    ax.set_title("PLANAR_2D OptiX — loaded" if loaded else "PLANAR_2D OptiX — unloaded")


def _comparison_limits(case: Any, unloaded_optics: Any) -> tuple[float, float, float, float]:
    result = case.fea.result
    if result is None or result.reference_mesh is None:
        raise RuntimeError("case FEA reference mesh is unavailable")
    reference, _ = _mesh_arrays(result.reference_mesh)
    loaded, _ = _mesh_arrays(result.deformed_mesh)
    geometries = [
        case.fingertip.geometry.raw_material_geometry,
        case.fingertip.geometry.outer_pad_geometry,
    ]
    points = [reference, loaded]
    for raw in (unloaded_optics, case.raytracing.raw):
        x_edges, y_edges, _, _ = _optical_grid(raw)
        points.append(np.asarray([[x_edges[0], y_edges[0]], [x_edges[-1], y_edges[-1]]]))
    if result.indenter_pose is not None:
        x, y = result.indenter_pose.carrier_geometry.exterior.xy
        points.append(np.column_stack((x, y)))
    geometry_points = []
    for geometry in geometries:
        min_x, min_y, max_x, max_y = geometry.bounds
        geometry_points.append(np.asarray([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]], dtype=float))
    return bounds_from_points(*(points + geometry_points), padding=0.05, minimum_span=2.0)


def plot_case_comparison(
    case: Any,
    unloaded_optics: Any,
    *,
    unloaded_pose: Any | None = None,
    show_exits: bool = False,
    title: str | None = None,
) -> Figure:
    """Compose shared-renderer FEA and PLANAR_2D optical paper panels."""
    required = ("fingertip", "fea", "raytracing")
    if any(not hasattr(case, name) for name in required):
        raise TypeError("case must expose the FingertipCase visualization contract")
    if case.fea.result is None or case.fea.result.indenter_pose is None:
        raise RuntimeError("loaded FEA result and indenter pose are required")
    if case.fea.result.reference_mesh is None or case.raytracing.raw is None:
        raise RuntimeError("completed FEA and optical results are required")
    loaded_optics = case.raytracing.raw
    _optical_grid(unloaded_optics)
    _optical_grid(loaded_optics)
    stress_mapping = case.fea.result.element_von_mises_stress_mpa
    if stress_mapping is None:
        raise RuntimeError("loaded FEA result has no von Mises stress field")
    stress_values = _stress_values(case.fea.result.reference_mesh, dict(stress_mapping))
    stress_vmax = float(np.max(stress_values))
    if not np.isfinite(stress_vmax) or stress_vmax <= 0.0:
        raise ValueError("loaded FEA stress maximum must be finite and positive")
    stress_norm = Normalize(vmin=0.0, vmax=stress_vmax)
    stress_cmap = plt.get_cmap(MECHANICS_CMAP)
    optical_norm, optical_cmap = shared_optical_normalization((unloaded_optics, loaded_optics))

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 10.0), constrained_layout=True, squeeze=False)
    _plot_fea_panel(axes[0, 0], case, loaded=False, stress_norm=stress_norm, stress_cmap=stress_cmap, unloaded_pose=unloaded_pose)
    _plot_fea_panel(axes[0, 1], case, loaded=True, stress_norm=stress_norm, stress_cmap=stress_cmap, unloaded_pose=None)
    _plot_optical_panel(axes[1, 0], case, unloaded_optics, loaded=False, optical_norm=optical_norm, optical_cmap=optical_cmap, debug=show_exits)
    _plot_optical_panel(axes[1, 1], case, loaded_optics, loaded=True, optical_norm=optical_norm, optical_cmap=optical_cmap, debug=show_exits)
    bounds = _comparison_limits(case, unloaded_optics)
    for axis in axes.flat:
        apply_physical_axes(axis, bounds)

    stress_mappable = ScalarMappable(norm=stress_norm, cmap=stress_cmap)
    stress_mappable.set_array(np.asarray([], dtype=float))
    figure.colorbar(stress_mappable, ax=axes[0, :].tolist(), fraction=0.046, pad=0.04).set_label("Cauchy von Mises stress [MPa]")
    optical_mappable = ScalarMappable(norm=optical_norm, cmap=optical_cmap)
    optical_mappable.set_array(np.asarray([], dtype=float))
    figure.colorbar(optical_mappable, ax=axes[1, :].tolist(), fraction=0.046, pad=0.04).set_label("Weighted optical path density")
    figure.suptitle(title or "Fingertip unloaded vs loaded comparison")
    return figure


__all__ = ["plot_case_comparison"]
