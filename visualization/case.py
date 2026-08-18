"""Matplotlib comparison view for one unloaded and loaded fingertip case."""

from __future__ import annotations

from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.figure import Figure
import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from visualization.geometry import plot_fingertip


OPTICAL_DISPLAY_GAMMA = 0.45
OPTICAL_UPPER_PERCENTILE = 99.5
OPTICAL_LOW_WEIGHT_FRACTION = 2.5e-3
OPTICAL_SMOOTHING_RADIUS_CELLS = 1


def _line_parts(geometry: Any) -> tuple[Any, ...]:
    if geometry is None:
        return ()
    if isinstance(geometry, LineString):
        return (geometry,)
    if isinstance(geometry, MultiLineString):
        return tuple(geometry.geoms)
    raise TypeError("line geometry must be a LineString or MultiLineString")


def _polygon_parts(geometry: Any) -> tuple[Any, ...]:
    if isinstance(geometry, Polygon):
        return (geometry,)
    if isinstance(geometry, MultiPolygon):
        return tuple(geometry.geoms)
    raise TypeError("polygon geometry must be a Polygon or MultiPolygon")


def _plot_geometry_outline(
    ax: Any,
    geometry: Any,
    *,
    color: str,
    linewidth: float,
    linestyle: str = "-",
    label: str,
    zorder: int = 6,
) -> None:
    for index, polygon in enumerate(_polygon_parts(geometry)):
        x, y = polygon.exterior.xy
        ax.plot(
            x,
            y,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label if index == 0 else "_nolegend_",
            zorder=zorder,
        )


def _plot_mesh_boundary(
    ax: Any,
    mesh: Any,
    tag: str,
    *,
    color: str,
    linewidth: float,
    linestyle: str = "-",
    label: str,
    zorder: int = 7,
) -> None:
    boundaries = getattr(mesh, "boundaries", {})
    if tag not in boundaries:
        return
    for index, edge in enumerate(boundaries[tag]):
        coordinates = np.asarray(mesh.coordinates[np.asarray(edge, dtype=int)])
        ax.plot(
            coordinates[:, 0],
            coordinates[:, 1],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label if index == 0 else "_nolegend_",
            zorder=zorder,
        )


def _plot_pose(
    ax: Any,
    pose: Any | None,
    *,
    show_contact_patch: bool,
) -> None:
    if pose is None:
        return
    carrier_x, carrier_y = pose.carrier_geometry.exterior.xy
    ax.fill(
        carrier_x,
        carrier_y,
        facecolor="#B8C2CC",
        edgecolor="#495057",
        alpha=0.30,
        label="Indenter pose",
        zorder=8,
    )
    if not show_contact_patch:
        return
    for index, line in enumerate(_line_parts(pose.contact_patch)):
        patch_x, patch_y = line.xy
        ax.plot(
            patch_x,
            patch_y,
            color="#D62728",
            linewidth=2.4,
            label="Exact mechanical contact patch" if index == 0 else "_nolegend_",
            zorder=10,
        )


def _pad_view(mesh: Any) -> Any:
    return mesh.pad if hasattr(mesh, "pad") else mesh


def _stress_values(
    reference_mesh: Any,
    stress_by_element_id: dict[int, float],
) -> np.ndarray:
    pad = _pad_view(reference_mesh)
    if hasattr(reference_mesh, "pad_elements"):
        element_ids = [int(element.id) for element in reference_mesh.pad_elements]
    else:
        element_ids = list(range(len(pad.triangles)))
    values = np.asarray(
        [stress_by_element_id[element_id] for element_id in element_ids],
        dtype=float,
    )
    if values.shape != (len(pad.triangles),) or not np.all(np.isfinite(values)):
        raise ValueError("loaded FEA stress must contain one finite value per pad element")
    if np.any(values < 0.0):
        raise ValueError("loaded FEA von Mises stress must be nonnegative")
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
    if result is None:
        raise RuntimeError("case FEA result is unavailable")
    reference_mesh = result.reference_mesh
    if reference_mesh is None:
        raise RuntimeError("case FEA reference mesh is unavailable")
    reference_pad = _pad_view(reference_mesh)
    plot_fingertip(
        case.fingertip,
        ax=ax,
        show_led=False,
        show_light_source=False,
        show_legend=False,
        title=None,
    )
    if loaded:
        if result.element_von_mises_stress_mpa is None:
            raise RuntimeError(
                "loaded FEA result has no extracted Cauchy von Mises stress"
            )
        mesh = result.deformed_mesh
        stress = _stress_values(
            reference_mesh,
            dict(result.element_von_mises_stress_mpa),
        )
        coordinates = np.asarray(mesh.coordinates, dtype=float)
        ax.tripcolor(
            coordinates[:, 0],
            coordinates[:, 1],
            reference_pad.triangles,
            facecolors=stress,
            shading="flat",
            cmap=stress_cmap,
            norm=stress_norm,
            edgecolors="#56616A",
            linewidth=0.18,
            alpha=0.88,
            zorder=2,
        )
        for tag in ("pad_outer_left", "pad_outer_arc", "pad_outer_right"):
            _plot_mesh_boundary(
                ax,
                mesh,
                tag,
                color="#343A40",
                linewidth=0.9,
                label="Deformed pad boundary" if tag == "pad_outer_arc" else "_nolegend_",
            )
        _plot_pose(ax, result.indenter_pose, show_contact_patch=True)
        ax.set_title("FEA — loaded")
    else:
        coordinates = np.asarray(reference_pad.coordinates, dtype=float)
        ax.triplot(
            coordinates[:, 0],
            coordinates[:, 1],
            reference_pad.triangles,
            color="#56616A",
            linewidth=0.42,
            label="Reference pad mesh",
            zorder=2,
        )
        zero_stress = np.zeros(len(reference_pad.triangles), dtype=float)
        ax.tripcolor(
            coordinates[:, 0],
            coordinates[:, 1],
            reference_pad.triangles,
            facecolors=zero_stress,
            shading="flat",
            cmap=stress_cmap,
            norm=stress_norm,
            alpha=0.18,
            zorder=1,
        )
        _plot_pose(ax, unloaded_pose, show_contact_patch=False)
        ax.set_title("FEA — unloaded reference (zero stress)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")


def _optical_field(raw: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_edges = getattr(raw, "projected_x_edges_mm", None)
    y_edges = getattr(raw, "projected_y_edges_mm", None)
    field = getattr(raw, "projected_weighted_path_density", None)
    if x_edges is None or y_edges is None or field is None:
        raise ValueError("PLANAR_2D raw result must retain its projected path-density field")
    values = np.asarray(field, dtype=float)
    x_values = np.asarray(x_edges, dtype=float)
    y_values = np.asarray(y_edges, dtype=float)
    if values.shape != (len(y_values) - 1, len(x_values) - 1):
        raise ValueError("projected path-density field shape does not match its axes")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("projected path-density field must be finite and nonnegative")
    return x_values, y_values, values


def _shared_optical_norm(
    raw_results: Iterable[Any],
    domain_masks: Iterable[np.ndarray] | None = None,
) -> tuple[PowerNorm, Any]:
    results = tuple(raw_results)
    fields = [_optical_field(raw)[2] for raw in results]
    if domain_masks is not None:
        masks = tuple(domain_masks)
        if len(masks) != len(fields) or any(
            mask.shape != field.shape for mask, field in zip(masks, fields)
        ):
            raise ValueError("optical domain masks must match both optical fields")
        fields = [field[mask] for field, mask in zip(fields, masks)]
    positive_fields = [field[field > 0.0] for field in fields if np.any(field > 0.0)]
    if not positive_fields:
        raise ValueError("unloaded and loaded optical fields contain no positive transport")
    positive = np.concatenate(positive_fields)
    vmax = float(np.percentile(positive, OPTICAL_UPPER_PERCENTILE))
    if not np.isfinite(vmax) or vmax <= 0.0:
        raise ValueError("unloaded and loaded optical fields have no finite positive transport")
    return (
        PowerNorm(
            gamma=OPTICAL_DISPLAY_GAMMA,
            vmin=0.0,
            vmax=vmax,
            clip=True,
        ),
        plt.get_cmap("magma").with_extremes(bad="#FFFFFF"),
    )


def _optical_display_floor(optical_norm: PowerNorm) -> float:
    """Return a shared display-only floor for numerical path traces."""
    return max(
        float(optical_norm.vmax) * OPTICAL_LOW_WEIGHT_FRACTION,
        float(np.finfo(float).tiny),
    )


def _grid_domain_mask(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    domain: Any,
) -> np.ndarray:
    """Classify projected cell centers against the physical silicone domain."""
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return np.asarray(
        [
            domain.covers(Point(float(x), float(y)))
            for y in y_centers
            for x in x_centers
        ],
        dtype=bool,
    ).reshape(len(y_centers), len(x_centers))


def _silicone_domain(case: Any, *, loaded: bool) -> Any:
    if not loaded:
        return case.fingertip.geometry.pad_material_geometry
    result = case.fea.result
    if result is None:
        raise RuntimeError("case FEA result is unavailable")
    mesh = _pad_view(result.deformed_mesh)
    coordinates = np.asarray(mesh.coordinates, dtype=float)
    triangle_polygons = [
        Polygon(coordinates[triangle]) for triangle in np.asarray(mesh.triangles, dtype=int)
    ]
    domain = unary_union(triangle_polygons)
    if domain.is_empty or not domain.is_valid:
        raise ValueError("loaded silicone mesh did not produce a valid display domain")
    return domain


def _smooth_display_field(
    field: np.ndarray,
    domain_mask: np.ndarray,
    *,
    radius_cells: int,
) -> np.ndarray:
    """Apply at most one-cell display smoothing without changing the raw field."""
    if radius_cells < 0:
        raise ValueError("radius_cells must be nonnegative")
    source = np.where(domain_mask, np.asarray(field, dtype=float), 0.0)
    if radius_cells == 0:
        return source.copy()
    if radius_cells != 1:
        raise ValueError("publication optical smoothing is limited to one grid cell")

    kernel_1d = np.asarray([1.0, 2.0, 1.0], dtype=float)
    kernel = np.outer(kernel_1d, kernel_1d)
    padded_source = np.pad(source, 1, mode="constant")
    padded_domain = np.pad(domain_mask.astype(float), 1, mode="constant")
    smoothed = np.zeros_like(source)
    weights = np.zeros_like(source)
    for row in range(3):
        for column in range(3):
            weight = kernel[row, column]
            source_slice = padded_source[row : row + source.shape[0], column : column + source.shape[1]]
            domain_slice = padded_domain[row : row + source.shape[0], column : column + source.shape[1]]
            smoothed += weight * source_slice
            weights += weight * domain_slice
    return np.divide(smoothed, weights, out=np.zeros_like(source), where=weights > 0.0)


def _display_optical_field(
    raw: Any,
    *,
    domain: Any,
    display_floor: float,
    smoothing_radius_cells: int = OPTICAL_SMOOTHING_RADIUS_CELLS,
) -> np.ma.MaskedArray:
    """Build a masked, smoothed display copy; transport data stay untouched."""
    x_edges, y_edges, field = _optical_field(raw)
    domain_mask = _grid_domain_mask(x_edges, y_edges, domain)
    display_field = _smooth_display_field(
        field,
        domain_mask,
        radius_cells=smoothing_radius_cells,
    )
    suppressed = (
        ~domain_mask
        | ~np.isfinite(display_field)
        | (display_field <= display_floor)
    )
    return np.ma.masked_where(suppressed, display_field)


def _plot_optical_panel(
    ax: Any,
    case: Any,
    raw: Any,
    *,
    loaded: bool,
    optical_norm: PowerNorm,
    optical_cmap: Any,
    show_exits: bool,
    display_floor: float,
    domain: Any,
) -> None:
    x_edges, y_edges, _ = _optical_field(raw)
    image_field = _display_optical_field(
        raw,
        domain=domain,
        display_floor=display_floor,
    )
    ax.set_facecolor("#FFFFFF")
    ax.pcolormesh(
        x_edges,
        y_edges,
        image_field,
        shading="flat",
        cmap=optical_cmap,
        norm=optical_norm,
    )
    result = case.fea.result
    if result is None or result.reference_mesh is None:
        raise RuntimeError("case FEA reference mesh is unavailable")
    if loaded:
        mesh = result.deformed_mesh
        for tag in ("pad_outer_left", "pad_outer_arc", "pad_outer_right"):
            _plot_mesh_boundary(
                ax,
                mesh,
                tag,
                color="#D9F99D",
                linewidth=1.0,
                label="Deformed fingertip outline" if tag == "pad_outer_arc" else "_nolegend_",
            )
        for index, line in enumerate(_line_parts(result.indenter_pose.contact_patch)):
            patch_x, patch_y = line.xy
            ax.plot(
                patch_x,
                patch_y,
                color="#FF6B6B",
                linewidth=2.0,
                label="Object-contact boundary" if index == 0 else "_nolegend_",
                zorder=8,
            )
    else:
        _plot_geometry_outline(
            ax,
            case.fingertip.geometry.outer_pad_geometry,
            color="#D9F99D",
            linewidth=1.0,
            label="Reference fingertip outline",
        )
    source_x, source_y = case.fingertip.led_source
    ax.scatter(
        [source_x],
        [source_y],
        marker="*",
        s=42.0,
        color="#F8E71C",
        edgecolors="#212529",
        linewidths=0.45,
        label="LED/source",
        zorder=9,
    )
    if show_exits:
        positions = np.asarray(raw.escape_positions_mm, dtype=float)
        weights = np.asarray(raw.escape_weights, dtype=float)
        if len(positions):
            selected = np.argsort(weights, kind="stable")[::-1][:24]
            ax.scatter(
                positions[selected, 0],
                positions[selected, 1],
                s=8.0,
                color="#F8F9FA",
                alpha=0.28,
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
                    color="#F8F9FA",
                    width=0.0018,
                    zorder=6,
                )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("PLANAR_2D OptiX — loaded" if loaded else "PLANAR_2D OptiX — unloaded")


def _comparison_limits(case: Any, unloaded_optics: Any) -> tuple[float, float, float, float]:
    result = case.fea.result
    if result is None or result.reference_mesh is None:
        raise RuntimeError("case FEA reference mesh is unavailable")
    reference = np.asarray(_pad_view(result.reference_mesh).coordinates, dtype=float)
    loaded = np.asarray(result.deformed_mesh.coordinates, dtype=float)
    points = [reference, loaded]
    for geometry in (
        case.fingertip.geometry.outer_pad_geometry,
        case.fingertip.geometry.link_geometry,
    ):
        min_x, min_y, max_x, max_y = geometry.bounds
        points.append(
            np.asarray(
                [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]],
                dtype=float,
            )
        )
    for raw in (unloaded_optics, case.raytracing.raw):
        x_edges, y_edges, _ = _optical_field(raw)
        points.append(np.asarray([[x_edges[0], y_edges[0]], [x_edges[-1], y_edges[-1]]]))
    if result.indenter_pose is not None:
        points.append(np.asarray(result.indenter_pose.carrier_geometry.exterior.coords))
    minimum = np.min(np.vstack(points), axis=0)
    maximum = np.max(np.vstack(points), axis=0)
    padding = 0.05 * max(float(np.max(maximum - minimum)), 1.0)
    return (
        float(minimum[0] - padding),
        float(maximum[0] + padding),
        float(minimum[1] - padding),
        float(maximum[1] + padding),
    )


def plot_case_comparison(
    case: Any,
    unloaded_optics: Any,
    *,
    unloaded_pose: Any | None = None,
    show_exits: bool = False,
    title: str | None = None,
) -> Figure:
    """Plot precomputed unloaded and loaded FEA/PLANAR_2D states in 2x2.

    The optical panels use one shared, robust ``PowerNorm`` and a copied
    display raster. Silicone-domain masking, low-weight suppression, and the
    one-cell raster smoothing are visualization-only; raw transport arrays
    remain the inputs to evaluation and are never modified here. Set
    ``show_exits=True`` only for the optional ray/exit debug overlay.
    """
    required = ("fingertip", "fea", "raytracing")
    if any(not hasattr(case, name) for name in required):
        raise TypeError("case must expose the FingertipCase visualization contract")
    if case.fea.result is None or case.fea.result.indenter_pose is None:
        raise RuntimeError("loaded FEA result and indenter pose are required")
    if case.fea.result.reference_mesh is None:
        raise RuntimeError("case FEA reference mesh is unavailable")
    if case.raytracing.raw is None:
        raise RuntimeError("loaded PLANAR_2D optical result is required")
    loaded_optics = case.raytracing.raw
    _optical_field(unloaded_optics)
    _optical_field(loaded_optics)
    stress_mapping = case.fea.result.element_von_mises_stress_mpa
    if stress_mapping is None:
        raise RuntimeError("loaded FEA result has no Cauchy von Mises stress field")
    stress_values = _stress_values(
        case.fea.result.reference_mesh,
        dict(stress_mapping),
    )
    loaded_vmax = float(np.max(stress_values))
    if not np.isfinite(loaded_vmax) or loaded_vmax <= 0.0:
        raise ValueError("loaded FEA stress maximum must be finite and positive")
    stress_norm = Normalize(vmin=0.0, vmax=loaded_vmax)
    stress_cmap = plt.get_cmap("viridis")
    optical_results = (unloaded_optics, loaded_optics)
    optical_domains = (
        _silicone_domain(case, loaded=False),
        _silicone_domain(case, loaded=True),
    )
    optical_domain_masks = tuple(
        _grid_domain_mask(_optical_field(raw)[0], _optical_field(raw)[1], domain)
        for raw, domain in zip(optical_results, optical_domains)
    )
    optical_norm, optical_cmap = _shared_optical_norm(
        optical_results,
        optical_domain_masks,
    )
    display_floor = _optical_display_floor(optical_norm)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14.0, 10.0),
        constrained_layout=True,
        squeeze=False,
    )
    _plot_fea_panel(
        axes[0, 0],
        case,
        loaded=False,
        stress_norm=stress_norm,
        stress_cmap=stress_cmap,
        unloaded_pose=unloaded_pose,
    )
    _plot_fea_panel(
        axes[0, 1],
        case,
        loaded=True,
        stress_norm=stress_norm,
        stress_cmap=stress_cmap,
        unloaded_pose=None,
    )
    _plot_optical_panel(
        axes[1, 0],
        case,
        unloaded_optics,
        loaded=False,
        optical_norm=optical_norm,
        optical_cmap=optical_cmap,
        show_exits=show_exits,
        display_floor=display_floor,
        domain=optical_domains[0],
    )
    _plot_optical_panel(
        axes[1, 1],
        case,
        loaded_optics,
        loaded=True,
        optical_norm=optical_norm,
        optical_cmap=optical_cmap,
        show_exits=show_exits,
        display_floor=display_floor,
        domain=optical_domains[1],
    )
    x_min, x_max, y_min, y_max = _comparison_limits(case, unloaded_optics)
    for axis in axes.flat:
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_aspect("equal", adjustable="box")

    stress_mappable = ScalarMappable(norm=stress_norm, cmap=stress_cmap)
    stress_mappable.set_array(np.asarray([], dtype=float))
    figure.colorbar(
        stress_mappable,
        ax=axes[0, :].tolist(),
        fraction=0.046,
        pad=0.04,
    ).set_label("Cauchy von Mises stress [MPa]")
    optical_mappable = ScalarMappable(norm=optical_norm, cmap=optical_cmap)
    optical_mappable.set_array(np.asarray([], dtype=float))
    figure.colorbar(
        optical_mappable,
        ax=axes[1, :].tolist(),
        fraction=0.046,
        pad=0.04,
    ).set_label("P2 weighted path density (display-only)")
    if title is None:
        title = "Fingertip unloaded vs loaded comparison"
    figure.suptitle(title)
    return figure


__all__ = ["plot_case_comparison"]
