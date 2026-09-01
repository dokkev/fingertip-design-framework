"""Export the four-panel fingertip geometry and optical overview."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from mpl_toolkits.mplot3d.art3d import (  # noqa: E402
    Line3DCollection,
    Poly3DCollection,
)
from shapely.geometry import LineString, Polygon as ShapelyPolygon  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from lumo.fingertip import (  # noqa: E402
    ACTIVE_Y_BOUNDS_MM,
    Fingertip,
    LED_RECESS_DEPTH_MM,
    TOTAL_Y_BOUNDS_MM,
)
from lumo.mesh import make_fingertip_mesh  # noqa: E402
from lumo.ray_tracing import (  # noqa: E402
    LED,
    OptixScene,
    emit_from_stem_window,
    sources_inside_silicone,
    trace_bounded_paths,
)
from lumo.visualization import (  # noqa: E402
    DEFAULT_STYLE,
    create_gridspec,
    plot_fingertip_parameterization,
    publication_context,
    save_figure,
)


_ROOT = Path(__file__).resolve().parents[2]
_INPUT = (
    _ROOT
    / "output"
    / "validation"
    / "fingertip_production_objective_freeze"
    / "nominal_fingertip_objectives.npz"
)
_OUTPUT_DIRECTORY = _ROOT / "output" / "figures"
_LOADED_SCENARIO = "sphere_15mm_y+0mm"
_LOADED_FORCE_N = 10.0

# A compact deterministic subset is enough to display the production transport
# model without turning the figure into a dense green block. Both states use
# the same angular, source-window, and path-branch samples.
_VISUALIZATION_SAMPLE_SIDE_COUNT = 8
_MAX_BOUNCES = 24
_RNG_SEED = 20260823
_SOURCE_RNG_SEED = 20260826
_CARRIER_ALBEDO = 0.7
_SLICE_TOLERANCE_M = 1.0e-10
_DISPLAY_SEGMENT_LENGTH_M = 0.5e-3


def _tet_slice_polygons_xz(
    vertices_m: np.ndarray,
    tetrahedra: np.ndarray,
) -> list[np.ndarray]:
    """Intersect the tetrahedral mesh with the physical Y=0 plane."""
    distances = vertices_m[:, 1]
    tet_distances = distances[tetrahedra]
    active_tetrahedra = tetrahedra[
        (tet_distances.min(axis=1) <= _SLICE_TOLERANCE_M)
        & (tet_distances.max(axis=1) >= -_SLICE_TOLERANCE_M)
    ]
    polygons: list[np.ndarray] = []
    for indices in active_tetrahedra:
        tet_vertices = vertices_m[indices]
        tet_plane_distances = tet_vertices[:, 1]
        points = [
            tet_vertices[index, (0, 2)]
            for index in range(4)
            if abs(tet_plane_distances[index]) <= _SLICE_TOLERANCE_M
        ]
        for first, second in combinations(range(4), 2):
            first_distance = tet_plane_distances[first]
            second_distance = tet_plane_distances[second]
            if first_distance * second_distance >= -_SLICE_TOLERANCE_M**2:
                continue
            fraction = first_distance / (first_distance - second_distance)
            intersection = tet_vertices[first] + fraction * (
                tet_vertices[second] - tet_vertices[first]
            )
            points.append(intersection[(0, 2),])

        unique_points: list[np.ndarray] = []
        for point in points:
            if not any(
                np.linalg.norm(point - existing) <= _SLICE_TOLERANCE_M
                for existing in unique_points
            ):
                unique_points.append(point)
        if len(unique_points) < 3:
            continue

        polygon = 1.0e3 * np.asarray(unique_points)
        center = polygon.mean(axis=0)
        angles = np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0])
        polygons.append(polygon[np.argsort(angles)])
    return polygons


def _make_leds(fingertip: Fingertip) -> tuple[LED, ...]:
    normal_W = np.array((0.0, 0.0, -1.0), dtype=np.float64)
    return tuple(
        LED(
            position_W_m=np.asarray(center_m, dtype=np.float64),
            normal_W=normal_W,
            parameters=fingertip.parameters.led,
        )
        for center_m in fingertip.led_source_centers_m
    )


def _optical_samples(ray_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_RNG_SEED)
    shape = (_MAX_BOUNCES, ray_count)
    return rng.random(shape), rng.random(shape), rng.random(shape)


def _emissions(scene: OptixScene, leds: tuple[LED, ...]) -> tuple[np.ndarray, ...]:
    side_count = _VISUALIZATION_SAMPLE_SIDE_COUNT
    coordinate = (np.arange(side_count, dtype=np.float64) + 0.5) / side_count
    angular_u1, angular_u2 = np.meshgrid(coordinate, coordinate, indexing="ij")
    angular_u1 = angular_u1.ravel()
    angular_u2 = angular_u2.ravel()
    source_coordinate = (
        np.arange(len(angular_u1), dtype=np.float64) + 0.5
    ) / len(angular_u1)
    source_rng = np.random.default_rng(_SOURCE_RNG_SEED)
    source_u_x = source_coordinate[source_rng.permutation(len(source_coordinate))]
    source_u_y = source_coordinate[source_rng.permutation(len(source_coordinate))]
    return tuple(
        emit_from_stem_window(
            scene,
            led,
            angular_u1,
            angular_u2,
            source_u_x,
            source_u_y,
        )
        for led in leds
    )


def _trace_state(
    scene: OptixScene,
    fingertip: Fingertip,
    leds: tuple[LED, ...],
    emissions: tuple[np.ndarray, ...],
    samples: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, float, float]:
    """Trace the current scene and retain power-carrying finite segments."""
    path_segments: list[np.ndarray] = []
    emitted_power = 0.0
    escaped_power = 0.0
    optics = fingertip.parameters.optics

    for led, emission in zip(leds, emissions, strict=True):
        inside_silicone = sources_inside_silicone(scene, led, emission)
        paths = trace_bounded_paths(
            scene,
            emission["origin_W_m"],
            emission["direction_W"],
            emission["power"],
            inside_silicone=inside_silicone,
            n_air=1.0,
            n_silicone=optics.refractive_index,
            extinction_coefficient_m_inv=optics.extinction_coefficient_m_inv,
            carrier_albedo=_CARRIER_ALBEDO,
            max_bounces=_MAX_BOUNCES,
            dielectric_branch_u=samples[0],
            carrier_u1=samples[1],
            carrier_u2=samples[2],
            record_segments=True,
        )
        path_segments.append(paths.path_segments)
        emitted_power += paths.emitted_power
        escaped_power += paths.escaped_power

    return np.concatenate(path_segments), emitted_power, escaped_power


def _add_optical_paths(
    axis,
    path_segments: np.ndarray,
    mesh_polygons: list[np.ndarray],
    *,
    reference_path_power: float,
    style,
) -> tuple[LineCollection, LineCollection]:
    active_lower_y_m, active_upper_y_m = 1.0e-3 * np.asarray(
        ACTIVE_Y_BOUNDS_MM,
        dtype=np.float64,
    )
    silicone_region = unary_union(
        tuple(ShapelyPolygon(polygon) for polygon in mesh_polygons)
    )
    visible_segments: list[np.ndarray] = []
    visible_power: list[float] = []
    for segment in path_segments:
        if not segment["inside_silicone"]:
            continue
        start = np.asarray(segment["origin_W_m"], dtype=np.float64)
        end = np.asarray(segment["end_W_m"], dtype=np.float64)
        delta = end - start
        if abs(delta[1]) <= np.finfo(np.float64).tiny:
            if not active_lower_y_m <= start[1] <= active_upper_y_m:
                continue
            lower_fraction, upper_fraction = 0.0, 1.0
        else:
            first = (active_lower_y_m - start[1]) / delta[1]
            second = (active_upper_y_m - start[1]) / delta[1]
            lower_fraction = max(0.0, min(first, second))
            upper_fraction = min(1.0, max(first, second))
            if lower_fraction > upper_fraction:
                continue

        clipped_start = start + lower_fraction * delta
        clipped_end = start + upper_fraction * delta
        clipped_delta = clipped_end - clipped_start
        clipped_length_m = float(np.linalg.norm(clipped_delta))
        subdivision_count = max(
            1,
            int(np.ceil(clipped_length_m / _DISPLAY_SEGMENT_LENGTH_M)),
        )
        fractions = np.linspace(
            lower_fraction,
            upper_fraction,
            subdivision_count + 1,
        )
        power_start = float(segment["power_start"])
        power_end = float(segment["power_end"])
        log_power_ratio = (
            np.log(power_end / power_start)
            if power_start > 0.0 and power_end > 0.0
            else None
        )
        for subdivision_index in range(subdivision_count):
            first_fraction = fractions[subdivision_index]
            second_fraction = fractions[subdivision_index + 1]
            midpoint_fraction = 0.5 * (first_fraction + second_fraction)
            subsegment = np.stack(
                (
                    start + first_fraction * delta,
                    start + second_fraction * delta,
                )
            )
            projected_xz_mm = 1.0e3 * subsegment[:, (0, 2)]
            clipped = LineString(projected_xz_mm).intersection(silicone_region)
            pending = [clipped]
            if log_power_ratio is None:
                midpoint_power = (1.0 - midpoint_fraction) * power_start + (
                    midpoint_fraction * power_end
                )
            else:
                midpoint_power = power_start * np.exp(
                    midpoint_fraction * log_power_ratio
                )
            while pending:
                geometry = pending.pop()
                if geometry.geom_type == "LineString" and not geometry.is_empty:
                    visible_segments.append(
                        np.asarray(geometry.coords, dtype=np.float64)
                    )
                    visible_power.append(midpoint_power)
                elif hasattr(geometry, "geoms"):
                    pending.extend(geometry.geoms)

    relative_power = np.clip(
        np.asarray(visible_power, dtype=np.float64) / reference_path_power,
        0.0,
        1.0,
    )
    display_power = np.sqrt(relative_power)
    optical_rgba = np.asarray(to_rgba(style.colors.optical), dtype=np.float64)
    glow_colors = np.repeat(optical_rgba[None, :], len(visible_segments), axis=0)
    glow_colors[:, 3] = 0.10 * display_power
    path_colors = glow_colors.copy()
    path_colors[:, 3] = 0.72 * display_power

    glow = LineCollection(
        visible_segments,
        colors=glow_colors,
        linewidths=2.5 * display_power,
        zorder=5,
    )
    axis.add_collection(glow)
    paths = LineCollection(
        visible_segments,
        colors=path_colors,
        linewidths=0.15 + 0.85 * display_power,
        zorder=6,
    )
    axis.add_collection(paths)
    return glow, paths


def _add_carrier(axis, fingertip: Fingertip, *, style) -> Polygon:
    section = np.asarray(fingertip.carrier.cross_section, dtype=np.float64).copy()
    stem_bottom_z_mm = float(section[:, 1].min())
    section[np.isclose(section[:, 1], stem_bottom_z_mm), 1] += LED_RECESS_DEPTH_MM
    patch = Polygon(
        section,
        closed=True,
        facecolor=style.colors.carrier,
        edgecolor=style.colors.carrier,
        linewidth=style.spine_width_pt,
        zorder=3,
    )
    axis.add_patch(patch)
    return patch


def _add_central_led(axis, fingertip: Fingertip, *, style) -> Rectangle:
    parameters = fingertip.parameters.led
    source_z_mm = 1.0e3 * fingertip.led_source_centers_m[2][2]
    package = Rectangle(
        (-0.5 * parameters.width_mm, source_z_mm),
        parameters.width_mm,
        parameters.height_mm,
        facecolor=style.colors.optical,
        edgecolor=style.colors.optical,
        linewidth=style.spine_width_pt,
        zorder=8,
    )
    axis.add_patch(package)
    axis.plot(
        (
            -0.5 * parameters.emitting_window_x_mm,
            0.5 * parameters.emitting_window_x_mm,
        ),
        (source_z_mm, source_z_mm),
        color=style.colors.optical,
        linewidth=2.0 * style.line_width_pt,
        solid_capstyle="round",
        zorder=9,
    )
    return package


def plot_fingertip_model_3d(
    axis,
    fingertip_mesh,
    *,
    carrier_color: str = "#B9BEC4",
    pad_edge_color: str | None = None,
    show_axes: bool = True,
    zoom: float = 1.0,
    style=DEFAULT_STYLE,
) -> None:
    """Plot the complete fingertip mesh on an existing 3D axis."""
    fingertip = fingertip_mesh.fingertip
    if pad_edge_color is None:
        pad_edge_color = style.colors.optical
    silicone_vertices_mm = 1.0e3 * np.asarray(
        fingertip_mesh.silicone.vertices,
        dtype=np.float64,
    )
    silicone_triangles = np.asarray(
        fingertip_mesh.silicone.surface_tri_indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    carrier_vertices_mm = 1.0e3 * np.asarray(
        fingertip_mesh.carrier.vertices,
        dtype=np.float64,
    )
    carrier_triangles = np.asarray(
        fingertip_mesh.carrier.indices,
        dtype=np.int32,
    ).reshape(-1, 3)

    silicone = Poly3DCollection(
        silicone_vertices_mm[silicone_triangles],
        facecolor=style.colors.silicone,
        edgecolor="none",
        alpha=0.20,
        rasterized=True,
        zorder=1,
    )
    axis.add_collection3d(silicone)
    carrier = Poly3DCollection(
        carrier_vertices_mm[carrier_triangles],
        facecolor=carrier_color,
        edgecolor="none",
        alpha=0.92,
        rasterized=True,
        zorder=2,
    )
    axis.add_collection3d(carrier)

    carrier_faces = carrier_vertices_mm[carrier_triangles]
    carrier_normals = np.cross(
        carrier_faces[:, 1] - carrier_faces[:, 0],
        carrier_faces[:, 2] - carrier_faces[:, 0],
    )
    carrier_normals /= np.linalg.norm(carrier_normals, axis=1)[:, None]
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, triangle in enumerate(carrier_triangles):
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            edge_faces.setdefault(edge, []).append(face_index)
    crease_cosine = np.cos(np.deg2rad(5.0))
    carrier_crease_edges = tuple(
        carrier_vertices_mm[np.asarray(edge)]
        for edge, face_indices in edge_faces.items()
        if len(face_indices) != 2
        or abs(
            float(
                np.dot(
                    carrier_normals[face_indices[0]],
                    carrier_normals[face_indices[1]],
                )
            )
        )
        < crease_cosine
    )
    axis.add_collection3d(
        Line3DCollection(
            carrier_crease_edges,
            colors="#41464D",
            linewidths=0.45,
            zorder=3,
        )
    )

    led_centers_mm = 1.0e3 * np.asarray(
        fingertip.led_source_centers_m,
        dtype=np.float64,
    )
    led = fingertip.parameters.led
    half_led_x_mm = 0.5 * led.emitting_window_x_mm
    half_led_y_mm = 0.5 * led.emitting_window_y_mm
    led_windows = tuple(
        np.asarray(
            (
                (-half_led_x_mm, center[1] - half_led_y_mm, center[2]),
                (half_led_x_mm, center[1] - half_led_y_mm, center[2]),
                (half_led_x_mm, center[1] + half_led_y_mm, center[2]),
                (-half_led_x_mm, center[1] + half_led_y_mm, center[2]),
            ),
            dtype=np.float64,
        )
        for center in led_centers_mm
    )
    axis.add_collection3d(
        Poly3DCollection(
            led_windows,
            facecolor=style.colors.optical,
            edgecolor="#006B4F",
            linewidth=0.7,
            alpha=1.0,
            zorder=5,
        )
    )

    silicone_geometry = fingertip.silicone
    ellipse_angle = np.linspace(np.pi, 0.0, 121)
    ellipse_x_mm = silicone_geometry.ellipse_radius_x_mm * np.cos(ellipse_angle)
    ellipse_z_mm = (
        silicone_geometry.ellipse_center_z_mm
        - silicone_geometry.ellipse_radius_z_mm * np.sin(ellipse_angle)
    )
    pad_outline_x_mm = np.concatenate(
        (
            (-silicone_geometry.half_width_mm,),
            ellipse_x_mm,
            (silicone_geometry.half_width_mm, -silicone_geometry.half_width_mm),
        )
    )
    pad_outline_z_mm = np.concatenate(
        (
            (silicone_geometry.bond_top_z_mm,),
            ellipse_z_mm,
            (silicone_geometry.bond_top_z_mm, silicone_geometry.bond_top_z_mm),
        )
    )
    for y_mm in TOTAL_Y_BOUNDS_MM:
        axis.plot(
            pad_outline_x_mm,
            np.full_like(pad_outline_x_mm, y_mm),
            pad_outline_z_mm,
            color=pad_edge_color,
            linewidth=1.15,
            zorder=4,
        )
    tip_z_mm = (
        silicone_geometry.ellipse_center_z_mm
        - silicone_geometry.ellipse_radius_z_mm
    )
    for x_mm, z_mm in (
        (-silicone_geometry.half_width_mm, silicone_geometry.bond_top_z_mm),
        (silicone_geometry.half_width_mm, silicone_geometry.bond_top_z_mm),
        (-silicone_geometry.half_width_mm, silicone_geometry.ellipse_center_z_mm),
        (silicone_geometry.half_width_mm, silicone_geometry.ellipse_center_z_mm),
        (0.0, tip_z_mm),
    ):
        axis.plot(
            (x_mm, x_mm),
            TOTAL_Y_BOUNDS_MM,
            (z_mm, z_mm),
            color=pad_edge_color,
            linewidth=0.85,
            zorder=4,
        )

    axis.computed_zorder = False
    axis.set_xlim(-16.0, 16.0)
    axis.set_ylim(TOTAL_Y_BOUNDS_MM[0] - 2.0, TOTAL_Y_BOUNDS_MM[1] + 2.0)
    axis.set_zlim(
        silicone_vertices_mm[:, 2].min() - 1.0,
        carrier_vertices_mm[:, 2].max() + 1.0,
    )
    axis.set_box_aspect((32.0, 64.0, 26.0), zoom=zoom)
    axis.view_init(elev=22.0, azim=-38.0)
    if show_axes:
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_zlabel("")
        axis.set_xticks((-10.0, 0.0, 10.0))
        axis.set_yticks((-20.0, 0.0, 20.0))
        axis.set_zticks((-10.0, 0.0, 10.0))
        axis.tick_params(
            which="major",
            width=style.tick_width_pt,
            length=style.tick_length_pt,
            labelsize=style.tick_font_size_pt,
            pad=-1.0,
        )
        axis.grid(False)
        for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
            coordinate_axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            coordinate_axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    else:
        axis.set_axis_off()


def plot_fingertip_mesh_state(
    axis,
    vertices_m: np.ndarray,
    tetrahedra: np.ndarray,
    fingertip: Fingertip,
    path_segments: np.ndarray,
    emitted_power: float,
    escaped_power: float,
    *,
    title: str,
    sphere_diameter_mm: float | None = None,
    indentation_mm: float | None = None,
    contact_positions_W_m: np.ndarray | None = None,
    show_axes: bool = True,
    show_legend: bool = True,
    style=DEFAULT_STYLE,
):
    mesh_polygons = _tet_slice_polygons_xz(vertices_m, tetrahedra)
    mesh = PolyCollection(
        mesh_polygons,
        facecolor=style.colors.silicone,
        edgecolor="#D8D8D8",
        linewidth=0.08,
        zorder=1,
    )
    axis.add_collection(mesh)
    carrier = _add_carrier(axis, fingertip, style=style)
    reference_path_power = (
        fingertip.parameters.led.normalized_power
        / _VISUALIZATION_SAMPLE_SIDE_COUNT**2
    )
    _add_optical_paths(
        axis,
        path_segments,
        mesh_polygons,
        reference_path_power=reference_path_power,
        style=style,
    )
    led = _add_central_led(axis, fingertip, style=style)

    legend_handles: list[object] = [
        Patch(
            facecolor=style.colors.silicone,
            edgecolor="#D8D8D8",
            label="Silicone tet mesh",
        ),
        Patch(
            facecolor=style.colors.carrier,
            edgecolor=style.colors.carrier,
            label="Carrier",
        ),
        Patch(
            facecolor=style.colors.silicone,
            edgecolor=style.colors.optical,
            label="Finite-area LED section",
        ),
        Line2D(
            (),
            (),
            color=style.colors.optical,
            linewidth=style.line_width_pt,
            label=r"OptiX light (display $\propto\sqrt{P}$)",
        ),
    ]
    if sphere_diameter_mm is not None and indentation_mm is not None:
        radius_mm = 0.5 * sphere_diameter_mm
        center_z_mm = 1.0e3 * fingertip.tip_z_m - radius_mm + indentation_mm
        angle = np.linspace(0.0, 2.0 * np.pi, 401)
        axis.plot(
            radius_mm * np.cos(angle),
            center_z_mm + radius_mm * np.sin(angle),
            color=style.colors.neutral,
            linestyle="--",
            linewidth=style.line_width_pt,
            zorder=2,
        )
        legend_handles.append(
            Line2D(
                (),
                (),
                color=style.colors.neutral,
                linestyle="--",
                label="Spherical indenter",
            )
        )

    if contact_positions_W_m is not None and len(contact_positions_W_m):
        contact_positions_mm = 1.0e3 * np.asarray(
            contact_positions_W_m,
            dtype=np.float64,
        )
        axis.scatter(
            contact_positions_mm[:, 0],
            contact_positions_mm[:, 2],
            s=5.0,
            color=style.colors.mechanical,
            edgecolors="none",
            alpha=0.85,
            zorder=10,
        )

    axis.set_xlim(-19.0, 19.0)
    axis.set_ylim(-18.0, 12.0)
    axis.set_aspect("equal", adjustable="box")
    if show_axes:
        axis.set_xlabel(r"Transverse coordinate, $X$ [mm]")
        axis.set_ylabel(r"Vertical coordinate, $Z$ [mm]")
        axis.set_title(title)
        axis.xaxis.set_major_locator(MultipleLocator(5.0))
        axis.yaxis.set_major_locator(MultipleLocator(5.0))
        axis.xaxis.set_minor_locator(MultipleLocator(1.0))
        axis.yaxis.set_minor_locator(MultipleLocator(1.0))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(
            which="major",
            width=1.2 * style.tick_width_pt,
            length=1.6 * style.tick_length_pt,
            direction="out",
        )
        axis.tick_params(
            which="minor",
            width=0.5 * style.tick_width_pt,
            length=0.5 * style.tick_length_pt,
            direction="out",
        )
    else:
        axis.set_axis_off()
    if show_legend:
        axis.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor=style.colors.grid,
        )
    return mesh, carrier, led


def _shared_legend_handles(style=DEFAULT_STYLE) -> tuple[object, ...]:
    return (
        Patch(
            facecolor=style.colors.silicone,
            edgecolor=style.colors.grid,
            label="Pad",
        ),
        Patch(
            facecolor=style.colors.carrier,
            edgecolor=style.colors.carrier,
            label="Carrier",
        ),
        Patch(
            facecolor=style.colors.silicone,
            edgecolor=style.colors.optical,
            label="LED",
        ),
        Line2D(
            (),
            (),
            color=style.colors.mechanical,
            linestyle="--",
            linewidth=style.line_width_pt,
            label="Bonding surface",
        ),
        Line2D(
            (),
            (),
            color=style.colors.optical,
            linewidth=style.line_width_pt,
            label=r"OptiX path ($\alpha\propto\sqrt{P}$)",
        ),
        Line2D(
            (),
            (),
            color=style.colors.neutral,
            linestyle="--",
            linewidth=style.line_width_pt,
            label="Spherical indenter",
        ),
    )


def main() -> None:
    if not _INPUT.is_file():
        raise FileNotFoundError(f"missing frozen production state: {_INPUT}")

    fingertip = Fingertip()
    fingertip_mesh = make_fingertip_mesh(fingertip, element_size_mm=1.0)
    with np.load(_INPUT, allow_pickle=False) as data:
        reference_vertices_m = np.asarray(data["reference_vertices_m"], dtype=np.float64)
        tetrahedra = np.asarray(data["tet_indices"], dtype=np.int32)
        scenario_matches = np.flatnonzero(data["scenario_names"] == _LOADED_SCENARIO)
        force_matches = np.flatnonzero(
            np.isclose(data["force_targets_n"], _LOADED_FORCE_N)
        )
        if len(scenario_matches) != 1 or len(force_matches) != 1:
            raise RuntimeError("requested loaded production checkpoint is not unique")
        scenario_index = int(scenario_matches[0])
        force_index = int(force_matches[0])
        loaded_vertices_m = np.asarray(
            data["silicone_vertices_m"][scenario_index, force_index],
            dtype=np.float64,
        )
        actual_force_n = float(data["actual_forces_n"][scenario_index, force_index])
        indentation_mm = 1.0e3 * float(
            data["indentations_m"][scenario_index, force_index]
        )
        sphere_diameter_mm = float(data["sphere_diameters_mm"][scenario_index])

    scene = OptixScene(fingertip_mesh)
    leds = _make_leds(fingertip)
    emissions = _emissions(scene, leds)
    samples = _optical_samples(len(emissions[0]))

    scene.update_silicone(reference_vertices_m)
    unloaded_paths = _trace_state(
        scene,
        fingertip,
        leds,
        emissions,
        samples,
    )
    scene.update_silicone(loaded_vertices_m)
    loaded_paths = _trace_state(
        scene,
        fingertip,
        leds,
        emissions,
        samples,
    )
    with publication_context():
        panel_width_ratios = (1.05, 1.70, 1.0, 1.0)
        figure, grid = create_gridspec(
            1,
            4,
            width="double",
            panel_aspect=1.30,
            width_ratios=panel_width_ratios,
        )
        figure.set_layout_engine(None)
        axes = np.asarray(
            (
                figure.add_subplot(grid[0, 0], projection="3d"),
                figure.add_subplot(grid[0, 1]),
                figure.add_subplot(grid[0, 2]),
                figure.add_subplot(grid[0, 3]),
            ),
            dtype=object,
        )
        plot_fingertip_model_3d(axes[0], fingertip_mesh)
        axes[0].set_title("(a) 3D overview", pad=2.0)
        plot_fingertip_parameterization(
            axes[1],
            fingertip,
            show_legend=False,
            style=replace(
                DEFAULT_STYLE,
                axis_label_font_size_pt=10.0,
                line_width_pt=1.35,
            ),
        )
        axes[1].set_xlabel(r"$X$ [mm]")
        axes[1].set_ylabel(r"$Z$ [mm]")
        axes[1].xaxis.set_major_locator(MultipleLocator(10.0))
        axes[1].set_title("(b) 2D parameterization", pad=2.0)
        plot_fingertip_mesh_state(
            axes[2],
            reference_vertices_m,
            tetrahedra,
            fingertip,
            *unloaded_paths,
            title="(c) Unloaded",
            show_legend=False,
        )
        plot_fingertip_mesh_state(
            axes[3],
            loaded_vertices_m,
            tetrahedra,
            fingertip,
            *loaded_paths,
            title=f"(d) Loaded, {actual_force_n:.2f} N",
            sphere_diameter_mm=sphere_diameter_mm,
            indentation_mm=indentation_mm,
            show_legend=False,
        )

        for axis in axes[2:]:
            axis.set_xlim(-16.0, 16.0)
            axis.set_ylim(-18.0, 14.5)
            axis.set_xlabel(r"$X$ [mm]")
            axis.set_anchor("C")
            axis.set_title(axis.get_title(), pad=2.0)
        axes[2].set_ylabel(r"$Z$ [mm]")
        axes[3].set_ylabel("")
        axes[3].tick_params(labelleft=False)

        panel_left = 0.03
        panel_right = 0.99
        panel_gaps = np.asarray((0.075, 0.06, 0.025), dtype=np.float64)
        panel_bottom = 0.18
        panel_top = 0.80
        available_panel_width = panel_right - panel_left - panel_gaps.sum()
        panel_widths = available_panel_width * np.asarray(
            panel_width_ratios,
            dtype=np.float64,
        ) / sum(panel_width_ratios)
        panel_height = panel_top - panel_bottom
        next_panel_left = panel_left
        for panel_index, (axis, panel_width) in enumerate(
            zip(axes, panel_widths, strict=True)
        ):
            axis.set_position(
                (
                    next_panel_left,
                    panel_bottom,
                    panel_width,
                    panel_height,
                )
            )
            next_panel_left += panel_width
            if panel_index < len(panel_gaps):
                next_panel_left += panel_gaps[panel_index]

        figure_width_in, figure_height_in = figure.get_size_inches()
        panel_box_aspects = (
            panel_widths * figure_width_in
        ) / (panel_height * figure_height_in)
        parameter_x_limits = axes[1].get_xlim()
        parameter_y_limits = axes[1].get_ylim()
        parameter_y_center = 0.5 * sum(parameter_y_limits)
        parameter_y_range = (
            parameter_x_limits[1] - parameter_x_limits[0]
        ) / panel_box_aspects[1]
        axes[1].set_ylim(
            parameter_y_center - 0.5 * parameter_y_range,
            parameter_y_center + 0.5 * parameter_y_range,
        )
        axes[1].set_aspect("auto")
        optical_y_range = 14.5 - (-18.0)
        for axis, panel_box_aspect in zip(
            axes[2:],
            panel_box_aspects[2:],
            strict=True,
        ):
            optical_x_half_range = 0.5 * optical_y_range * panel_box_aspect
            axis.set_xlim(-optical_x_half_range, optical_x_half_range)
            axis.set_aspect("auto")

        figure.legend(
            handles=_shared_legend_handles(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=6,
            columnspacing=0.9,
            handlelength=1.5,
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor=DEFAULT_STYLE.colors.grid,
        )

    paths = save_figure(
        figure,
        _OUTPUT_DIRECTORY / "fingertip_figure_2_overview",
        formats=("png",),
    )
    plt.close(figure)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
