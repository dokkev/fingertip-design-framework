"""Export the standalone B-D image assets used to assemble Figure 2."""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import newton  # noqa: E402
import newton.viewer  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

from figure_2_optomechanical_pipeline import _load_frozen_state  # noqa: E402
from fingertip_mesh_states import (  # noqa: E402
    _emissions,
    _make_leds,
    _optical_samples,
    _trace_state,
    plot_fingertip_mesh_state,
    plot_fingertip_model_3d,
)
from lumo.fingertip import Fingertip  # noqa: E402
from lumo.mesh import make_fingertip_mesh  # noqa: E402
from lumo.newton import Indenter, build_fingertip_newton_model  # noqa: E402
from lumo.ray_tracing import OptixScene  # noqa: E402
from lumo.visualization import DEFAULT_STYLE, publication_context  # noqa: E402


_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIRECTORY = _ROOT / "output" / "figures" / "figure2"
_FROZEN_INPUT = (
    _ROOT
    / "output"
    / "validation"
    / "fingertip_production_objective_freeze"
    / "nominal_fingertip_objectives.npz"
)
_CONTACT_PATCH_SCENARIO = "sphere_15mm_y+0mm"
_CONTACT_PATCH_FORCES_N = (1.0, 2.0, 5.0, 10.0)
_STRESS_COLOR_MAX_KPA = 100.0
_STRESS_COLOR_BINS = 12
_STRESS_COLOR_GAMMA = 0.30
_CONTACT_COLOR = wp.vec3(0.95, 0.48, 0.02)
_LED_COLOR = wp.vec3(0.0, 0.70, 0.38)


def _save_matplotlib_panel(figure, filename: str) -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        _OUTPUT_DIRECTORY / filename,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.01,
        facecolor="white",
    )
    plt.close(figure)


def _export_model_panel(fingertip_mesh) -> None:
    with publication_context():
        figure = plt.figure(figsize=(3.2, 3.2))
        axis = figure.add_subplot(111, projection="3d")
        plot_fingertip_model_3d(
            axis,
            fingertip_mesh,
            carrier_color=DEFAULT_STYLE.colors.carrier,
            pad_edge_color=DEFAULT_STYLE.colors.neutral,
            show_axes=False,
            zoom=0.90,
        )
        axis.set_position((0.04, 0.04, 0.92, 0.92))
    _save_matplotlib_panel(figure, "b_3d_model.png")


def _emphasize_cross_section_mesh(axis, fingertip_mesh) -> None:
    silicone_collections = [
        collection
        for collection in axis.collections
        if isinstance(collection, PolyCollection)
    ]
    if not silicone_collections:
        raise RuntimeError("silicone cross-section mesh is missing")
    silicone_collections[0].set_edgecolor("#B8C1BC")
    silicone_collections[0].set_linewidth(0.18)

    vertices_xz_mm = 1.0e3 * np.asarray(
        fingertip_mesh.carrier.vertices,
        dtype=np.float64,
    )[:, (0, 2)]
    triangles = np.asarray(
        fingertip_mesh.carrier.indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    unique_segments: dict[tuple[float, ...], np.ndarray] = {}
    for triangle in triangles:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            segment = vertices_xz_mm[triangle[[first, second]]]
            if np.linalg.norm(segment[1] - segment[0]) <= 1.0e-9:
                continue
            ordered = segment[np.lexsort((segment[:, 1], segment[:, 0]))]
            key = tuple(np.round(ordered, decimals=6).ravel())
            unique_segments[key] = ordered
    axis.add_collection(
        LineCollection(
            tuple(unique_segments.values()),
            colors="#383D43",
            linewidths=0.28,
            alpha=0.62,
            zorder=4,
        )
    )


def _export_optix_panel(fingertip_mesh, state: dict[str, object]) -> None:
    fingertip = fingertip_mesh.fingertip
    scene = OptixScene(fingertip_mesh)
    leds = _make_leds(fingertip)
    emissions = _emissions(scene, leds)
    samples = _optical_samples(len(emissions[0]))

    scene.update_silicone(state["reference_vertices_m"])
    unloaded_paths = _trace_state(scene, fingertip, leds, emissions, samples)
    scene.update_silicone(state["loaded_vertices_m"])
    loaded_paths = _trace_state(scene, fingertip, leds, emissions, samples)

    panels = (
        (
            "d_unloaded_optix.svg",
            state["reference_vertices_m"],
            unloaded_paths,
            {},
        ),
        (
            "d_loaded_optix.svg",
            state["loaded_vertices_m"],
            loaded_paths,
            {
                "sphere_diameter_mm": state["sphere_diameter_mm"],
                "indentation_mm": state["indentation_mm"],
            },
        ),
    )
    for filename, vertices, paths, loaded_arguments in panels:
        with publication_context():
            figure, axis = plt.subplots(figsize=(3.2, 3.2))
            plot_fingertip_mesh_state(
                axis,
                vertices,
                state["tetrahedra"],
                fingertip,
                *paths,
                title="",
                show_axes=False,
                show_legend=False,
                **loaded_arguments,
            )
            _emphasize_cross_section_mesh(axis, fingertip_mesh)
            axis.set_xlim(-16.0, 16.0)
            axis.set_ylim(-19.0, 14.0)
            axis.set_position((0.0, 0.0, 1.0, 1.0))
        _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        for output_name in (filename, filename.replace(".svg", ".png")):
            figure.savefig(
                _OUTPUT_DIRECTORY / output_name,
                dpi=600,
                bbox_inches="tight",
                pad_inches=0.01,
                facecolor="white",
            )
        plt.close(figure)


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) < 3:
        raise ValueError("contact patch needs at least three distinct points")

    def cross(origin, first, second) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _export_contact_patch_panel() -> None:
    with np.load(_FROZEN_INPUT, allow_pickle=False) as data:
        scenario_matches = np.flatnonzero(
            data["scenario_names"] == _CONTACT_PATCH_SCENARIO
        )
        if len(scenario_matches) != 1:
            raise RuntimeError("contact-patch scenario is not unique")
        if not np.array_equal(
            np.asarray(data["force_targets_n"], dtype=np.float64),
            np.asarray(_CONTACT_PATCH_FORCES_N, dtype=np.float64),
        ):
            raise RuntimeError("frozen force targets do not match Figure 2")

        scenario_index = int(scenario_matches[0])
        contours = []
        for force_index, force_n in enumerate(_CONTACT_PATCH_FORCES_N):
            offset, count = np.asarray(
                data["contact_record_offsets"][scenario_index, force_index],
                dtype=np.int64,
            )
            positions_xy_mm = 1.0e3 * np.asarray(
                data["contact_positions_W_m"][offset : offset + count, :2],
                dtype=np.float64,
            )
            contours.append((force_n, _convex_hull_2d(positions_xy_mm)))

    colors = {
        1.0: "#FEE8C8",
        2.0: "#FDBB84",
        5.0: "#FC8D59",
        10.0: "#D7301F",
    }
    with publication_context():
        figure, axis = plt.subplots(figsize=(3.2, 3.2))
        handles = {}
        for force_n, contour in reversed(contours):
            closed = np.vstack((contour, contour[0]))
            handles[force_n] = axis.fill(
                closed[:, 0],
                closed[:, 1],
                facecolor=colors[force_n],
                edgecolor=colors[force_n],
                linewidth=1.25,
                alpha=0.58,
                label=f"{force_n:g} N",
                zorder=10 - int(force_n),
            )[0]
        axis.axhline(0.0, color="#D5D5D5", linewidth=0.55, zorder=0)
        axis.axvline(0.0, color="#D5D5D5", linewidth=0.55, zorder=0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-5.4, 5.4)
        axis.set_ylim(-5.4, 5.4)
        axis.set_xlabel(r"$X$ (mm)", labelpad=1.0)
        axis.set_ylabel(r"$Y$ (mm)", labelpad=1.0)
        axis.set_xticks((-4.0, -2.0, 0.0, 2.0, 4.0))
        axis.set_yticks((-4.0, -2.0, 0.0, 2.0, 4.0))
        axis.legend(
            [handles[force] for force in _CONTACT_PATCH_FORCES_N],
            [f"{force:g} N" for force in _CONTACT_PATCH_FORCES_N],
            loc="upper left",
            frameon=False,
            handlelength=1.15,
            borderaxespad=0.2,
        )
        axis.spines[["top", "right"]].set_visible(False)
        figure.subplots_adjust(left=0.16, right=0.99, bottom=0.14, top=0.99)
    _save_matplotlib_panel(figure, "e_contact_patch.png")


def _set_shape_colors(model, indenter: Indenter, carrier_body: int) -> None:
    if model.shape_color is None or model.shape_body is None:
        return
    colors = model.shape_color.numpy()
    shape_bodies = model.shape_body.numpy()
    colors[shape_bodies == indenter.body_index] = (0.44, 0.46, 0.49)
    colors[shape_bodies == carrier_body] = (0.25, 0.27, 0.30)
    model.shape_color.assign(colors)


def _surface_von_mises_stress_pa(
    fingertip,
    reference_vertices_m: np.ndarray,
    deformed_vertices_m: np.ndarray,
    tetrahedra: np.ndarray,
    surface_triangles: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(reference_vertices_m, dtype=np.float64)
    deformed = np.asarray(deformed_vertices_m, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32).reshape(-1, 4)
    surface_triangles = np.asarray(surface_triangles, dtype=np.int32).reshape(-1, 3)

    reference_edges = np.stack(
        (
            reference[tetrahedra[:, 1]] - reference[tetrahedra[:, 0]],
            reference[tetrahedra[:, 2]] - reference[tetrahedra[:, 0]],
            reference[tetrahedra[:, 3]] - reference[tetrahedra[:, 0]],
        ),
        axis=2,
    )
    deformed_edges = np.stack(
        (
            deformed[tetrahedra[:, 1]] - deformed[tetrahedra[:, 0]],
            deformed[tetrahedra[:, 2]] - deformed[tetrahedra[:, 0]],
            deformed[tetrahedra[:, 3]] - deformed[tetrahedra[:, 0]],
        ),
        axis=2,
    )
    deformation_gradient = deformed_edges @ np.linalg.inv(reference_edges)
    determinant = np.linalg.det(deformation_gradient)
    if np.any(determinant <= 0.0):
        raise RuntimeError("cannot render stress for an inverted tetrahedron")

    mechanics = fingertip.parameters.mechanics
    mu = float(mechanics.shear_modulus_pa)
    lambda_nh = float(mechanics.lame_lambda_pa) + mu
    alpha = 1.0 + mu / lambda_nh
    cofactor = determinant[:, None, None] * np.swapaxes(
        np.linalg.inv(deformation_gradient),
        1,
        2,
    )
    first_piola = mu * deformation_gradient + (
        lambda_nh * (determinant - alpha)
    )[:, None, None] * cofactor
    cauchy = (
        first_piola @ np.swapaxes(deformation_gradient, 1, 2)
    ) / determinant[:, None, None]
    deviatoric = cauchy - (
        np.trace(cauchy, axis1=1, axis2=2)[:, None, None]
        * np.eye(3, dtype=np.float64)[None, :, :]
        / 3.0
    )
    tet_stress = np.sqrt(1.5 * np.sum(deviatoric * deviatoric, axis=(1, 2)))

    vertex_stress = np.zeros(len(reference), dtype=np.float64)
    incident_tet_count = np.zeros(len(reference), dtype=np.int32)
    for local_vertex in range(4):
        np.add.at(vertex_stress, tetrahedra[:, local_vertex], tet_stress)
        np.add.at(incident_tet_count, tetrahedra[:, local_vertex], 1)
    if np.any(incident_tet_count[surface_triangles] == 0):
        raise RuntimeError("surface vertex has no incident tetrahedron")
    vertex_stress /= np.maximum(incident_tet_count, 1)
    return vertex_stress[surface_triangles].mean(axis=1)


def _crop_viewer_background(frame: np.ndarray, padding_px: int = 40) -> np.ndarray:
    row_background = 0.5 * (
        frame[:, :1].astype(np.float32) + frame[:, -1:].astype(np.float32)
    )
    foreground = np.max(
        np.abs(frame.astype(np.float32) - row_background),
        axis=2,
    ) > 10.0
    rows, columns = np.nonzero(foreground)
    if len(rows) == 0:
        return frame
    row_start = max(0, int(rows.min()) - padding_px)
    row_stop = min(frame.shape[0], int(rows.max()) + padding_px + 1)
    column_start = max(0, int(columns.min()) - padding_px)
    column_stop = min(frame.shape[1], int(columns.max()) + padding_px + 1)
    return frame[row_start:row_stop, column_start:column_stop]


def _export_newton_panel(
    fingertip_mesh,
    state: dict[str, object],
) -> None:
    fingertip = fingertip_mesh.fingertip
    sphere_radius_m = 0.5e-3 * float(state["sphere_diameter_mm"])
    sphere_center_z_m = (
        fingertip.tip_z_m
        - sphere_radius_m
        + 1.0e-3 * float(state["indentation_mm"])
    )
    sphere_pose = wp.transform(
        wp.vec3(0.0, 0.0, sphere_center_z_m),
        wp.quat_identity(),
    )
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, 0.0))
    sphere_resource = files("lumo").joinpath(
        "assets",
        "objects",
        "urdf",
        "sphere_15mm.urdf",
    )
    with as_file(sphere_resource) as urdf_path:
        indenter = Indenter.add_urdf(builder, urdf_path, tf=sphere_pose)
    fingertip_model = build_fingertip_newton_model(
        fingertip_mesh,
        builder=builder,
    )
    model = fingertip_model.model
    newton_state = model.state()
    loaded_vertices = wp.array(
        np.asarray(state["loaded_vertices_m"], dtype=np.float32),
        dtype=wp.vec3,
        device=model.device,
    )
    wp.copy(
        newton_state.particle_q,
        loaded_vertices,
        dest_offset=fingertip_model.silicone_particle_start,
        count=fingertip_model.silicone_particle_count,
    )
    _set_shape_colors(model, indenter, fingertip_model.carrier_body)

    surface_triangles = np.asarray(
        fingertip_mesh.silicone.surface_tri_indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    triangle_stress_pa = _surface_von_mises_stress_pa(
        fingertip,
        state["reference_vertices_m"],
        state["loaded_vertices_m"],
        state["tetrahedra"],
        surface_triangles,
    )
    stress_norm = matplotlib.colors.PowerNorm(
        gamma=_STRESS_COLOR_GAMMA,
        vmin=0.0,
        vmax=_STRESS_COLOR_MAX_KPA,
        clip=True,
    )
    stress_normalized = stress_norm(triangle_stress_pa * 1.0e-3)
    stress_bin_ids = np.minimum(
        (stress_normalized * _STRESS_COLOR_BINS).astype(np.int32),
        _STRESS_COLOR_BINS - 1,
    )
    stress_colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "silicone_stress",
        (
            (0.00, (0.95, 0.94, 0.88)),
            (0.20, (1.00, 0.91, 0.55)),
            (0.38, (0.99, 0.72, 0.27)),
            (0.55, (0.96, 0.40, 0.12)),
            (0.72, (0.82, 0.10, 0.08)),
            (1.00, (0.48, 0.00, 0.06)),
        ),
    )
    stress_meshes = []
    for bin_index in range(_STRESS_COLOR_BINS):
        triangles = surface_triangles[stress_bin_ids == bin_index]
        if len(triangles) == 0:
            continue
        indices = wp.array(
            triangles.reshape(-1) + fingertip_model.silicone_particle_start,
            dtype=wp.int32,
            device=model.device,
        )
        color = tuple(
            float(channel)
            for channel in stress_colormap(
                (bin_index + 0.5) / _STRESS_COLOR_BINS
            )[:3]
        )
        stress_meshes.append((bin_index, indices, color))
    contact_positions = wp.array(
        np.asarray(state["contact_positions_m"], dtype=np.float32),
        dtype=wp.vec3,
        device=model.device,
    )
    contact_colors = wp.full(
        len(contact_positions),
        _CONTACT_COLOR,
        dtype=wp.vec3,
        device=model.device,
    )
    led_centers = wp.array(
        fingertip.led_source_centers_m,
        dtype=wp.vec3,
        device=model.device,
    )
    led_colors = wp.full(
        len(fingertip.led_source_centers_m),
        _LED_COLOR,
        dtype=wp.vec3,
        device=model.device,
    )

    viewer = newton.viewer.ViewerGL(
        width=1400,
        height=1400,
        vsync=False,
        headless=True,
    )
    try:
        viewer.set_model(model)
        viewer.show_triangles = False
        viewer.show_particles = False
        viewer.renderer.draw_fps = False
        viewer.renderer.draw_edges = False
        viewer.renderer.sky_upper = (1.0, 1.0, 1.0)
        viewer.renderer.sky_lower = (1.0, 1.0, 1.0)
        viewer.renderer.ambient_sky = (0.82, 0.82, 0.84)
        viewer.renderer.ambient_ground = (0.45, 0.45, 0.47)
        viewer.renderer._exposure = 1.25
        camera_position = wp.vec3(0.055, -0.120, -0.014)
        scene_center = wp.vec3(0.0, 0.0, -0.006)
        viewer.set_camera(camera_position, 0.0, 0.0)
        viewer.camera.look_at(scene_center)
        viewer.camera.fov = 38.0

        for _ in range(2):
            viewer.begin_frame(0.0)
            viewer.log_state(newton_state)
            for bin_index, indices, color in stress_meshes:
                viewer.log_mesh(
                    f"/figure2/silicone_stress_{bin_index:02d}",
                    newton_state.particle_q,
                    indices,
                    color=color,
                    roughness=0.82,
                    metallic=0.0,
                    backface_culling=False,
                )
            viewer.log_points(
                "/figure2/contact_patch",
                contact_positions,
                radii=9.0e-4,
                colors=contact_colors,
            )
            viewer.log_points(
                "/figure2/leds",
                led_centers,
                radii=7.0e-4,
                colors=led_colors,
            )
            viewer.end_frame()
        frame = viewer.get_frame(render_ui=False).numpy()
    finally:
        viewer.close()

    with publication_context():
        figure = plt.figure(figsize=(4.0, 3.25))
        image_axis = figure.add_axes((0.0, 0.13, 1.0, 0.87))
        image_axis.imshow(_crop_viewer_background(frame))
        image_axis.set_axis_off()
        colorbar_axis = figure.add_axes((0.20, 0.055, 0.60, 0.035))
        scalar_map = matplotlib.cm.ScalarMappable(
            norm=stress_norm,
            cmap=stress_colormap,
        )
        colorbar = figure.colorbar(
            scalar_map,
            cax=colorbar_axis,
            orientation="horizontal",
            ticks=(0.0, 5.0, 10.0, 25.0, 50.0, 100.0),
        )
        colorbar.set_label("Elastic von Mises stress (kPa)", labelpad=1.0)
    _save_matplotlib_panel(figure, "c_newton_mechanics.png")


def main() -> None:
    state = _load_frozen_state()
    fingertip_mesh = make_fingertip_mesh(Fingertip(), element_size_mm=1.0)
    _export_model_panel(fingertip_mesh)
    _export_newton_panel(fingertip_mesh, state)
    _export_optix_panel(fingertip_mesh, state)
    _export_contact_patch_panel()
    for filename in (
        "b_3d_model.png",
        "c_newton_mechanics.png",
        "d_unloaded_optix.svg",
        "d_unloaded_optix.png",
        "d_loaded_optix.svg",
        "d_loaded_optix.png",
        "e_contact_patch.png",
    ):
        print(_OUTPUT_DIRECTORY / filename)


if __name__ == "__main__":
    main()
