"""Plot the current 3D one-LED sensing paths in the center X-Z section."""

from __future__ import annotations

import argparse
from importlib.resources import as_file, files
from pathlib import Path

import numpy as np
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import FingertipMesh, make_fingertip_mesh
from lumo.newton import Indenter
from lumo.optimization import sensing_descriptors
from lumo.ray_tracing import (
    LED,
    OptixScene,
    safe_secondary_origins,
    side_view_observation,
    trace_bounded_paths,
)
from lumo.simulation import DesignStudy, DesignTrial, LumoSimulation


_SILICONE_INSTANCE_ID = 1
_CARRIER_INSTANCE_ID = 2
_SILICONE_MASK = 0x01
_CARRIER_MASK = 0x02
_ALL_MASK = _SILICONE_MASK | _CARRIER_MASK

_SAMPLE_SIDE_COUNT = 64
_BOUNCE_CAP = 24
_RNG_SEED = 20260823
_CARRIER_ALBEDO = 0.7
_PLOTTED_PATH_COUNT = 150

_SPHERE_RADIUS_M = 7.5e-3
_INITIAL_CLEARANCE_M = 1.0e-3
_APPROACH_SPEED_M_S = 2.5e-2
_TARGET_FORCE_N = 20.0
_FORCE_TOLERANCE_N = 1.0
_SETTLE_DURATION_S = 5.0
_MAX_SIM_TIME_S = 20.0
_SIM_FREQUENCY_HZ = 500.0
_VBD_ITERATIONS = 10
_CONTACT_STIFFNESS_N_M = 3.0e4
_CONTACT_DAMPING_N_S_M = 0.28228017516945547


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize projected 3D paths before and after center load."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="save the figure instead of opening an interactive window",
    )
    return parser.parse_args()


def _make_led(fingertip: Fingertip, fingertip_mesh: FingertipMesh) -> LED:
    vertices = np.asarray(fingertip_mesh.silicone.vertices, dtype=np.float64)
    center_y_m = 0.5 * float(vertices[:, 1].min() + vertices[:, 1].max())
    stem_bottom_z_m = -1.0e-3 * fingertip.parameters.geometry.stem_height_mm
    return LED(
        position_W_m=np.array((0.0, center_y_m, stem_bottom_z_m)),
        normal_W=np.array((0.0, 0.0, -1.0)),
        parameters=fingertip.parameters.led,
    )


def _surface_section_xz(
    vertices_W_m: np.ndarray,
    triangles: np.ndarray,
    *,
    section_y_m: float,
) -> np.ndarray:
    """Intersect surface triangles with one Y plane for this plot only."""
    vertices = np.asarray(vertices_W_m, dtype=np.float64)
    triangle_indices = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    tolerance_m = 1.0e-9
    sections: list[np.ndarray] = []

    for indices in triangle_indices:
        triangle = vertices[indices]
        distance = triangle[:, 1] - section_y_m
        points: list[np.ndarray] = []

        for vertex, signed_distance in zip(triangle, distance, strict=True):
            if abs(signed_distance) <= tolerance_m:
                points.append(vertex)

        for first, second in ((0, 1), (1, 2), (2, 0)):
            first_distance = distance[first]
            second_distance = distance[second]
            if (first_distance < -tolerance_m < second_distance) or (
                second_distance < -tolerance_m < first_distance
            ):
                fraction = first_distance / (first_distance - second_distance)
                points.append(
                    triangle[first]
                    + fraction * (triangle[second] - triangle[first])
                )

        unique_points: list[np.ndarray] = []
        for point in points:
            if not any(
                np.linalg.norm(point - existing) <= tolerance_m
                for existing in unique_points
            ):
                unique_points.append(point)
        if len(unique_points) < 2:
            continue

        best_pair = (unique_points[0], unique_points[1])
        best_distance = -1.0
        for first in range(len(unique_points) - 1):
            for second in range(first + 1, len(unique_points)):
                distance_m = float(
                    np.linalg.norm(unique_points[first] - unique_points[second])
                )
                if distance_m > best_distance:
                    best_distance = distance_m
                    best_pair = (unique_points[first], unique_points[second])
        sections.append(np.asarray(best_pair)[:, (0, 2)])

    if not sections:
        raise RuntimeError("silicone surface does not intersect the LED Y plane")
    section = np.stack(sections)
    if not np.all(np.isfinite(section)):
        raise RuntimeError("silicone section contains non-finite coordinates")
    return section


def _source_inside_silicone(
    scene: OptixScene,
    led: LED,
    emission: np.ndarray,
    *,
    state_label: str,
) -> bool:
    initial_hits = scene.trace_closest(
        emission["origin_W_m"],
        emission["direction_W"],
        mask=_ALL_MASK,
    )
    if not np.all(initial_hits["hit"]) or np.any(
        initial_hits["instance_id"] != _SILICONE_INSTANCE_ID
    ):
        raise AssertionError(f"{state_label} has an obstructed source ray")

    origin = emission["origin_W_m"][:1]
    direction = led.normal_W[None, :]
    silicone_hit = scene.trace_closest(origin, direction, mask=_SILICONE_MASK)[0]
    if not silicone_hit["hit"]:
        raise AssertionError(f"{state_label} invalidated the stem source path")
    normal_projection = float(np.dot(silicone_hit["normal_W"], led.normal_W))
    if abs(normal_projection) <= 1.0e-6:
        raise AssertionError(f"{state_label} has an ambiguous source interface")
    return normal_projection > 0.0


def _emit_from_stem_boundary(
    scene: OptixScene,
    led: LED,
    u1: np.ndarray,
    u2: np.ndarray,
) -> np.ndarray:
    """Emit from the OTK-safe pad side of the physical source point."""
    probe_distance_m = 0.5e-3 * led.parameters.height_mm
    probe_origin = (
        led.position_W_m - probe_distance_m * led.normal_W
    )[None, :]
    direction = led.normal_W[None, :]
    carrier_hit = scene.trace_closest(
        probe_origin,
        direction,
        mask=_CARRIER_MASK,
    )
    if not carrier_hit["hit"][0]:
        raise AssertionError("carrier probe did not find the stem boundary")
    hit_position = probe_origin[0] + carrier_hit["t"][0] * led.normal_W
    if not np.allclose(
        hit_position,
        led.position_W_m,
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise AssertionError("carrier probe found the wrong source boundary")

    safe_origin = safe_secondary_origins(carrier_hit, direction)[0]
    emission = led.emit(u1, u2)
    emission["origin_W_m"] = safe_origin
    return emission


def _trace_state(
    scene: OptixScene,
    fingertip: Fingertip,
    emission: np.ndarray,
    inside_silicone: bool,
    dielectric_branch_u: np.ndarray,
    carrier_u1: np.ndarray,
    carrier_u2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    optics = fingertip.parameters.optical
    result = trace_bounded_paths(
        scene,
        emission["origin_W_m"],
        emission["direction_W"],
        emission["power"],
        inside_silicone=inside_silicone,
        n_air=1.0,
        n_silicone=optics.refractive_index,
        extinction_coefficient_m_inv=optics.extinction_coefficient_m_inv,
        carrier_albedo=_CARRIER_ALBEDO,
        max_bounces=_BOUNCE_CAP,
        dielectric_branch_u=dielectric_branch_u,
        carrier_u1=carrier_u1,
        carrier_u2=carrier_u2,
        silicone_instance_id=_SILICONE_INSTANCE_ID,
        carrier_instance_id=_CARRIER_INSTANCE_ID,
        mask=_ALL_MASK,
        record_segments=True,
    )
    escaped = result.escaped_rays
    segments = result.segments
    if segments is None:
        raise AssertionError("requested path segments were not recorded")
    if not np.isclose(
        result.accounted_power,
        result.emitted_power,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("bounded transport failed energy closure")

    if np.any(escaped["bounce"] == 0):
        raise AssertionError("stem source escaped without a silicone hit")
    observation = side_view_observation(escaped, fingertip=fingertip)
    visible = escaped["direction_W"][:, 1] > 0.0
    if not np.isclose(
        float(observation.sum()),
        float(escaped["power"][visible].sum()),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("escape markers and quadrant observation disagree")
    for values in (
        segments["start_W_m"],
        segments["end_W_m"],
        segments["power"],
        escaped["origin_W_m"],
        escaped["power"],
        observation,
    ):
        if not np.all(np.isfinite(values)):
            raise AssertionError("optical plotting data contains NaN or Inf")
    return escaped, segments, observation


def _make_center_trial(fingertip: Fingertip, sphere_urdf_path: Path) -> DesignTrial:
    initial_center_z_m = (
        fingertip.tip_z_m
        - _INITIAL_CLEARANCE_M
        - _SPHERE_RADIUS_M
    )
    return DesignTrial(
        name="center_contact",
        urdf_path=sphere_urdf_path,
        initial_tf=wp.transform(
            wp.vec3(0.0, 0.0, initial_center_z_m),
            wp.quat_identity(),
        ),
        motion_direction_W=wp.vec3(0.0, 0.0, 1.0),
        approach_speed_m_s=_APPROACH_SPEED_M_S,
        target_force_n=_TARGET_FORCE_N,
        max_sim_time_s=_MAX_SIM_TIME_S,
    )


def _plot_panel(
    axis,
    *,
    title: str,
    section_xz_m: np.ndarray,
    carrier_xz_mm: np.ndarray,
    led: LED,
    segments: np.ndarray,
    escaped: np.ndarray,
    observation: np.ndarray,
    selected_ray_ids: np.ndarray,
    quadrant_center_xz_mm: np.ndarray,
    intensity_response: float,
    indenter_center_xz_mm: np.ndarray | None = None,
) -> None:
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    selected = segments[np.isin(segments["ray_id"], selected_ray_ids)]
    ray_lines_mm = np.stack(
        (selected["start_W_m"][:, (0, 2)], selected["end_W_m"][:, (0, 2)]),
        axis=1,
    ) * 1.0e3
    if len(selected):
        power_scale = max(float(selected["power"].max()), np.finfo(float).tiny)
        strength = np.sqrt(selected["power"] / power_scale)
        colors = np.zeros((len(selected), 4))
        colors[:, :3] = (1.0, 0.45, 0.0)
        colors[:, 3] = 0.04 + 0.30 * strength
        axis.add_collection(
            LineCollection(
                ray_lines_mm,
                colors=colors,
                linewidths=0.25 + 0.65 * strength,
                zorder=2,
                label="projected 3D paths",
            )
        )

    axis.add_collection(
        LineCollection(
            section_xz_m * 1.0e3,
            colors="#1565c0",
            linewidths=1.8,
            zorder=4,
            label="silicone section",
        )
    )
    axis.fill(
        carrier_xz_mm[:, 0],
        carrier_xz_mm[:, 1],
        color="0.55",
        alpha=0.22,
        label="carrier",
        zorder=1,
    )

    visible = escaped["direction_W"][:, 1] > 0.0
    visible_escaped = escaped[visible]
    if len(visible_escaped):
        escape_power_scale = max(
            float(visible_escaped["power"].max()),
            np.finfo(float).tiny,
        )
        marker_size = 4.0 + 18.0 * np.sqrt(
            visible_escaped["power"] / escape_power_scale
        )
        axis.scatter(
            1.0e3 * visible_escaped["origin_W_m"][:, 0],
            1.0e3 * visible_escaped["origin_W_m"][:, 2],
            s=marker_size,
            color="#d81b60",
            alpha=0.35,
            edgecolors="none",
            zorder=5,
            label="+Y escapes",
        )

    led_xz_mm = 1.0e3 * led.position_W_m[[0, 2]]
    led_normal_xz = led.normal_W[[0, 2]]
    axis.scatter(
        *led_xz_mm,
        marker="*",
        s=110,
        color="#2e7d32",
        edgecolors="black",
        linewidths=0.5,
        zorder=7,
        label="LED",
    )
    axis.arrow(
        *led_xz_mm,
        *(3.0 * led_normal_xz),
        width=0.08,
        head_width=0.7,
        head_length=0.9,
        length_includes_head=True,
        color="#2e7d32",
        zorder=6,
    )

    center_x_mm, center_z_mm = quadrant_center_xz_mm
    axis.axvline(center_x_mm, color="0.35", linestyle="--", linewidth=0.8)
    axis.axhline(center_z_mm, color="0.35", linestyle="--", linewidth=0.8)
    axis.text(center_x_mm + 0.5, center_z_mm + 0.5, "Q1", fontsize=8)
    axis.text(center_x_mm - 1.8, center_z_mm + 0.5, "Q2", fontsize=8)
    axis.text(center_x_mm - 1.8, center_z_mm - 1.2, "Q3", fontsize=8)
    axis.text(center_x_mm + 0.5, center_z_mm - 1.2, "Q4", fontsize=8)

    if indenter_center_xz_mm is not None:
        axis.add_patch(
            Circle(
                indenter_center_xz_mm,
                radius=1.0e3 * _SPHERE_RADIUS_M,
                fill=False,
                edgecolor="black",
                linewidth=1.0,
                linestyle=":",
                zorder=6,
                label="indenter",
            )
        )

    observation_text = np.array2string(
        observation,
        precision=4,
        suppress_small=True,
    )
    subtitle = f"Q={observation_text}"
    if intensity_response:
        subtitle += f"\nΔI/I₀={100.0 * intensity_response:+.2f}%"
    axis.set_title(f"{title}\n{subtitle}", fontsize=10)
    axis.set_xlabel("X [mm]")
    axis.grid(alpha=0.18)


def main() -> None:
    args = _parse_args()
    if args.output is not None:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fingertip = Fingertip(FingertipParameters())
    fingertip_mesh = make_fingertip_mesh(fingertip)
    reference_vertices = np.asarray(
        fingertip_mesh.silicone.vertices,
        dtype=np.float64,
    )
    triangles = np.asarray(
        fingertip_mesh.silicone.surface_tri_indices,
        dtype=np.int64,
    )
    led = _make_led(fingertip, fingertip_mesh)
    section_y_m = float(led.position_W_m[1])
    scene = OptixScene(
        fingertip_mesh,
        silicone_instance_id=_SILICONE_INSTANCE_ID,
        carrier_instance_id=_CARRIER_INSTANCE_ID,
        silicone_visibility_mask=_SILICONE_MASK,
        carrier_visibility_mask=_CARRIER_MASK,
    )

    sample_i, sample_j = np.meshgrid(
        np.arange(_SAMPLE_SIDE_COUNT),
        np.arange(_SAMPLE_SIDE_COUNT),
        indexing="ij",
    )
    emission_u1 = (sample_i.ravel() + 0.5) / _SAMPLE_SIDE_COUNT
    emission_u2 = (sample_j.ravel() + 0.5) / _SAMPLE_SIDE_COUNT
    emission = _emit_from_stem_boundary(
        scene,
        led,
        emission_u1,
        emission_u2,
    )
    source_inside_silicone = _source_inside_silicone(
        scene,
        led,
        emission,
        state_label="unloaded",
    )
    if not source_inside_silicone:
        raise AssertionError("reference stem source is not touching silicone")
    rng = np.random.default_rng(_RNG_SEED)
    dielectric_branch_u = rng.random((_BOUNCE_CAP, len(emission)))
    carrier_u1 = rng.random((_BOUNCE_CAP, len(emission)))
    carrier_u2 = rng.random((_BOUNCE_CAP, len(emission)))

    before_escaped, before_segments, before_observation = _trace_state(
        scene,
        fingertip,
        emission,
        source_inside_silicone,
        dielectric_branch_u,
        carrier_u1,
        carrier_u2,
    )

    loaded_vertices: np.ndarray | None = None
    loaded_escaped: np.ndarray | None = None
    loaded_segments: np.ndarray | None = None
    loaded_observation: np.ndarray | None = None
    loaded_force_n: float | None = None
    loaded_travel_m: float | None = None
    indenter_center_xz_mm: np.ndarray | None = None

    def inspect_loaded(
        trial: DesignTrial,
        simulation: LumoSimulation,
        indenter: Indenter,
    ) -> None:
        nonlocal loaded_vertices
        nonlocal loaded_escaped
        nonlocal loaded_segments
        nonlocal loaded_observation
        nonlocal loaded_force_n
        nonlocal loaded_travel_m
        nonlocal indenter_center_xz_mm

        if simulation.soft_contact_count(indenter.body_index) == 0:
            raise AssertionError("loaded state has no sphere contact")
        vertices = simulation.silicone_vertices()
        if vertices.shape != reference_vertices.shape or not np.all(
            np.isfinite(vertices)
        ):
            raise AssertionError("loaded silicone vertices are invalid")
        trial_reference_vertices = np.asarray(
            simulation.fingertip_mesh.silicone.vertices,
            dtype=np.float64,
        )
        if not np.allclose(
            trial_reference_vertices,
            reference_vertices,
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise AssertionError("loaded trial changed silicone vertex ordering")
        if not np.array_equal(
            simulation.fingertip_mesh.silicone.surface_tri_indices,
            triangles,
        ):
            raise AssertionError("loaded silicone topology changed")
        if (
            trial.final_tf is None
            or trial.reaction_force_n is None
            or trial.travel_m is None
        ):
            raise AssertionError("loaded trial has no force-stable result")

        scene.update_silicone(vertices)
        source_inside_silicone = _source_inside_silicone(
            scene,
            led,
            emission,
            state_label="loaded",
        )
        sphere_top_z_m = (
            float(np.asarray(trial.final_tf, dtype=np.float64)[2])
            + _SPHERE_RADIUS_M
        )
        if led.position_W_m[2] <= sphere_top_z_m:
            raise AssertionError("loaded sphere reached the LED source plane")
        loaded_escaped, loaded_segments, loaded_observation = _trace_state(
            scene,
            fingertip,
            emission,
            source_inside_silicone,
            dielectric_branch_u,
            carrier_u1,
            carrier_u2,
        )
        loaded_vertices = vertices.copy()
        loaded_force_n = float(trial.reaction_force_n)
        loaded_travel_m = float(trial.travel_m)
        indenter_pose = np.asarray(trial.final_tf, dtype=np.float64)
        indenter_center_xz_mm = 1.0e3 * indenter_pose[[0, 2]]

    sphere_resource = files("lumo").joinpath(
        "assets",
        "objects",
        "urdf",
        "sphere_15mm.urdf",
    )
    with as_file(sphere_resource) as sphere_urdf_path:
        trial = _make_center_trial(fingertip, sphere_urdf_path)
        DesignStudy(
            fingertip,
            (trial,),
            fingertip_mesh=fingertip_mesh,
            sim_frequency=_SIM_FREQUENCY_HZ,
            force_tolerance_n=_FORCE_TOLERANCE_N,
            settle_duration_s=_SETTLE_DURATION_S,
            iterations=_VBD_ITERATIONS,
            contact_stiffness_n_m=_CONTACT_STIFFNESS_N_M,
            contact_damping_n_s_m=_CONTACT_DAMPING_N_S_M,
        ).run(inspect_trial=inspect_loaded)

    if any(
        value is None
        for value in (
            loaded_vertices,
            loaded_escaped,
            loaded_segments,
            loaded_observation,
            loaded_force_n,
            loaded_travel_m,
            indenter_center_xz_mm,
        )
    ):
        raise RuntimeError("loaded visualization state was not captured")

    before_section = _surface_section_xz(
        reference_vertices,
        triangles,
        section_y_m=section_y_m,
    )
    loaded_section = _surface_section_xz(
        loaded_vertices,
        triangles,
        section_y_m=section_y_m,
    )
    maximum_displacement_m = float(
        np.linalg.norm(loaded_vertices - reference_vertices, axis=1).max()
    )
    if not np.isfinite(maximum_displacement_m) or maximum_displacement_m <= 0.0:
        raise AssertionError("loaded silicone did not deform")

    responses = np.stack((before_observation, loaded_observation))
    intensity, _ = sensing_descriptors(responses)
    selected_ray_ids = np.unique(
        np.linspace(
            0,
            len(emission) - 1,
            num=min(_PLOTTED_PATH_COUNT, len(emission)),
            dtype=np.int64,
        )
    )
    carrier_xz_mm = np.asarray(fingertip.carrier.cross_section, dtype=np.float64)
    left, right = fingertip.silicone.semiellipse_endpoints
    quadrant_center_xz_mm = np.array(
        (
            0.5 * (left[0] + right[0]),
            fingertip.silicone.ellipse_center_z_mm,
        )
    )

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), sharex=True, sharey=True)
    _plot_panel(
        axes[0],
        title="No load",
        section_xz_m=before_section,
        carrier_xz_mm=carrier_xz_mm,
        led=led,
        segments=before_segments,
        escaped=before_escaped,
        observation=before_observation,
        selected_ray_ids=selected_ray_ids,
        quadrant_center_xz_mm=quadrant_center_xz_mm,
        intensity_response=float(intensity[0]),
    )
    _plot_panel(
        axes[1],
        title="20 N load",
        section_xz_m=loaded_section,
        carrier_xz_mm=carrier_xz_mm,
        led=led,
        segments=loaded_segments,
        escaped=loaded_escaped,
        observation=loaded_observation,
        selected_ray_ids=selected_ray_ids,
        quadrant_center_xz_mm=quadrant_center_xz_mm,
        intensity_response=float(intensity[1]),
        indenter_center_xz_mm=indenter_center_xz_mm,
    )

    all_section_points_mm = np.concatenate(
        (before_section.reshape(-1, 2), loaded_section.reshape(-1, 2)),
        axis=0,
    ) * 1.0e3
    x_values_mm = np.concatenate(
        (
            all_section_points_mm[:, 0],
            carrier_xz_mm[:, 0],
            np.array((1.0e3 * led.position_W_m[0],)),
        )
    )
    z_values_mm = np.concatenate(
        (
            all_section_points_mm[:, 1],
            carrier_xz_mm[:, 1],
            np.array((1.0e3 * led.position_W_m[2],)),
        )
    )
    x_margin_mm = 2.0
    z_margin_mm = 2.0
    for axis in axes:
        axis.set_xlim(x_values_mm.min() - x_margin_mm, x_values_mm.max() + x_margin_mm)
        axis.set_ylim(z_values_mm.min() - z_margin_mm, z_values_mm.max() + z_margin_mm)
        axis.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Z [mm]")
    axes[0].legend(loc="lower left", fontsize=7, framealpha=0.9)
    figure.suptitle(
        "Projection of full 3D bounded optical paths onto the LED-center "
        f"X-Z section (Y={1.0e3 * section_y_m:.3f} mm)",
        fontsize=12,
    )
    figure.tight_layout()

    print(f"LED position W [m]: {led.position_W_m}")
    print(f"diagnostic optical paths: {len(emission)} ({_BOUNCE_CAP} bounces)")
    print(f"selected original ray IDs: {len(selected_ray_ids)} (same in both panels)")
    print(f"before quadrant power: {before_observation}")
    print(f"loaded quadrant power: {loaded_observation}")
    print(f"loaded normalized intensity response: {float(intensity[1]):+.9e}")
    print(f"loaded contact force: {loaded_force_n:.6f} N")
    print(f"loaded travel: {1.0e3 * loaded_travel_m:.6f} mm")
    print(
        "loaded indentation: "
        f"{1.0e3 * (loaded_travel_m - _INITIAL_CLEARANCE_M):.6f} mm"
    )
    print(f"maximum silicone displacement: {1.0e3 * maximum_displacement_m:.6f} mm")

    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=180, bbox_inches="tight")
        print(f"saved visualization: {args.output.resolve()}")
    plt.close(figure)


if __name__ == "__main__":
    main()
