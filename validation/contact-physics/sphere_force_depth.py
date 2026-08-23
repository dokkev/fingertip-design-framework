"""Measure force and contact integrity during continuous sphere indentation."""

from __future__ import annotations

import csv
from importlib.resources import as_file, files
from pathlib import Path

import matplotlib
import newton
import numpy as np
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.newton import Indenter
from lumo.simulation import LumoSimulation


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


_SIM_FREQUENCY_HZ = 1.0e3
_VBD_ITERATIONS = 10
_SPHERE_RADIUS_M = 7.5e-3
_INITIAL_CLEARANCE_M = 1.0e-3
_APPROACH_SPEED_M_S = 2.5e-2
_MAX_INDENTATION_DEPTH_M = 10.0e-3
_TARGET_FORCE_N = 20.0
_FIXED_POSE_HOLD_DURATION_S = 1.0
_REPORT_INTERVAL_TICKS = 25
_MOTION_DIRECTION_W = wp.vec3(0.0, 0.0, 1.0)
_OUTPUT_DIRECTORY = Path("output/validation")


def _sphere_penetration_depths_m(
    positions_m: np.ndarray,
    sphere_center_m: np.ndarray,
) -> np.ndarray:
    return np.maximum(
        0.0,
        _SPHERE_RADIUS_M - np.linalg.norm(positions_m - sphere_center_m, axis=1),
    )


def _six_tet_volumes(
    positions_m: np.ndarray,
    tet_indices: np.ndarray,
) -> np.ndarray:
    tets = positions_m[tet_indices]
    return np.einsum(
        "ij,ij->i",
        tets[:, 1] - tets[:, 0],
        np.cross(tets[:, 2] - tets[:, 0], tets[:, 3] - tets[:, 0]),
    )


def _make_simulation(
    fingertip: Fingertip,
    initial_pose: wp.transform,
) -> tuple[LumoSimulation, Indenter]:
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, 0.0))
    sphere_resource = files("lumo").joinpath(
        "assets",
        "objects",
        "urdf",
        "sphere_15mm.urdf",
    )
    with as_file(sphere_resource) as urdf_path:
        indenter = Indenter.add_urdf(builder, urdf_path, tf=initial_pose)

    simulation = LumoSimulation(
        fingertip,
        builder=builder,
        sim_frequency=_SIM_FREQUENCY_HZ,
        iterations=_VBD_ITERATIONS,
    )
    if simulation.soft_contact_count(indenter.body_index) != 0:
        raise RuntimeError("15 mm sphere has contacts before prescribed motion")
    return simulation, indenter


def _print_fixed_pose_force_decay(
    fingertip: Fingertip,
    initial_pose: wp.transform,
    initial_sphere_z_m: float,
) -> None:
    simulation, indenter = _make_simulation(fingertip, initial_pose)
    position_step_m = _APPROACH_SPEED_M_S / simulation.sim_frequency
    travel_m = 0.0

    maximum_travel_m = _INITIAL_CLEARANCE_M + _MAX_INDENTATION_DEPTH_M
    maximum_step_count = int(np.ceil(maximum_travel_m / position_step_m))
    for approach_step in range(1, maximum_step_count + 1):
        travel_m = approach_step * position_step_m
        sphere_center_m = wp.vec3(
            0.0,
            0.0,
            initial_sphere_z_m + travel_m,
        )
        simulation.apply_indenter_pose(
            indenter,
            wp.transform(sphere_center_m, wp.quat_identity()),
        )
        simulation.step()
        reaction_force_n = simulation.indenter_reaction_force(
            indenter,
            motion_direction_W=_MOTION_DIRECTION_W,
        )
        if reaction_force_n >= _TARGET_FORCE_N:
            break
    else:
        raise RuntimeError(
            "15 mm sphere did not reach 20 N before 10 mm indentation"
        )

    checkpoint_ticks = {
        0,
        round(5.0e-3 * simulation.sim_frequency),
        round(100.0e-3 * simulation.sim_frequency),
        round(_FIXED_POSE_HOLD_DURATION_S * simulation.sim_frequency),
    }
    checkpoint_forces_n = {0: reaction_force_n}
    for hold_tick in range(1, max(checkpoint_ticks) + 1):
        simulation.step()
        if hold_tick in checkpoint_ticks:
            checkpoint_forces_n[hold_tick] = simulation.indenter_reaction_force(
                indenter,
                motion_direction_W=_MOTION_DIRECTION_W,
            )

    print()
    print(
        "Fresh fixed-pose hold after first 20 N crossing at "
        f"{1.0e3 * (travel_m - _INITIAL_CLEARANCE_M):.6f} mm:"
    )
    for hold_tick in sorted(checkpoint_forces_n):
        hold_time_s = hold_tick / simulation.sim_frequency
        print(
            f"  F({1.0e3 * hold_time_s:g} ms) = "
            f"{checkpoint_forces_n[hold_tick]:.9f} N"
        )


def main() -> None:
    fingertip = Fingertip(FingertipParameters())
    initial_sphere_z_m = (
        fingertip.tip_z_m
        - _INITIAL_CLEARANCE_M
        - _SPHERE_RADIUS_M
    )
    initial_pose = wp.transform(
        wp.vec3(0.0, 0.0, initial_sphere_z_m),
        wp.quat_identity(),
    )

    simulation, indenter = _make_simulation(fingertip, initial_pose)

    model = simulation.fingertip_model.model
    shape_bodies = model.shape_body.numpy()
    sphere_shape_stiffnesses_n_m = np.unique(
        model.shape_material_ke.numpy()[shape_bodies == indenter.body_index]
    )
    sphere_shape_dampings_n_s_m = np.unique(
        model.shape_material_kd.numpy()[shape_bodies == indenter.body_index]
    )
    effective_sphere_stiffnesses_n_m = 0.5 * (
        model.soft_contact_ke + sphere_shape_stiffnesses_n_m
    )
    effective_sphere_dampings_n_s_m = 0.5 * (
        model.soft_contact_kd + sphere_shape_dampings_n_s_m
    )
    print(
        "sphere shape contact stiffness [N/m]: "
        f"{sphere_shape_stiffnesses_n_m.tolist()}"
    )
    print(
        "silicone particle contact stiffness [N/m]: "
        f"{model.soft_contact_ke:g}"
    )
    print(
        "sphere shape contact damping [N s/m]: "
        f"{sphere_shape_dampings_n_s_m.tolist()}"
    )
    print(
        "silicone particle contact damping [N s/m]: "
        f"{model.soft_contact_kd:g}"
    )
    print(
        "effective averaged sphere contact stiffness [N/m]: "
        f"{effective_sphere_stiffnesses_n_m.tolist()}"
    )
    print(
        "effective averaged sphere contact damping [N s/m]: "
        f"{effective_sphere_dampings_n_s_m.tolist()}"
    )

    reference_positions_m = np.asarray(
        simulation.fingertip_mesh.silicone.vertices,
        dtype=np.float64,
    )
    bonded_indices = simulation.fingertip_model.bonded_particle_indices.numpy()
    nonbonded = np.ones(len(reference_positions_m), dtype=bool)
    nonbonded[bonded_indices] = False
    surface_triangles = np.asarray(
        simulation.fingertip_mesh.silicone.surface_tri_indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    free_surface_triangles = surface_triangles[
        np.all(nonbonded[surface_triangles], axis=1)
    ]
    free_surface_vertices = np.unique(free_surface_triangles)
    tet_indices = np.asarray(
        simulation.fingertip_mesh.silicone.tet_indices,
        dtype=np.int32,
    ).reshape(-1, 4)
    free_tet_indices = tet_indices[np.all(nonbonded[tet_indices], axis=1)]
    reference_six_volumes = _six_tet_volumes(
        reference_positions_m,
        tet_indices,
    )
    if np.any(np.abs(reference_six_volumes) <= 1.0e-18):
        raise RuntimeError("reference silicone mesh contains a degenerate tet")

    columns: dict[str, list[float | int]] = {
        "simulation_time_s": [],
        "sphere_travel_m": [],
        "indentation_depth_m": [],
        "reaction_force_n": [],
        "sphere_contact_count": [],
        "maximum_nonbonded_particle_penetration_m": [],
        "nonbonded_particle_inside_count": [],
        "maximum_surface_vertex_penetration_m": [],
        "surface_vertex_inside_count": [],
        "maximum_surface_centroid_penetration_m": [],
        "surface_centroid_inside_count": [],
        "maximum_free_tet_center_penetration_m": [],
        "free_tet_center_inside_count": [],
        "minimum_det_f": [],
        "minimum_absolute_tet_volume_ratio": [],
        "inverted_tet_count": [],
    }

    position_step_m = _APPROACH_SPEED_M_S / simulation.sim_frequency
    maximum_travel_m = _INITIAL_CLEARANCE_M + _MAX_INDENTATION_DEPTH_M
    maximum_step_count = int(np.ceil(maximum_travel_m / position_step_m))
    target_crossing_index: int | None = None

    print(
        "Continuous 15 mm sphere indentation: "
        f"speed={1.0e3 * _APPROACH_SPEED_M_S:.3f} mm/s, "
        f"maximum depth={1.0e3 * _MAX_INDENTATION_DEPTH_M:.3f} mm",
        flush=True,
    )
    print(
        "depth_mm   force_N  contacts  particle_um  surface_um  "
        "centroid_um  tet_um   min_detF  inverted",
        flush=True,
    )

    for approach_step in range(1, maximum_step_count + 1):
        travel_m = approach_step * position_step_m
        indentation_depth_m = travel_m - _INITIAL_CLEARANCE_M
        sphere_center_m = np.array(
            (0.0, 0.0, initial_sphere_z_m + travel_m),
            dtype=np.float64,
        )
        simulation.apply_indenter_pose(
            indenter,
            wp.transform(
                wp.vec3(*sphere_center_m),
                wp.quat_identity(),
            ),
        )
        simulation.step()
        reaction_force_n = simulation.indenter_reaction_force(
            indenter,
            motion_direction_W=_MOTION_DIRECTION_W,
        )
        sphere_contact_count = simulation.soft_contact_count(indenter.body_index)
        positions_m = np.asarray(
            simulation.silicone_vertices(),
            dtype=np.float64,
        )
        velocities_m_s = simulation.state.particle_qd.numpy()
        if not np.all(np.isfinite(positions_m)) or not np.all(
            np.isfinite(velocities_m_s)
        ):
            raise RuntimeError("continuous indentation produced a non-finite state")

        particle_penetrations_m = _sphere_penetration_depths_m(
            positions_m[nonbonded],
            sphere_center_m,
        )
        surface_vertex_penetrations_m = _sphere_penetration_depths_m(
            positions_m[free_surface_vertices],
            sphere_center_m,
        )
        surface_centroid_penetrations_m = _sphere_penetration_depths_m(
            positions_m[free_surface_triangles].mean(axis=1),
            sphere_center_m,
        )
        free_tet_center_penetrations_m = _sphere_penetration_depths_m(
            positions_m[free_tet_indices].mean(axis=1),
            sphere_center_m,
        )
        current_six_volumes = _six_tet_volumes(positions_m, tet_indices)
        det_f = current_six_volumes / reference_six_volumes
        if not np.all(np.isfinite(det_f)):
            raise RuntimeError("continuous indentation produced non-finite det(F)")

        metrics = {
            "simulation_time_s": simulation.time_s,
            "sphere_travel_m": travel_m,
            "indentation_depth_m": indentation_depth_m,
            "reaction_force_n": reaction_force_n,
            "sphere_contact_count": sphere_contact_count,
            "maximum_nonbonded_particle_penetration_m": float(
                particle_penetrations_m.max()
            ),
            "nonbonded_particle_inside_count": int(
                np.count_nonzero(particle_penetrations_m > 0.0)
            ),
            "maximum_surface_vertex_penetration_m": float(
                surface_vertex_penetrations_m.max()
            ),
            "surface_vertex_inside_count": int(
                np.count_nonzero(surface_vertex_penetrations_m > 0.0)
            ),
            "maximum_surface_centroid_penetration_m": float(
                surface_centroid_penetrations_m.max()
            ),
            "surface_centroid_inside_count": int(
                np.count_nonzero(surface_centroid_penetrations_m > 0.0)
            ),
            "maximum_free_tet_center_penetration_m": float(
                free_tet_center_penetrations_m.max()
            ),
            "free_tet_center_inside_count": int(
                np.count_nonzero(free_tet_center_penetrations_m > 0.0)
            ),
            # For a linear tetrahedron, the signed volume ratio equals det(F).
            "minimum_det_f": float(det_f.min()),
            "minimum_absolute_tet_volume_ratio": float(np.abs(det_f).min()),
            "inverted_tet_count": int(np.count_nonzero(det_f <= 0.0)),
        }
        for key, value in metrics.items():
            columns[key].append(value)

        if target_crossing_index is None and reaction_force_n >= _TARGET_FORCE_N:
            target_crossing_index = len(columns["reaction_force_n"]) - 1
            print("--- first transient 20 N crossing ---", flush=True)

        if (
            approach_step % _REPORT_INTERVAL_TICKS == 0
            or approach_step == maximum_step_count
            or target_crossing_index == len(columns["reaction_force_n"]) - 1
        ):
            print(
                f"{1.0e3 * indentation_depth_m:8.3f} "
                f"{reaction_force_n:9.4f} "
                f"{sphere_contact_count:9d} "
                f"{1.0e6 * metrics['maximum_nonbonded_particle_penetration_m']:11.3f} "
                f"{1.0e6 * metrics['maximum_surface_vertex_penetration_m']:10.3f} "
                f"{1.0e6 * metrics['maximum_surface_centroid_penetration_m']:11.3f} "
                f"{1.0e6 * metrics['maximum_free_tet_center_penetration_m']:8.3f} "
                f"{metrics['minimum_det_f']:10.5f} "
                f"{metrics['inverted_tet_count']:8d}",
                flush=True,
            )

    force_n = np.asarray(columns["reaction_force_n"], dtype=np.float64)
    depth_m = np.asarray(columns["indentation_depth_m"], dtype=np.float64)
    peak_index = int(np.argmax(force_n))
    print()
    print(
        f"peak force: {force_n[peak_index]:.9f} N at "
        f"{1.0e3 * depth_m[peak_index]:.6f} mm indentation"
    )
    print(
        f"final force: {force_n[-1]:.9f} N at {1.0e3 * depth_m[-1]:.6f} mm indentation"
    )
    if target_crossing_index is None:
        print("transient 20 N target was not reached")
    else:
        print(
            "first 20 N crossing depth: "
            f"{1.0e3 * depth_m[target_crossing_index]:.6f} mm"
        )
    print(
        "maximum sphere penetration [um]: "
        f"particle={1.0e6 * max(columns['maximum_nonbonded_particle_penetration_m']):.3f}, "
        f"surface={1.0e6 * max(columns['maximum_surface_vertex_penetration_m']):.3f}, "
        f"centroid={1.0e6 * max(columns['maximum_surface_centroid_penetration_m']):.3f}, "
        f"tet={1.0e6 * max(columns['maximum_free_tet_center_penetration_m']):.3f}"
    )
    print(f"minimum det(F): {min(columns['minimum_det_f']):.9f}")
    print(f"maximum inverted tet count: {max(columns['inverted_tet_count'])}")

    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    csv_path = _OUTPUT_DIRECTORY / "sphere_force_depth.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values(), strict=True))

    depth_mm = 1.0e3 * depth_m
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), sharex=True)
    axes[0, 0].plot(depth_mm, force_n)
    axes[0, 0].axhline(
        _TARGET_FORCE_N,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    axes[0, 0].set_ylabel("reaction force [N]")

    axes[0, 1].plot(depth_mm, columns["sphere_contact_count"])
    axes[0, 1].set_ylabel("sphere contact count")

    penetration_columns = (
        ("particle", "maximum_nonbonded_particle_penetration_m"),
        ("surface vertex", "maximum_surface_vertex_penetration_m"),
        ("surface centroid", "maximum_surface_centroid_penetration_m"),
        ("free tet center", "maximum_free_tet_center_penetration_m"),
    )
    for label, column in penetration_columns:
        axes[1, 0].plot(
            depth_mm,
            1.0e6 * np.asarray(columns[column]),
            label=label,
        )
    axes[1, 0].set_xlabel("indentation depth [mm]")
    axes[1, 0].set_ylabel("sphere penetration [um]")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        depth_mm,
        columns["minimum_det_f"],
        label="minimum det(F)",
    )
    axes[1, 1].plot(
        depth_mm,
        columns["minimum_absolute_tet_volume_ratio"],
        label="minimum |volume ratio|",
    )
    axes[1, 1].axhline(0.0, color="black", linewidth=1.0)
    axes[1, 1].set_xlabel("indentation depth [mm]")
    axes[1, 1].set_ylabel("tet deformation ratio")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.suptitle("Continuous 15 mm sphere indentation diagnostics")
    figure.tight_layout()
    figure_path = _OUTPUT_DIRECTORY / "sphere_force_depth.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print(f"trajectory CSV: {csv_path}")
    print(f"trajectory plot: {figure_path}")

    del simulation, indenter
    _print_fixed_pose_force_decay(
        fingertip,
        initial_pose,
        initial_sphere_z_m,
    )


if __name__ == "__main__":
    main()
