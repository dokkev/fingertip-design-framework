"""Tune centered-sphere contact stiffness, damping, and timestep together."""

from __future__ import annotations

import csv
from importlib.resources import as_file, files
from math import ceil, sqrt
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


_PRIMARY_FREQUENCY_HZ = 2.0e3
_TIMESTEP_CHECK_FREQUENCY_HZ = 4.0e3
_VBD_ITERATIONS = 10
_SPHERE_RADIUS_M = 7.5e-3
_INITIAL_CLEARANCE_M = 1.0e-3
_APPROACH_SPEED_M_S = 2.5e-2
_TARGET_FORCE_N = 20.0
_MAX_INDENTATION_DEPTH_M = 10.0e-3
_POST_TARGET_TRAVEL_M = 0.5e-3
_HOLD_DURATION_S = 1.0
_SHAPE_STIFFNESSES_N_M = (1.0e4, 3.0e4, 1.0e5)
_DAMPING_RATIOS = (0.5, 1.0, 2.0)
_MOTION_DIRECTION_W = wp.vec3(0.0, 0.0, 1.0)
_OUTPUT_DIRECTORY = Path("output/validation")


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


def _sphere_penetration_m(
    positions_m: np.ndarray,
    sphere_center_m: np.ndarray,
) -> np.ndarray:
    return np.maximum(
        0.0,
        _SPHERE_RADIUS_M
        - np.linalg.norm(positions_m - sphere_center_m, axis=1),
    )


def _make_simulation(
    fingertip: Fingertip,
    *,
    sim_frequency_hz: float,
    shape_stiffness_n_m: float | None,
    shape_damping_n_s_m: float | None,
) -> tuple[LumoSimulation, Indenter, float]:
    initial_sphere_z_m = (
        fingertip.tip_z_m
        - _INITIAL_CLEARANCE_M
        - _SPHERE_RADIUS_M
    )
    initial_tf = wp.transform(
        wp.vec3(0.0, 0.0, initial_sphere_z_m),
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
        indenter = Indenter.add_urdf(
            builder,
            urdf_path,
            tf=initial_tf,
            contact_stiffness_n_m=shape_stiffness_n_m,
            contact_damping_n_s_m=shape_damping_n_s_m,
        )

    simulation = LumoSimulation(
        fingertip,
        builder=builder,
        sim_frequency=sim_frequency_hz,
        iterations=_VBD_ITERATIONS,
        soft_contact_stiffness_n_m=shape_stiffness_n_m,
        soft_contact_damping_n_s_m=shape_damping_n_s_m,
    )
    if simulation.soft_contact_count(indenter.body_index) != 0:
        raise RuntimeError("15 mm sphere has contacts before prescribed motion")
    return simulation, indenter, initial_sphere_z_m


def _sphere_contact_effective_masses_kg(
    simulation: LumoSimulation,
    indenter: Indenter,
) -> np.ndarray:
    count = simulation.soft_contact_count()
    shapes = simulation.contacts.soft_contact_shape.numpy()[:count]
    valid_shapes = shapes >= 0
    shape_bodies = simulation.fingertip_model.model.shape_body.numpy()
    sphere_contacts = valid_shapes.copy()
    sphere_contacts[valid_shapes] = (
        shape_bodies[shapes[valid_shapes]] == indenter.body_index
    )

    indices = simulation.contacts.soft_contact_indices.numpy()[:count][
        sphere_contacts
    ]
    barycentric = simulation.contacts.soft_contact_barycentric.numpy()[
        :count
    ][sphere_contacts]
    particle_masses_kg = (
        simulation.fingertip_model.model.particle_mass.numpy()
    )
    effective_masses_kg: list[float] = []
    for particle_indices, weights in zip(
        indices,
        barycentric,
        strict=True,
    ):
        valid_particles = particle_indices >= 0
        masses_kg = particle_masses_kg[
            particle_indices[valid_particles]
        ]
        weights = weights[valid_particles]
        positive_mass = masses_kg > 0.0
        if np.any(positive_mass):
            inverse_mass_kg = np.sum(
                weights[positive_mass] ** 2
                / masses_kg[positive_mass]
            )
            effective_masses_kg.append(1.0 / inverse_mass_kg)

    if not effective_masses_kg:
        raise RuntimeError("sphere contacts contain no positive particle mass")
    return np.asarray(effective_masses_kg, dtype=np.float64)


def _approach_to_target(
    simulation: LumoSimulation,
    indenter: Indenter,
    initial_sphere_z_m: float,
) -> tuple[float, float]:
    position_step_m = _APPROACH_SPEED_M_S / simulation.sim_frequency
    maximum_travel_m = _INITIAL_CLEARANCE_M + _MAX_INDENTATION_DEPTH_M
    maximum_step_count = ceil(maximum_travel_m / position_step_m)
    for approach_step in range(1, maximum_step_count + 1):
        travel_m = approach_step * position_step_m
        simulation.apply_indenter_pose(
            indenter,
            wp.transform(
                wp.vec3(0.0, 0.0, initial_sphere_z_m + travel_m),
                wp.quat_identity(),
            ),
        )
        simulation.step()
        reaction_force_n = simulation.indenter_reaction_force(
            indenter,
            motion_direction_W=_MOTION_DIRECTION_W,
        )
        if reaction_force_n >= _TARGET_FORCE_N:
            return travel_m, reaction_force_n
    raise RuntimeError("sphere did not reach 20 N before 10 mm indentation")


def _probe_contact_mass(fingertip: Fingertip) -> float:
    simulation, indenter, initial_sphere_z_m = _make_simulation(
        fingertip,
        sim_frequency_hz=_PRIMARY_FREQUENCY_HZ,
        shape_stiffness_n_m=None,
        shape_damping_n_s_m=None,
    )
    travel_m, force_n = _approach_to_target(
        simulation,
        indenter,
        initial_sphere_z_m,
    )
    model = simulation.fingertip_model.model
    shape_bodies = model.shape_body.numpy()
    sphere_shapes = shape_bodies == indenter.body_index
    shape_ke = np.unique(model.shape_material_ke.numpy()[sphere_shapes])
    shape_kd = np.unique(model.shape_material_kd.numpy()[sphere_shapes])
    contact_masses_kg = _sphere_contact_effective_masses_kg(
        simulation,
        indenter,
    )
    contact_count = simulation.soft_contact_count()
    contact_shapes = simulation.contacts.soft_contact_shape.numpy()[
        :contact_count
    ]
    valid_shapes = contact_shapes >= 0
    sphere_records = valid_shapes.copy()
    sphere_records[valid_shapes] = (
        shape_bodies[contact_shapes[valid_shapes]] == indenter.body_index
    )
    sphere_indices = simulation.contacts.soft_contact_indices.numpy()[
        :contact_count
    ][sphere_records]
    particle_record_count = int(
        np.count_nonzero(sphere_indices[:, 1] < 0)
    )
    full_surface_record_count = len(sphere_indices) - particle_record_count

    print(f"Newton {newton.__version__} nominal contact values:")
    print(f"  indenter shape ke: {shape_ke.tolist()} N/m")
    print(f"  indenter shape kd: {shape_kd.tolist()} N s/m")
    print(f"  soft_contact_ke: {model.soft_contact_ke:g} N/m")
    print(f"  soft_contact_kd: {model.soft_contact_kd:g} N s/m")
    print(
        "  effective contact mass at first 20 N crossing: "
        f"median={np.median(contact_masses_kg):.9e} kg, "
        f"range=[{contact_masses_kg.min():.9e}, "
        f"{contact_masses_kg.max():.9e}] kg"
    )
    print(
        f"  crossing: depth={1.0e3 * (travel_m - _INITIAL_CLEARANCE_M):.6f} mm, "
        f"force={force_n:.6f} N, contacts={len(contact_masses_kg)} "
        f"(particle={particle_record_count}, "
        f"edge/face={full_surface_record_count})"
    )
    return float(np.median(contact_masses_kg))


def _run_force_depth(
    fingertip: Fingertip,
    *,
    sim_frequency_hz: float,
    shape_stiffness_n_m: float,
    shape_damping_n_s_m: float,
) -> tuple[dict[str, float | int | bool], list[dict[str, float | int]]]:
    simulation, indenter, initial_sphere_z_m = _make_simulation(
        fingertip,
        sim_frequency_hz=sim_frequency_hz,
        shape_stiffness_n_m=shape_stiffness_n_m,
        shape_damping_n_s_m=shape_damping_n_s_m,
    )
    model = simulation.fingertip_model.model
    reference_positions_m = np.asarray(
        simulation.fingertip_mesh.silicone.vertices,
        dtype=np.float64,
    )
    bonded_indices = (
        simulation.fingertip_model.bonded_particle_indices.numpy()
    )
    nonbonded = np.ones(len(reference_positions_m), dtype=bool)
    nonbonded[bonded_indices] = False
    surface_triangles = np.asarray(
        simulation.fingertip_mesh.silicone.surface_tri_indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    free_surface_triangles = surface_triangles[
        np.all(nonbonded[surface_triangles], axis=1)
    ]
    tet_indices = np.asarray(
        simulation.fingertip_mesh.silicone.tet_indices,
        dtype=np.int32,
    ).reshape(-1, 4)
    free_tet_indices = tet_indices[
        np.all(nonbonded[tet_indices], axis=1)
    ]
    reference_six_volumes = _six_tet_volumes(
        reference_positions_m,
        tet_indices,
    )

    position_step_m = _APPROACH_SPEED_M_S / simulation.sim_frequency
    maximum_travel_m = _INITIAL_CLEARANCE_M + _MAX_INDENTATION_DEPTH_M
    maximum_step_count = ceil(maximum_travel_m / position_step_m)
    post_target_step_count = ceil(
        _POST_TARGET_TRAVEL_M / position_step_m
    )
    target_step: int | None = None
    target_metrics: dict[str, float | int] | None = None
    rows: list[dict[str, float | int]] = []

    for approach_step in range(1, maximum_step_count + 1):
        travel_m = approach_step * position_step_m
        indentation_depth_m = travel_m - _INITIAL_CLEARANCE_M
        sphere_center_m = np.asarray(
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
        positions_m = np.asarray(
            simulation.silicone_vertices(),
            dtype=np.float64,
        )
        velocities_m_s = simulation.state.particle_qd.numpy()
        if not np.all(np.isfinite(positions_m)) or not np.all(
            np.isfinite(velocities_m_s)
        ):
            raise RuntimeError("contact tuning produced a non-finite state")

        particle_penetration_m = _sphere_penetration_m(
            positions_m[nonbonded],
            sphere_center_m,
        )
        surface_centroid_penetration_m = _sphere_penetration_m(
            positions_m[free_surface_triangles].mean(axis=1),
            sphere_center_m,
        )
        free_tet_center_penetration_m = _sphere_penetration_m(
            positions_m[free_tet_indices].mean(axis=1),
            sphere_center_m,
        )
        det_f = (
            _six_tet_volumes(positions_m, tet_indices)
            / reference_six_volumes
        )
        row: dict[str, float | int] = {
            "indentation_depth_m": indentation_depth_m,
            "reaction_force_n": reaction_force_n,
            "sphere_contact_count": simulation.soft_contact_count(
                indenter.body_index
            ),
            "particle_penetration_m": float(
                particle_penetration_m.max()
            ),
            "surface_centroid_penetration_m": float(
                surface_centroid_penetration_m.max()
            ),
            "free_tet_center_penetration_m": float(
                free_tet_center_penetration_m.max()
            ),
            "minimum_det_f": float(det_f.min()),
            "inverted_tet_count": int(np.count_nonzero(det_f <= 0.0)),
        }
        rows.append(row)

        if target_step is None and reaction_force_n >= _TARGET_FORCE_N:
            target_step = approach_step
            target_metrics = row.copy()
        if (
            target_step is not None
            and approach_step >= target_step + post_target_step_count
        ):
            break

    if target_step is None or target_metrics is None:
        raise RuntimeError("sphere did not reach 20 N before 10 mm indentation")

    target_row_index = target_step - 1
    post_target_forces_n = np.asarray(
        [row["reaction_force_n"] for row in rows[target_row_index:]],
        dtype=np.float64,
    )
    force_steps_n = np.diff(post_target_forces_n)
    contact_masses_kg = _sphere_contact_effective_masses_kg(
        simulation,
        indenter,
    )
    effective_ke_n_m = 0.5 * (model.soft_contact_ke + shape_stiffness_n_m)
    effective_kd_n_s_m = 0.5 * (
        model.soft_contact_kd + shape_damping_n_s_m
    )
    critical_kd_n_s_m = 2.0 * sqrt(
        effective_ke_n_m * float(np.median(contact_masses_kg))
    )
    summary: dict[str, float | int | bool] = {
        "sim_frequency_hz": sim_frequency_hz,
        "shape_ke_n_m": shape_stiffness_n_m,
        "shape_kd_n_s_m": shape_damping_n_s_m,
        "soft_contact_ke_n_m": model.soft_contact_ke,
        "soft_contact_kd_n_s_m": model.soft_contact_kd,
        "effective_ke_n_m": effective_ke_n_m,
        "effective_kd_n_s_m": effective_kd_n_s_m,
        "actual_damping_ratio": effective_kd_n_s_m / critical_kd_n_s_m,
        "target_depth_m": target_metrics["indentation_depth_m"],
        "target_force_n": target_metrics["reaction_force_n"],
        "target_contact_count": target_metrics["sphere_contact_count"],
        "target_particle_penetration_m": target_metrics[
            "particle_penetration_m"
        ],
        "target_surface_centroid_penetration_m": target_metrics[
            "surface_centroid_penetration_m"
        ],
        "target_free_tet_center_penetration_m": target_metrics[
            "free_tet_center_penetration_m"
        ],
        "target_minimum_det_f": target_metrics["minimum_det_f"],
        "minimum_det_f": min(float(row["minimum_det_f"]) for row in rows),
        "maximum_inverted_tet_count": max(
            int(row["inverted_tet_count"]) for row in rows
        ),
        "post_target_min_force_n": float(post_target_forces_n.min()),
        "post_target_max_force_n": float(post_target_forces_n.max()),
        "maximum_one_step_force_drop_n": float(
            max(0.0, -force_steps_n.min())
            if force_steps_n.size
            else 0.0
        ),
        "force_dropout": bool(np.any(post_target_forces_n <= 1.0e-6)),
        "body_particle_buffer_size": (
            simulation.solver.body_particle_contact_buffer_pre_alloc
        ),
        "body_particle_buffer_overflow_max": int(
            simulation.solver.body_particle_contact_overflow_max.numpy()[0]
        ),
    }
    return summary, rows


def _run_fixed_pose_hold(
    fingertip: Fingertip,
    *,
    sim_frequency_hz: float,
    shape_stiffness_n_m: float,
    shape_damping_n_s_m: float,
) -> dict[str, float | int]:
    simulation, indenter, initial_sphere_z_m = _make_simulation(
        fingertip,
        sim_frequency_hz=sim_frequency_hz,
        shape_stiffness_n_m=shape_stiffness_n_m,
        shape_damping_n_s_m=shape_damping_n_s_m,
    )
    travel_m, crossing_force_n = _approach_to_target(
        simulation,
        indenter,
        initial_sphere_z_m,
    )
    checkpoint_ticks = {
        0,
        round(5.0e-3 * simulation.sim_frequency),
        round(100.0e-3 * simulation.sim_frequency),
        round(_HOLD_DURATION_S * simulation.sim_frequency),
    }
    checkpoint_forces_n = {0: crossing_force_n}
    checkpoint_contacts = {
        0: simulation.soft_contact_count(indenter.body_index)
    }
    hold_min_force_n = crossing_force_n
    for hold_tick in range(1, max(checkpoint_ticks) + 1):
        simulation.step()
        reaction_force_n = simulation.indenter_reaction_force(
            indenter,
            motion_direction_W=_MOTION_DIRECTION_W,
        )
        hold_min_force_n = min(hold_min_force_n, reaction_force_n)
        if hold_tick in checkpoint_ticks:
            checkpoint_forces_n[hold_tick] = reaction_force_n
            checkpoint_contacts[hold_tick] = (
                simulation.soft_contact_count(indenter.body_index)
            )

    return {
        "repeat_target_depth_m": travel_m - _INITIAL_CLEARANCE_M,
        "hold_force_0_ms_n": checkpoint_forces_n[0],
        "hold_force_5_ms_n": checkpoint_forces_n[
            round(5.0e-3 * simulation.sim_frequency)
        ],
        "hold_force_100_ms_n": checkpoint_forces_n[
            round(100.0e-3 * simulation.sim_frequency)
        ],
        "hold_force_1000_ms_n": checkpoint_forces_n[
            round(_HOLD_DURATION_S * simulation.sim_frequency)
        ],
        "hold_contact_count_1000_ms": checkpoint_contacts[
            round(_HOLD_DURATION_S * simulation.sim_frequency)
        ],
        "hold_min_force_n": hold_min_force_n,
        "hold_final_max_particle_speed_m_s": (
            simulation.maximum_active_particle_speed_m_s()
        ),
    }


def _run_case(
    fingertip: Fingertip,
    *,
    sim_frequency_hz: float,
    shape_stiffness_n_m: float,
    shape_damping_n_s_m: float,
    nominal_damping_ratio: float,
) -> tuple[dict[str, float | int | bool], list[dict[str, float | int]]]:
    summary, rows = _run_force_depth(
        fingertip,
        sim_frequency_hz=sim_frequency_hz,
        shape_stiffness_n_m=shape_stiffness_n_m,
        shape_damping_n_s_m=shape_damping_n_s_m,
    )
    summary.update(
        _run_fixed_pose_hold(
            fingertip,
            sim_frequency_hz=sim_frequency_hz,
            shape_stiffness_n_m=shape_stiffness_n_m,
            shape_damping_n_s_m=shape_damping_n_s_m,
        )
    )
    summary["nominal_damping_ratio"] = nominal_damping_ratio
    summary["crossing_depth_difference_m"] = abs(
        float(summary["target_depth_m"])
        - float(summary["repeat_target_depth_m"])
    )
    summary["hard_valid"] = bool(
        int(summary["target_contact_count"]) > 0
        and float(summary["minimum_det_f"]) > 0.2
        and int(summary["maximum_inverted_tet_count"]) == 0
        and not bool(summary["force_dropout"])
        and float(summary["hold_min_force_n"]) > 1.0e-6
        and int(summary["hold_contact_count_1000_ms"]) > 0
        and int(summary["body_particle_buffer_overflow_max"]) == 0
    )
    return summary, rows


def _print_result(result: dict[str, float | int | bool]) -> None:
    print(
        f"  ke={float(result['shape_ke_n_m']):8.1e} "
        f"kd={float(result['shape_kd_n_s_m']):8.4f} "
        f"zeta={float(result['actual_damping_ratio']):5.2f} | "
        f"depth={1.0e3 * float(result['target_depth_m']):6.3f} mm "
        f"pen={1.0e6 * float(result['target_particle_penetration_m']):7.2f} um "
        f"detF={float(result['target_minimum_det_f']):6.3f} "
        f"drop={float(result['maximum_one_step_force_drop_n']):7.2f} N | "
        f"hold=[{float(result['hold_force_0_ms_n']):6.2f}, "
        f"{float(result['hold_force_5_ms_n']):6.2f}, "
        f"{float(result['hold_force_100_ms_n']):6.2f}, "
        f"{float(result['hold_force_1000_ms_n']):6.2f}] N | "
        f"{'VALID' if result['hard_valid'] else 'INVALID'}",
        flush=True,
    )


def main() -> None:
    fingertip = Fingertip(FingertipParameters())
    reference_mass_kg = _probe_contact_mass(fingertip)
    print()
    print(
        "Each tuning case sets shape ke/kd and soft_contact ke/kd to the "
        "same pair material."
    )

    primary_cases: list[tuple[float, float, float]] = []
    for shape_ke_n_m in _SHAPE_STIFFNESSES_N_M:
        critical_effective_kd_n_s_m = 2.0 * sqrt(
            shape_ke_n_m * reference_mass_kg
        )
        for damping_ratio in _DAMPING_RATIOS:
            shape_kd_n_s_m = (
                damping_ratio * critical_effective_kd_n_s_m
            )
            primary_cases.append(
                (shape_ke_n_m, shape_kd_n_s_m, damping_ratio)
            )

    summaries: list[dict[str, float | int | bool]] = []
    trajectories: list[
        tuple[dict[str, float | int | bool], list[dict[str, float | int]]]
    ] = []
    print()
    print("2 kHz ke-kd sweep:")
    for shape_ke_n_m, shape_kd_n_s_m, damping_ratio in primary_cases:
        result, rows = _run_case(
            fingertip,
            sim_frequency_hz=_PRIMARY_FREQUENCY_HZ,
            shape_stiffness_n_m=shape_ke_n_m,
            shape_damping_n_s_m=shape_kd_n_s_m,
            nominal_damping_ratio=damping_ratio,
        )
        summaries.append(result)
        trajectories.append((result, rows))
        _print_result(result)

    candidates = [result for result in summaries if result["hard_valid"]]
    candidates.sort(
        key=lambda result: (
            abs(
                float(result["hold_force_1000_ms_n"])
                - float(result["hold_force_100_ms_n"])
            ),
            float(result["maximum_one_step_force_drop_n"]),
            float(result["crossing_depth_difference_m"]),
            float(result["target_particle_penetration_m"]),
        )
    )
    selected = candidates[:2]

    print()
    print("4 kHz timestep checks:")
    if not selected:
        print("  no hard-valid 2 kHz case; timestep check skipped")
    for candidate in selected:
        result, rows = _run_case(
            fingertip,
            sim_frequency_hz=_TIMESTEP_CHECK_FREQUENCY_HZ,
            shape_stiffness_n_m=float(candidate["shape_ke_n_m"]),
            shape_damping_n_s_m=float(candidate["shape_kd_n_s_m"]),
            nominal_damping_ratio=float(
                candidate["nominal_damping_ratio"]
            ),
        )
        summaries.append(result)
        trajectories.append((result, rows))
        _print_result(result)

    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary_path = _OUTPUT_DIRECTORY / "sphere_contact_tuning.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    trajectory_path = (
        _OUTPUT_DIRECTORY / "sphere_contact_tuning_force_depth.csv"
    )
    trajectory_fields = [
        "sim_frequency_hz",
        "shape_ke_n_m",
        "nominal_damping_ratio",
        "shape_kd_n_s_m",
        *list(trajectories[0][1][0]),
    ]
    with trajectory_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=trajectory_fields)
        writer.writeheader()
        for result, rows in trajectories:
            for row in rows:
                writer.writerow(
                    {
                        "sim_frequency_hz": result["sim_frequency_hz"],
                        "shape_ke_n_m": result["shape_ke_n_m"],
                        "nominal_damping_ratio": result[
                            "nominal_damping_ratio"
                        ],
                        "shape_kd_n_s_m": result["shape_kd_n_s_m"],
                        **row,
                    }
                )

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.0), sharey=True)
    for axis, damping_ratio in zip(axes, _DAMPING_RATIOS, strict=True):
        for result, rows in trajectories:
            if float(result["nominal_damping_ratio"]) != damping_ratio:
                continue
            depth_mm = 1.0e3 * np.asarray(
                [row["indentation_depth_m"] for row in rows]
            )
            force_n = np.asarray(
                [row["reaction_force_n"] for row in rows]
            )
            frequency_hz = float(result["sim_frequency_hz"])
            linestyle = "-" if frequency_hz == 2.0e3 else "--"
            axis.plot(
                depth_mm,
                force_n,
                linestyle=linestyle,
                label=(
                    f"ke={float(result['shape_ke_n_m']):.0e}, "
                    f"{frequency_hz / 1.0e3:g} kHz"
                ),
            )
        axis.axhline(_TARGET_FORCE_N, color="black", linewidth=1.0)
        axis.set_title(f"nominal damping ratio {damping_ratio:g}")
        axis.set_xlabel("indentation depth [mm]")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("reaction force [N]")
    figure.tight_layout()
    figure_path = _OUTPUT_DIRECTORY / "sphere_contact_tuning.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print()
    print(f"summary CSV: {summary_path}")
    print(f"force-depth CSV: {trajectory_path}")
    print(f"plot: {figure_path}")


if __name__ == "__main__":
    main()
