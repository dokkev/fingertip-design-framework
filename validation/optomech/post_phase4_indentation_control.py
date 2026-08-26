"""Gate A for validation-local fixed-indentation quasi-static loading."""

from __future__ import annotations

import csv
from importlib.resources import as_file, files
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import newton
import numpy as np
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import make_fingertip_5led_mesh
from lumo.newton import Indenter
from lumo.optimization.evaluator import (
    _indenter_contact_records,
    _six_tet_volumes,
)
from lumo.optimization.objective import (
    _active_surface_triangles,
    _surface_incidence,
    _triangle_areas,
)
from lumo.simulation import LumoSimulation


_OUTPUT_DIRECTORY = (
    Path("output/validation/production_evaluator_acceleration")
    / "post_phase4_indentation_control"
)
_REFERENCE_DIRECTORY = (
    Path("output/validation/production_evaluator_acceleration")
    / "phase3_dwell"
    / "five_second_convergence"
)
_SPHERE_DIAMETER_MM = 20.0
_CONTACT_Y_MM = 22.0
_REFERENCE_FORCE_INDEX = 2
_OBSERVATION_TIMES_S = (5.0, 7.5, 10.0)
_WINDOW_DURATION_S = 0.5
_SIM_FREQUENCY_HZ = 100.0
_TIME_STEP_S = 1.0 / _SIM_FREQUENCY_HZ
_APPROACH_SPEED_M_S = 5.0e-3
_INITIAL_CLEARANCE_M = 1.0e-3


def _active_speeds(simulation: LumoSimulation) -> np.ndarray:
    velocities = simulation.state.particle_qd
    flags = simulation.fingertip_model.model.particle_flags
    if velocities is None or flags is None:
        raise RuntimeError("Newton state lacks particle-speed diagnostics")
    start = simulation.fingertip_model.silicone_particle_start
    stop = start + simulation.fingertip_model.silicone_particle_count
    velocity_values = velocities.numpy()[start:stop]
    flag_values = flags.numpy()[start:stop]
    active = (flag_values & int(newton.ParticleFlags.ACTIVE)) != 0
    return np.linalg.norm(velocity_values[active], axis=1)


def _patch_state(
    simulation: LumoSimulation,
    indenter: Indenter,
    vertices_m: np.ndarray,
    surface_triangles: np.ndarray,
    incidence: tuple[object, object, object],
) -> dict[str, object]:
    records = _indenter_contact_records(simulation, indenter, vertices_m)
    if len(records[0]) == 0:
        raise RuntimeError("fixed-indentation state has no indenter contact")
    support = frozenset(
        _active_surface_triangles(
            records[0],
            vertex_triangles=incidence[0],
            edge_triangles=incidence[1],
            triangle_ids=incidence[2],
        )
    )
    normal = np.mean(records[3], axis=0)
    normal /= np.linalg.norm(normal)
    return {
        "support": support,
        "area_m2": float(
            _triangle_areas(vertices_m, surface_triangles)[list(support)].sum()
        ),
        "centroid_m": np.mean(records[2], axis=0),
        "normal": normal,
        "contact_count": len(records[0]),
    }


def _weighted_iou(
    first: frozenset[int],
    second: frozenset[int],
    reference_areas_m2: np.ndarray,
) -> float:
    union = first | second
    if not union:
        return 1.0
    return float(
        reference_areas_m2[list(first & second)].sum()
        / reference_areas_m2[list(union)].sum()
    )


def _reference_data() -> tuple[float, list[dict[str, object]]]:
    rows = []
    indentation_m = np.nan
    for duration_s, label in ((5.0, "5p0"), (7.5, "7p5"), (10.0, "10p0")):
        path = _REFERENCE_DIRECTORY / f"dwell_{label}_contact_limiter.npz"
        with np.load(path) as saved:
            indentation_m = float(
                saved["indentations_m"][0, _REFERENCE_FORCE_INDEX]
            )
            rows.append(
                {
                    "time_s": duration_s,
                    "force_n": float(
                        saved["actual_forces_n"][0, _REFERENCE_FORCE_INDEX]
                    ),
                    "indentation_m": indentation_m,
                    "vertices_m": np.asarray(
                        saved["silicone_vertices_m"][0, _REFERENCE_FORCE_INDEX],
                        dtype=np.float64,
                    ),
                    "particle_rms_m_s": float(
                        saved["rms_particle_speeds_m_s"][
                            0, _REFERENCE_FORCE_INDEX
                        ]
                    ),
                    "particle_p95_m_s": float(
                        saved["particle_speed_p95_m_s"][
                            0, _REFERENCE_FORCE_INDEX
                        ]
                    ),
                    "force_window_drift_n": float(
                        saved["settle_window_force_drifts_n"][
                            0, _REFERENCE_FORCE_INDEX
                        ]
                    ),
                    "indentation_window_drift_m": float(
                        saved["settle_window_indentation_drifts_m"][
                            0, _REFERENCE_FORCE_INDEX
                        ]
                    ),
                }
            )
    return indentation_m, rows


def _run_fixed_indentation(indentation_m: float) -> dict[str, object]:
    fingertip = Fingertip(FingertipParameters())
    fingertip_mesh = make_fingertip_5led_mesh(fingertip, element_size_mm=1.0)
    reference_vertices_m = np.asarray(
        fingertip_mesh.silicone.vertices,
        dtype=np.float64,
    )
    surface_triangles = np.asarray(
        fingertip_mesh.silicone.surface_tri_indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    tet_indices = np.asarray(
        fingertip_mesh.silicone.tet_indices,
        dtype=np.int32,
    ).reshape(-1, 4)
    reference_six_volumes_m3 = _six_tet_volumes(
        reference_vertices_m,
        tet_indices,
    )
    reference_areas_m2 = _triangle_areas(
        reference_vertices_m,
        surface_triangles,
    )
    incidence = _surface_incidence(surface_triangles)
    radius_m = 0.5e-3 * _SPHERE_DIAMETER_MM
    initial_translation_m = np.array(
        (
            0.0,
            1.0e-3 * _CONTACT_Y_MM,
            fingertip.tip_z_m - _INITIAL_CLEARANCE_M - radius_m,
        ),
        dtype=np.float64,
    )
    initial_tf = wp.transform(
        wp.vec3(*initial_translation_m),
        wp.quat_identity(),
    )
    resource = files("lumo.assets.objects.urdf").joinpath("sphere_20mm.urdf")
    with as_file(resource) as sphere_path:
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, 0.0))
        indenter = Indenter.add_urdf(
            builder,
            sphere_path,
            tf=initial_tf,
            contact_stiffness_n_m=3.0e4,
            contact_damping_n_s_m=0.28228017516945547,
        )
        simulation = LumoSimulation(
            fingertip,
            builder=builder,
            fingertip_mesh=fingertip_mesh,
            sim_frequency=_SIM_FREQUENCY_HZ,
            iterations=10,
            soft_contact_margin_m=1.0e-4,
            soft_contact_stiffness_n_m=3.0e4,
            soft_contact_damping_n_s_m=0.28228017516945547,
            element_size_mm=1.0,
            carrier_contact_stiffness_n_m=1.0e6,
            use_cuda_graph=False,
        )

        total_travel_m = _INITIAL_CLEARANCE_M + indentation_m
        travel_m = 0.0
        approach_indentation_m = []
        approach_force_n = []
        start_s = perf_counter()
        while travel_m < total_travel_m:
            travel_m = min(
                total_travel_m,
                travel_m + _APPROACH_SPEED_M_S * _TIME_STEP_S,
            )
            translation_m = initial_translation_m + np.array(
                (0.0, 0.0, travel_m)
            )
            simulation.apply_indenter_pose(
                indenter,
                wp.transform(wp.vec3(*translation_m), wp.quat_identity()),
            )
            simulation.step()
            approach_indentation_m.append(
                max(0.0, travel_m - _INITIAL_CLEARANCE_M)
            )
            approach_force_n.append(
                simulation.indenter_reaction_force(
                    indenter,
                    motion_direction_W=wp.vec3(0.0, 0.0, 1.0),
                )
            )

        max_hold_ticks = round(max(_OBSERVATION_TIMES_S) * _SIM_FREQUENCY_HZ)
        observation_ticks = {
            round(time_s * _SIM_FREQUENCY_HZ): time_s
            for time_s in _OBSERVATION_TIMES_S
        }
        window_ticks = round(_WINDOW_DURATION_S * _SIM_FREQUENCY_HZ)
        window_start_ticks = {
            tick - window_ticks: tick for tick in observation_ticks
        }
        force_history_n = np.empty(max_hold_ticks + 1, dtype=np.float64)
        rms_speed_history_m_s = np.empty(max_hold_ticks + 1, dtype=np.float64)
        p95_speed_history_m_s = np.empty(max_hold_ticks + 1, dtype=np.float64)
        vertices_at_window_start: dict[int, np.ndarray] = {}
        observations: list[dict[str, object]] = []

        force_history_n[0] = approach_force_n[-1]
        speeds = _active_speeds(simulation)
        rms_speed_history_m_s[0] = np.sqrt(np.mean(speeds**2))
        p95_speed_history_m_s[0] = np.percentile(speeds, 95.0)
        for hold_tick in range(1, max_hold_ticks + 1):
            simulation.step()
            force_history_n[hold_tick] = simulation.indenter_reaction_force(
                indenter,
                motion_direction_W=wp.vec3(0.0, 0.0, 1.0),
            )
            speeds = _active_speeds(simulation)
            rms_speed_history_m_s[hold_tick] = np.sqrt(np.mean(speeds**2))
            p95_speed_history_m_s[hold_tick] = np.percentile(speeds, 95.0)
            if hold_tick in window_start_ticks:
                vertices_at_window_start[window_start_ticks[hold_tick]] = (
                    simulation.silicone_vertices().astype(np.float64)
                )
            if hold_tick not in observation_ticks:
                continue

            vertices_m = simulation.silicone_vertices().astype(np.float64)
            patch = _patch_state(
                simulation,
                indenter,
                vertices_m,
                surface_triangles,
                incidence,
            )
            det_f = (
                _six_tet_volumes(vertices_m, tet_indices)
                / reference_six_volumes_m3
            )
            window_slice = slice(hold_tick - window_ticks, hold_tick + 1)
            window_forces = force_history_n[window_slice]
            window_start_vertices = vertices_at_window_start[hold_tick]
            vertex_delta = vertices_m - window_start_vertices
            observations.append(
                {
                    "time_s": observation_ticks[hold_tick],
                    "force_n": force_history_n[hold_tick],
                    "force_window_mean_n": float(np.mean(window_forces)),
                    "force_window_drift_n": float(
                        window_forces[-1] - window_forces[0]
                    ),
                    "force_window_std_n": float(np.std(window_forces)),
                    "vertex_window_rms_m": float(
                        np.sqrt(np.mean(vertex_delta**2))
                    ),
                    "vertex_window_max_m": float(np.max(np.abs(vertex_delta))),
                    "particle_rms_m_s": rms_speed_history_m_s[hold_tick],
                    "particle_p95_m_s": p95_speed_history_m_s[hold_tick],
                    "patch": patch,
                    "minimum_det_f": float(np.min(det_f)),
                    "inverted_tet_count": int(np.count_nonzero(det_f <= 0.0)),
                    "contact_buffer_overflow": int(
                        simulation.solver.body_particle_contact_overflow_max.numpy()[
                            0
                        ]
                    ),
                    "vertices_m": vertices_m,
                }
            )
        wall_s = perf_counter() - start_s

    np.savez_compressed(
        _OUTPUT_DIRECTORY / "fixed_indentation_gate_a.npz",
        indentation_m=np.asarray(indentation_m),
        approach_indentation_m=np.asarray(approach_indentation_m),
        approach_force_n=np.asarray(approach_force_n),
        hold_time_s=np.arange(max_hold_ticks + 1) * _TIME_STEP_S,
        hold_force_n=force_history_n,
        hold_particle_rms_speed_m_s=rms_speed_history_m_s,
        hold_particle_p95_speed_m_s=p95_speed_history_m_s,
        observation_times_s=np.asarray(
            [row["time_s"] for row in observations]
        ),
        observation_forces_n=np.asarray(
            [row["force_n"] for row in observations]
        ),
        observation_vertices_m=np.asarray(
            [row["vertices_m"] for row in observations],
            dtype=np.float32,
        ),
        observation_patch_support=np.asarray(
            [
                ",".join(str(index) for index in sorted(row["patch"]["support"]))
                for row in observations
            ]
        ),
        surface_triangles=surface_triangles,
        reference_vertices_m=reference_vertices_m,
        wall_s=np.asarray(wall_s),
    )
    return {
        "observations": observations,
        "approach_indentation_m": np.asarray(approach_indentation_m),
        "approach_force_n": np.asarray(approach_force_n),
        "hold_time_s": np.arange(max_hold_ticks + 1) * _TIME_STEP_S,
        "hold_force_n": force_history_n,
        "hold_particle_rms_speed_m_s": rms_speed_history_m_s,
        "hold_particle_p95_speed_m_s": p95_speed_history_m_s,
        "reference_areas_m2": reference_areas_m2,
        "wall_s": wall_s,
    }


def _write_outputs(
    fixed: dict[str, object],
    reference_rows: list[dict[str, object]],
    indentation_m: float,
) -> bool:
    observations = fixed["observations"]
    areas = fixed["reference_areas_m2"]
    fixed_interval = []
    servo_interval = []
    for first, second in zip(observations[:-1], observations[1:], strict=True):
        delta = second["vertices_m"] - first["vertices_m"]
        fixed_interval.append(
            {
                "force_change_n": abs(second["force_n"] - first["force_n"]),
                "vertex_rms_m": float(np.sqrt(np.mean(delta**2))),
                "vertex_max_m": float(np.max(np.abs(delta))),
                "patch_area_change_m2": abs(
                    second["patch"]["area_m2"] - first["patch"]["area_m2"]
                ),
                "patch_iou": _weighted_iou(
                    first["patch"]["support"],
                    second["patch"]["support"],
                    areas,
                ),
            }
        )
    for first, second in zip(
        reference_rows[:-1], reference_rows[1:], strict=True
    ):
        delta = second["vertices_m"] - first["vertices_m"]
        servo_interval.append(
            {
                "force_change_n": abs(second["force_n"] - first["force_n"]),
                "vertex_rms_m": float(np.sqrt(np.mean(delta**2))),
                "vertex_max_m": float(np.max(np.abs(delta))),
            }
        )

    safety_pass = all(
        row["minimum_det_f"] > 0.0
        and row["inverted_tet_count"] == 0
        and row["contact_buffer_overflow"] == 0
        for row in observations
    )
    decay_pass = (
        fixed_interval[1]["force_change_n"] < fixed_interval[0]["force_change_n"]
        and fixed_interval[1]["vertex_rms_m"] < fixed_interval[0]["vertex_rms_m"]
        and fixed_interval[1]["patch_area_change_m2"]
        < fixed_interval[0]["patch_area_change_m2"]
        and abs(observations[-1]["force_window_drift_n"])
        < abs(observations[0]["force_window_drift_n"])
        and observations[-1]["particle_p95_m_s"]
        < observations[0]["particle_p95_m_s"]
    )
    topology_pass = min(row["patch_iou"] for row in fixed_interval) >= 0.95
    gate_a_pass = safety_pass and decay_pass and topology_pass

    with (_OUTPUT_DIRECTORY / "summary.csv").open("w", newline="") as stream:
        fieldnames = (
            "protocol",
            "observation_time_s",
            "force_n",
            "force_window_mean_n",
            "force_window_drift_n",
            "force_window_std_n",
            "vertex_window_rms_um",
            "vertex_window_max_um",
            "particle_rms_m_s",
            "particle_p95_m_s",
            "patch_area_mm2",
            "contact_count",
            "minimum_det_f",
            "inverted_tet_count",
            "contact_buffer_overflow",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in observations:
            writer.writerow(
                {
                    "protocol": "fixed_indentation",
                    "observation_time_s": row["time_s"],
                    "force_n": row["force_n"],
                    "force_window_mean_n": row["force_window_mean_n"],
                    "force_window_drift_n": row["force_window_drift_n"],
                    "force_window_std_n": row["force_window_std_n"],
                    "vertex_window_rms_um": 1.0e6
                    * row["vertex_window_rms_m"],
                    "vertex_window_max_um": 1.0e6
                    * row["vertex_window_max_m"],
                    "particle_rms_m_s": row["particle_rms_m_s"],
                    "particle_p95_m_s": row["particle_p95_m_s"],
                    "patch_area_mm2": 1.0e6 * row["patch"]["area_m2"],
                    "contact_count": row["patch"]["contact_count"],
                    "minimum_det_f": row["minimum_det_f"],
                    "inverted_tet_count": row["inverted_tet_count"],
                    "contact_buffer_overflow": row["contact_buffer_overflow"],
                }
            )

    plt.figure(figsize=(7.0, 4.2))
    plt.plot(
        1.0e3 * fixed["approach_indentation_m"],
        fixed["approach_force_n"],
    )
    plt.xlabel("prescribed indentation [mm]")
    plt.ylabel("reaction force [N]")
    plt.title("Dynamic approach trace (not relaxed F(delta))")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(_OUTPUT_DIRECTORY / "force_vs_indentation.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.2))
    plt.plot(fixed["hold_time_s"], fixed["hold_force_n"])
    plt.scatter(
        [row["time_s"] for row in observations],
        [row["force_n"] for row in observations],
        color="tab:red",
        zorder=3,
    )
    plt.xlabel("fixed-pose hold time [s]")
    plt.ylabel("reaction force [N]")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(_OUTPUT_DIRECTORY / "force_drift_vs_hold_time.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.2))
    plt.semilogy(
        fixed["hold_time_s"],
        fixed["hold_particle_rms_speed_m_s"],
        label="RMS",
    )
    plt.semilogy(
        fixed["hold_time_s"],
        fixed["hold_particle_p95_speed_m_s"],
        label="P95",
    )
    plt.xlabel("fixed-pose hold time [s]")
    plt.ylabel("active particle speed [m/s]")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(_OUTPUT_DIRECTORY / "particle_speed_vs_time.png", dpi=180)
    plt.close()

    lines = [
        "# Post-Phase 4 indentation-control study — Gate A",
        "",
        f"Result: {'PASS' if gate_a_pass else 'FAIL'}",
        "",
        "This validation-local test prescribed the 5 s force-servo reference's "
        "15 N indentation, then held the 20 mm sphere at Y=+22 mm exactly fixed "
        "for 10 s. Production loading code and objectives were not changed.",
        "",
        f"- prescribed indentation: {1.0e3 * indentation_m:.6f} mm",
        f"- direct validation wall time: {fixed['wall_s']:.3f} s",
        "",
        "| hold time | force | 0.5 s force mean/drift/std | 0.5 s vertex RMS/max | particle RMS/P95 | patch area | contact count | min det(F) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in observations:
        lines.append(
            f"| {row['time_s']:.1f} s | {row['force_n']:.6f} N | "
            f"{row['force_window_mean_n']:.6f} / {row['force_window_drift_n']:+.6f} / {row['force_window_std_n']:.6f} N | "
            f"{1.0e6 * row['vertex_window_rms_m']:.3f} / {1.0e6 * row['vertex_window_max_m']:.3f} um | "
            f"{row['particle_rms_m_s']:.3e} / {row['particle_p95_m_s']:.3e} m/s | "
            f"{1.0e6 * row['patch']['area_m2']:.3f} mm2 | "
            f"{row['patch']['contact_count']} | {row['minimum_det_f']:.6f} |"
        )
    lines.extend(("", "## Interval decay", ""))
    for index, label in enumerate(("5 to 7.5 s", "7.5 to 10 s")):
        row = fixed_interval[index]
        lines.append(
            f"- {label}: force change={row['force_change_n']:.6f} N, "
            f"vertex RMS/max={1.0e6 * row['vertex_rms_m']:.3f}/"
            f"{1.0e6 * row['vertex_max_m']:.3f} um, patch-area change="
            f"{1.0e6 * row['patch_area_change_m2']:.3f} mm2, "
            f"patch IoU={row['patch_iou']:.6f}"
        )
    lines.extend(("", "## Force-servo reference interval context", ""))
    for index, label in enumerate(("5 to 7.5 s", "7.5 to 10 s")):
        row = servo_interval[index]
        lines.append(
            f"- {label}: force change={row['force_change_n']:.6f} N, "
            f"vertex RMS/max={1.0e6 * row['vertex_rms_m']:.3f}/"
            f"{1.0e6 * row['vertex_max_m']:.3f} um"
        )
    lines.extend(
        (
            "",
            "## Gate A decision",
            "",
            f"- decaying stationary diagnostics: {'PASS' if decay_pass else 'FAIL'}",
            f"- patch support IoU >= 0.95: {'PASS' if topology_pass else 'FAIL'}",
            f"- inversion/contact-buffer safety: {'PASS' if safety_pass else 'FAIL'}",
            f"- continue to adaptive force-conditioned extraction: {'YES' if gate_a_pass else 'NO'}",
            "",
            "The approach plot is a dynamic prescribed-motion trace and is not "
            "claimed as a converged force-indentation curve.",
        )
    )
    (_OUTPUT_DIRECTORY / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return gate_a_pass


def main() -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    indentation_m, reference_rows = _reference_data()
    fixed = _run_fixed_indentation(indentation_m)
    gate_a_pass = _write_outputs(fixed, reference_rows, indentation_m)
    print((_OUTPUT_DIRECTORY / "report.md").read_text(encoding="utf-8"))
    if not gate_a_pass:
        print("Gate A failed; adaptive F(delta) extraction was not implemented.")


if __name__ == "__main__":
    main()
