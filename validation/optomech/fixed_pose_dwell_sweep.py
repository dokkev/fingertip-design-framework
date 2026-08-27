"""Sweep fixed-pose hold time after first force-threshold crossings."""

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
from lumo.mesh import make_fingertip_mesh
from lumo.newton import Indenter
from lumo.optimization.evaluator import _six_tet_volumes
from lumo.simulation import LumoSimulation


_OUTPUT_DIRECTORY = Path("output/validation/fixed_pose_dwell_sweep")
_SCENARIOS = (
    ("sphere_20mm_y+22mm", "sphere_20mm.urdf", 20.0, 22.0),
    ("sphere_10mm_y+11mm", "sphere_10mm.urdf", 10.0, 11.0),
    ("sphere_10mm_y+22mm", "sphere_10mm.urdf", 10.0, 22.0),
    ("sphere_15mm_y+0mm", "sphere_15mm.urdf", 15.0, 0.0),
)
_FORCE_THRESHOLDS_N = (5.0, 10.0, 15.0, 20.0)
_HOLD_TIMES_S = (0.5, 1.0, 2.0, 5.0, 7.5, 10.0)
_SIM_FREQUENCY_HZ = 100.0
_TIME_STEP_S = 1.0 / _SIM_FREQUENCY_HZ
_APPROACH_SPEED_M_S = 5.0e-3
_INITIAL_CLEARANCE_M = 1.0e-3
_FORCE_WINDOW_S = 0.5


def _run_scenario(
    fingertip: Fingertip,
    fingertip_mesh: object,
    scenario: tuple[str, str, float, float],
    tet_indices: np.ndarray,
    reference_six_volumes_m3: np.ndarray,
) -> tuple[list[dict[str, float | int | str]], float]:
    name, urdf_name, diameter_mm, contact_y_mm = scenario
    radius_m = 0.5e-3 * diameter_mm
    initial_translation_m = np.asarray(
        (
            0.0,
            1.0e-3 * contact_y_mm,
            fingertip.tip_z_m - _INITIAL_CLEARANCE_M - radius_m,
        ),
        dtype=np.float64,
    )
    initial_tf = wp.transform(
        wp.vec3(*initial_translation_m),
        wp.quat_identity(),
    )
    resource = files("lumo.assets.objects.urdf").joinpath(urdf_name)
    start_s = perf_counter()
    rows: list[dict[str, float | int | str]] = []

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
        )

        travel_m = 0.0
        reaction_force_n = 0.0
        step_m = _APPROACH_SPEED_M_S * _TIME_STEP_S
        max_hold_ticks = round(max(_HOLD_TIMES_S) * _SIM_FREQUENCY_HZ)
        observation_ticks = {
            round(time_s * _SIM_FREQUENCY_HZ): time_s
            for time_s in _HOLD_TIMES_S
        }
        window_ticks = round(_FORCE_WINDOW_S * _SIM_FREQUENCY_HZ)

        for threshold_n in _FORCE_THRESHOLDS_N:
            while reaction_force_n < threshold_n:
                travel_m += step_m
                if travel_m > 12.0e-3:
                    raise RuntimeError(
                        f"{name} did not cross {threshold_n:g} N within "
                        "11 mm indentation"
                    )
                translation_m = initial_translation_m + np.asarray(
                    (0.0, 0.0, travel_m),
                    dtype=np.float64,
                )
                simulation.apply_indenter_pose(
                    indenter,
                    wp.transform(
                        wp.vec3(*translation_m),
                        wp.quat_identity(),
                    ),
                )
                simulation.step()
                reaction_force_n = simulation.indenter_reaction_force(
                    indenter,
                    motion_direction_W=wp.vec3(0.0, 0.0, 1.0),
                )

            indentation_m = max(0.0, travel_m - _INITIAL_CLEARANCE_M)
            trigger_force_n = reaction_force_n
            if simulation.state.body_q is None:
                raise RuntimeError("Newton state has no rigid-body poses")
            fixed_pose = np.asarray(
                simulation.state.body_q.numpy()[indenter.body_index],
                dtype=np.float64,
            )
            force_history_n = np.empty(max_hold_ticks + 1, dtype=np.float64)
            force_history_n[0] = trigger_force_n

            for hold_tick in range(1, max_hold_ticks + 1):
                simulation.step()
                reaction_force_n = simulation.indenter_reaction_force(
                    indenter,
                    motion_direction_W=wp.vec3(0.0, 0.0, 1.0),
                )
                force_history_n[hold_tick] = reaction_force_n
                if hold_tick not in observation_ticks:
                    continue

                pose = np.asarray(
                    simulation.state.body_q.numpy()[indenter.body_index],
                    dtype=np.float64,
                )
                vertices_m = simulation.silicone_vertices().astype(np.float64)
                det_f = (
                    _six_tet_volumes(vertices_m, tet_indices)
                    / reference_six_volumes_m3
                )
                start_tick = max(0, hold_tick - window_ticks)
                window_forces_n = force_history_n[start_tick : hold_tick + 1]
                rows.append(
                    {
                        "scenario": name,
                        "sphere_diameter_mm": diameter_mm,
                        "contact_y_mm": contact_y_mm,
                        "force_threshold_n": threshold_n,
                        "trigger_force_n": trigger_force_n,
                        "indentation_mm": 1.0e3 * indentation_m,
                        "hold_time_s": observation_ticks[hold_tick],
                        "snapshot_force_n": reaction_force_n,
                        "force_change_from_trigger_n": (
                            reaction_force_n - trigger_force_n
                        ),
                        "force_window_drift_n": (
                            window_forces_n[-1] - window_forces_n[0]
                        ),
                        "force_window_std_n": float(np.std(window_forces_n)),
                        "maximum_pose_error": float(
                            np.max(np.abs(pose - fixed_pose))
                        ),
                        "minimum_det_f": float(np.min(det_f)),
                        "inverted_tet_count": int(
                            np.count_nonzero(det_f <= 0.0)
                        ),
                        "contact_buffer_overflow": int(
                            simulation.solver.body_particle_contact_overflow_max.numpy()[
                                0
                            ]
                        ),
                    }
                )

    return rows, perf_counter() - start_s


def _add_reference_differences(
    rows: list[dict[str, float | int | str]],
) -> None:
    reference_force = {
        (str(row["scenario"]), float(row["force_threshold_n"])): float(
            row["snapshot_force_n"]
        )
        for row in rows
        if float(row["hold_time_s"]) == max(_HOLD_TIMES_S)
    }
    for row in rows:
        reference_n = reference_force[
            (str(row["scenario"]), float(row["force_threshold_n"]))
        ]
        difference_n = float(row["snapshot_force_n"]) - reference_n
        row["force_difference_from_10s_n"] = difference_n
        row["force_relative_difference_from_10s"] = (
            abs(difference_n) / abs(reference_n)
        )


def _write_outputs(
    rows: list[dict[str, float | int | str]],
    scenario_runtimes_s: dict[str, float],
) -> None:
    _add_reference_differences(rows)
    fieldnames = tuple(rows[0])
    with (_OUTPUT_DIRECTORY / "fixed_pose_dwell_sweep.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    duration_summary = []
    for hold_time_s in _HOLD_TIMES_S:
        selected = [
            row for row in rows if float(row["hold_time_s"]) == hold_time_s
        ]
        differences_n = np.asarray(
            [float(row["force_difference_from_10s_n"]) for row in selected]
        )
        duration_summary.append(
            {
                "hold_time_s": hold_time_s,
                "force_difference_rms_n": float(
                    np.sqrt(np.mean(differences_n**2))
                ),
                "force_difference_max_n": float(
                    np.max(np.abs(differences_n))
                ),
                "force_relative_difference_max_percent": 100.0
                * max(
                    float(row["force_relative_difference_from_10s"])
                    for row in selected
                ),
                "force_window_drift_max_n": max(
                    abs(float(row["force_window_drift_n"])) for row in selected
                ),
                "force_window_std_max_n": max(
                    float(row["force_window_std_n"]) for row in selected
                ),
                "maximum_pose_error": max(
                    float(row["maximum_pose_error"]) for row in selected
                ),
                "minimum_det_f": min(
                    float(row["minimum_det_f"]) for row in selected
                ),
                "inverted_tet_count": sum(
                    int(row["inverted_tet_count"]) for row in selected
                ),
                "contact_buffer_overflow": max(
                    int(row["contact_buffer_overflow"]) for row in selected
                ),
            }
        )

    with (_OUTPUT_DIRECTORY / "duration_summary.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(duration_summary[0]))
        writer.writeheader()
        writer.writerows(duration_summary)

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True)
    for axis, threshold_n in zip(axes.flat, _FORCE_THRESHOLDS_N, strict=True):
        for scenario in _SCENARIOS:
            name = scenario[0]
            selected = [
                row
                for row in rows
                if row["scenario"] == name
                and float(row["force_threshold_n"]) == threshold_n
            ]
            axis.plot(
                [float(row["hold_time_s"]) for row in selected],
                [float(row["snapshot_force_n"]) for row in selected],
                marker="o",
                label=name,
            )
        axis.set_title(f"first crossing {threshold_n:g} N")
        axis.set_ylabel("snapshot force [N]")
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("fixed-pose hold N [s]")
    axes[0, 0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(_OUTPUT_DIRECTORY / "force_vs_hold_time.png", dpi=180)
    plt.close(figure)

    plt.figure(figsize=(7.0, 4.2))
    plt.plot(
        [row["hold_time_s"] for row in duration_summary],
        [row["force_difference_max_n"] for row in duration_summary],
        marker="o",
        label="max",
    )
    plt.plot(
        [row["hold_time_s"] for row in duration_summary],
        [row["force_difference_rms_n"] for row in duration_summary],
        marker="o",
        label="RMS",
    )
    plt.xlabel("fixed-pose hold N [s]")
    plt.ylabel("force difference from 10 s [N]")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(_OUTPUT_DIRECTORY / "force_convergence_to_10s.png", dpi=180)
    plt.close()

    lines = [
        "# Fixed-pose dwell-time sweep",
        "",
        "For each sentinel scenario, the indenter moved monotonically at "
        "5 mm/s until the first crossing of each 5/10/15/20 N threshold. "
        "The indenter pose was then held exactly fixed for 10 s. The listed "
        "candidate N values are samples of that same relaxation trajectory; "
        "no force servo was active during a hold.",
        "",
        f"- scenarios: {len(_SCENARIOS)}",
        f"- hold candidates: {list(_HOLD_TIMES_S)} s",
        f"- physics: {_SIM_FREQUENCY_HZ:g} Hz, 10 VBD iterations",
        f"- total wall time: {sum(scenario_runtimes_s.values()):.3f} s",
        "",
        "## Convergence to the 10 s snapshot",
        "",
        "| N [s] | force RMS difference [N] | force max difference [N] | max relative difference | max last-0.5 s drift [N] | max last-0.5 s std [N] |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in duration_summary:
        lines.append(
            f"| {row['hold_time_s']:.1f} | "
            f"{row['force_difference_rms_n']:.6f} | "
            f"{row['force_difference_max_n']:.6f} | "
            f"{row['force_relative_difference_max_percent']:.3f}% | "
            f"{row['force_window_drift_max_n']:.6f} | "
            f"{row['force_window_std_max_n']:.6f} |"
        )

    lines.extend(
        (
            "",
            "## Threshold crossing and 10 s snapshot",
            "",
            "| scenario | threshold [N] | trigger force [N] | indentation [mm] | force after 5 s [N] | force after 10 s [N] | 5-to-10 s change [N] |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for scenario in _SCENARIOS:
        for threshold_n in _FORCE_THRESHOLDS_N:
            selected = [
                row
                for row in rows
                if row["scenario"] == scenario[0]
                and float(row["force_threshold_n"]) == threshold_n
            ]
            five = next(
                row for row in selected if float(row["hold_time_s"]) == 5.0
            )
            ten = next(
                row for row in selected if float(row["hold_time_s"]) == 10.0
            )
            lines.append(
                f"| {scenario[0]} | {threshold_n:.0f} | "
                f"{float(ten['trigger_force_n']):.6f} | "
                f"{float(ten['indentation_mm']):.6f} | "
                f"{float(five['snapshot_force_n']):.6f} | "
                f"{float(ten['snapshot_force_n']):.6f} | "
                f"{float(ten['snapshot_force_n']) - float(five['snapshot_force_n']):+.6f} |"
            )

    safety_pass = all(
        float(row["maximum_pose_error"]) == 0.0
        and float(row["minimum_det_f"]) > 0.0
        and int(row["inverted_tet_count"]) == 0
        and int(row["contact_buffer_overflow"]) == 0
        for row in rows
    )
    lines.extend(
        (
            "",
            "## Safety and interpretation",
            "",
            f"- exact fixed pose, finite positive det(F), no inversion, and no contact-buffer overflow: {'PASS' if safety_pass else 'FAIL'}",
            "- indentation is constant within each hold by construction; the "
            "table reports the first-crossing pose used for every N sample.",
            "- no automatic acceptance tolerance is encoded. The shortest "
            "appropriate N must be chosen from the measured force convergence "
            "rather than by changing mechanics to force a pass.",
        )
    )
    (_OUTPUT_DIRECTORY / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fingertip = Fingertip(FingertipParameters())
    fingertip_mesh = make_fingertip_mesh(fingertip, element_size_mm=1.0)
    reference_vertices_m = np.asarray(
        fingertip_mesh.silicone.vertices,
        dtype=np.float64,
    )
    tet_indices = np.asarray(
        fingertip_mesh.silicone.tet_indices,
        dtype=np.int32,
    ).reshape(-1, 4)
    reference_six_volumes_m3 = _six_tet_volumes(
        reference_vertices_m,
        tet_indices,
    )

    rows: list[dict[str, float | int | str]] = []
    scenario_runtimes_s: dict[str, float] = {}
    for scenario in _SCENARIOS:
        scenario_rows, runtime_s = _run_scenario(
            fingertip,
            fingertip_mesh,
            scenario,
            tet_indices,
            reference_six_volumes_m3,
        )
        rows.extend(scenario_rows)
        scenario_runtimes_s[scenario[0]] = runtime_s
        print(
            f"{scenario[0]}: {runtime_s:.3f} s, "
            f"{len(scenario_rows)} snapshots"
        )

    _write_outputs(rows, scenario_runtimes_s)
    print((_OUTPUT_DIRECTORY / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
