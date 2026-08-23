"""Diagnose fixed-pose force decay after a transient 20 N trigger."""

from __future__ import annotations

import csv
from importlib.resources import as_file, files
from math import ceil
from pathlib import Path

import matplotlib
import newton
import numpy as np
import warp as wp
from shapely import contains_xy, distance, points
from shapely.geometry import Polygon

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.newton import Indenter
from lumo.simulation import LumoSimulation


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


_SPHERE_RADIUS_M = 7.5e-3
_INITIAL_CLEARANCE_M = 1.0e-3
_APPROACH_SPEED_M_S = 2.5e-2
_TARGET_FORCE_N = 20.0
_MAX_APPROACH_TIME_S = 30.0
_HOLD_DURATION_S = 10.0
_CHECKPOINT_TIMES_S = (
    0.0,
    0.005,
    0.020,
    0.050,
    0.100,
    0.500,
    1.000,
    2.000,
    5.000,
    10.000,
)
_MOTION_DIRECTION_W = wp.vec3(0.0, 0.0, 1.0)
_OUTPUT_DIRECTORY = Path("output/validation")
_CASES = (
    ("baseline_1000Hz_10iter", 1000.0, 10, False),
    ("zero_velocity_1000Hz_10iter", 1000.0, 10, True),
    ("iterations_1000Hz_30iter", 1000.0, 30, False),
    ("timestep_2000Hz_20iter", 2000.0, 20, False),
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


def _carrier_particle_penetration_m(
    positions_m: np.ndarray,
    *,
    nonbonded: np.ndarray,
    carrier_cross_section: Polygon,
    carrier_y_limits_m: tuple[float, float],
) -> float:
    y_min_m, y_max_m = carrier_y_limits_m
    inside = (
        nonbonded
        & (positions_m[:, 1] > y_min_m)
        & (positions_m[:, 1] < y_max_m)
        & contains_xy(
            carrier_cross_section,
            1.0e3 * positions_m[:, 0],
            1.0e3 * positions_m[:, 2],
        )
    )
    if not np.any(inside):
        return 0.0
    return 1.0e-3 * float(
        np.max(
            distance(
                carrier_cross_section.boundary,
                points(
                    1.0e3 * positions_m[inside, 0],
                    1.0e3 * positions_m[inside, 2],
                ),
            )
        )
    )


def _run_case(
    fingertip: Fingertip,
    urdf_path: Path,
    *,
    name: str,
    sim_frequency: float,
    iterations: int,
    zero_velocity_at_trigger: bool,
) -> dict[str, object]:
    initial_sphere_z_m = (
        fingertip.tip_z_m
        - _INITIAL_CLEARANCE_M
        - _SPHERE_RADIUS_M
    )
    initial_pose = wp.transform(
        wp.vec3(0.0, 0.0, initial_sphere_z_m),
        wp.quat_identity(),
    )
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, 0.0))
    indenter = Indenter.add_urdf(builder, urdf_path, tf=initial_pose)
    simulation = LumoSimulation(
        fingertip,
        builder=builder,
        sim_frequency=sim_frequency,
        iterations=iterations,
    )
    if simulation.soft_contact_count(indenter.body_index) != 0:
        raise RuntimeError(f"{name} has contacts before prescribed motion")

    reference_positions_m = np.asarray(
        simulation.fingertip_mesh.silicone.vertices,
        dtype=np.float64,
    )
    tet_indices = np.asarray(
        simulation.fingertip_mesh.silicone.tet_indices,
        dtype=np.int32,
    ).reshape(-1, 4)
    reference_volume_m3 = float(
        np.abs(_six_tet_volumes(reference_positions_m, tet_indices)).sum()
    )
    bonded_indices = (
        simulation.fingertip_model.bonded_particle_indices.numpy()
    )
    nonbonded = np.ones(len(reference_positions_m), dtype=bool)
    nonbonded[bonded_indices] = False
    carrier_cross_section = Polygon(fingertip.carrier.cross_section)
    carrier_vertices_m = np.asarray(
        simulation.fingertip_mesh.carrier.vertices,
        dtype=np.float64,
    )
    carrier_y_limits_m = (
        float(carrier_vertices_m[:, 1].min()),
        float(carrier_vertices_m[:, 1].max()),
    )

    position_step_m = _APPROACH_SPEED_M_S / simulation.sim_frequency
    max_approach_steps = int(
        _MAX_APPROACH_TIME_S * simulation.sim_frequency
    )
    reaction_force_n = 0.0
    print(
        f"[{name}] approach: {1.0e3 * _APPROACH_SPEED_M_S:.3f} mm/s, "
        f"dt={1.0e3 * simulation.time_step_s:.3f} ms, "
        f"iterations={iterations}",
        flush=True,
    )
    for approach_step in range(1, max_approach_steps + 1):
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
            break
    else:
        raise RuntimeError(
            f"{name} did not reach transient 20 N within "
            f"{_MAX_APPROACH_TIME_S:g} s; last force was "
            f"{reaction_force_n:.9e} N"
        )

    if zero_velocity_at_trigger:
        if simulation.state.particle_qd is None:
            raise RuntimeError(f"{name} has no silicone particle velocities")
        # Diagnostic only: all Newton particles in this model are silicone.
        simulation.state.particle_qd.zero_()

    hold_times_s: list[float] = []
    reaction_forces_n: list[float] = []
    maximum_speeds_m_s: list[float] = []
    indenter_contact_counts: list[int] = []
    carrier_contact_counts: list[int] = []
    maximum_carrier_penetrations_m: list[float] = []
    total_tet_volume_ratios: list[float] = []

    def record(hold_time_s: float, force_n: float) -> None:
        positions_m = np.asarray(
            simulation.silicone_vertices(),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(positions_m)):
            raise RuntimeError(f"{name} produced non-finite positions")
        current_volume_m3 = float(
            np.abs(_six_tet_volumes(positions_m, tet_indices)).sum()
        )
        hold_times_s.append(hold_time_s)
        reaction_forces_n.append(force_n)
        maximum_speeds_m_s.append(
            simulation.maximum_active_particle_speed_m_s()
        )
        indenter_contact_counts.append(
            simulation.soft_contact_count(indenter.body_index)
        )
        carrier_contact_counts.append(
            simulation.soft_contact_count(
                simulation.fingertip_model.carrier_body
            )
        )
        maximum_carrier_penetrations_m.append(
            _carrier_particle_penetration_m(
                positions_m,
                nonbonded=nonbonded,
                carrier_cross_section=carrier_cross_section,
                carrier_y_limits_m=carrier_y_limits_m,
            )
        )
        total_tet_volume_ratios.append(
            current_volume_m3 / reference_volume_m3
        )

    record(0.0, reaction_force_n)
    trigger_time_s = simulation.time_s
    print(
        f"[{name}] trigger: t={trigger_time_s:.6f} s | "
        f"travel={1.0e3 * travel_m:.6f} mm | "
        f"F={reaction_force_n:.9f} N | "
        f"velocity_reset={zero_velocity_at_trigger}",
        flush=True,
    )

    hold_step_count = ceil(_HOLD_DURATION_S * simulation.sim_frequency)
    checkpoint_ticks = {
        round(time_s * simulation.sim_frequency)
        for time_s in _CHECKPOINT_TIMES_S
    }
    for hold_step in range(1, hold_step_count + 1):
        # The sphere pose is not updated anywhere in this hold loop.
        simulation.step()
        reaction_force_n = simulation.indenter_reaction_force(
            indenter,
            motion_direction_W=_MOTION_DIRECTION_W,
        )
        record(hold_step * simulation.time_step_s, reaction_force_n)
        if hold_step in checkpoint_ticks:
            print(
                f"[{name}] hold="
                f"{hold_step / simulation.sim_frequency:.3f} s | "
                f"F={reaction_force_n:.6f} N | "
                f"vmax={maximum_speeds_m_s[-1]:.3e} m/s | "
                f"sphere_contacts={indenter_contact_counts[-1]} | "
                f"carrier_contacts={carrier_contact_counts[-1]}",
                flush=True,
            )

    return {
        "name": name,
        "sim_frequency_hz": simulation.sim_frequency,
        "iterations": iterations,
        "velocity_reset": zero_velocity_at_trigger,
        "trigger_time_s": trigger_time_s,
        "travel_m": travel_m,
        "hold_time_s": hold_times_s,
        "reaction_force_n": reaction_forces_n,
        "maximum_speed_m_s": maximum_speeds_m_s,
        "indenter_contact_count": indenter_contact_counts,
        "carrier_contact_count": carrier_contact_counts,
        "maximum_nonbonded_particle_carrier_penetration_m": (
            maximum_carrier_penetrations_m
        ),
        "total_tet_volume_ratio": total_tet_volume_ratios,
    }


def main() -> None:
    fingertip = Fingertip(FingertipParameters())
    sphere_resource = files("lumo").joinpath(
        "assets",
        "objects",
        "urdf",
        "sphere_15mm.urdf",
    )
    results = []
    with as_file(sphere_resource) as urdf_path:
        for name, frequency_hz, iterations, zero_velocity in _CASES:
            results.append(
                _run_case(
                    fingertip,
                    urdf_path,
                    name=name,
                    sim_frequency=frequency_hz,
                    iterations=iterations,
                    zero_velocity_at_trigger=zero_velocity,
                )
            )

    print()
    print(
        "case                            hold_ms   force_N    vmax_m_s  "
        "sphere_ct carrier_ct carrier_pen_um volume_ratio"
    )
    for result in results:
        hold_times_s = result["hold_time_s"]
        for checkpoint_time_s in _CHECKPOINT_TIMES_S:
            checkpoint_index = round(
                checkpoint_time_s
                * float(result["sim_frequency_hz"])
            )
            print(
                f"{str(result['name']):31s} "
                f"{1.0e3 * hold_times_s[checkpoint_index]:7.1f} "
                f"{result['reaction_force_n'][checkpoint_index]:9.5f} "
                f"{result['maximum_speed_m_s'][checkpoint_index]:9.3e} "
                f"{result['indenter_contact_count'][checkpoint_index]:9d} "
                f"{result['carrier_contact_count'][checkpoint_index]:10d} "
                f"{1.0e6 * result['maximum_nonbonded_particle_carrier_penetration_m'][checkpoint_index]:14.3f} "
                f"{result['total_tet_volume_ratio'][checkpoint_index]:12.8f}"
            )

    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    csv_path = _OUTPUT_DIRECTORY / "force_traj.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            (
                "case",
                "sim_frequency_hz",
                "iterations",
                "velocity_reset",
                "hold_time_s",
                "reaction_force_n",
                "maximum_speed_m_s",
                "indenter_contact_count",
                "carrier_contact_count",
                "maximum_nonbonded_particle_carrier_penetration_m",
                "total_tet_volume_ratio",
            )
        )
        for result in results:
            row_count = len(result["hold_time_s"])
            for index in range(row_count):
                writer.writerow(
                    (
                        result["name"],
                        result["sim_frequency_hz"],
                        result["iterations"],
                        result["velocity_reset"],
                        result["hold_time_s"][index],
                        result["reaction_force_n"][index],
                        result["maximum_speed_m_s"][index],
                        result["indenter_contact_count"][index],
                        result["carrier_contact_count"][index],
                        result[
                            "maximum_nonbonded_particle_carrier_penetration_m"
                        ][index],
                        result["total_tet_volume_ratio"][index],
                    )
                )

    figure, axes = plt.subplots(3, 2, figsize=(12.0, 12.0), sharex=True)
    for result in results:
        hold_time_ms = 1.0e3 * np.asarray(result["hold_time_s"])
        label = str(result["name"])
        axes[0, 0].plot(
            hold_time_ms,
            result["reaction_force_n"],
            label=label,
        )
        axes[0, 1].plot(
            hold_time_ms,
            result["maximum_speed_m_s"],
            label=label,
        )
        axes[1, 0].plot(
            hold_time_ms,
            result["indenter_contact_count"],
            label=label,
        )
        axes[1, 1].plot(
            hold_time_ms,
            result["carrier_contact_count"],
            label=label,
        )
        axes[2, 0].plot(
            hold_time_ms,
            1.0e6
            * np.asarray(
                result[
                    "maximum_nonbonded_particle_carrier_penetration_m"
                ]
            ),
            label=label,
        )
        axes[2, 1].plot(
            hold_time_ms,
            result["total_tet_volume_ratio"],
            label=label,
        )

    axes[0, 0].axhline(
        _TARGET_FORCE_N,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    axes[0, 0].set_ylabel("reaction force [N]")
    axes[0, 1].set_ylabel("maximum active speed [m/s]")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_ylabel("sphere contact count")
    axes[1, 1].set_ylabel("carrier contact count")
    axes[2, 0].set_ylabel("carrier particle penetration [um]")
    axes[2, 1].set_ylabel("total tet volume ratio")
    axes[2, 0].set_xlabel("hold time [ms]")
    axes[2, 1].set_xlabel("hold time [ms]")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Fixed-pose force-decay diagnostics")
    figure.tight_layout()
    figure_path = _OUTPUT_DIRECTORY / "force_traj.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print()
    print(f"trajectory CSV: {csv_path}")
    print(f"trajectory plot: {figure_path}")


if __name__ == "__main__":
    main()
