"""View one 15 mm sphere pressing the fingertip to a transient 20 N."""

from __future__ import annotations

from importlib.resources import as_file, files
from math import ceil

import newton
import newton.viewer
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.newton import Indenter
from lumo.simulation import LumoSimulation
from lumo.util.viewer_util import configure_fingertip_camera


_SIM_FREQUENCY_HZ = 5.0e2
_SPHERE_RADIUS_M = 7.5e-3
_INITIAL_CLEARANCE_M = 1.0e-3
_APPROACH_SPEED_M_S = 2.5e-2
_TARGET_FORCE_N = 29.0
_HOLD_DURATION_S = 10.0
_MAX_SIM_TIME_S = 30.0
_REPORT_INTERVAL_TICKS = 100
_MOTION_DIRECTION_W = wp.vec3(0.0, 0.0, 1.0)


def _render(
    viewer: newton.viewer.ViewerGL,
    simulation: LumoSimulation,
) -> None:
    viewer.begin_frame(simulation.time_s)
    viewer.log_state(simulation.state)
    viewer.log_contacts(simulation.contacts, simulation.state)
    viewer.end_frame()


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
            tf=initial_pose,
        )

    simulation = LumoSimulation(
        fingertip,
        builder=builder,
        sim_frequency=_SIM_FREQUENCY_HZ,
    )
    viewer = newton.viewer.ViewerGL(vsync=False)
    viewer.set_model(simulation.fingertip_model.model)
    configure_fingertip_camera(viewer)

    translation_step_m = _APPROACH_SPEED_M_S / simulation.sim_frequency
    max_step_count = int(_MAX_SIM_TIME_S * simulation.sim_frequency)
    reaction_force_n = 0.0

    try:
        initial_contact_count = simulation.soft_contact_count(
            indenter.body_index
        )
        if initial_contact_count != 0:
            raise RuntimeError(
                "15 mm sphere has soft contacts before prescribed motion"
            )

        _render(viewer, simulation)
        print(
            "15 mm sphere viewer: moving at "
            f"{1.0e3 * _APPROACH_SPEED_M_S:.3f} mm/s toward "
            f"{_TARGET_FORCE_N:.1f} N",
            flush=True,
        )

        for approach_step in range(1, max_step_count + 1):
            if not viewer.is_running():
                print("viewer closed before the target force was reached")
                return

            travel_m = approach_step * translation_step_m
            sphere_z_m = initial_sphere_z_m + travel_m
            simulation.apply_indenter_pose(
                indenter,
                wp.transform(
                    wp.vec3(0.0, 0.0, sphere_z_m),
                    wp.quat_identity(),
                ),
            )
            simulation.step()
            reaction_force_n = simulation.indenter_reaction_force(
                indenter,
                motion_direction_W=_MOTION_DIRECTION_W,
            )
            _render(viewer, simulation)

            if (
                approach_step % _REPORT_INTERVAL_TICKS == 0
                or reaction_force_n >= _TARGET_FORCE_N
            ):
                maximum_speed_m_s = (
                    simulation.maximum_active_particle_speed_m_s()
                )
                sphere_contact_count = simulation.soft_contact_count(
                    indenter.body_index
                )
                print(
                    f"t={simulation.time_s:7.3f} s | "
                    f"travel={1.0e3 * travel_m:7.3f} mm | "
                    f"F={reaction_force_n:9.4f} N | "
                    f"vmax={maximum_speed_m_s:9.3e} m/s | "
                    f"contacts={sphere_contact_count}",
                    flush=True,
                )

            if reaction_force_n >= _TARGET_FORCE_N:
                print(
                    f"transient 20 N target reached at "
                    f"{1.0e3 * travel_m:.4f} mm travel; "
                    f"holding the sphere pose for {_HOLD_DURATION_S:g} s.",
                    flush=True,
                )

                hold_step_count = ceil(
                    _HOLD_DURATION_S * simulation.sim_frequency
                )
                for hold_step in range(1, hold_step_count + 1):
                    if not viewer.is_running():
                        print("viewer closed during the fixed-pose hold")
                        return

                    # Do not update the sphere pose during settling.
                    simulation.step()
                    reaction_force_n = simulation.indenter_reaction_force(
                        indenter,
                        motion_direction_W=_MOTION_DIRECTION_W,
                    )
                    _render(viewer, simulation)

                    if (
                        hold_step % _REPORT_INTERVAL_TICKS == 0
                        or hold_step == hold_step_count
                    ):
                        hold_time_s = (
                            hold_step / simulation.sim_frequency
                        )
                        maximum_speed_m_s = (
                            simulation.maximum_active_particle_speed_m_s()
                        )
                        sphere_contact_count = simulation.soft_contact_count(
                            indenter.body_index
                        )
                        print(
                            f"hold={hold_time_s:7.3f} s | "
                            f"F={reaction_force_n:9.4f} N | "
                            f"vmax={maximum_speed_m_s:9.3e} m/s | "
                            f"contacts={sphere_contact_count}",
                            flush=True,
                        )

                print(
                    f"{_HOLD_DURATION_S:g} s fixed-pose hold complete; "
                    "close the viewer to finish.",
                    flush=True,
                )
                while viewer.is_running():
                    _render(viewer, simulation)
                return

        raise RuntimeError(
            "15 mm sphere did not reach the transient 20 N target within "
            f"{_MAX_SIM_TIME_S:g} s; last force was "
            f"{reaction_force_n:.9e} N"
        )
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
