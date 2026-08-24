"""Move the flat-plate URDF until a transient contact force is reached."""

from __future__ import annotations

import sys
from importlib.resources import as_file, files

import newton
import newton.viewer
import numpy as np
import warp as wp
from shapely import contains_xy, distance, points
from shapely.geometry import Polygon

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.newton import Indenter
from lumo.simulation import LumoSimulation
from lumo.util.viewer_util import configure_fingertip_camera


_SIM_FREQUENCY_HZ = 1.0e3
_PLATE_STEP_M = 2.5e-5
_INITIAL_CLEARANCE_M = 1.0e-3
_PLATE_HALF_THICKNESS_M = 1.0e-3
_MAX_SIM_TIME_S = 30.0
_FORCE_THRESHOLD_N = 15.0
# One percent of the default 1 mm silicone mesh spacing. This rejects the
# millimetre-scale regression while allowing the VBD contact penalty to settle.
_MAX_CARRIER_PENETRATION_M = 1.0e-5
_MAX_BONDED_DRIFT_M = 1.0e-8


def _carrier_interior_depths_m(
    positions_m: np.ndarray,
    *,
    carrier_cross_section: Polygon,
    carrier_y_limits_m: tuple[float, float],
) -> np.ndarray:
    """Return cross-section penetration depth for points inside the carrier."""
    y_min_m, y_max_m = carrier_y_limits_m
    inside = (
        (positions_m[:, 1] > y_min_m)
        & (positions_m[:, 1] < y_max_m)
        & contains_xy(
            carrier_cross_section,
            1.0e3 * positions_m[:, 0],
            1.0e3 * positions_m[:, 2],
        )
    )
    depths_m = np.zeros(positions_m.shape[0], dtype=np.float64)
    if np.any(inside):
        depths_m[inside] = 1.0e-3 * distance(
            carrier_cross_section.boundary,
            points(
                1.0e3 * positions_m[inside, 0],
                1.0e3 * positions_m[inside, 2],
            ),
        )
    return depths_m


def _render(
    viewer: newton.viewer.ViewerGL,
    simulation: LumoSimulation,
) -> None:
    viewer.begin_frame(simulation.time_s)
    viewer.log_state(simulation.state)
    viewer.log_contacts(simulation.contacts, simulation.state)
    viewer.end_frame()


def main(*, show_viewer: bool = False) -> None:
    fingertip = Fingertip(FingertipParameters())
    initial_plate_z_m = (
        fingertip.tip_z_m
        - _INITIAL_CLEARANCE_M
        - _PLATE_HALF_THICKNESS_M
    )
    initial_plate_pose = wp.transform(
        wp.vec3(0.0, 0.0, initial_plate_z_m),
        wp.quat_identity(),
    )

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, 0.0))

    flat_plate_resource = files("lumo").joinpath(
        "assets",
        "objects",
        "flat_plate.urdf",
    )
    with as_file(flat_plate_resource) as urdf_path:
        indenter = Indenter.add_urdf(
            builder,
            urdf_path,
            tf=initial_plate_pose,
        )
    simulation = LumoSimulation(
        fingertip,
        builder=builder,
        sim_frequency=_SIM_FREQUENCY_HZ,
    )
    reference_positions = simulation.state.particle_q.numpy().copy()
    bonded_indices = (
        simulation.fingertip_model.bonded_particle_indices.numpy()
    )
    nonbonded = np.ones(reference_positions.shape[0], dtype=bool)
    nonbonded[bonded_indices] = False
    surface_indices = np.unique(
        np.asarray(
            simulation.fingertip_mesh.silicone.surface_tri_indices,
            dtype=np.int32,
        ).reshape(-1)
    )
    surface_indices = surface_indices[nonbonded[surface_indices]]
    tet_indices = np.asarray(
        simulation.fingertip_mesh.silicone.tet_indices,
        dtype=np.int32,
    ).reshape(-1, 4)
    carrier_cross_section = Polygon(fingertip.carrier.cross_section)
    carrier_vertices = np.asarray(
        simulation.fingertip_mesh.carrier.vertices,
        dtype=np.float64,
    )
    carrier_y_limits_m = (
        float(carrier_vertices[:, 1].min()),
        float(carrier_vertices[:, 1].max()),
    )

    viewer = None
    if show_viewer:
        viewer = newton.viewer.ViewerGL(vsync=True)
        viewer.set_model(simulation.fingertip_model.model)
        configure_fingertip_camera(viewer)

    try:
        initial_plate_contact_count = simulation.soft_contact_count(
            indenter.body_index
        )
        initial_carrier_contact_count = simulation.soft_contact_count(
            simulation.fingertip_model.carrier_body
        )
        if initial_plate_contact_count != 0:
            raise RuntimeError(
                "flat plate has soft contacts before prescribed motion"
            )
        reference_carrier_depths_m = _carrier_interior_depths_m(
            reference_positions,
            carrier_cross_section=carrier_cross_section,
            carrier_y_limits_m=carrier_y_limits_m,
        )
        reference_carrier_depths_m[~nonbonded] = 0.0
        if np.any(reference_carrier_depths_m > 0.0):
            raise RuntimeError(
                "nonbonded silicone starts inside the carrier"
            )

        if viewer is not None and viewer.is_running():
            _render(viewer, simulation)

        max_sim_steps = int(_MAX_SIM_TIME_S * simulation.sim_frequency)
        if max_sim_steps < 1:
            raise ValueError(
                "maximum simulation time must include at least one tick"
            )
        for approach_step in range(1, max_sim_steps + 1):
            plate_z_m = initial_plate_z_m + approach_step * _PLATE_STEP_M
            simulation.apply_indenter_pose(
                indenter,
                wp.transform(
                    wp.vec3(0.0, 0.0, plate_z_m),
                    wp.quat_identity(),
                ),
            )
            simulation.step()

            reaction_force_n = simulation.indenter_reaction_force(
                indenter,
                motion_direction_W=wp.vec3(0.0, 0.0, _PLATE_STEP_M),
            )
            if viewer is not None and viewer.is_running():
                _render(viewer, simulation)
            if reaction_force_n >= _FORCE_THRESHOLD_N:
                break
        else:
            raise RuntimeError(
                "flat plate reached its maximum simulation time "
                f"({_MAX_SIM_TIME_S:.9e} s) without reaching the transient "
                "force threshold; last force was "
                f"{reaction_force_n:.9e} N"
            )

        contact_count = int(
            simulation.contacts.soft_contact_count.numpy()[0]
        )
        plate_contact_count = simulation.soft_contact_count(
            indenter.body_index
        )
        carrier_contact_count = simulation.soft_contact_count(
            simulation.fingertip_model.carrier_body
        )
        if plate_contact_count == 0:
            raise RuntimeError(
                "force target was reached without flat-plate contact"
            )
        if carrier_contact_count == 0:
            raise RuntimeError(
                "force target was reached without carrier contact"
            )

        particle_q = simulation.state.particle_q.numpy()
        particle_qd = simulation.state.particle_qd.numpy()
        if not np.all(np.isfinite(particle_q)) or not np.all(
            np.isfinite(particle_qd)
        ):
            raise RuntimeError(
                "contact step produced a non-finite silicone state"
            )

        carrier_depths_m = _carrier_interior_depths_m(
            particle_q,
            carrier_cross_section=carrier_cross_section,
            carrier_y_limits_m=carrier_y_limits_m,
        )
        carrier_depths_m[~nonbonded] = 0.0
        surface_carrier_depths_m = carrier_depths_m[surface_indices]
        tet_centers_m = particle_q[tet_indices].mean(axis=1)
        tet_carrier_depths_m = _carrier_interior_depths_m(
            tet_centers_m,
            carrier_cross_section=carrier_cross_section,
            carrier_y_limits_m=carrier_y_limits_m,
        )
        max_carrier_penetration_m = float(carrier_depths_m.max())
        deep_nonbonded_count = int(
            np.count_nonzero(
                carrier_depths_m > _MAX_CARRIER_PENETRATION_M
            )
        )
        deep_surface_count = int(
            np.count_nonzero(
                surface_carrier_depths_m
                > _MAX_CARRIER_PENETRATION_M
            )
        )
        deep_tet_count = int(
            np.count_nonzero(
                tet_carrier_depths_m > _MAX_CARRIER_PENETRATION_M
            )
        )
        if deep_nonbonded_count or deep_surface_count or deep_tet_count:
            raise RuntimeError(
                "silicone penetrated the carrier beyond the allowed "
                f"{_MAX_CARRIER_PENETRATION_M:.9e} m contact tolerance"
            )

        bonded_drift_m = np.linalg.norm(
            particle_q[bonded_indices]
            - reference_positions[bonded_indices],
            axis=1,
        )
        max_bonded_drift_m = float(bonded_drift_m.max())
        if max_bonded_drift_m > _MAX_BONDED_DRIFT_M:
            raise RuntimeError("bonded silicone vertices drifted")

        applied_steps = simulation.step_count
        travel_m = applied_steps * _PLATE_STEP_M
        plate_z_m = initial_plate_z_m + travel_m
        print(
            "initial plate contacts:  "
            f"{initial_plate_contact_count}"
        )
        print(
            "initial carrier candidates: "
            f"{initial_carrier_contact_count}"
        )
        print(f"force threshold:       {_FORCE_THRESHOLD_N:.9e} N")
        print(f"applied steps:         {applied_steps}")
        print(f"simulation time:       {simulation.time_s:.9e} s")
        print(f"plate travel:          {travel_m:.9e} m")
        print(f"plate center z:        {plate_z_m:.9e} m")
        print(f"soft contacts:         {contact_count}")
        print(f"plate contacts:        {plate_contact_count}")
        print(f"carrier contacts:      {carrier_contact_count}")
        print(f"transient reaction:    {reaction_force_n:.9e} N")
        print(
            "nonbonded vertices inside carrier: "
            f"{int(np.count_nonzero(carrier_depths_m > 0.0))}"
        )
        print(
            "surface vertices inside carrier:   "
            f"{int(np.count_nonzero(surface_carrier_depths_m > 0.0))}"
        )
        print(
            "tet centers inside carrier:        "
            f"{int(np.count_nonzero(tet_carrier_depths_m > 0.0))}"
        )
        print(
            "maximum carrier penetration:       "
            f"{max_carrier_penetration_m:.9e} m"
        )
        print(
            "penetrations beyond tolerance:     "
            f"vertices={deep_nonbonded_count}, "
            f"surface={deep_surface_count}, tets={deep_tet_count}"
        )
        print(
            "maximum bonded drift:               "
            f"{max_bonded_drift_m:.9e} m"
        )
        print("flat-plate transient-force contact: PASS")

        if viewer is not None and viewer.is_running():
            print("ViewerGL is active; close the window to finish.")
            while viewer.is_running():
                _render(viewer, simulation)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments not in ([], ["--viewer"]):
        raise SystemExit(
            "usage: python validation/contact-physics/"
            "flat_plate_contact.py [--viewer]"
        )
    main(show_viewer=arguments == ["--viewer"])
