"""Indent three spherical URDFs at different fingertip locations to 20 N."""

from __future__ import annotations

from importlib.resources import as_file, files

import newton
import numpy as np
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.indentation import IndentationCase
from lumo.newton import Indenter
from lumo.simulation import LumoSimulation


_SIM_FREQUENCY_HZ = 1.0e3
_TRANSLATION_STEP_M = 2.5e-5
_INITIAL_CLEARANCE_M = 1.0e-3
_MAX_SIM_TIME_S = 30.0
_TARGET_FORCE_N = 20.0
_MAX_BONDED_DRIFT_M = 1.0e-8

# Sphere size is diameter. Each independent case uses a different X location.
_CASES = (
    ("sphere_5mm.urdf", 5.0, -7.5),
    ("sphere_10mm.urdf", 10.0, 0.0),
    ("sphere_20mm.urdf", 20.0, 7.5),
)


def _soft_contact_count_for_body(
    simulation: LumoSimulation,
    body_index: int,
) -> int:
    contact_count = int(
        simulation.contacts.soft_contact_count.numpy()[0]
    )
    shape_indices = simulation.contacts.soft_contact_shape.numpy()[
        :contact_count
    ]
    valid = shape_indices >= 0
    shape_bodies = simulation.fingertip_model.model.shape_body.numpy()
    return int(
        np.count_nonzero(
            shape_bodies[shape_indices[valid]] == body_index
        )
    )


def _run_indentation(
    parameters: FingertipParameters,
    urdf_filename: str,
    *,
    sphere_diameter_mm: float,
    contact_x_mm: float,
) -> None:
    fingertip = Fingertip(parameters)
    fingertip_tip_z_m = 1.0e-3 * (
        fingertip.silicone.ellipse_center_z_mm
        - fingertip.silicone.ellipse_radius_z_mm
    )
    sphere_radius_m = 0.5e-3 * sphere_diameter_mm
    initial_center_z_m = (
        fingertip_tip_z_m
        - _INITIAL_CLEARANCE_M
        - sphere_radius_m
    )
    contact_x_m = 1.0e-3 * contact_x_mm
    initial_pose = wp.transform(
        wp.vec3(contact_x_m, 0.0, initial_center_z_m),
        wp.quat_identity(),
    )

    builder = newton.ModelBuilder(gravity=0.0)
    sphere_resource = files("lumo").joinpath(
        "assets",
        "objects",
        "urdf",
        urdf_filename,
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
    reference_positions = simulation.state.particle_q.numpy().copy()
    bonded_indices = (
        simulation.fingertip_model.bonded_particle_indices.numpy()
    )

    simulation.collision_pipeline.collide(
        simulation.state,
        simulation.contacts,
    )
    initial_contact_count = _soft_contact_count_for_body(
        simulation,
        indenter.body_index,
    )
    if initial_contact_count != 0:
        raise RuntimeError(
            f"{urdf_filename} has soft contacts before prescribed motion"
        )

    case = IndentationCase(
        simulation,
        indenter,
        name=urdf_filename,
        initial_tf=initial_pose,
        translation_step_W_m=wp.vec3(
            0.0,
            0.0,
            _TRANSLATION_STEP_M,
        ),
        target_force_n=_TARGET_FORCE_N,
        max_sim_time_s=_MAX_SIM_TIME_S,
    )
    while not case.target_reached:
        case.apply_next_pose()
        simulation.step()
        case.observe_step()

    sphere_contact_count = _soft_contact_count_for_body(
        simulation,
        indenter.body_index,
    )
    if sphere_contact_count == 0:
        raise RuntimeError(
            f"{urdf_filename} reached the force target without contact"
        )

    particle_q = simulation.state.particle_q.numpy()
    particle_qd = simulation.state.particle_qd.numpy()
    if not np.all(np.isfinite(particle_q)) or not np.all(
        np.isfinite(particle_qd)
    ):
        raise RuntimeError(
            f"{urdf_filename} produced a non-finite silicone state"
        )

    bonded_drift_m = np.linalg.norm(
        particle_q[bonded_indices] - reference_positions[bonded_indices],
        axis=1,
    )
    max_bonded_drift_m = float(bonded_drift_m.max())
    if max_bonded_drift_m > _MAX_BONDED_DRIFT_M:
        raise RuntimeError(
            f"{urdf_filename} caused bonded silicone vertices to drift"
        )

    print(f"sphere diameter:       {sphere_diameter_mm:.1f} mm")
    print(f"contact location X:    {contact_x_mm:.1f} mm")
    print(f"target force:          {_TARGET_FORCE_N:.9e} N")
    print(f"transient reaction:    {case.reaction_force_n:.9e} N")
    print(f"simulation time:       {case.elapsed_time_s:.9e} s")
    print(f"sphere travel:         {case.travel_m:.9e} m")
    print(f"sphere contacts:       {sphere_contact_count}")
    print(f"maximum bonded drift:  {max_bonded_drift_m:.9e} m")
    print()


def main() -> None:
    morphology = FingertipParameters()
    for urdf_filename, diameter_mm, contact_x_mm in _CASES:
        _run_indentation(
            morphology,
            urdf_filename,
            sphere_diameter_mm=diameter_mm,
            contact_x_mm=contact_x_mm,
        )

    print("three-location spherical indentation: PASS")


if __name__ == "__main__":
    main()
