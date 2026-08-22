"""Indent three spherical URDFs at different fingertip locations to 20 N."""

from __future__ import annotations

from contextlib import ExitStack
from importlib.resources import as_file, files
from pathlib import Path

import numpy as np
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.simulation import LumoSimulation
from lumo.simulation.indentation import IndentationCase, IndentationStudy


_SIM_FREQUENCY_HZ = 1.0e3
_TRANSLATION_STEP_M = 2.5e-5
_INITIAL_CLEARANCE_M = 1.0e-3
_MAX_SIM_TIME_S = 30.0
_TARGET_FORCE_N = 20.0
_MAX_BONDED_DRIFT_M = 1.0e-8


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


def _make_case(
    fingertip: Fingertip,
    urdf_path: Path,
    urdf_filename: str,
    *,
    sphere_diameter_mm: float,
    contact_x_mm: float,
) -> IndentationCase:
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

    return IndentationCase(
        name=urdf_filename,
        urdf_path=urdf_path,
        initial_tf=initial_pose,
        translation_step_W_m=wp.vec3(
            0.0,
            0.0,
            _TRANSLATION_STEP_M,
        ),
        target_force_n=_TARGET_FORCE_N,
        max_sim_time_s=_MAX_SIM_TIME_S,
    )


def _validate_and_report(case: IndentationCase) -> None:
    simulation = case.simulation
    indenter = case.indenter
    reaction_force_n = case.reaction_force_n
    if simulation is None or indenter is None or reaction_force_n is None:
        raise RuntimeError(f"{case.name} has not completed")

    contact_x_mm = 1.0e3 * float(np.asarray(case.initial_tf)[0])
    bonded_indices = (
        simulation.fingertip_model.bonded_particle_indices.numpy()
    )

    sphere_contact_count = _soft_contact_count_for_body(
        simulation,
        indenter.body_index,
    )
    if sphere_contact_count == 0:
        raise RuntimeError(
            f"{case.name} reached the force target without contact"
        )

    particle_q = simulation.state.particle_q.numpy()
    particle_qd = simulation.state.particle_qd.numpy()
    if not np.all(np.isfinite(particle_q)) or not np.all(
        np.isfinite(particle_qd)
    ):
        raise RuntimeError(
            f"{case.name} produced a non-finite silicone state"
        )

    bonded_drift_m = np.linalg.norm(
        particle_q[bonded_indices]
        - simulation.fingertip_model.bonded_local_positions.numpy(),
        axis=1,
    )
    max_bonded_drift_m = float(bonded_drift_m.max())
    if max_bonded_drift_m > _MAX_BONDED_DRIFT_M:
        raise RuntimeError(
            f"{case.name} caused bonded silicone vertices to drift"
        )

    print(f"case:                  {case.name}")
    print(f"contact location X:    {contact_x_mm:.1f} mm")
    print(f"target force:          {case.target_force_n:.9e} N")
    elapsed_time_s = case.step_count / simulation.sim_frequency
    travel_m = case.step_count * float(
        np.linalg.norm(case.translation_step_W_m)
    )
    print(f"transient reaction:    {reaction_force_n:.9e} N")
    print(f"simulation time:       {elapsed_time_s:.9e} s")
    print(f"sphere travel:         {travel_m:.9e} m")
    print(f"sphere contacts:       {sphere_contact_count}")
    print(f"maximum bonded drift:  {max_bonded_drift_m:.9e} m")
    print()


def main() -> None:
    fingertip = Fingertip(FingertipParameters())
    resource_root = files("lumo").joinpath(
        "assets",
        "objects",
        "urdf",
    )

    with ExitStack() as resources:
        sphere_5mm = _make_case(
            fingertip,
            resources.enter_context(
                as_file(resource_root.joinpath("sphere_5mm.urdf"))
            ),
            "sphere_5mm.urdf",
            sphere_diameter_mm=5.0,
            contact_x_mm=-7.5,
        )
        sphere_10mm = _make_case(
            fingertip,
            resources.enter_context(
                as_file(resource_root.joinpath("sphere_10mm.urdf"))
            ),
            "sphere_10mm.urdf",
            sphere_diameter_mm=10.0,
            contact_x_mm=0.0,
        )
        sphere_20mm = _make_case(
            fingertip,
            resources.enter_context(
                as_file(resource_root.joinpath("sphere_20mm.urdf"))
            ),
            "sphere_20mm.urdf",
            sphere_diameter_mm=20.0,
            contact_x_mm=7.5,
        )
        study = IndentationStudy(
            fingertip,
            (sphere_5mm, sphere_10mm, sphere_20mm),
            sim_frequency=_SIM_FREQUENCY_HZ,
        )
        study.run()

        _validate_and_report(sphere_5mm)
        _validate_and_report(sphere_10mm)
        _validate_and_report(sphere_20mm)

    print("three-location spherical indentation: PASS")


if __name__ == "__main__":
    main()
