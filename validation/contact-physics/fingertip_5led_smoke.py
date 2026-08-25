"""Run a short central-contact Newton smoke on the full five-LED mesh."""

from __future__ import annotations

from importlib.resources import as_file, files

import newton
import numpy as np
import warp as wp

from lumo.fingertip import Fingertip
from lumo.mesh import make_fingertip_5led_mesh
from lumo.newton import Indenter
from lumo.simulation import LumoSimulation


_SIM_FREQUENCY_HZ = 100.0
_VBD_ITERATIONS = 10
_CONTACT_STIFFNESS_N_M = 3.0e4
_CONTACT_DAMPING_N_S_M = 0.28228017516945547
_SPHERE_RADIUS_M = 5.0e-3
_INITIAL_CLEARANCE_M = 0.25e-3
_TRANSLATION_STEP_M = 25.0e-6
_MAX_STEPS = 120
_POST_CONTACT_STEPS = 10


def main() -> None:
    fingertip = Fingertip()
    fingertip_mesh = make_fingertip_5led_mesh(
        fingertip,
        element_size_mm=1.0,
    )
    initial_z_m = fingertip.tip_z_m - _SPHERE_RADIUS_M - _INITIAL_CLEARANCE_M
    initial_tf = wp.transform(
        wp.vec3(0.0, 0.0, initial_z_m),
        wp.quat_identity(),
    )

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, 0.0))
    sphere_resource = files("lumo").joinpath(
        "assets",
        "objects",
        "urdf",
        "sphere_10mm.urdf",
    )
    with as_file(sphere_resource) as sphere_path:
        indenter = Indenter.add_urdf(
            builder,
            sphere_path,
            tf=initial_tf,
            contact_stiffness_n_m=_CONTACT_STIFFNESS_N_M,
            contact_damping_n_s_m=_CONTACT_DAMPING_N_S_M,
        )

    simulation = LumoSimulation(
        fingertip,
        builder=builder,
        fingertip_mesh=fingertip_mesh,
        sim_frequency=_SIM_FREQUENCY_HZ,
        iterations=_VBD_ITERATIONS,
        soft_contact_stiffness_n_m=_CONTACT_STIFFNESS_N_M,
        soft_contact_damping_n_s_m=_CONTACT_DAMPING_N_S_M,
    )
    if simulation.soft_contact_count(indenter.body_index) != 0:
        raise RuntimeError("central sphere contacts the full fingertip initially")

    reference_vertices = simulation.silicone_vertices()
    contact_step = None
    final_contact_count = 0
    for step in range(1, _MAX_STEPS + 1):
        travel_m = step * _TRANSLATION_STEP_M
        tf = wp.transform(
            wp.vec3(0.0, 0.0, initial_z_m + travel_m),
            wp.quat_identity(),
        )
        simulation.apply_indenter_pose(indenter, tf)
        simulation.step()
        final_contact_count = simulation.soft_contact_count(indenter.body_index)
        if final_contact_count and contact_step is None:
            contact_step = step
        if contact_step is not None and step >= contact_step + _POST_CONTACT_STEPS:
            break

    if contact_step is None:
        raise RuntimeError("central sphere created no contact in the smoke travel")
    final_vertices = simulation.silicone_vertices()
    if not np.all(np.isfinite(final_vertices)):
        raise RuntimeError("full fingertip produced non-finite silicone vertices")
    displacement_m = np.linalg.norm(final_vertices - reference_vertices, axis=1)
    bonded_displacement_m = displacement_m[fingertip_mesh.bonded_vertex_indices]
    if float(displacement_m.max()) <= 1.0e-7:
        raise RuntimeError("central contact did not deform the silicone")
    if float(bonded_displacement_m.max()) > 1.0e-8:
        raise RuntimeError("full fingertip kinematic bond drifted")

    print("full five-LED Newton smoke: PASS")
    print(f"particles: {simulation.fingertip_model.model.particle_count}")
    print(f"tetrahedra: {simulation.fingertip_model.model.tet_count}")
    print(f"first contact step: {contact_step}")
    print(f"final indenter contacts: {final_contact_count}")
    print(f"max silicone displacement: {1.0e3 * displacement_m.max():.6f} mm")
    print(
        "max bonded displacement: "
        f"{1.0e6 * bonded_displacement_m.max():.6f} um"
    )


if __name__ == "__main__":
    main()
