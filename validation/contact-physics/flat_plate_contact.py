"""Move the flat-plate URDF until a transient contact force is reached."""

from __future__ import annotations

import argparse
from pathlib import Path

import newton
import numpy as np
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import make_fingertip_mesh
from lumo.newton import (
    Indenter,
    build_fingertip_newton_model,
)
from lumo.simulation import LumoSimulation


_TIME_STEP_S = 1.0e-3
_PLATE_STEP_M = 2.5e-5
_INITIAL_CLEARANCE_M = 1.0e-3
_PLATE_HALF_THICKNESS_M = 1.0e-3
_MAX_STEPS = 80
_SOFT_CONTACT_MARGIN_M = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Move a kinematic flat plate into the fingertip until its "
            "transient -Z reaction force reaches a prescribed threshold."
        )
    )
    parser.add_argument(
        "--force-threshold-n",
        type=float,
        required=True,
        help="positive transient -Z reaction-force threshold in newtons",
    )
    args = parser.parse_args()
    force_threshold_n = args.force_threshold_n
    if not np.isfinite(force_threshold_n) or force_threshold_n <= 0.0:
        parser.error("--force-threshold-n must be finite and positive")

    fingertip = Fingertip(FingertipParameters())
    mesh = make_fingertip_mesh(fingertip)

    reference_positions = np.asarray(mesh.silicone.vertices, dtype=np.float64)
    fingertip_tip_z_m = float(reference_positions[:, 2].min())
    initial_plate_z_m = (
        fingertip_tip_z_m
        - _INITIAL_CLEARANCE_M
        - _PLATE_HALF_THICKNESS_M
    )
    initial_plate_pose = wp.transform(
        wp.vec3(0.0, 0.0, initial_plate_z_m),
        wp.quat_identity(),
    )

    builder = newton.ModelBuilder(gravity=0.0)
    # Use the silicone surface itself as the contact boundary. Newton's
    # generic 0.1 m particle-radius default is not meaningful at this scale.
    builder.default_particle_radius = 0.0

    urdf_path = (
        Path(__file__).resolve().parents[2]
        / "assets"
        / "objects"
        / "flat_plate.urdf"
    )
    indenter = Indenter.add_urdf(
        builder,
        urdf_path,
        tf=initial_plate_pose,
    )
    model = build_fingertip_newton_model(mesh, builder=builder)
    simulation = LumoSimulation(
        model,
        time_step_s=_TIME_STEP_S,
        soft_contact_margin_m=_SOFT_CONTACT_MARGIN_M,
    )

    simulation.collision_pipeline.collide(
        simulation.state,
        simulation.contacts,
    )
    initial_contact_count = int(
        simulation.contacts.soft_contact_count.numpy()[0]
    )
    if initial_contact_count != 0:
        raise RuntimeError(
            "flat plate has soft contacts before prescribed motion"
        )

    body_to_wrench = np.full(model.model.body_count, -1, dtype=np.int32)
    body_to_wrench[indenter.body_index] = 0
    body_to_wrench_device = wp.array(
        body_to_wrench,
        dtype=wp.int32,
        device=model.model.device,
    )
    indenter_wrench = wp.zeros(
        1,
        dtype=wp.spatial_vector,
        device=model.model.device,
    )

    carrier_pose = wp.transform_identity()
    contact_step = None
    threshold_step = None
    contact_count = 0
    plate_z_m = initial_plate_z_m
    transient_reaction_force_n = 0.0
    peak_transient_reaction_force_n = 0.0

    for step in range(1, _MAX_STEPS + 1):
        plate_z_m = initial_plate_z_m + step * _PLATE_STEP_M
        plate_pose = wp.transform(
            wp.vec3(0.0, 0.0, plate_z_m),
            wp.quat_identity(),
        )
        simulation.apply_carrier_pose(carrier_pose)
        simulation.apply_indenter_pose(indenter, plate_pose)

        if simulation.state.body_qd is None:
            raise RuntimeError("simulation state has no rigid-body velocities")
        body_qd_before = wp.clone(simulation.state.body_qd)
        state_before = simulation.state
        simulation.step()

        contact_count = int(
            simulation.contacts.soft_contact_count.numpy()[0]
        )
        if contact_count > 0 and contact_step is None:
            contact_step = step

        simulation.solver.coupling_harvest_proxy_wrenches(
            body_to_wrench_device,
            indenter_wrench,
            body_qd_before=body_qd_before,
            state=state_before,
            state_out=simulation.state,
            contacts=simulation.contacts,
            dt=simulation.time_step_s,
        )
        wrench = indenter_wrench.numpy()[0]
        if not np.all(np.isfinite(wrench)):
            raise RuntimeError("contact step produced a non-finite wrench")

        transient_reaction_force_n = max(0.0, -float(wrench[2]))
        peak_transient_reaction_force_n = max(
            peak_transient_reaction_force_n,
            transient_reaction_force_n,
        )
        if transient_reaction_force_n >= force_threshold_n:
            threshold_step = step
            break

    if contact_step is None:
        raise RuntimeError(
            "flat plate reached its maximum travel without soft contact"
        )
    if threshold_step is None:
        raise RuntimeError(
            "flat plate reached its maximum travel without reaching the "
            f"transient force threshold; peak force was "
            f"{peak_transient_reaction_force_n:.9e} N"
        )

    particle_q = simulation.state.particle_q.numpy()
    particle_qd = simulation.state.particle_qd.numpy()
    if not np.all(np.isfinite(particle_q)) or not np.all(
        np.isfinite(particle_qd)
    ):
        raise RuntimeError("contact step produced a non-finite silicone state")

    travel_m = threshold_step * _PLATE_STEP_M
    print(f"initial soft contacts: {initial_contact_count}")
    print(f"first contact step:    {contact_step}")
    print(f"force threshold:       {force_threshold_n:.9e} N")
    print(f"threshold step:        {threshold_step}")
    print(f"plate travel:          {travel_m:.9e} m")
    print(f"plate center z:        {plate_z_m:.9e} m")
    print(f"soft contacts:         {contact_count}")
    print(f"transient reaction:    {transient_reaction_force_n:.9e} N")
    print("flat-plate transient-force contact: PASS")


if __name__ == "__main__":
    main()
