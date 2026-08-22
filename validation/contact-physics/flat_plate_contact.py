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
    FingertipNewtonSolver,
    Indenter,
    build_fingertip_newton_model,
)


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
    solver = FingertipNewtonSolver(
        model,
        time_step_s=_TIME_STEP_S,
        indenter=indenter,
        soft_contact_margin_m=_SOFT_CONTACT_MARGIN_M,
    )

    solver.collision_pipeline.collide(solver.state, solver.contacts)
    initial_contact_count = int(
        solver.contacts.soft_contact_count.numpy()[0]
    )
    if initial_contact_count != 0:
        raise RuntimeError(
            "flat plate has soft contacts before prescribed motion"
        )

    carrier_pose = wp.transform_identity()
    transient_reaction_force_n = indenter.move_until_force(
        solver,
        dx_m=0.0,
        dy_m=0.0,
        dz_m=_PLATE_STEP_M,
        f_des_n=force_threshold_n,
        carrier_pose=carrier_pose,
        max_steps=_MAX_STEPS,
    )

    contact_count = int(solver.contacts.soft_contact_count.numpy()[0])
    if contact_count <= 0:
        raise RuntimeError("force threshold was reached without soft contact")

    particle_q = solver.state.particle_q.numpy()
    particle_qd = solver.state.particle_qd.numpy()
    if not np.all(np.isfinite(particle_q)) or not np.all(
        np.isfinite(particle_qd)
    ):
        raise RuntimeError("contact step produced a non-finite silicone state")

    plate_pose = indenter.get_current_pose(solver.state)
    plate_z_m = float(plate_pose[2])
    travel_m = plate_z_m - initial_plate_z_m
    applied_steps = int(round(travel_m / _PLATE_STEP_M))
    print(f"initial soft contacts: {initial_contact_count}")
    print(f"force threshold:       {force_threshold_n:.9e} N")
    print(f"applied steps:         {applied_steps}")
    print(f"plate travel:          {travel_m:.9e} m")
    print(f"plate center z:        {plate_z_m:.9e} m")
    print(f"soft contacts:         {contact_count}")
    print(f"transient reaction:    {transient_reaction_force_n:.9e} N")
    print("flat-plate transient-force contact: PASS")


if __name__ == "__main__":
    main()
