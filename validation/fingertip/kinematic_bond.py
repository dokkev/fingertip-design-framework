"""Check the Newton fingertip fixed boundary condition."""

from __future__ import annotations

import numpy as np
import newton
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.newton import build_fingertip_newton_model
from lumo.mesh import make_fingertip_mesh


def main() -> None:
    mesh = make_fingertip_mesh(
        Fingertip(FingertipParameters()),
        element_size_mm=1.0,
    )
    fingertip_newton = build_fingertip_newton_model(mesh)
    model = fingertip_newton.model

    if not model.body_flags.numpy()[fingertip_newton.carrier_body] & int(
        newton.BodyFlags.KINEMATIC
    ):
        raise RuntimeError("carrier body is not kinematic")

    bonded_indices = fingertip_newton.bonded_particle_indices.numpy()
    particle_flags = model.particle_flags.numpy()[bonded_indices]
    if np.any(particle_flags & int(newton.ParticleFlags.ACTIVE)):
        raise RuntimeError("bonded particles must be non-active")

    state_0 = model.state()
    state_1 = model.state()
    fingertip_newton.prepare_step(state_0, state_1)

    reference = fingertip_newton.bonded_reference_positions.numpy()
    identity = np.asarray(wp.transform_identity())
    for state in (state_0, state_1):
        bonded_positions = state.particle_q.numpy()[bonded_indices]
        carrier_pose = state.body_q.numpy()[fingertip_newton.carrier_body]
        np.testing.assert_allclose(
            bonded_positions,
            reference,
            rtol=0.0,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            carrier_pose,
            identity,
            rtol=0.0,
            atol=1.0e-7,
        )

    print("fixed fingertip bond: PASS")
    print(f"carrier body:             {fingertip_newton.carrier_body}")
    print(f"bonded particles:          {bonded_indices.size}")


if __name__ == "__main__":
    main()
