"""Check the first Newton fingertip kinematic boundary condition."""

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
    identity = wp.transform_identity()
    fingertip_newton.prepare_step(state_0, state_1, identity)

    reference = model.particle_q.numpy()[bonded_indices]
    identity_positions = state_0.particle_q.numpy()[bonded_indices]
    np.testing.assert_allclose(identity_positions, reference, rtol=0.0, atol=1.0e-7)

    translation = wp.transform(
        p=wp.vec3(1.0e-3, -2.0e-3, 3.0e-3),
        q=wp.quat_identity(),
    )
    fingertip_newton.prepare_step(state_0, state_1, translation)
    translated = state_0.particle_q.numpy()[bonded_indices]
    expected = fingertip_newton.bonded_local_positions.numpy() + np.asarray(
        [1.0e-3, -2.0e-3, 3.0e-3],
        dtype=np.float32,
    )
    np.testing.assert_allclose(translated, expected, rtol=0.0, atol=1.0e-7)

    print("kinematic fingertip bond: PASS")
    print(f"carrier body:             {fingertip_newton.carrier_body}")
    print(f"bonded particles:          {bonded_indices.size}")


if __name__ == "__main__":
    main()
