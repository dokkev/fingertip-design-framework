"""Check that SolverVBD preserves the unloaded fingertip reference state."""

from __future__ import annotations

import numpy as np
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import make_fingertip_mesh
from lumo.newton import build_fingertip_newton_model
from lumo.simulation import LumoSimulation


_BONDED_DISPLACEMENT_TOLERANCE_M = 1.0e-7
_ZERO_LOAD_DRIFT_TOLERANCE_M = 1.0e-6
_TIME_STEP_S = 1.0e-3


def main() -> None:
    fingertip = Fingertip(FingertipParameters())
    mesh = make_fingertip_mesh(fingertip)
    model = build_fingertip_newton_model(mesh)
    simulation = LumoSimulation(model, time_step_s=_TIME_STEP_S)

    q0 = simulation.state.particle_q.numpy().copy()
    pose = wp.transform_identity()

    for _ in range(100):
        simulation.apply_carrier_pose(pose)
        simulation.step()

    q1 = simulation.state.particle_q.numpy()
    qd1 = simulation.state.particle_qd.numpy()

    if not np.all(np.isfinite(q0)):
        raise RuntimeError("reference particle positions are not finite")
    if not np.all(np.isfinite(q1)) or not np.all(np.isfinite(qd1)):
        raise RuntimeError("zero-load SolverVBD state is not finite")

    displacement = np.linalg.norm(q1 - q0, axis=1)
    bonded_indices = model.bonded_particle_indices.numpy()
    bonded_displacement = displacement[bonded_indices]

    max_displacement = float(displacement.max())
    max_bonded_displacement = float(bonded_displacement.max())

    print(f"max displacement:         {max_displacement:.9e} m")
    print(f"max bonded displacement: {max_bonded_displacement:.9e} m")

    if max_bonded_displacement > _BONDED_DISPLACEMENT_TOLERANCE_M:
        raise RuntimeError("bonded vertices drifted under zero load")
    if max_displacement > _ZERO_LOAD_DRIFT_TOLERANCE_M:
        raise RuntimeError("silicone reference shape drifted under zero load")

    print("zero-load SolverVBD validation: PASS")


if __name__ == "__main__":
    main()
