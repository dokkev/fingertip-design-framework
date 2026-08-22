"""Check that SolverVBD preserves the unloaded fingertip reference state."""

from __future__ import annotations

import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.simulation import LumoSimulation


_BONDED_DISPLACEMENT_TOLERANCE_M = 1.0e-7
_ZERO_LOAD_DRIFT_TOLERANCE_M = 1.0e-6
_SIM_FREQUENCY_HZ = 1.0e3


def main() -> None:
    fingertip = Fingertip(FingertipParameters())
    simulation = LumoSimulation(
        fingertip,
        sim_frequency=_SIM_FREQUENCY_HZ,
    )

    q0 = simulation.state.particle_q.numpy().copy()

    for _ in range(100):
        simulation.step()

    q1 = simulation.state.particle_q.numpy()
    qd1 = simulation.state.particle_qd.numpy()

    if not np.all(np.isfinite(q0)):
        raise RuntimeError("reference particle positions are not finite")
    if not np.all(np.isfinite(q1)) or not np.all(np.isfinite(qd1)):
        raise RuntimeError("zero-load SolverVBD state is not finite")

    displacement = np.linalg.norm(q1 - q0, axis=1)
    bonded_indices = (
        simulation.fingertip_model.bonded_particle_indices.numpy()
    )
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
