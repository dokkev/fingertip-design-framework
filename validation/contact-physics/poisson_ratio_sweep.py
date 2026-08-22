"""Compare near-incompressible silicone parameters under flat-plate contact."""

from __future__ import annotations

from importlib.resources import as_file, files

import newton
import numpy as np
import warp as wp

from lumo.fingertip import (
    Fingertip,
    FingertipParameters,
    ViscoelasticParameters,
)
from lumo.newton import Indenter
from lumo.simulation import LumoSimulation


_POISSON_RATIOS = (0.48, 0.49, 0.495)
_DENSITY_KG_M3 = 1070.0
_K_MU_PA = 1.06e5
_DAMPING_PA_S = 10.0

_SIM_FREQUENCY_HZ = 1.0e3
_PLATE_STEP_M = 2.5e-5
_INITIAL_CLEARANCE_M = 1.0e-3
_PLATE_HALF_THICKNESS_M = 1.0e-3
_MAX_SIM_TIME_S = 30.0
_FORCE_THRESHOLD_N = 15.0


def _lame_lambda_pa(poisson_ratio: float) -> float:
    return 2.0 * _K_MU_PA * poisson_ratio / (1.0 - 2.0 * poisson_ratio)


def _six_tet_volumes(
    positions: np.ndarray,
    tet_indices: np.ndarray,
) -> np.ndarray:
    tets = tet_indices.reshape(-1, 4)
    p0 = positions[tets[:, 0]]
    p1 = positions[tets[:, 1]]
    p2 = positions[tets[:, 2]]
    p3 = positions[tets[:, 3]]
    return np.einsum(
        "ij,ij->i",
        p1 - p0,
        np.cross(p2 - p0, p3 - p0),
    )


def _run_case(poisson_ratio: float) -> None:
    k_lambda_pa = _lame_lambda_pa(poisson_ratio)
    fingertip = Fingertip(
        FingertipParameters(
            viscoelastic=ViscoelasticParameters(
                density_kg_m3=_DENSITY_KG_M3,
                k_mu_pa=_K_MU_PA,
                k_lambda_pa=k_lambda_pa,
                damping=_DAMPING_PA_S,
            )
        )
    )
    fingertip_tip_z_m = 1.0e-3 * (
        fingertip.silicone.ellipse_center_z_mm
        - fingertip.silicone.ellipse_radius_z_mm
    )
    initial_plate_z_m = (
        fingertip_tip_z_m
        - _INITIAL_CLEARANCE_M
        - _PLATE_HALF_THICKNESS_M
    )

    builder = newton.ModelBuilder(gravity=0.0)

    flat_plate_resource = files("lumo").joinpath(
        "assets",
        "objects",
        "flat_plate.urdf",
    )
    with as_file(flat_plate_resource) as urdf_path:
        indenter = Indenter.add_urdf(
            builder,
            urdf_path,
            tf=wp.transform(
                wp.vec3(0.0, 0.0, initial_plate_z_m),
                wp.quat_identity(),
            ),
        )

    simulation = LumoSimulation(
        fingertip,
        builder=builder,
        sim_frequency=_SIM_FREQUENCY_HZ,
    )
    reference_positions = simulation.state.particle_q.numpy().copy()
    tet_indices = simulation.fingertip_mesh.silicone.tet_indices
    reference_six_volumes = _six_tet_volumes(
        reference_positions,
        tet_indices,
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
            f"nu={poisson_ratio:g} has soft contacts before prescribed motion"
        )

    max_sim_steps = int(_MAX_SIM_TIME_S * simulation.sim_frequency)
    if max_sim_steps < 1:
        raise ValueError("maximum simulation time must include at least one tick")

    had_soft_contact = False
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

        contact_count = int(
            simulation.contacts.soft_contact_count.numpy()[0]
        )
        had_soft_contact = had_soft_contact or contact_count > 0
        reaction_force_n = simulation.indenter_reaction_force(
            indenter,
            motion_direction_W=wp.vec3(0.0, 0.0, _PLATE_STEP_M),
        )
        if reaction_force_n >= _FORCE_THRESHOLD_N:
            target_reached = True
            break
    else:
        target_reached = False

    if not had_soft_contact:
        raise RuntimeError(f"nu={poisson_ratio:g} produced no soft contact")

    particle_q = simulation.state.particle_q.numpy()
    particle_qd = simulation.state.particle_qd.numpy()
    if not np.all(np.isfinite(particle_q)) or not np.all(
        np.isfinite(particle_qd)
    ):
        raise RuntimeError(f"nu={poisson_ratio:g} produced a non-finite state")

    current_six_volumes = _six_tet_volumes(particle_q, tet_indices)
    tet_j = current_six_volumes / reference_six_volumes
    if not np.all(np.isfinite(tet_j)):
        raise RuntimeError(f"nu={poisson_ratio:g} produced non-finite tet J")

    total_volume_ratio = float(
        np.abs(current_six_volumes).sum()
        / np.abs(reference_six_volumes).sum()
    )
    travel_m = simulation.step_count * _PLATE_STEP_M
    status = "TARGET REACHED" if target_reached else "TIME LIMIT"

    print()
    print(f"Poisson ratio:       {poisson_ratio:.6f}")
    print(f"k_mu:               {_K_MU_PA:.9e} Pa")
    print(f"k_lambda:           {k_lambda_pa:.9e} Pa")
    print(f"status:              {status}")
    print(f"simulation time:     {simulation.time_s:.9e} s")
    print(f"plate travel:        {travel_m:.9e} m")
    print(f"reaction force:      {reaction_force_n:.9e} N")
    print(f"total volume ratio:  {total_volume_ratio:.9e}")
    print(f"minimum tet J:       {float(tet_j.min()):.9e}")
    print(f"maximum tet J:       {float(tet_j.max()):.9e}")


def main() -> None:
    print("Dragon Skin 10 NV Poisson-ratio sweep")
    print(f"density:             {_DENSITY_KG_M3:.9e} kg/m^3")
    print(f"damping:             {_DAMPING_PA_S:.9e} Pa s")
    print(f"force threshold:     {_FORCE_THRESHOLD_N:.9e} N")
    print(f"maximum sim time:    {_MAX_SIM_TIME_S:.9e} s")

    for poisson_ratio in _POISSON_RATIOS:
        _run_case(poisson_ratio)


if __name__ == "__main__":
    main()
