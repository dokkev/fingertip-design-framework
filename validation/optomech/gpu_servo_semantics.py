"""Short regression for direct and GPU force-servo checkpoint semantics."""

from __future__ import annotations

from importlib.resources import as_file, files

import numpy as np
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import make_fingertip_5led_mesh
from lumo.simulation import DesignStudy, DesignTrial


def _run(fingertip, mesh, sphere_path, *, use_cuda_graph: bool):
    trial = DesignTrial(
        name="gpu_servo_semantics",
        urdf_path=sphere_path,
        initial_tf=wp.transform(
            wp.vec3(0.0, 0.0, fingertip.tip_z_m - 11.0e-3),
            wp.quat_identity(),
        ),
        motion_direction_W=wp.vec3(0.0, 0.0, 1.0),
        approach_speed_m_s=5.0e-3,
        target_force_n=5.0,
        max_sim_time_s=10.0,
        initial_clearance_m=1.0e-3,
    )
    checkpoints = []

    def collect(completed_trial, simulation, indenter):
        checkpoints.append(
            (
                completed_trial.step_count,
                completed_trial.reaction_force_n,
                completed_trial.travel_m,
                simulation.soft_contact_count(indenter.body_index),
                simulation.silicone_vertices().copy(),
            )
        )

    DesignStudy(
        fingertip,
        (trial,),
        fingertip_mesh=mesh,
        sim_frequency=100.0,
        settle_duration_s=0.05,
        force_tolerance_fraction=0.1,
        force_targets_n=(5.0,),
        force_gain_m_s_n=2.5e-4,
        iterations=10,
        soft_contact_margin_m=1.0e-4,
        carrier_contact_stiffness_n_m=1.0e6,
        contact_stiffness_n_m=3.0e4,
        contact_damping_n_s_m=0.28228017516945547,
        use_cuda_graph=use_cuda_graph,
    ).run(inspect_trial=collect)
    if len(checkpoints) != 1:
        raise RuntimeError("the one-target study did not emit exactly one checkpoint")
    return checkpoints[0]


def main() -> None:
    fingertip = Fingertip(FingertipParameters())
    mesh = make_fingertip_5led_mesh(fingertip, element_size_mm=1.0)
    resource = files("lumo.assets.objects.urdf").joinpath("sphere_20mm.urdf")
    with as_file(resource) as sphere_path:
        direct = _run(fingertip, mesh, sphere_path, use_cuda_graph=False)
        graph = _run(fingertip, mesh, sphere_path, use_cuda_graph=True)

    if abs(direct[0] - graph[0]) > 2:
        raise RuntimeError("GPU servo changed the accepted checkpoint by over two ticks")
    if not (4.5 <= direct[1] <= 5.5 and 4.5 <= graph[1] <= 5.5):
        raise RuntimeError("a backend accepted force outside the 5 N band")
    if direct[3] <= 0 or graph[3] <= 0:
        raise RuntimeError("a backend accepted a checkpoint without contact")
    vertex_rms_m = float(np.sqrt(np.mean((direct[4] - graph[4]) ** 2)))
    if vertex_rms_m > 1.0e-4:
        raise RuntimeError("short GPU servo state is not mechanically equivalent")
    print(
        "PASS | "
        f"step direct/graph={direct[0]}/{graph[0]} | "
        f"force={direct[1]:.6f}/{graph[1]:.6f} N | "
        f"travel={1.0e3 * direct[2]:.6f}/{1.0e3 * graph[2]:.6f} mm | "
        f"vertex RMS={1.0e6 * vertex_rms_m:.3f} um"
    )


if __name__ == "__main__":
    main()
