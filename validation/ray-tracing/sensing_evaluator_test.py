"""Validate one production Newton-to-OptiX sensing evaluation."""

from __future__ import annotations

from importlib.resources import as_file, files
from time import perf_counter

import numpy as np
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.optimization.evaluator import evaluate_contact_sensing
from lumo.ray_tracing import PathTraceResult
from lumo.simulation import DesignTrial


_SPHERE_RADIUS_M = 7.5e-3
_INITIAL_CLEARANCE_M = 1.0e-3


def _print_state(
    label: str,
    response: np.ndarray,
    paths: PathTraceResult,
) -> None:
    if response.shape != (4,) or not np.all(np.isfinite(response)):
        raise AssertionError(f"{label} has an invalid side-view response")
    if not np.isclose(
        paths.accounted_power,
        paths.emitted_power,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError(f"{label} path-energy ledger does not close")

    print(label)
    print(f"  side-view [Q1 Q2 Q3 Q4]: {response}")
    print(
        "  path energy: "
        f"emitted={paths.emitted_power:.9e} | "
        f"escaped={paths.escaped_power:.9e} | "
        f"carrier_absorbed={paths.absorbed_power:.9e} | "
        f"bulk_loss={paths.bulk_loss_power:.9e} | "
        f"internal_miss={paths.unresolved_internal_miss_power:.9e} | "
        f"remaining={paths.remaining_power:.9e} | "
        f"accounted={paths.accounted_power:.9e} | "
        f"closure={paths.closure_error:+.3e}"
    )
    print(
        f"  escaped rays={paths.escaped_ray_count} | "
        f"remaining rays={paths.remaining_ray_count}"
    )


def main() -> None:
    wall_start = perf_counter()
    fingertip = Fingertip(FingertipParameters())
    initial_center_z_m = (
        fingertip.tip_z_m - _INITIAL_CLEARANCE_M - _SPHERE_RADIUS_M
    )
    sphere_resource = files("lumo").joinpath(
        "assets",
        "objects",
        "urdf",
        "sphere_15mm.urdf",
    )
    with as_file(sphere_resource) as sphere_urdf_path:
        trial = DesignTrial(
            name="centered_15mm_sphere",
            urdf_path=sphere_urdf_path,
            initial_tf=wp.transform(
                wp.vec3(0.0, 0.0, initial_center_z_m),
                wp.quat_identity(),
            ),
            motion_direction_W=wp.vec3(0.0, 0.0, 1.0),
            approach_speed_m_s=2.5e-2,
            target_force_n=20.0,
            max_sim_time_s=20.0,
        )
        evaluation = evaluate_contact_sensing(fingertip, trial)

    if (
        trial.reaction_force_n is None
        or trial.travel_m is None
        or abs(trial.reaction_force_n - trial.target_force_n) > 1.0
    ):
        raise AssertionError("contact trial did not reach the accepted force")

    np.set_printoptions(precision=9, suppress=True, linewidth=140)
    print("Production Newton -> OptiX sensing path")
    print("side convention: +Y; quadrants Q1,Q2,Q3,Q4 in canonical X-Z")
    _print_state(
        "no contact",
        evaluation.no_contact_response,
        evaluation.no_contact_paths,
    )
    _print_state(
        "20 N contact",
        evaluation.contact_response,
        evaluation.contact_paths,
    )
    print(
        "side-view delta: "
        f"{evaluation.contact_response - evaluation.no_contact_response}"
    )
    print(
        f"mechanics: F={trial.reaction_force_n:.6f} N | "
        f"travel={1.0e3 * trial.travel_m:.6f} mm | "
        f"indentation={1.0e3 * (trial.travel_m - _INITIAL_CLEARANCE_M):.6f} mm | "
        f"ticks={trial.step_count} | t={trial.simulation_time_s:.3f} s"
    )
    print(f"wall runtime: {perf_counter() - wall_start:.3f} s")
    print("Production Newton -> OptiX sensing path: PASS")


if __name__ == "__main__":
    main()
