"""Validate the production 3-sphere by 3-location sensing matrix."""

from __future__ import annotations

from contextlib import ExitStack
from importlib.resources import as_file, files
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.fingertip.geometric_param import semiellipse_depth_at_x_mm
from lumo.optimization.evaluator import evaluate_contact_sensing
from lumo.simulation import DesignTrial


_INITIAL_CLEARANCE_M = 1.0e-3
_SPHERES = (
    ("sphere_5mm.urdf", 5.0),
    ("sphere_10mm.urdf", 10.0),
    ("sphere_20mm.urdf", 20.0),
)
_CONTACT_X_MM = (-7.5, 0.0, 7.5)


def _print_energy_ledger(fields: tuple[str, ...], energy: np.ndarray) -> None:
    values = " | ".join(
        f"{name}={value:.9e}" for name, value in zip(fields, energy, strict=True)
    )
    print(f"    energy: {values}")


def _validate_optical_state(
    label: str,
    response: np.ndarray,
    energy_fields: tuple[str, ...],
    energy: np.ndarray,
) -> None:
    if response.shape != (4,) or not np.all(np.isfinite(response)):
        raise AssertionError(f"{label} has an invalid side-view response")
    if energy.shape != (len(energy_fields),) or not np.all(np.isfinite(energy)):
        raise AssertionError(f"{label} has an invalid energy ledger")
    closure_error = energy[energy_fields.index("closure_error")]
    if abs(closure_error) > 1.0e-12:
        raise AssertionError(f"{label} path-energy ledger does not close")


def _make_trial(
    fingertip: Fingertip,
    urdf_path: Path,
    urdf_filename: str,
    *,
    sphere_diameter_mm: float,
    contact_x_mm: float,
) -> DesignTrial:
    surface_z_mm = fingertip.silicone.ellipse_center_z_mm - semiellipse_depth_at_x_mm(
        half_width_mm=fingertip.silicone.ellipse_radius_x_mm,
        height_mm=fingertip.silicone.ellipse_radius_z_mm,
        x_mm=contact_x_mm,
    )
    sphere_radius_m = 0.5e-3 * sphere_diameter_mm
    initial_center_z_m = 1.0e-3 * surface_z_mm - _INITIAL_CLEARANCE_M - sphere_radius_m
    return DesignTrial(
        name=f"{Path(urdf_filename).stem}_x{contact_x_mm:+g}mm",
        urdf_path=urdf_path,
        initial_tf=wp.transform(
            wp.vec3(1.0e-3 * contact_x_mm, 0.0, initial_center_z_m),
            wp.quat_identity(),
        ),
        motion_direction_W=wp.vec3(0.0, 0.0, 1.0),
        approach_speed_m_s=2.5e-2,
        target_force_n=20.0,
        max_sim_time_s=30.0,
        initial_clearance_m=_INITIAL_CLEARANCE_M,
    )


def main() -> None:
    wall_start = perf_counter()
    fingertip = Fingertip(FingertipParameters())
    resource_root = files("lumo").joinpath(
        "assets",
        "objects",
        "urdf",
    )

    with ExitStack() as resources:
        trials = []
        for urdf_filename, sphere_diameter_mm in _SPHERES:
            urdf_path = resources.enter_context(
                as_file(resource_root.joinpath(urdf_filename))
            )
            for contact_x_mm in _CONTACT_X_MM:
                trials.append(
                    _make_trial(
                        fingertip,
                        urdf_path,
                        urdf_filename,
                        sphere_diameter_mm=sphere_diameter_mm,
                        contact_x_mm=contact_x_mm,
                    )
                )
        evaluation = evaluate_contact_sensing(fingertip, trials)

    np.set_printoptions(precision=9, suppress=True, linewidth=140)
    print("Production Newton -> OptiX 3 x 3 x 4 sensing matrix")
    print("side convention: +Y; quadrants Q1,Q2,Q3,Q4 in canonical X-Z")
    _validate_optical_state(
        "no contact",
        evaluation.no_contact_response,
        evaluation.energy_fields,
        evaluation.no_contact_energy,
    )
    print(f"no-contact Q: {evaluation.no_contact_response}")
    _print_energy_ledger(
        evaluation.energy_fields,
        evaluation.no_contact_energy,
    )
    expected_names = tuple(trial.name for trial in trials)
    if evaluation.scenario_names != expected_names:
        raise AssertionError("response-matrix rows do not match scenario order")

    for scenario_index, trial in enumerate(trials):
        print(trial.name)
        if trial.reaction_force_n is None or trial.travel_m is None:
            raise AssertionError(f"{trial.name} has no final mechanics result")
        for force_index, target_force_n in enumerate(evaluation.force_targets_n):
            response = evaluation.response_matrix[scenario_index, force_index]
            energy = evaluation.energy_matrix[scenario_index, force_index]
            actual_force_n = evaluation.actual_forces_n[
                scenario_index,
                force_index,
            ]
            tolerance_n = 0.1 * target_force_n
            if abs(actual_force_n - target_force_n) > tolerance_n:
                raise AssertionError(
                    f"{trial.name} missed the {target_force_n:g} N checkpoint"
                )
            _validate_optical_state(
                f"{trial.name} at {target_force_n:g} N",
                response,
                evaluation.energy_fields,
                energy,
            )
            print(
                f"  target={target_force_n:4.1f} N | "
                f"F={actual_force_n:9.6f} N | "
                f"indentation="
                f"{1.0e3 * evaluation.indentations_m[scenario_index, force_index]:.6f} mm"
            )
            print(f"    Q:     {response}")
            print(f"    delta: {response - evaluation.no_contact_response}")
            _print_energy_ledger(evaluation.energy_fields, energy)
        print(f"  scenario wall={evaluation.scenario_runtime_s[scenario_index]:.3f} s")

    if evaluation.response_matrix.shape != (9, 4, 4):
        raise AssertionError("response matrix must have shape (9, 4, 4)")
    print("scenario x force x quadrant response matrix:")
    print(evaluation.response_matrix)
    print(f"wall runtime: {perf_counter() - wall_start:.3f} s")
    print("Production Newton -> OptiX 3 x 3 x 4 sensing matrix: PASS")


if __name__ == "__main__":
    main()
