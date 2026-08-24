"""Compare adaptive force-servo settling with the former 5 s dwell."""

from __future__ import annotations

import csv
from contextlib import ExitStack
from importlib.resources import as_file, files
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.optimization import sensing_objectives
from lumo.optimization.evaluator import evaluate_contact_sensing
from lumo.simulation import DesignTrial


_INITIAL_CLEARANCE_M = 1.0e-3
_APPROACH_SPEED_M_S = 5.0e-3
_MAX_SIM_TIME_S = 60.0
_ADAPTIVE_DURATION_S = 0.2
_ADAPTIVE_DISPLACEMENT_TOLERANCE_M = 1.0e-6
_REFERENCE_DURATION_S = 5.0
_SPHERES = (
    ("sphere_5mm.urdf", 5.0),
    ("sphere_10mm.urdf", 10.0),
    ("sphere_20mm.urdf", 20.0),
)
_OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "validation"
    / "adaptive_settling.csv"
)


def _make_trials(
    fingertip: Fingertip,
    sphere_resources: tuple[tuple[Path, float], ...],
) -> list[DesignTrial]:
    trials = []
    for urdf_path, diameter_mm in sphere_resources:
        radius_m = 0.5e-3 * diameter_mm
        trials.append(
            DesignTrial(
                name=f"sphere_{diameter_mm:g}mm_center",
                urdf_path=urdf_path,
                initial_tf=wp.transform(
                    wp.vec3(
                        0.0,
                        0.0,
                        fingertip.tip_z_m - _INITIAL_CLEARANCE_M - radius_m,
                    ),
                    wp.quat_identity(),
                ),
                motion_direction_W=wp.vec3(0.0, 0.0, 1.0),
                approach_speed_m_s=_APPROACH_SPEED_M_S,
                target_force_n=20.0,
                max_sim_time_s=_MAX_SIM_TIME_S,
                initial_clearance_m=_INITIAL_CLEARANCE_M,
            )
        )
    return trials


def _checkpoint_intervals(checkpoint_times_s: np.ndarray) -> np.ndarray:
    return np.diff(
        np.column_stack(
            (
                np.zeros(len(checkpoint_times_s), dtype=np.float64),
                checkpoint_times_s,
            )
        ),
        axis=1,
    )


def main() -> None:
    fingertip = Fingertip(FingertipParameters())
    resource_root = files("lumo").joinpath("assets", "objects", "urdf")
    with ExitStack() as resources:
        sphere_resources = tuple(
            (
                resources.enter_context(as_file(resource_root.joinpath(filename))),
                diameter_mm,
            )
            for filename, diameter_mm in _SPHERES
        )

        reference_start_s = perf_counter()
        reference = evaluate_contact_sensing(
            fingertip,
            _make_trials(fingertip, sphere_resources),
            settle_duration_s=_REFERENCE_DURATION_S,
            settle_displacement_tolerance_m=None,
        )
        reference_wall_s = perf_counter() - reference_start_s

        adaptive_start_s = perf_counter()
        adaptive = evaluate_contact_sensing(
            fingertip,
            _make_trials(fingertip, sphere_resources),
            settle_duration_s=_ADAPTIVE_DURATION_S,
            settle_displacement_tolerance_m=(_ADAPTIVE_DISPLACEMENT_TOLERANCE_M),
        )
        adaptive_wall_s = perf_counter() - adaptive_start_s

    if reference.scenario_names != adaptive.scenario_names:
        raise RuntimeError("reference and adaptive scenario orders differ")
    if not np.array_equal(reference.force_targets_n, adaptive.force_targets_n):
        raise RuntimeError("reference and adaptive force targets differ")

    reference_intervals_s = _checkpoint_intervals(reference.checkpoint_times_s)
    adaptive_intervals_s = _checkpoint_intervals(adaptive.checkpoint_times_s)
    fieldnames = (
        "scenario",
        "sphere_diameter_mm",
        "target_force_n",
        "reference_checkpoint_time_s",
        "adaptive_checkpoint_time_s",
        "reference_interval_s",
        "adaptive_interval_s",
        "reference_force_n",
        "adaptive_force_n",
        "reference_indentation_mm",
        "adaptive_indentation_mm",
        "indentation_difference_mm",
        "quadrant_l2_difference",
        "side_power_difference",
    )
    rows = []
    print("Adaptive settled-state comparison against the former 5 s dwell")
    for sphere_index, ((_, diameter_mm), scenario_name) in enumerate(
        zip(_SPHERES, adaptive.scenario_names, strict=True)
    ):
        print(f"{scenario_name}")
        for force_index, target_force_n in enumerate(adaptive.force_targets_n):
            reference_response = reference.response_matrix[
                sphere_index,
                force_index,
            ]
            adaptive_response = adaptive.response_matrix[sphere_index, force_index]
            indentation_difference_mm = 1.0e3 * (
                adaptive.indentations_m[sphere_index, force_index]
                - reference.indentations_m[sphere_index, force_index]
            )
            quadrant_l2_difference = float(
                np.linalg.norm(adaptive_response - reference_response)
            )
            side_power_difference = float(
                adaptive_response.sum() - reference_response.sum()
            )
            row = {
                "scenario": scenario_name,
                "sphere_diameter_mm": diameter_mm,
                "target_force_n": target_force_n,
                "reference_checkpoint_time_s": reference.checkpoint_times_s[
                    sphere_index,
                    force_index,
                ],
                "adaptive_checkpoint_time_s": adaptive.checkpoint_times_s[
                    sphere_index,
                    force_index,
                ],
                "reference_interval_s": reference_intervals_s[
                    sphere_index,
                    force_index,
                ],
                "adaptive_interval_s": adaptive_intervals_s[
                    sphere_index,
                    force_index,
                ],
                "reference_force_n": reference.actual_forces_n[
                    sphere_index,
                    force_index,
                ],
                "adaptive_force_n": adaptive.actual_forces_n[
                    sphere_index,
                    force_index,
                ],
                "reference_indentation_mm": 1.0e3
                * reference.indentations_m[sphere_index, force_index],
                "adaptive_indentation_mm": 1.0e3
                * adaptive.indentations_m[sphere_index, force_index],
                "indentation_difference_mm": indentation_difference_mm,
                "quadrant_l2_difference": quadrant_l2_difference,
                "side_power_difference": side_power_difference,
            }
            rows.append(row)
            print(
                f"  target={target_force_n:4.1f} N | "
                f"interval adaptive/reference="
                f"{row['adaptive_interval_s']:.3f}/{row['reference_interval_s']:.3f} s | "
                f"F={row['adaptive_force_n']:.6f}/{row['reference_force_n']:.6f} N | "
                f"d_indent={indentation_difference_mm:+.6f} mm | "
                f"dQ_L2={quadrant_l2_difference:.9e} | "
                f"dP_side={side_power_difference:+.9e}"
            )

    reference_objectives = sensing_objectives(
        reference.response_matrix,
        no_contact_response=reference.no_contact_response,
    )
    adaptive_objectives = sensing_objectives(
        adaptive.response_matrix,
        no_contact_response=adaptive.no_contact_response,
    )
    reference_intensity_by_sphere = reference_objectives[0]
    reference_spatial_by_sphere = reference_objectives[1]
    adaptive_intensity_by_sphere = adaptive_objectives[0]
    adaptive_spatial_by_sphere = adaptive_objectives[1]
    print("per-sphere objective differences (adaptive - reference):")
    for sphere_index, (_, diameter_mm) in enumerate(_SPHERES):
        print(
            f"  {diameter_mm:g} mm: "
            f"dJ_intensity="
            f"{adaptive_intensity_by_sphere[sphere_index] - reference_intensity_by_sphere[sphere_index]:+.9e} | "
            f"dJ_spatial="
            f"{adaptive_spatial_by_sphere[sphere_index] - reference_spatial_by_sphere[sphere_index]:+.9e}"
        )
    print(
        "worst-diameter objectives: "
        f"J_intensity adaptive/reference="
        f"{adaptive_objectives[2]:.9e}/{reference_objectives[2]:.9e} "
        f"(difference={adaptive_objectives[2] - reference_objectives[2]:+.9e})"
    )
    print(
        "                           "
        f"J_spatial adaptive/reference="
        f"{adaptive_objectives[3]:.9e}/{reference_objectives[3]:.9e} "
        f"(difference={adaptive_objectives[3] - reference_objectives[3]:+.9e})"
    )
    print(
        f"wall runtime adaptive/reference="
        f"{adaptive_wall_s:.3f}/{reference_wall_s:.3f} s | "
        f"speedup={reference_wall_s / adaptive_wall_s:.3f}x"
    )

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
