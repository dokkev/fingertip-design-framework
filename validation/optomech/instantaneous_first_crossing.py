"""Validate production instantaneous first-force-crossing checkpoints."""

from __future__ import annotations

import csv
from importlib.resources import as_file, files
from pathlib import Path
from time import perf_counter

import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.optimization.evaluator import evaluate_full_finger
from lumo.optimization.objective import compute_objectives_from_raw


_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "validation"
    / "instantaneous_first_crossing"
)
_CONTACT_Y_MM = (-11.0, -5.5, 0.0, 11.0)
_FORCE_TARGETS_N = (5.0, 10.0, 15.0, 20.0)
_APPROACH_SPEED_M_S = 5.0e-3


def main() -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    sphere_resource = files("lumo.assets.objects.urdf").joinpath(
        "sphere_10mm.urdf"
    )
    start_s = perf_counter()
    with as_file(sphere_resource) as sphere_path:
        evaluation = evaluate_full_finger(
            Fingertip(FingertipParameters()),
            (sphere_path,),
            (10.0,),
            _CONTACT_Y_MM,
            force_targets_n=_FORCE_TARGETS_N,
            initial_clearance_m=1.0e-3,
            approach_speed_m_s=_APPROACH_SPEED_M_S,
            max_sim_time_s=60.0,
            parallel_world_count=4,
        )
    runtime_s = perf_counter() - start_s

    targets = np.asarray(_FORCE_TARGETS_N, dtype=np.float64)[None, :]
    if evaluation.mechanics_backend != "cuda_graph_parallel_4":
        raise RuntimeError("production evaluator used an unexpected backend")
    if np.any(evaluation.actual_forces_n < targets):
        raise RuntimeError("a saved checkpoint precedes its force crossing")
    if np.any(np.diff(evaluation.indentations_m, axis=1) <= 0.0):
        raise RuntimeError("indentation is not strictly increasing")
    if np.any(np.diff(evaluation.checkpoint_steps, axis=1) <= 0):
        raise RuntimeError("checkpoint steps are not strictly increasing")
    if not np.allclose(
        evaluation.indentation_rates_m_s,
        _APPROACH_SPEED_M_S,
        rtol=1.0e-6,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            "a checkpoint was not captured during constant approach: "
            f"{evaluation.indentation_rates_m_s}"
        )
    if np.any(evaluation.indenter_contact_counts <= 0):
        raise RuntimeError("a checkpoint has no indenter contact")
    if np.any(evaluation.contact_buffer_overflow != 0):
        raise RuntimeError("a checkpoint overflowed the contact buffer")
    if np.any(evaluation.inverted_tet_counts != 0):
        raise RuntimeError("a checkpoint contains an inverted tetrahedron")
    if np.any(evaluation.minimum_det_f <= 0.0):
        raise RuntimeError("a checkpoint has nonpositive det(F)")
    if not np.all(np.isfinite(evaluation.response_matrix)):
        raise RuntimeError("optical responses are non-finite")
    closure_index = evaluation.energy_fields.index("closure_error")
    max_closure_error = float(
        np.max(np.abs(evaluation.energy_matrix[..., closure_index]))
    )
    if max_closure_error > 1.0e-12:
        raise RuntimeError("optical energy ledger does not close")

    contact, observation = compute_objectives_from_raw(vars(evaluation))
    overshoot_n = evaluation.actual_forces_n - targets
    csv_path = _OUTPUT_DIRECTORY / "checkpoints.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "scenario",
                "target_force_n",
                "actual_force_n",
                "overshoot_n",
                "indentation_mm",
                "checkpoint_step",
                "checkpoint_time_s",
                "contact_count",
                "min_det_f",
            )
        )
        for scenario_index, scenario_name in enumerate(
            evaluation.scenario_names
        ):
            for force_index, target_force_n in enumerate(_FORCE_TARGETS_N):
                writer.writerow(
                    (
                        scenario_name,
                        target_force_n,
                        evaluation.actual_forces_n[scenario_index, force_index],
                        overshoot_n[scenario_index, force_index],
                        1.0e3
                        * evaluation.indentations_m[
                            scenario_index, force_index
                        ],
                        evaluation.checkpoint_steps[scenario_index, force_index],
                        evaluation.checkpoint_times_s[scenario_index, force_index],
                        evaluation.indenter_contact_counts[
                            scenario_index, force_index
                        ],
                        evaluation.minimum_det_f[scenario_index, force_index],
                    )
                )

    report_path = _OUTPUT_DIRECTORY / "report.md"
    report_path.write_text(
        "\n".join(
            (
                "# Instantaneous first-crossing production validation",
                "",
                "Result: PASS",
                "",
                f"- backend: `{evaluation.mechanics_backend}`",
                "- loading protocol: `constant_speed_force_thresholds`",
                f"- approach speed: `{_APPROACH_SPEED_M_S:g} m/s`",
                f"- scenarios: `{len(evaluation.scenario_names)}`",
                f"- checkpoints: `{evaluation.actual_forces_n.size}`",
                f"- maximum force overshoot: `{float(overshoot_n.max()):.9f} N`",
                f"- minimum det(F): `{float(evaluation.minimum_det_f.min()):.9f}`",
                f"- maximum energy closure error: `{max_closure_error:.3e}`",
                f"- J_contact: `{contact.J_contact:.9f}`",
                f"- J_obs: `{observation.J_obs:.9f}`",
                f"- total runtime: `{runtime_s:.3f} s`",
                "",
                "Every checkpoint was copied on the first Newton tick whose "
                "reaction force met or exceeded its ordered threshold. No "
                "force-band hold or force-feedback correction was applied.",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    print("Instantaneous first-crossing production validation: PASS")
    print(f"backend={evaluation.mechanics_backend}")
    print(f"maximum overshoot={float(overshoot_n.max()):.6f} N")
    print(f"J_contact={contact.J_contact:.9f}")
    print(f"J_obs={observation.J_obs:.9f}")
    print(f"runtime={runtime_s:.3f} s")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
