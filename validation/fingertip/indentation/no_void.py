"""Run the resumable Phase 4J no-void external-contact-only queue."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

from fem.indentation import (
    IndentationSettings,
    inspect_indentation_runtime_contract,
    run_indentation_case,
)
from fem.kratos_settings import (
    CONSTITUTIVE_LAW,
    MIXED_PAD_ELEMENT,
    POISSON_RATIO,
)
from validation.common.io import (
    atomic_write_json,
    strict_read_json,
    write_indentation_case_outputs,
)
from validation.common.runner import run_isolated
from validation.fingertip.indentation.metrics import (
    achieved_contact_centroid,
    append_or_replace_case,
    completed_case_result,
    no_void_geometry_contract,
    reaction_work_proxy,
)
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


DEFAULT_OUTPUT = Path("output/validation/fingertip/indentation/no_void")
J0_INDENTATION_MM = 0.25 / 48.0
CANONICAL_INDENTATION_MM = 1.5
CANONICAL_STEPS = 48


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_case-name", help=argparse.SUPPRESS)
    parser.add_argument("--_case-directory", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_mesh-level", choices=("medium", "fine"))
    parser.add_argument("--_indentation-mm", type=float)
    parser.add_argument("--_steps", type=int)
    return parser.parse_args()


def _smoke_acceptance(result: Mapping[str, Any]) -> dict[str, bool]:
    history = result.get("history", [])
    final = history[-1] if history else {}
    external = final.get("contact_groups", {}).get(
        "external_pad_indenter", {}
    )
    det_f = final.get("pad_strain_det_f", {}).get("det_f", {})
    return {
        "first_step_converged": result.get("solve_status") == "PASS"
        and len(history) == 1,
        "finite_fields": bool(final.get("finite_fields")),
        "positive_reaction_sign": float(
            final.get("indenter_normal_reaction_n", 0.0)
        )
        > 0.0,
        "positive_det_f": det_f.get("nonpositive_count") == 0,
        "external_contact_active": int(
            external.get("active_condition_count", 0)
        )
        > 0,
        "no_internal_contact_lm_assembly": bool(
            result.get("assembled_contact_lm_contract", {}).get(
                "no_internal_contact_lm_assembly"
            )
        ),
    }


def _run_child(arguments: argparse.Namespace) -> int:
    required = (
        arguments._case_name,
        arguments._case_directory,
        arguments._mesh_level,
        arguments._indentation_mm,
        arguments._steps,
    )
    if any(value is None for value in required):
        raise ValueError("internal child arguments are incomplete")
    case_name = str(arguments._case_name)
    case_directory = Path(arguments._case_directory).resolve()
    model = FingertipModel(FingertipParameters())
    result, artifacts = run_indentation_case(
        model,
        arguments._mesh_level,
        IndentationSettings(
            float(arguments._indentation_mm),
            int(arguments._steps),
        ),
        internal_contact_configuration="none",
    )
    original_status = result["status"]
    result.update(
        {
            "phase": "4J",
            "case_name": case_name,
            "solver_case_status": original_status,
            "no_void_geometry_contract": no_void_geometry_contract(model),
            "strain_energy": {
                "available": False,
                "reason": (
                    "Element STRAIN_ENERGY was not an accepted Phase 1/2 "
                    "runtime output and is not introduced as a new acceptance "
                    "metric in Phase 4J."
                ),
            },
            "external_reaction_work": reaction_work_proxy(
                result.get("history", [])
            ),
        }
    )
    if artifacts is not None and result.get("history"):
        active_ids = result["history"][-1]["contact_groups"][
            "external_pad_indenter"
        ]["active_slave_node_ids"]
        result["achieved_contact_centroid"] = achieved_contact_centroid(
            artifacts.mesh, active_ids
        )
    else:
        result["achieved_contact_centroid"] = {
            "available": False,
            "reference_centroid_mm": None,
            "active_slave_node_count": 0,
        }
    if case_name == "J0":
        smoke = _smoke_acceptance(result)
        result["smoke_acceptance_checks"] = smoke
        result["status"] = "PASS" if all(smoke.values()) else "FAIL"
        if result["status"] == "PASS":
            result.pop("failure_reason", None)
        else:
            result["failure_reason"] = "j0_smoke_acceptance_failed"

    outputs = write_indentation_case_outputs(
        result, artifacts, case_directory
    )
    result["outputs"] = outputs
    atomic_write_json(case_directory / "result.json", result)
    print(
        f"Phase 4J {case_name}: {result['status']} "
        f"({len(result.get('history', []))}/{arguments._steps} steps)"
    )
    return 0 if result["status"] == "PASS" else 1


def _case_command(
    output: Path,
    case_name: str,
    case_directory: Path,
    mesh_level: str,
    indentation_mm: float,
    steps: int,
) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "validation.fingertip.indentation.no_void",
        "--output-directory",
        str(output),
        "--_child",
        "--_case-name",
        case_name,
        "--_case-directory",
        str(case_directory),
        "--_mesh-level",
        mesh_level,
        "--_indentation-mm",
        str(indentation_mm),
        "--_steps",
        str(steps),
    ]


def _run_case(
    output: Path,
    run_state: dict[str, Any],
    *,
    case_name: str,
    directory_name: str,
    mesh_level: str,
    indentation_mm: float,
    steps: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    case_directory = output / directory_name
    result_path = case_directory / "result.json"
    existing = completed_case_result(result_path, case_name)
    command = _case_command(
        output,
        case_name,
        case_directory,
        mesh_level,
        indentation_mm,
        steps,
    )
    if existing is not None:
        postprocess_recovered = (
            existing.get("postprocess_recovery", {}).get("status") == "PASS"
        )
        checkpoint = {
            "case_name": case_name,
            "command": command,
            "start_time": None,
            "end_time": _now(),
            "duration_seconds": 0.0,
            "exit_code": existing.get("process_exit_code"),
            "status": existing["status"],
            "retry_count": existing.get("retry_count", 0),
            "artifact_directory": str(case_directory.resolve()),
            "failure_reason": existing.get("failure_reason"),
            "resumed_from_valid_artifact": True,
            "postprocess_recovered": postprocess_recovered,
            "process_exit_context": (
                "Numerical solve PASS; the original nonzero exit occurred "
                "after checkpoint and its output was recovered."
                if postprocess_recovered
                else None
            ),
        }
        append_or_replace_case(run_state, checkpoint)
        atomic_write_json(output / "run_state.json", run_state)
        return existing

    case_directory.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONFAULTHANDLER": "1",
        }
    )
    started_at = _now()
    start = time.perf_counter()
    final_exit_code: int | None = None
    retry_count = 0
    timed_out = False
    for attempt in range(2):
        attempt_log = case_directory / f"solver_attempt_{attempt}.log"
        completed = run_isolated(
            command,
            cwd=Path(__file__).resolve().parents[3],
            environment=environment,
            output_path=attempt_log,
            timeout_seconds=timeout_seconds,
            disable_core_dumps=True,
        )
        final_exit_code = completed.return_code
        timed_out = timed_out or completed.timed_out
        if result_path.is_file():
            break
        if attempt == 0:
            retry_count = 1
            continue
        break

    duration = time.perf_counter() - start
    if result_path.is_file():
        result = strict_read_json(result_path)
        recovered_postprocess = (
            result.get("postprocess_recovery", {}).get("status") == "PASS"
        )
        if (
            result.get("status") == "PASS"
            and final_exit_code not in (None, 0)
            and not recovered_postprocess
        ):
            result["status"] = "FAIL"
            result["failure_reason"] = (
                "child_exited_nonzero_after_writing_pass_result"
            )
        result["process_exit_code"] = final_exit_code
        result["retry_count"] = retry_count
        result["reproduction_command"] = command
        result.setdefault("outputs", {})["solver_log"] = str(
            (case_directory / f"solver_attempt_{retry_count}.log").resolve()
        )
    else:
        result = {
            "phase": "4J",
            "case_name": case_name,
            "mesh_level": mesh_level,
            "status": "TIMEOUT" if timed_out else "FAIL",
            "solve_status": "FAIL",
            "history": [],
            "failure_reason": (
                "timeout_after_one_retry"
                if timed_out
                else "child_terminated_without_result_after_one_retry"
            ),
            "process_exit_code": final_exit_code,
            "retry_count": retry_count,
            "reproduction_command": command,
        }
    atomic_write_json(result_path, result)
    checkpoint = {
        "case_name": case_name,
        "command": command,
        "start_time": started_at,
        "end_time": _now(),
        "duration_seconds": duration,
        "exit_code": final_exit_code,
        "status": result["status"],
        "retry_count": retry_count,
        "artifact_directory": str(case_directory.resolve()),
        "failure_reason": result.get("failure_reason"),
        "resumed_from_valid_artifact": False,
    }
    append_or_replace_case(run_state, checkpoint)
    atomic_write_json(output / "run_state.json", run_state)
    return result


def _skip_case(
    output: Path,
    run_state: dict[str, Any],
    case_name: str,
    directory_name: str,
    reason: str,
) -> dict[str, Any]:
    directory = output / directory_name
    existing = completed_case_result(directory / "result.json", case_name)
    if existing is not None:
        record = {
            "case_name": case_name,
            "command": [],
            "start_time": None,
            "end_time": _now(),
            "duration_seconds": 0.0,
            "exit_code": existing.get("process_exit_code"),
            "status": existing["status"],
            "retry_count": existing.get("retry_count", 0),
            "artifact_directory": str(directory.resolve()),
            "failure_reason": existing.get("reason")
            or existing.get("failure_reason"),
            "resumed_from_valid_artifact": True,
        }
        append_or_replace_case(run_state, record)
        atomic_write_json(output / "run_state.json", run_state)
        return existing
    directory.mkdir(parents=True, exist_ok=True)
    result = {
        "phase": "4J",
        "case_name": case_name,
        "status": "SKIPPED",
        "reason": reason,
    }
    atomic_write_json(directory / "result.json", result)
    record = {
        "case_name": case_name,
        "command": [],
        "start_time": None,
        "end_time": _now(),
        "duration_seconds": 0.0,
        "exit_code": None,
        "status": "SKIPPED",
        "retry_count": 0,
        "artifact_directory": str(directory.resolve()),
        "failure_reason": reason,
        "resumed_from_valid_artifact": False,
    }
    append_or_replace_case(run_state, record)
    atomic_write_json(output / "run_state.json", run_state)
    return result


def _write_csv(
    path: Path,
    columns: list[str],
    rows: list[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _baseline_row(
    case_name: str, result: Mapping[str, Any]
) -> dict[str, Any]:
    final = result.get("final") or (
        result.get("history", [])[-1] if result.get("history") else {}
    )
    return {
        "case": case_name,
        "mesh_level": result.get("mesh_level"),
        "status": result.get("status"),
        "completed_steps": len(result.get("history", [])),
        "final_indentation_mm": final.get("achieved_indentation_mm"),
        "final_reaction_n": final.get("indenter_normal_reaction_n"),
        "minimum_det_f": result.get("minimum_pad_det_f"),
        "maximum_principal_strain": result.get("maximum_pad_strain"),
        "maximum_nonlinear_iterations": result.get(
            "maximum_nonlinear_iterations"
        ),
        "solve_wall_clock_seconds": result.get(
            "solve_wall_clock_seconds"
        ),
        "external_work_n_mm": result.get(
            "external_reaction_work", {}
        ).get("value_n_mm"),
        "strain_energy_available": result.get("strain_energy", {}).get(
            "available"
        ),
        "failure_reason": result.get("failure_reason"),
    }


def _relative_difference(first: float, second: float) -> float:
    return abs(first - second) / abs(second) if second else float("inf")


def _run_parent(arguments: argparse.Namespace) -> int:
    output = arguments.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "run_state.json"
    if state_path.is_file():
        run_state = strict_read_json(state_path)
    else:
        run_state = {
            "phase": "4J",
            "status": "RUNNING",
            "created_at": _now(),
            "cases": [],
        }
        atomic_write_json(state_path, run_state)

    model = FingertipModel(FingertipParameters())
    geometry_contract = no_void_geometry_contract(model)
    preflight_path = output / "preflight.json"
    if preflight_path.is_file():
        preflight = strict_read_json(preflight_path)
    else:
        runtime = inspect_indentation_runtime_contract(
            model,
            "medium",
            IndentationSettings(J0_INDENTATION_MM, 1),
            internal_contact_configuration="none",
        )
        registration = runtime["internal_contact_registration"]
        external = runtime["runtime_contact_contract"]["groups"][
            "external_pad_indenter"
        ]
        checks = {
            "no_void_geometry": geometry_contract["pass"],
            "no_internal_contact_process": (
                runtime["contact_process_count"] == 1
                and not registration["registered_group_names"]
            ),
            "no_internal_contact_submodel_part": not registration[
                "internal_contact_submodel_parts_present"
            ],
            "no_internal_generated_contact_condition": (
                set(runtime["runtime_contact_contract"]["groups"])
                == {"external_pad_indenter"}
            ),
            "external_slave_role": external["checks"][
                "slave_runtime_role"
            ],
            "external_master_role": external["checks"][
                "master_runtime_role"
            ],
            "external_normal_finite": all(
                abs(float(value)) < float("inf")
                for value in (
                    *external["slave_mean_runtime_normal"],
                    *external["master_mean_runtime_normal"],
                )
            ),
            "mixed_t3_element": MIXED_PAD_ELEMENT
            == "TotalLagrangianMixedVolumetricStrainElement2D3N",
            "poisson_ratio_0p49": POISSON_RATIO == 0.49,
        }
        preflight = {
            "phase": "4J",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "geometry_contract": geometry_contract,
            "runtime": runtime,
            "formulation": {
                "element": MIXED_PAD_ELEMENT,
                "constitutive_law": CONSTITUTIVE_LAW,
                "poisson_ratio": POISSON_RATIO,
            },
            "checks": checks,
            "lm_dof_caveat": {
                "literal_absence_on_internal_semantic_nodes": (
                    registration[
                        "internal_source_nodes_with_root_level_lm_dof"
                    ]
                    == 0
                ),
                "explanation": registration[
                    "root_level_lm_dof_explanation"
                ],
                "acceptance_interpretation": (
                    "Phase 4J requires no internal contact-coupled LM "
                    "assembly. J0 records the actual assembled DOF set."
                ),
            },
        }
        atomic_write_json(preflight_path, preflight)

    j0 = _run_case(
        output,
        run_state,
        case_name="J0",
        directory_name="j0_smoke",
        mesh_level="medium",
        indentation_mm=J0_INDENTATION_MM,
        steps=1,
        timeout_seconds=1800,
    )
    j1 = _run_case(
        output,
        run_state,
        case_name="J1",
        directory_name="j1_medium",
        mesh_level="medium",
        indentation_mm=CANONICAL_INDENTATION_MM,
        steps=CANONICAL_STEPS,
        timeout_seconds=7200,
    )
    if j1["status"] == "PASS":
        j2 = _run_case(
            output,
            run_state,
            case_name="J2",
            directory_name="j2_fine",
            mesh_level="fine",
            indentation_mm=CANONICAL_INDENTATION_MM,
            steps=CANONICAL_STEPS,
            timeout_seconds=7200,
        )
    else:
        j2 = _skip_case(
            output,
            run_state,
            "J2",
            "j2_fine",
            "J1 medium baseline did not pass the fine-mesh gate.",
        )
    j3 = _skip_case(
        output,
        run_state,
        "J3",
        "j3_location_sweep",
        (
            "The repository roadmap defines only the symbolic location x_c; "
            "it contains no discrete, documented contact-location cases."
        ),
    )
    j4 = _skip_case(
        output,
        run_state,
        "J4",
        "j4_parameter_sweep",
        (
            "The repository contains no documented no-void mechanics "
            "candidate list. A new design space is outside the bounded task."
        ),
    )

    baseline_rows = [
        _baseline_row(name, result)
        for name, result in (("J0", j0), ("J1", j1), ("J2", j2))
    ]
    _write_csv(
        output / "baseline_comparison.csv",
        list(baseline_rows[0]),
        baseline_rows,
    )
    _write_csv(
        output / "contact_location_sweep.csv",
        ["case", "status", "prescribed_location", "reason"],
        [
            {
                "case": "J3",
                "status": j3["status"],
                "prescribed_location": None,
                "reason": j3["reason"],
            }
        ],
    )

    comparison: dict[str, Any] = {"available": False}
    if j1.get("history") and j2.get("history"):
        medium = float(j1["history"][-1]["indenter_normal_reaction_n"])
        fine = float(j2["history"][-1]["indenter_normal_reaction_n"])
        comparison = {
            "available": True,
            "medium_final_reaction_n": medium,
            "fine_final_reaction_n": fine,
            "relative_difference": _relative_difference(medium, fine),
            "below_10_percent": _relative_difference(medium, fine) < 0.10,
        }

    if preflight["status"] != "PASS" or j1["status"] != "PASS":
        phase_status = "FAIL"
    elif j2["status"] == "PASS":
        phase_status = "PASS"
    else:
        phase_status = "CONDITIONAL_PASS"
    summary = {
        "phase": "4J",
        "status": phase_status,
        "phase4i_internal_contact_baseline": "BLOCKED",
        "phase4ig_crosspoint_treatment": "FAIL_BLOCKED",
        "preflight_status": preflight["status"],
        "case_status": {
            "J0": j0["status"],
            "J1": j1["status"],
            "J2": j2["status"],
            "J3": j3["status"],
            "J4": j4["status"],
        },
        "medium_fine_reaction": comparison,
        "strain_energy_policy": (
            "Unavailable; external reaction work is stored separately and "
            "is not relabeled as element strain energy."
        ),
        "location_sweep": j3,
        "parameter_sweep": j4,
        "queue_continuation": (
            "Independent J3/J4 decisions were recorded even when an earlier "
            "numerical case failed; J2 alone is gated on J1."
        ),
        "artifacts": {
            "run_state": str(state_path),
            "preflight": str(preflight_path),
            "baseline_comparison": str(
                (output / "baseline_comparison.csv").resolve()
            ),
            "contact_location_sweep": str(
                (output / "contact_location_sweep.csv").resolve()
            ),
        },
    }
    atomic_write_json(output / "summary.json", summary)
    run_state["status"] = "COMPLETE"
    run_state["completed_at"] = _now()
    run_state["phase_status"] = phase_status
    atomic_write_json(state_path, run_state)
    print(output / "summary.json")
    return 0 if phase_status in {"PASS", "CONDITIONAL_PASS"} else 1


def main() -> int:
    arguments = _arguments()
    if arguments._child:
        return _run_child(arguments)
    return _run_parent(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
