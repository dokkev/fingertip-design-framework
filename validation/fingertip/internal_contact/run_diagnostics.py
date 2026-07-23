"""Run isolated Phase 4I-D contact configurations in child processes."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from typing import Any, Mapping, Sequence

from validation.fingertip.internal_contact.diagnostics import (
    CASE_CONFIGURATIONS,
    CASE_DIRECTORY_NAMES,
    assemble_first_step_diagnostics,
    common_settings,
    configuration_for_case,
    run_continuous_u_full_trial,
    run_first_step_case,
)


DEFAULT_OUTPUT = Path("output/validation/fingertip/internal_contact/diagnostics")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(CASE_CONFIGURATIONS),
    )
    selection.add_argument(
        "--case", choices=tuple(CASE_CONFIGURATIONS)
    )
    parser.add_argument(
        "--mesh-level",
        choices=("medium",),
        default="medium",
        help="Phase 4I-D holds the mesh at medium",
    )
    parser.add_argument("--first-step-only", action="store_true")
    parser.add_argument("--run-full-trial", action="store_true")
    parser.add_argument("--indentation-mm", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT
    )
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_child-case", help=argparse.SUPPRESS)
    parser.add_argument(
        "--_case-directory", type=Path, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--_full-trial-child", action="store_true", help=argparse.SUPPRESS
    )
    return parser.parse_args()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_value,
        )
        + "\n",
        encoding="utf-8",
    )


def _csv_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _write_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(
        {key for record in records for key in record}
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        if not fields:
            stream.write("")
            return
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {field: _csv_cell(record.get(field)) for field in fields}
            )


def _run_case_child(arguments: argparse.Namespace) -> int:
    if not arguments._child_case or arguments._case_directory is None:
        raise ValueError("internal child case and directory are required")
    case = arguments._child_case
    directory = arguments._case_directory
    directory.mkdir(parents=True, exist_ok=True)
    diagnostic, dof_rows, contact_records = (
        assemble_first_step_diagnostics(case, arguments.mesh_level)
    )
    _write_json(directory / "runtime_contract.json", diagnostic)
    _write_json(
        directory / "matrix_diagnostics.json",
        diagnostic.get(
            "matrix_diagnostics",
            {
                "available": False,
                "reason": diagnostic.get("failure_reason"),
                "exception": diagnostic.get("exception"),
            },
        ),
    )
    _write_json(
        directory / "corner_contract.json",
        diagnostic.get(
            "corner_contract",
            {
                "available": False,
                "reason": diagnostic.get("failure_reason"),
            },
        ),
    )
    _write_records(directory / "dof_map.csv", dof_rows)
    _write_records(
        directory / "contact_conditions.csv", contact_records
    )
    first_step = run_first_step_case(case, arguments.mesh_level)
    _write_json(directory / "first_step_result.json", first_step)
    combined = {
        "phase": "4I-D",
        "case": case,
        "configuration": configuration_for_case(case),
        "diagnostic": diagnostic,
        "first_step": first_step,
        "status": first_step["status"],
        "outputs": {
            "runtime_contract": str(
                (directory / "runtime_contract.json").resolve()
            ),
            "matrix_diagnostics": str(
                (directory / "matrix_diagnostics.json").resolve()
            ),
            "corner_contract": str(
                (directory / "corner_contract.json").resolve()
            ),
            "dof_map": str((directory / "dof_map.csv").resolve()),
            "contact_conditions": str(
                (directory / "contact_conditions.csv").resolve()
            ),
            "first_step_result": str(
                (directory / "first_step_result.json").resolve()
            ),
        },
    }
    _write_json(directory / "result.json", combined)
    print(
        f"PHASE4ID_CHILD_RESULT {case} {combined['status']}",
        flush=True,
    )
    return 0 if combined["status"] == "PASS" else 1


def _history_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in result.get("history", []):
        groups = point.get("contact_groups", {})
        external = groups.get("external_pad_indenter", {})
        internal = groups.get("internal_u", {})
        semantic = internal.get("semantic_regions", {})
        rows.append(
            {
                "step": point.get("step"),
                "prescribed_indenter_travel_mm": point.get(
                    "prescribed_indenter_travel_mm"
                ),
                "achieved_indentation_mm": point.get(
                    "achieved_indentation_mm"
                ),
                "indenter_normal_reaction_n": point.get(
                    "indenter_normal_reaction_n"
                ),
                "support_signed_reaction_along_loading_n": point.get(
                    "support_signed_reaction_along_loading_n"
                ),
                "force_equilibrium_error": point.get(
                    "force_equilibrium_error"
                ),
                "nonlinear_iterations": point.get(
                    "nonlinear_iterations"
                ),
                "active_set_converged": point.get(
                    "active_set_converged"
                ),
                "finite_fields": point.get("finite_fields"),
                "minimum_det_f": point.get(
                    "pad_strain_det_f", {}
                ).get("det_f", {}).get("min"),
                "external_generated_conditions": external.get(
                    "generated_condition_count"
                ),
                "external_active_conditions": external.get(
                    "active_condition_count"
                ),
                "internal_u_generated_conditions": internal.get(
                    "generated_condition_count"
                ),
                "internal_u_active_conditions": internal.get(
                    "active_condition_count"
                ),
                "internal_left_active_nodes": semantic.get(
                    "left", {}
                ).get("active_node_count"),
                "internal_bottom_active_nodes": semantic.get(
                    "bottom", {}
                ).get("active_node_count"),
                "internal_right_active_nodes": semantic.get(
                    "right", {}
                ).get("active_node_count"),
                "external_penetration_pass": external.get(
                    "penetration_pass"
                ),
                "internal_u_penetration_pass": internal.get(
                    "penetration_pass"
                ),
            }
        )
    return rows


def _run_full_trial_child(arguments: argparse.Namespace) -> int:
    if arguments._case_directory is None:
        raise ValueError("internal case directory is required")
    result = run_continuous_u_full_trial(
        arguments.mesh_level,
        arguments.indentation_mm,
        arguments.steps,
    )
    _write_json(
        arguments._case_directory / "full_trial_result.json", result
    )
    _write_records(
        arguments._case_directory / "history.csv",
        _history_rows(result),
    )
    print(
        f"PHASE4ID_FULL_TRIAL_RESULT {result['status']}", flush=True
    )
    return 0 if result["status"] == "PASS" else 1


def _child_command(
    arguments: argparse.Namespace,
    case: str,
    directory: Path,
    full_trial: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-B",
        "-m",
        "validation.fingertip.internal_contact.run_diagnostics",
        "--mesh-level",
        arguments.mesh_level,
        "--output-directory",
        str(arguments.output_directory),
        "--_case-directory",
        str(directory),
    ]
    if full_trial:
        command.extend(
            [
                "--_full-trial-child",
                "--indentation-mm",
                str(arguments.indentation_mm),
                "--steps",
                str(arguments.steps),
            ]
        )
    else:
        command.extend(["--_child", "--_child-case", case])
    return command


def _run_process(command: Sequence[str], log_path: Path) -> int:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONFAULTHANDLER"] = "1"
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=7200,
        preexec_fn=lambda: resource.setrlimit(
            resource.RLIMIT_CORE, (0, 0)
        ),
    )
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.write(completed.stdout)
        stream.write(completed.stderr)
        stream.write(f"\n[process_exit_code={completed.returncode}]\n")
    return completed.returncode


def _load_case(
    directory: Path, exit_code: int, case: str
) -> dict[str, Any]:
    path = directory / "result.json"
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if not value.get("run_pending"):
            value["process_exit_code"] = exit_code
            return value
    runtime_path = directory / "runtime_contract.json"
    diagnostic = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime_path.is_file()
        else {}
    )
    log_path = directory / "solver.log"
    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.is_file()
        else ""
    )
    first_step = {
        "phase": "4I-D",
        "case": case,
        "configuration": configuration_for_case(case),
        "status": "FAIL",
        "solver_converged": False,
        "nonlinear_iterations": None,
        "failed_iteration": 1,
        "failed_iteration_basis": (
            "The native abort followed the first Skyline factorization and "
            "preceded any completed nonlinear-iteration callback; ProcessInfo "
            "could not be read after SIGABRT."
        ),
        "failure_reason": "native_solver_process_aborted",
        "process_exit_code": exit_code,
        "process_signal": -exit_code if exit_code < 0 else None,
        "skyline_zero_sum_reported": (
            "LUSkylineFactorization::factorize: Error zero sum"
            in log_text
        ),
        "nonfinite_contact_normal_reported": (
            "normal norm is zero or almost zero" in log_text
        ),
        "process_output_tail": log_text[-12000:],
    }
    _write_json(directory / "first_step_result.json", first_step)
    return {
        "phase": "4I-D",
        "case": case,
        "configuration": configuration_for_case(case),
        "status": "FAIL",
        "failure_reason": "child_process_terminated_without_result",
        "process_exit_code": exit_code,
        "diagnostic": diagnostic,
        "first_step": first_step,
    }


def _git_state() -> dict[str, Any]:
    def command(*values: str) -> str:
        completed = subprocess.run(
            values,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "head": command("git", "rev-parse", "HEAD"),
        "branch": command("git", "branch", "--show-current"),
        "worktree_status": command("git", "status", "--short").splitlines(),
        "identifier": "HEAD plus preserved dirty-worktree listing",
    }


def _case_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    first = result.get("first_step", {})
    diagnostic = result.get("diagnostic", {})
    dofs = diagnostic.get("dof_summary", {})
    matrix = diagnostic.get("matrix_diagnostics", {})
    purity = diagnostic.get("contact_pair_purity", {})
    groups = purity.get("groups", {})
    return {
        "case": result.get("case"),
        "configuration": result.get("configuration"),
        "first_step_status": first.get("status"),
        "solver_converged": first.get("solver_converged"),
        "nonlinear_iterations": first.get("nonlinear_iterations"),
        "failed_iteration": first.get("failed_iteration"),
        "total_assembled_dofs": dofs.get(
            "assembled_total_dof_count"
        ),
        "contact_lm_dofs": dofs.get(
            "assembled_contact_lm_dof_count"
        ),
        "duplicate_equation_id_count": len(
            dofs.get("duplicate_equation_ids", [])
        ),
        "zero_rows": matrix.get("exact_zero_row_count"),
        "near_zero_rows": matrix.get("near_zero_row_count"),
        "near_zero_offenders": matrix.get("near_zero_rows"),
        "external_generated_conditions": groups.get(
            "external_pad_indenter", {}
        ).get("generated_condition_count"),
        "external_active_conditions_at_assembly": groups.get(
            "external_pad_indenter", {}
        ).get("active_condition_count"),
        "internal_generated_conditions": sum(
            value.get("generated_condition_count", 0)
            for name, value in groups.items()
            if name != "external_pad_indenter"
        ),
        "internal_active_conditions_at_assembly": sum(
            value.get("active_condition_count", 0)
            for name, value in groups.items()
            if name != "external_pad_indenter"
        ),
        "pair_purity_pass": purity.get(
            "all_generated_conditions_pair_pure"
        ),
    }


def _corner_comparison(
    results: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    output: dict[str, Any] = {"available": False}
    if "D" not in results or "E" not in results:
        return output
    d_contract = (
        results["D"]
        .get("diagnostic", {})
        .get("corner_contract", {})
        .get("corners", {})
    )
    e_contract = (
        results["E"]
        .get("diagnostic", {})
        .get("corner_contract", {})
        .get("corners", {})
    )
    if not d_contract or not e_contract:
        return output
    comparisons: dict[str, Any] = {}
    for label in sorted(set(d_contract).intersection(e_contract)):
        d_value = d_contract[label]
        e_value = e_contract[label]
        comparisons[label] = {
            "same_physical_node_id": (
                d_value["node_id"] == e_value["node_id"]
            ),
            "same_incident_condition_ids": (
                [
                    item["condition_id"]
                    for item in d_value["incident_conditions"]
                ]
                == [
                    item["condition_id"]
                    for item in e_value["incident_conditions"]
                ]
            ),
            "case_d_contact_process_registration_count": d_value[
                "contact_process_registration_count"
            ],
            "case_e_contact_process_registration_count": e_value[
                "contact_process_registration_count"
            ],
            "case_d_contact_dof": d_value["contact_related_dofs"],
            "case_e_contact_dof": e_value["contact_related_dofs"],
            "case_d_duplicate_incident_connectivity_count": d_value[
                "duplicate_incident_connectivity_count"
            ],
            "case_e_duplicate_incident_connectivity_count": e_value[
                "duplicate_incident_connectivity_count"
            ],
        }
    return {
        "available": True,
        "comparisons": comparisons,
        "all_same_physical_nodes": all(
            value["same_physical_node_id"]
            for value in comparisons.values()
        ),
        "all_reuse_same_source_conditions": all(
            value["same_incident_condition_ids"]
            for value in comparisons.values()
        ),
        "all_e_registration_counts_at_most_one": all(
            value["case_e_contact_process_registration_count"] <= 1
            for value in comparisons.values()
        ),
        "all_e_incident_connectivity_unique": all(
            value["case_e_duplicate_incident_connectivity_count"] == 0
            for value in comparisons.values()
        ),
    }


def _conclusion(
    results: Mapping[str, Mapping[str, Any]],
    full_trial: Mapping[str, Any] | None,
) -> dict[str, Any]:
    statuses = {
        case: result.get("first_step", {}).get("status")
        for case, result in results.items()
    }
    if statuses.get("A") == "FAIL":
        root = (
            "Case A failed: internal zero-clearance contact is not a "
            "sufficient condition for the regression. Start from the "
            "external-only assembly/solve."
        )
    elif statuses.get("B") == "FAIL":
        root = (
            "A passed but B failed: pair splitting and U-corners alone do not "
            "explain the failure; one conforming zero-clearance ALM pair is "
            "already sufficient."
        )
    elif statuses.get("C") == "FAIL":
        left = statuses.get("C-left")
        right = statuses.get("C-right")
        if left == "PASS" and right == "PASS":
            root = (
                "A/B and each single side passed, but C failed: simultaneous "
                "opposing-side assembly or process interference is implicated."
            )
        elif left == "FAIL" or right == "FAIL":
            failed_sides = [
                name
                for name, status in (
                    ("left", left),
                    ("right", right),
                )
                if status == "FAIL"
            ]
            root = (
                "A/B passed and C failed; the auxiliary controls show that "
                f"{'/'.join(failed_sides)} side contact alone is sufficient "
                "to reproduce failure. Opposing-pair interaction is therefore "
                "not required. The corresponding endpoint LM row, surface "
                "orientation/normal, and zero-clearance activation remain "
                "candidate mechanisms."
            )
        else:
            root = (
                "A/B passed but C failed. Single-side controls were not both "
                "available, so simultaneous side-pair interference is not yet "
                "distinguished from one defective side."
            )
    elif statuses.get("D") == "FAIL" and statuses.get("E") == "PASS":
        root = (
            "A/B/C passed, D failed, and E passed: the failure is associated "
            "with the three-separate-pair solver topology. Continuous U is "
            "only a validated recovery if its gated 48-step Trial also passes."
        )
    elif statuses.get("D") == "FAIL" and statuses.get("E") == "FAIL":
        root = (
            "D and E both failed: merging the separate pairs is not a "
            "solution; zero-clearance activation, corner normals, LM "
            "assembly, or boundary-condition rank deficiency remains."
        )
    else:
        root = (
            "The observed pattern does not isolate the prior failure to one "
            "listed topology; matrix, DOF, and pair-purity evidence must be "
            "used without a stronger causal claim."
        )
    full_status = full_trial.get("status") if full_trial else None
    adopted = (
        statuses.get("E") == "PASS" and full_status == "PASS"
    )
    return {
        "first_step_pattern": statuses,
        "evidence_backed_conclusion": root,
        "case_d_prior_failure_reproduced": statuses.get("D") == "FAIL",
        "continuous_u_full_trial_status": full_status or "NOT_RUN",
        "continuous_u_pair_decision": (
            "ADOPT" if adopted else "REJECT"
            if statuses.get("E") == "FAIL" or full_status == "FAIL"
            else "INCONCLUSIVE"
        ),
        "phase4i_d_verdict": "PASS" if adopted else "FAIL"
        if statuses.get("E") == "FAIL" or full_status == "FAIL"
        else "INCOMPLETE",
        "phase4i_status": "still incomplete",
        "next_step_is_medium_fine_1p5mm_baseline": adopted,
    }


def _write_comparison(
    path: Path, case_summaries: Sequence[Mapping[str, Any]]
) -> None:
    _write_records(path, case_summaries)


def _main_parent(arguments: argparse.Namespace) -> int:
    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)
    requested = (
        arguments.cases
        if arguments.cases
        else [arguments.case]
        if arguments.case
        else ["A", "B", "C", "D", "E"]
    )
    results: dict[str, dict[str, Any]] = {}
    commands: list[list[str]] = []
    for case in requested:
        directory = output / CASE_DIRECTORY_NAMES[case]
        directory.mkdir(parents=True, exist_ok=True)
        log = directory / "solver.log"
        log.write_text("", encoding="utf-8")
        _write_json(
            directory / "result.json",
            {
                "phase": "4I-D",
                "case": case,
                "configuration": configuration_for_case(case),
                "status": "PENDING",
                "run_pending": True,
            },
        )
        command = _child_command(arguments, case, directory)
        commands.append(command)
        exit_code = _run_process(command, log)
        result = _load_case(directory, exit_code, case)
        results[case] = result
        _write_json(directory / "result.json", result)

    if (
        "C" in results
        and results["C"].get("first_step", {}).get("status") == "FAIL"
    ):
        for case in ("C-left", "C-right"):
            directory = output / CASE_DIRECTORY_NAMES[case]
            directory.mkdir(parents=True, exist_ok=True)
            log = directory / "solver.log"
            log.write_text("", encoding="utf-8")
            _write_json(
                directory / "result.json",
                {
                    "phase": "4I-D",
                    "case": case,
                    "configuration": configuration_for_case(case),
                    "status": "PENDING",
                    "run_pending": True,
                },
            )
            command = _child_command(arguments, case, directory)
            commands.append(command)
            exit_code = _run_process(command, log)
            result = _load_case(directory, exit_code, case)
            results[case] = result
            _write_json(directory / "result.json", result)

    full_trial: dict[str, Any] | None = None
    if arguments.run_full_trial:
        if requested != ["E"]:
            raise ValueError("--run-full-trial requires --case E")
        first_status = results["E"].get("first_step", {}).get("status")
        if first_status == "PASS":
            directory = output / CASE_DIRECTORY_NAMES["E"]
            command = _child_command(
                arguments, "E", directory, full_trial=True
            )
            commands.append(command)
            exit_code = _run_process(
                command, directory / "solver.log"
            )
            path = directory / "full_trial_result.json"
            if path.is_file():
                full_trial = json.loads(path.read_text(encoding="utf-8"))
                full_trial["process_exit_code"] = exit_code
                _write_json(path, full_trial)
            else:
                full_trial = {
                    "status": "FAIL",
                    "failure_reason": (
                        "full_trial_child_terminated_without_result"
                    ),
                    "process_exit_code": exit_code,
                }
        else:
            full_trial = {
                "status": "NOT_RUN",
                "failure_reason": (
                    "Case E first-step gate did not pass"
                ),
            }
    elif (
        "E" in results
        and results["E"].get("first_step", {}).get("status") != "PASS"
    ):
        full_trial = {
            "phase": "4I-D",
            "diagnostic_case": "E",
            "status": "NOT_RUN",
            "gate": "Case E first-step acceptance",
            "reason": (
                "Continuous-U first step failed; the 0.25 mm / 48-step "
                "Trial is prohibited by the gate."
            ),
            "requested_indentation_mm": arguments.indentation_mm,
            "requested_steps": arguments.steps,
        }
        _write_json(
            output
            / CASE_DIRECTORY_NAMES["E"]
            / "full_trial_result.json",
            full_trial,
        )

    case_summaries = [
        _case_summary(results[case]) for case in results
    ]
    conclusion = _conclusion(results, full_trial)
    summary = {
        "phase": "4I-D",
        "git_state": _git_state(),
        "common_settings": common_settings(arguments.mesh_level),
        "commands": commands,
        "cases": case_summaries,
        "corner_contract_comparison_d_vs_e": _corner_comparison(
            results
        ),
        "full_trial": full_trial,
        "conclusion": conclusion,
        "artifacts_are_separate_from_prior_phase4i": True,
    }
    _write_json(output / "summary.json", summary)
    _write_comparison(output / "comparison.csv", case_summaries)
    print(
        "Phase 4I-D: "
        f"{conclusion['phase4i_d_verdict']}; continuous U: "
        f"{conclusion['continuous_u_pair_decision']}"
    )
    return 0 if conclusion["phase4i_d_verdict"] == "PASS" else 1


def main() -> int:
    arguments = _parse_arguments()
    if arguments._child:
        return _run_case_child(arguments)
    if arguments._full_trial_child:
        return _run_full_trial_child(arguments)
    return _main_parent(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
