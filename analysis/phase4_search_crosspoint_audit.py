"""Run Phase 4I-F search/crosspoint cases in isolated Python processes."""

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

from fem.search_crosspoint_audit import (
    CAUSAL_VARIANTS,
    run_lifecycle_case,
    source_trace,
    unavailable_case_records,
)


DEFAULT_OUTPUT = Path("output/phase4_search_crosspoint_audit")
CASE_DIRECTORIES = {
    "L00": "left_control",
    "F00": "f00_original",
    "F02": "f02_invalid_inactive",
}
UNAVAILABLE_DIRECTORIES = {
    "F01": "f01_invalid_removed",
    "F03": "f03_valid_only",
    "symmetric_control": "symmetric_control",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT
    )
    parser.add_argument("--mesh-level", choices=("medium",), default="medium")
    parser.add_argument("--_child-case", choices=tuple(CASE_DIRECTORIES))
    parser.add_argument("--_case-directory", type=Path)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for record in records for key in record})
    with path.open("w", encoding="utf-8", newline="") as stream:
        if not fields:
            return
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {key: _csv_value(record.get(key)) for key in fields}
            )


def _flatten_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    dofs = snapshot.get("dofs", {})
    flags = snapshot.get("node_flags", {})
    local = snapshot.get("local_lm_assembly", {})
    global_row = snapshot.get("global_lm_assembly", {})
    aggregate = local.get("aggregate_before_dirichlet", {})
    valid = local.get("valid_pairs_only_before_dirichlet", {})
    pairs = snapshot.get("incident_generated_conditions", [])
    return {
        "variant": snapshot.get("variant"),
        "side": snapshot.get("side"),
        "stage": snapshot.get("stage"),
        "iteration": snapshot.get("iteration"),
        "endpoint_node_id": snapshot.get("endpoint_node_id"),
        "active": flags.get("ACTIVE"),
        "slave": flags.get("SLAVE"),
        "master": flags.get("MASTER"),
        "lm_pressure": snapshot.get(
            "lagrange_multiplier_contact_pressure"
        ),
        "weighted_gap": snapshot.get("weighted_gap"),
        "normal_gap": snapshot.get("normal_gap"),
        "nodal_area": snapshot.get("nodal_area"),
        "nodal_h": snapshot.get("nodal_h"),
        "augmented_normal_contact_pressure": snapshot.get(
            "augmented_normal_contact_pressure_recomputed"
        ),
        "active_set_computed": snapshot.get("process_flags", {}).get(
            "ACTIVE_SET_COMPUTED"
        ),
        "contact_converged": snapshot.get("process_flags", {}).get(
            "CONTACT_CONVERGED"
        ),
        "lm_equation_id": dofs.get(
            "LAGRANGE_MULTIPLIER_CONTACT_PRESSURE", {}
        ).get("equation_id"),
        "lm_fixed": dofs.get(
            "LAGRANGE_MULTIPLIER_CONTACT_PRESSURE", {}
        ).get("fixed"),
        "displacement_x_equation_id": dofs.get(
            "DISPLACEMENT_X", {}
        ).get("equation_id"),
        "displacement_x_fixed": dofs.get(
            "DISPLACEMENT_X", {}
        ).get("fixed"),
        "displacement_y_equation_id": dofs.get(
            "DISPLACEMENT_Y", {}
        ).get("equation_id"),
        "displacement_y_fixed": dofs.get(
            "DISPLACEMENT_Y", {}
        ).get("fixed"),
        "generated_pair_ids": [
            record["generated_condition_id"] for record in pairs
        ],
        "active_pair_ids": [
            record["generated_condition_id"]
            for record in pairs
            if record["condition_active"]
        ],
        "invalid_pair_ids": [
            record["generated_condition_id"]
            for record in pairs
            if record["out_of_domain_extra_pair"]
        ],
        "pre_dirichlet_row_norm": aggregate.get("row_norm_all_columns"),
        "pre_dirichlet_free_column_norm": aggregate.get(
            "row_norm_free_columns"
        ),
        "valid_only_free_column_norm": valid.get(
            "row_norm_free_columns"
        ),
        "post_dirichlet_row_norm": global_row.get(
            "row_norm_all_columns"
        ),
        "post_dirichlet_free_column_norm": global_row.get(
            "row_norm_free_columns"
        ),
        "rhs_entry": global_row.get("rhs_entry"),
        "crosspoint": snapshot.get("crosspoint"),
        "pairs": pairs,
        "local_condition_rows": local.get("condition_rows"),
    }


def _write_case(
    directory: Path,
    result: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
    contact_records: Sequence[Mapping[str, Any]],
) -> None:
    _write_json(directory / "result.json", result)
    _write_json(directory / "lifecycle.json", list(snapshots))
    _write_json(directory / "contact_conditions.json", list(contact_records))
    _write_csv(
        directory / "endpoint_lifecycle.csv",
        [_flatten_snapshot(snapshot) for snapshot in snapshots],
    )


def _run_child(arguments: argparse.Namespace) -> int:
    if arguments._child_case is None or arguments._case_directory is None:
        raise ValueError("child case and case directory are required")
    result, snapshots, contact_records = run_lifecycle_case(
        CAUSAL_VARIANTS[arguments._child_case],
        arguments.mesh_level,
    )
    _write_case(
        arguments._case_directory, result, snapshots, contact_records
    )
    print(
        "PHASE4IF_CHILD_RESULT "
        f"{arguments._child_case} {result['status']}",
        flush=True,
    )
    return 0


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
    log_path.write_text(
        "$ "
        + " ".join(command)
        + "\n"
        + completed.stdout
        + completed.stderr
        + f"\n[process_exit_code={completed.returncode}]\n",
        encoding="utf-8",
    )
    return completed.returncode


def _load_case(directory: Path, label: str, exit_code: int) -> dict[str, Any]:
    path = directory / "result.json"
    if path.is_file():
        result = json.loads(path.read_text(encoding="utf-8"))
    else:
        result = {
            "phase": "4I-F",
            "variant": {"name": label},
            "status": "FAIL",
            "solve_converged": False,
            "failure_reason": "child_process_terminated_without_result",
        }
    result["process_exit_code"] = exit_code
    _write_json(path, result)
    return result


def _load_lifecycle(directory: Path) -> list[dict[str, Any]]:
    path = directory / "lifecycle.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def _stage(
    snapshots: Sequence[Mapping[str, Any]],
    name: str,
    iteration: int | None = None,
) -> Mapping[str, Any]:
    matches = [
        record
        for record in snapshots
        if record.get("stage") == name
        and (iteration is None or record.get("iteration") == iteration)
    ]
    return matches[-1] if matches else {}


def _active_history(
    snapshots: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "iteration": record["iteration"],
            "active": record["node_flags"]["ACTIVE"],
            "weighted_gap": record.get("weighted_gap"),
            "lm_pressure": record.get(
                "lagrange_multiplier_contact_pressure"
            ),
            "augmented_pressure": record.get(
                "augmented_normal_contact_pressure_recomputed"
            ),
            "active_pair_ids": [
                pair["generated_condition_id"]
                for pair in record.get(
                    "incident_generated_conditions", []
                )
                if pair["condition_active"]
            ],
        }
        for record in snapshots
        if record.get("stage") == "after_active_set_convergence_check"
    ]


def _pair_comparison_rows(
    lifecycles: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, snapshots in lifecycles.items():
        search = _stage(snapshots, "after_contact_search")
        for pair in search.get("incident_generated_conditions", []):
            rows.append(
                {
                    "variant": label,
                    "side": search.get("side"),
                    "endpoint_node_id": search.get("endpoint_node_id"),
                    "generated_condition_id": pair[
                        "generated_condition_id"
                    ],
                    "condition_active": pair["condition_active"],
                    "slave_node_ids": pair["slave_node_ids"],
                    "master_node_ids": pair["master_node_ids"],
                    "segment_fraction": pair["endpoint_projection"].get(
                        "segment_fraction"
                    ),
                    "local_coordinate": pair["endpoint_projection"].get(
                        "local_coordinate"
                    ),
                    "inside_local_domain": pair[
                        "endpoint_projection"
                    ].get("inside_local_domain"),
                    "projection_point_mm": pair[
                        "endpoint_projection"
                    ].get("projection_point_mm"),
                    "exact_overlap_length_mm": pair[
                        "exact_overlap"
                    ].get("overlap_length_mm"),
                    "positive_exact_overlap": pair[
                        "exact_overlap"
                    ].get("positive_overlap"),
                    "valid_endpoint_pair": pair["valid_endpoint_pair"],
                    "out_of_domain_extra_pair": pair[
                        "out_of_domain_extra_pair"
                    ],
                }
            )
    return rows


def _crosspoint_rows(
    lifecycles: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, snapshots in lifecycles.items():
        assembly = _stage(snapshots, "after_tangent_assembly", 1)
        if not assembly:
            assembly = _stage(snapshots, "after_initialize_solution_step")
        flat = _flatten_snapshot(assembly)
        rows.append(
            {
                key: value
                for key, value in flat.items()
                if key
                in {
                    "variant",
                    "side",
                    "endpoint_node_id",
                    "lm_equation_id",
                    "lm_fixed",
                    "displacement_x_equation_id",
                    "displacement_x_fixed",
                    "displacement_y_equation_id",
                    "displacement_y_fixed",
                    "pre_dirichlet_row_norm",
                    "pre_dirichlet_free_column_norm",
                    "valid_only_free_column_norm",
                    "post_dirichlet_row_norm",
                    "post_dirichlet_free_column_norm",
                    "rhs_entry",
                    "crosspoint",
                    "local_condition_rows",
                }
            }
        )
    return rows


def _summarize(
    results: Mapping[str, Mapping[str, Any]],
    lifecycles: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    unavailable = unavailable_case_records()
    f00_search = _stage(lifecycles["F00"], "after_contact_search")
    f02_mutated = _stage(
        lifecycles["F02"], "after_diagnostic_invalid_pair_deactivation"
    )
    left_search = _stage(lifecycles["L00"], "after_contact_search")
    f00_invalid = [
        pair
        for pair in f00_search.get("incident_generated_conditions", [])
        if pair["out_of_domain_extra_pair"]
    ]
    f02_invalid_inactive = bool(f02_mutated) and all(
        not pair["condition_active"]
        for pair in f02_mutated.get("incident_generated_conditions", [])
        if pair["out_of_domain_extra_pair"]
    )
    f00_assembly = _stage(
        lifecycles["F00"], "after_tangent_assembly", 1
    )
    f02_assembly = _stage(
        lifecycles["F02"], "after_tangent_assembly", 1
    )
    left_assembly = _stage(
        lifecycles["L00"], "after_tangent_assembly", 1
    )
    f00_valid_free = (
        f00_assembly.get("local_lm_assembly", {})
        .get("valid_pairs_only_before_dirichlet", {})
        .get("row_norm_free_columns")
    )
    right_deficient = (
        f00_valid_free is not None and f00_valid_free < 1.0e-12
    )
    left_history = _active_history(lifecycles["L00"])
    right_history = _active_history(lifecycles["F00"])
    left_deactivation_iterations = [
        record["iteration"]
        for record in left_history
        if not record["active"]
    ]
    f02_still_fails = (
        f02_invalid_inactive and not results["F02"]["solve_converged"]
    )
    source_correction_validated = False
    regressions = {
        "status": "NOT_RUN",
        "reason": (
            "The regression gate requires a validated physical source-level "
            "correction; F01/F03 cannot be constructed through a safe public "
            "Kratos API and F02 is diagnostic-only."
        ),
        "requested_cases": ["A", "B", "C-left", "C-right", "C", "D", "E"],
    }
    full_trials = {
        "status": "NOT_RUN",
        "reason": (
            "No corrected D/E first-step PASS exists; the 0.25 mm x 48-step "
            "trial gate is closed."
        ),
        "one_point_five_mm_trial_run": False,
    }
    return {
        "phase": "4I-F",
        "status": "FAIL",
        "mesh_level": "medium",
        "first_step_travel_mm": 0.25 / 48.0,
        "cases": {**results, **unavailable},
        "first_left_right_asymmetry_trigger": {
            "stage": "after_contact_search",
            "left_upper_generated_pair_count": len(
                left_search.get("incident_generated_conditions", [])
            ),
            "right_upper_generated_pair_count": len(
                f00_search.get("incident_generated_conditions", [])
            ),
            "right_out_of_domain_pair_count": len(f00_invalid),
            "right_out_of_domain_local_coordinates": [
                pair["endpoint_projection"]["local_coordinate"]
                for pair in f00_invalid
            ],
            "right_out_of_domain_exact_overlap_mm": [
                pair["exact_overlap"].get("overlap_length_mm")
                for pair in f00_invalid
            ],
        },
        "causal_matrix_conclusion": {
            "F00_reproduced_failure": not results["F00"][
                "solve_converged"
            ],
            "F02_invalid_condition_reliably_inactive": (
                f02_invalid_inactive
            ),
            "F02_still_failed": f02_still_fails,
            "invalid_pair_active_state_is_necessary_for_failure": False
            if f02_still_fails
            else None,
            "invalid_pair_presence_effect_isolated": False,
            "reason": (
                "F02 separates condition ACTIVE state from presence and still "
                "fails. Safe container removal (F01/F03) is unavailable, so "
                "the effect of mere pair presence remains unconfirmed."
            ),
        },
        "direct_zero_row_mechanism": {
            "right_endpoint_xy_free_primal_dofs": f00_assembly.get(
                "crosspoint", {}
            ).get("xy_free_primal_dof_count"),
            "right_valid_pair_only_free_column_norm": f00_valid_free,
            "right_valid_pair_deficient": right_deficient,
            "right_post_dirichlet_global_row": f00_assembly.get(
                "global_lm_assembly"
            ),
            "right_f02_post_dirichlet_global_row": f02_assembly.get(
                "global_lm_assembly"
            ),
            "left_post_dirichlet_global_row": left_assembly.get(
                "global_lm_assembly"
            ),
            "mechanism": (
                "The active endpoint LM couples locally to displacement "
                "columns, but its endpoint X/Y primal DOFs are fixed. The "
                "remaining valid-pair coupling to free columns is near zero; "
                "Dirichlet elimination therefore leaves a near-zero global LM "
                "row. The zero-overlap extra condition contributes a zero "
                "local row but is not required for the deficiency."
            ),
        },
        "left_success_dependency": {
            "left_solve_converged": results["L00"]["solve_converged"],
            "left_newton_iterations": results["L00"][
                "newton_iterations"
            ],
            "first_inactive_iteration": min(
                left_deactivation_iterations, default=None
            ),
            "depends_on_endpoint_deactivation": bool(
                results["L00"]["solve_converged"]
                and left_deactivation_iterations
            ),
            "active_history": left_history,
        },
        "right_active_history": right_history,
        "library_level_behavior": (
            "Kratos' broad-phase search can create an ACTIVE paired condition "
            "whose endpoint projection is outside the Line2 segment and whose "
            "exact overlap is numerical zero. Gap/active state is then "
            "aggregated at the slave node."
        ),
        "application_level_modeling_susceptibility": (
            "The pad internal-contact endpoint is also on the fully constrained "
            "pad-bond boundary, so an active scalar contact LM has no meaningful "
            "free endpoint primal direction after Dirichlet elimination."
        ),
        "production_correction": {
            "validated": source_correction_validated,
            "candidate_a_pair_acceptance": (
                "NOT_ADOPTED: pair-state deactivation does not recover F02, "
                "and safe pair-removal causality is unavailable."
            ),
            "candidate_b_crosspoint_multiplier": (
                "IDENTIFIED_FOR_FUTURE_WORK, NOT_ADOPTED: no official Kratos "
                "automatic crosspoint LM treatment was found, and endpoint/LM "
                "deletion, forced fixity, contact truncation, or forced "
                "inactivation are prohibited without a mesh-independent "
                "mathematical rule."
            ),
        },
        "regressions": regressions,
        "full_trials": full_trials,
        "confirmed_facts": [
            "The first asymmetry occurs during contact search.",
            "The right extra pair has xi outside [-1,1] and numerical-zero exact overlap.",
            "F02 keeps the pair but makes its condition inactive through a supported flag API.",
            "The valid right endpoint pair alone has a near-zero free-column LM coupling.",
            "Both upper pad endpoints are contact/bond/Dirichlet crosspoints.",
        ],
        "unconfirmed": [
            "Whether removing only the invalid pair would change convergence.",
            "Whether search traversal or insertion-order tie handling creates the left/right candidate asymmetry.",
            "A physically justified mesh-independent production crosspoint multiplier rule.",
        ],
        "acceptance": {
            "root_causes_isolated": True,
            "physical_source_level_correction_validated": False,
            "C_right_and_C_first_step_pass": False,
            "D_or_E_48_step_trial_pass": False,
            "phase_pass": False,
        },
    }


def main() -> int:
    arguments = _arguments()
    if arguments._child_case:
        return _run_child(arguments)

    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "source_trace.json", source_trace())

    results: dict[str, dict[str, Any]] = {}
    lifecycles: dict[str, list[dict[str, Any]]] = {}
    commands: list[dict[str, Any]] = []
    for label in ("L00", "F00", "F02"):
        directory = output / CASE_DIRECTORIES[label]
        directory.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-B",
            "-m",
            "analysis.phase4_search_crosspoint_audit",
            "--mesh-level",
            arguments.mesh_level,
            "--output-directory",
            str(output),
            "--_child-case",
            label,
            "--_case-directory",
            str(directory),
        ]
        exit_code = _run_process(command, directory / "solver.log")
        results[label] = _load_case(directory, label, exit_code)
        lifecycles[label] = _load_lifecycle(directory)
        commands.append(
            {
                "case": label,
                "command": command,
                "exit_code": exit_code,
            }
        )

    unavailable = unavailable_case_records()
    for label, directory_name in UNAVAILABLE_DIRECTORIES.items():
        directory = output / directory_name
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(directory / "result.json", unavailable[label])

    regression_directory = output / "regressions"
    trial_directory = output / "full_trials"
    regression_directory.mkdir(parents=True, exist_ok=True)
    trial_directory.mkdir(parents=True, exist_ok=True)

    lifecycle_rows = [
        _flatten_snapshot(snapshot)
        for snapshots in lifecycles.values()
        for snapshot in snapshots
    ]
    pair_rows = _pair_comparison_rows(lifecycles)
    crosspoint_rows = _crosspoint_rows(lifecycles)
    _write_csv(output / "endpoint_lifecycle.csv", lifecycle_rows)
    _write_csv(output / "search_pair_comparison.csv", pair_rows)
    _write_csv(output / "crosspoint_dof_map.csv", crosspoint_rows)

    summary = _summarize(results, lifecycles)
    summary["commands"] = commands
    _write_json(output / "summary.json", summary)
    _write_json(
        regression_directory / "result.json", summary["regressions"]
    )
    _write_json(trial_directory / "result.json", summary["full_trials"])
    print(
        "Phase 4I-F "
        f"{summary['status']}: correction gate "
        f"{'open' if summary['production_correction']['validated'] else 'closed'}"
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
