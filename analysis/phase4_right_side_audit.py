"""Run the Phase 4I-E left/right orientation audit in isolated processes."""

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

from fem.right_side_audit import (
    ORIENTATION_VARIANTS,
    audit_side_orientation,
    common_audit_mesh,
    left_right_mirror_contract,
)


DEFAULT_OUTPUT = Path("output/phase4_right_side_audit")
CASE_DIRECTORIES = {
    "L00": "left_oracle",
    "R00": "right_r00",
    "R10": "right_r10",
    "R01": "right_r01",
    "R11": "right_r11",
}


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--run-orientation-matrix", action="store_true")
    mode.add_argument("--run-regression-cases", action="store_true")
    mode.add_argument("--run-full-trials", action="store_true")
    parser.add_argument("--mesh-level", choices=("medium",), default="medium")
    parser.add_argument("--indentation-mm", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT
    )
    parser.add_argument("--_audit-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_audit-label", help=argparse.SUPPRESS)
    parser.add_argument("--_case-directory", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _csv_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for record in records for field in record})
    with path.open("w", encoding="utf-8", newline="") as stream:
        if not fields:
            return
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {field: _csv_cell(record.get(field)) for field in fields}
            )


def _write_case_artifacts(
    directory: Path,
    result: Mapping[str, Any],
    dof_rows: Sequence[Mapping[str, Any]],
    contact_records: Sequence[Mapping[str, Any]],
) -> None:
    _write_json(directory / "result.json", result)
    _write_json(
        directory / "endpoint_assembly.json",
        result.get("diagnostic", {}).get(
            "endpoint_assembly",
            {"available": False, "reason": result.get("failure_reason")},
        ),
    )
    _write_json(
        directory / "matrix_diagnostics.json",
        result.get("diagnostic", {}).get(
            "matrix_diagnostics",
            {"available": False, "reason": result.get("failure_reason")},
        ),
    )
    _write_json(
        directory / "normal_contract.json",
        {
            "source_orientation_contract": result.get(
                "source_orientation_contract"
            ),
            "stage_snapshots": result.get("stage_snapshots"),
        },
    )
    _write_csv(directory / "dof_map.csv", dof_rows)
    _write_csv(directory / "contact_conditions.csv", contact_records)


def _run_child(arguments: argparse.Namespace) -> int:
    if arguments._audit_label is None or arguments._case_directory is None:
        raise ValueError("audit child label and directory are required")
    label = arguments._audit_label
    directory = arguments._case_directory
    directory.mkdir(parents=True, exist_ok=True)

    def checkpoint(
        partial: Mapping[str, Any],
        dof_rows: Sequence[Mapping[str, Any]],
        contact_records: Sequence[Mapping[str, Any]],
    ) -> None:
        _write_case_artifacts(
            directory, partial, dof_rows, contact_records
        )

    if label == "L00":
        result, dof_rows, contact_records = audit_side_orientation(
            "left",
            arguments.mesh_level,
            pre_solve_callback=checkpoint,
        )
    else:
        result, dof_rows, contact_records = audit_side_orientation(
            "right",
            arguments.mesh_level,
            ORIENTATION_VARIANTS[label],
            pre_solve_callback=checkpoint,
        )
    _write_case_artifacts(directory, result, dof_rows, contact_records)
    print(f"PHASE4IE_CHILD_RESULT {label} {result['status']}", flush=True)
    return 0 if result["status"] == "PASS" else 1


def _child_command(
    arguments: argparse.Namespace, label: str, directory: Path
) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "analysis.phase4_right_side_audit",
        "--mesh-level",
        arguments.mesh_level,
        "--output-directory",
        str(arguments.output_directory),
        "--_audit-child",
        "--_audit-label",
        label,
        "--_case-directory",
        str(directory),
    ]


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


def _load_child_result(
    directory: Path, label: str, exit_code: int
) -> dict[str, Any]:
    path = directory / "result.json"
    if path.is_file():
        result = json.loads(path.read_text(encoding="utf-8"))
    else:
        result = {
            "phase": "4I-E",
            "variant": {"name": label},
            "status": "FAIL",
            "failure_reason": "child_process_terminated_without_artifact",
        }
    if result.get("status") == "PENDING_SOLVE":
        log = (directory / "solver.log").read_text(
            encoding="utf-8", errors="replace"
        )
        result["status"] = "FAIL"
        result["failure_reason"] = "native_solver_process_aborted"
        result["solver_converged"] = False
        result["process_signal"] = -exit_code if exit_code < 0 else None
        result["skyline_zero_sum_reported"] = (
            "LUSkylineFactorization::factorize: Error zero sum" in log
        )
        result["nonfinite_contact_normal_reported"] = (
            "normal norm is zero or almost zero" in log
        )
    result["process_exit_code"] = exit_code
    _write_json(path, result)
    return result


def _upper_endpoint(
    result: Mapping[str, Any], stage: str, tag: str
) -> Mapping[str, Any]:
    return result["stage_snapshots"][stage]["surfaces"][tag][
        "upper_endpoint"
    ]


def _reflection_error(
    left: Sequence[float], right: Sequence[float]
) -> float:
    return ((right[0] + left[0]) ** 2 + (right[1] - left[1]) ** 2) ** 0.5


def _mirror_case_comparison(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for stage in (
        "before_contact_process",
        "after_execute_initialize",
        "after_contact_search",
        "after_first_newton_assembly",
    ):
        left_stages = left.get("stage_snapshots", {})
        right_stages = right.get("stage_snapshots", {})
        if stage not in left_stages or stage not in right_stages:
            stages[stage] = {
                "available": False,
                "reason": (
                    "right diagnostic stopped before this stage"
                    if stage not in right_stages
                    else "left oracle stage unavailable"
                ),
            }
            continue
        left_node = _upper_endpoint(
            left, stage, "pad_cutout_left"
        )
        right_node = _upper_endpoint(
            right, stage, "pad_cutout_right"
        )
        left_normal = left_node.get("nodal_normal")
        right_normal = right_node.get("nodal_normal")
        normal_error = (
            _reflection_error(left_normal, right_normal)
            if left_normal is not None and right_normal is not None
            else None
        )
        stages[stage] = {
            "available": True,
            "left_node_id": left_node["node_id"],
            "right_node_id": right_node["node_id"],
            "coordinate_reflection_error_mm": _reflection_error(
                left_node["reference_coordinate_mm"],
                right_node["reference_coordinate_mm"],
            ),
            "left_nodal_normal": left_normal,
            "right_nodal_normal": right_normal,
            "normal_reflection_error": normal_error,
            "left_flags": left_node["flags"],
            "right_flags": right_node["flags"],
            "left_weighted_gap": left_node["weighted_gap"],
            "right_weighted_gap": right_node["weighted_gap"],
        }
    left_endpoint = left.get("diagnostic", {}).get(
        "endpoint_assembly", {}
    )
    right_endpoint = right.get("diagnostic", {}).get(
        "endpoint_assembly", {}
    )
    left_projections = left_endpoint.get("pairing_projection", [])
    right_projections = right_endpoint.get("pairing_projection", [])
    search_asymmetry = {
        "available": bool(left_endpoint) and bool(right_endpoint),
        "left_upper_generated_pair_count": (
            len(left_projections) if left_endpoint else None
        ),
        "right_upper_generated_pair_count": (
            len(right_projections) if right_endpoint else None
        ),
        "left_all_endpoint_projections_successful": bool(
            left_projections
        )
        and all(
            record["endpoint_projection"]["success"]
            for record in left_projections
        ),
        "right_all_endpoint_projections_successful": bool(
            right_projections
        )
        and all(
            record["endpoint_projection"]["success"]
            for record in right_projections
        ),
    }
    first_stage = "none"
    reason = "all inspected contracts remain mirrored"
    source_physical = right.get("source_orientation_contract", {}).get(
        "all_ordering_normals_physical", False
    )
    if not source_physical:
        first_stage = "before_contact_process"
        reason = "right source Line2 ordering normal is non-physical"
    elif any(
        value.get("available", False)
        and value["normal_reflection_error"] is not None
        and value["normal_reflection_error"] > 1.0e-12
        for key, value in stages.items()
        if key in ("after_execute_initialize", "after_contact_search")
    ):
        first_stage = next(
            key
            for key in (
                "after_execute_initialize",
                "after_contact_search",
            )
            if stages[key].get("available", False)
            and stages[key]["normal_reflection_error"] is not None
            and stages[key]["normal_reflection_error"] > 1.0e-12
        )
        reason = "upper-endpoint nodal normals stop satisfying reflection"
    elif search_asymmetry["available"] and (
        search_asymmetry["left_upper_generated_pair_count"]
        != search_asymmetry["right_upper_generated_pair_count"]
        or search_asymmetry[
            "left_all_endpoint_projections_successful"
        ]
        != search_asymmetry[
            "right_all_endpoint_projections_successful"
        ]
    ):
        first_stage = "after_contact_search"
        reason = (
            "right upper slave condition receives an extra/invalid master "
            "projection compared with the left oracle"
        )
    elif left_endpoint and right_endpoint and (
        left_endpoint.get("near_zero") != right_endpoint.get("near_zero")
        or len(left_endpoint.get("local_condition_contributors", []))
        != len(right_endpoint.get("local_condition_contributors", []))
    ):
        first_stage = "after_first_newton_assembly"
        reason = "LM row contribution structure differs"
    return {
        "right_variant": right["variant"]["name"],
        "stages": stages,
        "search_and_assembly": {
            **search_asymmetry,
            "left_lm_row_norm": left_endpoint.get(
                "global_tangent_row_norm"
            ),
            "right_lm_row_norm": right_endpoint.get(
                "global_tangent_row_norm"
            ),
            "left_local_contributor_condition_ids": [
                record["condition_id"]
                for record in left_endpoint.get(
                    "local_condition_contributors", []
                )
            ],
            "right_local_contributor_condition_ids": [
                record["condition_id"]
                for record in right_endpoint.get(
                    "local_condition_contributors", []
                )
            ],
        },
        "first_asymmetric_stage": first_stage,
        "first_asymmetry_reason": reason,
    }


def _orientation_row(result: Mapping[str, Any]) -> dict[str, Any]:
    endpoint = result.get("diagnostic", {}).get(
        "endpoint_assembly", {}
    )
    projections = endpoint.get("pairing_projection", [])
    solve = result.get("solve_result", {})
    history = solve.get("history", [])
    point = history[-1] if history else {}
    group = point.get("contact_groups", {}).get(
        "internal_right", {}
    )
    source_physical = (
        result.get("acceptance_checks", {}).get(
            "source_ordering_normals_physical"
        )
        if result.get("acceptance_checks") is not None
        else result.get("source_orientation_contract", {}).get(
            "all_ordering_normals_physical"
        )
    )
    exception_lines = str(result.get("exception", "")).splitlines()
    return {
        "variant": result.get("variant", {}).get("name"),
        "reverse_slave": result.get("variant", {}).get(
            "reverse_slave"
        ),
        "reverse_master": result.get("variant", {}).get(
            "reverse_master"
        ),
        "source_ordering_physical": source_physical,
        "solver_converged": result.get(
            "acceptance_checks", {}
        ).get("first_step_solver_converged"),
        "case_status": result.get("status"),
        "upper_endpoint_generated_pair_count": len(projections),
        "all_endpoint_projections_successful": bool(projections)
        and all(
            record["endpoint_projection"]["success"]
            for record in projections
        ),
        "upper_endpoint_lm_row_norm": endpoint.get(
            "global_tangent_row_norm"
        ),
        "upper_endpoint_lm_near_zero": endpoint.get("near_zero"),
        "local_contributor_count": len(
            endpoint.get("local_condition_contributors", [])
        ),
        "internal_active_condition_count": group.get(
            "active_condition_count"
        ),
        "reaction_n": point.get("indenter_normal_reaction_n"),
        "minimum_det_f": point.get(
            "pad_strain_det_f", {}
        ).get("det_f", {}).get("min"),
        "pair_purity": result.get("diagnostic", {})
        .get("pair_purity", {})
        .get("all_generated_conditions_pair_pure"),
        "failure_reason": result.get("failure_reason"),
        "exception_summary": exception_lines[0] if exception_lines else None,
    }


def _write_not_run_case(
    directory: Path, label: str, reason: str
) -> None:
    result = {
        "phase": "4I-E",
        "case": label,
        "status": "NOT_RUN",
        "reason": reason,
    }
    _write_case_artifacts(directory, result, [], [])
    (directory / "solver.log").write_text(
        f"NOT_RUN: {reason}\n", encoding="utf-8"
    )


def _git_state() -> dict[str, Any]:
    def command(*values: str) -> str:
        completed = subprocess.run(
            values, check=False, capture_output=True, text=True
        )
        return completed.stdout.strip()

    return {
        "head": command("git", "rev-parse", "HEAD"),
        "branch": command("git", "branch", "--show-current"),
        "worktree_status": command(
            "git", "status", "--short"
        ).splitlines(),
    }


def _run_orientation_matrix(arguments: argparse.Namespace) -> int:
    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)
    fingertip_model, mesh = common_audit_mesh(arguments.mesh_level)
    source_mirror = left_right_mirror_contract(
        mesh, fingertip_model
    )
    results: dict[str, dict[str, Any]] = {}
    commands: list[list[str]] = []
    for label in ("L00", "R00", "R10", "R01", "R11"):
        directory = output / CASE_DIRECTORIES[label]
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(
            directory / "result.json",
            {
                "phase": "4I-E",
                "variant": {"name": label},
                "status": "PENDING_CHILD",
            },
        )
        command = _child_command(arguments, label, directory)
        commands.append(command)
        exit_code = _run_process(
            command, directory / "solver.log"
        )
        results[label] = _load_child_result(
            directory, label, exit_code
        )

    comparisons = [
        _mirror_case_comparison(results["L00"], results[label])
        for label in ("R00", "R10", "R01", "R11")
    ]
    orientation_rows = [
        _orientation_row(results[label])
        for label in ("R00", "R10", "R01", "R11")
    ]
    valid_candidates = [
        row["variant"]
        for row in orientation_rows
        if row["case_status"] == "PASS"
        and row["source_ordering_physical"]
    ]
    physical_baseline_only = (
        orientation_rows[0]["source_ordering_physical"] is True
        and all(
            row["source_ordering_physical"] is False
            for row in orientation_rows[1:]
        )
    )
    orientation_decision = {
        "physically_valid_passing_variants": valid_candidates,
        "hypothesis": (
            "CONFIRMED"
            if valid_candidates
            else "REJECTED"
            if physical_baseline_only
            else "REJECTED"
            if any(
                row["solver_converged"]
                and not row["source_ordering_physical"]
                for row in orientation_rows
            )
            else "INCONCLUSIVE"
        ),
        "source_level_fix_applied": False,
        "source_level_fix_reason": (
            "R00 already has the physical outward ordering. Reversing either "
            "right surface violates that contract and produces a zero nodal "
            "normal at a shared endpoint during ExecuteInitialize, so no "
            "orientation edit is a valid production correction."
        ),
    }
    first_r00_asymmetry = next(
        comparison
        for comparison in comparisons
        if comparison["right_variant"] == "R00"
    )
    not_run_reason = (
        "no physically validated source-level correction; regression and "
        "full-trial gates remain closed"
    )
    for label, directory_name in (
        ("A", "regression_a"),
        ("B", "regression_b"),
        ("C-left", "regression_left"),
        ("C-right", "regression_right"),
        ("C", "regression_c"),
        ("D", "regression_d"),
        ("E", "regression_e"),
    ):
        _write_not_run_case(
            output / directory_name, label, not_run_reason
        )
    summary = {
        "phase": "4I-E",
        "git_state": _git_state(),
        "common_settings": {
            "mesh_level": arguments.mesh_level,
            "first_step_travel_mm": 0.25 / 48.0,
            "geometry": "default zero-clearance FingertipModel",
            "parameter_tuning": "none",
        },
        "commands": commands,
        "source_mesh_mirror_contract": source_mirror,
        "orientation_matrix": orientation_rows,
        "mirror_comparisons": comparisons,
        "orientation_decision": orientation_decision,
        "diagnostic_conclusion": {
            "first_r00_asymmetric_stage": first_r00_asymmetry[
                "first_asymmetric_stage"
            ],
            "first_r00_asymmetry_reason": first_r00_asymmetry[
                "first_asymmetry_reason"
            ],
            "confirmed_root_cause": None,
            "narrowed_scope": (
                "right upper-endpoint contact search/pair generation and "
                "subsequent LM assembly/active-set handling"
            ),
            "left_oracle_note": (
                "The left endpoint also has a near-zero first-assembly LM "
                "row, but it has one valid generated pair and the nonlinear "
                "solve deactivates that endpoint and converges. R00 has an "
                "additional invalid adjacent-master pair with a zero local "
                "LM contribution and does not converge."
            ),
        },
        "regression_cases": {
            "status": "NOT_RUN",
            "gate": "physically valid source-level correction",
        },
        "full_trials": {
            "status": "NOT_RUN",
            "gate": "C-right/C/D/E first-step PASS",
        },
        "phase4i_e_verdict": (
            "INCOMPLETE" if valid_candidates else "FAIL"
        ),
        "phase4i_baseline_resume": False,
        "medium_fine_1p5mm_baseline_allowed": False,
    }
    _write_json(output / "summary.json", summary)
    _write_csv(output / "mirror_comparison.csv", comparisons)
    _write_csv(output / "orientation_matrix.csv", orientation_rows)
    print(
        "Phase 4I-E orientation matrix: "
        f"{summary['phase4i_e_verdict']} "
        f"({orientation_decision['hypothesis']})"
    )
    return 0 if valid_candidates else 1


def _gate_only_mode(
    arguments: argparse.Namespace, requested: str
) -> int:
    summary_path = arguments.output_directory / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            "run --run-orientation-matrix before gated modes"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if requested == "regression":
        allowed = summary["orientation_decision"][
            "source_level_fix_applied"
        ]
        reason = "no physically validated source-level fix"
        key = "regression_cases"
    else:
        allowed = (
            summary.get("regression_cases", {}).get("status")
            == "PASS"
        )
        reason = "regression first-step gate did not pass"
        key = "full_trials"
    if not allowed:
        summary[key] = {
            "status": "NOT_RUN",
            "reason": reason,
        }
        _write_json(summary_path, summary)
        print(f"Phase 4I-E {requested}: NOT_RUN ({reason})")
        return 1
    raise RuntimeError(
        f"{requested} execution requires the validated source correction "
        "to be implemented before this gate can open"
    )


def main() -> int:
    arguments = _parse_arguments()
    if arguments._audit_child:
        return _run_child(arguments)
    if arguments.run_regression_cases:
        return _gate_only_mode(arguments, "regression")
    if arguments.run_full_trials:
        return _gate_only_mode(arguments, "full trial")
    return _run_orientation_matrix(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
