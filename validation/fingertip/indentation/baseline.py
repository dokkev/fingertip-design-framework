"""Run and compare the Phase 4I central fingertip indentation cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from fem.indentation import (
    IndentationSettings,
    inspect_indentation_runtime_contract,
    run_indentation_case,
)
from fem.results import profile_error_metrics
from validation.common.io import write_indentation_case_outputs
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    level_group = parser.add_mutually_exclusive_group()
    level_group.add_argument(
        "--mesh-level", choices=("medium", "fine"), default="medium"
    )
    level_group.add_argument(
        "--mesh-levels", nargs="+", choices=("medium", "fine")
    )
    parser.add_argument("--indentation-mm", type=float, default=1.5)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("output/validation/fingertip/indentation/baseline"),
    )
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_case-directory", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _depth_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _case_directory(
    root: Path,
    mesh_level: str,
    indentation_mm: float,
    trial: bool,
) -> Path:
    if trial:
        return root / f"trial_{mesh_level}_{_depth_label(indentation_mm)}"
    return root / f"baseline_{mesh_level}"


def _run_child(arguments: argparse.Namespace) -> int:
    if arguments._case_directory is None:
        raise ValueError("--_case-directory is required for an internal child run")
    model = FingertipModel(FingertipParameters())
    settings = IndentationSettings(
        indentation_mm=arguments.indentation_mm,
        number_of_steps=arguments.steps,
    )
    result, artifacts = run_indentation_case(
        model,
        arguments.mesh_level,
        settings,
    )
    outputs = write_indentation_case_outputs(
        result, artifacts, arguments._case_directory
    )
    result["outputs"] = outputs
    _write_json(arguments._case_directory / "result.json", result)
    print(
        f"Phase 4I {arguments.mesh_level}: {result['status']} "
        f"({len(result.get('history', []))}/{arguments.steps} steps)"
    )
    return 0 if result["status"] == "PASS" else 1


def _run_case_subprocess(
    mesh_level: str,
    indentation_mm: float,
    steps: int,
    trial: bool,
    output_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    case_directory = _case_directory(
        output_root, mesh_level, indentation_mm, trial
    )
    case_directory.mkdir(parents=True, exist_ok=True)
    settings = IndentationSettings(indentation_mm, steps)
    model = FingertipModel(FingertipParameters())
    preflight = inspect_indentation_runtime_contract(
        model, mesh_level, settings
    )
    _write_json(case_directory / "preflight.json", preflight)
    command = [
        sys.executable,
        "-B",
        "-m",
        "validation.fingertip.indentation.baseline",
        "--mesh-level",
        mesh_level,
        "--indentation-mm",
        str(indentation_mm),
        "--steps",
        str(steps),
        "--output-directory",
        str(output_root),
        "--_child",
        "--_case-directory",
        str(case_directory),
    ]
    if trial:
        command.append("--trial")
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONFAULTHANDLER"] = "1"
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=7200,
        preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)),
    )
    solver_log = completed.stdout + completed.stderr
    (case_directory / "solver.log").write_text(solver_log, encoding="utf-8")
    result_path = case_directory / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = {
            "phase": "4I",
            "mesh_level": mesh_level,
            "status": "FAIL",
            "solve_status": "FAIL",
            "history": [],
            "failure_reason": "case_process_terminated_without_result",
            "process_exit_code": completed.returncode,
            "process_signal": (
                -completed.returncode if completed.returncode < 0 else None
            ),
            "process_output_tail": solver_log[-12000:],
            "preflight": preflight,
            "reproduction_command": command,
        }
    result["preflight"] = preflight
    result["process_exit_code"] = completed.returncode
    result["reproduction_command"] = command
    result.setdefault("outputs", {})["solver_log"] = str(
        (case_directory / "solver.log").resolve()
    )
    result["outputs"]["preflight"] = str(
        (case_directory / "preflight.json").resolve()
    )
    _write_json(result_path, result)
    return result, command


def _history_point_at_depth(
    result: Mapping[str, Any], depth_mm: float
) -> Mapping[str, Any] | None:
    candidates = [
        point
        for point in result.get("history", [])
        if abs(float(point["achieved_indentation_mm"]) - depth_mm) <= 1.0e-9
    ]
    return candidates[0] if candidates else None


def _read_profile(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    numeric_fields = (
        "node_id",
        "reference_x_mm",
        "reference_y_mm",
        "normalized_arc_coordinate",
        "tangent_coordinate_from_crown_mm",
        "ux_mm",
        "uy_mm",
        "local_normal_displacement_mm",
        "local_tangential_displacement_mm",
        "deformed_x_mm",
        "deformed_y_mm",
    )
    for record in records:
        for field in numeric_fields:
            record[field] = int(record[field]) if field == "node_id" else float(record[field])
    return records


def _interpolated_profile_values(
    profile: Sequence[Mapping[str, Any]],
    common: np.ndarray,
    field: str,
) -> np.ndarray:
    coordinate = np.asarray(
        [record["normalized_arc_coordinate"] for record in profile], dtype=float
    )
    values = np.asarray([record[field] for record in profile], dtype=float)
    return np.interp(common, coordinate, values)


def _symmetry_metrics(
    profile: Sequence[Mapping[str, Any]],
    floor_mm: float,
) -> dict[str, Any]:
    half = np.linspace(0.0, 0.5, 251)
    mirror = 1.0 - half
    normal_first = _interpolated_profile_values(
        profile, half, "local_normal_displacement_mm"
    )
    normal_second = _interpolated_profile_values(
        profile, mirror, "local_normal_displacement_mm"
    )
    tangent_first = _interpolated_profile_values(
        profile, half, "local_tangential_displacement_mm"
    )
    tangent_second = _interpolated_profile_values(
        profile, mirror, "local_tangential_displacement_mm"
    )
    return {
        "normal_even_symmetry": profile_error_metrics(
            normal_first, normal_second, floor_mm
        ),
        "tangential_odd_symmetry": profile_error_metrics(
            tangent_first, -tangent_second, floor_mm
        ),
    }


def _compare_profiles(
    medium_profile: Sequence[Mapping[str, Any]],
    fine_profile: Sequence[Mapping[str, Any]],
    active_fine_node_ids: Sequence[int],
    floor_mm: float,
) -> dict[str, Any]:
    common = np.linspace(0.0, 1.0, 501)
    result: dict[str, Any] = {
        "common_grid_point_count": len(common),
        "normal": profile_error_metrics(
            _interpolated_profile_values(
                medium_profile, common, "local_normal_displacement_mm"
            ),
            _interpolated_profile_values(
                fine_profile, common, "local_normal_displacement_mm"
            ),
            floor_mm,
        ),
        "tangential": profile_error_metrics(
            _interpolated_profile_values(
                medium_profile, common, "local_tangential_displacement_mm"
            ),
            _interpolated_profile_values(
                fine_profile, common, "local_tangential_displacement_mm"
            ),
            floor_mm,
        ),
        "medium_symmetry": _symmetry_metrics(medium_profile, floor_mm),
        "fine_symmetry": _symmetry_metrics(fine_profile, floor_mm),
    }
    active_set = set(active_fine_node_ids)
    active_coordinates = [
        float(record["normalized_arc_coordinate"])
        for record in fine_profile
        if int(record["node_id"]) in active_set
    ]
    if active_coordinates:
        lower, upper = min(active_coordinates), max(active_coordinates)
        masks = {
            "contact_zone": (common >= lower) & (common <= upper),
            "side_region": (common < lower) | (common > upper),
        }
        for name, mask in masks.items():
            if np.count_nonzero(mask) >= 2:
                result[name] = profile_error_metrics(
                    _interpolated_profile_values(
                        medium_profile, common, "local_normal_displacement_mm"
                    )[mask],
                    _interpolated_profile_values(
                        fine_profile, common, "local_normal_displacement_mm"
                    )[mask],
                    floor_mm,
                )
        result["fine_contact_zone_normalized_arc_range"] = [lower, upper]
    return result


def _relative_difference(first: float, reference: float) -> float:
    return abs(first - reference) / abs(reference) if reference else math.inf


def _compare_cases(
    medium: Mapping[str, Any],
    fine: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    depths = (0.5, 1.0, 1.5)
    settings = medium.get("configuration", {}).get("indentation", {})
    floor = float(settings.get("profile_displacement_floor_mm", 1.0e-5))
    comparisons: dict[str, Any] = {}
    for depth in depths:
        medium_point = _history_point_at_depth(medium, depth)
        fine_point = _history_point_at_depth(fine, depth)
        entry: dict[str, Any] = {
            "available": medium_point is not None and fine_point is not None
        }
        if medium_point is not None and fine_point is not None:
            medium_reaction = float(medium_point["indenter_normal_reaction_n"])
            fine_reaction = float(fine_point["indenter_normal_reaction_n"])
            entry.update(
                {
                    "reaction_n": {
                        "medium": medium_reaction,
                        "fine": fine_reaction,
                    },
                    "reaction_relative_difference": _relative_difference(
                        medium_reaction, fine_reaction
                    ),
                    "contact_width_mm": {
                        "medium_chord": medium_point["external_contact_width"]["chord_width_mm"],
                        "fine_chord": fine_point["external_contact_width"]["chord_width_mm"],
                        "medium_arc": medium_point["external_contact_width"]["arc_length_mm"],
                        "fine_arc": fine_point["external_contact_width"]["arc_length_mm"],
                    },
                }
            )
            medium_profile_path = output_root / "baseline_medium" / "profiles" / f"profile_{_depth_label(depth)}.csv"
            fine_profile_path = output_root / "baseline_fine" / "profiles" / f"profile_{_depth_label(depth)}.csv"
            if medium_profile_path.is_file() and fine_profile_path.is_file():
                entry["outer_arc_profile"] = _compare_profiles(
                    _read_profile(medium_profile_path),
                    _read_profile(fine_profile_path),
                    fine_point["contact_groups"]["external_pad_indenter"]["active_slave_node_ids"],
                    floor,
                )
        comparisons[f"{depth:g}"] = entry
    final = comparisons["1.5"]
    case_pass = medium.get("status") == "PASS" and fine.get("status") == "PASS"
    reaction_pass = bool(final.get("available")) and final.get(
        "reaction_relative_difference", math.inf
    ) < 0.10
    profile_pass = bool(final.get("outer_arc_profile")) and final[
        "outer_arc_profile"
    ]["normal"]["relative_l2_error"] < 0.10
    return {
        "phase": "4I",
        "depth_comparisons_mm": comparisons,
        "acceptance": {
            "both_cases_pass": case_pass,
            "final_reaction_relative_difference_below_10_percent": reaction_pass,
            "final_outer_normal_profile_relative_error_below_10_percent": profile_pass,
            "phase4i_pass": case_pass and reaction_pass and profile_pass,
        },
        "profile_comparison": {
            "coordinate": "reference normalized PadOuterArc arc coordinate",
            "common_grid_points": 501,
            "absolute_displacement_floor_mm": floor,
        },
    }


def _save_mesh_convergence_plot(
    medium: Mapping[str, Any], fine: Mapping[str, Any], path: Path
) -> None:
    import matplotlib.pyplot as plt

    if not medium.get("history") or not fine.get("history"):
        return
    figure, axis = plt.subplots(figsize=(6.6, 4.4))
    for label, result in (("medium", medium), ("fine", fine)):
        axis.plot(
            [point["achieved_indentation_mm"] for point in result["history"]],
            [point["indenter_normal_reaction_n"] for point in result["history"]],
            label=label,
        )
    axis.set(xlabel="Indentation [mm]", ylabel="Normal reaction [N]", title="Phase 4I mesh convergence")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _run_parent(arguments: argparse.Namespace) -> int:
    output_root = arguments.output_directory.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    levels = arguments.mesh_levels or [arguments.mesh_level]
    commands: list[list[str]] = []
    results: dict[str, dict[str, Any]] = {}
    for level in levels:
        result, command = _run_case_subprocess(
            level,
            arguments.indentation_mm,
            arguments.steps,
            arguments.trial,
            output_root,
        )
        results[level] = result
        commands.append(command)
        print(
            f"{level}: {result['status']} "
            f"({len(result.get('history', []))}/{arguments.steps} steps)"
        )
    comparison = None
    if arguments.compare and {"medium", "fine"}.issubset(results):
        comparison = _compare_cases(results["medium"], results["fine"], output_root)
        _write_json(output_root / "mesh_convergence.json", comparison)
        _save_mesh_convergence_plot(
            results["medium"],
            results["fine"],
            output_root / "mesh_convergence.png",
        )
    summary = {
        "phase": "4I",
        "trial": arguments.trial,
        "requested_indentation_mm": arguments.indentation_mm,
        "requested_steps": arguments.steps,
        "case_status": {level: result["status"] for level, result in results.items()},
        "phase4i_status": (
            "PASS"
            if comparison is not None
            and comparison["acceptance"]["phase4i_pass"]
            else (
                "TRIAL_PASS"
                if arguments.trial and all(result["status"] == "PASS" for result in results.values())
                else "FAIL"
            )
        ),
        "comparison": comparison,
        "commands": commands,
    }
    _write_json(output_root / "phase4i_summary.json", summary)
    print(output_root / "phase4i_summary.json")
    return 0 if all(result["status"] == "PASS" for result in results.values()) else 1


def main() -> int:
    arguments = _parse_arguments()
    if arguments._child:
        return _run_child(arguments)
    return _run_parent(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
