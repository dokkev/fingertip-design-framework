"""Strict, atomic validation artifact I/O."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def strict_read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object while rejecting NaN and Infinity."""
    resolved = Path(path)
    value = json.loads(
        resolved.read_text(encoding="utf-8"),
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {constant}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {resolved}")
    return value


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Strictly serialize JSON and atomically replace one artifact."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, resolved)


def write_csv(
    path: str | Path,
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> None:
    """Write a deterministic CSV artifact with an explicit schema."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def write_indentation_history(
    path: Path,
    history: Sequence[Mapping[str, Any]],
) -> None:
    fields = (
        "step",
        "pseudo_time",
        "prescribed_indenter_travel_mm",
        "achieved_indentation_mm",
        "indenter_normal_reaction_n",
        "support_signed_reaction_along_loading_n",
        "force_equilibrium_error",
        "nonlinear_iterations",
        "solver_converged",
        "active_set_converged",
        "external_active_condition_count",
        "internal_left_active_condition_count",
        "internal_right_active_condition_count",
        "internal_bottom_active_condition_count",
        "external_weighted_gap_min",
        "external_weighted_gap_mean",
        "external_contact_chord_width_mm",
        "external_contact_arc_length_mm",
        "maximum_principal_green_lagrange_strain",
        "minimum_pad_det_f",
        "maximum_pad_displacement_mm",
        "maximum_contact_penetration_mm",
        "solve_wall_clock_seconds",
    )
    rows = []
    for point in history:
        groups = point["contact_groups"]
        external = groups["external_pad_indenter"]
        rows.append(
            (
                point["step"],
                point["pseudo_time"],
                point["prescribed_indenter_travel_mm"],
                point["achieved_indentation_mm"],
                point["indenter_normal_reaction_n"],
                point["support_signed_reaction_along_loading_n"],
                point["force_equilibrium_error"],
                point["nonlinear_iterations"],
                point["solver_converged"],
                point["active_set_converged"],
                external["active_condition_count"],
                groups.get("internal_left", {}).get("active_condition_count", 0),
                groups.get("internal_right", {}).get("active_condition_count", 0),
                groups.get("internal_bottom", {}).get("active_condition_count", 0),
                external["weighted_gap"]["min"],
                external["weighted_gap"]["mean"],
                point["external_contact_width"]["chord_width_mm"],
                point["external_contact_width"]["arc_length_mm"],
                point["pad_strain_det_f"][
                    "maximum_principal_green_lagrange_strain"
                ]["value"],
                point["pad_strain_det_f"]["det_f"]["min"],
                point["maximum_pad_displacement_mm"],
                max(
                    float(
                        group["signed_geometric_gap"].get(
                            "maximum_penetration_mm"
                        )
                        or 0.0
                    )
                    for group in groups.values()
                ),
                point["solve_wall_clock_seconds"],
            )
        )
    write_csv(path, fields, rows)


def _write_indentation_profile(
    path: Path,
    profile: Sequence[Mapping[str, Any]],
) -> None:
    fields = (
        "node_id",
        "reference_x_mm",
        "reference_y_mm",
        "normalized_arc_coordinate",
        "tangent_coordinate_from_crown_mm",
        "side",
        "ux_mm",
        "uy_mm",
        "local_normal_displacement_mm",
        "local_tangential_displacement_mm",
        "deformed_x_mm",
        "deformed_y_mm",
    )
    write_csv(
        path,
        fields,
        ([record[field] for field in fields] for record in profile),
    )


def write_indentation_case_outputs(
    result: Mapping[str, Any],
    artifacts: Any | None,
    output_directory: str | Path,
) -> dict[str, str]:
    """Write validation artifacts without adding I/O to the FEM backend."""
    from visualization.indentation import (
        save_deformed_mesh_plot,
        save_history_plots,
        save_outer_profile_plot,
    )

    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    history_path = directory / "history.csv"
    atomic_write_json(result_path, result)
    write_indentation_history(history_path, result.get("history", []))
    outputs = {"result": str(result_path), "history": str(history_path)}
    if artifacts is None:
        return outputs
    profiles_directory = directory / "profiles"
    plots_directory = directory / "plots"
    for key, snapshot in artifacts.snapshots.items():
        label = str(key).replace(".", "p")
        profile_path = profiles_directory / f"profile_{label}.csv"
        _write_indentation_profile(profile_path, snapshot["profile"])
        outputs[f"profile_{label}"] = str(profile_path)
        deformed_path = plots_directory / f"deformed_mesh_{label}.png"
        save_deformed_mesh_plot(artifacts, snapshot, deformed_path)
        outputs[f"deformed_mesh_{label}"] = str(deformed_path)
    save_history_plots(result, plots_directory)
    save_outer_profile_plot(
        artifacts.snapshots, plots_directory / "outer_arc_profiles.png"
    )
    if artifacts.snapshots:
        final_key = max(artifacts.snapshots, key=float)
        final_path = directory / "deformed_mesh.png"
        save_deformed_mesh_plot(
            artifacts, artifacts.snapshots[final_key], final_path
        )
        outputs["deformed_mesh"] = str(final_path)
    return outputs
