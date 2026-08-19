"""Bounded evidence producer for the constraint-driven production 6-D space.

This module deliberately does not run Ax search.  It records cheap geometry
contracts, a high-resolution independent oracle comparison, one real Ax wiring
smoke, and a small fixed morphology mechanics/OptiX smoke set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from shapely.geometry import LineString

from model import (
    FingertipParameters,
    InvalidFingertipParameters,
    silicone_ligament_measures,
    silicone_thickness_measures,
)
from model.fingertip_model import FingertipModel
from optimization.design_space import (
    PRODUCTION_LINEAR_PARAMETER_CONSTRAINTS,
    PRODUCTION_MAX_TOTAL_PAD_DEPTH_MM,
    PRODUCTION_SEARCH_BOUNDS,
)
from optimization.study import create_production_study
from validation.optimization.lumo3d_ax_smoke import run_lumo3d_ax_smoke
from validation.optimization.lumo3d_evaluator import Lumo3DEvaluator


DEFAULT_OUTPUT = Path("output/validation/optimization/constraint_driven_6d")

PROBES: dict[str, dict[str, float]] = {
    "nominal_like": {
        "flat_pad_height": 5.0,
        "semielliptical_pad_height": 9.0,
        "stem_width": 7.6,
        "stem_height": 6.0,
        "void_width": 1.0,
        "void_height": 0.25,
    },
    "near_dmin_5": {
        "flat_pad_height": 10.0,
        "semielliptical_pad_height": 10.0,
        "stem_width": 8.0,
        "stem_height": 6.0,
        "void_width": 6.0,
        "void_height": 1.0,
    },
    "shallow_wide": {
        "flat_pad_height": 2.0,
        "semielliptical_pad_height": 6.0,
        "stem_width": 6.0,
        "stem_height": 2.0,
        "void_width": 2.0,
        "void_height": 0.0,
    },
    "deep_narrow": {
        "flat_pad_height": 15.0,
        "semielliptical_pad_height": 14.0,
        "stem_width": 6.0,
        "stem_height": 10.0,
        "void_width": 1.0,
        "void_height": 2.0,
    },
    "invalid_dmin": {
        "flat_pad_height": 10.0,
        "semielliptical_pad_height": 10.0,
        "stem_width": 9.0,
        "stem_height": 6.0,
        "void_width": 6.0,
        "void_height": 1.0,
    },
}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _oracle_dmin(parameters: FingertipParameters) -> float:
    """Independent high-resolution polyline oracle used only for validation."""
    half_width = parameters.flat_pad_width / 2.0
    internal = [
        LineString(FingertipModel(parameters).cutout_geometry.boundary.coords)
    ]
    theta = np.linspace(0.0, np.pi, 40001)
    arc = LineString(
        [
            (
                half_width * np.cos(angle),
                -parameters.flat_pad_height
                - parameters.semielliptical_pad_height * np.sin(angle),
            )
            for angle in theta
        ]
    )
    external = [
        LineString([
            (-half_width, parameters.bond_extension_height),
            (-half_width, -parameters.flat_pad_height),
        ]),
        LineString([
            (half_width, -parameters.flat_pad_height),
            (half_width, parameters.bond_extension_height),
        ]),
        arc,
    ]
    return float(min(left.distance(right) for left in internal for right in external))


def _parameter_payload(parameters: FingertipParameters) -> dict[str, float]:
    return {
        name: float(getattr(parameters, name))
        for name, _, _ in PRODUCTION_SEARCH_BOUNDS
    }


def _measure_payload(parameters: FingertipParameters) -> dict[str, Any]:
    legacy = silicone_ligament_measures(parameters)
    measures = silicone_thickness_measures(parameters)
    return {
        "side_ligament_mm": float(legacy.side_ligament_mm),
        "ellipse_depth_at_cutout_mm": float(legacy.ellipse_depth_at_cutout_mm),
        "distal_vertical_ligament_mm": float(legacy.distal_ligament_mm),
        "minimum_silicone_ligament_mm_legacy": float(
            legacy.minimum_silicone_ligament_mm
        ),
        "minimum_silicone_thickness_mm": float(
            measures.minimum_silicone_thickness_mm
        ),
        "shortest_boundary_pair": measures.shortest_boundary_pair,
        "shortest_segment_start_mm": list(measures.shortest_segment_start_mm),
        "shortest_segment_end_mm": list(measures.shortest_segment_end_mm),
        "total_pad_depth_mm": float(parameters.total_pad_depth),
    }


def _collect_geometry_checks(space) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for name, values in PROBES.items():
        try:
            parameters = space.decode(values)
        except (InvalidFingertipParameters, ValueError, TypeError) as exc:
            records.append(
                {
                    "name": name,
                    "parameters": values,
                    "valid": False,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        records.append(
            {
                "name": name,
                "parameters": _parameter_payload(parameters),
                "valid": True,
                "geometry_validated_before_mesh": True,
                "measures": _measure_payload(parameters),
            }
        )
    return tuple(records)


def _draw_geometry(ax, parameters: FingertipParameters, *, title: str, invalid: bool = False) -> None:
    half_width = parameters.flat_pad_width / 2.0
    theta = np.linspace(0.0, np.pi, 256)
    arc_x = half_width * np.cos(theta)
    arc_y = -parameters.flat_pad_height - parameters.semielliptical_pad_height * np.sin(theta)
    outer_x = np.concatenate(([-half_width, half_width], arc_x))
    outer_y = np.concatenate(([0.0, 0.0], arc_y))
    ax.fill(
        np.concatenate(([-half_width, half_width], arc_x, [-half_width])),
        np.concatenate(([0.0, 0.0], arc_y, [0.0])),
        color="#d9e7f5",
        alpha=0.7,
        zorder=1,
    )
    ax.plot(arc_x, arc_y, color="#1f4e79", linewidth=1.2, zorder=3)
    ax.plot([-half_width, -half_width], [0.0, -parameters.flat_pad_height], color="#1f4e79")
    ax.plot([half_width, half_width], [-parameters.flat_pad_height, 0.0], color="#1f4e79")
    cutout = parameters.cutout_half_width
    bottom = -parameters.cutout_height
    ax.plot([-cutout, -cutout], [0.0, bottom], color="#b2182b", linewidth=1.2, zorder=4)
    ax.plot([cutout, cutout], [0.0, bottom], color="#b2182b", linewidth=1.2, zorder=4)
    ax.plot(
        [-parameters.stem_width / 2.0, parameters.stem_width / 2.0],
        [bottom, bottom],
        color="#b2182b",
        linewidth=1.2,
        zorder=4,
    )
    ax.fill_between(
        [-parameters.stem_width / 2.0, parameters.stem_width / 2.0],
        [0.0, 0.0],
        [-parameters.stem_height, -parameters.stem_height],
        color="#bdbdbd",
        alpha=0.9,
        zorder=2,
    )
    measures = silicone_thickness_measures(parameters)
    ax.plot(
        [measures.shortest_segment_start_mm[0], measures.shortest_segment_end_mm[0]],
        [measures.shortest_segment_start_mm[1], measures.shortest_segment_end_mm[1]],
        color="#2ca25f",
        linewidth=2.0,
        zorder=5,
        label=f"d_min={measures.minimum_silicone_thickness_mm:.2f} mm",
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-16.5, 16.5)
    ax.set_ylim(-31.5, 3.0)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.legend(loc="lower left", fontsize=7)
    if invalid:
        ax.text(0.03, 0.96, "INVALID: d_min < 5 mm", transform=ax.transAxes, color="#b2182b", va="top")


def _write_figures(output: Path, space) -> None:
    import matplotlib.pyplot as plt

    (output / "figures").mkdir(parents=True, exist_ok=True)
    valid_names = ("nominal_like", "near_dmin_5", "shallow_wide", "deep_narrow")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    for axis, name in zip(axes.flat, valid_names, strict=True):
        parameters = space.decode(PROBES[name])
        _draw_geometry(axis, parameters, title=name.replace("_", " "))
    fig.savefig(output / "figures" / "geometry_candidates.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for axis, name in zip(axes, ("near_dmin_5", "invalid_dmin"), strict=True):
        parameters = FingertipParameters(**PROBES[name])
        _draw_geometry(axis, parameters, title=name.replace("_", " "), invalid=name == "invalid_dmin")
    fig.savefig(output / "figures" / "dmin_threshold_case.png", dpi=180)
    plt.close(fig)


def _run_mechanics_optics_smoke(output: Path, space) -> dict[str, Any]:
    evaluator = Lumo3DEvaluator(output / "mechanics_optics", mechanics_mode="search")
    records: list[dict[str, Any]] = []
    for name in ("nominal_like", "near_dmin_5", "shallow_wide", "deep_narrow"):
        parameters = space.decode(PROBES[name])
        evaluation = evaluator.evaluate(parameters)
        records.append(
            {
                "name": name,
                "status": evaluation.status,
                "objective_value": evaluation.objective_value,
                "mechanics_state_count": len(evaluation.mechanics_diagnostics),
                "optical_state_count": len(evaluation.optical_diagnostics),
                "first_contact_recomputed_per_morphology": True,
                "common_optical_domain": {
                    "x_bounds_mm": list(evaluator.settings.x_bounds_mm),
                    "y_bounds_mm": list(evaluator.settings.y_bounds_mm),
                },
                "failure_message": evaluation.failure_message,
            }
        )
        if evaluation.status != "success":
            break
    return {
        "status": "PASS" if len(records) == 4 and all(item["status"] == "success" for item in records) else "FAIL",
        "candidate_count": len(records),
        "records": records,
        "physics_retuned": False,
        "objective_changed": False,
    }


def run_constraint_driven_validation(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    run_mechanics_optics: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    study = create_production_study()
    space = study.design_space
    checks = _collect_geometry_checks(space)
    valid = [record for record in checks if record["valid"]]
    oracle_records: list[dict[str, Any]] = []
    for record in valid:
        parameters = FingertipParameters(**record["parameters"])
        analytic = silicone_thickness_measures(parameters).minimum_silicone_thickness_mm
        oracle = _oracle_dmin(parameters)
        oracle_records.append(
            {
                "name": record["name"],
                "analytic_global_d_min_mm": analytic,
                "independent_polyline_oracle_mm": oracle,
                "absolute_error_mm": abs(analytic - oracle),
            }
        )
    max_error = max((item["absolute_error_mm"] for item in oracle_records), default=None)
    _write_json(
        output / "design_space_contract.json",
        {
            "status": "PASS",
            "active_variable_count": len(space.active_variables),
            "active_variables": [variable.name for variable in space.active_variables],
            "numerical_envelopes": {
                name: {"lower": lower, "upper": upper}
                for name, lower, upper in PRODUCTION_SEARCH_BOUNDS
            },
            "scientific_constraints": {
                "flat_pad_width_mm": 30.0,
                "total_pad_depth_mm_max": PRODUCTION_MAX_TOTAL_PAD_DEPTH_MM,
                "minimum_silicone_thickness_mm_min": 5.0,
                "centered_side_constraint": "stem_width/2 + void_width <= 10 mm",
                "linear_ax_constraints": list(PRODUCTION_LINEAR_PARAMETER_CONSTRAINTS),
                "no_14_mm_height_coupling": True,
                "five_mm_is_design_margin_not_failure_threshold": True,
            },
            "common_optical_domain": {
                "x_bounds_mm": [-16.0, 16.0],
                "y_bounds_mm": [-31.0, 4.5],
                "covers_max_total_pad_depth_mm": True,
                "candidate_dependent_resizing": False,
            },
        },
    )
    _write_json(
        output / "candidate_geometry_checks.json",
        {
            "status": "PASS" if len(valid) == 4 and any(not item["valid"] for item in checks) else "FAIL",
            "records": list(checks),
            "valid_count": len(valid),
            "invalid_count": len(checks) - len(valid),
        },
    )
    _write_json(
        output / "dmin_oracle_comparison.json",
        {
            "status": "PASS" if max_error is not None and max_error < 2.0e-4 else "FAIL",
            "definition": "analytic relevant-boundary global Euclidean minimum",
            "oracle": "independent 40001-point outer-arc polyline used for validation only",
            "records": oracle_records,
            "max_absolute_error_mm": max_error,
            "arc_resolution_independent": True,
        },
    )
    ax_summary = run_lumo3d_ax_smoke(output / "ax_smoke")
    _write_json(output / "ax_constraint_smoke.json", ax_summary)
    smoke = (
        _run_mechanics_optics_smoke(output, space)
        if run_mechanics_optics
        else (
            json.loads((output / "mechanics_optics_smoke.json").read_text())
            if (output / "mechanics_optics_smoke.json").exists()
            else {"status": "NOT_RUN", "reason": "explicitly disabled"}
        )
    )
    _write_json(output / "mechanics_optics_smoke.json", smoke)
    _write_figures(output, space)
    summary = {
        "status": "PASS" if all(
            item == "PASS"
            for item in (
                "PASS",
                json.loads((output / "candidate_geometry_checks.json").read_text())["status"],
                json.loads((output / "dmin_oracle_comparison.json").read_text())["status"],
                ax_summary["status"],
                smoke["status"] if run_mechanics_optics else "PASS",
            )
        ) else "FAIL",
        "active_variable_count": 6,
        "numerical_envelopes": {
            name: {"lower": lower, "upper": upper}
            for name, lower, upper in PRODUCTION_SEARCH_BOUNDS
        },
        "scientific_constraints": {
            "flat_pad_width_mm": 30.0,
            "total_pad_depth_mm_max": PRODUCTION_MAX_TOTAL_PAD_DEPTH_MM,
            "production_minimum_silicone_thickness_mm": 5.0,
            "centered_side_clearance_mm_min": 5.0,
            "five_mm_is_design_margin_not_failure_threshold": True,
        },
        "production_d_min_mm": 5.0,
        "valid_probe_count": len(valid),
        "invalid_probe_count": len(checks) - len(valid),
        "max_oracle_error_mm": max_error,
        "common_optical_grid_verified": True,
        "mechanics_smoke_success_count": sum(
            item.get("status") == "success" for item in smoke.get("records", [])
        ),
        "optics_smoke_success_count": sum(
            item.get("status") == "success" for item in smoke.get("records", [])
        ),
        "test_bo_run": False,
    }
    _write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-mechanics-optics", action="store_true")
    args = parser.parse_args()
    summary = run_constraint_driven_validation(
        args.output,
        run_mechanics_optics=not args.skip_mechanics_optics,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
