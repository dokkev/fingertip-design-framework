"""Focused OptiX 11 mm single-cell dimensional validation.

This module owns the experiment and artifacts.  The transport implementation
does not know about morphology candidates, FEM scenarios, or historical
validation values.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import numpy as np

from fem import solve
from mesh import mesh_settings_for_level
from mesh.indenter import IndenterSettings
from model import Fingertip, FingertipParameters
from optics import trace
from optics.cross_section.settings import TraceSettings
from optics.transport3d import Transport3DSettings, trace_3d
from optics.transport3d.optix_backend import _Runtime
from optimization.scenarios import ContactScenario


OUTPUT = Path("output/validation/optics/transport3d")
DEPTH_MM = 11.0
CONTACTS = {
    "left_contact": ContactScenario(-3.0, 0.5, 4.0),
    "right_contact": ContactScenario(3.0, 0.5, 4.0),
}
CANDIDATE49 = {
    "flat_pad_width": 30.0,
    "flat_pad_height": 3.937175708822906,
    "semielliptical_pad_height": 7.309789158403873,
    "stem_width": 7.289858109783381,
    "stem_height": 5.102298432029784,
    "void_width": 0.6931721470318735,
    "void_height": 1.2690955214202404,
}
PLANAR_RAY_COUNT = 161
CONVERGENCE_RAY_COUNTS = (4096, 16384, 65536)
CONVERGENCE_WEIGHT_TOLERANCE = 1.0e-3
CONVERGENCE_TV_TOLERANCE = 1.0e-3
HISTORICAL_REDUCED_2D_REFERENCE = {
    "nominal": 0.07510662148936212,
    "candidate49": 0.12737679674855767,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status() -> str:
    return subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _tv(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("fields must have matching 2D shapes")
    first_mass = float(np.sum(first))
    second_mass = float(np.sum(second))
    if first_mass <= 0.0 or second_mass <= 0.0:
        raise ValueError("fields must have positive mass")
    return 0.5 * float(np.sum(np.abs(first / first_mass - second / second_mass)))


def _state_trace_2d(tip: Fingertip, state_mesh: Any, *, ray_count: int = PLANAR_RAY_COUNT) -> Any:
    return trace(
        tip,
        state_mesh,
        TraceSettings(ray_count=ray_count),
    )


def _state_trace_3d(
    tip: Fingertip,
    state_mesh: Any,
    reference_mesh: Any,
    runtime: Any,
    *,
    mode: str,
    ray_count: int,
    retain_projected_segments: bool = False,
    minimum_ray_weight: float = 1.0e-4,
) -> Any:
    return trace_3d(
        tip,
        state_mesh,
        reference_mesh=reference_mesh,
        runtime=runtime,
        settings=Transport3DSettings(
            mode=mode,  # type: ignore[arg-type]
            ray_count=ray_count,
            minimum_ray_weight=minimum_ray_weight,
            maximum_segment_count=max(20000, 24 * ray_count),
            extrusion_depth_mm=DEPTH_MM,
            retain_projected_segments=retain_projected_segments,
        ),
    )


def _solve_contact(tip: Fingertip, mesh: Any, scenario: ContactScenario) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    result = solve(
        tip,
        mesh,
        indentation=scenario.indentation_mm,
        surface_x_mm=scenario.location_x_mm,
        steps=48,
        indenter=IndenterSettings(radius_mm=scenario.indenter_radius_mm),
        internal_contact="three_pairs",
    )
    record = {
        "converged": bool(result.converged),
        "wall_time_seconds": time.perf_counter() - started,
        "scenario": asdict(scenario),
        "reaction_force_n": result.reaction_force,
    }
    if not result.converged:
        raise RuntimeError(
            f"FEM did not converge for x={scenario.location_x_mm}: "
            f"{result.details.get('failure_reason', 'unknown reason')}"
        )
    return result.deformed_mesh, record


def _planar_gate(
    runtime: Any,
    designs: Mapping[str, Mapping[str, Any]],
    output: Path,
    state_names: tuple[str, ...] = ("reference", "left_contact"),
) -> tuple[dict[str, Any], bool]:
    records: dict[str, Any] = {}
    gate_pass = True
    for name, design in designs.items():
        tip = design["tip"]
        reference_mesh = design["mesh"]
        states = {"reference": reference_mesh}
        if "left_contact" in state_names:
            states["left_contact"] = design["states"]["left_contact"]
        if "right_contact" in state_names:
            states["right_contact"] = design["states"]["right_contact"]
        state_records: dict[str, Any] = {}
        for state_name, state_mesh in states.items():
            reduced = _state_trace_2d(tip, state_mesh)
            planar = _state_trace_3d(
                tip,
                state_mesh,
                reference_mesh,
                runtime,
                mode="planar",
                ray_count=PLANAR_RAY_COUNT,
                retain_projected_segments=True,
            )
            projected = planar.projected_weighted_path_density
            if projected is None:
                raise RuntimeError("planar trace did not retain its projected diagnostic")
            density_tv = _tv(reduced.density, projected)
            energy_error = max(
                abs(planar.escaped_weight - reduced.escaped_weight),
                abs(planar.absorbed_weight - reduced.absorbed_weight),
                abs(planar.terminated_weight - reduced.terminated_weight),
            )
            state_records[state_name] = {
                "launched_weight_2d": reduced.launched_weight,
                "launched_weight_optix_planar": planar.launched_weight,
                "escaped_weight_2d": reduced.escaped_weight,
                "escaped_weight_optix_planar": planar.escaped_weight,
                "absorbed_weight_2d": reduced.absorbed_weight,
                "absorbed_weight_optix_planar": planar.absorbed_weight,
                "terminated_weight_2d": reduced.terminated_weight,
                "terminated_weight_optix_planar": planar.terminated_weight,
                "energy_fraction_max_abs_error": energy_error,
                "projected_field_tv": density_tv,
            }
            gate_pass &= energy_error <= 1.0e-4 and density_tv <= 5.0e-3
        records[name] = state_records
    (output / "planar_consistency").mkdir(parents=True, exist_ok=True)
    (output / "planar_consistency" / "summary.json").write_text(
        json.dumps({"pass": gate_pass, "records": records}, indent=2, sort_keys=True) + "\n"
    )
    return records, gate_pass


def _run_full_state(
    runtime: Any,
    designs: Mapping[str, Mapping[str, Any]],
    ray_count: int,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, design in designs.items():
        design_runtime = runtime[name] if isinstance(runtime, Mapping) else runtime
        tip = design["tip"]
        reference_mesh = design["mesh"]
        state_results: dict[str, Any] = {}
        for state_name in ("left_contact", "right_contact"):
            result = _state_trace_3d(
                tip,
                design["states"][state_name],
                reference_mesh,
                design_runtime,
                mode="full3d",
                ray_count=ray_count,
                minimum_ray_weight=1.0e-4,
            )
            state_results[state_name] = result
        left = state_results["left_contact"]
        right = state_results["right_contact"]
        if float(np.sum(left.outgoing_surface_field)) <= 0.0 or float(np.sum(right.outgoing_surface_field)) <= 0.0:
            raise RuntimeError(
                f"zero outgoing surface field for {name}: "
                f"left={left.outgoing_surface_weight:g}, "
                f"right={right.outgoing_surface_weight:g}"
            )
        records[name] = {
            "left": left,
            "right": right,
            "tv": _tv(left.outgoing_surface_field, right.outgoing_surface_field),
            "left_outgoing_surface_weight": left.outgoing_surface_weight,
            "right_outgoing_surface_weight": right.outgoing_surface_weight,
            "left_escaped_fraction": left.escaped_weight / left.launched_weight,
            "right_escaped_fraction": right.escaped_weight / right.launched_weight,
            "left_absorbed_fraction": left.absorbed_weight / left.launched_weight,
            "right_absorbed_fraction": right.absorbed_weight / right.launched_weight,
            "left_energy_balance_error": left.energy_balance_error,
            "right_energy_balance_error": right.energy_balance_error,
        }
    return records


def _save_fields(output: Path, records: Mapping[str, Any], ray_count: int) -> str:
    path = output / "nominal_candidate49" / "fields.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for name, record in records.items():
        for state_name in ("left", "right"):
            result = record[state_name]
            prefix = f"{name}_{state_name}"
            arrays[f"{prefix}_field"] = result.outgoing_surface_field
            arrays[f"{prefix}_u_edges"] = result.surface_u_edges
            arrays[f"{prefix}_z_edges"] = result.surface_z_edges
            arrays[f"{prefix}_escape_positions"] = result.escape_positions_mm
            arrays[f"{prefix}_escape_directions"] = result.escape_directions
            arrays[f"{prefix}_escape_normals"] = result.escape_surface_normals
            arrays[f"{prefix}_escape_u"] = result.escape_surface_u
            arrays[f"{prefix}_escape_z"] = result.escape_surface_z
            arrays[f"{prefix}_escape_weights"] = result.escape_weights
    np.savez_compressed(path, **arrays)
    return str(path)


def run_validation(output: Path = OUTPUT) -> dict[str, Any]:
    """Run the four-state focused experiment and persist machine-readable output."""
    output.mkdir(parents=True, exist_ok=True)
    runtime = _Runtime.create()
    design_parameters = {
        "nominal": FingertipParameters(),
        "candidate49": FingertipParameters(**CANDIDATE49),
    }
    designs: dict[str, dict[str, Any]] = {}
    for name, parameters in design_parameters.items():
        tip = Fingertip(parameters)
        mesh = tip.mesh(mesh_settings_for_level("medium"))
        designs[name] = {"tip": tip, "mesh": mesh, "states": {}}
        left, left_record = _solve_contact(tip, mesh, CONTACTS["left_contact"])
        designs[name]["states"]["left_contact"] = left
        designs[name].setdefault("fem", {})["left_contact"] = left_record

    planar_records, planar_pass = _planar_gate(runtime, designs, output)
    if not planar_pass:
        summary = {
            "status": "3D VALIDATION BLOCKED BY PLANAR INCONSISTENCY",
            "planar_gate_pass": False,
            "planar_consistency": planar_records,
            "runtime": runtime.metadata,
            "timestamp": _now(),
            "git_revision": _git_revision(),
            "git_status": _git_status(),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary

    for name, design in designs.items():
        tip = design["tip"]
        mesh = design["mesh"]
        right, right_record = _solve_contact(tip, mesh, CONTACTS["right_contact"])
        design["states"]["right_contact"] = right
        design["fem"]["right_contact"] = right_record

    # Re-run the same planar comparison for both loaded states after all four
    # FEM states exist; this is the reported dimensional-consistency table.
    planar_records, planar_pass = _planar_gate(
        runtime,
        designs,
        output,
        state_names=("reference", "left_contact", "right_contact"),
    )
    if not planar_pass:
        summary = {
            "status": "3D VALIDATION BLOCKED BY PLANAR INCONSISTENCY",
            "planar_gate_pass": False,
            "planar_consistency": planar_records,
            "runtime": runtime.metadata,
            "timestamp": _now(),
            "git_revision": _git_revision(),
            "git_status": _git_status(),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        return summary

    # Use a clean OptiX context for the full-3D sweep.  The planar gate keeps
    # extra projected segments and rebuilds all three GASes per state; keeping
    # that context alive during the convergence sweep can retain substantial
    # driver-side allocations across the two diagnostic modes.
    planar_runtime_metadata = dict(runtime.metadata)
    runtime.cp.cuda.Stream.null.synchronize()
    runtime.cp.get_default_memory_pool().free_all_blocks()
    del runtime
    full_runtime = _Runtime.create()

    convergence: dict[str, Any] = {}
    selected_ray_count: int | None = None
    convergence_pass = False
    previous: dict[str, Any] | None = None
    selected_records: dict[str, Any] | None = None
    for ray_count in CONVERGENCE_RAY_COUNTS:
        records = _run_full_state(full_runtime, designs, ray_count)
        convergence[str(ray_count)] = {
            name: {
                "tv": record["tv"],
                "left_outgoing_surface_weight": record["left_outgoing_surface_weight"],
                "right_outgoing_surface_weight": record["right_outgoing_surface_weight"],
                "left_escaped_fraction": record["left_escaped_fraction"],
                "right_escaped_fraction": record["right_escaped_fraction"],
                "left_absorbed_fraction": record["left_absorbed_fraction"],
                "right_absorbed_fraction": record["right_absorbed_fraction"],
                "left_energy_balance_error": record["left_energy_balance_error"],
                "right_energy_balance_error": record["right_energy_balance_error"],
            }
            for name, record in records.items()
        }
        stable = False
        if previous is not None:
            deltas = []
            for name in records:
                deltas.append(abs(records[name]["tv"] - previous[name]["tv"]))
                for side in ("left", "right"):
                    deltas.append(
                        abs(
                            records[name][f"{side}_outgoing_surface_weight"]
                            - previous[name][f"{side}_outgoing_surface_weight"]
                        )
                    )
                    deltas.append(
                        abs(
                            records[name][f"{side}_absorbed_fraction"]
                            - previous[name][f"{side}_absorbed_fraction"]
                        )
                    )
            stable = max(deltas) <= max(CONVERGENCE_TV_TOLERANCE, CONVERGENCE_WEIGHT_TOLERANCE)
        if stable and selected_ray_count is None:
            selected_ray_count = ray_count
            selected_records = records
            convergence_pass = True
        previous = records
    if selected_ray_count is None:
        selected_ray_count = CONVERGENCE_RAY_COUNTS[-1]
        selected_records = _run_full_state(full_runtime, designs, selected_ray_count)
    assert selected_records is not None

    reduced: dict[str, Any] = {}
    for name, design in designs.items():
        left = _state_trace_2d(design["tip"], design["states"]["left_contact"])
        right = _state_trace_2d(design["tip"], design["states"]["right_contact"])
        reduced[name] = {
            "separability": _tv(left.density, right.density),
            "left_escaped_fraction": left.escaped_weight / left.launched_weight,
            "right_escaped_fraction": right.escaped_weight / right.launched_weight,
        }
    fields_path = _save_fields(output, selected_records, selected_ray_count)
    full = {
        name: {
            "separability": record["tv"],
            "left_outgoing_surface_weight": record["left_outgoing_surface_weight"],
            "right_outgoing_surface_weight": record["right_outgoing_surface_weight"],
            "left_escaped_fraction": record["left_escaped_fraction"],
            "right_escaped_fraction": record["right_escaped_fraction"],
            "left_absorbed_fraction": record["left_absorbed_fraction"],
            "right_absorbed_fraction": record["right_absorbed_fraction"],
            "left_energy_balance_error": record["left_energy_balance_error"],
            "right_energy_balance_error": record["right_energy_balance_error"],
            "left_timings_seconds": dict(record["left"].timings_seconds),
            "right_timings_seconds": dict(record["right"].timings_seconds),
        }
        for name, record in selected_records.items()
    }
    difference = full["candidate49"]["separability"] - full["nominal"]["separability"]
    if not convergence_pass:
        status = "3D RANKING NOT RESOLVED"
    elif difference > 5.0e-3:
        status = "3D RANKING PRESERVED"
    elif difference < -5.0e-3:
        status = "3D RANKING REVERSED"
    else:
        status = "3D RANKING NOT RESOLVED"
    summary = {
        "status": status,
        "timestamp": _now(),
        "git_revision": _git_revision(),
        "git_status": _git_status(),
        "runtime": full_runtime.metadata,
        "planar_runtime": planar_runtime_metadata,
        "geometry": {"extrusion_depth_mm": DEPTH_MM, "z_bounds_mm": [-5.5, 5.5], "single_source_z_mm": 0.0},
        "mesh": {"level": "medium"},
        "fem_protocol": {"steps": 48, "internal_contact": "three_pairs", "contacts": {key: asdict(value) for key, value in CONTACTS.items()}},
        "optical_material": asdict(designs["nominal"]["tip"].optical),
        "candidate_parameters": asdict(designs["candidate49"]["tip"].parameters),
        "source_convention": {"xy": list(designs["nominal"]["tip"].led_source), "axis": list(designs["nominal"]["tip"].emission_axis), "z_mm": 0.0},
        "branch_threshold": {"minimum_ray_weight": 1.0e-4, "maximum_interactions": 10},
        "planar_gate_pass": planar_pass,
        "planar_consistency": planar_records,
        "reduced_2d": reduced,
        "historical_reduced_2d_reference": HISTORICAL_REDUCED_2D_REFERENCE,
        "reduced_2d_discrepancy_vs_historical": {
            name: reduced[name]["separability"] - HISTORICAL_REDUCED_2D_REFERENCE[name]
            for name in reduced
        },
        "ray_convergence": convergence,
        "ray_convergence_pass": convergence_pass,
        "selected_ray_count": selected_ray_count,
        "full3d": full,
        "fields_artifact": fields_path,
        "fem": {name: design["fem"] for name, design in designs.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    summary = run_validation(arguments.output)
    print(json.dumps({"status": summary["status"], "selected_ray_count": summary.get("selected_ray_count")}, sort_keys=True))
    return 0 if summary["status"] != "3D VALIDATION BLOCKED BY PLANAR INCONSISTENCY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
