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
from optics.metrics import field_difference
from optics.transport3d import Transport3DSettings, trace_3d
from optics.transport3d.optix_backend import create_runtime
from optimization.scenarios import ContactScenario


OUTPUT = Path("output/validation/optics/transport3d/internal_bridge_convergence")
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
CONVERGENCE_RAY_COUNTS = (16384, 65536, 262144)
CONVERGENCE_TV_TOLERANCE = 1.0e-2
VALIDATION_MAX_PERIODIC_WRAPS = 512
FIELD_RESOLUTIONS = {
    "low": {"x_bins": 192, "y_bins": 192, "z_bins": 16},
    "current": {"x_bins": 240, "y_bins": 240, "z_bins": 32},
    "high": {"x_bins": 288, "y_bins": 288, "z_bins": 48},
}
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


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _overlap_fractions(target_edges: np.ndarray, source_edges: np.ndarray) -> np.ndarray:
    left = np.maximum(target_edges[:-1, None], source_edges[None, :-1])
    right = np.minimum(target_edges[1:, None], source_edges[None, 1:])
    return np.maximum(0.0, right - left) / np.diff(source_edges)[None, :]


def _common_edges(first: np.ndarray, second: np.ndarray, count: int) -> np.ndarray:
    lower = max(float(first[0]), float(second[0]))
    upper = min(float(first[-1]), float(second[-1]))
    if not upper > lower:
        raise ValueError("fields have no common comparison support")
    return np.linspace(lower, upper, count)


def _resample_mass(
    field: np.ndarray,
    source_x_edges: np.ndarray,
    source_y_edges: np.ndarray,
    target_x_edges: np.ndarray,
    target_y_edges: np.ndarray,
) -> np.ndarray:
    x_fraction = _overlap_fractions(target_x_edges, source_x_edges)
    y_fraction = _overlap_fractions(target_y_edges, source_y_edges)
    return y_fraction @ np.asarray(field, dtype=float) @ x_fraction.T


def _tv_with_edges(
    first: np.ndarray,
    first_x_edges: np.ndarray,
    first_y_edges: np.ndarray,
    second: np.ndarray,
    second_x_edges: np.ndarray,
    second_y_edges: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    target_x = _common_edges(
        first_x_edges,
        second_x_edges,
        max(len(first_x_edges), len(second_x_edges)),
    )
    target_y = _common_edges(
        first_y_edges,
        second_y_edges,
        max(len(first_y_edges), len(second_y_edges)),
    )
    first_mass = _resample_mass(
        first,
        first_x_edges,
        first_y_edges,
        target_x,
        target_y,
    )
    second_mass = _resample_mass(
        second,
        second_x_edges,
        second_y_edges,
        target_x,
        target_y,
    )
    first_total = float(np.sum(first_mass))
    second_total = float(np.sum(second_mass))
    if first_total <= 0.0 or second_total <= 0.0:
        raise ValueError("fields must have positive mass on common support")
    tv = 0.5 * float(
        np.sum(np.abs(first_mass / first_total - second_mass / second_total))
    )
    return min(1.0, max(0.0, tv)), {
        "x_min_mm": float(target_x[0]),
        "x_max_mm": float(target_x[-1]),
        "y_min_mm": float(target_y[0]),
        "y_max_mm": float(target_y[-1]),
        "x_bins": len(target_x) - 1,
        "y_bins": len(target_y) - 1,
        "resampling": "mass-preserving overlap fractions on common intersection support",
    }


def _internal_path_tv(first: Any, second: Any) -> tuple[float, dict[str, Any]]:
    if (
        first.internal_z_integrated_path_density is None
        or second.internal_z_integrated_path_density is None
    ):
        raise RuntimeError("internal path field was not retained")
    return _tv_with_edges(
        first.internal_z_integrated_path_density,
        first.internal_path_x_edges_mm,
        first.internal_path_y_edges_mm,
        second.internal_z_integrated_path_density,
        second.internal_path_x_edges_mm,
        second.internal_path_y_edges_mm,
    )


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
    retain_internal_path_field: bool = False,
    field_resolution: Mapping[str, int] | None = None,
    minimum_ray_weight: float = 1.0e-4,
) -> Any:
    resolution = field_resolution or FIELD_RESOLUTIONS["current"]
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
            maximum_periodic_wraps=VALIDATION_MAX_PERIODIC_WRAPS,
            terminate_on_periodic_wrap_limit=True,
            terminate_on_no_event=True,
            extrusion_depth_mm=DEPTH_MM,
            retain_projected_segments=retain_projected_segments,
            retain_internal_path_field=retain_internal_path_field,
            internal_grid_width=resolution["x_bins"],
            internal_grid_height=resolution["y_bins"],
            internal_z_bins=resolution["z_bins"],
        ),
    )


def _solve_contact(
    tip: Fingertip,
    mesh: Any,
    scenario: ContactScenario,
    *,
    cache_path: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            displacement = np.asarray(data["displacement"], dtype=float)
        loaded_mesh = mesh.pad.deformed(
            displacement,
            metadata={"condition": "cached_contact", "scenario": asdict(scenario)},
        )
        return loaded_mesh, {
            "converged": True,
            "cached": True,
            "wall_time_seconds": 0.0,
            "scenario": asdict(scenario),
            "reaction_force_n": None,
        }
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
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            displacement=np.asarray(result.deformed_mesh.displacement, dtype=float),
        )
    return result.deformed_mesh, record


def _planar_hit_distance_stats(reduced: Any, planar: Any) -> dict[str, Any]:
    reduced_records = sorted(
        (
            int(segment.ray_index),
            int(segment.interaction_index),
            float(
                np.linalg.norm(
                    np.asarray(segment.end, dtype=float)
                    - np.asarray(segment.start, dtype=float)
                )
            ),
        )
        for segment in reduced.segments
    )
    optix_lengths = planar.retained_segment_lengths_mm
    optix_primary = planar.retained_segment_primary_ray_indices
    optix_interactions = planar.retained_segment_interaction_counts
    if optix_lengths is None or optix_primary is None or optix_interactions is None:
        raise RuntimeError("planar trace did not retain segment distances")
    optix_records = sorted(
        (
            int(primary),
            int(interaction),
            float(length),
        )
        for primary, interaction, length in zip(
            np.asarray(optix_primary, dtype=np.int64),
            np.asarray(optix_interactions, dtype=np.int64),
            np.asarray(optix_lengths, dtype=float),
        )
    )
    compared = min(len(reduced_records), len(optix_records))
    if compared:
        reduced_keys = [record[:2] for record in reduced_records[:compared]]
        optix_keys = [record[:2] for record in optix_records[:compared]]
        errors = np.abs(
            np.asarray([record[2] for record in reduced_records[:compared]])
            - np.asarray([record[2] for record in optix_records[:compared]])
        )
    else:
        reduced_keys = []
        optix_keys = []
        errors = np.asarray([], dtype=float)
    mismatch_count = abs(len(reduced_records) - len(optix_records)) + sum(
        first != second for first, second in zip(reduced_keys, optix_keys)
    )
    return {
        "maximum_absolute_hit_distance_error_mm": float(np.max(errors)) if len(errors) else 0.0,
        "mean_absolute_hit_distance_error_mm": float(np.mean(errors)) if len(errors) else 0.0,
        "p95_absolute_hit_distance_error_mm": float(np.percentile(errors, 95.0)) if len(errors) else 0.0,
        "compared_hits": int(compared),
        "mismatch_count": int(mismatch_count),
    }


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
            hit_distance_stats = _planar_hit_distance_stats(reduced, planar)
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
                "hit_distance_stats": hit_distance_stats,
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
    field_resolution: Mapping[str, int],
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, design in designs.items():
        started = time.perf_counter()
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
                retain_internal_path_field=True,
                field_resolution=field_resolution,
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
        path_tv, comparison_grid = _internal_path_tv(left, right)
        records[name] = {
            "left": left,
            "right": right,
            "j3d_path": path_tv,
            "j3d_path_comparison_grid": comparison_grid,
            "j3d_surface": _tv(left.outgoing_surface_field, right.outgoing_surface_field),
            "left_outgoing_surface_weight": left.outgoing_surface_weight,
            "right_outgoing_surface_weight": right.outgoing_surface_weight,
            "left_escaped_fraction": left.escaped_weight / left.launched_weight,
            "right_escaped_fraction": right.escaped_weight / right.launched_weight,
            "left_absorbed_fraction": left.absorbed_weight / left.launched_weight,
            "right_absorbed_fraction": right.absorbed_weight / right.launched_weight,
            "left_energy_balance_error": left.energy_balance_error,
            "right_energy_balance_error": right.energy_balance_error,
            "left_processed_segment_count": left.geometry_metadata["processed_segment_count"],
            "right_processed_segment_count": right.geometry_metadata["processed_segment_count"],
            "left_timings_seconds": dict(left.timings_seconds),
            "right_timings_seconds": dict(right.timings_seconds),
            "runtime_seconds": time.perf_counter() - started,
        }
    return records


def _save_fields(
    output: Path,
    reduced_results: Mapping[str, Any],
    final_records: Mapping[str, Mapping[str, Any]],
    ray_count: int,
) -> str:
    path = output / "fields.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for name, reduced in reduced_results.items():
        arrays[f"{name}_2d_left_field"] = reduced["left"].density
        arrays[f"{name}_2d_left_x_edges"] = reduced["left"].x_edges
        arrays[f"{name}_2d_left_y_edges"] = reduced["left"].y_edges
        arrays[f"{name}_2d_right_field"] = reduced["right"].density
        arrays[f"{name}_2d_right_x_edges"] = reduced["right"].x_edges
        arrays[f"{name}_2d_right_y_edges"] = reduced["right"].y_edges
    for resolution_name, records in final_records.items():
        for name, record in records.items():
            resolution_prefix = f"{name}_{resolution_name}"
            for state_name in ("left", "right"):
                result = record[state_name]
                prefix = f"{resolution_prefix}_{state_name}"
                if result.internal_weighted_path_density_3d is None:
                    raise RuntimeError("selected result is missing the P3 field")
                arrays[f"{prefix}_p3"] = result.internal_weighted_path_density_3d
                arrays[f"{prefix}_p3_x_edges"] = result.internal_path_x_edges_mm
                arrays[f"{prefix}_p3_y_edges"] = result.internal_path_y_edges_mm
                arrays[f"{prefix}_p3_z_edges"] = result.internal_path_z_edges_mm
                arrays[f"{prefix}_p3_xy"] = result.internal_z_integrated_path_density
                arrays[f"{prefix}_surface_field"] = result.outgoing_surface_field
                arrays[f"{prefix}_surface_u_edges"] = result.surface_u_edges
                arrays[f"{prefix}_surface_z_edges"] = result.surface_z_edges
                arrays[f"{prefix}_escape_positions"] = result.escape_positions_mm
                arrays[f"{prefix}_escape_directions"] = result.escape_directions
                arrays[f"{prefix}_escape_normals"] = result.escape_surface_normals
                arrays[f"{prefix}_escape_u"] = result.escape_surface_u
                arrays[f"{prefix}_escape_z"] = result.escape_surface_z
                arrays[f"{prefix}_escape_weights"] = result.escape_weights
                arrays[f"{prefix}_escape_primary_ray_indices"] = result.escape_primary_ray_indices
                arrays[f"{prefix}_escape_interaction_counts"] = result.escape_interaction_counts
                arrays[f"{prefix}_escape_path_lengths_mm"] = result.escape_path_lengths_mm
    arrays["artifact_ray_count"] = np.asarray([ray_count], dtype=np.int64)
    arrays["artifact_field_levels"] = np.asarray(tuple(final_records), dtype="U16")
    np.savez_compressed(path, **arrays)
    return str(path)


def _full_record_summary(
    record: Mapping[str, Any],
    field_resolution: Mapping[str, int],
) -> dict[str, Any]:
    left = record["left"]
    right = record["right"]
    return {
        "observable_types": {
            "j3d_path": "3D internal weighted optical path-length distribution contact-state separability",
            "j3d_surface": "camera-independent outgoing surface transport contact-state separability",
        },
        "j3d_path": float(record["j3d_path"]),
        "j3d_surface": float(record["j3d_surface"]),
        "j3d_path_comparison_grid": record["j3d_path_comparison_grid"],
        "field_resolution": dict(field_resolution),
        "left_total_launched_weight": left.launched_weight,
        "right_total_launched_weight": right.launched_weight,
        "left_escaped_weight": left.escaped_weight,
        "right_escaped_weight": right.escaped_weight,
        "left_absorbed_weight": left.absorbed_weight,
        "right_absorbed_weight": right.absorbed_weight,
        "left_terminated_weight": left.terminated_weight,
        "right_terminated_weight": right.terminated_weight,
        "left_energy_balance_residual": left.energy_balance_error,
        "right_energy_balance_residual": right.energy_balance_error,
        "left_processed_segment_count": record["left_processed_segment_count"],
        "right_processed_segment_count": record["right_processed_segment_count"],
        "left_periodic_wrap_termination": _plain_json(left.geometry_metadata["periodic_wrap_termination"]),
        "right_periodic_wrap_termination": _plain_json(right.geometry_metadata["periodic_wrap_termination"]),
        "left_no_event_termination": _plain_json(left.geometry_metadata["no_event_termination"]),
        "right_no_event_termination": _plain_json(right.geometry_metadata["no_event_termination"]),
        "left_runtime_seconds": record["left_timings_seconds"],
        "right_runtime_seconds": record["right_timings_seconds"],
        "design_runtime_seconds": record["runtime_seconds"],
    }


def _ordering(value: float, tolerance: float = 1.0e-12) -> str:
    if value > tolerance:
        return "candidate49_gt_nominal"
    if value < -tolerance:
        return "candidate49_lt_nominal"
    return "tie"


def _convergence_diagnostics(convergence: Mapping[str, Any]) -> dict[str, Any]:
    final_ray = str(CONVERGENCE_RAY_COUNTS[-1])
    previous_ray = str(CONVERGENCE_RAY_COUNTS[-2])
    ray_deltas: dict[str, Any] = {}
    criterion_values: list[float] = []
    for resolution_name in ("current", "high"):
        ray_deltas[resolution_name] = {}
        for morphology in ("nominal", "candidate49"):
            final = convergence[final_ray][resolution_name][morphology]
            previous = convergence[previous_ray][resolution_name][morphology]
            path_delta = abs(final["j3d_path"] - previous["j3d_path"])
            surface_delta = abs(final["j3d_surface"] - previous["j3d_surface"])
            ray_deltas[resolution_name][morphology] = {
                "absolute_j3d_path_delta": path_delta,
                "absolute_j3d_surface_delta": surface_delta,
                "criterion_pass": (
                    path_delta <= CONVERGENCE_TV_TOLERANCE
                    and surface_delta <= CONVERGENCE_TV_TOLERANCE
                ),
            }
            criterion_values.extend((path_delta, surface_delta))

    resolution_deltas: dict[str, Any] = {}
    for morphology in ("nominal", "candidate49"):
        current = convergence[final_ray]["current"][morphology]
        high = convergence[final_ray]["high"][morphology]
        path_delta = abs(current["j3d_path"] - high["j3d_path"])
        surface_delta = abs(current["j3d_surface"] - high["j3d_surface"])
        resolution_deltas[morphology] = {
            "absolute_current_to_high_j3d_path_delta": path_delta,
            "absolute_current_to_high_j3d_surface_delta": surface_delta,
            "criterion_pass": (
                path_delta <= CONVERGENCE_TV_TOLERANCE
                and surface_delta <= CONVERGENCE_TV_TOLERANCE
            ),
        }
        criterion_values.extend((path_delta, surface_delta))

    ordering_by_resolution: dict[str, Any] = {}
    for resolution_name in FIELD_RESOLUTIONS:
        nominal = convergence[final_ray][resolution_name]["nominal"]
        candidate = convergence[final_ray][resolution_name]["candidate49"]
        ordering_by_resolution[resolution_name] = {
            "j3d_path": _ordering(candidate["j3d_path"] - nominal["j3d_path"]),
            "j3d_surface": _ordering(candidate["j3d_surface"] - nominal["j3d_surface"]),
        }
    path_orderings = {item["j3d_path"] for item in ordering_by_resolution.values()}
    surface_orderings = {item["j3d_surface"] for item in ordering_by_resolution.values()}
    ordering_stable = len(path_orderings) == 1 and len(surface_orderings) == 1
    return {
        "criterion": {
            "successive_ray_tv_absolute_tolerance": CONVERGENCE_TV_TOLERANCE,
            "requires_current_and_high_resolution": True,
            "requires_current_to_high_resolution_stability": True,
            "requires_morphology_ordering_stability": True,
        },
        "successive_ray_deltas": ray_deltas,
        "current_to_high_resolution_deltas": resolution_deltas,
        "final_ray_ordering_by_resolution": ordering_by_resolution,
        "morphology_ordering_stable": ordering_stable,
        "maximum_criterion_delta": max(criterion_values) if criterion_values else 0.0,
        "pass": (
            all(
                ray_deltas[level][morphology]["criterion_pass"]
                for level in ("current", "high")
                for morphology in ("nominal", "candidate49")
            )
            and all(item["criterion_pass"] for item in resolution_deltas.values())
            and ordering_stable
        ),
    }


def run_validation(output: Path = OUTPUT) -> dict[str, Any]:
    """Run the focused 2D/internal-3D/outgoing-3D comparison."""
    output.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime()
    design_parameters = {
        "nominal": FingertipParameters(),
        "candidate49": FingertipParameters(**CANDIDATE49),
    }
    designs: dict[str, dict[str, Any]] = {}
    for name, parameters in design_parameters.items():
        tip = Fingertip(parameters)
        mesh = tip.mesh(mesh_settings_for_level("medium"))
        designs[name] = {"tip": tip, "mesh": mesh, "states": {}}
        left, left_record = _solve_contact(
            tip,
            mesh,
            CONTACTS["left_contact"],
            cache_path=output / "fea_states" / f"{name}_left_contact.npz",
        )
        designs[name]["states"]["left_contact"] = left
        designs[name].setdefault("fem", {})["left_contact"] = left_record

    planar_records, planar_pass = _planar_gate(runtime, designs, output)
    if not planar_pass:
        summary = {
            "status": "3D RANKING NOT RESOLVED",
            "blocking_reason": "planar correctness gate failed",
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
        right, right_record = _solve_contact(
            tip,
            mesh,
            CONTACTS["right_contact"],
            cache_path=output / "fea_states" / f"{name}_right_contact.npz",
        )
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
            "status": "3D RANKING NOT RESOLVED",
            "blocking_reason": "planar correctness gate failed",
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
    full_runtime = create_runtime()

    reduced_results: dict[str, Any] = {}
    reduced: dict[str, Any] = {}
    for name, design in designs.items():
        left = _state_trace_2d(design["tip"], design["states"]["left_contact"])
        right = _state_trace_2d(design["tip"], design["states"]["right_contact"])
        reduced_results[name] = {"left": left, "right": right}
        reduced[name] = {
            "observable_type": "2D internal weighted optical path-length distribution contact-state separability",
            "j2d": field_difference(left, right),
            "left_escaped_fraction": left.escaped_weight / left.launched_weight,
            "right_escaped_fraction": right.escaped_weight / right.launched_weight,
        }

    convergence: dict[str, Any] = {}
    final_records: dict[str, Mapping[str, Any]] = {}
    for ray_count in CONVERGENCE_RAY_COUNTS:
        ray_records: dict[str, Any] = {}
        for resolution_name, resolution in FIELD_RESOLUTIONS.items():
            records = _run_full_state(
                full_runtime,
                designs,
                ray_count,
                resolution,
            )
            ray_records[resolution_name] = {
                name: _full_record_summary(record, resolution)
                for name, record in records.items()
            }
            if ray_count == CONVERGENCE_RAY_COUNTS[-1]:
                final_records[resolution_name] = records
        convergence[str(ray_count)] = ray_records

    convergence_diagnostics = _convergence_diagnostics(convergence)
    convergence_pass = bool(convergence_diagnostics["pass"])
    selected_ray_count = CONVERGENCE_RAY_COUNTS[-1]
    fields_path = _save_fields(output, reduced_results, final_records, selected_ray_count)

    final_current = convergence[str(selected_ray_count)]["current"]
    j2d_ordering = _ordering(reduced["candidate49"]["j2d"] - reduced["nominal"]["j2d"])
    j3d_path_ordering = _ordering(
        final_current["candidate49"]["j3d_path"]
        - final_current["nominal"]["j3d_path"]
    )
    j3d_surface_ordering = _ordering(
        final_current["candidate49"]["j3d_surface"]
        - final_current["nominal"]["j3d_surface"]
    )
    if not convergence_pass:
        status = "3D RANKING NOT RESOLVED"
    elif j2d_ordering == j3d_path_ordering == j3d_surface_ordering:
        status = "PRELIMINARY 2D SURROGATE CONSISTENCY"
    elif j2d_ordering == j3d_path_ordering:
        status = "OBSERVABLE MISMATCH"
    else:
        status = "DIMENSIONAL REDUCTION MISMATCH"
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
        "branch_threshold": {
            "minimum_ray_weight_fraction": 1.0e-4,
            "maximum_interactions": 10,
            "convention": "primary and first-generation branches are retained; cutoff applies when interaction_count > 1",
            "matches_2d": True,
        },
        "periodic_wrap_guard": {
            "maximum_periodic_wraps": VALIDATION_MAX_PERIODIC_WRAPS,
            "terminate_on_limit": True,
            "interpretation": "only pathological branches reaching the safety bound are booked as terminated; ordinary periodic crossings remain unchanged",
        },
        "no_event_guard": {
            "terminate_on_no_event": True,
            "interpretation": "only branches with neither a physical OptiX hit nor a valid periodic event are booked as terminated",
        },
        "planar_gate_pass": planar_pass,
        "planar_consistency": planar_records,
        "reduced_2d": reduced,
        "historical_reduced_2d_reference": HISTORICAL_REDUCED_2D_REFERENCE,
        "reduced_2d_discrepancy_vs_historical": {
            name: reduced[name]["j2d"] - HISTORICAL_REDUCED_2D_REFERENCE[name]
            for name in reduced
        },
        "ray_convergence": convergence,
        "ray_convergence_pass": convergence_pass,
        "convergence_diagnostics": convergence_diagnostics,
        "selected_ray_count": selected_ray_count,
        "final_ordering": {
            "j2d": j2d_ordering,
            "j3d_path": j3d_path_ordering,
            "j3d_surface": j3d_surface_ordering,
        },
        "observable_definitions": {
            "j2d": "2D internal weighted optical path-length distribution TV between left/right loaded states",
            "j3d_path": "TV between z-integrated 3D internal weighted path-length distributions",
            "j3d_surface": "TV between camera-independent outgoing surface transport fields",
        },
        "internal_path_field_definition": {
            "field": "P3(x,y,z), raw weighted path length accumulated for transport segments in the accessible optical domain",
            "segment_medium_scope": "air and silicone, matching the existing 2D internal field",
            "voxel_value_definition": "weighted path length per voxel before TV normalization",
            "z_integral": "sum of z-bin path masses to form P3_xy; no extra z-width multiplier",
            "line_sampling": "deterministic segment midpoint sampling, at most 32 samples per segment",
            "comparison_grid": "mass-preserving overlap resampling on the common x/y intersection support",
            "raw_total": "sum of persisted P3 voxels in fields.npz",
            "source_z_mm": 0.0,
            "extrusion_depth_mm": DEPTH_MM,
            "field_levels": FIELD_RESOLUTIONS,
        },
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
