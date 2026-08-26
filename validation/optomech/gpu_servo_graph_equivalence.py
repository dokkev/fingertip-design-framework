"""Validate the GPU-resident force-servo graph against direct Newton runs."""

from __future__ import annotations

import csv
from importlib.resources import as_file, files
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.optimization.evaluator import evaluate_full_finger
from lumo.optimization.objective import (
    _active_surface_triangles,
    _mean_contact_normal,
    _surface_incidence,
    _triangle_areas,
    compute_objectives_from_raw,
)
from lumo.simulation import REFERENCE_DWELL_LOADING


_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIRECTORY = (
    _ROOT
    / "output"
    / "validation"
    / "production_evaluator_acceleration"
    / "phase1_cuda_graph"
)
_REPEAT_COUNT = 5
_FORCE_TARGETS_N = np.array((5.0, 10.0, 15.0, 20.0))
_CONTACT_Y_MM = (11.0, 22.0)
_EMITTED_POWER = 5.0


def _supports_and_patch_metrics(evaluation):
    triangles = np.asarray(evaluation.surface_triangles, dtype=np.int32)
    incidence = _surface_incidence(triangles)
    supports = np.empty(evaluation.actual_forces_n.shape, dtype=object)
    patch_area_m2 = np.empty(evaluation.actual_forces_n.shape, dtype=np.float64)
    mean_normals = np.empty((*evaluation.actual_forces_n.shape, 3))
    for scenario_index in range(len(evaluation.scenario_names)):
        for force_index in range(len(evaluation.force_targets_n)):
            start, count = evaluation.contact_record_offsets[
                scenario_index,
                force_index,
            ]
            indices = evaluation.contact_particle_indices[start : start + count]
            support = frozenset(
                _active_surface_triangles(
                    indices,
                    vertex_triangles=incidence[0],
                    edge_triangles=incidence[1],
                    triangle_ids=incidence[2],
                )
            )
            supports[scenario_index, force_index] = support
            patch_area_m2[scenario_index, force_index] = float(
                _triangle_areas(
                    evaluation.silicone_vertices_m[
                        scenario_index,
                        force_index,
                    ],
                    triangles,
                )[list(support)].sum()
            )
            mean_normals[scenario_index, force_index] = _mean_contact_normal(
                evaluation.contact_normals_W[start : start + count]
            )
    return supports, patch_area_m2, mean_normals


def _run(use_cuda_graph: bool, sphere_path: Path) -> dict[str, object]:
    fingertip = Fingertip(FingertipParameters())
    start_s = perf_counter()
    evaluation = evaluate_full_finger(
        fingertip,
        (sphere_path,),
        (10.0,),
        _CONTACT_Y_MM,
        force_targets_n=_FORCE_TARGETS_N,
        settle_duration_s=5.0,
        force_tolerance_fraction=0.1,
        initial_clearance_m=1.0e-3,
        approach_speed_m_s=5.0e-3,
        max_sim_time_s=60.0,
        loading_mode=REFERENCE_DWELL_LOADING,
        use_cuda_graph=use_cuda_graph,
    )
    wp.synchronize()
    total_wall_s = perf_counter() - start_s
    contact, observation = compute_objectives_from_raw(vars(evaluation))
    supports, patch_area_m2, mean_normals = _supports_and_patch_metrics(evaluation)
    normalized_response = (
        evaluation.combined_response_matrix
        - evaluation.combined_no_contact_response
    ) / _EMITTED_POWER
    return {
        "evaluation": evaluation,
        "contact": contact,
        "observation": observation,
        "supports": supports,
        "patch_area_m2": patch_area_m2,
        "mean_normals": mean_normals,
        "normalized_response": normalized_response,
        "total_wall_s": total_wall_s,
    }


def _group_array(runs, getter):
    return np.stack([np.asarray(getter(run), dtype=np.float64) for run in runs])


def _relative_mean_shift(direct: np.ndarray, graph: np.ndarray) -> float:
    direct_mean = direct.mean(axis=0)
    scale = np.maximum(np.abs(direct_mean), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(graph.mean(axis=0) - direct_mean) / scale))


def _stat_row(name: str, direct: np.ndarray, graph: np.ndarray) -> dict[str, object]:
    return {
        "quantity": name,
        "direct_mean": float(direct.mean()),
        "direct_std": float(direct.std()),
        "direct_min": float(direct.min()),
        "direct_max": float(direct.max()),
        "graph_mean": float(graph.mean()),
        "graph_std": float(graph.std()),
        "graph_min": float(graph.min()),
        "graph_max": float(graph.max()),
        "mean_shift": float(graph.mean() - direct.mean()),
        "maximum_relative_mean_shift": _relative_mean_shift(direct, graph),
    }


def _pairwise_geometry_variability(runs) -> tuple[float, float]:
    maximum_rms = 0.0
    maximum_absolute = 0.0
    for first, second in combinations(runs, 2):
        delta = (
            first["evaluation"].silicone_vertices_m
            - second["evaluation"].silicone_vertices_m
        )
        maximum_rms = max(maximum_rms, float(np.sqrt(np.mean(delta**2))))
        maximum_absolute = max(maximum_absolute, float(np.max(np.abs(delta))))
    return maximum_rms, maximum_absolute


def _graph_to_direct_geometry(graph_runs, direct_runs) -> tuple[float, float]:
    maximum_nearest_rms = 0.0
    maximum_nearest_absolute = 0.0
    for graph in graph_runs:
        distances = []
        for direct in direct_runs:
            delta = (
                graph["evaluation"].silicone_vertices_m
                - direct["evaluation"].silicone_vertices_m
            )
            distances.append(
                (
                    float(np.sqrt(np.mean(delta**2))),
                    float(np.max(np.abs(delta))),
                )
            )
        nearest = min(distances)
        maximum_nearest_rms = max(maximum_nearest_rms, nearest[0])
        maximum_nearest_absolute = max(maximum_nearest_absolute, nearest[1])
    return maximum_nearest_rms, maximum_nearest_absolute


def _support_iou(first: frozenset[int], second: frozenset[int]) -> float:
    return len(first & second) / len(first | second)


def _direct_patch_floor(runs) -> float:
    floor = 1.0
    for first, second in combinations(runs, 2):
        for index in np.ndindex(first["supports"].shape):
            floor = min(
                floor,
                _support_iou(first["supports"][index], second["supports"][index]),
            )
    return floor


def _graph_patch_iou(graph_runs, direct_runs) -> float:
    minimum = 1.0
    for graph in graph_runs:
        for index in np.ndindex(graph["supports"].shape):
            best_direct = max(
                _support_iou(graph["supports"][index], direct["supports"][index])
                for direct in direct_runs
            )
            minimum = min(minimum, best_direct)
    return minimum


def _evaluate(direct_runs, graph_runs):
    quantities = {
        "checkpoint_step": lambda run: run["evaluation"].checkpoint_steps,
        "checkpoint_time_s": lambda run: run["evaluation"].checkpoint_times_s,
        "force_n": lambda run: run["evaluation"].actual_forces_n,
        "indentation_m": lambda run: run["evaluation"].indentations_m,
        "patch_area_m2": lambda run: run["patch_area_m2"],
        "contact_count": lambda run: run["evaluation"].indenter_contact_counts,
        "minimum_det_f": lambda run: run["evaluation"].minimum_det_f,
        "q_form": lambda run: run["contact"].q_form,
        "q_stable": lambda run: run["contact"].q_stable,
        "q_stiff": lambda run: run["contact"].q_stiff,
        "q_contact": lambda run: run["contact"].q_contact,
        "J_obs": lambda run: np.asarray(run["observation"].J_obs),
    }
    grouped = {
        name: (_group_array(direct_runs, getter), _group_array(graph_runs, getter))
        for name, getter in quantities.items()
    }
    rows = [_stat_row(name, *values) for name, values in grouped.items()]

    direct_geometry = _pairwise_geometry_variability(direct_runs)
    graph_geometry = _pairwise_geometry_variability(graph_runs)
    graph_to_direct_geometry = _graph_to_direct_geometry(graph_runs, direct_runs)
    geometry_pass = (
        graph_to_direct_geometry[0] <= 10.0 * direct_geometry[0]
        and graph_to_direct_geometry[1] <= 10.0 * direct_geometry[1]
    )

    direct_patch_floor = _direct_patch_floor(direct_runs)
    graph_patch_iou = _graph_patch_iou(graph_runs, direct_runs)
    patch_threshold = min(0.95, direct_patch_floor)
    patch_pass = graph_patch_iou >= patch_threshold

    q_relative = {
        name: _relative_mean_shift(*grouped[name])
        for name in ("q_form", "q_stable", "q_stiff", "q_contact")
    }
    q_pass = (
        q_relative["q_form"] <= 0.05
        and q_relative["q_stable"] <= 0.05
        and q_relative["q_stiff"] <= 0.05
        and q_relative["q_contact"] <= 0.02
    )

    direct_response = _group_array(
        direct_runs,
        lambda run: run["normalized_response"],
    )
    graph_response = _group_array(
        graph_runs,
        lambda run: run["normalized_response"],
    )
    response_mean_delta = graph_response.mean(axis=0) - direct_response.mean(axis=0)
    response_rms = float(np.sqrt(np.mean(response_mean_delta**2)))
    response_max = float(np.max(np.abs(response_mean_delta)))
    observation_relative = _relative_mean_shift(*grouped["J_obs"])
    optical_pass = observation_relative <= 0.02

    direct_visible = _group_array(
        direct_runs,
        lambda run: run["evaluation"].visible_side_power.sum(axis=2),
    )
    graph_visible = _group_array(
        graph_runs,
        lambda run: run["evaluation"].visible_side_power.sum(axis=2),
    )
    visible_delta = graph_visible.mean(axis=0) - direct_visible.mean(axis=0)
    direct_outside = _group_array(
        direct_runs,
        lambda run: run["evaluation"].outside_roi_power.sum(axis=2),
    )
    graph_outside = _group_array(
        graph_runs,
        lambda run: run["evaluation"].outside_roi_power.sum(axis=2),
    )
    outside_delta = graph_outside.mean(axis=0) - direct_outside.mean(axis=0)

    graph_evaluations = [run["evaluation"] for run in graph_runs]
    finite_pass = all(
        np.all(np.isfinite(evaluation.silicone_vertices_m))
        and np.all(np.isfinite(evaluation.actual_forces_n))
        and np.all(np.isfinite(evaluation.response_matrix))
        for evaluation in graph_evaluations
    )
    safety_pass = all(
        np.count_nonzero(evaluation.contact_buffer_overflow) == 0
        and np.count_nonzero(evaluation.inverted_tet_counts) == 0
        and np.all(evaluation.indenter_contact_counts > 0)
        for evaluation in graph_evaluations
    )
    force_band_pass = all(
        np.all(
            np.abs(evaluation.actual_forces_n - _FORCE_TARGETS_N[None, :])
            <= 0.1 * _FORCE_TARGETS_N[None, :]
        )
        for evaluation in graph_evaluations
    )
    ordering_pass = all(
        np.all(np.diff(evaluation.checkpoint_steps, axis=1) > 0)
        and np.array_equal(
            evaluation.checkpoint_steps,
            np.rint(evaluation.checkpoint_times_s * 100.0).astype(np.int64),
        )
        for evaluation in graph_evaluations
    )
    accepted = all(
        (
            geometry_pass,
            patch_pass,
            q_pass,
            optical_pass,
            finite_pass,
            safety_pass,
            force_band_pass,
            ordering_pass,
        )
    )
    diagnostics = {
        "geometry_pass": geometry_pass,
        "direct_geometry_pair_rms_m": direct_geometry[0],
        "direct_geometry_pair_max_m": direct_geometry[1],
        "graph_geometry_pair_rms_m": graph_geometry[0],
        "graph_geometry_pair_max_m": graph_geometry[1],
        "graph_to_direct_nearest_rms_m": graph_to_direct_geometry[0],
        "graph_to_direct_nearest_max_m": graph_to_direct_geometry[1],
        "patch_pass": patch_pass,
        "direct_patch_iou_floor": direct_patch_floor,
        "graph_to_direct_patch_iou": graph_patch_iou,
        "patch_iou_threshold": patch_threshold,
        "q_pass": q_pass,
        **{f"{name}_relative_shift": value for name, value in q_relative.items()},
        "optical_pass": optical_pass,
        "J_obs_relative_shift": observation_relative,
        "response_rms_difference": response_rms,
        "response_max_bin_difference": response_max,
        "visible_power_max_difference": float(np.max(np.abs(visible_delta))),
        "outside_roi_max_difference": float(np.max(np.abs(outside_delta))),
        "finite_pass": finite_pass,
        "safety_pass": safety_pass,
        "force_band_pass": force_band_pass,
        "ordering_pass": ordering_pass,
    }
    return accepted, rows, diagnostics


def _write_outputs(direct_runs, graph_runs, accepted, rows, diagnostics):
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (_OUTPUT_DIRECTORY / "gpu_servo_scientific_statistics.csv").open(
        "w",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    direct_wall = np.array([run["total_wall_s"] for run in direct_runs])
    graph_wall = np.array([run["total_wall_s"] for run in graph_runs])
    direct_scenario_wall = np.array(
        [run["evaluation"].scenario_runtime_s.sum() for run in direct_runs]
    )
    graph_scenario_wall = np.array(
        [run["evaluation"].scenario_runtime_s.sum() for run in graph_runs]
    )
    speedup = float(direct_scenario_wall.mean() / graph_scenario_wall.mean())
    graph_replays = _group_array(
        graph_runs,
        lambda run: run["evaluation"].graph_replay_counts,
    )
    graph_interventions = _group_array(
        graph_runs,
        lambda run: run["evaluation"].force_servo_host_intervention_counts,
    )
    graph_syncs = _group_array(
        graph_runs,
        lambda run: run["evaluation"].force_servo_host_sync_counts,
    )
    ticks_per_intervention = _group_array(
        graph_runs,
        lambda run: run[
            "evaluation"
        ].force_servo_average_ticks_per_host_intervention,
    )
    direct_simulated_s = np.array(
        [run["evaluation"].checkpoint_times_s[:, -1].sum() for run in direct_runs]
    )
    graph_simulated_s = np.array(
        [run["evaluation"].checkpoint_times_s[:, -1].sum() for run in graph_runs]
    )

    stats = {row["quantity"]: row for row in rows}
    table = [
        ("checkpoint timing [s]", stats["checkpoint_time_s"], "diagnostic/pass"),
        ("force [N]", stats["force_n"], "force band"),
        ("indentation [m]", stats["indentation_m"], "diagnostic"),
        ("patch area [m2]", stats["patch_area_m2"], "q_form / topology"),
        ("q_form", stats["q_form"], "<=5%"),
        ("q_stable", stats["q_stable"], "<=5%"),
        ("q_stiff", stats["q_stiff"], "<=5%"),
        ("q_contact", stats["q_contact"], "<=2%"),
        ("J_obs", stats["J_obs"], "<=2%"),
    ]
    lines = [
        "# Phase 1-B GPU-resident force-servo graph",
        "",
        f"Result: {'PASS' if accepted else 'FAIL'}",
        "",
        "Bitwise equality is not required because the direct Newton full-surface contact backend exhibits intrinsic GPU atomic-ordering nondeterminism.",
        "",
        "Production finite-area 1.8 x 1.6 mm emission and hard 11-bin +X observation were used. The representative set is the known difficult 10 mm sphere, Y=+11/+22 mm pair at 5/10/15/20 N with the unchanged 5 s dwell.",
        "",
        "## Scientific comparison",
        "",
        "| Quantity | Direct mean | GPU graph mean | Mean shift | Relative shift | Acceptance |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, row, rule in table:
        lines.append(
            f"| {name} | {row['direct_mean']:.9g} | {row['graph_mean']:.9g} | "
            f"{row['mean_shift']:.9g} | "
            f"{100.0 * row['maximum_relative_mean_shift']:.3f}% | {rule} |"
        )
    lines.extend(
        (
            "",
            "## Geometry, contact, and safety",
            "",
            f"- direct/direct maximum vertex RMS: {diagnostics['direct_geometry_pair_rms_m']:.9e} m",
            f"- graph/graph maximum vertex RMS: {diagnostics['graph_geometry_pair_rms_m']:.9e} m",
            f"- graph/nearest-direct maximum vertex RMS: {diagnostics['graph_to_direct_nearest_rms_m']:.9e} m",
            f"- direct/direct maximum vertex absolute difference: {diagnostics['direct_geometry_pair_max_m']:.9e} m",
            f"- graph/nearest-direct maximum vertex absolute difference: {diagnostics['graph_to_direct_nearest_max_m']:.9e} m",
            f"- geometry same-order gate: {'PASS' if diagnostics['geometry_pass'] else 'FAIL'}",
            f"- direct intrinsic patch-IoU floor: {diagnostics['direct_patch_iou_floor']:.6f}",
            f"- graph/nearest-direct patch IoU: {diagnostics['graph_to_direct_patch_iou']:.6f}",
            f"- patch-IoU threshold: {diagnostics['patch_iou_threshold']:.6f}",
            f"- patch topology: {'PASS' if diagnostics['patch_pass'] else 'FAIL'}",
            f"- finite state: {'PASS' if diagnostics['finite_pass'] else 'FAIL'}",
            f"- inversion/contact-buffer/contact existence: {'PASS' if diagnostics['safety_pass'] else 'FAIL'}",
            f"- force band and target order: {'PASS' if diagnostics['force_band_pass'] and diagnostics['ordering_pass'] else 'FAIL'}",
            "",
            "## Optical state",
            "",
            f"- normalized 11D response RMS difference: {diagnostics['response_rms_difference']:.9e}",
            f"- normalized 11D maximum-bin difference: {diagnostics['response_max_bin_difference']:.9e}",
            f"- total +X visible-power maximum difference: {diagnostics['visible_power_max_difference']:.9e}",
            f"- outside-ROI maximum difference: {diagnostics['outside_roi_max_difference']:.9e}",
            f"- difficult-pair separation relative shift: {100.0 * diagnostics['J_obs_relative_shift']:.3f}%",
            f"- optical <=2% gate: {'PASS' if diagnostics['optical_pass'] else 'FAIL'}",
            "",
            "## Performance",
            "",
            f"- direct total evaluation mean: {direct_wall.mean():.3f} s",
            f"- graph total evaluation mean: {graph_wall.mean():.3f} s",
            f"- direct scenario-loop mean: {direct_scenario_wall.mean():.3f} s",
            f"- graph scenario-loop mean: {graph_scenario_wall.mean():.3f} s",
            f"- scenario-loop speedup: {speedup:.3f}x",
            f"- direct simulated-second/wall-second: {(direct_simulated_s / direct_scenario_wall).mean():.3f}",
            f"- graph simulated-second/wall-second: {(graph_simulated_s / graph_scenario_wall).mean():.3f}",
            f"- graph replay count per two-scenario evaluation: {graph_replays.sum(axis=1).mean():.1f}",
            f"- force-control host interventions: {graph_interventions.sum(axis=1).mean():.1f}",
            f"- force-control host synchronizations: {graph_syncs.sum(axis=1).mean():.1f}",
            f"- average physics ticks per host intervention: {ticks_per_intervention.mean():.3f}",
            "",
            "## Design-conclusion interpretation",
            "",
            "No suitable existing full-finger morphology comparison uses this exact finite-area production contract, so a morphology-ranking preservation ratio cannot be manufactured. The gate therefore reports mechanical components and the known limiting optical separation directly.",
            "",
        )
    )
    (_OUTPUT_DIRECTORY / "gpu_servo_graph_equivalence.md").write_text(
        "\n".join(lines)
    )
    np.savez_compressed(
        _OUTPUT_DIRECTORY / "gpu_servo_graph_repeats.npz",
        direct_forces=np.stack(
            [run["evaluation"].actual_forces_n for run in direct_runs]
        ),
        graph_forces=np.stack(
            [run["evaluation"].actual_forces_n for run in graph_runs]
        ),
        direct_indentations_m=np.stack(
            [run["evaluation"].indentations_m for run in direct_runs]
        ),
        graph_indentations_m=np.stack(
            [run["evaluation"].indentations_m for run in graph_runs]
        ),
        direct_J_obs=np.array([run["observation"].J_obs for run in direct_runs]),
        graph_J_obs=np.array([run["observation"].J_obs for run in graph_runs]),
        direct_wall_s=direct_wall,
        graph_wall_s=graph_wall,
        accepted=np.asarray(accepted),
    )


def main() -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    resource = files("lumo.assets.objects.urdf").joinpath("sphere_10mm.urdf")
    direct_runs = []
    graph_runs = []
    with as_file(resource) as sphere_path:
        for use_graph, output in ((False, direct_runs), (True, graph_runs)):
            backend = "GPU graph" if use_graph else "direct"
            for repeat_index in range(_REPEAT_COUNT):
                print(
                    f"{backend} repeat {repeat_index + 1}/{_REPEAT_COUNT}",
                    flush=True,
                )
                output.append(_run(use_graph, sphere_path))

    accepted, rows, diagnostics = _evaluate(direct_runs, graph_runs)
    _write_outputs(direct_runs, graph_runs, accepted, rows, diagnostics)
    report = _OUTPUT_DIRECTORY / "gpu_servo_graph_equivalence.md"
    print(report.read_text())
    if not accepted:
        raise RuntimeError("GPU-resident force-servo scientific gate failed")


if __name__ == "__main__":
    main()
