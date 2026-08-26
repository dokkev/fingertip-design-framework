"""Validate finalized-model reuse and complete runtime reset reuse."""

from __future__ import annotations

import csv
from importlib.resources import as_file, files
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.optimization.evaluator import FullFingerEvaluation, evaluate_full_finger
from lumo.optimization.objective import (
    _active_surface_triangles,
    _surface_incidence,
    compute_objectives_from_raw,
)
import lumo.simulation.runtime as runtime_module


_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIRECTORY = (
    _ROOT
    / "output"
    / "validation"
    / "production_evaluator_acceleration"
    / "phase2_runtime_reuse"
)
_LOCATIONS_Y_MM = (0.0, 5.5, 22.0)
_REPEAT_COUNT = 3


def _objective_data(result: FullFingerEvaluation) -> dict[str, object]:
    return dict(vars(result))


def _run(
    *,
    sphere_path: Path,
    reuse_finalized_models: bool,
    reuse_runtimes: bool = False,
) -> dict[str, object]:
    build_count = 0
    build_wall_s = 0.0
    original_build = runtime_module.build_fingertip_newton_model

    def timed_build(*args: object, **kwargs: object) -> object:
        nonlocal build_count, build_wall_s
        start_s = perf_counter()
        result = original_build(*args, **kwargs)
        build_wall_s += perf_counter() - start_s
        build_count += 1
        return result

    runtime_module.build_fingertip_newton_model = timed_build
    start_s = perf_counter()
    try:
        evaluation = evaluate_full_finger(
            Fingertip(FingertipParameters()),
            (sphere_path,),
            (20.0,),
            _LOCATIONS_Y_MM,
            reuse_finalized_models=reuse_finalized_models,
            reuse_runtimes=reuse_runtimes,
        )
    finally:
        runtime_module.build_fingertip_newton_model = original_build
    wall_s = perf_counter() - start_s
    contact, observation = compute_objectives_from_raw(_objective_data(evaluation))
    incidence = _surface_incidence(evaluation.surface_triangles)
    supports = np.empty(evaluation.actual_forces_n.shape, dtype=object)
    for scenario_index, force_index in np.ndindex(supports.shape):
        start, count = evaluation.contact_record_offsets[
            scenario_index,
            force_index,
        ]
        supports[scenario_index, force_index] = frozenset(
            _active_surface_triangles(
                evaluation.contact_particle_indices[start : start + count],
                vertex_triangles=incidence[0],
                edge_triangles=incidence[1],
                triangle_ids=incidence[2],
            )
        )
    return {
        "evaluation": evaluation,
        "contact": contact,
        "observation": observation,
        "wall_s": wall_s,
        "model_build_count": build_count,
        "model_build_wall_s": build_wall_s,
        "supports": supports,
    }


def _scalar_fields(run: dict[str, object]) -> dict[str, np.ndarray]:
    evaluation = run["evaluation"]
    contact = run["contact"]
    observation = run["observation"]
    return {
        "checkpoint_time_s": evaluation.checkpoint_times_s,
        "actual_force_n": evaluation.actual_forces_n,
        "indentation_m": evaluation.indentations_m,
        "contact_centroid_W_m": evaluation.contact_centroids_W_m,
        "contact_count": evaluation.indenter_contact_counts.astype(np.float64),
        "minimum_det_f": evaluation.minimum_det_f,
        "q_form": contact.q_form,
        "q_stable": contact.q_stable,
        "q_stiff": contact.q_stiff,
        "q_contact": contact.q_contact,
        "J_contact": np.asarray(contact.J_contact),
        "response_matrix": evaluation.response_matrix,
        "J_obs": np.asarray(observation.J_obs),
        "outside_roi_fraction": evaluation.outside_roi_power_fraction,
    }


def _pairwise_vertex_envelope(runs: list[dict[str, object]]) -> tuple[float, float]:
    rms = 0.0
    maximum = 0.0
    for first, second in combinations(runs, 2):
        delta = (
            first["evaluation"].silicone_vertices_m
            - second["evaluation"].silicone_vertices_m
        )
        rms = max(rms, float(np.sqrt(np.mean(delta**2))))
        maximum = max(maximum, float(np.max(np.abs(delta))))
    return rms, maximum


def _compare(
    fresh_runs: list[dict[str, object]],
    reuse_runs: list[dict[str, object]],
) -> tuple[bool, list[dict[str, object]], dict[str, float]]:
    rows = []
    fresh_fields = [_scalar_fields(run) for run in fresh_runs]
    reuse_fields = [_scalar_fields(run) for run in reuse_runs]
    for field in fresh_fields[0]:
        fresh = np.stack([values[field] for values in fresh_fields])
        reuse = np.stack([values[field] for values in reuse_fields])
        fresh_mean = fresh.mean(axis=0)
        scale = np.maximum(np.abs(fresh_mean), np.finfo(np.float64).tiny)
        rows.append(
            {
                "quantity": field,
                "fresh_mean": float(fresh.mean()),
                "fresh_std": float(fresh.std()),
                "fresh_range": float(fresh.max() - fresh.min()),
                "reuse_mean": float(reuse.mean()),
                "reuse_std": float(reuse.std()),
                "reuse_range": float(reuse.max() - reuse.min()),
                "mean_shift": float(reuse.mean() - fresh.mean()),
                "relative_mean_shift": float(
                    np.max(np.abs(reuse.mean(axis=0) - fresh_mean) / scale)
                ),
            }
        )

    fresh_rms, fresh_max = _pairwise_vertex_envelope(fresh_runs)
    reuse_rms, reuse_max = _pairwise_vertex_envelope(reuse_runs)
    reuse_to_fresh_rms = 0.0
    reuse_to_fresh_max = 0.0
    for reuse in reuse_runs:
        nearest = min(
            (
                float(
                    np.sqrt(
                        np.mean(
                            (
                                reuse["evaluation"].silicone_vertices_m
                                - fresh["evaluation"].silicone_vertices_m
                            )
                            ** 2
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            reuse["evaluation"].silicone_vertices_m
                            - fresh["evaluation"].silicone_vertices_m
                        )
                    )
                ),
            )
            for fresh in fresh_runs
        )
        reuse_to_fresh_rms = max(reuse_to_fresh_rms, nearest[0])
        reuse_to_fresh_max = max(reuse_to_fresh_max, nearest[1])
    geometry_pass = (
        reuse_to_fresh_rms <= 10.0 * fresh_rms
        and reuse_to_fresh_max <= 10.0 * fresh_max
    )
    fresh_patch_floor = 1.0
    for first, second in combinations(fresh_runs, 2):
        for index in np.ndindex(first["supports"].shape):
            first_patch = first["supports"][index]
            second_patch = second["supports"][index]
            fresh_patch_floor = min(
                fresh_patch_floor,
                len(first_patch & second_patch) / len(first_patch | second_patch),
            )
    reuse_patch_iou = 1.0
    for reuse in reuse_runs:
        for index in np.ndindex(reuse["supports"].shape):
            reuse_patch = reuse["supports"][index]
            reuse_patch_iou = min(
                reuse_patch_iou,
                max(
                    len(reuse_patch & fresh["supports"][index])
                    / len(reuse_patch | fresh["supports"][index])
                    for fresh in fresh_runs
                ),
            )
    patch_threshold = min(0.95, fresh_patch_floor)
    patch_pass = reuse_patch_iou >= patch_threshold
    relative_shifts = {row["quantity"]: row["relative_mean_shift"] for row in rows}
    objective_pass = (
        relative_shifts["q_form"] <= 0.05
        and relative_shifts["q_stable"] <= 0.05
        and relative_shifts["q_stiff"] <= 0.05
        and relative_shifts["q_contact"] <= 0.02
        and relative_shifts["J_contact"] <= 0.02
        and relative_shifts["J_obs"] <= 0.02
    )
    safety_pass = all(
        np.all(run["evaluation"].inverted_tet_counts == 0)
        and np.all(run["evaluation"].contact_buffer_overflow == 0)
        and np.all(run["evaluation"].indenter_contact_counts > 0)
        and np.all(np.isfinite(run["evaluation"].silicone_vertices_m))
        for run in reuse_runs
    )
    build_pass = all(run["model_build_count"] == 3 for run in fresh_runs) and all(
        run["model_build_count"] == 1 for run in reuse_runs
    )
    accepted = geometry_pass and patch_pass and objective_pass and safety_pass and build_pass
    diagnostics = {
        "fresh_vertex_rms_m": fresh_rms,
        "reuse_vertex_rms_m": reuse_rms,
        "reuse_to_fresh_nearest_rms_m": reuse_to_fresh_rms,
        "fresh_vertex_max_m": fresh_max,
        "reuse_vertex_max_m": reuse_max,
        "reuse_to_fresh_nearest_max_m": reuse_to_fresh_max,
        "geometry_pass": float(geometry_pass),
        "fresh_patch_iou_floor": fresh_patch_floor,
        "reuse_patch_iou": reuse_patch_iou,
        "patch_threshold": patch_threshold,
        "patch_pass": float(patch_pass),
        "objective_pass": float(objective_pass),
        "safety_pass": float(safety_pass),
        "build_pass": float(build_pass),
    }
    return accepted, rows, diagnostics


def _write_report(
    fresh_runs: list[dict[str, object]],
    reuse_runs: list[dict[str, object]],
    accepted: bool,
    rows: list[dict[str, object]],
    diagnostics: dict[str, float],
) -> None:
    fresh_wall = np.array([run["wall_s"] for run in fresh_runs])
    reuse_wall = np.array([run["wall_s"] for run in reuse_runs])
    fresh_build = np.array([run["model_build_wall_s"] for run in fresh_runs])
    reuse_build = np.array([run["model_build_wall_s"] for run in reuse_runs])
    speedup = float(fresh_wall.mean() / reuse_wall.mean())
    with (_OUTPUT_DIRECTORY / "reuse_envelope.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Newton finalized-model reuse",
        "",
        f"Stage A result: {'PASS' if accepted else 'FAIL'}",
        "",
        "## Dependency audit",
        "",
        "- morphology only: full-finger tetrahedral topology, material arrays, bonded indices, carrier geometry/SDF",
        "- morphology + sphere diameter: indenter geometry, finalized Model, particle/body coloring",
        "- morphology + sphere diameter + Y/history: State A/B, SolverVBD, CollisionPipeline, Contacts, control, wrench/diagnostic buffers, servo state",
        "- Stage A shares only the finalized model/coloring and creates every runtime/history owner fresh.",
        "",
        "## Scientific equivalence",
        "",
        "| quantity | fresh mean | fresh std | reuse mean | reuse std | relative mean shift |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['quantity']} | {row['fresh_mean']:.9g} | "
            f"{row['fresh_std']:.9g} | {row['reuse_mean']:.9g} | "
            f"{row['reuse_std']:.9g} | "
            f"{100.0 * row['relative_mean_shift']:.3f}% |"
        )
    lines.extend(
        (
            "",
            f"- fresh pairwise vertex RMS envelope: {diagnostics['fresh_vertex_rms_m']:.9e} m",
            f"- reuse pairwise vertex RMS: {diagnostics['reuse_vertex_rms_m']:.9e} m",
            f"- reuse-to-nearest-fresh vertex RMS: {diagnostics['reuse_to_fresh_nearest_rms_m']:.9e} m",
            f"- geometry: {'PASS' if diagnostics['geometry_pass'] else 'FAIL'}",
            f"- fresh intrinsic patch-IoU floor: {diagnostics['fresh_patch_iou_floor']:.6f}",
            f"- reuse/nearest-fresh patch IoU: {diagnostics['reuse_patch_iou']:.6f}",
            f"- patch threshold: {diagnostics['patch_threshold']:.6f}",
            f"- patch topology: {'PASS' if diagnostics['patch_pass'] else 'FAIL'}",
            f"- objective inputs: {'PASS' if diagnostics['objective_pass'] else 'FAIL'}",
            f"- inversion/overflow: {'PASS' if diagnostics['safety_pass'] else 'FAIL'}",
            f"- expected 3-to-1 model build count: {'PASS' if diagnostics['build_pass'] else 'FAIL'}",
            "",
            "## Performance for three Y scenarios",
            "",
            f"- fresh model builds/run: 3; mean build wall {fresh_build.mean():.3f} s",
            f"- reused model builds/run: 1; mean build wall {reuse_build.mean():.3f} s",
            f"- fresh evaluation mean: {fresh_wall.mean():.3f} s",
            f"- Stage A evaluation mean: {reuse_wall.mean():.3f} s",
            f"- measured speedup: {speedup:.3f}x",
            "",
            "Stage B runs only after this Stage A gate passes.",
            "",
        )
    )
    (_OUTPUT_DIRECTORY / "report.md").write_text("\n".join(lines))


def _append_stage_b_report(
    fresh_runs: list[dict[str, object]],
    reset_runs: list[dict[str, object]],
    accepted: bool,
    rows: list[dict[str, object]],
    diagnostics: dict[str, float],
) -> None:
    fresh_wall = np.array([run["wall_s"] for run in fresh_runs])
    reset_wall = np.array([run["wall_s"] for run in reset_runs])
    with (_OUTPUT_DIRECTORY / "runtime_reset_envelope.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        (_OUTPUT_DIRECTORY / "report.md").read_text().rstrip(),
        "",
        "## Stage B complete-runtime reset",
        "",
        f"Stage B result: {'PASS' if accepted else 'FAIL'}",
        "",
        "The reset restores both Newton states, solver history, collision/contact "
        "buffers, indenter pose, wrench/diagnostic buffers, servo/checkpoint "
        "buffers, state parity, counters, and graph-visible initial transform. "
        "The finalized model, SolverVBD, CollisionPipeline, Contacts allocations, "
        "and two-tick CUDA graphs remain allocated.",
        "",
        "| quantity | fresh mean | fresh std | reset mean | reset std | relative mean shift |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['quantity']} | {row['fresh_mean']:.9g} | "
            f"{row['fresh_std']:.9g} | {row['reuse_mean']:.9g} | "
            f"{row['reuse_std']:.9g} | "
            f"{100.0 * row['relative_mean_shift']:.3f}% |"
        )
    lines.extend(
        (
            "",
            f"- fresh pairwise vertex RMS envelope: {diagnostics['fresh_vertex_rms_m']:.9e} m",
            f"- reset pairwise vertex RMS: {diagnostics['reuse_vertex_rms_m']:.9e} m",
            f"- reset-to-nearest-fresh vertex RMS: {diagnostics['reuse_to_fresh_nearest_rms_m']:.9e} m",
            f"- geometry: {'PASS' if diagnostics['geometry_pass'] else 'FAIL'}",
            f"- reset/nearest-fresh patch IoU: {diagnostics['reuse_patch_iou']:.6f}",
            f"- patch topology: {'PASS' if diagnostics['patch_pass'] else 'FAIL'}",
            f"- objective inputs: {'PASS' if diagnostics['objective_pass'] else 'FAIL'}",
            f"- inversion/overflow: {'PASS' if diagnostics['safety_pass'] else 'FAIL'}",
            f"- expected 3-to-1 model build count: {'PASS' if diagnostics['build_pass'] else 'FAIL'}",
            f"- fresh evaluation mean: {fresh_wall.mean():.3f} s",
            f"- Stage B evaluation mean: {reset_wall.mean():.3f} s",
            f"- measured Stage B speedup: {fresh_wall.mean() / reset_wall.mean():.3f}x",
            "",
        )
    )
    (_OUTPUT_DIRECTORY / "report.md").write_text("\n".join(lines))


def main() -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    resource = files("lumo.assets.objects.urdf").joinpath("sphere_20mm.urdf")
    fresh_runs = []
    reuse_runs = []
    with as_file(resource) as sphere_path:
        for reuse, output in ((False, fresh_runs), (True, reuse_runs)):
            label = "stage_a" if reuse else "fresh"
            for repeat in range(_REPEAT_COUNT):
                print(f"{label} repeat {repeat + 1}/{_REPEAT_COUNT}", flush=True)
                output.append(
                    _run(
                        sphere_path=sphere_path,
                        reuse_finalized_models=reuse,
                    )
                )
    accepted, rows, diagnostics = _compare(fresh_runs, reuse_runs)
    _write_report(fresh_runs, reuse_runs, accepted, rows, diagnostics)
    if not accepted:
        print((_OUTPUT_DIRECTORY / "report.md").read_text())
        raise RuntimeError("Stage A finalized-model reuse gate failed")

    reset_runs = []
    with as_file(resource) as sphere_path:
        for repeat in range(_REPEAT_COUNT):
            print(f"stage_b repeat {repeat + 1}/{_REPEAT_COUNT}", flush=True)
            reset_runs.append(
                _run(
                    sphere_path=sphere_path,
                    reuse_finalized_models=True,
                    reuse_runtimes=True,
                )
            )
    reset_accepted, reset_rows, reset_diagnostics = _compare(
        fresh_runs,
        reset_runs,
    )
    _append_stage_b_report(
        fresh_runs,
        reset_runs,
        reset_accepted,
        reset_rows,
        reset_diagnostics,
    )
    print((_OUTPUT_DIRECTORY / "report.md").read_text())
    if not reset_accepted:
        raise RuntimeError("Stage B complete-runtime reset gate failed")


if __name__ == "__main__":
    main()
