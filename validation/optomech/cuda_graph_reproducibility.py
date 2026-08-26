"""Compare partial CUDA graphs with Newton's intrinsic direct variability."""

from __future__ import annotations

import csv
from importlib.resources import as_file, files
from itertools import combinations
from pathlib import Path

import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import make_fingertip_5led_mesh
from lumo.optimization.objective import (
    _active_surface_triangles,
    _mean_contact_normal,
    _surface_incidence,
    _triangle_areas,
    compute_contact_objective,
)

import cuda_graph_equivalence as strict_validation


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


def _derive_run(
    profile: dict[str, object],
    *,
    reference_vertices_m: np.ndarray,
    surface_triangles: np.ndarray,
) -> dict[str, object]:
    incidence = _surface_incidence(surface_triangles)
    supports = []
    patch_areas = []
    patch_centroids = []
    patch_normals = []
    contact_indices = []
    contact_normals = []
    offsets = np.empty((1, 4, 2), dtype=np.int64)
    record_start = 0
    checkpoints = profile["checkpoints"]
    for force_index, checkpoint in enumerate(checkpoints):
        indices = np.asarray(checkpoint["contact_indices"], dtype=np.int32)
        barycentric = np.asarray(
            checkpoint["contact_barycentric"],
            dtype=np.float64,
        )
        vertices = np.asarray(checkpoint["silicone_vertices_m"], dtype=np.float64)
        normals = np.asarray(checkpoint["contact_normals_W"], dtype=np.float64)
        support = _active_surface_triangles(
            indices,
            vertex_triangles=incidence[0],
            edge_triangles=incidence[1],
            triangle_ids=incidence[2],
        )
        supports.append(frozenset(support))
        patch_areas.append(
            float(_triangle_areas(vertices, surface_triangles)[list(support)].sum())
        )
        positions = np.empty((len(indices), 3), dtype=np.float64)
        for record_index, (record, weights) in enumerate(
            zip(indices, barycentric, strict=True)
        ):
            present = record >= 0
            positions[record_index] = np.sum(
                vertices[record[present]] * weights[present, None],
                axis=0,
            )
        patch_centroids.append(positions.mean(axis=0))
        patch_normals.append(_mean_contact_normal(normals))
        offsets[0, force_index] = (record_start, len(indices))
        record_start += len(indices)
        contact_indices.append(indices)
        contact_normals.append(normals)

    forces = np.array(
        [float(checkpoint["actual_force_n"]) for checkpoint in checkpoints]
    )[None, :]
    indentations = np.array(
        [float(checkpoint["indentation_m"]) for checkpoint in checkpoints]
    )[None, :]
    vertices = np.stack(
        [checkpoint["silicone_vertices_m"] for checkpoint in checkpoints]
    )[None, ...]
    contact = compute_contact_objective(
        reference_vertices_m=reference_vertices_m,
        surface_triangles=surface_triangles,
        scenario_names=("sphere_20mm_y+0mm",),
        sphere_diameters_mm=np.array((20.0,)),
        force_targets_n=_FORCE_TARGETS_N,
        actual_forces_n=forces,
        indentations_m=indentations,
        contact_record_offsets=offsets,
        contact_particle_indices=np.concatenate(contact_indices),
        contact_normals_W=np.concatenate(contact_normals),
        silicone_vertices_m=vertices,
    )
    scalar_fields = {
        "checkpoint_step": np.array(
            [int(checkpoint["step_index"]) for checkpoint in checkpoints],
            dtype=np.float64,
        ),
        "checkpoint_time_s": np.array(
            [float(checkpoint["simulation_time_s"]) for checkpoint in checkpoints]
        ),
        "actual_force_n": forces[0],
        "indentation_m": indentations[0],
        "patch_area_m2": np.asarray(patch_areas),
        "patch_centroid_W_m": np.asarray(patch_centroids),
        "patch_normal_W": np.asarray(patch_normals),
        "contact_count": np.array(
            [int(checkpoint["contact_count"]) for checkpoint in checkpoints],
            dtype=np.float64,
        ),
        "minimum_det_f": np.array(
            [float(checkpoint["minimum_det_f"]) for checkpoint in checkpoints]
        ),
        "q_form": contact.q_form,
        "q_stable": contact.q_stable,
        "q_stiff": contact.q_stiff,
        "q_contact": contact.q_contact,
    }
    return {
        "profile": profile,
        "scalars": scalar_fields,
        "supports": tuple(supports),
        "vertices": vertices[0],
        "contact": contact,
    }


def _pairwise_geometry_distances(runs: list[dict[str, object]]) -> tuple[float, float]:
    maximum_rms = 0.0
    maximum_absolute = 0.0
    for first, second in combinations(runs, 2):
        delta = np.asarray(first["vertices"]) - np.asarray(second["vertices"])
        maximum_rms = max(maximum_rms, float(np.sqrt(np.mean(delta**2))))
        maximum_absolute = max(maximum_absolute, float(np.max(np.abs(delta))))
    return maximum_rms, maximum_absolute


def _minimum_support_iou(runs: list[dict[str, object]]) -> float:
    minimum = 1.0
    for first, second in combinations(runs, 2):
        for first_patch, second_patch in zip(
            first["supports"],
            second["supports"],
            strict=True,
        ):
            minimum = min(
                minimum,
                len(first_patch & second_patch) / len(first_patch | second_patch),
            )
    return minimum


def _evaluate_envelope(
    direct_runs: list[dict[str, object]],
    graph_runs: list[dict[str, object]],
) -> tuple[bool, list[dict[str, object]], dict[str, float]]:
    rows = []
    accepted = True
    for field in direct_runs[0]["scalars"]:
        direct = np.stack([run["scalars"][field] for run in direct_runs])
        graph = np.stack([run["scalars"][field] for run in graph_runs])
        direct_min = direct.min(axis=0)
        direct_max = direct.max(axis=0)
        graph_min = graph.min(axis=0)
        graph_max = graph.max(axis=0)
        inside = bool(np.all((graph >= direct_min) & (graph <= direct_max)))
        comparable = bool(np.all(graph_max - graph_min <= direct_max - direct_min))
        mean_inside = bool(
            np.all((graph.mean(axis=0) >= direct_min) & (graph.mean(axis=0) <= direct_max))
        )
        field_pass = inside and comparable and mean_inside
        accepted &= field_pass
        rows.append(
            {
                "quantity": field,
                "direct_min": float(direct.min()),
                "direct_max": float(direct.max()),
                "direct_range_max": float(np.max(direct_max - direct_min)),
                "graph_min": float(graph.min()),
                "graph_max": float(graph.max()),
                "graph_range_max": float(np.max(graph_max - graph_min)),
                "all_graph_inside_direct_envelope": inside,
                "graph_variation_not_larger": comparable,
                "graph_mean_inside_direct_envelope": mean_inside,
                "pass": field_pass,
            }
        )

    direct_geometry_rms, direct_geometry_max = _pairwise_geometry_distances(
        direct_runs
    )
    graph_geometry_rms, graph_geometry_max = _pairwise_geometry_distances(graph_runs)
    graph_to_direct_rms = 0.0
    graph_to_direct_max = 0.0
    for graph in graph_runs:
        distances = []
        for direct in direct_runs:
            delta = np.asarray(graph["vertices"]) - np.asarray(direct["vertices"])
            distances.append(
                (float(np.sqrt(np.mean(delta**2))), float(np.max(np.abs(delta))))
            )
        nearest = min(distances)
        graph_to_direct_rms = max(graph_to_direct_rms, nearest[0])
        graph_to_direct_max = max(graph_to_direct_max, nearest[1])
    geometry_pass = (
        graph_geometry_rms <= direct_geometry_rms
        and graph_geometry_max <= direct_geometry_max
        and graph_to_direct_rms <= direct_geometry_rms
        and graph_to_direct_max <= direct_geometry_max
    )
    accepted &= geometry_pass

    direct_support_iou = _minimum_support_iou(direct_runs)
    graph_support_iou = _minimum_support_iou(graph_runs)
    graph_to_direct_iou = 1.0
    for graph in graph_runs:
        for force_index, graph_patch in enumerate(graph["supports"]):
            graph_to_direct_iou = min(
                graph_to_direct_iou,
                max(
                    len(graph_patch & direct["supports"][force_index])
                    / len(graph_patch | direct["supports"][force_index])
                    for direct in direct_runs
                ),
            )
    support_pass = (
        graph_support_iou >= direct_support_iou
        and graph_to_direct_iou >= direct_support_iou
    )
    accepted &= support_pass

    safety_pass = all(
        int(checkpoint["inversion_count"]) == 0
        and int(checkpoint["buffer_overflow"]) == 0
        for run in graph_runs
        for checkpoint in run["profile"]["checkpoints"]
    )
    accepted &= safety_pass
    diagnostics = {
        "direct_geometry_pair_rms_m": direct_geometry_rms,
        "graph_geometry_pair_rms_m": graph_geometry_rms,
        "graph_to_direct_nearest_rms_m": graph_to_direct_rms,
        "direct_geometry_pair_max_m": direct_geometry_max,
        "graph_geometry_pair_max_m": graph_geometry_max,
        "graph_to_direct_nearest_max_m": graph_to_direct_max,
        "direct_min_patch_iou": direct_support_iou,
        "graph_min_patch_iou": graph_support_iou,
        "graph_to_direct_min_nearest_patch_iou": graph_to_direct_iou,
        "geometry_pass": float(geometry_pass),
        "support_pass": float(support_pass),
        "safety_pass": float(safety_pass),
    }
    return accepted, rows, diagnostics


def _write_outputs(
    direct_runs: list[dict[str, object]],
    graph_runs: list[dict[str, object]],
    accepted: bool,
    rows: list[dict[str, object]],
    diagnostics: dict[str, float],
) -> None:
    with (_OUTPUT_DIRECTORY / "numerical_envelope.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    direct_wall = np.array(
        [run["profile"]["loop_wall_s"] for run in direct_runs],
        dtype=np.float64,
    )
    graph_wall = np.array(
        [run["profile"]["loop_wall_s"] for run in graph_runs],
        dtype=np.float64,
    )
    speedup = float(direct_wall.mean() / graph_wall.mean())
    direct_steps_s = np.array(
        [run["profile"]["step_count"] / run["profile"]["loop_wall_s"] for run in direct_runs]
    )
    graph_steps_s = np.array(
        [run["profile"]["step_count"] / run["profile"]["loop_wall_s"] for run in graph_runs]
    )
    lines = [
        "# Partial CUDA graph intrinsic-reproducibility gate",
        "",
        f"Result: {'PASS' if accepted else 'FAIL'}",
        "",
        "The numerical envelope is measured from five fresh direct runs; no arbitrary floating-point tolerance was introduced.",
        "Raw atomic contact-record order is excluded from the gate. Material-surface support, patch geometry, normalized mean normal, and objective components are compared instead.",
        "",
        "## Envelope results",
        "",
        "| quantity | direct range | graph range | graph inside | graph variation <= direct | graph mean inside | pass |",
        "|---|---:|---:|:---:|:---:|:---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['quantity']} | {row['direct_range_max']:.9g} | "
            f"{row['graph_range_max']:.9g} | {row['all_graph_inside_direct_envelope']} | "
            f"{row['graph_variation_not_larger']} | "
            f"{row['graph_mean_inside_direct_envelope']} | {row['pass']} |"
        )
    lines.extend(
        (
            "",
            "## Geometry and contact support",
            "",
            f"- direct pairwise vertex RMS envelope: {diagnostics['direct_geometry_pair_rms_m']:.9e} m",
            f"- graph pairwise vertex RMS: {diagnostics['graph_geometry_pair_rms_m']:.9e} m",
            f"- worst graph-to-nearest-direct vertex RMS: {diagnostics['graph_to_direct_nearest_rms_m']:.9e} m",
            f"- direct pairwise vertex max envelope: {diagnostics['direct_geometry_pair_max_m']:.9e} m",
            f"- graph pairwise vertex max: {diagnostics['graph_geometry_pair_max_m']:.9e} m",
            f"- worst graph-to-nearest-direct vertex max: {diagnostics['graph_to_direct_nearest_max_m']:.9e} m",
            f"- minimum direct/direct patch IoU: {diagnostics['direct_min_patch_iou']:.9f}",
            f"- minimum graph/graph patch IoU: {diagnostics['graph_min_patch_iou']:.9f}",
            f"- minimum graph/nearest-direct patch IoU: {diagnostics['graph_to_direct_min_nearest_patch_iou']:.9f}",
            f"- geometry envelope: {'PASS' if diagnostics['geometry_pass'] else 'FAIL'}",
            f"- patch-support envelope: {'PASS' if diagnostics['support_pass'] else 'FAIL'}",
            f"- inversion/overflow safety: {'PASS' if diagnostics['safety_pass'] else 'FAIL'}",
            "",
            "## Performance",
            "",
            f"- direct loop mean: {direct_wall.mean():.3f} s",
            f"- graph loop mean: {graph_wall.mean():.3f} s",
            f"- direct mean throughput: {direct_steps_s.mean():.3f} steps/s",
            f"- graph mean throughput: {graph_steps_s.mean():.3f} steps/s",
            f"- measured loop speedup: {speedup:.3f}x",
            "",
            "The speedup is production-acceptable only when the numerical gate passes.",
            "",
        )
    )
    (_OUTPUT_DIRECTORY / "cuda_graph_reproducibility.md").write_text(
        "\n".join(lines)
    )
    np.savez_compressed(
        _OUTPUT_DIRECTORY / "cuda_graph_repeats.npz",
        direct_checkpoint_steps=np.stack(
            [run["scalars"]["checkpoint_step"] for run in direct_runs]
        ),
        graph_checkpoint_steps=np.stack(
            [run["scalars"]["checkpoint_step"] for run in graph_runs]
        ),
        direct_forces=np.stack(
            [run["scalars"]["actual_force_n"] for run in direct_runs]
        ),
        graph_forces=np.stack(
            [run["scalars"]["actual_force_n"] for run in graph_runs]
        ),
        direct_indentations_m=np.stack(
            [run["scalars"]["indentation_m"] for run in direct_runs]
        ),
        graph_indentations_m=np.stack(
            [run["scalars"]["indentation_m"] for run in graph_runs]
        ),
        direct_q_contact=np.array(
            [run["contact"].J_contact for run in direct_runs]
        ),
        graph_q_contact=np.array(
            [run["contact"].J_contact for run in graph_runs]
        ),
        direct_loop_wall_s=direct_wall,
        graph_loop_wall_s=graph_wall,
        accepted=np.asarray(accepted),
    )


def main() -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    strict_validation._OUTPUT_DIRECTORY = _OUTPUT_DIRECTORY
    fingertip = Fingertip(FingertipParameters())
    mesh = make_fingertip_5led_mesh(fingertip, element_size_mm=1.0)
    reference_vertices_m = np.asarray(mesh.silicone.vertices, dtype=np.float32)
    surface_triangles = np.asarray(
        mesh.silicone.surface_tri_indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    tet_indices = np.asarray(mesh.silicone.tet_indices, dtype=np.int32).reshape(-1, 4)
    reference_six_volumes_m3 = strict_validation._six_tet_volumes(
        reference_vertices_m,
        tet_indices,
    )
    resource = files("lumo.assets.objects.urdf").joinpath("sphere_20mm.urdf")
    direct_runs = []
    graph_runs = []
    with as_file(resource) as sphere_path:
        for use_graph, output in ((False, direct_runs), (True, graph_runs)):
            mode = "graph" if use_graph else "direct"
            for repeat in range(_REPEAT_COUNT):
                print(f"{mode} repeat {repeat + 1}/{_REPEAT_COUNT}", flush=True)
                profile = strict_validation._run_scenario(
                    use_cuda_graph=use_graph,
                    fingertip=fingertip,
                    fingertip_mesh=mesh,
                    sphere_urdf_path=sphere_path,
                    tet_indices=tet_indices,
                    reference_six_volumes_m3=reference_six_volumes_m3,
                )
                output.append(
                    _derive_run(
                        profile,
                        reference_vertices_m=reference_vertices_m,
                        surface_triangles=surface_triangles,
                    )
                )

    accepted, rows, diagnostics = _evaluate_envelope(direct_runs, graph_runs)
    _write_outputs(direct_runs, graph_runs, accepted, rows, diagnostics)
    print((_OUTPUT_DIRECTORY / "cuda_graph_reproducibility.md").read_text())
    if not accepted:
        raise RuntimeError("partial CUDA graph reproducibility-envelope gate failed")


if __name__ == "__main__":
    main()
