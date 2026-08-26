"""Select the shortest sentinel-validated force-band dwell."""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path
from time import perf_counter

import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.optimization.evaluator import evaluate_full_finger
from lumo.optimization.objective import (
    _active_surface_triangles,
    _mean_contact_normal,
    _surface_incidence,
    compute_contact_objective,
    compute_objectives_from_raw,
)
from lumo.simulation import REFERENCE_DWELL_LOADING


_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIRECTORY = (
    _ROOT
    / "output"
    / "validation"
    / "production_evaluator_acceleration"
    / "phase3_dwell"
)
_GROUPS = (
    ("contact_limiter", "sphere_20mm.urdf", 20.0, (22.0,)),
    ("observation_pair", "sphere_10mm.urdf", 10.0, (11.0, 22.0)),
    ("interior", "sphere_15mm.urdf", 15.0, (0.0,)),
)


def _saved_evaluation(path: Path) -> dict[str, object]:
    with np.load(path) as stored:
        return {key: stored[key] for key in stored.files}


def _save_evaluation(path: Path, evaluation: object) -> dict[str, object]:
    data = {}
    for key, value in vars(evaluation).items():
        if value is None:
            continue
        if isinstance(value, (np.ndarray, tuple, str, float, int)):
            data[key] = np.asarray(value)
    np.savez_compressed(path, **data)
    return data


def _run_dwell(
    dwell_s: float,
    *,
    output_directory: Path = _OUTPUT_DIRECTORY,
) -> dict[str, object]:
    label = str(dwell_s).replace(".", "p")
    result: dict[str, object] = {"dwell_s": dwell_s, "groups": {}}
    total_start_s = perf_counter()
    resource_root = files("lumo.assets.objects.urdf")
    for name, filename, diameter_mm, locations_y_mm in _GROUPS:
        output_path = output_directory / f"dwell_{label}_{name}.npz"
        if output_path.is_file():
            print(f"dwell {dwell_s:g} s / {name}: loading", flush=True)
            data = _saved_evaluation(output_path)
        else:
            print(f"dwell {dwell_s:g} s / {name}: running", flush=True)
            resource = resource_root.joinpath(filename)
            with as_file(resource) as sphere_path:
                evaluation = evaluate_full_finger(
                    Fingertip(FingertipParameters()),
                    (sphere_path,),
                    (diameter_mm,),
                    locations_y_mm,
                    settle_duration_s=dwell_s,
                    loading_mode=REFERENCE_DWELL_LOADING,
                )
            data = _save_evaluation(output_path, evaluation)
        result["groups"][name] = data
    result["wall_s"] = perf_counter() - total_start_s
    return result


def _contact(data: dict[str, object]):
    return compute_contact_objective(
        reference_vertices_m=np.asarray(data["reference_vertices_m"]),
        surface_triangles=np.asarray(data["surface_triangles"]),
        scenario_names=tuple(str(value) for value in data["scenario_names"]),
        sphere_diameters_mm=np.asarray(data["sphere_diameters_mm"]),
        force_targets_n=np.asarray(data["force_targets_n"]),
        actual_forces_n=np.asarray(data["actual_forces_n"]),
        indentations_m=np.asarray(data["indentations_m"]),
        contact_record_offsets=np.asarray(data["contact_record_offsets"]),
        contact_particle_indices=np.asarray(data["contact_particle_indices"]),
        contact_normals_W=np.asarray(data["contact_normals_W"]),
        silicone_vertices_m=np.asarray(data["silicone_vertices_m"]),
    )


def _summary(run: dict[str, object]) -> dict[str, object]:
    contacts = {
        name: _contact(data) for name, data in run["groups"].items()
    }
    scenario_names = tuple(
        scenario
        for contact in contacts.values()
        for scenario in contact.scenario_names
    )
    q_form = np.concatenate([contact.q_form for contact in contacts.values()])
    q_stable = np.concatenate(
        [contact.q_stable for contact in contacts.values()]
    )
    q_stiff = np.concatenate([contact.q_stiff for contact in contacts.values()])
    q_contact = np.concatenate(
        [contact.q_contact for contact in contacts.values()]
    )
    limiting_contact_index = int(np.argmin(q_contact))
    _, observation = compute_objectives_from_raw(
        run["groups"]["observation_pair"]
    )
    return {
        "scenario_names": scenario_names,
        "q_form": q_form,
        "q_stable": q_stable,
        "q_stiff": q_stiff,
        "q_contact": q_contact,
        "J_contact": float(q_contact[limiting_contact_index]),
        "limiting_contact": scenario_names[limiting_contact_index],
        "observation": observation,
    }


def _relative(candidate: np.ndarray | float, reference: np.ndarray | float) -> float:
    candidate_values = np.asarray(candidate, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    scale = np.maximum(np.abs(reference_values), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(candidate_values - reference_values) / scale))


def _patch_and_normal_diagnostics(
    candidate: dict[str, object],
    reference: dict[str, object],
) -> tuple[float, float]:
    triangles = np.asarray(reference["surface_triangles"], dtype=np.int32)
    incidence = _surface_incidence(triangles)
    minimum_iou = 1.0
    minimum_normal_score = 1.0
    for scenario_index, force_index in np.ndindex(
        np.asarray(reference["actual_forces_n"]).shape
    ):
        patches = []
        normals = []
        for data in (reference, candidate):
            start, count = np.asarray(data["contact_record_offsets"])[
                scenario_index, force_index
            ]
            indices = np.asarray(data["contact_particle_indices"])[
                start : start + count
            ]
            patches.append(
                _active_surface_triangles(
                    indices,
                    vertex_triangles=incidence[0],
                    edge_triangles=incidence[1],
                    triangle_ids=incidence[2],
                )
            )
            normals.append(
                _mean_contact_normal(
                    np.asarray(data["contact_normals_W"])[start : start + count]
                )
            )
        minimum_iou = min(
            minimum_iou,
            len(patches[0] & patches[1]) / len(patches[0] | patches[1]),
        )
        minimum_normal_score = min(
            minimum_normal_score,
            0.5 * (1.0 + float(np.dot(normals[0], normals[1]))),
        )
    return minimum_iou, minimum_normal_score


def _compare(candidate: dict[str, object], reference: dict[str, object]) -> dict[str, object]:
    candidate_summary = _summary(candidate)
    reference_summary = _summary(reference)
    force_abs_n = 0.0
    indentation_abs_m = 0.0
    centroid_abs_m = 0.0
    speed_abs_m_s = 0.0
    minimum_patch_iou = 1.0
    minimum_normal_score = 1.0
    safety_pass = True
    for name in reference["groups"]:
        candidate_group = candidate["groups"][name]
        reference_group = reference["groups"][name]
        force_abs_n = max(
            force_abs_n,
            float(
                np.max(
                    np.abs(
                        candidate_group["actual_forces_n"]
                        - reference_group["actual_forces_n"]
                    )
                )
            ),
        )
        indentation_abs_m = max(
            indentation_abs_m,
            float(
                np.max(
                    np.abs(
                        candidate_group["indentations_m"]
                        - reference_group["indentations_m"]
                    )
                )
            ),
        )
        centroid_abs_m = max(
            centroid_abs_m,
            float(
                np.max(
                    np.linalg.norm(
                        candidate_group["contact_centroids_W_m"]
                        - reference_group["contact_centroids_W_m"],
                        axis=-1,
                    )
                )
            ),
        )
        speed_abs_m_s = max(
            speed_abs_m_s,
            float(
                np.max(
                    np.abs(
                        candidate_group["maximum_particle_speeds_m_s"]
                        - reference_group["maximum_particle_speeds_m_s"]
                    )
                )
            ),
        )
        patch_iou, normal_score = _patch_and_normal_diagnostics(
            candidate_group,
            reference_group,
        )
        minimum_patch_iou = min(minimum_patch_iou, patch_iou)
        minimum_normal_score = min(minimum_normal_score, normal_score)
        safety_pass = safety_pass and bool(
            np.all(np.isfinite(candidate_group["silicone_vertices_m"]))
            and np.all(candidate_group["inverted_tet_counts"] == 0)
            and np.all(candidate_group["contact_buffer_overflow"] == 0)
            and np.all(candidate_group["indenter_contact_counts"] > 0)
        )
    candidate_observation = candidate_summary["observation"]
    reference_observation = reference_summary["observation"]
    limiting_cases_pass = (
        candidate_summary["limiting_contact"]
        == reference_summary["limiting_contact"]
        and candidate_observation.limiting_force_n
        == reference_observation.limiting_force_n
        and candidate_observation.limiting_contact_y_pair_mm
        == reference_observation.limiting_contact_y_pair_mm
    )
    metrics = {
        "force_abs_n": force_abs_n,
        "indentation_abs_m": indentation_abs_m,
        "centroid_abs_m": centroid_abs_m,
        "maximum_speed_abs_m_s": speed_abs_m_s,
        "minimum_patch_iou": minimum_patch_iou,
        "minimum_normal_score": minimum_normal_score,
        "q_form_relative": _relative(
            candidate_summary["q_form"], reference_summary["q_form"]
        ),
        "q_stable_relative": _relative(
            candidate_summary["q_stable"], reference_summary["q_stable"]
        ),
        "q_stiff_relative": _relative(
            candidate_summary["q_stiff"], reference_summary["q_stiff"]
        ),
        "q_contact_relative": _relative(
            candidate_summary["q_contact"], reference_summary["q_contact"]
        ),
        "J_contact_relative": _relative(
            candidate_summary["J_contact"], reference_summary["J_contact"]
        ),
        "J_obs_relative": _relative(
            candidate_observation.J_obs, reference_observation.J_obs
        ),
        "candidate_J_contact": candidate_summary["J_contact"],
        "candidate_J_obs": candidate_observation.J_obs,
        "limiting_cases_pass": limiting_cases_pass,
        "safety_pass": safety_pass,
    }
    metrics["accepted"] = bool(
        metrics["q_form_relative"] <= 0.05
        and metrics["q_stable_relative"] <= 0.05
        and metrics["q_stiff_relative"] <= 0.05
        and metrics["q_contact_relative"] <= 0.02
        and metrics["J_contact_relative"] <= 0.02
        and metrics["J_obs_relative"] <= 0.02
        and minimum_patch_iou >= 0.95
        and limiting_cases_pass
        and safety_pass
    )
    return metrics


def _write_report(
    reference: dict[str, object],
    candidates: list[tuple[float, dict[str, object], dict[str, object]]],
    selected_dwell_s: float,
) -> None:
    reference_summary = _summary(reference)
    lines = [
        "# Force-dwell sentinel screen",
        "",
        "The accepted GPU-resident graph and complete-runtime reset backend is "
        "used without changing force-servo, Newton, contact, mesh, or optical "
        "settings. The 5 s force-band dwell is the gold reference.",
        "",
        f"- reference J_contact: {reference_summary['J_contact']:.9f}",
        f"- reference J_obs difficult pair: {reference_summary['observation'].J_obs:.9f}",
        f"- selected sentinel dwell: {selected_dwell_s:g} s",
        "",
        "| dwell [s] | pass | q_form | q_stable | q_stiff | q_contact | J_contact | J_obs pair | min patch IoU | indentation max [um] | wall [s] |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dwell_s, run, metrics in candidates:
        lines.append(
            f"| {dwell_s:g} | {'PASS' if metrics['accepted'] else 'FAIL'} | "
            f"{100.0 * metrics['q_form_relative']:.3f}% | "
            f"{100.0 * metrics['q_stable_relative']:.3f}% | "
            f"{100.0 * metrics['q_stiff_relative']:.3f}% | "
            f"{100.0 * metrics['q_contact_relative']:.3f}% | "
            f"{100.0 * metrics['J_contact_relative']:.3f}% | "
            f"{100.0 * metrics['J_obs_relative']:.3f}% | "
            f"{metrics['minimum_patch_iou']:.6f} | "
            f"{1.0e6 * metrics['indentation_abs_m']:.3f} | {run['wall_s']:.3f} |"
        )
    lines.extend(("", "## Diagnostics", ""))
    for dwell_s, _, metrics in candidates:
        lines.extend(
            (
                f"### {dwell_s:g} s",
                "",
                f"- force maximum difference: {metrics['force_abs_n']:.6f} N",
                f"- contact centroid maximum difference: {1.0e6 * metrics['centroid_abs_m']:.3f} um",
                f"- maximum particle-speed difference: {metrics['maximum_speed_abs_m_s']:.6e} m/s",
                f"- minimum normal score: {metrics['minimum_normal_score']:.9f}",
                f"- limiting scenarios preserved: {'PASS' if metrics['limiting_cases_pass'] else 'FAIL'}",
                f"- finite/inversion/overflow/contact: {'PASS' if metrics['safety_pass'] else 'FAIL'}",
                "",
            )
        )
    (_OUTPUT_DIRECTORY / "report.md").write_text("\n".join(lines))


def main() -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    reference = _run_dwell(5.0)
    candidates = []
    two_second = _run_dwell(2.0)
    two_metrics = _compare(two_second, reference)
    candidates.append((2.0, two_second, two_metrics))
    if two_metrics["accepted"]:
        one_second = _run_dwell(1.0)
        one_metrics = _compare(one_second, reference)
        candidates.append((1.0, one_second, one_metrics))
        if one_metrics["accepted"]:
            half_second = _run_dwell(0.5)
            half_metrics = _compare(half_second, reference)
            candidates.append((0.5, half_second, half_metrics))
    else:
        three_second = _run_dwell(3.0)
        three_metrics = _compare(three_second, reference)
        candidates.append((3.0, three_second, three_metrics))

    passing = [dwell for dwell, _, metrics in candidates if metrics["accepted"]]
    selected_dwell_s = min(passing) if passing else 5.0
    _write_report(reference, candidates, selected_dwell_s)
    print((_OUTPUT_DIRECTORY / "report.md").read_text())


if __name__ == "__main__":
    main()
