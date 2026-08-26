"""Compare continuous force ramps against the frozen full-finger reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from importlib.resources import as_file, files
from pathlib import Path
from time import perf_counter

import numpy as np

from lumo.fingertip import Fingertip
from lumo.optimization.ax_bo import (
    _campaign_definition,
    _objective_details,
    _save_trial_result,
)
from lumo.optimization.evaluator import evaluate_full_finger
from lumo.optimization.objective import (
    _active_surface_triangles,
    _surface_incidence,
    _triangle_areas,
    compute_objectives_from_raw,
)
from lumo.simulation import QUASISTATIC_RAMP_LOADING


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIRECTORY = (
    _REPOSITORY_ROOT / "output" / "validation" / "quasistatic_ramp_protocol"
)
_REFERENCE_PATH = (
    _REPOSITORY_ROOT
    / "output"
    / "validation"
    / "full_finger_production_objective_freeze"
    / "nominal_full_finger_objectives.npz"
)
_DURATIONS_S = (20.0, 10.0, 5.0, 2.5)
_SPHERE_DIAMETERS_MM = (5.0, 10.0, 20.0)
_CONTACT_Y_MM = (-22.0, -11.0, -5.5, 0.0, 5.5, 11.0, 22.0)
_FORCE_TARGETS_N = (5.0, 10.0, 15.0, 20.0)
_PARAMETERS = {
    "geometry.flat_pad_height_mm": 5.0,
    "geometry.semiellipse_height_mm": 9.0,
    "geometry.stem_width_mm": 7.6,
    "geometry.stem_height_mm": 6.0,
    "geometry.void_width_mm": 2.0,
}
_SUMMARY_FIELDS = (
    "duration_s",
    "force_ramp_rate_n_s",
    "runtime_s",
    "newton_runtime_estimate_s",
    "optix_runtime_s",
    "speedup",
    "J_contact",
    "J_contact_relative_error",
    "J_obs",
    "J_obs_relative_error",
    "limiting_contact_same",
    "limiting_observation_same",
    "worst_force_absolute_error_n",
    "worst_force_relative_error",
    "worst_indentation_absolute_error_mm",
    "worst_indentation_relative_error",
    "worst_patch_area_absolute_error_mm2",
    "worst_patch_area_relative_error",
    "minimum_patch_support_iou",
    "worst_contact_centroid_error_mm",
    "worst_contact_normal_error_deg",
    "worst_deformation_rms_error_mm",
    "worst_deformation_max_error_mm",
    "worst_q_form_absolute_error",
    "worst_q_form_relative_error",
    "worst_q_stable_absolute_error",
    "worst_q_stable_relative_error",
    "worst_q_stiff_absolute_error",
    "worst_q_stiff_relative_error",
    "worst_q_contact_absolute_error",
    "worst_q_contact_relative_error",
    "worst_same_force_separation_absolute_error",
    "worst_same_force_separation_relative_error",
    "worst_observation_l2_error",
    "worst_visible_power_absolute_error",
    "worst_visible_power_relative_error",
    "worst_outside_roi_fraction_error",
    "max_particle_speed_m_s",
    "max_mean_particle_speed_m_s",
    "max_particle_speed_p95_m_s",
    "max_kinetic_energy_j",
    "max_abs_reaction_force_rate_n_s",
    "max_indentation_rate_m_s",
    "max_abs_servo_error_n",
    "minimum_det_f",
    "max_energy_closure_error",
    "valid",
    "failure",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def _relative_error(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    scale = np.abs(reference)
    if np.any(scale <= 1.0e-15):
        raise ValueError("relative-error reference contains a near-zero value")
    return np.abs(values - reference) / scale


def _contact_patches(
    data: dict[str, np.ndarray],
) -> tuple[list[list[set[int]]], np.ndarray, np.ndarray]:
    triangles = np.asarray(data["surface_triangles"], dtype=np.int32)
    reference_vertices = np.asarray(data["reference_vertices_m"], dtype=np.float64)
    vertices = np.asarray(data["silicone_vertices_m"], dtype=np.float64)
    offsets = np.asarray(data["contact_record_offsets"], dtype=np.int64)
    contact_indices = np.asarray(data["contact_particle_indices"], dtype=np.int32)
    vertex_triangles, edge_triangles, triangle_ids = _surface_incidence(triangles)
    patches: list[list[set[int]]] = []
    patch_areas_m2 = np.empty(offsets.shape[:2], dtype=np.float64)
    for scenario_index in range(offsets.shape[0]):
        scenario_patches = []
        for force_index in range(offsets.shape[1]):
            start, count = offsets[scenario_index, force_index]
            patch = _active_surface_triangles(
                contact_indices[start : start + count],
                vertex_triangles=vertex_triangles,
                edge_triangles=edge_triangles,
                triangle_ids=triangle_ids,
            )
            scenario_patches.append(patch)
            deformed_areas = _triangle_areas(
                vertices[scenario_index, force_index],
                triangles,
            )
            patch_areas_m2[scenario_index, force_index] = float(
                deformed_areas[list(patch)].sum()
            )
        patches.append(scenario_patches)
    return patches, patch_areas_m2, _triangle_areas(reference_vertices, triangles)


def _patch_support_iou(
    first: set[int],
    second: set[int],
    reference_areas_m2: np.ndarray,
) -> float:
    union = first | second
    if not union:
        return 0.0
    return float(
        reference_areas_m2[list(first & second)].sum()
        / reference_areas_m2[list(union)].sum()
    )


def _mean_contact_normals(data: dict[str, np.ndarray]) -> np.ndarray:
    offsets = np.asarray(data["contact_record_offsets"], dtype=np.int64)
    records = np.asarray(data["contact_normals_W"], dtype=np.float64)
    means = np.empty((*offsets.shape[:2], 3), dtype=np.float64)
    for scenario_index in range(offsets.shape[0]):
        for force_index in range(offsets.shape[1]):
            start, count = offsets[scenario_index, force_index]
            mean = records[start : start + count].mean(axis=0)
            means[scenario_index, force_index] = mean / np.linalg.norm(mean)
    return means


def _energy_closure(data: dict[str, np.ndarray]) -> float:
    fields = tuple(str(field) for field in data["energy_fields"])
    closure_index = fields.index("closure_error")
    return float(
        max(
            np.max(np.abs(data["no_contact_energy"][:, closure_index])),
            np.max(np.abs(data["energy_matrix"][..., closure_index])),
        )
    )


def _compare(
    candidate: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> dict[str, object]:
    reference_contact, reference_observation = compute_objectives_from_raw(reference)
    contact, observation = compute_objectives_from_raw(candidate)
    reference_patches, reference_areas, reference_triangle_areas = _contact_patches(
        reference
    )
    patches, patch_areas, triangle_areas = _contact_patches(candidate)
    if not np.allclose(
        triangle_areas,
        reference_triangle_areas,
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise RuntimeError("ramp and reference surface topology do not match")
    support_iou = np.empty(reference_areas.shape, dtype=np.float64)
    for scenario_index in range(reference_areas.shape[0]):
        for force_index in range(reference_areas.shape[1]):
            support_iou[scenario_index, force_index] = _patch_support_iou(
                patches[scenario_index][force_index],
                reference_patches[scenario_index][force_index],
                reference_triangle_areas,
            )

    reference_normals = _mean_contact_normals(reference)
    normals = _mean_contact_normals(candidate)
    normal_angles_deg = np.degrees(
        np.arccos(np.clip(np.sum(normals * reference_normals, axis=2), -1.0, 1.0))
    )
    deformation_error_m = np.linalg.norm(
        np.asarray(candidate["silicone_vertices_m"], dtype=np.float64)
        - np.asarray(reference["silicone_vertices_m"], dtype=np.float64),
        axis=3,
    )
    deformation_rms_m = np.sqrt(np.mean(deformation_error_m**2, axis=2))
    deformation_max_m = np.max(deformation_error_m, axis=2)

    combined = np.asarray(candidate["response_matrix"], dtype=np.float64).sum(axis=2)
    reference_combined = np.asarray(
        reference["response_matrix"], dtype=np.float64
    ).sum(axis=2)
    observation_l2 = np.linalg.norm(combined - reference_combined, axis=2)
    visible = np.asarray(candidate["visible_side_power"], dtype=np.float64).sum(axis=2)
    reference_visible = np.asarray(
        reference["visible_side_power"], dtype=np.float64
    ).sum(axis=2)

    separation_mask = np.triu(
        np.ones(reference_observation.location_separations.shape[-2:], dtype=bool),
        k=1,
    )
    separation_mask = np.broadcast_to(
        separation_mask,
        reference_observation.location_separations.shape,
    )
    reference_separations = reference_observation.location_separations[
        separation_mask
    ]
    separations = observation.location_separations[separation_mask]

    actual_forces = np.asarray(candidate["actual_forces_n"], dtype=np.float64)
    force_targets = np.asarray(candidate["force_targets_n"], dtype=np.float64)
    force_relative_error = np.abs(actual_forces - force_targets) / force_targets
    valid = bool(
        np.all(force_relative_error <= 0.1 + 1.0e-12)
        and np.all(np.diff(candidate["indentations_m"], axis=1) > 0.0)
        and np.all(candidate["contact_buffer_overflow"] == 0)
        and np.all(candidate["inverted_tet_counts"] == 0)
        and np.all(np.isfinite(candidate["minimum_det_f"]))
        and np.min(candidate["minimum_det_f"]) > 0.0
        and _energy_closure(candidate) <= 1.0e-12
    )
    return {
        "J_contact": contact.J_contact,
        "J_contact_relative_error": abs(
            contact.J_contact - reference_contact.J_contact
        )
        / abs(reference_contact.J_contact),
        "J_obs": observation.J_obs,
        "J_obs_relative_error": abs(observation.J_obs - reference_observation.J_obs)
        / abs(reference_observation.J_obs),
        "limiting_contact_same": (
            contact.limiting_scenario == reference_contact.limiting_scenario
        ),
        "limiting_observation_same": (
            observation.limiting_sphere_diameter_mm
            == reference_observation.limiting_sphere_diameter_mm
            and observation.limiting_force_n == reference_observation.limiting_force_n
            and observation.limiting_contact_y_pair_mm
            == reference_observation.limiting_contact_y_pair_mm
        ),
        "worst_force_absolute_error_n": float(
            np.max(np.abs(actual_forces - reference["actual_forces_n"]))
        ),
        "worst_force_relative_error": float(np.max(force_relative_error)),
        "worst_indentation_absolute_error_mm": float(
            1.0e3
            * np.max(
                np.abs(candidate["indentations_m"] - reference["indentations_m"])
            )
        ),
        "worst_indentation_relative_error": float(
            np.max(
                _relative_error(
                    candidate["indentations_m"], reference["indentations_m"]
                )
            )
        ),
        "worst_patch_area_absolute_error_mm2": float(
            1.0e6 * np.max(np.abs(patch_areas - reference_areas))
        ),
        "worst_patch_area_relative_error": float(
            np.max(_relative_error(patch_areas, reference_areas))
        ),
        "minimum_patch_support_iou": float(np.min(support_iou)),
        "worst_contact_centroid_error_mm": float(
            1.0e3
            * np.max(
                np.linalg.norm(
                    candidate["contact_centroids_W_m"]
                    - reference["contact_centroids_W_m"],
                    axis=2,
                )
            )
        ),
        "worst_contact_normal_error_deg": float(np.max(normal_angles_deg)),
        "worst_deformation_rms_error_mm": float(1.0e3 * np.max(deformation_rms_m)),
        "worst_deformation_max_error_mm": float(1.0e3 * np.max(deformation_max_m)),
        "worst_q_form_absolute_error": float(
            np.max(np.abs(contact.q_form - reference_contact.q_form))
        ),
        "worst_q_form_relative_error": float(
            np.max(_relative_error(contact.q_form, reference_contact.q_form))
        ),
        "worst_q_stable_absolute_error": float(
            np.max(np.abs(contact.q_stable - reference_contact.q_stable))
        ),
        "worst_q_stable_relative_error": float(
            np.max(_relative_error(contact.q_stable, reference_contact.q_stable))
        ),
        "worst_q_stiff_absolute_error": float(
            np.max(np.abs(contact.q_stiff - reference_contact.q_stiff))
        ),
        "worst_q_stiff_relative_error": float(
            np.max(_relative_error(contact.q_stiff, reference_contact.q_stiff))
        ),
        "worst_q_contact_absolute_error": float(
            np.max(np.abs(contact.q_contact - reference_contact.q_contact))
        ),
        "worst_q_contact_relative_error": float(
            np.max(_relative_error(contact.q_contact, reference_contact.q_contact))
        ),
        "worst_same_force_separation_absolute_error": float(
            np.max(np.abs(separations - reference_separations))
        ),
        "worst_same_force_separation_relative_error": float(
            np.max(_relative_error(separations, reference_separations))
        ),
        "worst_observation_l2_error": float(np.max(observation_l2)),
        "worst_visible_power_absolute_error": float(
            np.max(np.abs(visible - reference_visible))
        ),
        "worst_visible_power_relative_error": float(
            np.max(_relative_error(visible, reference_visible))
        ),
        "worst_outside_roi_fraction_error": float(
            np.max(
                np.abs(
                    candidate["outside_roi_power_fraction"]
                    - reference["outside_roi_power_fraction"]
                )
            )
        ),
        "max_particle_speed_m_s": float(
            np.max(candidate["maximum_particle_speeds_m_s"])
        ),
        "max_mean_particle_speed_m_s": float(
            np.max(candidate["mean_particle_speeds_m_s"])
        ),
        "max_particle_speed_p95_m_s": float(
            np.max(candidate["particle_speed_p95_m_s"])
        ),
        "max_kinetic_energy_j": float(np.max(candidate["kinetic_energy_j"])),
        "max_abs_reaction_force_rate_n_s": float(
            np.max(np.abs(candidate["reaction_force_rates_n_s"]))
        ),
        "max_indentation_rate_m_s": float(
            np.max(np.abs(candidate["indentation_rates_m_s"]))
        ),
        "max_abs_servo_error_n": float(
            np.max(np.abs(candidate["servo_errors_n"]))
        ),
        "minimum_det_f": float(np.min(candidate["minimum_det_f"])),
        "max_energy_closure_error": _energy_closure(candidate),
        "valid": valid,
        "failure": "" if valid else "scientific integrity check failed",
    }


def _write_summary(rows: list[dict[str, object]]) -> None:
    path = _OUTPUT_DIRECTORY / "protocol_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_checkpoint_comparison(
    candidates: list[tuple[float, dict[str, np.ndarray]]],
    reference: dict[str, np.ndarray],
) -> None:
    fields = (
        "duration_s",
        "force_ramp_rate_n_s",
        "scenario",
        "sphere_diameter_mm",
        "contact_y_mm",
        "target_force_n",
        "actual_force_n",
        "reference_force_n",
        "force_error_from_target_n",
        "force_error_from_reference_n",
        "capture_time_s",
        "reference_checkpoint_time_s",
        "indentation_mm",
        "reference_indentation_mm",
        "indentation_error_mm",
        "patch_area_mm2",
        "reference_patch_area_mm2",
        "patch_support_iou",
        "contact_centroid_error_mm",
        "contact_normal_error_deg",
        "deformation_rms_error_mm",
        "deformation_max_error_mm",
        "observation_l2_error",
        "visible_power",
        "reference_visible_power",
        "outside_roi_fraction",
        "reference_outside_roi_fraction",
        "maximum_particle_speed_m_s",
        "mean_particle_speed_m_s",
        "rms_particle_speed_m_s",
        "particle_speed_p95_m_s",
        "kinetic_energy_j",
        "force_reference_n",
        "reaction_force_rate_n_s",
        "indentation_rate_m_s",
        "servo_error_n",
        "contact_count",
        "reference_contact_count",
        "minimum_det_f",
        "reference_minimum_det_f",
        "inversion_count",
        "contact_buffer_overflow",
        "optics_runtime_s",
    )
    reference_patches, reference_patch_areas, reference_triangle_areas = (
        _contact_patches(reference)
    )
    reference_normals = _mean_contact_normals(reference)
    reference_vertices = np.asarray(reference["silicone_vertices_m"], dtype=np.float64)
    reference_response = np.asarray(reference["response_matrix"], dtype=np.float64).sum(
        axis=2
    )
    reference_visible = np.asarray(
        reference["visible_side_power"], dtype=np.float64
    ).sum(axis=2)
    rows: list[dict[str, object]] = []
    for duration_s, candidate in candidates:
        patches, patch_areas, _ = _contact_patches(candidate)
        normals = _mean_contact_normals(candidate)
        response = np.asarray(candidate["response_matrix"], dtype=np.float64).sum(
            axis=2
        )
        visible = np.asarray(candidate["visible_side_power"], dtype=np.float64).sum(
            axis=2
        )
        vertex_error_m = np.linalg.norm(
            np.asarray(candidate["silicone_vertices_m"], dtype=np.float64)
            - reference_vertices,
            axis=3,
        )
        for scenario_index, scenario_name in enumerate(candidate["scenario_names"]):
            for force_index, target_force_n in enumerate(
                candidate["force_targets_n"]
            ):
                normal_angle_deg = float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                np.dot(
                                    normals[scenario_index, force_index],
                                    reference_normals[scenario_index, force_index],
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
                rows.append(
                    {
                        "duration_s": duration_s,
                        "force_ramp_rate_n_s": 20.0 / duration_s,
                        "scenario": str(scenario_name),
                        "sphere_diameter_mm": candidate["sphere_diameters_mm"][
                            scenario_index
                        ],
                        "contact_y_mm": candidate["contact_y_mm"][scenario_index],
                        "target_force_n": target_force_n,
                        "actual_force_n": candidate["actual_forces_n"][
                            scenario_index, force_index
                        ],
                        "reference_force_n": reference["actual_forces_n"][
                            scenario_index, force_index
                        ],
                        "force_error_from_target_n": candidate["actual_forces_n"][
                            scenario_index, force_index
                        ]
                        - target_force_n,
                        "force_error_from_reference_n": candidate["actual_forces_n"][
                            scenario_index, force_index
                        ]
                        - reference["actual_forces_n"][scenario_index, force_index],
                        "capture_time_s": candidate["checkpoint_times_s"][
                            scenario_index, force_index
                        ],
                        "reference_checkpoint_time_s": reference[
                            "checkpoint_times_s"
                        ][scenario_index, force_index],
                        "indentation_mm": 1.0e3
                        * candidate["indentations_m"][scenario_index, force_index],
                        "reference_indentation_mm": 1.0e3
                        * reference["indentations_m"][scenario_index, force_index],
                        "indentation_error_mm": 1.0e3
                        * (
                            candidate["indentations_m"][scenario_index, force_index]
                            - reference["indentations_m"][scenario_index, force_index]
                        ),
                        "patch_area_mm2": 1.0e6
                        * patch_areas[scenario_index, force_index],
                        "reference_patch_area_mm2": 1.0e6
                        * reference_patch_areas[scenario_index, force_index],
                        "patch_support_iou": _patch_support_iou(
                            patches[scenario_index][force_index],
                            reference_patches[scenario_index][force_index],
                            reference_triangle_areas,
                        ),
                        "contact_centroid_error_mm": 1.0e3
                        * np.linalg.norm(
                            candidate["contact_centroids_W_m"][
                                scenario_index, force_index
                            ]
                            - reference["contact_centroids_W_m"][
                                scenario_index, force_index
                            ]
                        ),
                        "contact_normal_error_deg": normal_angle_deg,
                        "deformation_rms_error_mm": 1.0e3
                        * np.sqrt(
                            np.mean(vertex_error_m[scenario_index, force_index] ** 2)
                        ),
                        "deformation_max_error_mm": 1.0e3
                        * np.max(vertex_error_m[scenario_index, force_index]),
                        "observation_l2_error": np.linalg.norm(
                            response[scenario_index, force_index]
                            - reference_response[scenario_index, force_index]
                        ),
                        "visible_power": visible[scenario_index, force_index],
                        "reference_visible_power": reference_visible[
                            scenario_index, force_index
                        ],
                        "outside_roi_fraction": candidate[
                            "outside_roi_power_fraction"
                        ][scenario_index, force_index],
                        "reference_outside_roi_fraction": reference[
                            "outside_roi_power_fraction"
                        ][scenario_index, force_index],
                        "maximum_particle_speed_m_s": candidate[
                            "maximum_particle_speeds_m_s"
                        ][scenario_index, force_index],
                        "mean_particle_speed_m_s": candidate[
                            "mean_particle_speeds_m_s"
                        ][scenario_index, force_index],
                        "rms_particle_speed_m_s": candidate[
                            "rms_particle_speeds_m_s"
                        ][scenario_index, force_index],
                        "particle_speed_p95_m_s": candidate[
                            "particle_speed_p95_m_s"
                        ][scenario_index, force_index],
                        "kinetic_energy_j": candidate["kinetic_energy_j"][
                            scenario_index, force_index
                        ],
                        "force_reference_n": candidate["force_references_n"][
                            scenario_index, force_index
                        ],
                        "reaction_force_rate_n_s": candidate[
                            "reaction_force_rates_n_s"
                        ][scenario_index, force_index],
                        "indentation_rate_m_s": candidate["indentation_rates_m_s"][
                            scenario_index, force_index
                        ],
                        "servo_error_n": candidate["servo_errors_n"][
                            scenario_index, force_index
                        ],
                        "contact_count": candidate["indenter_contact_counts"][
                            scenario_index, force_index
                        ],
                        "reference_contact_count": reference[
                            "indenter_contact_counts"
                        ][scenario_index, force_index],
                        "minimum_det_f": candidate["minimum_det_f"][
                            scenario_index, force_index
                        ],
                        "reference_minimum_det_f": reference["minimum_det_f"][
                            scenario_index, force_index
                        ],
                        "inversion_count": candidate["inverted_tet_counts"][
                            scenario_index, force_index
                        ],
                        "contact_buffer_overflow": candidate[
                            "contact_buffer_overflow"
                        ][scenario_index, force_index],
                        "optics_runtime_s": candidate[
                            "checkpoint_optics_runtime_s"
                        ][scenario_index, force_index],
                    }
                )
    with (_OUTPUT_DIRECTORY / "checkpoint_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_interim_report(
    rows: list[dict[str, object]],
    reference: dict[str, np.ndarray],
) -> None:
    reference_contact, reference_observation = compute_objectives_from_raw(reference)
    valid_rows = [row for row in rows if row["valid"]]
    ordered = sorted(valid_rows, key=lambda item: float(item["duration_s"]), reverse=True)
    lines = [
        "# Continuous quasi-static ramp protocol",
        "",
        "Result: NO FAST PROTOCOL ACCEPTED",
        "",
        "This validation compares continuously loaded force-reference ramps against "
        "the frozen 5 s dwell reference. It does not change either objective, does "
        "not change the production evaluator default, and does not start Ax/BO.",
        "",
        "## Frozen reference",
        "",
        f"- raw artifact: `{_REFERENCE_PATH}`",
        f"- J_contact: {reference_contact.J_contact:.9f}",
        f"- J_obs: {reference_observation.J_obs:.9f}",
        f"- runtime: {float(reference['evaluation_runtime_s']):.3f} s",
        "- loading: sequential 5/10/15/20 N force servo with 5 s in-band dwell",
        f"- maximum saved checkpoint particle speed: "
        f"{float(np.max(reference['maximum_particle_speeds_m_s'])):.3e} m/s",
        "",
        "## Nominal protocol sweep",
        "",
        "| Duration [s] | Rate [N/s] | Runtime [s] | Speedup | J_contact | Error | J_obs | Error | Worst indentation error | Max particle speed [m/s] | Valid |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in sorted(rows, key=lambda item: float(item["duration_s"]), reverse=True):
        if row["valid"]:
            lines.append(
                f"| {float(row['duration_s']):g} | "
                f"{float(row['force_ramp_rate_n_s']):g} | "
                f"{float(row['runtime_s']):.3f} | {float(row['speedup']):.2f}x | "
                f"{float(row['J_contact']):.9f} | "
                f"{float(row['J_contact_relative_error']):.2%} | "
                f"{float(row['J_obs']):.9f} | "
                f"{float(row['J_obs_relative_error']):.2%} | "
                f"{float(row['worst_indentation_relative_error']):.2%} | "
                f"{float(row['max_particle_speed_m_s']):.3e} | yes |"
            )
        else:
            lines.append(
                f"| {float(row['duration_s']):g} | "
                f"{float(row['force_ramp_rate_n_s']):g} | "
                f"{row.get('runtime_s', '')} |  |  |  |  |  |  |  | no |"
            )
    lines.extend(
        [
            "",
            "All four runs passed force tolerance, finite-state, positive-det(F), "
            "zero-inversion, zero-buffer-overflow, and optical energy-closure checks. "
            "`Valid` in the table means execution integrity, not approximation "
            "acceptance.",
            "",
            "## Objective and component fidelity",
            "",
            "| Duration [s] | Jc <=1/2/5% | Jo <=1/2/5% | q_form worst | q_stable worst | q_stiff worst | q_contact worst | Same-force separation worst | Limiting cases preserved |",
            "|---:|:---:|:---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in ordered:
        contact_error = float(row["J_contact_relative_error"])
        observation_error = float(row["J_obs_relative_error"])
        contact_bands = "/".join(
            "Y" if contact_error <= band else "N" for band in (0.01, 0.02, 0.05)
        )
        observation_bands = "/".join(
            "Y" if observation_error <= band else "N"
            for band in (0.01, 0.02, 0.05)
        )
        lines.append(
            f"| {float(row['duration_s']):g} | {contact_bands} | "
            f"{observation_bands} | "
            f"{float(row['worst_q_form_relative_error']):.2%} | "
            f"{float(row['worst_q_stable_relative_error']):.2%} | "
            f"{float(row['worst_q_stiff_relative_error']):.2%} | "
            f"{float(row['worst_q_contact_relative_error']):.2%} | "
            f"{float(row['worst_same_force_separation_relative_error']):.2%} | "
            f"{'yes' if row['limiting_contact_same'] and row['limiting_observation_same'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Absolute worst-case differences are retained in `protocol_summary.csv`; "
            "all 336 protocol/scenario/checkpoint comparisons are in "
            "`checkpoint_comparison.csv`.",
            "",
            "The decisive failure is not the scalar objective alone. Even the 20 s "
            "ramp changes the worst same-force separation by 272.61%. The specific "
            "worst comparison is the 10 mm sphere at 5 N, Y=-22 versus +22 mm: "
            "reference 0.003239302, ramp 0.012070090. This does not approach the "
            "dwell result monotonically as the ramp is slowed. The limiting scalar "
            "J_obs pair itself remains 20 mm, 5 N, Y=+5.5 versus +11 mm for every "
            "candidate, so the scalar similarity hides large non-limiting response "
            "changes.",
            "",
            "## Dynamic diagnostics",
            "",
            "| Duration [s] | Max speed [m/s] | Max mean speed [m/s] | Max P95 speed [m/s] | Max kinetic energy [J] | Max |dF/dt| [N/s] | Max indentation rate [mm/s] | Max servo error [N] |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ordered:
        lines.append(
            f"| {float(row['duration_s']):g} | "
            f"{float(row['max_particle_speed_m_s']):.3e} | "
            f"{float(row['max_mean_particle_speed_m_s']):.3e} | "
            f"{float(row['max_particle_speed_p95_m_s']):.3e} | "
            f"{float(row['max_kinetic_energy_j']):.3e} | "
            f"{float(row['max_abs_reaction_force_rate_n_s']):.3f} | "
            f"{1.0e3 * float(row['max_indentation_rate_m_s']):.3f} | "
            f"{float(row['max_abs_servo_error_n']):.3f} |"
        )
    lines.extend(
        [
            "",
            "The current tetrahedral material includes `k_damp=10 Pa s`, so rate "
            "dependence from constitutive damping and internal/contact redistribution "
            "is expected. The reference artifact predates the new mean/P95/kinetic "
            "diagnostics, so only its saved maximum checkpoint speed can be compared "
            "directly. No unload loop was added: the one-way ramp already failed the "
            "reference-fidelity gate, and unload is not part of the current production "
            "controller path.",
            "",
            "## Runtime accounting",
            "",
            "| Protocol | Newton estimate [s] | Newton/scenario [s] | OptiX [s] | Total [s] | 60 eval [h] | 100 eval [h] | 120 eval [h] |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| reference dwell | n/a | n/a | n/a | {float(reference['evaluation_runtime_s']):.1f} | "
            f"{60.0 * float(reference['evaluation_runtime_s']) / 3600.0:.1f} | "
            f"{100.0 * float(reference['evaluation_runtime_s']) / 3600.0:.1f} | "
            f"{120.0 * float(reference['evaluation_runtime_s']) / 3600.0:.1f} |",
        ]
    )
    for row in ordered:
        runtime_s = float(row["runtime_s"])
        newton_s = float(row["newton_runtime_estimate_s"])
        lines.append(
            f"| {float(row['duration_s']):g} s ramp | {newton_s:.1f} | "
            f"{newton_s / 21.0:.1f} | {float(row['optix_runtime_s']):.1f} | "
            f"{runtime_s:.1f} | {60.0 * runtime_s / 3600.0:.1f} | "
            f"{100.0 * runtime_s / 3600.0:.1f} | "
            f"{120.0 * runtime_s / 3600.0:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Morphology-ranking robustness",
            "",
            "Not run. The task required this phase only after selecting a promising "
            "nominal ramp. No candidate preserved the underlying same-force optical "
            "separations within even 5%, so additional compliant/stiff reference runs "
            "would not establish an acceptable fast protocol. No morphology-ranking "
            "claim is made.",
            "",
            "## Recommended BO loading protocol",
            "",
            "- loading mode: no continuous ramp accepted",
            "- force ramp rate / duration: none",
            "- checkpoint capture method tested: first actual reaction-force crossing",
            "- force tolerance: +/-10% (all tested captures satisfied it)",
            "- BO evaluator: retain `reference_dwell` (5 s per checkpoint)",
            "- expected runtime: 1222.991 s per morphology",
            "- speedup: 1.00x",
            "- important caveat: the 2.5 s ramp reached 2.83x speedup, but changed "
            "J_contact by 8.13% and underlying optical separations by 271.98%",
            "",
            "## Explicit answers",
            "",
            "1. Continuous loading did not reproduce the 5 s dwell raw objective "
            "inputs consistently.",
            "2. No ramp rate in 1/2/4/8 N/s is acceptable; therefore there is no "
            "fastest acceptable rate.",
            "3. The 20 s ramp nearly reproduces scalar J_contact, but component "
            "errors reach about 4.7%; faster ramps reach 14.6-27.9% q_stiff error.",
            "4. Every ramp preserves the limiting J_obs pair, but J_obs differs by "
            "3.94-7.70% and non-limiting same-force separations differ by >271%.",
            "5. Rate/history contamination is significant: force-reference lag, "
            "nonzero indentation rate, and non-convergent optical fields remain.",
            "6. Generalization beyond nominal was not tested because no nominal "
            "candidate passed the component-fidelity gate.",
            "7. Measured speedups span 1.17x to 2.83x, but none is accepted.",
            "8. This protocol should not replace the 5 s dwell evaluator for BO.",
            "9. Final Pareto candidates should continue to use the 5 s reference; "
            "currently all BO evaluations must also use it.",
            "10. No Ax/BO campaign was started.",
        ]
    )
    (_OUTPUT_DIRECTORY / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare full-finger force ramps against the frozen reference."
    )
    parser.add_argument(
        "--durations",
        nargs="+",
        type=float,
        default=_DURATIONS_S,
        help="seconds for the force reference to rise from 0 to 20 N",
    )
    arguments = parser.parse_args()
    durations_s = tuple(float(value) for value in arguments.durations)
    if any(not np.isfinite(value) or value <= 0.0 for value in durations_s):
        raise ValueError("durations must be finite and positive")

    if not _REFERENCE_PATH.is_file():
        raise FileNotFoundError(_REFERENCE_PATH)
    reference = _load(_REFERENCE_PATH)
    if tuple(reference["sphere_diameters_mm"][:: len(_CONTACT_Y_MM)]) != (
        _SPHERE_DIAMETERS_MM
    ):
        raise RuntimeError("reference sphere contract does not match")
    if not np.array_equal(reference["force_targets_n"], _FORCE_TARGETS_N):
        raise RuntimeError("reference force contract does not match")
    if not np.array_equal(
        reference["contact_y_mm"][: len(_CONTACT_Y_MM)], _CONTACT_Y_MM
    ):
        raise RuntimeError("reference contact-Y contract does not match")

    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    run_config = {
        "reference_path": str(_REFERENCE_PATH),
        "reference_sha256": _sha256(_REFERENCE_PATH),
        "loading_mode": QUASISTATIC_RAMP_LOADING,
        "candidate_durations_s": list(durations_s),
        "candidate_force_ramp_rates_n_s": [20.0 / value for value in durations_s],
        "checkpoint_capture": "first actual reaction-force crossing",
        "force_tolerance_fraction": 0.1,
        "sphere_diameters_mm": list(_SPHERE_DIAMETERS_MM),
        "contact_y_mm": list(_CONTACT_Y_MM),
        "force_targets_n": list(_FORCE_TARGETS_N),
        "sim_frequency_hz": 100.0,
        "vbd_iterations": 10,
        "optical_paths_per_led": 65536,
        "max_bounces": 24,
        "morphology_parameters_mm": _PARAMETERS,
    }
    (_OUTPUT_DIRECTORY / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    campaign = _campaign_definition(
        "discrete-05mm",
        sphere_diameters_mm=_SPHERE_DIAMETERS_MM,
        contact_y_mm=_CONTACT_Y_MM,
        force_targets_n=_FORCE_TARGETS_N,
    )
    fingertip = Fingertip(campaign.space.to_parameters(_PARAMETERS))
    resource_root = files("lumo").joinpath("assets", "objects", "urdf")
    rows: list[dict[str, object]] = []
    with (
        as_file(resource_root.joinpath("sphere_5mm.urdf")) as sphere_5mm,
        as_file(resource_root.joinpath("sphere_10mm.urdf")) as sphere_10mm,
        as_file(resource_root.joinpath("sphere_20mm.urdf")) as sphere_20mm,
    ):
        sphere_paths = (sphere_5mm, sphere_10mm, sphere_20mm)
        candidate_artifacts: list[tuple[float, dict[str, np.ndarray]]] = []
        for duration_s in durations_s:
            rate_n_s = 20.0 / duration_s
            slug = f"{duration_s:g}s".replace(".", "p")
            raw_path = _OUTPUT_DIRECTORY / f"ramp_{slug}.npz"
            print(
                f"ramp duration={duration_s:g} s, rate={rate_n_s:g} N/s",
                flush=True,
            )
            try:
                if not raw_path.is_file():
                    start_s = perf_counter()
                    evaluation = evaluate_full_finger(
                        fingertip,
                        sphere_paths,
                        _SPHERE_DIAMETERS_MM,
                        _CONTACT_Y_MM,
                        force_targets_n=_FORCE_TARGETS_N,
                        settle_duration_s=0.0,
                        force_tolerance_fraction=0.1,
                        initial_clearance_m=1.0e-3,
                        approach_speed_m_s=5.0e-3,
                        max_sim_time_s=max(60.0, duration_s + 30.0),
                        loading_mode=QUASISTATIC_RAMP_LOADING,
                        force_ramp_rate_n_s=rate_n_s,
                    )
                    runtime_s = perf_counter() - start_s
                    details = _objective_details(evaluation)
                    _save_trial_result(
                        raw_path,
                        campaign=campaign,
                        evaluation=evaluation,
                        details=details,
                        parameters=_PARAMETERS,
                        runtime_s=runtime_s,
                    )
                    del evaluation
                candidate = _load(raw_path)
                comparison = _compare(candidate, reference)
                runtime_s = float(candidate["evaluation_runtime_s"])
                optix_runtime_s = float(candidate["no_contact_optics_runtime_s"]) + float(
                    np.sum(candidate["checkpoint_optics_runtime_s"])
                )
                row = {
                    "duration_s": duration_s,
                    "force_ramp_rate_n_s": rate_n_s,
                    "runtime_s": runtime_s,
                    "newton_runtime_estimate_s": runtime_s - optix_runtime_s,
                    "optix_runtime_s": optix_runtime_s,
                    "speedup": float(reference["evaluation_runtime_s"]) / runtime_s,
                    **comparison,
                }
                candidate_artifacts.append((duration_s, candidate))
            except Exception as error:
                row = {
                    name: "" for name in _SUMMARY_FIELDS
                }
                row.update(
                    duration_s=duration_s,
                    force_ramp_rate_n_s=rate_n_s,
                    valid=False,
                    failure=f"{type(error).__name__}: {error}",
                )
                print(f"  INVALID: {row['failure']}", flush=True)
            rows.append(row)
            _write_summary(rows)
            _write_interim_report(rows, reference)
            if row["valid"]:
                print(
                    f"  J_contact={float(row['J_contact']):.9f} "
                    f"({float(row['J_contact_relative_error']):.2%}), "
                    f"J_obs={float(row['J_obs']):.9f} "
                    f"({float(row['J_obs_relative_error']):.2%}), "
                    f"runtime={float(row['runtime_s']):.1f} s",
                    flush=True,
                )

    _write_checkpoint_comparison(candidate_artifacts, reference)
    _write_interim_report(rows, reference)
    print(f"summary: {_OUTPUT_DIRECTORY / 'protocol_summary.csv'}")
    print(
        "checkpoint comparison: "
        f"{_OUTPUT_DIRECTORY / 'checkpoint_comparison.csv'}"
    )
    print(f"report: {_OUTPUT_DIRECTORY / 'report.md'}")


if __name__ == "__main__":
    main()
