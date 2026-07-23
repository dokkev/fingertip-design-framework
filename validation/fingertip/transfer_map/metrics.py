"""Numerical synthesis for the Phase 4K transfer-map validation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from fem.indentation import IndentationSettings

REPRESENTATIVE_DEPTHS_MM = (0.25, 0.5, 1.0, 1.5)
MEDIUM_LOCATIONS = (0.20, 0.35, 0.50, 0.65, 0.80)
FINE_LOCATIONS = (0.20, 0.50, 0.80)
SIDE_NAMES = ("left", "right")

def _depth_step(depth: float) -> int:
    return int(round(depth / 1.5 * 48.0)) - 1


def signature(record: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(row["u_normal_mm"])
            for side in SIDE_NAMES
            for row in record["observation_sidewalls"][side]
        ],
        dtype=float,
    )


def tangent_signature(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    delta = np.asarray([record["delta_n_mm"] for record in records], dtype=float)
    values = np.asarray([signature(record) for record in records])
    return np.gradient(values, delta, axis=0, edge_order=1)


def _signature_norm(vector: np.ndarray, eta: np.ndarray) -> float:
    count = len(eta)
    return math.sqrt(
        float(np.trapezoid(vector[:count] ** 2, eta))
        + float(np.trapezoid(vector[count:] ** 2, eta))
    )


def _distance_matrix(
    signatures: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    count = signatures.shape[0]
    result = np.zeros((count, count), dtype=float)
    for first in range(count):
        for second in range(count):
            result[first, second] = _signature_norm(
                signatures[first] - signatures[second], eta
            )
    return result


def _shape_distance_matrix(
    signatures: np.ndarray,
    eta: np.ndarray,
    floor_mm: float,
) -> np.ndarray:
    norms = np.asarray(
        [_signature_norm(signature, eta) for signature in signatures]
    )
    normalized = np.asarray(
        [
            signature / max(norm, floor_mm)
            for signature, norm in zip(signatures, norms)
        ]
    )
    return _distance_matrix(normalized, eta)


def _normalized_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(
        second_centered
    )
    return (
        float(np.dot(first_centered, second_centered) / denominator)
        if denominator > 1.0e-14
        else None
    )


def _profile_difference(
    medium: np.ndarray,
    fine: np.ndarray,
    floor_mm: float,
) -> dict[str, Any]:
    difference = medium - fine
    denominator = max(
        float(np.linalg.norm(fine)),
        floor_mm * math.sqrt(fine.size),
    )
    return {
        "relative_l2_difference": float(np.linalg.norm(difference))
        / denominator,
        "maximum_absolute_difference_mm": float(
            np.max(np.abs(difference))
        ),
        "normalized_shape_correlation": _normalized_correlation(
            medium, fine
        ),
    }


def assemble_arrays(
    loaded: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    case_count = len(loaded)
    step_count = 48
    side_count = 2
    sample_count = 41
    u_xy = np.full(
        (case_count, step_count, side_count, sample_count, 2),
        np.nan,
    )
    u_normal = np.full(
        (case_count, step_count, side_count, sample_count), np.nan
    )
    u_tangent = np.full_like(u_normal, np.nan)
    delta = np.full((case_count, step_count), np.nan)
    force = np.full_like(delta, np.nan)
    centroid = np.full_like(delta, np.nan)
    length = np.full_like(delta, np.nan)
    valid = np.zeros((case_count, step_count), dtype=bool)
    xi = np.asarray(
        [float(spec["xi_cmd"]) for spec, _, _ in loaded], dtype=float
    )
    eta = np.tile(np.linspace(0.0, 1.0, sample_count), (2, 1))
    rows: list[dict[str, Any]] = []
    for case_index, (spec, result, records) in enumerate(loaded):
        for record in records:
            step_index = int(record["step"]) - 1
            if not 0 <= step_index < step_count:
                continue
            delta[case_index, step_index] = float(record["delta_n_mm"])
            force[case_index, step_index] = float(
                record["canonical_normal_reaction_n"]
            )
            contact = record["contact"]
            if contact["xi_centroid"] is not None:
                centroid[case_index, step_index] = float(
                    contact["xi_centroid"]
                )
            if contact["contact_length_mm"] is not None:
                length[case_index, step_index] = float(
                    contact["contact_length_mm"]
                )
            record_valid = bool(
                record["solver_converged"]
                and record["finite_fields"]
                and float(record["minimum_det_f"]) > 0.0
            )
            valid[case_index, step_index] = record_valid
            for side_index, side in enumerate(SIDE_NAMES):
                for sample_index, sample in enumerate(
                    record["observation_sidewalls"][side]
                ):
                    ux = float(sample["ux_mm"])
                    uy = float(sample["uy_mm"])
                    normal = float(sample["u_normal_mm"])
                    tangent = float(sample["u_tangent_mm"])
                    u_xy[
                        case_index,
                        step_index,
                        side_index,
                        sample_index,
                    ] = (ux, uy)
                    u_normal[
                        case_index,
                        step_index,
                        side_index,
                        sample_index,
                    ] = normal
                    u_tangent[
                        case_index,
                        step_index,
                        side_index,
                        sample_index,
                    ] = tangent
                    rows.append(
                        {
                            "case": spec["case_name"],
                            "mesh": spec["mesh"],
                            "step": record["step"],
                            "delta_n": record["delta_n_mm"],
                            "xi_cmd": spec["xi_cmd"],
                            "xi_centroid": contact["xi_centroid"],
                            "F_n": record["canonical_normal_reaction_n"],
                            "contact_length": contact["contact_length_mm"],
                            "side_name": side,
                            "eta": sample["eta"],
                            "X0_x": sample["reference_x_mm"],
                            "X0_y": sample["reference_y_mm"],
                            "u_x": ux,
                            "u_y": uy,
                            "u_normal": normal,
                            "u_tangent": tangent,
                            "deformed_x": sample["deformed_x_mm"],
                            "deformed_y": sample["deformed_y_mm"],
                            "min_detF": record["minimum_det_f"],
                            "strain_metric": record[
                                "canonical_strain_metric"
                            ]["value"],
                            "valid": record_valid,
                        }
                    )
    arrays = {
        "xi_cmd": xi,
        "delta_n": delta,
        "eta": eta,
        "u_xy": u_xy,
        "u_normal": u_normal,
        "u_tangent": u_tangent,
        "F_n": force,
        "xi_centroid": centroid,
        "contact_length": length,
        "valid_mask": valid,
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        arrays["G_secant"] = u_normal / delta[:, :, None, None]
    tangent_gain = np.full_like(u_normal, np.nan)
    for case_index in range(case_count):
        case_valid = valid[case_index]
        if np.count_nonzero(case_valid) >= 2:
            tangent_gain[case_index, case_valid] = np.gradient(
                u_normal[case_index, case_valid],
                delta[case_index, case_valid],
                axis=0,
                edge_order=1,
            )
    arrays["G_tangent"] = tangent_gain
    medium_indices = [
        next(
            index
            for index, (spec, _, _) in enumerate(loaded)
            if spec["mesh"] == "medium"
            and abs(float(spec["xi_cmd"]) - xi_value) <= 1.0e-15
        )
        for xi_value in MEDIUM_LOCATIONS
    ]
    arrays["medium_xi"] = np.asarray(MEDIUM_LOCATIONS, dtype=float)
    arrays["S_location"] = np.gradient(
        u_normal[medium_indices],
        np.asarray(MEDIUM_LOCATIONS, dtype=float),
        axis=0,
        edge_order=1,
    )
    return arrays, rows



def synthesize_metrics(
    loaded: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ],
) -> dict[str, Any]:
    by_key = {
        (str(spec["mesh"]), float(spec["xi_cmd"])): (result, records)
        for spec, result, records in loaded
    }
    medium = [
        (xi, *by_key[("medium", xi)]) for xi in MEDIUM_LOCATIONS
    ]
    eta = np.linspace(0.0, 1.0, 41)
    floor = IndentationSettings(1.5, 48).profile_displacement_floor_mm
    slices: dict[str, Any] = {}
    all_medium_complete = all(len(records) == 48 for _, _, records in medium)
    for depth in REPRESENTATIVE_DEPTHS_MM:
        step_index = _depth_step(depth)
        if not all_medium_complete:
            slices[f"{depth:g}"] = {
                "available": False,
                "reason": "one or more medium cases incomplete",
            }
            continue
        signatures = np.asarray(
            [signature(records[step_index]) for _, _, records in medium]
        )
        norms = np.asarray(
            [_signature_norm(signature, eta) for signature in signatures]
        )
        centered = signatures - np.mean(signatures, axis=0, keepdims=True)
        singular_values = np.linalg.svd(
            centered, full_matrices=False, compute_uv=False
        )
        location_sensitivity = np.gradient(
            signatures,
            np.asarray(MEDIUM_LOCATIONS),
            axis=0,
            edge_order=1,
        )
        slices[f"{depth:g}"] = {
            "available": True,
            "step": step_index + 1,
            "interpolated": False,
            "lateral_signal_norm_mm": norms.tolist(),
            "indentation_normalized_gain": (norms / depth).tolist(),
            "fixed_indentation_distance_matrix_mm": _distance_matrix(
                signatures, eta
            ).tolist(),
            "amplitude_normalized_shape_distance_matrix": (
                _shape_distance_matrix(signatures, eta, floor).tolist()
            ),
            "location_sensitivity_l2_mm_per_xi": [
                _signature_norm(value, eta)
                for value in location_sensitivity
            ],
            "signature_singular_values_mm": singular_values.tolist(),
            "svd_interpretation": (
                "descriptive only; no optical-noise observability threshold"
            ),
        }

    force_conditioned: dict[str, Any]
    if not all(len(records) == 48 for _, _, records in medium):
        force_conditioned = {
            "available": False,
            "reason": "one or more force curves are incomplete",
        }
    else:
        lower = max(
            min(
                float(record["canonical_normal_reaction_n"])
                for record in records
            )
            for _, _, records in medium
        )
        upper = min(
            max(
                float(record["canonical_normal_reaction_n"])
                for record in records
            )
            for _, _, records in medium
        )
        force_levels = np.linspace(lower, upper, 5)
        matrices = []
        crossing_counts: list[list[int]] = []
        interpolation_failed = False
        for target_force in force_levels:
            signatures = []
            level_crossing_counts = []
            for _, _, records in medium:
                forces = np.asarray(
                    [
                        record["canonical_normal_reaction_n"]
                        for record in records
                    ],
                    dtype=float,
                )
                profiles = np.asarray(
                    [signature(record) for record in records]
                )
                crossings = [
                    index
                    for index in range(len(forces) - 1)
                    if (
                        min(forces[index], forces[index + 1])
                        <= target_force
                        <= max(forces[index], forces[index + 1])
                    )
                    and abs(forces[index + 1] - forces[index]) > 1.0e-14
                ]
                level_crossing_counts.append(len(crossings))
                if not crossings:
                    interpolation_failed = True
                    break
                index = crossings[0]
                weight = (
                    (target_force - forces[index])
                    / (forces[index + 1] - forces[index])
                )
                signatures.append(
                    (1.0 - weight) * profiles[index]
                    + weight * profiles[index + 1]
                )
            crossing_counts.append(level_crossing_counts)
            if interpolation_failed:
                break
            matrices.append(
                _distance_matrix(np.asarray(signatures), eta).tolist()
            )
        if interpolation_failed:
            force_conditioned = {
                "available": False,
                "reason": "a common force level has no loading-path crossing",
                "common_force_range_n": [lower, upper],
            }
        else:
            force_conditioned = {
                "available": True,
                "common_force_range_n": [lower, upper],
                "force_levels_n": force_levels.tolist(),
                "distance_matrices_mm": matrices,
                "crossing_counts_by_force_and_location": crossing_counts,
                "interpolation": (
                    "piecewise linear at the first crossing along the unmodified "
                    "loading path; no monotonicization or smoothing"
                ),
                "multiple_crossing_policy": (
                    "first loading-path crossing is used and all crossing "
                    "counts are reported"
                ),
            }

    mesh_comparison: dict[str, Any] = {}
    for xi in FINE_LOCATIONS:
        medium_records = by_key[("medium", xi)][1]
        fine_records = by_key[("fine", xi)][1]
        if len(medium_records) != 48 or len(fine_records) != 48:
            mesh_comparison[f"{xi:.2f}"] = {
                "available": False,
                "reason": "medium or fine case incomplete",
            }
            continue
        depth_rows: dict[str, Any] = {}
        medium_tangent = tangent_signature(medium_records)
        fine_tangent = tangent_signature(fine_records)
        for depth in REPRESENTATIVE_DEPTHS_MM:
            index = _depth_step(depth)
            normal = _profile_difference(
                signature(medium_records[index]),
                signature(fine_records[index]),
                floor,
            )
            gain_medium = _signature_norm(
                signature(medium_records[index]), eta
            ) / depth
            gain_fine = _signature_norm(
                signature(fine_records[index]), eta
            ) / depth
            normal["transfer_gain_medium"] = gain_medium
            normal["transfer_gain_fine"] = gain_fine
            normal["transfer_gain_relative_difference"] = (
                abs(gain_medium - gain_fine)
                / max(abs(gain_fine), floor)
            )
            normal["tangent_gain_profile"] = _profile_difference(
                medium_tangent[index],
                fine_tangent[index],
                floor,
            )
            depth_rows[f"{depth:g}"] = normal
        medium_force = float(
            medium_records[-1]["canonical_normal_reaction_n"]
        )
        fine_force = float(fine_records[-1]["canonical_normal_reaction_n"])
        mesh_comparison[f"{xi:.2f}"] = {
            "available": True,
            "final_reaction_relative_difference": abs(
                medium_force - fine_force
            )
            / abs(fine_force),
            "final_centroids": {
                "medium": medium_records[-1]["contact"]["xi_centroid"],
                "fine": fine_records[-1]["contact"]["xi_centroid"],
            },
            "final_contact_lengths_mm": {
                "medium": medium_records[-1]["contact"]["contact_length_mm"],
                "fine": fine_records[-1]["contact"]["contact_length_mm"],
            },
            "minimum_det_f": {
                "medium": min(
                    record["minimum_det_f"] for record in medium_records
                ),
                "fine": min(
                    record["minimum_det_f"] for record in fine_records
                ),
            },
            "profiles_by_depth": depth_rows,
        }

    stabilization: dict[str, Any] = {}
    for xi, _, records in medium:
        if len(records) != 48:
            stabilization[f"{xi:.2f}"] = {"available": False}
            continue
        delta = np.asarray(
            [record["delta_n_mm"] for record in records], dtype=float
        )
        force = np.asarray(
            [record["canonical_normal_reaction_n"] for record in records],
            dtype=float,
        )
        stiffness = np.gradient(force, delta, edge_order=1)
        verified = [
            record
            for record in records
            if record["contact"]["verification"] == "VERIFIED"
        ]
        stabilization[f"{xi:.2f}"] = {
            "available": True,
            "verified_contact_step_count": len(verified),
            "centroid_drift": (
                max(record["contact"]["xi_centroid"] for record in verified)
                - min(record["contact"]["xi_centroid"] for record in verified)
                if verified
                else None
            ),
            "contact_length_range_mm": (
                [
                    min(
                        record["contact"]["contact_length_mm"]
                        for record in verified
                    ),
                    max(
                        record["contact"]["contact_length_mm"]
                        for record in verified
                    ),
                ]
                if verified
                else None
            ),
            "dF_d_delta_n_per_mm": stiffness.tolist(),
            "early_incremental_stiffness_n_per_mm": float(
                (force[15] - force[7]) / (delta[15] - delta[7])
            ),
            "late_incremental_stiffness_n_per_mm": float(
                (force[47] - force[31]) / (delta[47] - delta[31])
            ),
        }

    return {
        "medium_locations": list(MEDIUM_LOCATIONS),
        "representative_slices": slices,
        "force_conditioned_separability": force_conditioned,
        "mesh_comparison": mesh_comparison,
        "contact_stabilization": stabilization,
        "local_transfer_jacobian": {
            "definition": (
                "columns are finite-difference partial signature/partial xi "
                "and partial signature/partial delta around a nonlinear "
                "operating point"
            ),
            "status": (
                "available from location_sensitivity and tangent transfer "
                "arrays; it is not a compliance gradient"
            ),
        },
    }

