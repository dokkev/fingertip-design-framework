"""Pure run aggregation and morphology-comparison metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .optical import rms_profile_distance


SATURATION_WARNING_FRACTION = 0.01


def actual_force_magnitude(fx_n: float, fy_n: float, fz_n: float) -> float:
    """Return synchronized three-axis force magnitude in newtons."""

    return float(np.sqrt(fx_n * fx_n + fy_n * fy_n + fz_n * fz_n))


def aggregate_run_force_frames(
    frame_rows: list[dict[str, Any]],
    frame_profiles: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Reduce repeated frames within each force hold to one observation."""

    profiles = np.asarray(frame_profiles, dtype=np.float64)
    if profiles.ndim != 2 or len(frame_rows) != len(profiles):
        raise ValueError("frame_rows and frame_profiles must have equal length")
    if len(profiles) and not np.all(np.isfinite(profiles)):
        raise ValueError("frame_profiles must be finite")

    groups: dict[tuple[str, str, float], list[int]] = defaultdict(list)
    for index, row in enumerate(frame_rows):
        groups[
            (
                str(row["specimen_id"]),
                str(row["run_id"]),
                float(row["target_force_n"]),
            )
        ].append(index)

    output_rows: list[dict[str, Any]] = []
    output_profiles: list[np.ndarray] = []
    for indices in (groups[key] for key in sorted(groups)):
        rows = [frame_rows[index] for index in indices]
        first = rows[0]
        force = _values(rows, "actual_force_n")
        median_profile = np.median(profiles[indices], axis=0)
        variation = np.sqrt(
            np.mean((profiles[indices] - median_profile[None, :]) ** 2, axis=1)
        )
        result = {
            key: first[key]
            for key in (
                "specimen_id",
                "material",
                "morphology",
                "run_id",
                "run_status",
                "indenter",
                "hole_index",
                "repetition_index",
                "target_force_n",
                "expected_frame_count",
                "force_tolerance_n",
                "acquisition_target_forces_n",
            )
        }
        result.update(
            {
                "frame_count": len(rows),
                "actual_force_median_n": float(np.median(force)),
                "actual_force_mean_n": float(np.mean(force)),
                "actual_force_std_n": float(np.std(force, ddof=1))
                if len(force) > 1
                else 0.0,
                "actual_force_min_n": float(np.min(force)),
                "actual_force_max_n": float(np.max(force)),
                "within_hold_optical_variation_dn": float(np.median(variation)),
                "within_hold_optical_variation_max_dn": float(np.max(variation)),
            }
        )
        for name in ("Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm"):
            result[f"{name}_median"] = float(np.median(_values(rows, name)))
        sync = np.abs(_values(rows, "camera_bota_time_delta_ms"))
        result["camera_bota_sync_median_ms"] = float(np.median(sync))
        result["camera_bota_sync_max_ms"] = float(np.max(sync))
        for channel in "RGB":
            for name in (
                f"image_mean_{channel}_dn",
                f"saturation_ge250_{channel}_fraction",
                f"saturation_eq255_{channel}_fraction",
            ):
                result[name] = float(np.median(_values(rows, name)))
        result["qc_flags"] = ";".join(_run_force_qc_flags(result))
        output_rows.append(result)
        output_profiles.append(median_profile)
    return output_rows, np.asarray(output_profiles, dtype=np.float64)


def fit_load_responses(
    run_force_rows: list[dict[str, Any]],
    run_force_profiles: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Fit signed 128-bin Green profiles against actual force for each run."""

    profiles = np.asarray(run_force_profiles, dtype=np.float64)
    if profiles.ndim != 2 or len(run_force_rows) != len(profiles):
        raise ValueError("run_force_rows and profiles must have equal length")
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(run_force_rows):
        groups[(str(row["specimen_id"]), str(row["run_id"]))].append(index)

    output_rows: list[dict[str, Any]] = []
    slopes: list[np.ndarray] = []
    for indices in (groups[key] for key in sorted(groups)):
        indices = sorted(
            indices, key=lambda index: float(run_force_rows[index]["target_force_n"])
        )
        rows = [run_force_rows[index] for index in indices]
        first = rows[0]
        targets = np.asarray(
            [float(row["target_force_n"]) for row in rows], dtype=np.float64
        )
        forces = np.asarray(
            [float(row["actual_force_median_n"]) for row in rows], dtype=np.float64
        )
        values = profiles[indices]
        try:
            slope, _, residual_rms, r_squared = fit_profile_slopes(forces, values)
        except ValueError:
            slope = np.full(values.shape[1], np.nan, dtype=np.float64)
            residual_rms = r_squared = float("nan")
        expected = _parse_force_list(str(first["acquisition_target_forces_n"]))
        flags = sorted(
            {
                flag
                for row in rows
                for flag in str(row["qc_flags"]).split(";")
                if flag
            }
        )
        missing = sorted(set(expected) - set(targets))
        if missing:
            flags.append("missing_force")
        result = {
            key: first[key]
            for key in (
                "specimen_id",
                "material",
                "morphology",
                "run_id",
                "run_status",
                "indenter",
                "hole_index",
                "repetition_index",
            )
        }
        result.update(
            {
                "available_force_states_n": ";".join(f"{value:g}" for value in targets),
                "force_state_count": len(targets),
                "actual_force_span_n": float(np.ptp(forces)),
                "S_load_DN_per_N": float(np.sqrt(np.mean(slope**2)))
                if np.all(np.isfinite(slope))
                else float("nan"),
                "fit_residual_rms_dn": residual_rms,
                "fit_r_squared": r_squared,
                "2_to_5_DN_per_N": _finite_difference(
                    targets, forces, values, 2.0, 5.0
                ),
                "5_to_10_DN_per_N": _finite_difference(
                    targets, forces, values, 5.0, 10.0
                ),
                "10_to_15_DN_per_N": _finite_difference(
                    targets, forces, values, 10.0, 15.0
                ),
                "2_to_15_DN_per_N": _finite_difference(
                    targets, forces, values, 2.0, 15.0
                ),
                "qc_flags": ";".join(sorted(set(flags))),
                "S_OM_DN_per_mm": float("nan"),
                "S_OM_status": "unavailable: no trusted mechanical deformation input",
            }
        )
        output_rows.append(result)
        slopes.append(slope)
    return output_rows, np.asarray(slopes, dtype=np.float64)


def fit_profile_slopes(
    actual_forces_n: np.ndarray,
    profiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit ``s(v,F)=a(v)+b(v)F`` and return b, a, residual RMS, and R²."""

    force = np.asarray(actual_forces_n, dtype=np.float64)
    values = np.asarray(profiles, dtype=np.float64)
    if (
        force.ndim != 1
        or values.ndim != 2
        or len(force) != len(values)
        or len(force) < 2
        or not np.all(np.isfinite(force))
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("at least two finite force/profile observations are required")
    centered = force - np.mean(force)
    denominator = float(np.dot(centered, centered))
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("actual forces must contain at least two distinct values")
    slope = np.sum(centered[:, None] * values, axis=0) / denominator
    intercept = np.mean(values, axis=0) - slope * np.mean(force)
    prediction = intercept[None, :] + force[:, None] * slope[None, :]
    residual_sum = float(np.sum((values - prediction) ** 2))
    total_sum = float(np.sum((values - np.mean(values, axis=0)) ** 2))
    r_squared = float("nan") if total_sum <= 0.0 else 1.0 - residual_sum / total_sum
    residual_rms = float(np.sqrt(np.mean((values - prediction) ** 2)))
    return slope, intercept, residual_rms, r_squared


def spatial_metrics(
    run_rows: list[dict[str, Any]],
    slope_profiles: np.ndarray,
    *,
    hole_spacing_mm: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return neighboring-hole distances and independent-run variability."""

    profiles = np.asarray(slope_profiles, dtype=np.float64)
    if profiles.ndim != 2 or len(run_rows) != len(profiles):
        raise ValueError("run_rows and slope_profiles must have equal length")
    groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(run_rows):
        if not np.all(np.isfinite(profiles[index])):
            continue
        groups[
            (
                str(row["specimen_id"]),
                str(row["indenter"]),
                int(row["hole_index"]),
            )
        ].append(index)

    templates: dict[tuple[str, str, int], np.ndarray] = {}
    variability_rows: list[dict[str, Any]] = []
    for key, indices in sorted(groups.items()):
        template = np.median(profiles[indices], axis=0)
        templates[key] = template
        estimable = len(indices) >= 2
        for index in indices:
            row = run_rows[index]
            variability_rows.append(
                {
                    key: row[key]
                    for key in (
                        "specimen_id",
                        "material",
                        "morphology",
                        "run_id",
                        "run_status",
                        "indenter",
                        "hole_index",
                        "repetition_index",
                    )
                }
                | {
                    "same_location_repeat_count": len(indices),
                    "repeat_variability_DN_per_N": rms_profile_distance(
                        profiles[index], template
                    )
                    if estimable
                    else float("nan"),
                }
            )

    grouped_indices = {index for indices in groups.values() for index in indices}
    for index, row in enumerate(run_rows):
        if index in grouped_indices:
            continue
        variability_rows.append(
            {
                key: row[key]
                for key in (
                    "specimen_id",
                    "material",
                    "morphology",
                    "run_id",
                    "run_status",
                    "indenter",
                    "hole_index",
                    "repetition_index",
                )
            }
            | {
                "same_location_repeat_count": 0,
                "repeat_variability_DN_per_N": float("nan"),
            }
        )

    neighboring_rows: list[dict[str, Any]] = []
    specimen_indenters = sorted({(key[0], key[1]) for key in templates})
    for specimen_id, indenter in specimen_indenters:
        holes = sorted(
            hole
            for specimen, candidate_indenter, hole in templates
            if specimen == specimen_id and candidate_indenter == indenter
        )
        for hole_i, hole_j in zip(holes, holes[1:]):
            if hole_j != hole_i + 1:
                continue
            first_indices = groups[(specimen_id, indenter, hole_i)]
            second_indices = groups[(specimen_id, indenter, hole_j)]
            identity = run_rows[first_indices[0]]
            distance = rms_profile_distance(
                templates[(specimen_id, indenter, hole_i)],
                templates[(specimen_id, indenter, hole_j)],
            )
            neighboring_rows.append(
                {
                    "specimen_id": specimen_id,
                    "material": identity["material"],
                    "morphology": identity["morphology"],
                    "indenter": indenter,
                    "hole_i": hole_i,
                    "hole_j": hole_j,
                    "run_count_i": len(first_indices),
                    "run_count_j": len(second_indices),
                    "D_neighbor_DN_per_N": distance,
                    "hole_spacing_mm": "" if hole_spacing_mm is None else hole_spacing_mm,
                    "D_neighbor_DN_per_N_per_mm": ""
                    if hole_spacing_mm is None
                    else distance / hole_spacing_mm,
                }
            )
    return neighboring_rows, variability_rows


def morphology_metrics(
    run_rows: list[dict[str, Any]],
    neighboring_rows: list[dict[str, Any]],
    variability_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize headline quantities without combining them into one score."""

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        groups[(str(row["material"]), str(row["morphology"]), str(row["indenter"]))].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        material, morphology, indenter = key
        load = _finite([row["S_load_DN_per_N"] for row in rows])
        neighbor = _finite(
            row["D_neighbor_DN_per_N"]
            for row in neighboring_rows
            if _summary_key(row) == key
        )
        variation = _finite(
            row["repeat_variability_DN_per_N"]
            for row in variability_rows
            if _summary_key(row) == key
        )
        d_median = _median_or_nan(neighbor)
        w_median = _median_or_nan(variation)
        output.append(
            {
                "material": material,
                "morphology": morphology,
                "indenter": indenter,
                "physical_specimen_count": len({str(row["specimen_id"]) for row in rows}),
                "recorded_run_count": len(rows),
                "independent_run_count": len(load),
                "S_load_median_DN_per_N": _median_or_nan(load),
                "S_load_IQR_DN_per_N": _iqr_or_nan(load),
                "neighbor_pair_count": len(neighbor),
                "D_neighbor_median_DN_per_N": d_median,
                "D_neighbor_IQR_DN_per_N": _iqr_or_nan(neighbor),
                "repeat_variability_run_count": len(variation),
                "W_median_DN_per_N": w_median,
                "W_IQR_DN_per_N": _iqr_or_nan(variation),
                "D_neighbor_over_W": d_median / w_median
                if np.isfinite(w_median) and w_median > 0.0
                else float("nan"),
                "S_OM_DN_per_mm": float("nan"),
                "S_OM_status": "unavailable: no trusted mechanical deformation input",
            }
        )
    return output


def _run_force_qc_flags(row: dict[str, Any]) -> list[str]:
    flags = []
    if str(row["run_status"]) != "complete":
        flags.append("run_not_complete")
    if int(row["frame_count"]) != int(row["expected_frame_count"]):
        flags.append("wrong_frame_count")
    tolerance = float(row["force_tolerance_n"])
    target = float(row["target_force_n"])
    if abs(float(row["actual_force_median_n"]) - target) > tolerance:
        flags.append("force_out_of_band")
    if float(row["actual_force_max_n"]) - float(row["actual_force_min_n"]) > tolerance:
        flags.append("large_force_spread")
    if any(
        float(row[f"saturation_ge250_{channel}_fraction"])
        > SATURATION_WARNING_FRACTION
        for channel in "RGB"
    ):
        flags.append("saturation")
    return flags


def _finite_difference(
    targets: np.ndarray,
    forces: np.ndarray,
    profiles: np.ndarray,
    low_target: float,
    high_target: float,
) -> float:
    low = np.flatnonzero(np.isclose(targets, low_target))
    high = np.flatnonzero(np.isclose(targets, high_target))
    if len(low) != 1 or len(high) != 1:
        return float("nan")
    denominator = float(forces[high[0]] - forces[low[0]])
    if abs(denominator) <= np.finfo(np.float64).eps:
        return float("nan")
    return rms_profile_distance(profiles[high[0]], profiles[low[0]]) / abs(
        denominator
    )


def _parse_force_list(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(";") if item)


def _values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _finite(values: Any) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def _median_or_nan(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def _iqr_or_nan(values: np.ndarray) -> float:
    if not len(values):
        return float("nan")
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def _summary_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["material"]), str(row["morphology"]), str(row["indenter"])


__all__ = [
    "SATURATION_WARNING_FRACTION",
    "actual_force_magnitude",
    "aggregate_run_force_frames",
    "fit_load_responses",
    "fit_profile_slopes",
    "morphology_metrics",
    "spatial_metrics",
]
