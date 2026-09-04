"""Aggregate repeated frames into independent run and condition observations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .spatial_signature import pairwise_rows, repeat_variability


def aggregate_analysis(
    frame_rows: list[dict[str, Any]],
    frame_signatures: np.ndarray,
    *,
    hole_spacing_mm: float | None = None,
) -> dict[str, Any]:
    """Build all numerical summaries from compact per-frame extracted features."""

    groups: dict[tuple[str, str, float], list[int]] = defaultdict(list)
    for index, row in enumerate(frame_rows):
        groups[
            (str(row["specimen_id"]), str(row["run_id"]), float(row["target_force_n"]))
        ].append(index)

    run_rows: list[dict[str, Any]] = []
    run_signatures: list[np.ndarray] = []
    for _, indices in sorted(groups.items()):
        rows = [frame_rows[index] for index in indices]
        first = rows[0]
        actual_forces = _values(rows, "actual_force_n")
        valid_deformation = [row for row in rows if _as_bool(row["deformation_valid"])]
        run = {
            key: first[key]
            for key in (
                "specimen_id",
                "material",
                "morphology",
                "run_id",
                "indenter",
                "hole_index",
                "repetition_index",
                "target_force_n",
            )
        }
        run.update(
            {
                "frame_count": len(rows),
                "actual_force_mean_n": float(np.mean(actual_forces)),
                "actual_force_std_n": float(np.std(actual_forces, ddof=1))
                if len(actual_forces) > 1
                else 0.0,
                "actual_force_min_n": float(np.min(actual_forces)),
                "actual_force_max_n": float(np.max(actual_forces)),
                "actual_force_median_n": float(np.median(actual_forces)),
                "deformation_valid_frame_count": len(valid_deformation),
                "deformation_valid": len(valid_deformation) == len(rows),
                "deformation_invalid_reasons": ";".join(
                    sorted(
                        {
                            str(row["deformation_invalid_reason"])
                            for row in rows
                            if not _as_bool(row["deformation_valid"])
                        }
                    )
                ),
            }
        )
        for channel in "RGB":
            for metric in ("mae", "rms", "signed_mean"):
                name = f"optical_{metric}_{channel}_dn"
                run[name] = _median(rows, name)
            for metric in ("mae", "rms"):
                name = f"optical_{metric}_{channel}_dn_per_n"
                run[name] = _median(rows, name)
            for metric in ("ge250", "eq255"):
                name = f"saturation_{metric}_{channel}_fraction"
                run[name] = _median(rows, name)
        for metric in ("rms", "p95", "max"):
            name = f"deformation_{metric}_px"
            run[name] = _median(valid_deformation, name)
        for metric in ("mae", "rms"):
            name = f"optical_{metric}_G_dn_per_deformation_px"
            run[name] = _median(valid_deformation, name)
        run_rows.append(run)
        run_signatures.append(np.median(frame_signatures[indices], axis=0))

    signature_array = np.asarray(run_signatures, dtype=np.float64)
    condition_rows = _condition_rows(run_rows, signature_array)
    return {
        "run_rows": run_rows,
        "run_signatures": signature_array,
        "condition_rows": condition_rows,
        "pairwise_rows": pairwise_rows(
            run_rows, signature_array, hole_spacing_mm=hole_spacing_mm
        ),
        "fit_rows": _force_response_fits(run_rows),
    }


def _condition_rows(
    run_rows: list[dict[str, Any]],
    signatures: np.ndarray,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, float], list[int]] = defaultdict(list)
    for index, row in enumerate(run_rows):
        groups[
            (
                str(row["specimen_id"]),
                str(row["indenter"]),
                int(row["hole_index"]),
                float(row["target_force_n"]),
            )
        ].append(index)
    output = []
    metrics = (
        "actual_force_mean_n",
        "optical_mae_G_dn",
        "optical_rms_G_dn",
        "optical_mae_G_dn_per_n",
        "optical_rms_G_dn_per_n",
        "deformation_rms_px",
        "deformation_p95_px",
        "deformation_max_px",
        "optical_mae_G_dn_per_deformation_px",
        "optical_rms_G_dn_per_deformation_px",
    )
    for _, indices in sorted(groups.items()):
        rows = [run_rows[index] for index in indices]
        first = rows[0]
        result = {
            key: first[key]
            for key in (
                "specimen_id",
                "material",
                "morphology",
                "indenter",
                "hole_index",
                "target_force_n",
            )
        }
        result["independent_run_count"] = len(rows)
        for metric in metrics:
            values = _finite_values(rows, metric)
            result[f"{metric}_median"] = _nan_stat(values, np.median)
            result[f"{metric}_iqr"] = _nan_stat(
                values, lambda x: np.percentile(x, 75) - np.percentile(x, 25)
            )
        _, deviations = repeat_variability(signatures[indices])
        result["signature_repeat_variability_median_dn"] = float(np.median(deviations))
        result["signature_repeat_variability_iqr_dn"] = float(
            np.percentile(deviations, 75) - np.percentile(deviations, 25)
        )
        output.append(result)
    return output


def _force_response_fits(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        groups[
            (str(row["specimen_id"]), str(row["indenter"]), int(row["hole_index"]))
        ].append(row)
    output = []
    for _, rows in sorted(groups.items()):
        x = _values(rows, "actual_force_mean_n")
        y = _values(rows, "optical_mae_G_dn")
        first = rows[0]
        if len(np.unique(x)) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            prediction = slope * x + intercept
        else:
            slope = intercept = float("nan")
            prediction = np.full_like(y, np.nan)
        denominator = float(np.dot(x, x))
        origin_slope = (
            float(np.dot(x, y) / denominator) if denominator > 0.0 else float("nan")
        )
        origin_prediction = origin_slope * x
        output.append(
            {
                "specimen_id": first["specimen_id"],
                "material": first["material"],
                "morphology": first["morphology"],
                "indenter": first["indenter"],
                "hole_index": first["hole_index"],
                "independent_run_force_count": len(rows),
                "ordinary_slope_dn_per_n": float(slope),
                "ordinary_intercept_dn": float(intercept),
                "ordinary_r_squared": _r_squared(y, prediction)
                if np.all(np.isfinite(prediction))
                else float("nan"),
                "origin_slope_dn_per_n": origin_slope,
                "origin_r_squared": _r_squared(y, origin_prediction),
            }
        )
    return output


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((observed - np.mean(observed)) ** 2))
    if denominator <= np.finfo(np.float64).eps:
        return float("nan")
    return 1.0 - float(np.sum((observed - predicted) ** 2)) / denominator


def _values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = _values(rows, key)
    return values[np.isfinite(values)]


def _median(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    values = _finite_values(rows, key)
    return float(np.median(values)) if len(values) else float("nan")


def _nan_stat(values: np.ndarray, statistic: Any) -> float:
    return float(statistic(values)) if len(values) else float("nan")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


__all__ = ["aggregate_analysis"]
