"""Deterministic run-level QC from cached longitudinal signatures."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from .optical import rms_profile_distance


@dataclass(frozen=True)
class RunQCConfig:
    """Small-sample support and two explicit optical QC thresholds."""

    minimum_template_runs: int = 3
    minimum_robust_scale_runs: int = 4
    robust_outlier_threshold: float = 3.5
    alternative_distance_ratio: float = 0.75
    minimum_alternative_preference_count: int = 3
    minimum_alternative_preference_fraction: float = 0.75
    summary_run_count: int = 15


@dataclass(frozen=True)
class _RunRepresentation:
    identity: dict[str, Any]
    low_target_force_n: float
    signatures: dict[float, np.ndarray]
    differentials: dict[float, np.ndarray]
    slope: np.ndarray | None
    force_residuals: dict[float, float]


def analyze_run_qc(
    run_rows: list[dict[str, Any]],
    run_signatures: np.ndarray,
    coverage_rows: list[dict[str, Any]],
    *,
    config: RunQCConfig = RunQCConfig(),
) -> dict[str, Any]:
    """Rank loaded runs without changing any stored experimental identity."""

    signatures = np.asarray(run_signatures, dtype=np.float64)
    if (
        signatures.ndim != 2
        or len(run_rows) != len(signatures)
        or not np.all(np.isfinite(signatures))
    ):
        raise ValueError("run_rows and finite run_signatures must have equal length")
    if config.minimum_template_runs < 1 or config.minimum_robust_scale_runs < 3:
        raise ValueError("QC sample-count requirements are invalid")
    if not 0.0 < config.alternative_distance_ratio < 1.0:
        raise ValueError("alternative_distance_ratio must be between zero and one")
    if not 0.0 < config.minimum_alternative_preference_fraction <= 1.0:
        raise ValueError(
            "minimum_alternative_preference_fraction must be in (0, 1]"
        )
    if config.minimum_alternative_preference_count < 1:
        raise ValueError("minimum_alternative_preference_count must be positive")
    if config.robust_outlier_threshold <= 0.0 or config.summary_run_count < 1:
        raise ValueError("QC threshold and summary count must be positive")
    representations = _run_representations(run_rows, signatures)
    if not representations:
        return {
            "rows": [],
            "profiles": {},
            "differential_targets_n": [],
            "config": asdict(config),
        }
    differential_targets = sorted(
        {target for run in representations for target in run.differentials}
    )
    conditions: dict[tuple[str, str, int], list[_RunRepresentation]] = defaultdict(
        list
    )
    for run in representations:
        conditions[_condition_key(run)].append(run)
    metadata_reasons = _metadata_anomaly_reasons(coverage_rows, representations)

    rows: list[dict[str, Any]] = []
    profiles: dict[str, dict[str, Any]] = {}
    for run in representations:
        same_distances, same_templates, same_slope_distance, same_slope = (
            _same_hole_comparison(run, conditions, config)
        )
        alternatives = _alternative_hole_comparisons(
            run,
            same_distances,
            same_slope_distance,
            conditions,
            config,
        )
        selected_hole, selected = _select_consistent_alternative(alternatives)
        reasons = metadata_reasons.get(_run_key(run), ())
        row = _base_qc_row(run, reasons)
        differential_values = [
            distance for distance in same_distances.values() if np.isfinite(distance)
        ]
        row.update(
            {
                "same_hole_distance_median_dn": _median_or_nan(
                    differential_values
                ),
                "same_hole_distance_max_dn": _max_or_nan(differential_values),
                "same_hole_slope_distance_dn_per_n": same_slope_distance,
                "nearest_alternative_hole": selected_hole or "",
                "nearest_alt_distance_median_dn": selected.get(
                    "differential_distance_median", float("nan")
                ),
                "recorded_hole_distance_median_dn": _median_or_nan(
                    [
                        same_distances[target]
                        for target in selected.get("differential_distances", {})
                        if np.isfinite(same_distances.get(target, float("nan")))
                    ]
                ),
                "alternative_vs_recorded_ratio": selected.get(
                    "median_component_ratio", float("nan")
                ),
                "alt_preferred_count": selected.get("preferred_count", 0),
                "comparison_count": selected.get("comparison_count", 0),
                "slope_prefers_alternative": selected.get(
                    "slope_preferred", False
                ),
                "worst_force_progression_residual_dn": _max_or_nan(
                    list(run.force_residuals.values())
                ),
                "worst_force_target_n": _largest_value_key(run.force_residuals),
            }
        )
        for target in differential_targets:
            key = _force_key(target)
            row[f"same_hole_distance_{key}_dn"] = same_distances.get(
                target, float("nan")
            )
            nearest_hole, nearest_distance = _nearest_hole_at_force(
                alternatives, target
            )
            row[f"nearest_alternative_hole_{key}"] = nearest_hole or ""
            row[f"nearest_alternative_distance_{key}_dn"] = nearest_distance
            row[f"selected_alternative_distance_{key}_dn"] = selected.get(
                "differential_distances", {}
            ).get(target, float("nan"))
        nearest_slope_hole, nearest_slope_distance = _nearest_slope_hole(alternatives)
        row["nearest_alternative_slope_hole"] = nearest_slope_hole or ""
        row["nearest_alternative_slope_distance_dn_per_n"] = nearest_slope_distance
        row["selected_alternative_slope_distance_dn_per_n"] = selected.get(
            "slope_distance", float("nan")
        )
        for target in sorted(run.force_residuals):
            row[f"force_progression_residual_{_force_key(target)}_dn"] = (
                run.force_residuals[target]
            )
        rows.append(row)
        profiles[_profile_key(run)] = {
            "run": run,
            "same_templates": same_templates,
            "same_slope_template": same_slope,
            "alternative_hole": selected_hole,
            "alternative_templates": selected.get("templates", {}),
            "alternative_slope_template": selected.get("slope_template"),
        }

    _add_robust_scores(rows, config)
    for row in rows:
        _finalize_qc_classification(row, config)
    rows.sort(key=lambda row: (_identity_sort_key(row)))
    return {
        "rows": rows,
        "profiles": profiles,
        "differential_targets_n": differential_targets,
        "config": asdict(config),
    }


def _run_representations(
    rows: list[dict[str, Any]], signatures: np.ndarray
) -> list[_RunRepresentation]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[
            (str(row["specimen_id"]), str(row["indenter"]), str(row["run_id"]))
        ].append(index)
    output = []
    for _, indices in sorted(groups.items()):
        ordered = sorted(indices, key=lambda index: float(rows[index]["target_force_n"]))
        first = rows[ordered[0]]
        by_target = {
            float(rows[index]["target_force_n"]): signatures[index].copy()
            for index in ordered
        }
        if len(by_target) != len(ordered):
            raise ValueError(f"duplicate run-force observation: {first['run_id']}")
        low_target = min(by_target)
        differentials = {
            target: signature - by_target[low_target]
            for target, signature in by_target.items()
            if target != low_target
        }
        forces = np.asarray(
            [float(rows[index]["actual_force_median_n"]) for index in ordered],
            dtype=np.float64,
        )
        values = signatures[ordered]
        slope, residuals = _slope_and_residuals(forces, values)
        identity = {
            key: first[key]
            for key in (
                "specimen_id",
                "material",
                "morphology",
                "indenter",
                "run_id",
                "hole_index",
                "repetition_index",
            )
        }
        output.append(
            _RunRepresentation(
                identity=identity,
                low_target_force_n=low_target,
                signatures=by_target,
                differentials=differentials,
                slope=slope,
                force_residuals={
                    float(rows[index]["target_force_n"]): float(residual)
                    for index, residual in zip(ordered, residuals, strict=True)
                },
            )
        )
    return output


def _slope_and_residuals(
    forces: np.ndarray, signatures: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray]:
    if len(forces) < 2:
        return None, np.full(len(forces), np.nan)
    centered = forces - np.mean(forces)
    denominator = float(np.dot(centered, centered))
    if denominator <= np.finfo(np.float64).eps:
        return None, np.full(len(forces), np.nan)
    slope = np.sum(centered[:, None] * signatures, axis=0) / denominator
    intercept = np.mean(signatures, axis=0) - slope * np.mean(forces)
    prediction = intercept[None, :] + forces[:, None] * slope[None, :]
    residuals = np.sqrt(np.mean((signatures - prediction) ** 2, axis=1))
    return slope, residuals


def _same_hole_comparison(
    run: _RunRepresentation,
    conditions: dict[tuple[str, str, int], list[_RunRepresentation]],
    config: RunQCConfig,
) -> tuple[dict[float, float], dict[float, np.ndarray], float, np.ndarray | None]:
    peers = conditions[_condition_key(run)]
    distances: dict[float, float] = {}
    templates: dict[float, np.ndarray] = {}
    for target, signature in run.differentials.items():
        candidates = [
            candidate.differentials[target]
            for candidate in peers
            if _run_key(candidate) != _run_key(run)
            and candidate.low_target_force_n == run.low_target_force_n
            and target in candidate.differentials
        ]
        template = _median_template(candidates, config.minimum_template_runs)
        if template is not None:
            templates[target] = template
            distances[target] = rms_profile_distance(signature, template)
        else:
            distances[target] = float("nan")
    slope_candidates = [
        candidate.slope
        for candidate in peers
        if _run_key(candidate) != _run_key(run) and candidate.slope is not None
    ]
    slope_template = _median_template(slope_candidates, config.minimum_template_runs)
    slope_distance = (
        rms_profile_distance(run.slope, slope_template)
        if run.slope is not None and slope_template is not None
        else float("nan")
    )
    return distances, templates, slope_distance, slope_template


def _alternative_hole_comparisons(
    run: _RunRepresentation,
    same_distances: dict[float, float],
    same_slope_distance: float,
    conditions: dict[tuple[str, str, int], list[_RunRepresentation]],
    config: RunQCConfig,
) -> dict[int, dict[str, Any]]:
    specimen = str(run.identity["specimen_id"])
    indenter = str(run.identity["indenter"])
    recorded_hole = int(run.identity["hole_index"])
    holes = sorted(
        hole
        for candidate_specimen, candidate_indenter, hole in conditions
        if candidate_specimen == specimen
        and candidate_indenter == indenter
        and hole != recorded_hole
    )
    output: dict[int, dict[str, Any]] = {}
    for hole in holes:
        peers = conditions[(specimen, indenter, hole)]
        distances: dict[float, float] = {}
        templates: dict[float, np.ndarray] = {}
        component_ratios: list[float] = []
        preferred_count = 0
        differential_preferred_count = 0
        comparison_count = 0
        for target, signature in run.differentials.items():
            candidates = [
                candidate.differentials[target]
                for candidate in peers
                if candidate.low_target_force_n == run.low_target_force_n
                and target in candidate.differentials
            ]
            template = _median_template(candidates, config.minimum_template_runs)
            if template is None:
                continue
            templates[target] = template
            distance = rms_profile_distance(signature, template)
            distances[target] = distance
            same = same_distances.get(target, float("nan"))
            if np.isfinite(same) and same > 0.0:
                ratio = distance / same
                component_ratios.append(ratio)
                comparison_count += 1
                if ratio < config.alternative_distance_ratio:
                    preferred_count += 1
                    differential_preferred_count += 1
        slope_candidates = [
            candidate.slope for candidate in peers if candidate.slope is not None
        ]
        slope_template = _median_template(
            slope_candidates, config.minimum_template_runs
        )
        slope_distance = (
            rms_profile_distance(run.slope, slope_template)
            if run.slope is not None and slope_template is not None
            else float("nan")
        )
        slope_preferred = False
        if (
            np.isfinite(slope_distance)
            and np.isfinite(same_slope_distance)
            and same_slope_distance > 0.0
        ):
            ratio = slope_distance / same_slope_distance
            component_ratios.append(ratio)
            comparison_count += 1
            slope_preferred = ratio < config.alternative_distance_ratio
            preferred_count += int(slope_preferred)
        output[hole] = {
            "differential_distances": distances,
            "differential_distance_median": _median_or_nan(list(distances.values())),
            "templates": templates,
            "slope_distance": slope_distance,
            "slope_template": slope_template,
            "median_component_ratio": _median_or_nan(component_ratios),
            "preferred_count": preferred_count,
            "differential_preferred_count": differential_preferred_count,
            "comparison_count": comparison_count,
            "slope_preferred": slope_preferred,
        }
    return output


def _select_consistent_alternative(
    alternatives: dict[int, dict[str, Any]],
) -> tuple[int | None, dict[str, Any]]:
    comparable = [
        (float(values["differential_distance_median"]), hole)
        for hole, values in alternatives.items()
        if np.isfinite(values["differential_distance_median"])
    ]
    if not comparable:
        comparable = [
            (float(values["slope_distance"]), hole)
            for hole, values in alternatives.items()
            if np.isfinite(values["slope_distance"])
        ]
    if not comparable:
        return None, {}
    _, hole = min(comparable)
    return hole, alternatives[hole]


def _nearest_hole_at_force(
    alternatives: dict[int, dict[str, Any]], target: float
) -> tuple[int | None, float]:
    candidates = [
        (float(values["differential_distances"][target]), hole)
        for hole, values in alternatives.items()
        if target in values["differential_distances"]
    ]
    if not candidates:
        return None, float("nan")
    distance, hole = min(candidates)
    return hole, distance


def _nearest_slope_hole(
    alternatives: dict[int, dict[str, Any]],
) -> tuple[int | None, float]:
    candidates = [
        (float(values["slope_distance"]), hole)
        for hole, values in alternatives.items()
        if np.isfinite(values["slope_distance"])
    ]
    if not candidates:
        return None, float("nan")
    distance, hole = min(candidates)
    return hole, distance


def _base_qc_row(
    run: _RunRepresentation, metadata_reasons: tuple[str, ...]
) -> dict[str, Any]:
    identity = run.identity
    return {
        "specimen_id": identity["specimen_id"],
        "material": identity["material"],
        "morphology": identity["morphology"],
        "indenter": identity["indenter"],
        "run_id": identity["run_id"],
        "recorded_hole": identity["hole_index"],
        "repetition_index": identity["repetition_index"],
        "lowest_target_force_n": run.low_target_force_n,
        "metadata_anomaly": bool(metadata_reasons),
        "metadata_anomaly_reason": "; ".join(metadata_reasons),
    }


def _metadata_anomaly_reasons(
    coverage_rows: list[dict[str, Any]],
    runs: list[_RunRepresentation],
) -> dict[tuple[str, str, str], tuple[str, ...]]:
    group_reasons: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    run_reasons: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in coverage_rows:
        validity = str(row["validity"])
        if validity == "valid":
            continue
        specimen = str(row["specimen_id"])
        indenter = str(row["indenter"])
        hole = int(row["hole_index"])
        repetition = int(row["repetition_index"])
        target = float(row["target_force_n"])
        if validity == "missing_run":
            group_reasons[(specimen, indenter, hole)].add(
                f"missing repetition {repetition}"
            )
            continue
        reason = f"{validity}: repetition {repetition}, {target:g} N"
        for run_id in str(row["run_ids"]).split(";"):
            if run_id:
                run_reasons[(specimen, indenter, run_id)].add(reason)
    output = {}
    for run in runs:
        key = _run_key(run)
        reasons = set(run_reasons.get(key, set()))
        reasons.update(group_reasons.get(_condition_key(run), set()))
        output[key] = tuple(sorted(reasons))
    return output


def _add_robust_scores(rows: list[dict[str, Any]], config: RunQCConfig) -> None:
    distance_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("same_hole_distance_") and key.endswith("_dn")
        }
    )
    distance_fields.append("same_hole_slope_distance_dn_per_n")
    residual_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("force_progression_residual_")
        }
    )
    groups: dict[tuple[str, str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_row_scale_key(row)].append(row)
    for group_rows in groups.values():
        for row in group_rows:
            distance_scores = {
                field: _robust_upper_score(
                    float(row[field]),
                    [float(peer[field]) for peer in group_rows],
                    config.minimum_robust_scale_runs,
                )
                for field in distance_fields
                if field in row
            }
            residual_scores = {
                field: _robust_upper_score(
                    float(row[field]),
                    [float(peer.get(field, float("nan"))) for peer in group_rows],
                    config.minimum_robust_scale_runs,
                )
                for field in residual_fields
                if field in row
            }
            row["robust_same_hole_outlier_score"] = _max_or_nan(
                list(distance_scores.values())
            )
            row["robust_same_hole_outlier_component"] = _largest_finite_key(
                distance_scores
            )
            row["force_progression_outlier_score"] = _max_or_nan(
                list(residual_scores.values())
            )


def _robust_upper_score(
    value: float, population: list[float], minimum_count: int
) -> float:
    values = np.asarray(population, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not np.isfinite(value) or len(values) < minimum_count:
        return float("nan")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= np.finfo(np.float64).eps:
        return 0.0 if value <= median + np.finfo(np.float64).eps else float("nan")
    return max(0.0, (value - median) / (1.4826 * mad))


def _finalize_qc_classification(row: dict[str, Any], config: RunQCConfig) -> None:
    comparisons = int(row["comparison_count"])
    preferred = int(row["alt_preferred_count"])
    required = max(
        config.minimum_alternative_preference_count,
        math.ceil(config.minimum_alternative_preference_fraction * comparisons),
    )
    differential_preferred = 0
    selected_hole = row["nearest_alternative_hole"]
    for key, value in row.items():
        if not key.startswith("selected_alternative_distance_") or not key.endswith(
            "_dn"
        ):
            continue
        force_key = key.removeprefix("selected_alternative_distance_").removesuffix(
            "_dn"
        )
        same = row.get(f"same_hole_distance_{force_key}_dn", float("nan"))
        if _substantially_better(value, same, config.alternative_distance_ratio):
            differential_preferred += 1
    possible_mislabel = bool(
        selected_hole != ""
        and comparisons >= config.minimum_alternative_preference_count
        and preferred >= required
        and differential_preferred >= 2
        and row["slope_prefers_alternative"]
    )
    same_score = float(row["robust_same_hole_outlier_score"])
    progression_score = float(row["force_progression_outlier_score"])
    repeat_outlier = bool(
        not possible_mislabel
        and np.isfinite(same_score)
        and same_score >= config.robust_outlier_threshold
    )
    progression_outlier = bool(
        np.isfinite(progression_score)
        and progression_score >= config.robust_outlier_threshold
    )
    row["possible_hole_mislabel"] = possible_mislabel
    row["repeat_outlier"] = repeat_outlier
    row["force_progression_anomaly"] = progression_outlier
    evidence_scores = [
        same_score if np.isfinite(same_score) else 0.0,
        progression_score if np.isfinite(progression_score) else 0.0,
        float(preferred) if possible_mislabel else 0.0,
        1.0 if row["metadata_anomaly"] else 0.0,
    ]
    row["suspect_score"] = max(evidence_scores)
    reasons = []
    if possible_mislabel:
        reasons.append(
            f"alternative hole {selected_hole} preferred in {preferred}/{comparisons} comparisons"
        )
    if repeat_outlier:
        reasons.append(
            f"same-hole outlier ({row['robust_same_hole_outlier_component']})"
        )
    if progression_outlier:
        reasons.append(
            f"force-progression residual at {float(row['worst_force_target_n']):g} N"
        )
    if row["metadata_anomaly"]:
        reasons.append("metadata anomaly")
    row["suspect_reasons"] = "; ".join(reasons)
    row["qc_interpretation"] = _interpretation(row)


def _interpretation(row: dict[str, Any]) -> str:
    if row["possible_hole_mislabel"]:
        return "possible hole mislabel; manual label inspection required"
    if row["repeat_outlier"]:
        return "repeat outlier; no consistently better alternative hole"
    if row["force_progression_anomaly"]:
        return "within-run force-progression anomaly"
    if row["metadata_anomaly"]:
        return "metadata-only anomaly"
    return "no automatic QC flag"


def _substantially_better(value: Any, reference: Any, ratio: float) -> bool:
    try:
        alternative = float(value)
        recorded = float(reference)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(alternative)
        and np.isfinite(recorded)
        and recorded > 0.0
        and alternative / recorded < ratio
    )


def _median_template(
    values: list[np.ndarray | None], minimum_count: int
) -> np.ndarray | None:
    finite = [np.asarray(value, dtype=np.float64) for value in values if value is not None]
    if len(finite) < minimum_count:
        return None
    return np.median(np.stack(finite), axis=0)


def _median_or_nan(values: list[float]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else float("nan")


def _max_or_nan(values: list[float]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if len(finite) else float("nan")


def _largest_value_key(values: dict[float, float]) -> float:
    finite = [(value, key) for key, value in values.items() if np.isfinite(value)]
    return float(max(finite)[1]) if finite else float("nan")


def _largest_finite_key(values: dict[str, float]) -> str:
    finite = [(value, key) for key, value in values.items() if np.isfinite(value)]
    return max(finite)[1] if finite else ""


def _force_key(force: float) -> str:
    return f"{force:g}".replace(".", "p") + "n"


def _condition_key(run: _RunRepresentation) -> tuple[str, str, int]:
    return (
        str(run.identity["specimen_id"]),
        str(run.identity["indenter"]),
        int(run.identity["hole_index"]),
    )


def _run_key(run: _RunRepresentation) -> tuple[str, str, str]:
    return (
        str(run.identity["specimen_id"]),
        str(run.identity["indenter"]),
        str(run.identity["run_id"]),
    )


def _profile_key(run: _RunRepresentation) -> str:
    return "|".join(_run_key(run))


def _row_scale_key(row: dict[str, Any]) -> tuple[str, str, int, float]:
    return (
        str(row["specimen_id"]),
        str(row["indenter"]),
        int(row["recorded_hole"]),
        float(row["lowest_target_force_n"]),
    )


def _identity_sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["specimen_id"]), str(row["indenter"]), str(row["run_id"])


__all__ = ["RunQCConfig", "analyze_run_qc"]
