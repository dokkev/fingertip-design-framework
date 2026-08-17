"""Artifact-only spatial-scale analysis for the localized-load OptiX study.

This module deliberately consumes persisted mechanics and optical artifacts.  It
does not import or call an FEA solver or an OptiX transport runner.  The native
P3 field remains authoritative; P3_xy results are emitted only as bridge
diagnostics.

The command writes only to ``output/.../spatial_scale_metric_study`` and never
rewrites the existing localized-load interim summaries.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import kendalltau, spearmanr

from validation import overnight_force_localized_trend as localized
from validation.common.io import atomic_write_json, strict_read_json
from validation.common.provenance import sha256_file


OUTPUT = localized.OUTPUT / "spatial_scale_metric_study"
SCALES_MM = (0.0, 0.25, 0.5, 1.0, 2.0)
PAIR_NEAR_TOLERANCES = (1.0e-12, 1.0e-8, 1.0e-6)
STATE_SCHEMA = "force-localized-spatial-scale-state-v1"
SUMMARY_SCHEMA = "force-localized-spatial-scale-summary-v1"
STUDY_ID = "force_localized_spatial_scale_tv_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _resolve_path(reference: str | Path, *, artifact_path: Path | None = None) -> Path:
    candidate = Path(str(reference))
    if candidate.is_absolute():
        return candidate
    candidates = [Path.cwd() / candidate]
    if artifact_path is not None:
        candidates.extend((artifact_path.parent / candidate, artifact_path.parent / candidate.name))
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _manifest() -> dict[str, Any]:
    manifest_path = localized.OUTPUT / "experiment_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("localized-load experiment manifest is missing")
    return localized._manifest()


def _canonical_parameters(parameters: Mapping[str, Any]) -> str:
    return json.dumps(dict(parameters), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _centers(axes: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    result = []
    for axis in axes:
        values = np.asarray(axis, dtype=float)
        if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
            raise ValueError("field axis is not finite and one-dimensional")
        if not np.all(np.diff(values) > 0.0):
            raise ValueError("field axis is not strictly increasing")
        result.append(0.5 * (values[:-1] + values[1:]))
    return tuple(result)


def _axis_spacing(axes: Sequence[np.ndarray]) -> list[float]:
    spacing = []
    for axis in axes:
        delta = np.diff(np.asarray(axis, dtype=float))
        representative = float(np.mean(delta))
        if representative <= 0.0 or not np.allclose(delta, representative, rtol=1.0e-9, atol=1.0e-12):
            raise ValueError("spatial-scale study requires uniform physical field axes")
        spacing.append(representative)
    return spacing


def _field_descriptors(field: np.ndarray, axes: Sequence[np.ndarray], result: Mapping[str, Any], raw: Mapping[str, Any] | None) -> dict[str, Any]:
    values = np.asarray(field, dtype=float)
    if values.ndim not in (2, 3) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("optical field is not a finite nonnegative 2D/3D field")
    mass = float(values.sum())
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("optical field has no positive mass")
    centers = _centers(axes)
    grids = np.meshgrid(*centers, indexing="ij")
    weights = values / mass
    coordinates = np.stack([grid.reshape(-1) for grid in grids], axis=1)
    flat_weights = weights.reshape(-1)
    centroid = flat_weights @ coordinates
    centered = coordinates - centroid
    covariance = (centered * flat_weights[:, None]).T @ centered
    spread = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    descriptors: dict[str, Any] = {
        "field_mass": mass,
        "total_transport": float(result.get("total_transport", math.nan)),
        "escaped_weight": float(result.get("escaped_weight", math.nan)),
        "absorbed_weight": float(result.get("absorbed_weight", math.nan)),
        "terminated_weight": float(result.get("terminated_weight", math.nan)),
        "centroid_mm": centroid.tolist(),
        "spread_mm": spread.tolist(),
        "covariance_mm2": covariance.tolist(),
        "ray_count": int(result.get("ray_count", 0)),
        "valid_ray_count": int(result.get("valid_ray_count", 0)),
        "terminated_ray_count": int(result.get("terminated_ray_count", 0)),
    }
    if descriptors["ray_count"]:
        descriptors["valid_ray_fraction"] = descriptors["valid_ray_count"] / descriptors["ray_count"]
        descriptors["terminated_ray_fraction"] = descriptors["terminated_ray_count"] / descriptors["ray_count"]
    else:
        descriptors["valid_ray_fraction"] = None
        descriptors["terminated_ray_fraction"] = None
    path_stats = (raw or {}).get("path_length_statistics_mm", {})
    descriptors["mean_optical_path_length_mm"] = path_stats.get("mean")
    descriptors["path_length_p95_mm"] = path_stats.get("p95")
    descriptors["escaped_ray_count"] = (raw or {}).get("escaped_ray_count")
    for key in (
        "branching_statistics",
        "escape_interaction_statistics",
        "reflection_refraction_tir_statistics",
        "termination_statistics",
        "z_fraction_away_from_central_region",
        "z_distribution",
        "processed_segment_count",
        "retained_segment_count",
    ):
        if raw is not None and key in raw:
            descriptors[key] = raw[key]
    for index, name in enumerate(("x", "y", "z")[: values.ndim]):
        descriptors[f"{name}_centroid_mm"] = float(centroid[index])
        descriptors[f"{name}_spread_mm"] = float(spread[index])
    return descriptors


def _validate_optix_artifact(
    path: Path,
    case: Mapping[str, Any],
    mechanics: Mapping[str, Any],
    mode: str,
    expected_experiment_fingerprint: str,
    expected_force_n: float,
) -> dict[str, Any]:
    metadata = strict_read_json(path)
    contract = metadata.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"missing optical contract: {path}")
    expected_source_key = "state_artifact" if mode == "PLANAR_2D" else "native_manifest"
    expected_source = mechanics.get(expected_source_key)
    checks = {
        "experiment_fingerprint": contract.get("experiment_fingerprint") == expected_experiment_fingerprint,
        "case_id": contract.get("case_id") == case["case_id"],
        "morphology_fingerprint": contract.get("morphology_fingerprint") == case["morphology_fingerprint"],
        "optical_mode": contract.get("optical_mode") == mode,
        "mechanics_source": contract.get("mechanics_source") == expected_source,
    }
    if not all(checks.values()):
        raise ValueError(f"optical provenance mismatch at {path}: {checks}")
    try:
        if not math.isclose(float(contract["force_target_n"]), expected_force_n, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"optical force contract mismatch at {path}")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid optical force contract at {path}") from exc
    field_path = _resolve_path(metadata.get("field_artifact", ""), artifact_path=path)
    if not field_path.exists() or metadata.get("field_sha256") != sha256_file(field_path):
        raise ValueError(f"optical field checksum mismatch at {path}")
    with np.load(field_path, allow_pickle=False) as archive:
        if "field" not in archive.files:
            raise ValueError(f"optical field array is missing at {field_path}")
        field = np.asarray(archive["field"], dtype=float)
        axes = tuple(np.asarray(archive[f"axis_{index}"], dtype=float) for index in range(field.ndim))
    expected_ndim = 2 if mode == "PLANAR_2D" else 3
    if field.ndim != expected_ndim or field.shape != tuple(len(axis) - 1 for axis in axes):
        raise ValueError(f"optical field dimensionality mismatch at {path}")
    raw_path = path.with_name(path.stem + "__raw.json")
    raw = strict_read_json(raw_path) if raw_path.exists() else None
    if raw is not None and raw.get("contract") != dict(contract):
        raise ValueError(f"raw optical contract mismatch at {raw_path}")
    result = metadata.get("result")
    if not isinstance(result, Mapping):
        raise ValueError(f"missing optical result metadata at {path}")
    return {
        "path": str(path),
        "field_path": str(field_path),
        "metadata": metadata,
        "contract": dict(contract),
        "field": field,
        "axes": axes,
        "result": dict(result),
        "raw": raw,
        "descriptors": _field_descriptors(field, axes, result, raw),
    }


def _eligible_states(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases = {case["case_id"]: case for case in localized._cases()}
    expected_fp = str(manifest["experiment_fingerprint"])
    force_n = float(manifest["production_force_n"])
    states: list[dict[str, Any]] = []
    for base_id in sorted({case["base_id"] for case in cases.values()}):
        for arm in ("FIXED", "VARIED"):
            state_id = f"{base_id}__{arm}"
            state_path = localized.INTERIM_OUTPUT / f"{state_id}.json"
            if not state_path.exists():
                continue
            state = strict_read_json(state_path)
            if state.get("schema") != "force-localized-interim-optix-state-v1" or state.get("status") != "PASS":
                continue
            if state.get("experiment_fingerprint") != expected_fp:
                continue
            authoritative_state_case = cases[f"{state_id}__left"]
            if (
                state.get("case_id") != state_id
                or state.get("base_id") != authoritative_state_case["base_id"]
                or state.get("arm") != authoritative_state_case["arm"]
                or _canonical_parameters(state.get("parameters", {})) != _canonical_parameters(authoritative_state_case["parameters"])
                or (state.get("morphology_fingerprint") is not None and state.get("morphology_fingerprint") != authoritative_state_case["morphology_fingerprint"])
            ):
                continue
            state_mechanics = state.get("mechanics")
            state_artifacts = state.get("optix_artifacts")
            if not isinstance(state_mechanics, Mapping) or not isinstance(state_artifacts, Mapping):
                continue
            side_artifacts: dict[str, dict[str, Any]] = {}
            valid = True
            for side in ("left", "right"):
                case = cases[f"{state_id}__{side}"]
                two_d = localized._read_case("fea2d", case)
                three_d = localized._read_case("fea3d", case)
                if two_d is None or two_d.get("status") != "PASS" or three_d is None or three_d.get("status") != "PASS":
                    valid = False
                    break
                mechanics_records = (two_d, three_d, state_mechanics.get(f"2d_{side}"), state_mechanics.get(f"3d_{side}"))
                for mechanics_record in mechanics_records:
                    if not isinstance(mechanics_record, Mapping):
                        valid = False
                        break
                    if (
                        mechanics_record.get("case_id") != case["case_id"]
                        or mechanics_record.get("base_id") != case["base_id"]
                        or mechanics_record.get("arm") != case["arm"]
                        or mechanics_record.get("side") != case["side"]
                        or mechanics_record.get("morphology_fingerprint") != case["morphology_fingerprint"]
                        or _canonical_parameters(mechanics_record.get("parameters", {})) != _canonical_parameters(case["parameters"])
                    ):
                        valid = False
                        break
                if not valid:
                    break
                if state_mechanics.get(f"2d_{side}", {}).get("case_id") != case["case_id"] or state_mechanics.get(f"3d_{side}", {}).get("case_id") != case["case_id"]:
                    valid = False
                    break
                try:
                    p2_path = _resolve_path(state_artifacts[f"P2_{side}"])
                    p3_path = _resolve_path(state_artifacts[f"P3_{side}"])
                    p2 = _validate_optix_artifact(p2_path, case, two_d, "PLANAR_2D", expected_fp, force_n)
                    p3 = _validate_optix_artifact(p3_path, case, three_d, "FULL_3D", expected_fp, force_n)
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    valid = False
                    break
                side_artifacts[side] = {"case": case, "mechanics_2d": two_d, "mechanics_3d": three_d, "P2": p2, "P3": p3}
            if not valid:
                continue
            states.append({
                "case_id": state_id,
                "base_id": base_id,
                "arm": arm,
                "parameters": dict(authoritative_state_case["parameters"]),
                "morphology_fingerprint": authoritative_state_case["morphology_fingerprint"],
                "state_artifact": str(state_path),
                "saved_metrics": {
                    "J2": state.get("J2", {}).get("normalized_redistribution_l1"),
                    "J3": state.get("J3", {}).get("normalized_redistribution_l1"),
                    "J3xy": state.get("J3_xy", {}).get("normalized_redistribution_l1"),
                },
                "sides": side_artifacts,
                "unique_physical_id": _canonical_parameters(authoritative_state_case["parameters"]),
            })
    return states


def _normalize(field: np.ndarray) -> np.ndarray:
    mass = float(np.sum(field))
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("cannot normalize a zero or non-finite field")
    return np.asarray(field, dtype=float) / mass


def _smooth(field: np.ndarray, axes: Sequence[np.ndarray], scale_mm: float) -> tuple[np.ndarray, dict[str, Any]]:
    normalized = _normalize(field)
    before = float(normalized.sum())
    if scale_mm == 0.0:
        return normalized, {"scale_mm": 0.0, "applied": False, "mass_before": before, "mass_after": before, "mass_error": 0.0, "boundary_mode": "none"}
    spacing = _axis_spacing(axes)
    sigma = [float(scale_mm / value) for value in spacing]
    filtered = gaussian_filter(normalized, sigma=sigma, mode="reflect", truncate=4.0)
    after = float(filtered.sum())
    return filtered, {
        "scale_mm": float(scale_mm),
        "applied": True,
        "sigma_bins_by_axis": sigma,
        "axis_spacing_mm": spacing,
        "mass_before": before,
        "mass_after": after,
        "mass_error": after - before,
        "boundary_mode": "reflect",
        "filter": "scipy.ndimage.gaussian_filter(truncate=4)",
    }


def _tv(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError("TV comparison requires matching field shapes")
    return 0.5 * float(np.sum(np.abs(first - second)))


def _directions(ndim: int) -> np.ndarray:
    if ndim == 2:
        angles = np.arange(8, dtype=float) * math.pi / 8.0
        return np.asarray([[math.cos(angle), math.sin(angle)] for angle in angles], dtype=float)
    if ndim == 3:
        raw = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1), (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1))
        values = np.asarray(raw, dtype=float)
        return values / np.linalg.norm(values, axis=1, keepdims=True)
    raise ValueError("sliced Wasserstein requires a 2D or 3D field")


def _sliced_wasserstein(first: np.ndarray, second: np.ndarray, axes: Sequence[np.ndarray], reference_length_mm: float) -> float:
    if first.shape != second.shape or first.ndim != len(axes):
        raise ValueError("sliced Wasserstein requires matching fields and axes")
    if reference_length_mm <= 0.0:
        raise ValueError("sliced Wasserstein reference length must be positive")
    weights_first = _normalize(first).reshape(-1)
    weights_second = _normalize(second).reshape(-1)
    centers = _centers(axes)
    grids = np.meshgrid(*centers, indexing="ij")
    coordinates = np.stack([grid.reshape(-1) for grid in grids], axis=1)
    values = []
    for direction in _directions(first.ndim):
        projection = coordinates @ direction
        order = np.argsort(projection, kind="mergesort")
        sorted_projection = projection[order]
        cumulative = np.cumsum(weights_first[order] - weights_second[order])
        values.append(float(np.sum(np.abs(cumulative[:-1]) * np.diff(sorted_projection))))
    return float(np.mean(values) / reference_length_mm)


def _p3_xy(field: np.ndarray, axes: Sequence[np.ndarray]) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    if field.ndim != 3 or len(axes) != 3:
        raise ValueError("P3_xy requires a native 3D field")
    return np.sum(field, axis=2), (axes[0], axes[1])


def _shared_reference_length(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    x_span = 0.0
    y_span = 0.0
    z_span = 0.0
    for state in states:
        for side in state["sides"].values():
            for label in ("P2", "P3"):
                axes = side[label]["axes"]
                x_span = max(x_span, float(axes[0][-1] - axes[0][0]))
                y_span = max(y_span, float(axes[1][-1] - axes[1][0]))
                if len(axes) == 3:
                    z_span = max(z_span, float(axes[2][-1] - axes[2][0]))
    if z_span <= 0.0:
        z_span = float(localized._manifest()["optix"]["extrusion_depth_mm"])
    length = math.sqrt(x_span * x_span + y_span * y_span + z_span * z_span)
    return {
        "length_mm": length,
        "definition": "one shared diagonal of the maximum physical x/y/z field envelope across the eligible exact-fingerprint artifact population; not per morphology",
        "x_span_mm": x_span,
        "y_span_mm": y_span,
        "z_span_mm": z_span,
    }


def _pair_relation(first: float, second: float, tolerance: float) -> str:
    if first == 0.0 or second == 0.0:
        return "exact_tie"
    if abs(first) <= tolerance or abs(second) <= tolerance:
        return "near_tie_uncertain"
    return "concordant" if first * second > 0.0 else "discordant"


def _rank_stats(rows: Sequence[Mapping[str, Any]], first_key: str, second_key: str) -> dict[str, Any]:
    first = np.asarray([float(row[first_key]) for row in rows], dtype=float)
    second = np.asarray([float(row[second_key]) for row in rows], dtype=float)
    if len(rows) < 3:
        return {"status": "INCONCLUSIVE", "n": len(rows)}
    exact = near = concordant = discordant = 0
    tolerance_counts = {}
    for tolerance in PAIR_NEAR_TOLERANCES:
        counts = {key: 0 for key in ("concordant", "discordant", "near_tie_uncertain", "exact_tie")}
        for index, other in itertools.combinations(range(len(rows)), 2):
            relation = _pair_relation(float(first[index] - first[other]), float(second[index] - second[other]), tolerance)
            counts[relation] += 1
        tolerance_counts[str(tolerance)] = counts
    for index, other in itertools.combinations(range(len(rows)), 2):
        relation = _pair_relation(float(first[index] - first[other]), float(second[index] - second[other]), PAIR_NEAR_TOLERANCES[-1])
        if relation == "exact_tie":
            exact += 1
        elif relation == "near_tie_uncertain":
            near += 1
        elif relation == "concordant":
            concordant += 1
        else:
            discordant += 1
    return {
        "status": "PASS",
        "n": len(rows),
        "spearman_rho": float(spearmanr(first, second).statistic),
        "kendall_tau": float(kendalltau(first, second).statistic),
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "exact_tied_pairs": exact,
        "near_tie_uncertain_pairs": near,
        "total_pairs": len(rows) * (len(rows) - 1) // 2,
        "pairwise_concordance_fraction": concordant / (concordant + discordant) if concordant + discordant else None,
        "first_dynamic_range": float(np.ptp(first)),
        "second_dynamic_range": float(np.ptp(second)),
        "primary_near_tie_tolerance": PAIR_NEAR_TOLERANCES[-1],
        "near_tie_sensitivity": tolerance_counts,
    }


def _pair_details(rows: Sequence[Mapping[str, Any]], first_key: str, second_key: str) -> dict[str, Any]:
    pairs = []
    for first, second in itertools.combinations(rows, 2):
        delta_first = float(first[first_key]) - float(second[first_key])
        delta_second = float(first[second_key]) - float(second[second_key])
        relation = _pair_relation(delta_first, delta_second, PAIR_NEAR_TOLERANCES[-1])
        pairs.append({
            "first": first["case_id"],
            "second": second["case_id"],
            "relation": relation,
            "abs_delta_first": abs(delta_first),
            "abs_delta_second": abs(delta_second),
        })
    pairs.sort(key=lambda row: row["abs_delta_first"] + row["abs_delta_second"], reverse=True)
    return {
        "strongest_inversions": [row for row in pairs if row["relation"] == "discordant"][:5],
        "strongest_preserved_order": [row for row in pairs if row["relation"] == "concordant"][:5],
        "exact_ties": [row for row in pairs if row["relation"] == "exact_tie"][:5],
        "near_ties": [row for row in pairs if row["relation"] == "near_tie_uncertain"][:5],
    }


def _state_descriptor_record(state: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in ("P2", "P3"):
        sides = state["sides"]
        left = sides["left"][label]["descriptors"]
        right = sides["right"][label]["descriptors"]
        result[label] = {
            "left": left,
            "right": right,
            "left_minus_right": {
                key: (float(left[key]) - float(right[key]))
                if isinstance(left.get(key), (int, float)) and isinstance(right.get(key), (int, float)) else None
                for key in ("field_mass", "total_transport", "escaped_weight", "x_centroid_mm", "y_centroid_mm")
                if key in left and key in right
            },
            "absolute_left_minus_right": {
                key: abs(float(left[key]) - float(right[key]))
                if isinstance(left.get(key), (int, float)) and isinstance(right.get(key), (int, float)) else None
                for key in ("field_mass", "total_transport", "escaped_weight", "x_centroid_mm", "y_centroid_mm")
                if key in left and key in right
            },
        }
    return result


def _state_metrics(state: Mapping[str, Any], reference_length_mm: float) -> dict[str, Any]:
    sides = state["sides"]
    p2_left, p2_right = sides["left"]["P2"], sides["right"]["P2"]
    p3_left, p3_right = sides["left"]["P3"], sides["right"]["P3"]
    if any(not np.array_equal(p2_left["axes"][index], p2_right["axes"][index]) for index in range(2)):
        raise ValueError(f"2D left/right grids differ for {state['case_id']}")
    if any(not np.array_equal(p3_left["axes"][index], p3_right["axes"][index]) for index in range(3)):
        raise ValueError(f"3D left/right grids differ for {state['case_id']}")
    p3xy_left, p3xy_axes = _p3_xy(p3_left["field"], p3_left["axes"])
    p3xy_right, _ = _p3_xy(p3_right["field"], p3_right["axes"])
    native_j2 = _tv(_normalize(p2_left["field"]), _normalize(p2_right["field"]))
    native_j3 = _tv(_normalize(p3_left["field"]), _normalize(p3_right["field"]))
    native_j3xy = _tv(_normalize(p3xy_left), _normalize(p3xy_right))
    scales: dict[str, Any] = {}
    for scale in SCALES_MM:
        p2_l, p2_l_diag = _smooth(p2_left["field"], p2_left["axes"], scale)
        p2_r, p2_r_diag = _smooth(p2_right["field"], p2_right["axes"], scale)
        p3_l, p3_l_diag = _smooth(p3_left["field"], p3_left["axes"], scale)
        p3_r, p3_r_diag = _smooth(p3_right["field"], p3_right["axes"], scale)
        p3xy_l, p3xy_l_diag = _smooth(p3xy_left, p3xy_axes, scale)
        p3xy_r, p3xy_r_diag = _smooth(p3xy_right, p3xy_axes, scale)
        scales[str(scale)] = {
            "scale_mm": scale,
            "J_TV_2D": _tv(p2_l, p2_r),
            "J_TV_native_3D": _tv(p3_l, p3_r),
            "J_TV_P3xy_bridge_only": _tv(p3xy_l, p3xy_r),
            "J_W_sliced_2D": _sliced_wasserstein(p2_l, p2_r, p2_left["axes"], reference_length_mm),
            "J_W_sliced_native_3D": _sliced_wasserstein(p3_l, p3_r, p3_left["axes"], reference_length_mm),
            "J_W_sliced_P3xy_bridge_only": _sliced_wasserstein(p3xy_l, p3xy_r, p3xy_axes, reference_length_mm),
            "smoothing": {"2D_left": p2_l_diag, "2D_right": p2_r_diag, "3D_left": p3_l_diag, "3D_right": p3_r_diag, "P3xy_left": p3xy_l_diag, "P3xy_right": p3xy_r_diag},
        }
    saved_j2 = state.get("saved_metrics", {}).get("J2")
    saved_j3 = state.get("saved_metrics", {}).get("J3")
    saved_j3xy = state.get("saved_metrics", {}).get("J3xy")
    return {
        "native": {
            "J_TV_native_2D": native_j2,
            "J_TV_native_3D": native_j3,
            "J_TV_P3xy_bridge_only": native_j3xy,
            "saved_metric_difference": {
                "J2": None if saved_j2 is None else native_j2 - float(saved_j2),
                "J3": None if saved_j3 is None else native_j3 - float(saved_j3),
                "J3xy": None if saved_j3xy is None else native_j3xy - float(saved_j3xy),
            },
        },
        "scales": scales,
    }


def _unique_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    selected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        selected.setdefault(str(row["unique_physical_id"]), row)
    return list(selected.values())


def _population_stats(rows: Sequence[Mapping[str, Any]], scale: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for population_name, population in (("raw_all", list(rows)), ("unique_all", _unique_rows(rows))):
        result[population_name] = {
            "eligible_state_count": len(population),
            "unique_physical_state_count": len(_unique_rows(population)),
            "fixed_count": sum(row["arm"] == "FIXED" for row in population),
            "varied_count": sum(row["arm"] == "VARIED" for row in population),
            "unique_fixed_count": len(_unique_rows([row for row in population if row["arm"] == "FIXED"])),
            "unique_varied_count": len(_unique_rows([row for row in population if row["arm"] == "VARIED"])),
            "J2_vs_native_J3": _rank_stats(population, "J2", "J3"),
            "J2_vs_P3xy_bridge": _rank_stats(population, "J2", "J3xy"),
        }
    for arm in ("FIXED", "VARIED"):
        population = [row for row in rows if row["arm"] == arm]
        unique_population = _unique_rows(population)
        result[arm] = {
            "eligible_state_count": len(population),
            "unique_physical_state_count": len(unique_population),
            "J2_vs_native_J3": _rank_stats(unique_population, "J2", "J3"),
            "J2_vs_P3xy_bridge": _rank_stats(unique_population, "J2", "J3xy"),
        }
    result["pair_details_unique"] = {
        "J2_vs_native_J3": _pair_details(_unique_rows(rows), "J2", "J3"),
        "J2_vs_P3xy_bridge": _pair_details(_unique_rows(rows), "J2", "J3xy"),
    }
    result["scale_mm"] = scale
    return result


def _classification(scale_summaries: Sequence[Mapping[str, Any]]) -> str:
    curves = [summary["unique_all"]["J2_vs_native_J3"] for summary in scale_summaries]
    if len(curves) < 2 or any(item.get("status") != "PASS" for item in curves):
        return "INCONCLUSIVE"
    rho = np.asarray([float(item["spearman_rho"]) for item in curves])
    tau = np.asarray([float(item["kendall_tau"]) for item in curves])
    if np.all(rho == rho[0]) and np.all(tau == tau[0]):
        return "SCALE_ROBUST"
    if np.all(np.diff(rho) >= 0.0) and np.all(np.diff(tau) >= 0.0) and (rho[-1] > rho[0] or tau[-1] > tau[0]):
        return "COARSE_SCALE_ALIGNMENT"
    if np.all(rho <= rho[0]) and np.all(tau <= tau[0]) and (rho[-1] < rho[0] or tau[-1] < tau[0]):
        return "NO_SCALE_RECOVERY"
    if np.any(np.diff(rho) > 0.0) and np.any(np.diff(rho) < 0.0) or np.any(np.diff(tau) > 0.0) and np.any(np.diff(tau) < 0.0):
        return "NON_MONOTONIC_SCALE_EFFECT"
    return "INCONCLUSIVE"


def _descriptor_correlations(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    unique = _unique_rows(rows)
    metrics = ("field_mass", "total_transport", "escaped_weight", "x_centroid_mm", "y_centroid_mm", "valid_ray_fraction", "mean_optical_path_length_mm")
    result: dict[str, Any] = {}
    for metric in metrics:
        p2_values = []
        p3_values = []
        for row in unique:
            p2 = row["descriptors"]["P2"]["mean"]
            p3 = row["descriptors"]["P3"]["mean"]
            if metric in p2 and metric in p3 and p2[metric] is not None and p3[metric] is not None:
                p2_values.append(float(p2[metric]))
                p3_values.append(float(p3[metric]))
        if len(p2_values) >= 3 and np.ptp(p2_values) > 0.0 and np.ptp(p3_values) > 0.0:
            result[metric] = {"n": len(p2_values), "spearman_rho": float(spearmanr(p2_values, p3_values).statistic), "kendall_tau": float(kendalltau(p2_values, p3_values).statistic), "interpretation": "descriptor rank correlation only; absolute P2/P3 scale is not equated"}
        else:
            result[metric] = {"status": "INCONCLUSIVE", "n": len(p2_values)}
    return result


def _anchor_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    requested = {
        "nominal": ["base_00_nominal__FIXED", "base_00_nominal__VARIED"],
        "candidate49_FIXED": ["base_01_candidate49__FIXED"],
        "candidate49_VARIED": ["base_01_candidate49__VARIED"],
        "base03_FIXED": ["base_03_lhs_02__FIXED"],
        "base03_VARIED": ["base_03_lhs_02__VARIED"],
    }
    by_id = {row["case_id"]: row for row in rows}
    output: dict[str, Any] = {}
    for label, ids in requested.items():
        output[label] = []
        for case_id in ids:
            row = by_id.get(case_id)
            if row is None:
                continue
            output[label].append({"case_id": case_id, "arm": row["arm"], "parameters": row["parameters"], "native": row["metrics"]["native"], "scales": {str(scale): row["metrics"]["scales"][str(scale)] for scale in SCALES_MM}})
    return output


def run() -> dict[str, Any]:
    manifest = _manifest()
    if manifest.get("production_force_n") is None:
        raise RuntimeError("production force is not frozen")
    states = _eligible_states(manifest)
    if not states:
        summary = {"schema": SUMMARY_SCHEMA, "study_id": STUDY_ID, "status": "INCONCLUSIVE", "reason": "no exact-fingerprint eligible states", "experiment_fingerprint": manifest["experiment_fingerprint"], "created_at": _now()}
        OUTPUT.mkdir(parents=True, exist_ok=True)
        atomic_write_json(OUTPUT / "spatial_scale_metric_summary.json", summary)
        return summary
    reference_length = _shared_reference_length(states)
    rows: list[dict[str, Any]] = []
    for state in states:
        descriptor = _state_descriptor_record(state)
        metrics = _state_metrics(state, float(reference_length["length_mm"]))
        row = {
            "schema": STATE_SCHEMA,
            "status": "PASS",
            "experiment_fingerprint": manifest["experiment_fingerprint"],
            "case_id": state["case_id"],
            "base_id": state["base_id"],
            "arm": state["arm"],
            "parameters": state["parameters"],
            "morphology_fingerprint": state["morphology_fingerprint"],
            "unique_physical_id": state["unique_physical_id"],
            "mechanics_artifacts": {side: {"2D": state["sides"][side]["mechanics_2d"].get("state_artifact"), "3D": state["sides"][side]["mechanics_3d"].get("native_manifest")} for side in ("left", "right")},
            "optix_artifacts": {side: {"P2": state["sides"][side]["P2"]["path"], "P3": state["sides"][side]["P3"]["path"]} for side in ("left", "right")},
            "descriptors": {"P2": {"left": descriptor["P2"]["left"], "right": descriptor["P2"]["right"], "mean": {}, "left_minus_right": descriptor["P2"]["left_minus_right"], "absolute_left_minus_right": descriptor["P2"]["absolute_left_minus_right"]}, "P3": {"left": descriptor["P3"]["left"], "right": descriptor["P3"]["right"], "mean": {}, "left_minus_right": descriptor["P3"]["left_minus_right"], "absolute_left_minus_right": descriptor["P3"]["absolute_left_minus_right"]}},
            "metrics": metrics,
        }
        for label in ("P2", "P3"):
            left = row["descriptors"][label]["left"]
            right = row["descriptors"][label]["right"]
            keys = set(left) & set(right)
            row["descriptors"][label]["mean"] = {key: (float(left[key]) + float(right[key])) * 0.5 for key in keys if isinstance(left[key], (int, float)) and isinstance(right[key], (int, float)) and math.isfinite(float(left[key])) and math.isfinite(float(right[key]))}
        rows.append(row)

    for row in rows:
        row["metrics"]["native"]["J2"] = row["metrics"]["native"].pop("J_TV_native_2D")
        row["metrics"]["native"]["J3"] = row["metrics"]["native"].pop("J_TV_native_3D")
        row["metrics"]["native"]["J3xy"] = row["metrics"]["native"].pop("J_TV_P3xy_bridge_only")
        for scale in SCALES_MM:
            scale_row = row["metrics"]["scales"][str(scale)]
            scale_row["J2"] = scale_row.pop("J_TV_2D")
            scale_row["J3"] = scale_row.pop("J_TV_native_3D")
            scale_row["J3xy"] = scale_row.pop("J_TV_P3xy_bridge_only")

    for row in rows:
        atomic_write_json(OUTPUT / f"{row['case_id']}.json", _jsonable(row))

    scale_summaries = []
    for scale in SCALES_MM:
        scale_rows = []
        for row in rows:
            scale_metrics = row["metrics"]["scales"][str(scale)]
            scale_rows.append({"case_id": row["case_id"], "arm": row["arm"], "unique_physical_id": row["unique_physical_id"], "J2": scale_metrics["J2"], "J3": scale_metrics["J3"], "J3xy": scale_metrics["J3xy"]})
        scale_summary = _population_stats(scale_rows, scale)
        scale_summaries.append(scale_summary)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "study_id": STUDY_ID,
        "label": "ARTIFACT-ONLY — SPATIAL-SCALE OPTICAL REDISTRIBUTION — NATIVE P3 AUTHORITATIVE — P3xy BRIDGE ONLY",
        "status": "PASS" if len(rows) >= 3 else "INCONCLUSIVE",
        "experiment_fingerprint": manifest["experiment_fingerprint"],
        "eligible_state_count": len(rows),
        "unique_physical_state_count": len(_unique_rows(rows)),
        "fixed_count": sum(row["arm"] == "FIXED" for row in rows),
        "varied_count": sum(row["arm"] == "VARIED" for row in rows),
        "unique_fixed_count": len(_unique_rows([row for row in rows if row["arm"] == "FIXED"])),
        "unique_varied_count": len(_unique_rows([row for row in rows if row["arm"] == "VARIED"])),
        "spatial_scales_mm": list(SCALES_MM),
        "pair_near_tolerances": list(PAIR_NEAR_TOLERANCES),
        "smoothing_contract": {"formula": "0.5 * ||G_l*p_left - G_l*p_right||_1", "normalization": "each side normalized by its own raw field mass before smoothing", "axis_sigma": "sigma_bins = l_mm / actual uniform field-axis spacing", "boundary": "reflect", "mass_policy": "record pre/post smoothing mass; no post-filter renormalization"},
        "wasserstein": {"status": "PASS_SLICED_APPROXIMATION", "metric": "deterministic sliced W1 over physical bin centers after the same declared spatial smoothing scale", "directions": {"2D": _directions(2).tolist(), "3D": _directions(3).tolist()}, "reference_length": reference_length, "native_exact_multidimensional_w1": "DEFERRED"},
        "scale_statistics": scale_summaries,
        "classification_native_J2_vs_J3": _classification(scale_summaries),
        "descriptor_correlations": _descriptor_correlations(rows),
        "anchor_values": _anchor_values(rows),
        "state_files": [str(OUTPUT / f"{row['case_id']}.json") for row in rows],
        "existing_summaries_unchanged": True,
        "fea_rerun": False,
        "optix_rerun": False,
        "created_at": _now(),
    }
    atomic_write_json(OUTPUT / "spatial_scale_metric_summary.json", _jsonable(summary))
    atomic_write_json(OUTPUT / "spatial_scale_metric_state_table.json", _jsonable({"schema": "force-localized-spatial-scale-state-table-v1", "experiment_fingerprint": manifest["experiment_fingerprint"], "records": rows}))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run the artifact-only study")
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required")
    result = run()
    print(json.dumps(_jsonable({"status": result.get("status"), "eligible_state_count": result.get("eligible_state_count", 0), "unique_physical_state_count": result.get("unique_physical_state_count", 0), "output": str(OUTPUT / 'spatial_scale_metric_summary.json')}), sort_keys=True))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
