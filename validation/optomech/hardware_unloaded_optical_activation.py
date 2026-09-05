"""Measure unloaded-referenced optical activation in the hardware datasets.

This validation reuses the production optical-strip extraction functions but
writes only a separate read-only study bundle. It does not modify raw data,
production metrics, morphology summaries, or Figure 5.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from itertools import combinations
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.analysis.dataset import index_session  # noqa: E402
from experiments.analysis.metrics import actual_force_magnitude  # noqa: E402
from experiments.analysis.optical import (  # noqa: E402
    PROFILE_BINS,
    calibrate_optical_strip,
    load_rgb,
    longitudinal_green_profile,
    rms_profile_distance,
    temporal_median_rgb,
)
from figures.figure5.config import (  # noqa: E402
    ANALYSIS_ROOTS,
    MORPHOLOGY_CONDITIONS,
)


OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "output" / "validation" / "hardware_unloaded_optical_activation"
)
TARGET_FORCES_N = (2.0, 5.0, 10.0, 15.0)
INDENTERS = ("sphere_10mm", "sphere_30mm")
HOLE_TO_CONTACT_X_MM = {
    1: 0.0,
    2: 10.0,
    3: 20.0,
    4: 30.0,
    5: 40.0,
    6: 50.0,
}
MORPHOLOGY_LABELS = {
    "baseline": "Baseline",
    "flat_opt": "Flat-opt",
    "angled_opt": "Angled-opt",
}
MATERIAL_LABELS = {"solaris": "Solaris", "dragon_skin": "Dragon Skin"}
INDENTER_LABELS = {"sphere_10mm": "10 mm sphere", "sphere_30mm": "30 mm sphere"}
COLORS = {"baseline": "#4c78a8", "flat_opt": "#f58518", "angled_opt": "#54a24b"}
MARKERS = {"baseline": "o", "flat_opt": "s", "angled_opt": "D"}
CSV_FLOAT_FORMAT = ".9g"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: format(value, CSV_FLOAT_FORMAT)
                    if isinstance(value, float)
                    else value
                    for key, value in row.items()
                }
            )


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _measurement_time_s(measurements: Any, source: str) -> float:
    if source == "camera_host_time_s":
        value = _finite_float(measurements.get("camera_host_time_s"))
    elif source == "camera_device_timestamp_ms":
        raw = _finite_float(measurements.get("camera_device_timestamp_ms"))
        value = None if raw is None else raw / 1000.0
    else:
        raise ValueError(f"unknown timestamp source: {source}")
    if value is None:
        raise ValueError(f"missing finite {source}")
    return value


def _select_time_source(frames: Iterable[Any]) -> str:
    records = list(frames)
    if all(
        _finite_float(frame.measurements.get("camera_host_time_s")) is not None
        for frame in records
    ):
        return "camera_host_time_s"
    if all(
        _finite_float(frame.measurements.get("camera_device_timestamp_ms")) is not None
        for frame in records
    ):
        return "camera_device_timestamp_ms"
    raise RuntimeError(
        "session has no timestamp source shared by loaded and unloaded data"
    )


def _force_n(measurements: Any) -> float:
    return actual_force_magnitude(
        float(measurements["Fx_N"]),
        float(measurements["Fy_N"]),
        float(measurements["Fz_N"]),
    )


def _existing_summary(condition: Any, specimen_id: str) -> dict[str, Any]:
    root = ANALYSIS_ROOTS[condition.material] / "raw_data_summary"
    dataset_matches = [
        row
        for row in _read_csv(root / "dataset_summary.csv")
        if row["specimen_id"] == specimen_id
    ]
    if len(dataset_matches) != 1:
        raise RuntimeError(f"missing unique dataset summary for {specimen_id}")
    unloaded_rows = [
        row
        for row in _read_csv(root / "unloaded_summary.csv")
        if row["specimen_id"] == specimen_id
    ]
    qc_rows = [
        row
        for row in _read_csv(root / "qc_summary.csv")
        if row["specimen_id"] in {specimen_id, "all"}
    ]
    dataset = dataset_matches[0]
    geometry_messages = [
        row["message"]
        for row in qc_rows
        if row["qc_code"] == "suspicious_camera_geometry_change"
    ]
    if "suspicious camera geometry" in dataset["coverage_warnings"].lower():
        geometry_messages.append(dataset["coverage_warnings"])
    return {
        "dataset": dataset,
        "unloaded_rows": unloaded_rows,
        "qc_rows": qc_rows,
        "geometry_messages": tuple(dict.fromkeys(geometry_messages)),
    }


def _group_session_frames(
    index: Any,
) -> tuple[dict[Path, list[Any]], dict[tuple[str, float], list[Any]]]:
    unloaded: dict[Path, list[Any]] = defaultdict(list)
    loaded: dict[tuple[str, float], list[Any]] = defaultdict(list)
    for frame in index.frames:
        if frame.run is None:
            unloaded[frame.segment_path].append(frame)
        elif (
            frame.target_force_n in TARGET_FORCES_N and frame.run.indenter in INDENTERS
        ):
            loaded[(frame.run.run_id, float(frame.target_force_n))].append(frame)
    for frames in (*unloaded.values(), *loaded.values()):
        frames.sort(key=lambda frame: int(frame.measurements["frame_index"]))
    return unloaded, loaded


def _valid_runs(
    index: Any, loaded: dict[tuple[str, float], list[Any]]
) -> tuple[list[Any], list[str]]:
    expected_frames = index.session.force_sequence.expected_record_frame_count
    valid = []
    excluded = []
    for run in index.runs:
        if run.indenter not in INDENTERS:
            continue
        reasons = []
        if run.status != "complete":
            reasons.append(f"status={run.status}")
        for force in TARGET_FORCES_N:
            count = len(loaded.get((run.run_id, force), ()))
            if count != expected_frames:
                reasons.append(f"{force:g}N_frames={count},expected={expected_frames}")
        if reasons:
            excluded.append(f"{run.run_id}: {', '.join(reasons)}")
        else:
            valid.append(run)
    return valid, excluded


def _run_qc_flags(existing: dict[str, Any], run_id: str) -> tuple[str, ...]:
    flags = []
    for row in existing["qc_rows"]:
        if row["message"].startswith(f"{run_id} "):
            flags.append(row["qc_code"])
    return tuple(sorted(set(flags)))


def _extract_session(condition: Any) -> dict[str, Any]:
    assert condition.session_path is not None
    index = index_session(condition.session_path, expected_repetitions=5)
    if (
        index.session.material != condition.material
        or index.session.morphology != condition.morphology
    ):
        raise RuntimeError(
            f"Figure 5 condition metadata mismatch: {condition.session_path}"
        )
    existing = _existing_summary(condition, index.session.specimen_id)
    unloaded_groups, loaded_groups = _group_session_frames(index)
    if not unloaded_groups:
        raise RuntimeError(f"{index.specimen_id} has no unloaded capture")
    all_unloaded_frames = [
        frame for path in sorted(unloaded_groups) for frame in unloaded_groups[path]
    ]
    time_source = _select_time_source(index.frames)
    unloaded_images = [load_rgb(frame.rgb_path) for frame in all_unloaded_frames]
    session_reference = temporal_median_rgb(unloaded_images)
    strip = calibrate_optical_strip(session_reference)

    capture_rows: list[dict[str, Any]] = []
    capture_profiles: dict[str, np.ndarray] = {}
    capture_profile_records: list[dict[str, Any]] = []
    for path in sorted(unloaded_groups):
        frames = unloaded_groups[path]
        profiles = []
        times = []
        host_times = []
        device_times = []
        forces = []
        for frame in frames:
            image = load_rgb(frame.rgb_path)
            profile, _ = longitudinal_green_profile(image, strip, bins=PROFILE_BINS)
            profiles.append(profile)
            times.append(_measurement_time_s(frame.measurements, time_source))
            host = _finite_float(frame.measurements.get("camera_host_time_s"))
            device = _finite_float(frame.measurements.get("camera_device_timestamp_ms"))
            if host is not None:
                host_times.append(host)
            if device is not None:
                device_times.append(device)
            forces.append(_force_n(frame.measurements))
        capture_id = path.name
        median_profile = np.median(np.asarray(profiles), axis=0)
        capture_profiles[capture_id] = median_profile
        capture_profile_records.append(
            {
                "specimen_id": index.specimen_id,
                "material": condition.material,
                "morphology": condition.morphology,
                "capture_id": capture_id,
                "profile": median_profile,
            }
        )
        capture_rows.append(
            {
                "specimen_id": index.specimen_id,
                "material": condition.material,
                "morphology": condition.morphology,
                "capture_id": capture_id,
                "frame_count": len(frames),
                "time_source": time_source,
                "capture_time_s": float(np.median(times)),
                "camera_host_time_start_s": min(host_times)
                if host_times
                else float("nan"),
                "camera_host_time_median_s": float(np.median(host_times))
                if host_times
                else float("nan"),
                "camera_host_time_end_s": max(host_times)
                if host_times
                else float("nan"),
                "camera_device_timestamp_start_ms": min(device_times)
                if device_times
                else float("nan"),
                "camera_device_timestamp_median_ms": float(np.median(device_times))
                if device_times
                else float("nan"),
                "camera_device_timestamp_end_ms": max(device_times)
                if device_times
                else float("nan"),
                "actual_force_median_n": float(np.median(forces)),
                "actual_force_std_n": float(np.std(forces, ddof=1))
                if len(forces) > 1
                else 0.0,
                "actual_force_min_n": float(np.min(forces)),
                "actual_force_max_n": float(np.max(forces)),
            }
        )

    valid_runs, excluded_runs = _valid_runs(index, loaded_groups)
    run_rows: list[dict[str, Any]] = []
    pairing_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    loaded_profile_records: list[dict[str, Any]] = []
    for run in valid_runs:
        run_frames = [
            frame
            for force in TARGET_FORCES_N
            for frame in loaded_groups[(run.run_id, force)]
        ]
        run_time = float(
            np.median(
                [
                    _measurement_time_s(frame.measurements, time_source)
                    for frame in run_frames
                ]
            )
        )
        paired = min(
            capture_rows,
            key=lambda row: (
                abs(float(row["capture_time_s"]) - run_time),
                str(row["capture_id"]),
            ),
        )
        signed_delta = float(paired["capture_time_s"]) - run_time
        relation = (
            "before"
            if signed_delta < 0.0
            else ("after" if signed_delta > 0.0 else "simultaneous")
        )
        pairing_rows.append(
            {
                "specimen_id": index.specimen_id,
                "material": condition.material,
                "morphology": condition.morphology,
                "run_id": run.run_id,
                "indenter": run.indenter,
                "hole_index": run.hole_index,
                "X_contact_mm": HOLE_TO_CONTACT_X_MM[run.hole_index],
                "repetition_index": run.repetition_index,
                "timestamp_source": time_source,
                "loaded_run_time_s": run_time,
                "paired_unloaded_capture_id": paired["capture_id"],
                "unloaded_capture_time_s": paired["capture_time_s"],
                "unloaded_time_delta_s": signed_delta,
                "unloaded_time_absolute_delta_s": abs(signed_delta),
                "unloaded_before_or_after": relation,
            }
        )
        force_rows = []
        for force in TARGET_FORCES_N:
            frames = loaded_groups[(run.run_id, force)]
            profiles = []
            forces = []
            for frame in frames:
                profile, _ = longitudinal_green_profile(
                    load_rgb(frame.rgb_path), strip, bins=PROFILE_BINS
                )
                profiles.append(profile)
                forces.append(_force_n(frame.measurements))
            loaded_profile = np.median(np.asarray(profiles), axis=0)
            changes = {
                capture_id: rms_profile_distance(loaded_profile, profile)
                for capture_id, profile in capture_profiles.items()
            }
            paired_profile = capture_profiles[str(paired["capture_id"])]
            delta_profile = loaded_profile - paired_profile
            primary = rms_profile_distance(loaded_profile, paired_profile)
            sensitivity_min = min(changes.values())
            sensitivity_max = max(changes.values())
            sensitivity_range = sensitivity_max - sensitivity_min
            flags = _run_qc_flags(existing, run.run_id)
            row = {
                "specimen_id": index.specimen_id,
                "material": condition.material,
                "morphology": condition.morphology,
                "run_id": run.run_id,
                "indenter": run.indenter,
                "hole_index": run.hole_index,
                "X_contact_mm": HOLE_TO_CONTACT_X_MM[run.hole_index],
                "repetition_index": run.repetition_index,
                "target_force_n": force,
                "actual_force_n": float(np.median(forces)),
                "frame_count": len(frames),
                "paired_unloaded_capture_id": paired["capture_id"],
                "loaded_run_time_s": run_time,
                "unloaded_capture_time_s": paired["capture_time_s"],
                "unloaded_time_delta_s": signed_delta,
                "unloaded_time_absolute_delta_s": abs(signed_delta),
                "unloaded_before_or_after": relation,
                "optical_change_rms_DN": primary,
                "reference_sensitivity_range_DN": sensitivity_range,
                "status": "valid" if not flags else f"valid;{';'.join(flags)}",
                "QC_flags": ";".join(flags),
            }
            run_rows.append(row)
            force_rows.append(row)
            sensitivity_rows.append(
                {
                    "specimen_id": index.specimen_id,
                    "material": condition.material,
                    "morphology": condition.morphology,
                    "run_id": run.run_id,
                    "indenter": run.indenter,
                    "hole_index": run.hole_index,
                    "X_contact_mm": HOLE_TO_CONTACT_X_MM[run.hole_index],
                    "repetition_index": run.repetition_index,
                    "target_force_n": force,
                    "paired_unloaded_capture_id": paired["capture_id"],
                    "paired_optical_change_rms_DN": primary,
                    "minimum_optical_change_rms_DN": sensitivity_min,
                    "maximum_optical_change_rms_DN": sensitivity_max,
                    "reference_sensitivity_range_DN": sensitivity_range,
                    "unloaded_capture_count": len(capture_profiles),
                    "capture_ids": ";".join(changes),
                    "optical_change_by_capture_DN": ";".join(
                        f"{capture_id}:{value:.9g}"
                        for capture_id, value in changes.items()
                    ),
                }
            )
            loaded_profile_records.append(
                {
                    "specimen_id": index.specimen_id,
                    "material": condition.material,
                    "morphology": condition.morphology,
                    "run_id": run.run_id,
                    "indenter": run.indenter,
                    "hole_index": run.hole_index,
                    "repetition_index": run.repetition_index,
                    "target_force_n": force,
                    "loaded_profile": loaded_profile,
                    "delta_profile": delta_profile,
                }
            )
        activation = [float(row["optical_change_rms_DN"]) for row in force_rows]
        monotonic = bool(np.all(np.diff(activation) >= 0.0))
        for row in force_rows:
            row["activation_monotonic"] = monotonic

    pair_distances = [
        rms_profile_distance(capture_profiles[first], capture_profiles[second])
        for first, second in combinations(capture_profiles, 2)
    ]
    stability = _stability_rows(
        condition,
        index.specimen_id,
        capture_rows,
        capture_profiles,
        pair_distances,
        existing,
    )
    return {
        "index": index,
        "condition": condition,
        "existing": existing,
        "capture_rows": capture_rows,
        "capture_profiles": capture_profile_records,
        "run_rows": run_rows,
        "pairing_rows": pairing_rows,
        "sensitivity_rows": sensitivity_rows,
        "loaded_profiles": loaded_profile_records,
        "stability_rows": stability,
        "unloaded_pair_distances": pair_distances,
        "excluded_runs": excluded_runs,
    }


def _span(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.ptp(values)) if values else float("nan")


def _stability_rows(
    condition: Any,
    specimen_id: str,
    capture_rows: list[dict[str, Any]],
    capture_profiles: dict[str, np.ndarray],
    pair_distances: list[float],
    existing: dict[str, Any],
) -> list[dict[str, Any]]:
    dataset = existing["dataset"]
    unloaded = existing["unloaded_rows"]
    pair_median = float(np.median(pair_distances)) if pair_distances else float("nan")
    pair_max = float(np.max(pair_distances)) if pair_distances else float("nan")
    common = {
        "specimen_id": specimen_id,
        "material": condition.material,
        "morphology": condition.morphology,
        "unloaded_capture_count": len(capture_rows),
        "pairwise_unloaded_distance_median_DN": pair_median,
        "pairwise_unloaded_distance_max_DN": pair_max,
        "existing_unloaded_centroid_span_px": float(
            dataset["unloaded_geometry_centroid_span_px"]
        ),
        "existing_orientation_span_deg": _span(
            unloaded, "optical_region_orientation_deg"
        ),
        "existing_longitudinal_extent_span_px": _span(
            unloaded, "optical_region_longitudinal_extent_px"
        ),
        "existing_transverse_extent_span_px": _span(
            unloaded, "optical_region_transverse_extent_px"
        ),
        "existing_optical_area_span_px": _span(unloaded, "optical_region_area_px"),
        "existing_geometry_warning": "; ".join(existing["geometry_messages"]),
        "existing_dataset_warnings": dataset["coverage_warnings"],
    }
    output = []
    by_id = {row["capture_id"]: row for row in capture_rows}
    if len(capture_profiles) == 1:
        capture_id = next(iter(capture_profiles))
        row = by_id[capture_id]
        output.append(
            {
                **common,
                "record_type": "capture",
                "capture_id_a": capture_id,
                "capture_id_b": "",
                "pairwise_unloaded_profile_distance_DN": float("nan"),
                "capture_a_frame_count": row["frame_count"],
                "capture_a_time_s": row["capture_time_s"],
                "capture_a_actual_force_median_n": row["actual_force_median_n"],
                "capture_b_frame_count": "",
                "capture_b_time_s": "",
                "capture_b_actual_force_median_n": "",
                "record_status": "single_capture_pairwise_stability_unavailable",
            }
        )
        return output
    for capture_id, row in by_id.items():
        output.append(
            {
                **common,
                "record_type": "capture",
                "capture_id_a": capture_id,
                "capture_id_b": "",
                "pairwise_unloaded_profile_distance_DN": "",
                "capture_a_frame_count": row["frame_count"],
                "capture_a_time_s": row["capture_time_s"],
                "capture_a_actual_force_median_n": row["actual_force_median_n"],
                "capture_b_frame_count": "",
                "capture_b_time_s": "",
                "capture_b_actual_force_median_n": "",
                "record_status": "capture_profile_preserved_in_npz",
            }
        )
    for first, second in combinations(capture_profiles, 2):
        output.append(
            {
                **common,
                "record_type": "pair",
                "capture_id_a": first,
                "capture_id_b": second,
                "pairwise_unloaded_profile_distance_DN": rms_profile_distance(
                    capture_profiles[first], capture_profiles[second]
                ),
                "capture_a_frame_count": by_id[first]["frame_count"],
                "capture_a_time_s": by_id[first]["capture_time_s"],
                "capture_a_actual_force_median_n": by_id[first][
                    "actual_force_median_n"
                ],
                "capture_b_frame_count": by_id[second]["frame_count"],
                "capture_b_time_s": by_id[second]["capture_time_s"],
                "capture_b_actual_force_median_n": by_id[second][
                    "actual_force_median_n"
                ],
                "record_status": "measured_pair",
            }
        )
    return output


def _aggregate_morphologies(
    run_rows: list[dict[str, Any]],
    extracted: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[
            (row["material"], row["morphology"], row["indenter"], row["target_force_n"])
        ].append(row)

    stability_by_specimen = {}
    for session in extracted:
        capture_rows = session["capture_rows"]
        pair_distances = session["unloaded_pair_distances"]
        existing = session["existing"]
        stability_by_specimen[session["index"].specimen_id] = {
            "capture_count": len(capture_rows),
            "pair_max": float(np.max(pair_distances))
            if pair_distances
            else float("nan"),
            "geometry_warning": bool(existing["geometry_messages"]),
            "geometry_warning_text": "; ".join(existing["geometry_messages"]),
        }

    rows = []
    status_by_specimen: dict[str, str] = {}
    for key in sorted(grouped):
        material, morphology, indenter, force = key
        values = grouped[key]
        optical = np.asarray([row["optical_change_rms_DN"] for row in values])
        sensitivity = np.asarray(
            [row["reference_sensitivity_range_DN"] for row in values]
        )
        specimen_id = str(values[0]["specimen_id"])
        reference_assessed = stability_by_specimen[specimen_id]["capture_count"] >= 2
        monotonic_by_run = {
            row["run_id"]: bool(row["activation_monotonic"]) for row in values
        }
        rows.append(
            {
                "specimen_id": specimen_id,
                "material": material,
                "morphology": morphology,
                "indenter": indenter,
                "target_force_n": force,
                "valid_run_count": len(values),
                "optical_change_median_DN": float(np.median(optical)),
                "optical_change_q1_DN": float(np.percentile(optical, 25)),
                "optical_change_q3_DN": float(np.percentile(optical, 75)),
                "optical_change_IQR_DN": float(
                    np.percentile(optical, 75) - np.percentile(optical, 25)
                ),
                "baseline_optical_change_median_DN": float("nan"),
                "baseline_gain_percent": float("nan"),
                "monotonic_run_count": sum(monotonic_by_run.values()),
                "monotonic_run_fraction": float(
                    np.mean(list(monotonic_by_run.values()))
                ),
                "unloaded_reference_sensitivity_median_DN": (
                    float(np.median(sensitivity))
                    if reference_assessed
                    else float("nan")
                ),
                "unloaded_reference_sensitivity_p95_DN": (
                    float(np.percentile(sensitivity, 95))
                    if reference_assessed
                    else float("nan")
                ),
                "unloaded_reference_sensitivity_max_DN": (
                    float(np.max(sensitivity)) if reference_assessed else float("nan")
                ),
                "unloaded_geometry_warning": stability_by_specimen[specimen_id][
                    "geometry_warning_text"
                ],
                "status": "pending_classification",
            }
        )

    lookup = {
        (
            row["material"],
            row["morphology"],
            row["indenter"],
            row["target_force_n"],
        ): row
        for row in rows
    }
    for row in rows:
        baseline = lookup[
            (row["material"], "baseline", row["indenter"], row["target_force_n"])
        ]
        baseline_value = float(baseline["optical_change_median_DN"])
        row["baseline_optical_change_median_DN"] = baseline_value
        row["baseline_gain_percent"] = 100.0 * (
            float(row["optical_change_median_DN"]) / baseline_value - 1.0
        )

    for specimen_id, stability in stability_by_specimen.items():
        specimen_rows = [row for row in rows if row["specimen_id"] == specimen_id]
        a2 = [row for row in specimen_rows if row["target_force_n"] == 2.0]
        a2_min = min(float(row["optical_change_median_DN"]) for row in a2)
        sensitivity_p95 = max(
            float(row["unloaded_reference_sensitivity_p95_DN"]) for row in a2
        )
        if stability["capture_count"] < 2:
            status = "unloaded-reference stability unavailable (single capture)"
        elif (
            stability["geometry_warning"]
            or stability["pair_max"] >= a2_min
            or sensitivity_p95 >= a2_min
        ):
            status = "unloaded-reference sensitive"
        else:
            status = "reference-stable in recorded captures"
        status_by_specimen[specimen_id] = status
        for row in specimen_rows:
            row["status"] = status

    for condition in MORPHOLOGY_CONDITIONS:
        if not condition.pending:
            continue
        for indenter in INDENTERS:
            for force in TARGET_FORCES_N:
                baseline = lookup[(condition.material, "baseline", indenter, force)]
                rows.append(
                    {
                        "specimen_id": "pending",
                        "material": condition.material,
                        "morphology": condition.morphology,
                        "indenter": indenter,
                        "target_force_n": force,
                        "valid_run_count": 0,
                        "optical_change_median_DN": float("nan"),
                        "optical_change_q1_DN": float("nan"),
                        "optical_change_q3_DN": float("nan"),
                        "optical_change_IQR_DN": float("nan"),
                        "baseline_optical_change_median_DN": baseline[
                            "optical_change_median_DN"
                        ],
                        "baseline_gain_percent": float("nan"),
                        "monotonic_run_count": 0,
                        "monotonic_run_fraction": float("nan"),
                        "unloaded_reference_sensitivity_median_DN": float("nan"),
                        "unloaded_reference_sensitivity_p95_DN": float("nan"),
                        "unloaded_reference_sensitivity_max_DN": float("nan"),
                        "unloaded_geometry_warning": "pending physical dataset",
                        "status": "pending",
                    }
                )
    rows.sort(
        key=lambda row: (
            row["material"],
            row["indenter"],
            row["morphology"],
            row["target_force_n"],
        )
    )
    return rows, status_by_specimen


def _write_profiles(
    path: Path,
    capture_profiles: list[dict[str, Any]],
    loaded_profiles: list[dict[str, Any]],
) -> None:
    np.savez_compressed(
        path,
        longitudinal_coordinate=np.linspace(0.0, 1.0, PROFILE_BINS, dtype=np.float32),
        unloaded_capture_profiles=np.asarray(
            [row["profile"] for row in capture_profiles], dtype=np.float32
        ),
        unloaded_specimen_id=np.asarray(
            [row["specimen_id"] for row in capture_profiles]
        ),
        unloaded_material=np.asarray([row["material"] for row in capture_profiles]),
        unloaded_morphology=np.asarray([row["morphology"] for row in capture_profiles]),
        unloaded_capture_id=np.asarray([row["capture_id"] for row in capture_profiles]),
        loaded_profiles=np.asarray(
            [row["loaded_profile"] for row in loaded_profiles], dtype=np.float32
        ),
        unloaded_referenced_delta_profiles=np.asarray(
            [row["delta_profile"] for row in loaded_profiles], dtype=np.float32
        ),
        loaded_specimen_id=np.asarray([row["specimen_id"] for row in loaded_profiles]),
        loaded_material=np.asarray([row["material"] for row in loaded_profiles]),
        loaded_morphology=np.asarray([row["morphology"] for row in loaded_profiles]),
        loaded_run_id=np.asarray([row["run_id"] for row in loaded_profiles]),
        loaded_indenter=np.asarray([row["indenter"] for row in loaded_profiles]),
        loaded_hole_index=np.asarray(
            [row["hole_index"] for row in loaded_profiles], dtype=np.int16
        ),
        loaded_repetition_index=np.asarray(
            [row["repetition_index"] for row in loaded_profiles], dtype=np.int16
        ),
        loaded_target_force_n=np.asarray(
            [row["target_force_n"] for row in loaded_profiles], dtype=np.float32
        ),
    )


def _morphology_lookup(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str, float], dict[str, Any]]:
    return {
        (
            row["material"],
            row["morphology"],
            row["indenter"],
            row["target_force_n"],
        ): row
        for row in rows
    }


def _plot_curves(output: Path, rows: list[dict[str, Any]]) -> None:
    lookup = _morphology_lookup(rows)
    figure, axes = plt.subplots(
        2, 2, figsize=(10.2, 7.0), sharey=False, constrained_layout=True
    )
    for row_index, material in enumerate(("solaris", "dragon_skin")):
        for column, indenter in enumerate(INDENTERS):
            axis = axes[row_index, column]
            for morphology in ("baseline", "flat_opt", "angled_opt"):
                values = [
                    lookup[(material, morphology, indenter, force)]
                    for force in TARGET_FORCES_N
                ]
                if values[0]["status"] == "pending":
                    continue
                median = np.asarray(
                    [0.0] + [row["optical_change_median_DN"] for row in values]
                )
                q1 = np.asarray([0.0] + [row["optical_change_q1_DN"] for row in values])
                q3 = np.asarray([0.0] + [row["optical_change_q3_DN"] for row in values])
                x = np.arange(5)
                axis.plot(
                    x,
                    median,
                    marker=MARKERS[morphology],
                    color=COLORS[morphology],
                    label=MORPHOLOGY_LABELS[morphology],
                )
                axis.fill_between(
                    x, q1, q3, color=COLORS[morphology], alpha=0.14, linewidth=0.0
                )
            axis.set_xticks(np.arange(5), ("Unloaded", "2", "5", "10", "15"))
            axis.set_title(f"{MATERIAL_LABELS[material]} · {INDENTER_LABELS[indenter]}")
            axis.set_xlabel("Target force state [N]")
            if column == 0:
                axis.set_ylabel("Optical change from unloaded [camera DN]")
            axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[0, 0].legend(frameon=False)
    figure.savefig(output / "unloaded_optical_activation_curves.png", dpi=220)
    plt.close(figure)


def _plot_2n(output: Path, rows: list[dict[str, Any]]) -> None:
    lookup = _morphology_lookup(rows)
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 6.8), constrained_layout=True)
    for row_index, material in enumerate(("solaris", "dragon_skin")):
        for column, indenter in enumerate(INDENTERS):
            axis = axes[row_index, column]
            for position, morphology in enumerate(
                ("baseline", "flat_opt", "angled_opt")
            ):
                row = lookup[(material, morphology, indenter, 2.0)]
                if row["status"] == "pending":
                    axis.text(
                        position,
                        0.5,
                        "Pending",
                        rotation=90,
                        ha="center",
                        va="center",
                        transform=axis.get_xaxis_transform(),
                        color="#777777",
                    )
                    continue
                median = float(row["optical_change_median_DN"])
                axis.errorbar(
                    position,
                    median,
                    yerr=[
                        [median - float(row["optical_change_q1_DN"])],
                        [float(row["optical_change_q3_DN"]) - median],
                    ],
                    fmt=MARKERS[morphology],
                    color=COLORS[morphology],
                    capsize=3,
                )
                if morphology != "baseline":
                    axis.annotate(
                        f"{row['baseline_gain_percent']:+.1f}%",
                        (position, median),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                    )
            axis.set_xticks(
                range(3), ("Baseline", "Flat-opt", "Angled-opt"), rotation=18
            )
            axis.set_title(f"{MATERIAL_LABELS[material]} · {INDENTER_LABELS[indenter]}")
            if column == 0:
                axis.set_ylabel("A(2 N) [camera DN]")
            axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    figure.savefig(output / "unloaded_optical_activation_2N.png", dpi=220)
    plt.close(figure)


def _plot_profiles(output: Path, loaded_profiles: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(
        2, 2, figsize=(10.2, 6.8), sharex=True, constrained_layout=True
    )
    force_colors = plt.cm.viridis(np.linspace(0.12, 0.9, len(TARGET_FORCES_N)))
    coordinate = np.linspace(0.0, 1.0, PROFILE_BINS)
    for row_index, material in enumerate(("solaris", "dragon_skin")):
        for column, indenter in enumerate(INDENTERS):
            axis = axes[row_index, column]
            subset = [
                row
                for row in loaded_profiles
                if row["material"] == material
                and row["morphology"] == "flat_opt"
                and row["indenter"] == indenter
            ]
            for force, color in zip(TARGET_FORCES_N, force_colors, strict=True):
                profiles = np.asarray(
                    [
                        row["delta_profile"]
                        for row in subset
                        if row["target_force_n"] == force
                    ]
                )
                median = np.median(profiles, axis=0)
                q1 = np.percentile(profiles, 25, axis=0)
                q3 = np.percentile(profiles, 75, axis=0)
                axis.plot(coordinate, median, color=color, label=f"{force:g} N")
                axis.fill_between(
                    coordinate, q1, q3, color=color, alpha=0.12, linewidth=0.0
                )
            axis.axhline(0.0, color="#888888", linewidth=0.7)
            axis.set_title(
                f"{MATERIAL_LABELS[material]} Flat-opt · {INDENTER_LABELS[indenter]}"
            )
            axis.set_xlabel("Normalized longitudinal coordinate (Distal → Proximal)")
            if column == 0:
                axis.set_ylabel("Loaded − paired unloaded [camera DN]")
    axes[0, 0].legend(frameon=False, ncol=2)
    figure.savefig(output / "unloaded_optical_activation_profile_examples.png", dpi=220)
    plt.close(figure)


def _plot_reference_qc(
    output: Path,
    morphology_rows: list[dict[str, Any]],
    extracted: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.1), constrained_layout=True)
    labels = [
        session["condition"].display_name.replace(" ", "\n", 1) for session in extracted
    ]
    pair_median = [
        float(np.median(session["unloaded_pair_distances"]))
        if session["unloaded_pair_distances"]
        else np.nan
        for session in extracted
    ]
    pair_max = [
        float(np.max(session["unloaded_pair_distances"]))
        if session["unloaded_pair_distances"]
        else np.nan
        for session in extracted
    ]
    x = np.arange(len(extracted))
    axes[0].scatter(x, pair_median, label="Median", color="#4c78a8")
    axes[0].scatter(x, pair_max, label="Maximum", marker="D", color="#e45756")
    axes[0].set_ylabel("Pairwise unloaded distance [DN]")
    axes[0].set_title("Unloaded profile drift")
    axes[0].legend(frameon=False)
    lookup = _morphology_lookup(morphology_rows)
    for position, session in enumerate(extracted):
        condition = session["condition"]
        values = [
            lookup[(condition.material, condition.morphology, indenter, 2.0)]
            for indenter in INDENTERS
        ]
        axes[1].scatter(
            [row["optical_change_median_DN"] for row in values],
            [row["unloaded_reference_sensitivity_p95_DN"] for row in values],
            marker=MARKERS[condition.morphology],
            color=COLORS[condition.morphology],
            label=condition.display_name,
        )
    limits = [0.0, max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])]
    axes[1].plot(limits, limits, "--", color="#999999", linewidth=0.8)
    axes[1].set_xlim(limits)
    axes[1].set_ylim(limits)
    axes[1].set_xlabel("A(2 N) median [DN]")
    axes[1].set_ylabel("Reference sensitivity p95 [DN]")
    axes[1].set_title("Reference choice versus low-load signal")
    centroid_spans = [
        float(session["existing"]["dataset"]["unloaded_geometry_centroid_span_px"])
        for session in extracted
    ]
    axes[2].bar(x, centroid_spans, color="#777777")
    axes[2].set_ylabel("Unloaded centroid span [px]")
    axes[2].set_title("Existing camera-geometry diagnostic")
    for axis in (axes[0], axes[2]):
        axis.set_xticks(x, labels, rotation=25, ha="right")
    figure.savefig(output / "unloaded_optical_activation_reference_qc.png", dpi=220)
    plt.close(figure)


def _case_for_morphology(
    lookup: dict[tuple[str, str, str, float], dict[str, Any]],
    material: str,
    morphology: str,
    indenter: str,
) -> str:
    row_2 = lookup[(material, morphology, indenter, 2.0)]
    if row_2["status"] == "pending":
        return "Pending"
    if morphology == "baseline":
        return "Reference"
    if row_2["status"] != "reference-stable in recorded captures":
        return "Case C — reference stability insufficient"
    baseline_2 = lookup[(material, "baseline", indenter, 2.0)]
    if row_2["optical_change_median_DN"] > baseline_2["optical_change_median_DN"]:
        return "Case A — larger low-load activation"
    high_larger = any(
        lookup[(material, morphology, indenter, force)]["optical_change_median_DN"]
        > lookup[(material, "baseline", indenter, force)]["optical_change_median_DN"]
        for force in (10.0, 15.0)
    )
    return (
        "Case B — high-load only"
        if high_larger
        else "Case D — baseline as large or larger"
    )


def _summary_markdown(
    morphology_rows: list[dict[str, Any]],
    pairing_rows: list[dict[str, Any]],
    extracted: list[dict[str, Any]],
) -> str:
    def value_or_dash(value: float) -> str:
        return f"{value:.4g}" if np.isfinite(value) else "—"

    lookup = _morphology_lookup(morphology_rows)
    lines = [
        "# Hardware unloaded-referenced optical activation feasibility",
        "",
        "This is a read-only optical-reference study.",
        "It does not define or register a paper metric.",
        "",
        "Every loaded and unloaded RGB frame was re-extracted through one",
        "session-global production `OpticalStrip`. Each run uses the unloaded",
        "capture nearest in camera host time for all four force states.",
        "`unloaded_time_delta_s = unloaded capture time - loaded run time`, so",
        "negative values mean that the unloaded capture occurred before the run.",
        "",
        "## Morphology activation curves",
        "",
    ]
    for material in ("solaris", "dragon_skin"):
        for indenter in INDENTERS:
            lines.extend(
                [
                    f"### {MATERIAL_LABELS[material]} · {INDENTER_LABELS[indenter]}",
                    "",
                    "| morphology | A_2N [DN] | A_5N [DN] | A_10N [DN] | A_15N [DN] | 2N gain vs baseline | early fraction | monotonic runs | reference sensitivity p95 [DN] | status | interpretation |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
                ]
            )
            for morphology in ("baseline", "flat_opt", "angled_opt"):
                values = [
                    lookup[(material, morphology, indenter, force)]
                    for force in TARGET_FORCES_N
                ]
                if values[0]["status"] == "pending":
                    lines.append(
                        f"| {MORPHOLOGY_LABELS[morphology]} | — | — | — | — | — | — | — | — | pending | Pending |"
                    )
                    continue
                medians = [float(row["optical_change_median_DN"]) for row in values]
                early = medians[0] / medians[-1] if medians[-1] > 0.0 else float("nan")
                lines.append(
                    f"| {MORPHOLOGY_LABELS[morphology]} | {medians[0]:.4g} | {medians[1]:.4g} | {medians[2]:.4g} | {medians[3]:.4g} | "
                    f"{values[0]['baseline_gain_percent']:+.2f}% | {early:.4g} | "
                    f"{values[0]['monotonic_run_count']}/{values[0]['valid_run_count']} "
                    f"({100.0 * values[0]['monotonic_run_fraction']:.1f}%) | "
                    f"{value_or_dash(float(values[0]['unloaded_reference_sensitivity_p95_DN']))} | "
                    f"{values[0]['status']} | {_case_for_morphology(lookup, material, morphology, indenter)} |"
                )
            lines.append("")

    lines.extend(["## Unloaded-reference stability", ""])
    for session in extracted:
        condition = session["condition"]
        distances = session["unloaded_pair_distances"]
        pair_median = float(np.median(distances)) if distances else float("nan")
        pair_max = float(np.max(distances)) if distances else float("nan")
        dataset = session["existing"]["dataset"]
        status = lookup[(condition.material, condition.morphology, INDENTERS[0], 2.0)][
            "status"
        ]
        lines.extend(
            [
                f"### {condition.display_name} (`{session['index'].specimen_id}`)",
                "",
                f"- unloaded captures: **{len(session['capture_rows'])}**",
                f"- pairwise unloaded-profile distance median/max [DN]: **{pair_median:.4g} / {pair_max:.4g}**"
                if distances
                else "- pairwise unloaded-profile distance: **unavailable (one capture)**",
                f"- existing unloaded optical-region centroid span: **{float(dataset['unloaded_geometry_centroid_span_px']):.4g} px**",
                f"- reference status: **{status}**",
            ]
        )
        geometry = session["existing"]["geometry_messages"]
        lines.append(
            f"- camera-geometry warning: **{'; '.join(geometry)}**"
            if geometry
            else "- camera-geometry warning: none in the existing summary"
        )
        if session["excluded_runs"]:
            lines.append(
                f"- excluded incomplete runs: **{'; '.join(session['excluded_runs'])}**"
            )
        lines.append("")

    lines.extend(
        [
            "## Explicit nearest-time pairing",
            "",
            "Complete run-level timestamps and signed differences are stored in",
            "`unloaded_optical_activation_pairing.csv`. The assignments are:",
            "",
        ]
    )
    grouped_pairing: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in pairing_rows:
        grouped_pairing[(row["specimen_id"], row["paired_unloaded_capture_id"])].append(
            row["run_id"]
        )
    for (specimen_id, capture_id), run_ids in sorted(grouped_pairing.items()):
        lines.append(f"- `{specimen_id}` / `{capture_id}`: `{', '.join(run_ids)}`")

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The tables distinguish larger low-load activation, high-load-only",
            "improvement, baseline-dominant results, and unloaded-reference",
            "sensitivity without redefining the metric. Any specimen marked",
            "`unloaded-reference sensitive` or with only one unloaded capture",
            "should not support a strong morphology-ranking claim from this study.",
            "",
            "No force normalization, per-run normalization, smoothing to enforce",
            "monotonicity, pixel-to-mm conversion, production registration, or",
            "Figure 5 modification was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    arguments = _arguments()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    available = [
        condition for condition in MORPHOLOGY_CONDITIONS if not condition.pending
    ]
    extracted = []
    for condition in available:
        print(f"extracting {condition.display_name} ...", flush=True)
        extracted.append(_extract_session(condition))

    run_rows = [row for session in extracted for row in session["run_rows"]]
    pairing_rows = [row for session in extracted for row in session["pairing_rows"]]
    sensitivity_rows = [
        row for session in extracted for row in session["sensitivity_rows"]
    ]
    stability_rows = [row for session in extracted for row in session["stability_rows"]]
    capture_profiles = [
        row for session in extracted for row in session["capture_profiles"]
    ]
    loaded_profiles = [
        row for session in extracted for row in session["loaded_profiles"]
    ]
    morphology_rows, _ = _aggregate_morphologies(run_rows, extracted)

    _write_csv(output / "unloaded_optical_activation_runs.csv", run_rows)
    _write_csv(output / "unloaded_optical_activation_morphologies.csv", morphology_rows)
    _write_csv(output / "unloaded_optical_activation_pairing.csv", pairing_rows)
    _write_csv(
        output / "unloaded_optical_activation_reference_sensitivity.csv",
        sensitivity_rows,
    )
    _write_csv(
        output / "unloaded_optical_activation_unloaded_stability.csv", stability_rows
    )
    _write_profiles(
        output / "unloaded_optical_activation_profiles.npz",
        capture_profiles,
        loaded_profiles,
    )
    _plot_curves(output, morphology_rows)
    _plot_2n(output, morphology_rows)
    _plot_profiles(output, loaded_profiles)
    _plot_reference_qc(output, morphology_rows, extracted)
    summary = _summary_markdown(morphology_rows, pairing_rows, extracted)
    (output / "unloaded_optical_activation_summary.md").write_text(
        summary, encoding="utf-8"
    )
    print(summary)


if __name__ == "__main__":
    main()
