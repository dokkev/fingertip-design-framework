#!/usr/bin/env python3
"""Replay learning-free optical localization and global torque calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from algorithm import (  # noqa: E402
    ContactLocalizationConfig,
    build_canonical_map,
    build_unloaded_reference,
    canonical_position_to_mm,
    causal_median_response,
    compute_longitudinal_response,
    detect_leds,
    landmark_longitudinal_coordinates,
    localize_response_profile,
    segment_fingertip,
    warp_to_canonical,
)


DATASET_ROOT = REPOSITORY_ROOT / "output" / "proprioceptive_robust_dataset"
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "output" / "validation" / "proprioceptive_robustness"
)
ROTATION_AXIS_TO_LED1_MM = 103.6
REFERENCE_FRAME_COUNT = 5
LOADED_FORCE_THRESHOLD_N = 2.0
RUN_GROUPS = {
    "Nominal": ("run_001", "run_002", "run_003"),
    "View A": ("run_007", "run_008", "run_009"),
    "View B": ("run_010", "run_011", "run_012"),
    "View C": ("run_013", "run_014", "run_015"),
    "Bright": ("run_016", "run_017", "run_018"),
    "Dark": ("run_019", "run_020", "run_021"),
}
CONTACT_LOCATIONS_MM = (0.0, 20.0, 40.0)
TRIAL_CORRECTIONS = {"run_018": 6, "run_021": 7}


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_run(root: Path, run_id: str) -> dict[str, object]:
    directory = root / run_id
    metadata = json.loads((directory / "metadata.json").read_text())
    camera = pd.read_csv(directory / "camera_timestamps.csv")
    sequence = pd.read_csv(directory / "force_sequence.csv")
    if camera.empty or sequence.empty:
        raise RuntimeError(f"{run_id} contains an empty acquisition table")
    return {
        "run_id": run_id,
        "directory": directory,
        "metadata": metadata,
        "camera": camera,
        "sequence": sequence,
    }


def _reference_indices(camera: pd.DataFrame) -> np.ndarray:
    force = camera["ft_contact_force_N"].to_numpy(dtype=np.float64)
    if len(force) < REFERENCE_FRAME_COUNT:
        raise RuntimeError("camera table has fewer than five frames")
    selected = np.argsort(force)[:REFERENCE_FRAME_COUNT]
    if float(np.max(force[selected])) >= 1.0:
        raise RuntimeError(
            "condition reference does not contain five observations below 1 N"
        )
    return np.sort(selected)


def _median_rgb(run: dict[str, object], indices: np.ndarray) -> np.ndarray:
    directory = run["directory"]
    camera = run["camera"]
    frames = [
        _read_rgb(directory / str(camera.iloc[index]["filename"]))
        for index in indices
    ]
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _nearest_indices(query: np.ndarray, source: np.ndarray) -> np.ndarray:
    insertion = np.searchsorted(source, query)
    insertion = np.clip(insertion, 0, len(source) - 1)
    previous = np.maximum(insertion - 1, 0)
    use_previous = np.abs(query - source[previous]) <= np.abs(
        source[insertion] - query
    )
    return np.where(use_previous, previous, insertion)


def _geometry_overlay(
    rgb: np.ndarray,
    fingertip: object,
    leds: object,
    title: str,
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(
        fingertip.mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(bgr, contours, -1, (0, 220, 255), 3, cv2.LINE_AA)
    for index, (x_coordinate, y_coordinate) in enumerate(
        leds.landmarks_xy_px,
        start=1,
    ):
        point = (round(float(x_coordinate)), round(float(y_coordinate)))
        cv2.circle(bgr, point, 8, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            bgr,
            str(index),
            (point[0] + 9, point[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        bgr,
        title,
        (18, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.resize(bgr, (480, 270), interpolation=cv2.INTER_AREA)


def _run_geometry(run: dict[str, object]) -> tuple[object, object, object, np.ndarray]:
    camera = run["camera"]
    force = camera["ft_contact_force_N"].to_numpy(dtype=np.float64)
    reference_index = int(np.argmin(force))
    reference_rgb = _read_rgb(
        run["directory"] / str(camera.iloc[reference_index]["filename"])
    )
    fingertip = segment_fingertip(reference_rgb)
    leds = detect_leds(reference_rgb, fingertip)
    canonical_map = build_canonical_map(fingertip)
    led_coordinates = landmark_longitudinal_coordinates(
        canonical_map,
        leds.landmarks_xy_px,
    )
    return fingertip, leds, canonical_map, led_coordinates


def _condition_reference(
    anchor_run: dict[str, object],
    config: ContactLocalizationConfig,
) -> object:
    indices = _reference_indices(anchor_run["camera"])
    median_rgb = _median_rgb(anchor_run, indices)
    fingertip = segment_fingertip(median_rgb)
    canonical_map = build_canonical_map(fingertip)
    frames = [
        warp_to_canonical(
            _read_rgb(
                anchor_run["directory"]
                / str(anchor_run["camera"].iloc[index]["filename"])
            ),
            canonical_map,
        )
        for index in indices
    ]
    return build_unloaded_reference(frames, config)


def _replay_run(
    run: dict[str, object],
    condition: str,
    location_mm: float,
    unloaded_reference: object,
    unloaded_reference_run_id: str,
    config: ContactLocalizationConfig,
) -> tuple[list[dict[str, object]], dict[str, object], np.ndarray]:
    fingertip, leds, canonical_map, led_coordinates = _run_geometry(run)
    camera = run["camera"]
    response_history: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for camera_row in camera.itertuples(index=False):
        rgb = _read_rgb(run["directory"] / str(camera_row.filename))
        canonical = warp_to_canonical(rgb, canonical_map)
        response, saturation_ge_250, saturation_eq_255 = (
            compute_longitudinal_response(
                canonical,
                unloaded_reference.canonical_rgb,
                brightest_fraction=config.brightest_fraction,
                smoothing_sigma=config.longitudinal_smoothing_sigma,
            )
        )
        response_history.append(response)
        filtered = causal_median_response(
            response_history,
            config.temporal_median_window,
        )
        result = localize_response_profile(
            filtered,
            unloaded_reference,
            led_coordinates,
            config,
            saturation_fraction_ge_250=saturation_ge_250,
            saturation_fraction_eq_255=saturation_eq_255,
        )
        evidence_total = float(np.sum(result.evidence_profile))
        raw_centroid = (
            float(
                np.sum(
                    np.linspace(0.0, 1.0, len(result.evidence_profile))
                    * result.evidence_profile
                )
                / evidence_total
            )
            if evidence_total > 0.0
            else np.nan
        )
        raw_geometry_position_mm = (
            canonical_position_to_mm(
                raw_centroid,
                led_coordinates,
                maximum_extrapolation_mm=1.0e6,
            )
            if np.isfinite(raw_centroid)
            else np.nan
        )
        rows.append(
            {
                "run_id": run["run_id"],
                "observation_condition": condition,
                "trial": TRIAL_CORRECTIONS.get(
                    run["run_id"],
                    int(run["metadata"]["experiment"]["trial"]),
                ),
                "contact_location_gt_mm": location_mm,
                "optical_reference_run_id": unloaded_reference_run_id,
                "camera_frame_index": int(camera_row.frame_index),
                "camera_timestamp_ns": int(camera_row.timestamp_ns),
                "ground_truth_force_n": float(camera_row.ft_contact_force_N),
                "optical_contact_detected": bool(result.contact_detected),
                "optical_location_valid": bool(
                    result.valid and result.contact_detected
                ),
                "estimated_contact_location_mm": (
                    result.position_mm
                    if result.valid and result.contact_detected
                    else np.nan
                ),
                "raw_response_centroid": raw_centroid,
                "raw_geometry_position_mm": raw_geometry_position_mm,
                "optical_status": result.status,
                "total_evidence_dn": result.total_evidence_dn,
                "peak_snr": result.peak_snr,
                "peak_to_background_ratio": result.peak_to_background_ratio,
                "normalized_spatial_entropy": result.normalized_spatial_entropy,
                "saturation_fraction_ge_250": saturation_ge_250,
                "saturation_fraction_eq_255": saturation_eq_255,
            }
        )
    reference_force = camera["ft_contact_force_N"].to_numpy(dtype=np.float64)
    reference_index = int(np.argmin(reference_force))
    geometry_row: dict[str, object] = {
        "run_id": run["run_id"],
        "observation_condition": condition,
        "contact_location_gt_mm": location_mm,
        "reference_frame_index": int(camera.iloc[reference_index]["frame_index"]),
        "reference_force_n": float(reference_force[reference_index]),
        "median_led_spacing_px": leds.median_spacing_px,
    }
    for index, (point, coordinate) in enumerate(
        zip(leds.landmarks_xy_px, led_coordinates, strict=True),
        start=1,
    ):
        geometry_row[f"led{index}_x_px"] = float(point[0])
        geometry_row[f"led{index}_y_px"] = float(point[1])
        geometry_row[f"led{index}_canonical"] = float(coordinate)
    overlay = _geometry_overlay(
        _read_rgb(
            run["directory"] / str(camera.iloc[reference_index]["filename"])
        ),
        fingertip,
        leds,
        f"{condition} | {run['run_id']} | x={location_mm:g} mm",
    )
    return rows, geometry_row, overlay


def _attach_motor_torque(
    camera_results: pd.DataFrame,
    runs: dict[str, dict[str, object]],
) -> pd.DataFrame:
    groups = []
    for run_id, group in camera_results.groupby("run_id", sort=False):
        sequence = runs[run_id]["sequence"]
        sequence_time = sequence["timestamp_ns"].to_numpy(dtype=np.int64)
        camera_time = group["camera_timestamp_ns"].to_numpy(dtype=np.int64)
        indices = _nearest_indices(camera_time, sequence_time)
        selected = sequence.iloc[indices]
        attached = group.copy()
        attached["sequence_timestamp_ns"] = selected["timestamp_ns"].to_numpy(
            dtype=np.int64
        )
        attached["camera_sequence_time_delta_ms"] = np.abs(
            camera_time - attached["sequence_timestamp_ns"].to_numpy(dtype=np.int64)
        ) / 1.0e6
        attached["motor_torque_nm"] = selected["motor_torque_Nm"].to_numpy(
            dtype=np.float64
        )
        attached["target_force_n"] = selected["target_force_N"].to_numpy(
            dtype=np.float64
        )
        groups.append(attached)
    return pd.concat(groups, ignore_index=True)


def _fit_global_affine(table: pd.DataFrame) -> tuple[float, float, int]:
    location = table["estimated_contact_location_mm"].to_numpy(dtype=np.float64)
    torque = table["motor_torque_nm"].to_numpy(dtype=np.float64)
    force = table["ground_truth_force_n"].to_numpy(dtype=np.float64)
    calibration = (
        (table["observation_condition"].to_numpy() == "Nominal")
        & (force >= LOADED_FORCE_THRESHOLD_N)
        & np.isfinite(location)
        & np.isfinite(torque)
    )
    if np.count_nonzero(calibration) < 10:
        raise RuntimeError("nominal runs contain too few valid calibration samples")
    moment_arm_m = (ROTATION_AXIS_TO_LED1_MM - location[calibration]) / 1000.0
    if np.any(moment_arm_m <= 0.0):
        raise RuntimeError("nominal optical location produced a nonpositive moment arm")
    target_torque_nm = force[calibration] * moment_arm_m
    design = np.column_stack(
        (torque[calibration], np.ones(np.count_nonzero(calibration)))
    )
    alpha_tau, bias_tau_nm = np.linalg.lstsq(
        design,
        target_torque_nm,
        rcond=None,
    )[0]
    return float(alpha_tau), float(bias_tau_nm), int(np.count_nonzero(calibration))


def _summarize(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, run_id), group in table.groupby(
        ["observation_condition", "run_id"],
        sort=False,
    ):
        force = group["ground_truth_force_n"].to_numpy(dtype=np.float64)
        location = group["estimated_contact_location_mm"].to_numpy(dtype=np.float64)
        loaded = force >= LOADED_FORCE_THRESHOLD_N
        localized = loaded & np.isfinite(location)
        force_estimated = localized & np.isfinite(
            group["estimated_force_n"].to_numpy(dtype=np.float64)
        )
        location_error = np.abs(
            location - float(group["contact_location_gt_mm"].iloc[0])
        )
        force_error = (
            group["estimated_force_n"].to_numpy(dtype=np.float64) - force
        )
        rows.append(
            {
                "run_id": run_id,
                "observation_condition": condition,
                "trial": int(group["trial"].iloc[0]),
                "contact_location_gt_mm": float(
                    group["contact_location_gt_mm"].iloc[0]
                ),
                "loaded_camera_samples": int(np.count_nonzero(loaded)),
                "localized_camera_samples": int(np.count_nonzero(localized)),
                "localization_coverage": float(
                    np.count_nonzero(localized) / max(np.count_nonzero(loaded), 1)
                ),
                "median_estimated_contact_location_mm": (
                    float(np.median(location[localized]))
                    if np.any(localized)
                    else np.nan
                ),
                "contact_location_mae_mm": (
                    float(np.mean(location_error[localized]))
                    if np.any(localized)
                    else np.nan
                ),
                "force_estimate_samples": int(np.count_nonzero(force_estimated)),
                "force_mae_n": (
                    float(np.mean(np.abs(force_error[force_estimated])))
                    if np.any(force_estimated)
                    else np.nan
                ),
                "force_rmse_n": (
                    float(np.sqrt(np.mean(force_error[force_estimated] ** 2)))
                    if np.any(force_estimated)
                    else np.nan
                ),
                "force_bias_n": (
                    float(np.mean(force_error[force_estimated]))
                    if np.any(force_estimated)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _fit_location_affine(
    raw_position_mm: np.ndarray,
    ground_truth_mm: np.ndarray,
) -> tuple[float, float, str]:
    design = np.column_stack((raw_position_mm, np.ones(len(raw_position_mm))))
    slope, intercept = np.linalg.lstsq(design, ground_truth_mm, rcond=None)[0]
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return np.nan, np.nan, "invalid: nonfinite affine fit"
    if slope <= 0.0:
        return float(slope), float(intercept), "invalid: nonmonotonic response"
    return float(slope), float(intercept), "valid"


def _condition_location_calibration(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare per-condition affine calibration with a held-out-run audit."""

    loaded = table[
        (table["ground_truth_force_n"] >= LOADED_FORCE_THRESHOLD_N)
        & np.isfinite(table["raw_geometry_position_mm"])
    ].copy()
    predictions: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for condition, condition_table in loaded.groupby(
        "observation_condition",
        sort=False,
    ):
        run_calibration = (
            condition_table.groupby(
                ["run_id", "contact_location_gt_mm"],
                as_index=False,
                sort=True,
            )["raw_geometry_position_mm"]
            .median()
            .sort_values("contact_location_gt_mm")
        )
        if len(run_calibration) != len(CONTACT_LOCATIONS_MM):
            raise RuntimeError(
                f"{condition} does not contain all three calibration locations"
            )

        protocols: list[tuple[str, str | None, pd.DataFrame]] = [
            ("three_run_affine_in_sample", None, run_calibration)
        ]
        protocols.extend(
            (
                "leave_one_run_out",
                held_out_run_id,
                run_calibration[run_calibration["run_id"] != held_out_run_id],
            )
            for held_out_run_id in run_calibration["run_id"]
        )
        for protocol, held_out_run_id, calibration_rows in protocols:
            slope, intercept, status = _fit_location_affine(
                calibration_rows["raw_geometry_position_mm"].to_numpy(
                    dtype=np.float64
                ),
                calibration_rows["contact_location_gt_mm"].to_numpy(
                    dtype=np.float64
                ),
            )
            evaluation = (
                condition_table
                if held_out_run_id is None
                else condition_table[condition_table["run_id"] == held_out_run_id]
            ).copy()
            evaluation["calibration_protocol"] = protocol
            evaluation["held_out_run_id"] = held_out_run_id or ""
            evaluation["calibration_run_ids"] = ",".join(
                calibration_rows["run_id"].astype(str)
            )
            evaluation["location_calibration_slope"] = slope
            evaluation["location_calibration_intercept_mm"] = intercept
            evaluation["location_calibration_status"] = status
            evaluation["condition_calibrated_location_mm"] = (
                slope * evaluation["raw_geometry_position_mm"] + intercept
                if status == "valid"
                else np.nan
            )
            evaluation["condition_calibrated_error_mm"] = (
                evaluation["condition_calibrated_location_mm"]
                - evaluation["contact_location_gt_mm"]
            )
            predictions.append(evaluation)

            finite = np.isfinite(evaluation["condition_calibrated_location_mm"])
            absolute_error = np.abs(
                evaluation.loc[finite, "condition_calibrated_error_mm"].to_numpy(
                    dtype=np.float64
                )
            )
            summaries.append(
                {
                    "observation_condition": condition,
                    "calibration_protocol": protocol,
                    "held_out_run_id": held_out_run_id or "",
                    "calibration_run_ids": ",".join(
                        calibration_rows["run_id"].astype(str)
                    ),
                    "location_calibration_slope": slope,
                    "location_calibration_intercept_mm": intercept,
                    "location_calibration_status": status,
                    "evaluation_sample_count": len(evaluation),
                    "finite_prediction_count": int(np.count_nonzero(finite)),
                    "location_mae_mm": (
                        float(np.mean(absolute_error))
                        if absolute_error.size
                        else np.nan
                    ),
                    "location_median_absolute_error_mm": (
                        float(np.median(absolute_error))
                        if absolute_error.size
                        else np.nan
                    ),
                    "within_one_led_pitch_fraction": (
                        float(np.mean(absolute_error <= 11.0))
                        if absolute_error.size
                        else np.nan
                    ),
                }
            )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(summaries)


def _plot_condition_location_calibration(
    predictions: pd.DataFrame,
    output_stem: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 3.1), sharex=True)
    protocols = (
        ("three_run_affine_in_sample", "Three-run condition fit"),
        ("leave_one_run_out", "Leave-one-run/location-out"),
    )
    colors = plt.get_cmap("tab10").colors
    for axis, (protocol, title) in zip(axes, protocols, strict=True):
        selected = predictions[predictions["calibration_protocol"] == protocol]
        for color, (condition, condition_table) in zip(
            colors,
            selected.groupby("observation_condition", sort=False),
            strict=False,
        ):
            run_medians = (
                condition_table.groupby("contact_location_gt_mm", as_index=False)[
                    "condition_calibrated_location_mm"
                ]
                .median()
                .sort_values("contact_location_gt_mm")
            )
            axis.plot(
                run_medians["contact_location_gt_mm"],
                run_medians["condition_calibrated_location_mm"],
                marker="o",
                linewidth=1.0,
                markersize=4.5,
                color=color,
                label=condition,
            )
        axis.plot([-5, 45], [-5, 45], "--", color="0.55", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlim(-5, 45)
        axis.set_xticks(CONTACT_LOCATIONS_MM)
        axis.margins(y=0.08)
        axis.grid(axis="both", color="0.9", linewidth=0.6)
    axes[0].set_ylabel("Estimated contact location [mm]")
    figure.supxlabel("Ground-truth contact location [mm]")
    axes[1].legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    figure.tight_layout()
    figure.savefig(output_stem.with_suffix(".png"), dpi=240)
    figure.savefig(output_stem.with_suffix(".pdf"))
    plt.close(figure)


def _summarize_condition_calibration(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (condition, protocol), group in predictions.groupby(
        ["observation_condition", "calibration_protocol"],
        sort=False,
    ):
        absolute_error = np.abs(
            group["condition_calibrated_error_mm"].to_numpy(dtype=np.float64)
        )
        finite = np.isfinite(absolute_error)
        rows.append(
            {
                "observation_condition": condition,
                "calibration_protocol": protocol,
                "evaluation_sample_count": len(group),
                "finite_prediction_count": int(np.count_nonzero(finite)),
                "location_mae_mm": (
                    float(np.mean(absolute_error[finite]))
                    if np.any(finite)
                    else np.nan
                ),
                "within_one_led_pitch_fraction": (
                    float(np.mean(absolute_error[finite] <= 11.0))
                    if np.any(finite)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_lookup: dict[str, dict[str, object]] = {}
    condition_references: dict[str, object] = {}
    config = ContactLocalizationConfig(
        unloaded_frame_count=REFERENCE_FRAME_COUNT,
        temporal_median_window=3,
    )
    for condition, run_ids in RUN_GROUPS.items():
        for run_id in run_ids:
            run_lookup[run_id] = _load_run(args.dataset_root, run_id)
        condition_references[condition] = _condition_reference(
            run_lookup[run_ids[0]],
            config,
        )

    camera_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    overlay_rows: list[np.ndarray] = []
    for condition, run_ids in RUN_GROUPS.items():
        condition_overlays = []
        for run_id, location_mm in zip(
            run_ids,
            CONTACT_LOCATIONS_MM,
            strict=True,
        ):
            try:
                unloaded_reference = _condition_reference(
                    run_lookup[run_id],
                    config,
                )
                unloaded_reference_run_id = run_id
            except RuntimeError:
                unloaded_reference = condition_references[condition]
                unloaded_reference_run_id = run_ids[0]
            rows, geometry, overlay = _replay_run(
                run_lookup[run_id],
                condition,
                location_mm,
                unloaded_reference,
                unloaded_reference_run_id,
                config,
            )
            camera_rows.extend(rows)
            geometry_rows.append(geometry)
            condition_overlays.append(overlay)
        overlay_rows.append(np.hstack(condition_overlays))

    camera_results = pd.DataFrame(camera_rows)
    estimates = _attach_motor_torque(camera_results, run_lookup)
    alpha_tau, bias_tau_nm, calibration_sample_count = _fit_global_affine(estimates)
    location = estimates["estimated_contact_location_mm"].to_numpy(dtype=np.float64)
    moment_arm_m = (ROTATION_AXIS_TO_LED1_MM - location) / 1000.0
    calibrated_torque_nm = (
        alpha_tau * estimates["motor_torque_nm"].to_numpy(dtype=np.float64)
        + bias_tau_nm
    )
    estimated_force_n = np.divide(
        calibrated_torque_nm,
        moment_arm_m,
        out=np.full(len(estimates), np.nan, dtype=np.float64),
        where=np.isfinite(moment_arm_m) & (moment_arm_m > 0.0),
    )
    estimates["moment_arm_m"] = moment_arm_m
    estimates["calibrated_torque_nm"] = calibrated_torque_nm
    estimates["global_alpha_tau"] = alpha_tau
    estimates["global_bias_tau_nm"] = bias_tau_nm
    estimates["estimated_force_n"] = estimated_force_n
    estimates["force_error_n"] = (
        estimated_force_n
        - estimates["ground_truth_force_n"].to_numpy(dtype=np.float64)
    )
    # Formula-facing aliases make the reproducibility artifact unambiguous.
    estimates["x_hat_mm"] = estimates["estimated_contact_location_mm"]
    estimates["f_gt_n"] = estimates["ground_truth_force_n"]
    estimates["f_hat_n"] = estimates["estimated_force_n"]
    estimates["tau_act_nm"] = estimates["motor_torque_nm"]
    summary = _summarize(estimates)
    calibrated_predictions, calibrated_summary = _condition_location_calibration(
        estimates
    )
    calibrated_condition_summary = _summarize_condition_calibration(
        calibrated_predictions
    )

    camera_results.to_csv(args.output_dir / "camera_localization.csv", index=False)
    pd.DataFrame(geometry_rows).to_csv(
        args.output_dir / "geometry_registration.csv",
        index=False,
    )
    estimates.to_csv(args.output_dir / "per_sample_estimates.csv", index=False)
    summary.to_csv(args.output_dir / "run_summary.csv", index=False)
    calibrated_predictions.to_csv(
        args.output_dir / "condition_calibrated_localization.csv",
        index=False,
    )
    calibrated_summary.to_csv(
        args.output_dir / "condition_calibrated_localization_summary.csv",
        index=False,
    )
    calibrated_condition_summary.to_csv(
        args.output_dir / "condition_calibrated_localization_by_condition.csv",
        index=False,
    )
    _plot_condition_location_calibration(
        calibrated_predictions,
        args.output_dir / "condition_calibrated_localization",
    )
    cv2.imwrite(
        str(args.output_dir / "geometry_registration.png"),
        np.vstack(overlay_rows),
    )

    localization_ready = bool(
        np.all(summary["localization_coverage"] >= 0.80)
        and np.all(summary["contact_location_mae_mm"] <= 11.0)
    )
    nominal_summary = summary[
        summary["observation_condition"] == "Nominal"
    ]
    calibration_location_ready = bool(
        np.all(nominal_summary["localization_coverage"] >= 0.80)
        and np.all(nominal_summary["contact_location_mae_mm"] <= 11.0)
    )
    condition_in_sample = calibrated_summary[
        calibrated_summary["calibration_protocol"]
        == "three_run_affine_in_sample"
    ]
    condition_leave_one_out = calibrated_summary[
        calibrated_summary["calibration_protocol"] == "leave_one_run_out"
    ]
    condition_in_sample_ready = bool(
        np.all(condition_in_sample["location_calibration_status"] == "valid")
        and np.all(condition_in_sample["within_one_led_pitch_fraction"] >= 0.80)
    )
    condition_leave_one_out_ready = bool(
        np.all(condition_leave_one_out["location_calibration_status"] == "valid")
        and np.all(
            condition_leave_one_out["within_one_led_pitch_fraction"] >= 0.80
        )
    )
    report = {
        "analysis_version": 1,
        "dataset_root": str(args.dataset_root),
        "led_on_run_count": len(summary),
        "excluded_led_off_runs": ["run_004", "run_005", "run_006"],
        "metadata_corrections": {
            "run_018": {"contact_location_gt_mm": 40.0, "trial": 6},
            "run_021": {"trial": 7},
        },
        "optical_reference_protocol": (
            "five below-1-N frames from the current run when present; otherwise "
            "the 0-mm run from the same observation condition; no loaded start "
            "is treated as unloaded"
        ),
        "force_model": (
            "F_hat=(alpha_tau*tau_act+b_tau)/((103.6-x_hat)/1000)"
        ),
        "global_alpha_tau": alpha_tau,
        "global_bias_tau_nm": bias_tau_nm,
        "calibration_conditions": ["run_001", "run_002", "run_003"],
        "calibration_sample_count": calibration_sample_count,
        "calibration_location_qc_passed": calibration_location_ready,
        "calibration_status": (
            "candidate_global_fit"
            if calibration_location_ready
            else "diagnostic_only_due_to_nominal_localization_failure"
        ),
        "per_run_torque_zeroing": False,
        "condition_location_calibration": {
            "unit": "one camera-extrinsic/lighting condition containing three runs",
            "contact_locations_mm": list(CONTACT_LOCATIONS_MM),
            "model": "positive-slope affine map from raw geometry position to contact location",
            "three_run_in_sample_qc_passed": condition_in_sample_ready,
            "leave_one_run_location_out_qc_passed": condition_leave_one_out_ready,
        },
        "fig6e_ready": localization_ready,
        "fig6e_blocker": (
            None
            if localization_ready
            else "one or more runs fail 80% coverage or one-LED-pitch location MAE"
        ),
    }
    (args.output_dir / "global_torque_calibration.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nalpha_tau={alpha_tau:.9f}")
    print(f"b_tau={bias_tau_nm:.9f} N m")
    print(f"Fig. 6(e) ready: {localization_ready}")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
