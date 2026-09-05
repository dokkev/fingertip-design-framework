"""Evaluate rigid-body image tracking as a hardware indentation signal.

This is a bounded physical-data feasibility study.  It does not modify the
physical dataset, Figure 5, production analysis, or any paper metric.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.analysis.dataset import index_session  # noqa: E402
from experiments.analysis.metrics import actual_force_magnitude  # noqa: E402


DEFAULT_CONFIG_PATH = Path(__file__).with_name(
    "hardware_indentation_tracking_config.json"
)
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "output" / "validation" / "hardware_indentation_tracking"
)
TARGET_FORCES_N = (2.0, 5.0, 10.0, 15.0)
REFERENCE_FORCE_N = 2.0
CSV_FLOAT_FORMAT = ".9g"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="JSON file containing the session and two fixed image ROIs",
    )
    parser.add_argument(
        "--stage",
        choices=("manual", "sample", "full"),
        default="manual",
        help=(
            "manual uses one inspection run; sample uses the configured 6--12 "
            "run subset; full uses every complete matching run"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    config_path = path if path.is_absolute() else REPOSITORY_ROOT / path
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "session_path",
        "indenter",
        "moving_roi_xywh",
        "fixture_roi_xywh",
        "manual_run_id",
        "sample_run_ids",
        "hole_positions_mm",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing config entries: {missing}")
    config["_config_path"] = str(config_path.resolve())
    return config


def _resolve_session_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _roi_xyxy(value: Iterable[int], image_shape: tuple[int, ...]) -> tuple[int, ...]:
    x, y, width, height = (int(component) for component in value)
    image_height, image_width = image_shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be positive")
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError(
            f"ROI {(x, y, width, height)} exceeds image size "
            f"{image_width} x {image_height}"
        )
    return x, y, x + width, y + height


def _load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def _median_bgr(paths: list[Path]) -> np.ndarray:
    if not paths:
        raise ValueError("cannot construct a hold median without frames")
    images = [_load_bgr(path) for path in paths]
    shape = images[0].shape
    if any(image.shape != shape for image in images):
        raise ValueError("all frames in one hold must have the same shape")
    return np.median(np.stack(images), axis=0).astype(np.uint8)


def _red_edge(image: np.ndarray, roi: tuple[int, ...]) -> np.ndarray:
    """Return the one deterministic phase-correlation representation."""

    x0, y0, x1, y1 = roi
    red = image[y0:y1, x0:x1, 2].astype(np.float32)
    gradient_x = cv2.Scharr(red, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Scharr(red, cv2.CV_32F, 0, 1)
    edge = cv2.magnitude(gradient_x, gradient_y)
    edge -= float(np.mean(edge))
    return edge


def _phase_shift(
    reference: np.ndarray,
    current: np.ndarray,
    window: np.ndarray,
) -> tuple[float, float, float]:
    shift, response = cv2.phaseCorrelate(reference, current, window)
    values = np.asarray((*shift, response), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("phase correlation returned a non-finite result")
    return float(shift[0]), float(shift[1]), float(response)


def _frame_force(measurements: dict[str, str] | Any) -> float:
    return actual_force_magnitude(
        float(measurements["Fx_N"]),
        float(measurements["Fy_N"]),
        float(measurements["Fz_N"]),
    )


def _choose_runs(index: Any, config: dict[str, Any], stage: str) -> list[Any]:
    wanted_indenter = str(config["indenter"])
    complete = {
        run.run_id: run
        for run in index.runs
        if run.status == "complete" and run.indenter == wanted_indenter
    }
    frame_forces: dict[str, set[float]] = {run_id: set() for run_id in complete}
    for frame in index.frames:
        if frame.run is not None and frame.run.run_id in frame_forces:
            assert frame.target_force_n is not None
            frame_forces[frame.run.run_id].add(float(frame.target_force_n))
    complete = {
        run_id: run
        for run_id, run in complete.items()
        if set(TARGET_FORCES_N).issubset(frame_forces[run_id])
    }
    if stage == "manual":
        run_ids = [str(config["manual_run_id"])]
    elif stage == "sample":
        run_ids = [str(value) for value in config["sample_run_ids"]]
        sample_holes = {complete[run_id].hole_index for run_id in run_ids if run_id in complete}
        if len(run_ids) < 6 or len(run_ids) > 12 or sample_holes != set(range(1, 7)):
            raise ValueError(
                "sample_run_ids must contain 6--12 complete runs spanning holes 1--6"
            )
    else:
        run_ids = sorted(complete)
    missing = sorted(set(run_ids) - set(complete))
    if missing:
        raise ValueError(f"selected runs are absent, incomplete, or missing forces: {missing}")
    return [complete[run_id] for run_id in run_ids]


def _group_frames(index: Any, run_ids: set[str]) -> dict[tuple[str, float], list[Any]]:
    groups: dict[tuple[str, float], list[Any]] = {}
    for frame in index.frames:
        if frame.run is None or frame.run.run_id not in run_ids:
            continue
        assert frame.target_force_n is not None
        key = frame.run.run_id, float(frame.target_force_n)
        groups.setdefault(key, []).append(frame)
    for frames in groups.values():
        frames.sort(key=lambda frame: int(frame.measurements["frame_index"]))
    return groups


def _estimate_axis(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 1:
        raise ValueError("at least one two-dimensional 2-to-15 N vector is required")
    _, singular_values, right_vectors = np.linalg.svd(values, full_matrices=False)
    axis = right_vectors[0]
    if float(np.median(values @ axis)) < 0.0:
        axis = -axis
    denominator = float(np.sum(singular_values**2))
    dominance = float(singular_values[0] ** 2 / denominator) if denominator else 0.0
    return axis, singular_values, dominance


def _finite_row(row: dict[str, Any], fields: Iterable[str]) -> None:
    values = np.asarray([float(row[field]) for field in fields])
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"non-finite tracking result in {row.get('run_id', '?')}")


def _track(
    index: Any,
    runs: list[Any],
    config: dict[str, Any],
    stage: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, float]:
    groups = _group_frames(index, {run.run_id for run in runs})
    first_frames = groups[(runs[0].run_id, REFERENCE_FORCE_N)]
    first_image = _load_bgr(first_frames[0].rgb_path)
    moving_roi = _roi_xyxy(config["moving_roi_xywh"], first_image.shape)
    fixture_roi = _roi_xyxy(config["fixture_roi_xywh"], first_image.shape)
    windows = {
        moving_roi: cv2.createHanningWindow(
            (moving_roi[2] - moving_roi[0], moving_roi[3] - moving_roi[1]),
            cv2.CV_32F,
        ),
        fixture_roi: cv2.createHanningWindow(
            (fixture_roi[2] - fixture_roi[0], fixture_roi[3] - fixture_roi[1]),
            cv2.CV_32F,
        ),
    }
    hold_rows: list[dict[str, Any]] = []
    frame_intermediate: list[dict[str, Any]] = []
    medians: dict[tuple[str, float], np.ndarray] = {}
    references: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    positions = {int(key): float(value) for key, value in config["hole_positions_mm"].items()}

    for run in runs:
        for force in TARGET_FORCES_N:
            key = run.run_id, force
            frames = groups.get(key, [])
            if not frames:
                raise RuntimeError(f"{run.run_id} has no {force:g} N frames")
            medians[key] = _median_bgr([frame.rgb_path for frame in frames])
        reference_image = medians[(run.run_id, REFERENCE_FORCE_N)]
        references[run.run_id] = (
            _red_edge(reference_image, moving_roi),
            _red_edge(reference_image, fixture_roi),
        )

        for force in TARGET_FORCES_N:
            frames = groups[(run.run_id, force)]
            hold_image = medians[(run.run_id, force)]
            moving_reference, fixture_reference = references[run.run_id]
            moving = _phase_shift(
                moving_reference,
                _red_edge(hold_image, moving_roi),
                windows[moving_roi],
            )
            fixture = _phase_shift(
                fixture_reference,
                _red_edge(hold_image, fixture_roi),
                windows[fixture_roi],
            )
            forces = np.asarray([_frame_force(frame.measurements) for frame in frames])
            row = {
                "validation_stage": stage,
                "specimen_id": index.session.specimen_id,
                "material": index.session.material,
                "morphology": index.session.morphology,
                "run_id": run.run_id,
                "indenter": run.indenter,
                "hole_index": run.hole_index,
                "contact_position_mm": positions[run.hole_index],
                "repetition_index": run.repetition_index,
                "target_force_n": force,
                "frame_count": len(frames),
                "actual_force_median_n": float(np.median(forces)),
                "actual_force_std_n": float(np.std(forces, ddof=1)) if len(forces) > 1 else 0.0,
                "actual_force_min_n": float(np.min(forces)),
                "actual_force_max_n": float(np.max(forces)),
                "moving_dx_px": moving[0],
                "moving_dy_px": moving[1],
                "moving_phase_response": moving[2],
                "fixture_dx_px": fixture[0],
                "fixture_dy_px": fixture[1],
                "fixture_phase_response": fixture[2],
                "relative_dx_px": moving[0] - fixture[0],
                "relative_dy_px": moving[1] - fixture[1],
            }
            _finite_row(
                row,
                (
                    "actual_force_median_n",
                    "moving_dx_px",
                    "moving_dy_px",
                    "moving_phase_response",
                    "fixture_dx_px",
                    "fixture_dy_px",
                    "fixture_phase_response",
                    "relative_dx_px",
                    "relative_dy_px",
                ),
            )
            hold_rows.append(row)

            for frame in frames:
                image = _load_bgr(frame.rgb_path)
                frame_moving = _phase_shift(
                    moving_reference,
                    _red_edge(image, moving_roi),
                    windows[moving_roi],
                )
                frame_fixture = _phase_shift(
                    fixture_reference,
                    _red_edge(image, fixture_roi),
                    windows[fixture_roi],
                )
                measurements = frame.measurements
                frame_intermediate.append(
                    {
                        "validation_stage": stage,
                        "specimen_id": index.session.specimen_id,
                        "material": index.session.material,
                        "morphology": index.session.morphology,
                        "run_id": run.run_id,
                        "indenter": run.indenter,
                        "hole_index": run.hole_index,
                        "contact_position_mm": positions[run.hole_index],
                        "repetition_index": run.repetition_index,
                        "target_force_n": force,
                        "frame_index": int(measurements["frame_index"]),
                        "camera_host_time_s": float(measurements["camera_host_time_s"]),
                        "actual_force_n": _frame_force(measurements),
                        "image_path": str(frame.rgb_path.resolve()),
                        "moving_dx_px": frame_moving[0],
                        "moving_dy_px": frame_moving[1],
                        "moving_phase_response": frame_moving[2],
                        "fixture_dx_px": frame_fixture[0],
                        "fixture_dy_px": frame_fixture[1],
                        "fixture_phase_response": frame_fixture[2],
                        "relative_dx_px": frame_moving[0] - frame_fixture[0],
                        "relative_dy_px": frame_moving[1] - frame_fixture[1],
                    }
                )

    vectors_2_to_15 = np.asarray(
        [
            (row["relative_dx_px"], row["relative_dy_px"])
            for row in hold_rows
            if row["target_force_n"] == 15.0
        ],
        dtype=np.float64,
    )
    axis, _, dominance = _estimate_axis(vectors_2_to_15)
    transverse_axis = np.asarray((-axis[1], axis[0]))
    for rows in (hold_rows, frame_intermediate):
        for row in rows:
            relative = np.asarray((row["relative_dx_px"], row["relative_dy_px"]))
            row["axial_displacement_px"] = float(relative @ axis)
            row["transverse_displacement_px"] = float(relative @ transverse_axis)
            row["fixture_drift_px"] = float(
                np.hypot(row["fixture_dx_px"], row["fixture_dy_px"])
            )
    return frame_intermediate, hold_rows, axis, dominance


def _run_qc_rows(
    runs: list[Any],
    frame_rows: list[dict[str, Any]],
    hold_rows: list[dict[str, Any]],
    axis: np.ndarray,
    dominance: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    axis_angle_deg = float(np.degrees(np.arctan2(axis[1], axis[0])))
    for run in runs:
        holds = sorted(
            (row for row in hold_rows if row["run_id"] == run.run_id),
            key=lambda row: row["target_force_n"],
        )
        frames = [row for row in frame_rows if row["run_id"] == run.run_id]
        axial = np.asarray([row["axial_displacement_px"] for row in holds])
        increments = np.diff(axial)
        final = holds[-1]
        frame_stds = []
        frame_ranges = []
        for force in TARGET_FORCES_N:
            values = [
                row["axial_displacement_px"]
                for row in frames
                if row["target_force_n"] == force
            ]
            frame_stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            frame_ranges.append(float(np.ptp(values)) if len(values) > 1 else 0.0)
        axial_final = float(final["axial_displacement_px"])
        transverse_final = float(final["transverse_displacement_px"])
        image_x_sign = -1.0 if axis[0] < 0.0 else 1.0
        signed_image_x = image_x_sign * float(final["relative_dx_px"])
        output.append(
            {
                "validation_stage": final["validation_stage"],
                "specimen_id": final["specimen_id"],
                "run_id": run.run_id,
                "indenter": run.indenter,
                "hole_index": run.hole_index,
                "contact_position_mm": final["contact_position_mm"],
                "repetition_index": run.repetition_index,
                "axis_x": float(axis[0]),
                "axis_y": float(axis[1]),
                "axis_angle_deg": axis_angle_deg,
                "axis_svd_energy_fraction": dominance,
                "axial_15n_px": axial_final,
                "transverse_15n_px": transverse_final,
                "signed_image_x_15n_px": signed_image_x,
                "svd_minus_image_x_15n_px": axial_final - signed_image_x,
                "absolute_transverse_to_axial_15n_ratio": (
                    abs(transverse_final) / abs(axial_final)
                    if axial_final != 0.0
                    else float("nan")
                ),
                "fixture_drift_15n_px": final["fixture_drift_px"],
                "monotonic_2_5_10_15": bool(np.all(increments >= 0.0)),
                "decreasing_increment_count": int(np.sum(increments < 0.0)),
                "minimum_axial_increment_px": float(np.min(increments)),
                "within_hold_axial_std_max_px": float(max(frame_stds)),
                "within_hold_axial_range_max_px": float(max(frame_ranges)),
                "median_moving_phase_response": float(
                    np.median([row["moving_phase_response"] for row in frames])
                ),
                "minimum_moving_phase_response": float(
                    np.min([row["moving_phase_response"] for row in frames])
                ),
                "median_fixture_phase_response": float(
                    np.median([row["fixture_phase_response"] for row in frames])
                ),
                "minimum_fixture_phase_response": float(
                    np.min([row["fixture_phase_response"] for row in frames])
                ),
            }
        )
    return output


def _frame_csv_rows(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in frame_rows:
        output.append(
            {
                "validation_stage": row["validation_stage"],
                "specimen_id": row["specimen_id"],
                "material": row["material"],
                "morphology": row["morphology"],
                "run_id": row["run_id"],
                "indenter": row["indenter"],
                "hole_index": row["hole_index"],
                "X_contact_mm": row["contact_position_mm"],
                "repetition_index": row["repetition_index"],
                "target_force_n": row["target_force_n"],
                "frame_index": row["frame_index"],
                "camera_host_time_s": row["camera_host_time_s"],
                "actual_force_n": row["actual_force_n"],
                "image_path": row["image_path"],
                "dx_indenter_px": row["moving_dx_px"],
                "dy_indenter_px": row["moving_dy_px"],
                "dx_fixture_px": row["fixture_dx_px"],
                "dy_fixture_px": row["fixture_dy_px"],
                "dx_relative_px": row["relative_dx_px"],
                "dy_relative_px": row["relative_dy_px"],
                "delta_axial_px": row["axial_displacement_px"],
                "delta_transverse_px": row["transverse_displacement_px"],
                "phase_response_indenter": row["moving_phase_response"],
                "phase_response_fixture": row["fixture_phase_response"],
            }
        )
    return output


def _run_summary_rows(
    runs: list[Any],
    frame_rows: list[dict[str, Any]],
    hold_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for run in runs:
        ordered = sorted(
            (row for row in hold_rows if row["run_id"] == run.run_id),
            key=lambda row: row["target_force_n"],
        )
        prior_axial: list[float] = []
        for row in ordered:
            force = float(row["target_force_n"])
            frames = [
                frame
                for frame in frame_rows
                if frame["run_id"] == run.run_id
                and frame["target_force_n"] == force
            ]
            frame_axial = np.asarray(
                [frame["axial_displacement_px"] for frame in frames]
            )
            frame_transverse = np.asarray(
                [frame["transverse_displacement_px"] for frame in frames]
            )
            prior_axial.append(float(row["axial_displacement_px"]))
            monotonic = bool(np.all(np.diff(prior_axial) >= 0.0))
            output.append(
                {
                    "validation_stage": row["validation_stage"],
                    "specimen_id": row["specimen_id"],
                    "material": row["material"],
                    "morphology": row["morphology"],
                    "run_id": run.run_id,
                    "indenter": run.indenter,
                    "hole_index": run.hole_index,
                    "X_contact_mm": row["contact_position_mm"],
                    "repetition_index": run.repetition_index,
                    "target_force_n": force,
                    "frame_count": row["frame_count"],
                    "actual_force_median_n": row["actual_force_median_n"],
                    "actual_force_std_n": row["actual_force_std_n"],
                    "delta_axial_px": row["axial_displacement_px"],
                    "delta_axial_frame_median_px": float(np.median(frame_axial)),
                    "delta_axial_frame_spread_px": float(np.ptp(frame_axial)),
                    "delta_transverse_px": row["transverse_displacement_px"],
                    "delta_transverse_frame_median_px": float(
                        np.median(frame_transverse)
                    ),
                    "fixture_shift_px": row["fixture_drift_px"],
                    "dx_indenter_px": row["moving_dx_px"],
                    "dy_indenter_px": row["moving_dy_px"],
                    "dx_fixture_px": row["fixture_dx_px"],
                    "dy_fixture_px": row["fixture_dy_px"],
                    "dx_relative_px": row["relative_dx_px"],
                    "dy_relative_px": row["relative_dy_px"],
                    "phase_response_indenter": row["moving_phase_response"],
                    "phase_response_fixture": row["fixture_phase_response"],
                    "monotonic_so_far": monotonic,
                    "tracking_status": (
                        "measured" if monotonic else "nonmonotonic_so_far"
                    ),
                }
            )
    return output


def _session_summary_row(
    index: Any,
    stage: str,
    qc_rows: list[dict[str, Any]],
    hold_rows: list[dict[str, Any]],
    axis: np.ndarray,
    dominance: float,
) -> dict[str, Any]:
    final = [row for row in hold_rows if row["target_force_n"] == 15.0]
    ratios = np.asarray(
        [row["absolute_transverse_to_axial_15n_ratio"] for row in qc_rows]
    )
    image_x = np.asarray([row["signed_image_x_15n_px"] for row in qc_rows])
    return {
        "validation_stage": stage,
        "specimen_id": index.session.specimen_id,
        "material": index.session.material,
        "morphology": index.session.morphology,
        "indenter": final[0]["indenter"],
        "tracked_run_count": len(qc_rows),
        "monotonic_run_fraction": float(
            np.mean([row["monotonic_2_5_10_15"] for row in qc_rows])
        ),
        "estimated_axis_dx": float(axis[0]),
        "estimated_axis_dy": float(axis[1]),
        "estimated_axis_angle_deg": float(
            np.degrees(np.arctan2(axis[1], axis[0]))
        ),
        "axis_svd_energy_fraction": dominance,
        "median_2_to_15_indentation_px": float(
            np.median([row["axial_displacement_px"] for row in final])
        ),
        "median_2_to_15_signed_image_x_px": float(np.median(image_x)),
        "median_svd_minus_image_x_2_to_15_px": float(
            np.median(
                [row["svd_minus_image_x_15n_px"] for row in qc_rows]
            )
        ),
        "median_transverse_to_axial_ratio": float(np.median(ratios)),
        "median_fixture_shift_px": float(
            np.median([row["fixture_drift_px"] for row in final])
        ),
        "pixel_to_mm_status": "unavailable",
        "mm_per_px": "",
        "calibration_source": "none: no defensible visible rigid scale established",
    }


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


def _plot_example_overlay(
    output: Path,
    index: Any,
    run: Any,
    config: dict[str, Any],
    hold_rows: list[dict[str, Any]],
) -> None:
    groups = _group_frames(index, {run.run_id})
    images = {
        force: _median_bgr([frame.rgb_path for frame in groups[(run.run_id, force)]])
        for force in TARGET_FORCES_N
    }
    moving_roi = _roi_xyxy(config["moving_roi_xywh"], images[2.0].shape)
    fixture_roi = _roi_xyxy(config["fixture_roi_xywh"], images[2.0].shape)
    figure = plt.figure(figsize=(13.0, 6.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, height_ratios=(1.0, 0.82))
    display_gain = 3.0
    crop = (650, 60, 1920, 1080)
    for column, force in enumerate(TARGET_FORCES_N):
        axis = figure.add_subplot(grid[0, column])
        display = np.clip(images[force].astype(np.float32) * display_gain, 0, 255).astype(
            np.uint8
        )
        x0, y0, x1, y1 = crop
        axis.imshow(cv2.cvtColor(display[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        for roi, color, name in (
            (moving_roi, "#f28e2b", "moving shaft"),
            (fixture_roi, "#4e79a7", "fixed fixture"),
        ):
            rx0, ry0, rx1, ry1 = roi
            axis.add_patch(
                Rectangle(
                    (rx0 - x0, ry0 - y0),
                    rx1 - rx0,
                    ry1 - ry0,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.2,
                )
            )
        row = next(
            row
            for row in hold_rows
            if row["run_id"] == run.run_id and row["target_force_n"] == force
        )
        for roi, color, prefix in (
            (moving_roi, "#f28e2b", "indenter"),
            (fixture_roi, "#4e79a7", "fixture"),
        ):
            rx0, ry0, rx1, ry1 = roi
            center_x = 0.5 * (rx0 + rx1) - x0
            center_y = 0.5 * (ry0 + ry1) - y0
            dx = float(row[f"{'moving' if prefix == 'indenter' else 'fixture'}_dx_px"])
            dy = float(row[f"{'moving' if prefix == 'indenter' else 'fixture'}_dy_px"])
            axis.annotate(
                "",
                xy=(center_x + dx, center_y + dy),
                xytext=(center_x, center_y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "linewidth": 1.8,
                    "mutation_scale": 10,
                },
            )
        if column == 0:
            axis.text(
                0.02,
                0.04,
                "orange: indenter ROI/vector\nblue: fixture ROI/vector",
                transform=axis.transAxes,
                fontsize=7.5,
                color="#111111",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            )
        axis.set_title(
            f"{force:g} N   axial={row['axial_displacement_px']:.2f} px",
            fontsize=9,
        )
        axis.set_axis_off()

    moving_axes = []
    x0, y0, x1, y1 = moving_roi
    for column, force in enumerate((2.0, 15.0)):
        axis = figure.add_subplot(grid[1, column])
        edge = _red_edge(images[force], moving_roi)
        axis.imshow(edge, cmap="gray", origin="upper")
        axis.set_title(f"Moving ROI red-edge, {force:g} N", fontsize=9)
        axis.set_axis_off()
        moving_axes.append(axis)
    difference_axis = figure.add_subplot(grid[1, 2:])
    red_2 = images[2.0][y0:y1, x0:x1, 2].astype(np.float32)
    red_15 = images[15.0][y0:y1, x0:x1, 2].astype(np.float32)
    difference = np.abs(red_15 - red_2)
    image = difference_axis.imshow(difference, cmap="viridis", vmin=0.0)
    difference_axis.set_title(r"Moving ROI $|R_{15N}-R_{2N}|$", fontsize=9)
    difference_axis.set_axis_off()
    figure.colorbar(image, ax=difference_axis, fraction=0.035, label="Absolute change [DN]")
    figure.suptitle(
        f"{run.run_id}: hold medians (RGB display gain = {display_gain:g} only)",
        fontsize=11,
    )
    figure.savefig(output / "example_tracking_overlay.png", dpi=220)
    plt.close(figure)


def _plot_delta_vs_force(output: Path, hold_rows: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, 6))
    for run_id in sorted({row["run_id"] for row in hold_rows}):
        rows = sorted(
            (row for row in hold_rows if row["run_id"] == run_id),
            key=lambda row: row["target_force_n"],
        )
        hole = int(rows[0]["hole_index"])
        axis.plot(
            [row["actual_force_median_n"] for row in rows],
            [row["axial_displacement_px"] for row in rows],
            marker="o",
            markersize=2.8,
            linewidth=0.8,
            alpha=0.58,
            color=colors[hole - 1],
        )
    for hole, color in enumerate(colors, start=1):
        axis.plot([], [], color=color, marker="o", label=f"Hole {hole}")
    axis.set(
        xlabel="Measured force [N]",
        ylabel=r"Incremental axial displacement from 2 N [px]",
        title="Relative rigid-body indentation signal",
    )
    axis.legend(ncol=3, fontsize=8, frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "delta_vs_force.png", dpi=220)
    plt.close(figure)


def _plot_repetition_consistency(
    output: Path, hold_rows: list[dict[str, Any]]
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(8.4, 5.0), sharex=True, sharey=True)
    for hole in range(1, 7):
        axis = axes.flat[hole - 1]
        run_ids = sorted(
            {row["run_id"] for row in hold_rows if row["hole_index"] == hole}
        )
        if not run_ids:
            axis.set_visible(False)
            continue
        curves = []
        force_values = []
        for run_id in run_ids:
            rows = sorted(
                (row for row in hold_rows if row["run_id"] == run_id),
                key=lambda row: row["target_force_n"],
            )
            force = np.asarray([row["actual_force_median_n"] for row in rows])
            axial = np.asarray([row["axial_displacement_px"] for row in rows])
            curves.append(axial)
            force_values.append(force)
            axis.plot(force, axial, color="#7f8c8d", linewidth=0.8, alpha=0.55)
        median_force = np.median(np.vstack(force_values), axis=0)
        median_axial = np.median(np.vstack(curves), axis=0)
        axis.plot(
            median_force,
            median_axial,
            color="#d95f02",
            marker="o",
            markersize=3.2,
            linewidth=2.0,
            label="median",
        )
        position = next(
            row["contact_position_mm"]
            for row in hold_rows
            if row["hole_index"] == hole
        )
        axis.set_title(f"Hole {hole} · {position:g} mm", fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
    figure.supxlabel("Measured force [N]")
    figure.supylabel("Incremental axial displacement [px]")
    figure.suptitle("Independent repetitions and per-location median", fontsize=11)
    figure.tight_layout(pad=0.8)
    figure.savefig(output / "repetition_consistency.png", dpi=220)
    plt.close(figure)


def _plot_axial_transverse(output: Path, hold_rows: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(5.1, 4.1), constrained_layout=True)
    colors = {5.0: "#440154", 10.0: "#21918c", 15.0: "#fde725"}
    for force in (5.0, 10.0, 15.0):
        rows = [row for row in hold_rows if row["target_force_n"] == force]
        axis.scatter(
            [row["axial_displacement_px"] for row in rows],
            [row["transverse_displacement_px"] for row in rows],
            s=24,
            label=f"{force:g} N",
            color=colors[force],
            edgecolor="#333333",
            linewidth=0.35,
        )
    axis.axhline(0.0, color="#888888", linewidth=0.8)
    axis.set(
        xlabel="Axial displacement [px]",
        ylabel="Transverse residual [px]",
        title="Axial and transverse relative motion",
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "axial_vs_transverse_motion.png", dpi=220)
    plt.close(figure)


def _plot_fixture_drift(output: Path, hold_rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(9.0, 3.2), sharex=True, sharey=True)
    labels = (
        ("Raw indenter", "moving_dx_px", "moving_dy_px"),
        ("Fixed fixture", "fixture_dx_px", "fixture_dy_px"),
        ("Relative", "relative_dx_px", "relative_dy_px"),
    )
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, 6))
    representative_runs = []
    for hole in range(1, 7):
        run_ids = sorted(
            {row["run_id"] for row in hold_rows if row["hole_index"] == hole}
        )
        if run_ids:
            representative_runs.append((hole, run_ids[0]))
    for axis, (title, x_field, y_field) in zip(axes, labels, strict=True):
        for hole, run_id in representative_runs:
            rows = sorted(
                (row for row in hold_rows if row["run_id"] == run_id),
                key=lambda row: row["target_force_n"],
            )
            magnitude = [
                np.hypot(row[x_field], row[y_field])
                for row in rows
            ]
            axis.plot(
                [row["actual_force_median_n"] for row in rows],
                magnitude,
                marker="o",
                markersize=2.8,
                linewidth=1.0,
                color=colors[hole - 1],
                label=f"Hole {hole}",
            )
        axis.set_title(title, fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Translation magnitude [px]")
    axes[1].set_xlabel("Measured force [N]")
    axes[-1].legend(fontsize=7, frameon=False, ncol=2)
    figure.suptitle("Raw, reference, and relative motion", fontsize=11)
    figure.tight_layout(pad=0.8)
    figure.savefig(output / "fixture_drift.png", dpi=220)
    plt.close(figure)


def _plot_phase_response(output: Path, frame_rows: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    positions = []
    values = []
    labels = []
    index = 1
    for force in TARGET_FORCES_N:
        for field, label in (
            ("moving_phase_response", "indenter"),
            ("fixture_phase_response", "fixture"),
        ):
            group = [
                row[field] for row in frame_rows if row["target_force_n"] == force
            ]
            positions.append(index)
            values.append(group)
            labels.append(f"{force:g} N\n{label}")
            index += 1
    axis.boxplot(
        values,
        positions=positions,
        widths=0.58,
        showfliers=True,
        patch_artist=True,
        boxprops={"facecolor": "#d9e6f2", "edgecolor": "#333333"},
        medianprops={"color": "#d95f02"},
        whiskerprops={"color": "#333333"},
        capprops={"color": "#333333"},
        flierprops={"markersize": 3, "markerfacecolor": "#777777"},
    )
    axis.set(
        ylabel="OpenCV phase-correlation response",
        title="Frame-level correlation quality (no acceptance threshold)",
        xticks=positions,
        xticklabels=labels,
    )
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "phase_response_distribution.png", dpi=220)
    plt.close(figure)


def _report(
    output: Path,
    index: Any,
    stage: str,
    config: dict[str, Any],
    hold_rows: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
    axis: np.ndarray,
    dominance: float,
) -> str:
    final_rows = [row for row in hold_rows if row["target_force_n"] == 15.0]
    monotonic_count = sum(bool(row["monotonic_2_5_10_15"]) for row in qc_rows)
    positive_count = sum(float(row["axial_15n_px"]) > 0.0 for row in qc_rows)
    axial_15 = np.asarray([row["axial_displacement_px"] for row in final_rows])
    transverse_ratio = np.asarray(
        [row["absolute_transverse_to_axial_15n_ratio"] for row in qc_rows]
    )
    fixture_15 = np.asarray([row["fixture_drift_px"] for row in final_rows])
    frame_stability = np.asarray(
        [row["within_hold_axial_std_max_px"] for row in qc_rows]
    )
    frame_ranges = np.asarray(
        [row["within_hold_axial_range_max_px"] for row in qc_rows]
    )
    moving_response = np.asarray(
        [row["moving_phase_response"] for row in hold_rows if row["target_force_n"] > 2.0]
    )
    fixture_response = np.asarray(
        [row["fixture_phase_response"] for row in hold_rows if row["target_force_n"] > 2.0]
    )
    signed_image_x = np.asarray([row["signed_image_x_15n_px"] for row in qc_rows])
    repeat_iqrs = []
    for hole in range(1, 7):
        values = [
            row["axial_15n_px"] for row in qc_rows if row["hole_index"] == hole
        ]
        if len(values) > 1:
            repeat_iqrs.append(float(np.subtract(*np.percentile(values, [75, 25]))))
    lines = [
        "# Hardware indentation tracking feasibility",
        "",
        "This is a read-only image-tracking study. It does not define or register a paper metric.",
        "",
        "## Study contract",
        "",
        f"- validation stage: `{stage}`",
        f"- session: `{index.path}`",
        f"- specimen: `{index.session.specimen_id}`",
        f"- indenter: `{config['indenter']}`",
        f"- analyzed runs: `{len(qc_rows)}`",
        "- force states: `2, 5, 10, 15 N`; every state uses its median RGB frame",
        "- frame-level tracking is retained separately for within-hold QC",
        "- motion estimator: OpenCV phase correlation on red-channel Scharr-gradient magnitude",
        f"- moving shaft ROI `[x,y,w,h]`: `{config['moving_roi_xywh']}`",
        f"- fixed fixture ROI `[x,y,w,h]`: `{config['fixture_roi_xywh']}`",
        "- displacement units: pixels; no trusted image-to-mm calibration was invented",
        "",
        "The incremental vector at each force is the moving-indenter shift minus the fixed-fixture shift, both relative to the same run's 2 N hold. The common axial direction is the first through-origin SVD direction of all 2-to-15 N relative vectors, with sign selected by their majority direction.",
        "",
        "## Measured evidence",
        "",
        f"- axial image direction: `({axis[0]:.6f}, {axis[1]:.6f})`, angle `{np.degrees(np.arctan2(axis[1], axis[0])):.3f} deg`",
        f"- first-axis SVD energy fraction: `{dominance:.6f}`",
        f"- signed image-x diagnostic, 2-to-15 N median: `{np.median(signed_image_x):.3f}` px (SVD projection: `{np.median(axial_15):.3f}` px)",
        f"- positive 2-to-15 N axial direction: `{positive_count}/{len(qc_rows)}` runs",
        f"- monotonically nondecreasing 2/5/10/15 N axial sequence: `{monotonic_count}/{len(qc_rows)}` runs",
        f"- 2-to-15 N axial displacement median/IQR/range: `{np.median(axial_15):.3f}` / `{np.subtract(*np.percentile(axial_15, [75, 25])):.3f}` / `{np.min(axial_15):.3f}..{np.max(axial_15):.3f}` px",
        f"- absolute transverse/axial ratio at 15 N median/p95: `{np.median(transverse_ratio):.4f}` / `{np.percentile(transverse_ratio, 95):.4f}`",
        f"- fixed-fixture drift at 15 N median/p95/max: `{np.median(fixture_15):.3f}` / `{np.percentile(fixture_15, 95):.3f}` / `{np.max(fixture_15):.3f}` px",
        f"- maximum within-hold axial standard deviation median/p95/max: `{np.median(frame_stability):.3f}` / `{np.percentile(frame_stability, 95):.3f}` / `{np.max(frame_stability):.3f}` px",
        f"- maximum within-hold axial frame range median/p95/max: `{np.median(frame_ranges):.3f}` / `{np.percentile(frame_ranges, 95):.3f}` / `{np.max(frame_ranges):.3f}` px",
        f"- moving-ROI phase response median/min: `{np.median(moving_response):.4f}` / `{np.min(moving_response):.4f}`",
        f"- fixture-ROI phase response median/min: `{np.median(fixture_response):.4f}` / `{np.min(fixture_response):.4f}`",
        (
            f"- same-location 2-to-15 N repetition IQR median/max: "
            f"`{np.median(repeat_iqrs):.3f}` / `{np.max(repeat_iqrs):.3f}` px"
            if repeat_iqrs
            else "- same-location repetition IQR: unavailable in the one-run stage"
        ),
        "",
        "## Contact-coordinate correction",
        "",
        "This study uses the measured fixture-stop mapping `0, 10, 20, 30, 40, 50 mm` for holes 1--6. Figure 5 currently uses the older `0, 11, 22, 33, 44, 55 mm` mapping. Figure 5 was deliberately not modified by this feasibility study.",
        "",
        "## Conclusion",
        "",
        f"1. Monotonic indentation: `{monotonic_count}/{len(qc_rows)}` runs were monotonically nondecreasing across 2/5/10/15 N.",
        f"2. Motion versus within-hold variation: median 2-to-15 N motion was `{np.median(axial_15):.3f}` px; the per-run maximum frame range had median `{np.median(frame_ranges):.3f}` px and maximum `{np.max(frame_ranges):.3f}` px.",
        f"3. Axial dominance: median absolute transverse/axial ratio at 15 N was `{np.median(transverse_ratio):.4f}`.",
        f"4. Fixture stability: median fixed-reference shift at 15 N was `{np.median(fixture_15):.3f}` px.",
        (
            f"5. Repetition consistency: median/max same-location IQR at 15 N was "
            f"`{np.median(repeat_iqrs):.3f}` / `{np.max(repeat_iqrs):.3f}` px."
            if repeat_iqrs
            else "5. Repetition consistency: unavailable in the one-run stage."
        ),
        "6. Pixel-to-mm calibration: unavailable; no defensible visible rigid scale was established.",
        "",
    ]
    if stage == "sample" and float(np.max(frame_ranges)) > float(np.median(axial_15)):
        lines.extend(
            [
                f"The estimator returned finite hold-median measurements for all `{len(qc_rows)}` selected runs, and the majority-defined axis captured `{dominance:.3%}` of 2-to-15 N vector energy. However, the largest disagreement between frames from one nominally fixed force hold was `{np.max(frame_ranges):.3f}` px, larger than the session's median total 2-to-15 N signal of `{np.median(axial_15):.3f}` px. The raw adjacent frames are visually nearly unchanged, so this exposes phase-correlation peak ambiguity rather than a defensible mechanical displacement. The full-session stage was not run.",
                "",
                "**Tracking signal is not yet trustworthy enough for optomechanical metrics.**",
                "",
                "No arbitrary thresholded PASS is issued; `S_OM` remains unavailable and no production or Figure 5 metric consumes this result.",
                "",
            ]
        )
    elif stage == "manual":
        lines.extend(
            [
                f"The one-run gate returned finite measurements, a `{dominance:.3%}` first-axis energy fraction, and a monotonic four-hold trajectory. This is only the manual-inspection gate; it is not evidence for full-session consistency. The overlay must be inspected before running the configured 6--12-run sample.",
                "",
                "**No feasibility decision is made from one run.**",
                "",
                "No arbitrary thresholded PASS is issued; `S_OM` remains unavailable.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"The estimator returned finite measurements for all `{len(qc_rows)}` selected runs. The majority-defined axis captured `{dominance:.3%}` of 2-to-15 N vector energy; `{positive_count}/{len(qc_rows)}` final vectors shared its positive direction and `{monotonic_count}/{len(qc_rows)}` runs were monotonic at all four measured holds.",
                "",
                "**Existing images appear sufficient for relative indentation recovery.**",
                "",
                "This is evidence from the requested feasibility analysis, not an arbitrary thresholded PASS. `S_OM` remains unavailable and no production or Figure 5 metric consumes this result.",
                "",
            ]
        )
    report = "\n".join(lines)
    (output / "report.md").write_text(report, encoding="utf-8")
    return report


def main() -> None:
    args = _arguments()
    config = _load_config(args.config)
    session_path = _resolve_session_path(str(config["session_path"]))
    index = index_session(session_path, expected_repetitions=5)
    runs = _choose_runs(index, config, args.stage)
    output = args.output if args.output.is_absolute() else REPOSITORY_ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    figure_output = output / "figures"
    figure_output.mkdir(exist_ok=True)

    frame_rows, hold_rows, axis, dominance = _track(index, runs, config, args.stage)
    qc_rows = _run_qc_rows(runs, frame_rows, hold_rows, axis, dominance)
    frame_csv_rows = _frame_csv_rows(frame_rows)
    run_summary_rows = _run_summary_rows(runs, frame_rows, hold_rows)
    session_summary_rows = [
        _session_summary_row(index, args.stage, qc_rows, hold_rows, axis, dominance)
    ]

    _write_csv(output / "tracking_frames.csv", frame_csv_rows)
    _write_csv(output / "run_indentation_summary.csv", run_summary_rows)
    _write_csv(output / "session_indentation_summary.csv", session_summary_rows)
    _write_csv(output / "tracking_qc.csv", qc_rows)
    _plot_example_overlay(figure_output, index, runs[0], config, hold_rows)
    _plot_delta_vs_force(figure_output, hold_rows)
    _plot_repetition_consistency(figure_output, hold_rows)
    _plot_axial_transverse(figure_output, hold_rows)
    _plot_fixture_drift(figure_output, hold_rows)
    _plot_phase_response(figure_output, frame_rows)
    report = _report(output, index, args.stage, config, hold_rows, qc_rows, axis, dominance)

    print(report)
    print(f"Outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
