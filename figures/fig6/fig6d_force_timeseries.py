"""Build the representative multi-location proprioceptive force time series."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from algorithm.contact_localization import (  # noqa: E402
    ContactLocalizationConfig,
    build_unloaded_reference,
    canonical_position_to_mm,
    causal_median_response,
    compute_longitudinal_response,
    localize_response_profile,
)
from experiments.localization.fixed_finger_calibration import (  # noqa: E402
    calibrate_fixed_finger,
    warp_with_fixed_finger_calibration,
)
from lumo.visualization import (  # noqa: E402
    DEFAULT_STYLE,
    MATERIAL_LABELS,
    PAPER_LABELS,
    publication_context,
    save_figure,
)


FIGURE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = REPOSITORY_ROOT / "output" / "proprioceptive_contact_dataset"
DEFAULT_OUTPUT_STEM = FIGURE_DIRECTORY / "fig6d_force_timeseries_multilocation"
DEFAULT_RUN_IDS = tuple(f"run_{index:03d}" for index in range(5, 11))
EXPECTED_LOCATIONS_MM = tuple(float(value) for value in range(0, 51, 10))
CALIBRATION_RUN_ID = "run_005"
CONTACT_FORCE_THRESHOLD_N = 2.0
UNLOADED_FORCE_THRESHOLD_N = 1.0
ROTATION_AXIS_TO_LED1_MM = 103.6
MINIMUM_MOMENT_ARM_MM = 1.0
ANALYSIS_VERSION = 1
EXPERIMENT_LABEL = f"{MATERIAL_LABELS['solaris']} {PAPER_LABELS['flat_opt']}"
TORQUE_TRACE_COLOR = "#A9ADB2"


def _nearest_indices(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if len(reference) < 2:
        raise ValueError("reference time series must contain at least two samples")
    indices = np.searchsorted(reference, query)
    indices = np.clip(indices, 1, len(reference) - 1)
    previous = indices - 1
    use_previous = np.abs(reference[previous] - query) <= np.abs(
        reference[indices] - query
    )
    return np.where(use_previous, previous, indices)


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _initial_unloaded_indices(force_n: np.ndarray) -> np.ndarray:
    contact = np.flatnonzero(force_n >= CONTACT_FORCE_THRESHOLD_N)
    if not len(contact):
        raise RuntimeError("run never reaches the 2 N contact threshold")
    candidates = np.flatnonzero(
        (np.arange(len(force_n)) < contact[0]) & (force_n < UNLOADED_FORCE_THRESHOLD_N)
    )
    if len(candidates) < 3:
        raise RuntimeError(
            "run requires at least three initial camera samples below 1 N"
        )
    return candidates


def _load_run(run_directory: Path) -> dict[str, object]:
    metadata = json.loads((run_directory / "metadata.json").read_text())
    camera = pd.read_csv(run_directory / "camera_timestamps.csv")
    motor = pd.read_csv(run_directory / "motor.csv")
    ft = pd.read_csv(run_directory / "ft.csv")
    force_sequence = pd.read_csv(run_directory / "force_sequence.csv")
    if metadata.get("status") != "complete":
        raise RuntimeError(f"{run_directory.name} is not complete")
    if len(camera) < 3 or len(motor) < 3 or len(ft) < 3 or len(force_sequence) < 3:
        raise RuntimeError(f"{run_directory.name} has an incomplete sensor stream")
    force_n = camera["ft_contact_force_N"].to_numpy(dtype=np.float64)
    unloaded_indices = _initial_unloaded_indices(force_n)
    return {
        "directory": run_directory,
        "metadata": metadata,
        "camera": camera,
        "motor": motor,
        "ft": ft,
        "force_sequence": force_sequence,
        "force_n": force_n,
        "unloaded_indices": unloaded_indices,
    }


def _validate_runs(runs: list[dict[str, object]], run_ids: tuple[str, ...]) -> None:
    observed_ids = tuple(str(run["metadata"]["run_id"]) for run in runs)
    if observed_ids != run_ids:
        raise RuntimeError(f"unexpected run order: {observed_ids}")
    observed_locations = tuple(
        float(run["metadata"]["experiment"]["contact_location_gt_mm"]) for run in runs
    )
    if observed_locations != EXPECTED_LOCATIONS_MM:
        raise RuntimeError(
            "expected contact locations "
            f"{EXPECTED_LOCATIONS_MM}, got {observed_locations}"
        )
    for run in runs:
        experiment = run["metadata"]["experiment"]
        expected = 10.0 * (int(experiment["hole_index"]) - 1)
        if float(experiment["contact_location_gt_mm"]) != expected:
            raise RuntimeError(f"{run['directory'].name} violates x_GT = 10(h-1) mm")
        if tuple(float(value) for value in experiment["target_forces_N"]) != (
            2.0,
            5.0,
            10.0,
            15.0,
            20.0,
        ):
            raise RuntimeError(f"{run['directory'].name} has unexpected force targets")
        sequence = run["force_sequence"]
        if float(sequence["actual_force_N"].iloc[0]) >= UNLOADED_FORCE_THRESHOLD_N:
            raise RuntimeError(f"{run['directory'].name} does not begin unloaded")
        if float(sequence["actual_force_N"].iloc[-1]) >= CONTACT_FORCE_THRESHOLD_N:
            raise RuntimeError(
                f"{run['directory'].name} does not include final release"
            )


def _reference_rgb(run: dict[str, object]) -> np.ndarray:
    directory = run["directory"]
    camera = run["camera"]
    frames = [
        _read_rgb(directory / camera.iloc[index]["filename"])
        for index in run["unloaded_indices"]
    ]
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def _torque_bias_nm(run: dict[str, object]) -> tuple[float, int]:
    camera = run["camera"]
    motor = run["motor"]
    ft = run["ft"]
    first_contact_index = int(
        np.flatnonzero(run["force_n"] >= CONTACT_FORCE_THRESHOLD_N)[0]
    )
    first_contact_time_ns = int(camera.iloc[first_contact_index]["timestamp_ns"])
    motor_time = motor["timestamp_ns"].to_numpy(dtype=np.int64)
    ft_time = ft["timestamp_ns"].to_numpy(dtype=np.int64)
    ft_force = np.sqrt(
        ft["fx_N"].to_numpy(dtype=np.float64) ** 2
        + ft["fy_N"].to_numpy(dtype=np.float64) ** 2
        + ft["fz_N"].to_numpy(dtype=np.float64) ** 2
    )
    ft_indices = _nearest_indices(motor_time, ft_time)
    unloaded = (
        (motor_time < first_contact_time_ns)
        & (ft_force[ft_indices] < UNLOADED_FORCE_THRESHOLD_N)
        & (np.abs(motor["velocity_rad_s"].to_numpy(dtype=np.float64)) < 0.15)
    )
    if np.count_nonzero(unloaded) < 3:
        raise RuntimeError(
            f"{run['directory'].name} has fewer than three unloaded torque samples"
        )
    return (
        float(np.median(motor["torque_Nm"].to_numpy(dtype=np.float64)[unloaded])),
        int(np.count_nonzero(unloaded)),
    )


def _optical_locations_mm(
    run: dict[str, object],
    fixed_calibration: object,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    directory = run["directory"]
    camera = run["camera"]
    unloaded_indices = run["unloaded_indices"]
    canonical_unloaded = [
        warp_with_fixed_finger_calibration(
            _read_rgb(directory / camera.iloc[index]["filename"]),
            fixed_calibration,
        )
        for index in unloaded_indices
    ]
    config = ContactLocalizationConfig(
        unloaded_frame_count=len(canonical_unloaded),
        temporal_median_window=3,
    )
    reference = build_unloaded_reference(canonical_unloaded, config)
    raw_position = np.full(len(camera), np.nan, dtype=np.float64)
    statuses: list[str] = []
    response_history: list[np.ndarray] = []
    for index, filename in enumerate(camera["filename"]):
        canonical = warp_with_fixed_finger_calibration(
            _read_rgb(directory / filename), fixed_calibration
        )
        response, saturation_ge_250, saturation_eq_255 = compute_longitudinal_response(
            canonical,
            reference.canonical_rgb,
            brightest_fraction=config.brightest_fraction,
            smoothing_sigma=config.longitudinal_smoothing_sigma,
        )
        response_history.append(response)
        filtered = causal_median_response(
            response_history, config.temporal_median_window
        )
        result = localize_response_profile(
            filtered,
            reference,
            fixed_calibration.led_longitudinal_fractions,
            config,
            saturation_fraction_ge_250=saturation_ge_250,
            saturation_fraction_eq_255=saturation_eq_255,
        )
        statuses.append(result.status)
        if (
            not result.contact_detected
            or result.total_evidence_dn <= 0.0
            or result.status == "invalid: saturated selected pixels"
        ):
            continue
        peak_coordinate = float(np.argmax(result.evidence_profile)) / (
            len(result.evidence_profile) - 1
        )
        try:
            raw_position[index] = canonical_position_to_mm(
                peak_coordinate,
                fixed_calibration.led_longitudinal_fractions,
                maximum_extrapolation_mm=config.maximum_extrapolation_mm,
            )
        except ValueError:
            continue

    finite = np.isfinite(raw_position)
    if np.count_nonzero(finite) < 3:
        raise RuntimeError(
            f"{directory.name} produced fewer than three optical location estimates"
        )
    filled_position = np.interp(
        np.arange(len(raw_position), dtype=np.float64),
        np.flatnonzero(finite).astype(np.float64),
        raw_position[finite],
    )
    return raw_position, filled_position, statuses


def build_force_timeseries(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    run_ids: tuple[str, ...] = DEFAULT_RUN_IDS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compute one audited six-location force trace from raw recorded runs."""

    root = Path(dataset_root)
    runs = [_load_run(root / run_id) for run_id in run_ids]
    _validate_runs(runs, run_ids)

    calibration_run = runs[run_ids.index(CALIBRATION_RUN_ID)]
    fixed_calibration = calibrate_fixed_finger(
        _reference_rgb(calibration_run),
        distal_orientation="minimum_longitudinal",
    )

    tables: list[pd.DataFrame] = []
    display_offset_s = 0.0
    for run in runs:
        camera = run["camera"].copy()
        sequence = run["force_sequence"].copy()
        camera_time = camera["timestamp_ns"].to_numpy(dtype=np.int64)
        sequence_time = sequence["timestamp_ns"].to_numpy(dtype=np.int64)
        torque_nm = sequence["motor_torque_Nm"].to_numpy(dtype=np.float64)
        motor_time = sequence["motor_timestamp_ns"].to_numpy(dtype=np.int64)
        motor_delta_ms = np.abs(motor_time - sequence_time) / 1.0e6
        tau0_nm, tau0_sample_count = _torque_bias_nm(run)
        raw_location_mm, filled_location_mm, statuses = _optical_locations_mm(
            run, fixed_calibration
        )
        location_mm = np.interp(sequence_time, camera_time, filled_location_mm)
        camera_indices = _nearest_indices(sequence_time, camera_time)
        nearest_raw_location_mm = raw_location_mm[camera_indices]
        moment_arm_mm = ROTATION_AXIS_TO_LED1_MM - location_mm
        if np.any(moment_arm_mm <= MINIMUM_MOMENT_ARM_MM):
            raise RuntimeError("optical location produced a non-physical moment arm")
        run_time_s = (sequence_time - sequence_time[0]) / 1.0e9
        if tables:
            previous_step_s = float(
                np.median(np.diff(tables[-1]["display_time_s"].to_numpy()))
            )
            display_offset_s = (
                float(tables[-1]["display_time_s"].iloc[-1]) + previous_step_s
            )
        display_time_s = display_offset_s + run_time_s
        metadata = run["metadata"]
        location_gt_mm = float(metadata["experiment"]["contact_location_gt_mm"])
        table = pd.DataFrame(
            {
                "analysis_version": ANALYSIS_VERSION,
                "display_time_s": display_time_s,
                "run_time_s": run_time_s,
                "run_id": metadata["run_id"],
                "hole_index": int(metadata["experiment"]["hole_index"]),
                "contact_location_gt_mm": location_gt_mm,
                "camera_timestamp_ns": camera_time[camera_indices],
                "camera_time_delta_ms": np.abs(
                    camera_time[camera_indices] - sequence_time
                )
                / 1.0e6,
                "sequence_timestamp_ns": sequence_time,
                "ground_truth_force_n": sequence["actual_force_N"].to_numpy(
                    dtype=np.float64
                ),
                "target_force_n": sequence["target_force_N"].to_numpy(dtype=np.float64),
                "sequence_state": sequence["state"].astype(str).to_numpy(),
                "motor_timestamp_ns": motor_time,
                "motor_time_delta_ms": motor_delta_ms,
                "motor_torque_nm": torque_nm,
                "tau0_nm": tau0_nm,
                "tau0_sample_count": tau0_sample_count,
                "optical_contact_detected": np.isfinite(nearest_raw_location_mm),
                "nearest_optical_location_raw_mm": nearest_raw_location_mm,
                "optical_location_mm": location_mm,
                "optical_status": np.asarray(statuses)[camera_indices],
                "moment_arm_m": moment_arm_mm / 1000.0,
                "unscaled_force_predictor_n": (torque_nm - tau0_nm)
                / (moment_arm_mm / 1000.0),
            }
        )
        tables.append(table)

    combined = pd.concat(tables, ignore_index=True)
    calibration_mask = (combined["run_id"] == CALIBRATION_RUN_ID) & (
        combined["ground_truth_force_n"] >= CONTACT_FORCE_THRESHOLD_N
    )
    predictor = combined.loc[calibration_mask, "unscaled_force_predictor_n"].to_numpy(
        dtype=np.float64
    )
    target = combined.loc[calibration_mask, "ground_truth_force_n"].to_numpy(
        dtype=np.float64
    )
    denominator = float(np.dot(predictor, predictor))
    if denominator <= np.finfo(np.float64).eps:
        raise RuntimeError("global force-scale calibration is rank deficient")
    global_alpha = float(np.dot(predictor, target) / denominator)
    combined["global_alpha"] = global_alpha
    combined["alpha_calibration_run"] = CALIBRATION_RUN_ID
    combined["estimated_force_n"] = np.maximum(
        global_alpha
        * combined["unscaled_force_predictor_n"].to_numpy(dtype=np.float64),
        0.0,
    )
    contact = combined["ground_truth_force_n"] >= CONTACT_FORCE_THRESHOLD_N
    contact_mae_n = float(
        np.mean(
            np.abs(
                combined.loc[contact, "estimated_force_n"].to_numpy()
                - combined.loc[contact, "ground_truth_force_n"].to_numpy()
            )
        )
    )
    if tuple(combined["run_id"].drop_duplicates()) != run_ids:
        raise RuntimeError("concatenated output changed the requested run order")
    summary = {
        "run_ids": run_ids,
        "locations_mm": EXPECTED_LOCATIONS_MM,
        "calibration_run_id": CALIBRATION_RUN_ID,
        "global_alpha": global_alpha,
        "contact_mae_n": contact_mae_n,
        "contact_threshold_n": CONTACT_FORCE_THRESHOLD_N,
        "moment_arm_definition": "r(x) = (103.6 - x) / 1000 m",
        "force_ground_truth": "camera-synchronized Rokubi force-vector magnitude",
        "location_method": (
            "dominant baseline-relative canonical Green response mapped through "
            "the detected five-LED geometry"
        ),
    }
    return combined, summary


def plot_force_timeseries(
    axis: plt.Axes,
    table: pd.DataFrame,
    summary: dict[str, object],
    *,
    show_title: bool = False,
    compact: bool = False,
) -> None:
    """Draw force and synchronized raw motor torque on a caller-owned axis."""

    torque_axis = axis.twinx()
    torque_axis.plot(
        table["display_time_s"],
        table["motor_torque_nm"],
        color=TORQUE_TRACE_COLOR,
        linewidth=0.75,
        alpha=0.72,
        zorder=1,
    )
    torque_axis.set_ylabel(
        "Motor torque [N m]",
        color=TORQUE_TRACE_COLOR,
        labelpad=2.0,
    )
    torque_axis.tick_params(
        axis="y",
        colors=TORQUE_TRACE_COLOR,
        labelsize=DEFAULT_STYLE.tick_font_size_pt,
        length=DEFAULT_STYLE.tick_length_pt,
        width=DEFAULT_STYLE.tick_width_pt,
        pad=1.5,
    )
    torque_axis.spines["top"].set_visible(False)
    torque_axis.spines["left"].set_visible(False)
    torque_axis.spines["right"].set_color(TORQUE_TRACE_COLOR)
    torque_axis.grid(False)

    axis.plot(
        table["display_time_s"],
        table["ground_truth_force_n"],
        color="#30343B",
        linewidth=1.05,
        label="Ground truth" if compact else "Ground-truth force",
        zorder=3,
    )
    axis.plot(
        table["display_time_s"],
        table["estimated_force_n"],
        color="#2C758E",
        linewidth=1.05,
        linestyle=(0, (3.0, 1.7)),
        label="Estimate" if compact else "Estimated force",
        zorder=3,
    )
    groups = list(table.groupby("run_id", sort=False))
    separators = [
        0.5
        * (
            float(groups[index - 1][1]["display_time_s"].iloc[-1])
            + float(groups[index][1]["display_time_s"].iloc[0])
        )
        for index in range(1, len(groups))
    ]
    if separators:
        axis.vlines(
            separators,
            0.0,
            1.0,
            transform=axis.get_xaxis_transform(),
            color="#A7A7A7",
            linewidth=0.55,
            linestyles=":",
            zorder=1,
        )
    for _, group in groups:
        center = 0.5 * (
            float(group["display_time_s"].iloc[0])
            + float(group["display_time_s"].iloc[-1])
        )
        location = int(round(float(group["contact_location_gt_mm"].iloc[0])))
        axis.text(
            center,
            1.018,
            f"{location} mm",
            transform=axis.get_xaxis_transform(),
            fontsize=DEFAULT_STYLE.minimum_font_size_pt,
            color="#555555",
            ha="center",
            va="bottom",
            clip_on=False,
        )
    maximum = max(
        float(table["ground_truth_force_n"].max()),
        float(table["estimated_force_n"].max()),
    )
    axis.set_xlim(
        float(table["display_time_s"].iloc[0]),
        float(table["display_time_s"].iloc[-1]),
    )
    axis.set_ylim(0.0, 5.0 * np.ceil(1.08 * maximum / 5.0))
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Force [N]")
    if show_title:
        axis.set_title(
            "Location-aware proprioceptive force estimation",
            fontsize=DEFAULT_STYLE.panel_title_font_size_pt,
            fontweight="normal",
            pad=10.0,
        )
    axis.grid(axis="y", color="#E3E3E3", linewidth=0.45, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#777777")
    axis.spines["bottom"].set_color("#777777")
    axis.tick_params(
        labelsize=DEFAULT_STYLE.tick_font_size_pt,
        length=DEFAULT_STYLE.tick_length_pt,
        width=DEFAULT_STYLE.tick_width_pt,
        pad=1.5,
    )
    axis.legend(
        loc="upper left",
        ncol=2,
        frameon=False,
        fontsize=DEFAULT_STYLE.minimum_font_size_pt,
        handlelength=1.8 if compact else 2.6,
        columnspacing=0.65 if compact else 1.1,
        borderaxespad=0.35,
    )
    axis.text(
        0.995,
        0.965,
        (
            f"MAE = {float(summary['contact_mae_n']):.2f} N"
            if compact
            else (
                f"{EXPERIMENT_LABEL}\n"
                f"Contact MAE = {float(summary['contact_mae_n']):.2f} N"
            )
        ),
        transform=axis.transAxes,
        fontsize=DEFAULT_STYLE.minimum_font_size_pt,
        color="#555555",
        ha="right",
        va="top",
    )


def write_force_timeseries_csv(table: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False, float_format="%.9g")
    return destination


def save_force_timeseries_panel(
    table: pd.DataFrame,
    summary: dict[str, object],
    output_stem: str | Path = DEFAULT_OUTPUT_STEM,
) -> tuple[Path, ...]:
    """Write the standalone audit render for Figure 6(d)."""

    figure, axis = plt.subplots(figsize=(DEFAULT_STYLE.double_column_width_in, 2.20))
    figure.subplots_adjust(left=0.085, right=0.925, bottom=0.20, top=0.84)
    plot_force_timeseries(axis, table, summary, show_title=True)
    outputs = save_figure(
        figure,
        Path(output_stem),
        formats=("pdf", "png"),
        bbox_inches=None,
        pad_inches=0.0,
    )
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    arguments = parser.parse_args()
    with publication_context(DEFAULT_STYLE):
        table, summary = build_force_timeseries(arguments.dataset_root)
        csv_path = write_force_timeseries_csv(
            table, arguments.output_stem.with_suffix(".csv")
        )
        outputs = save_force_timeseries_panel(table, summary, arguments.output_stem)
    print(csv_path)
    for path in outputs:
        print(path)
    print(
        f"alpha={summary['global_alpha']:.6f}; "
        f"contact_MAE={summary['contact_mae_n']:.3f} N"
    )


if __name__ == "__main__":
    main()
