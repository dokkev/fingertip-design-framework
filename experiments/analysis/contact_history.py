"""Read-only analysis of continuous cyclic contact-history experiments."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .optical import (
    PROFILE_BINS,
    OpticalStrip,
    calibrate_optical_strip,
    load_rgb,
    longitudinal_green_profile,
    temporal_median_rgb,
)


FORCE_GRID_N = np.arange(3.0, 14.0 + 0.25, 0.5, dtype=np.float64)
MINIMUM_CYCLE_COVERAGE_FRACTION = 0.5
MORPHOLOGY_ORDER = ("baseline", "flat_opt", "angled_opt")
MORPHOLOGY_LABELS = {
    "baseline": "Baseline",
    "flat_opt": "Flat-opt",
    "angled_opt": "Angled-opt",
}
COLORS = {
    "baseline": "#5f6368",
    "flat_opt": "#287d8e",
    "angled_opt": "#d27a32",
}


def analyze_contact_history(
    session_paths: list[str | Path],
    output_path: str | Path,
    *,
    repeat_metrics_path: str | Path | None = None,
) -> Path:
    """Analyze matched-force history dependence for three same-material sessions."""

    sessions = [_index_session(Path(path).resolve()) for path in session_paths]
    _validate_sessions(sessions)
    material = str(sessions[0]["material"])
    common_conditions = set.intersection(
        *[
            {(run["indenter"], run["hole_index"]) for run in session["runs"]}
            for session in sessions
        ]
    )
    if not common_conditions:
        raise RuntimeError("sessions have no common indenter/contact-location condition")

    for session in sessions:
        _finish_run_qc(session, common_conditions)

    primary_runs = [
        run
        for session in sessions
        for run in session["runs"]
        if run["condition"] in common_conditions and run["primary_valid"]
    ]
    missing = [
        morphology
        for morphology in MORPHOLOGY_ORDER
        if not any(run["morphology"] == morphology for run in primary_runs)
    ]
    if missing:
        raise RuntimeError(f"no QC-valid runs remain for: {', '.join(missing)}")

    calibration_cache: dict[tuple[str, str], OpticalStrip] = {}
    frame_rows: list[dict[str, Any]] = []
    frame_profiles: list[np.ndarray] = []
    for run in primary_runs:
        capture = _nearest_unloaded_capture(run, run["session"]["captures"])
        cache_key = (run["specimen_id"], capture["capture_id"])
        if cache_key not in calibration_cache:
            reference = temporal_median_rgb(
                [load_rgb(path) for path in capture["image_paths"]]
            )
            calibration_cache[cache_key] = calibrate_optical_strip(reference)
        run["calibration_capture_id"] = capture["capture_id"]
        strip = calibration_cache[cache_key]
        for row in run["frames"]:
            profile, _ = longitudinal_green_profile(
                load_rgb(run["trajectory_path"] / row["rgb_filename"]),
                strip,
                bins=PROFILE_BINS,
            )
            frame_rows.append(
                {
                    "specimen_id": run["specimen_id"],
                    "morphology": run["morphology"],
                    "run_id": run["run_id"],
                    "hole_index": run["hole_index"],
                    "repetition_index": run["repetition_index"],
                    "frame_index": int(row["frame_index"]),
                    "trajectory_elapsed_s": float(row["trajectory_elapsed_s"]),
                    "cycle_index": int(row["cycle_index"]),
                    "cycle_role": row["cycle_role"],
                    "phase": row["phase"],
                    "target_force_n": float(row["target_force_N"]),
                    "actual_force_n": float(row["force_magnitude_N"]),
                    "fz_share": float(row["fz_share"]),
                    "calibration_capture_id": capture["capture_id"],
                }
            )
            frame_profiles.append(profile)

    profiles = np.asarray(frame_profiles, dtype=np.float64)
    cycles = _cycle_profiles(frame_rows, profiles)
    measurement_cycles = [cycle for cycle in cycles if cycle["cycle_role"] == "measurement"]
    summary_grid = _common_summary_force_grid(measurement_cycles)

    history_rows, history_summary = _history_metrics(measurement_cycles, summary_grid)
    cycle_rows, cycle_summary = _cycle_repeatability(measurement_cycles, summary_grid)
    repeat_rows = _load_repeat_metrics(repeat_metrics_path, material)

    output = Path(output_path).resolve()
    results = output / "results"
    figures = output / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    qc_rows = [run for session in sessions for run in session["runs"]]
    _write_qc_csv(results / "run_qc.csv", qc_rows, common_conditions)
    _write_csv(results / "loading_unloading_gap.csv", history_rows)
    _write_csv(results / "history_summary.csv", history_summary)
    _write_csv(results / "same_contact_repeatability.csv", cycle_rows)
    _write_csv(results / "same_contact_repeatability_summary.csv", cycle_summary)
    _write_profile_cache(results / "longitudinal_profiles.npz", frame_rows, profiles)

    _plot_force_trajectories(figures, sessions, common_conditions)
    _plot_history_gap(figures, history_rows, summary_grid)
    _plot_relative_history(figures, history_summary)
    _plot_cycle_repeatability(figures, cycle_rows, summary_grid)
    _plot_conditioning(figures, cycles, summary_grid)
    if repeat_rows:
        _plot_repeat_comparison(figures, cycle_summary, repeat_rows)
    _write_report(
        output / "report.md",
        sessions,
        common_conditions,
        summary_grid,
        history_summary,
        cycle_summary,
        repeat_rows,
        material,
    )
    return output


def _index_session(path: Path) -> dict[str, Any]:
    session_json = _read_json(path / "session.json")
    specimen = session_json["specimen"]
    trajectory = session_json["trajectory"]
    total_cycles = int(trajectory["conditioning_cycles"]) + int(
        trajectory["measurement_cycles"]
    )
    cycle_s = (
        2.0
        * (float(trajectory["max_force_n"]) - float(trajectory["min_force_n"]))
        / float(trajectory["ramp_rate_n_per_s"])
        + float(trajectory["high_dwell_s"])
        + float(trajectory["low_dwell_s"])
    )
    expected_frames = int(
        round(total_cycles * cycle_s * float(trajectory["capture_rate_hz"]))
    )
    captures = []
    for capture_path in sorted((path / "unloaded").glob("capture_*")):
        rows = _read_csv(capture_path / "frames.csv")
        if not rows:
            continue
        captures.append(
            {
                "capture_id": capture_path.name,
                "host_time_s": float(
                    np.median([float(row["camera_host_time_s"]) for row in rows])
                ),
                "image_paths": [capture_path / row["rgb_filename"] for row in rows],
            }
        )
    if not captures:
        raise RuntimeError(f"{path} has no unloaded capture for optical calibration")

    session: dict[str, Any] = {
        "path": path,
        "session_json": session_json,
        "specimen_id": str(specimen["specimen_id"]),
        "material": str(specimen["material"]),
        "morphology": str(specimen["morphology"]),
        "expected_frames": expected_frames,
        "captures": captures,
        "runs": [],
    }
    for run_path in sorted((path / "runs").glob("run_*")):
        if not run_path.is_dir():
            continue
        metadata = _read_json(run_path / "run.json")
        trajectory_path = run_path / "trajectory"
        diagnostics = _read_json(trajectory_path / "trajectory.json")
        frames = _read_csv(trajectory_path / "frames.csv")
        if not frames:
            continue
        run = {
            "session": session,
            "session_path": str(path),
            "specimen_id": session["specimen_id"],
            "material": session["material"],
            "morphology": session["morphology"],
            "run_id": str(metadata["run_id"]),
            "status": str(metadata["status"]),
            "indenter": str(metadata["indenter"]),
            "hole_index": int(metadata["hole_index"]),
            "repetition_index": int(metadata["repetition_index"]),
            "trajectory_path": trajectory_path,
            "diagnostics": diagnostics,
            "frames": frames,
            "start_host_time_s": float(frames[0]["camera_host_time_s"]),
        }
        run["condition"] = (run["indenter"], run["hole_index"])
        session["runs"].append(run)
    return session


def _validate_sessions(sessions: list[dict[str, Any]]) -> None:
    if len(sessions) != 3:
        raise ValueError("exactly three baseline/flat-opt/angled-opt sessions are required")
    morphologies = {session["morphology"] for session in sessions}
    if morphologies != set(MORPHOLOGY_ORDER):
        raise ValueError(
            "sessions must contain exactly baseline, flat_opt, and angled_opt"
        )
    materials = {session["material"] for session in sessions}
    if len(materials) != 1:
        raise ValueError("all contact-history sessions must use the same material")
    camera_keys = (
        "model",
        "serial_number",
        "width",
        "height",
        "fps",
        "exposure_us",
        "gain",
        "white_balance_k",
    )
    camera_settings = {
        tuple(session["session_json"]["camera"].get(key) for key in camera_keys)
        for session in sessions
    }
    if len(camera_settings) != 1:
        raise RuntimeError("sessions do not share identical fixed camera settings")


def _finish_run_qc(
    session: dict[str, Any], common_conditions: set[tuple[str, int]]
) -> None:
    for run in session["runs"]:
        diagnostics = run["diagnostics"]
        frames = run["frames"]
        reasons: list[str] = []
        if run["condition"] not in common_conditions:
            reasons.append("condition_not_shared_by_all_morphologies")
        if run["status"] != "complete":
            reasons.append(f"run_status_{run['status']}")
        if bool(diagnostics["contact_loss_detected"]):
            reasons.append("contact_loss_detected")
        if len(frames) != session["expected_frames"]:
            reasons.append("unexpected_frame_count")
        for key in (
            "dropped_camera_frame_count",
            "dropped_writer_frame_count",
            "missed_capture_deadline_count",
        ):
            if int(diagnostics[key]) != 0:
                reasons.append(key)

        measurement = [row for row in frames if row["cycle_role"] == "measurement"]
        loading_fractions: list[float] = []
        unloading_fractions: list[float] = []
        overlap_low: list[float] = []
        overlap_high: list[float] = []
        for cycle_index in sorted({int(row["cycle_index"]) for row in measurement}):
            cycle = [row for row in measurement if int(row["cycle_index"]) == cycle_index]
            loading = np.asarray(
                [float(row["force_magnitude_N"]) for row in cycle if row["phase"] == "loading"]
            )
            unloading = np.asarray(
                [float(row["force_magnitude_N"]) for row in cycle if row["phase"] == "unloading"]
            )
            if len(loading) >= 2 and len(unloading) >= 2:
                loading_fractions.append(float(np.mean(np.diff(loading) >= 0.0)))
                unloading_fractions.append(float(np.mean(np.diff(unloading) <= 0.0)))
                overlap_low.append(float(max(loading.min(), unloading.min(), FORCE_GRID_N[0])))
                overlap_high.append(float(min(loading.max(), unloading.max(), FORCE_GRID_N[-1])))
        force = np.asarray([float(row["force_magnitude_N"]) for row in frames])
        fz_share = np.asarray([float(row["fz_share"]) for row in frames])
        run.update(
            {
                "primary_valid": not reasons,
                "qc_reasons": ";".join(reasons),
                "saved_frame_count": len(frames),
                "expected_frame_count": session["expected_frames"],
                "contact_loss_detected": bool(diagnostics["contact_loss_detected"]),
                "contact_loss_event_count": int(diagnostics["contact_loss_event_count"]),
                "minimum_force_seen_n": float(
                    diagnostics["min_actual_force_seen_during_cycling_N"]
                ),
                "actual_force_min_n": float(force.min()),
                "actual_force_max_n": float(force.max()),
                "fz_share_median": float(np.median(fz_share)),
                "fz_share_iqr": float(np.subtract(*np.percentile(fz_share, (75, 25)))),
                "loading_monotonic_fraction_median": _median_or_nan(loading_fractions),
                "unloading_monotonic_fraction_median": _median_or_nan(unloading_fractions),
                "actual_overlap_low_n_median": _median_or_nan(overlap_low),
                "actual_overlap_high_n_median": _median_or_nan(overlap_high),
            }
        )


def _nearest_unloaded_capture(
    run: dict[str, Any], captures: list[dict[str, Any]]
) -> dict[str, Any]:
    return min(
        captures,
        key=lambda capture: abs(capture["host_time_s"] - run["start_host_time_s"]),
    )


def _cycle_profiles(
    frame_rows: list[dict[str, Any]], profiles: np.ndarray
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(frame_rows):
        groups[(row["specimen_id"], row["run_id"], row["cycle_index"])].append(index)

    output = []
    for indices in groups.values():
        rows = [frame_rows[index] for index in indices]
        first = rows[0]
        branches = {}
        monotonic = {}
        for phase, expected_sign in (("loading", 1.0), ("unloading", -1.0)):
            selected = [index for index in indices if frame_rows[index]["phase"] == phase]
            force = np.asarray(
                [frame_rows[index]["actual_force_n"] for index in selected],
                dtype=np.float64,
            )
            values = profiles[selected]
            branches[phase] = _interpolate_profiles(force, values, FORCE_GRID_N)
            delta = expected_sign * np.diff(force)
            monotonic[phase] = float(np.mean(delta >= 0.0)) if len(delta) else float("nan")
        difference = branches["loading"] - branches["unloading"]
        valid = np.all(np.isfinite(difference), axis=1)
        gap = np.full(len(FORCE_GRID_N), np.nan, dtype=np.float64)
        gap[valid] = np.sqrt(np.mean(difference[valid] ** 2, axis=1))
        output.append(
            {
                "specimen_id": first["specimen_id"],
                "morphology": first["morphology"],
                "run_id": first["run_id"],
                "hole_index": first["hole_index"],
                "repetition_index": first["repetition_index"],
                "cycle_index": first["cycle_index"],
                "cycle_role": first["cycle_role"],
                "loading": branches["loading"],
                "unloading": branches["unloading"],
                "gap": gap,
                "loading_monotonic_fraction": monotonic["loading"],
                "unloading_monotonic_fraction": monotonic["unloading"],
            }
        )
    return output


def _interpolate_profiles(
    force: np.ndarray, profiles: np.ndarray, force_grid: np.ndarray
) -> np.ndarray:
    output = np.full((len(force_grid), PROFILE_BINS), np.nan, dtype=np.float64)
    if len(force) < 2:
        return output
    order = np.argsort(force, kind="stable")
    x = np.asarray(force[order], dtype=np.float64)
    y = np.asarray(profiles[order], dtype=np.float64)
    unique, inverse = np.unique(x, return_inverse=True)
    if len(unique) < 2:
        return output
    if len(unique) != len(x):
        reduced = np.empty((len(unique), y.shape[1]), dtype=np.float64)
        for index in range(len(unique)):
            reduced[index] = np.mean(y[inverse == index], axis=0)
        x, y = unique, reduced
    valid = (force_grid >= x[0]) & (force_grid <= x[-1])
    for profile_index in range(y.shape[1]):
        output[valid, profile_index] = np.interp(
            force_grid[valid], x, y[:, profile_index]
        )
    return output


def _common_summary_force_grid(cycles: list[dict[str, Any]]) -> np.ndarray:
    valid = np.ones(len(FORCE_GRID_N), dtype=bool)
    for morphology in MORPHOLOGY_ORDER:
        selected = [cycle for cycle in cycles if cycle["morphology"] == morphology]
        count = np.sum([np.isfinite(cycle["gap"]) for cycle in selected], axis=0)
        minimum = math.ceil(MINIMUM_CYCLE_COVERAGE_FRACTION * len(selected))
        valid &= count >= minimum
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        raise RuntimeError("fewer than two common actual-force grid points pass coverage")
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    longest = max(runs, key=len)
    return FORCE_GRID_N[longest]


def _history_metrics(
    cycles: list[dict[str, Any]], summary_grid: np.ndarray
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    force_indices = [int(np.flatnonzero(FORCE_GRID_N == force)[0]) for force in summary_grid]
    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for morphology in MORPHOLOGY_ORDER:
        selected = [cycle for cycle in cycles if cycle["morphology"] == morphology]
        for force, force_index in zip(summary_grid, force_indices, strict=True):
            values = np.asarray(
                [cycle["gap"][force_index] for cycle in selected], dtype=np.float64
            )
            values = values[np.isfinite(values)]
            curve_rows.append(
                {
                    "morphology": morphology,
                    "actual_force_n": force,
                    "cycle_count": len(values),
                    "H_median_dn": float(np.median(values)),
                    "H_q25_dn": float(np.percentile(values, 25)),
                    "H_q75_dn": float(np.percentile(values, 75)),
                }
            )

        cycle_metrics = []
        low_index, high_index = force_indices[0], force_indices[-1]
        for cycle in selected:
            if not np.isfinite(cycle["gap"][[low_index, high_index]]).all():
                continue
            mean_low = 0.5 * (
                cycle["loading"][low_index] + cycle["unloading"][low_index]
            )
            mean_high = 0.5 * (
                cycle["loading"][high_index] + cycle["unloading"][high_index]
            )
            signal_span = float(np.sqrt(np.mean((mean_high - mean_low) ** 2)))
            history = float(np.nanmedian(cycle["gap"][force_indices]))
            relative = history / signal_span if signal_span > 0.0 else float("nan")
            cycle_metrics.append((history, signal_span, relative))
        values = np.asarray(cycle_metrics, dtype=np.float64)
        summary_rows.append(
            {
                "morphology": morphology,
                "force_low_n": float(summary_grid[0]),
                "force_high_n": float(summary_grid[-1]),
                "eligible_cycle_count": len(values),
                "H_median_dn": float(np.median(values[:, 0])),
                "H_iqr_dn": float(np.subtract(*np.percentile(values[:, 0], (75, 25)))),
                "S_span_median_dn": float(np.median(values[:, 1])),
                "S_span_iqr_dn": float(np.subtract(*np.percentile(values[:, 1], (75, 25)))),
                "H_rel_median": float(np.median(values[:, 2])),
                "H_rel_q25": float(np.percentile(values[:, 2], 25)),
                "H_rel_q75": float(np.percentile(values[:, 2], 75)),
            }
        )
    return curve_rows, summary_rows


def _cycle_repeatability(
    cycles: list[dict[str, Any]], summary_grid: np.ndarray
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    force_indices = [int(np.flatnonzero(FORCE_GRID_N == force)[0]) for force in summary_grid]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for cycle in cycles:
        grouped[(cycle["morphology"], cycle["run_id"])].append(cycle)

    rows = []
    run_scalars: dict[str, list[float]] = defaultdict(list)
    for (morphology, run_id), run_cycles in sorted(grouped.items()):
        first = run_cycles[0]
        per_run = []
        for branch in ("loading", "unloading"):
            for force, force_index in zip(summary_grid, force_indices, strict=True):
                values = np.asarray(
                    [cycle[branch][force_index] for cycle in run_cycles],
                    dtype=np.float64,
                )
                valid = np.all(np.isfinite(values), axis=1)
                values = values[valid]
                if len(values) < 2:
                    repeatability = float("nan")
                else:
                    template = np.median(values, axis=0)
                    distances = np.sqrt(np.mean((values - template) ** 2, axis=1))
                    repeatability = float(np.median(distances))
                    per_run.append(repeatability)
                rows.append(
                    {
                        "morphology": morphology,
                        "run_id": run_id,
                        "hole_index": first["hole_index"],
                        "repetition_index": first["repetition_index"],
                        "branch": branch,
                        "actual_force_n": float(force),
                        "cycle_count": len(values),
                        "W_cycle_dn": repeatability,
                    }
                )
        if per_run:
            run_scalars[morphology].append(float(np.median(per_run)))

    summary = []
    for morphology in MORPHOLOGY_ORDER:
        values = np.asarray(run_scalars[morphology], dtype=np.float64)
        summary.append(
            {
                "morphology": morphology,
                "run_count": len(values),
                "W_cycle_median_dn": float(np.median(values)),
                "W_cycle_q25_dn": float(np.percentile(values, 25)),
                "W_cycle_q75_dn": float(np.percentile(values, 75)),
            }
        )
    return rows, summary


def _load_repeat_metrics(
    path: str | Path | None, material: str
) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows = _read_csv(Path(path))
    output = []
    for morphology in MORPHOLOGY_ORDER:
        matches = [
            row
            for row in rows
            if row["material"] == material
            and row["morphology"] == morphology
            and row["indenter"] == "sphere_10mm"
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one 10 mm W_repeat row for {morphology}")
        output.append(
            {
                "morphology": morphology,
                "W_repeat_dn_per_n": float(matches[0]["W_median_DN_per_N"]),
            }
        )
    return output


def _plot_force_trajectories(
    output: Path,
    sessions: list[dict[str, Any]],
    common_conditions: set[tuple[str, int]],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(9.0, 2.7), sharex=True, sharey=True)
    for axis, morphology in zip(axes, MORPHOLOGY_ORDER, strict=True):
        session = next(item for item in sessions if item["morphology"] == morphology)
        target_drawn = False
        for run in session["runs"]:
            if run["condition"] not in common_conditions:
                continue
            elapsed = np.asarray([float(row["trajectory_elapsed_s"]) for row in run["frames"]])
            force = np.asarray([float(row["force_magnitude_N"]) for row in run["frames"]])
            target = np.asarray([float(row["target_force_N"]) for row in run["frames"]])
            if run["primary_valid"]:
                color, alpha, width = COLORS[morphology], 0.28, 0.8
            else:
                color, alpha, width = "#c13b32", 0.28, 0.7
            axis.plot(elapsed, force, color=color, alpha=alpha, linewidth=width)
            if not target_drawn:
                axis.plot(elapsed, target, color="black", linestyle="--", linewidth=0.8)
                target_drawn = True
        axis.set_title(MORPHOLOGY_LABELS[morphology])
        axis.set_xlabel("Time [s]")
        axis.grid(axis="y", color="#dedede", linewidth=0.5)
    axes[0].set_ylabel("Actual force [N]")
    figure.suptitle("Actual force trajectories (red: excluded by run QC)", fontsize=10)
    _save(figure, output / "actual_force_trajectories")


def _plot_history_gap(
    output: Path, rows: list[dict[str, Any]], summary_grid: np.ndarray
) -> None:
    figure, axis = plt.subplots(figsize=(4.4, 3.0))
    for morphology in MORPHOLOGY_ORDER:
        selected = [row for row in rows if row["morphology"] == morphology]
        force = np.asarray([row["actual_force_n"] for row in selected])
        median = np.asarray([row["H_median_dn"] for row in selected])
        low = np.asarray([row["H_q25_dn"] for row in selected])
        high = np.asarray([row["H_q75_dn"] for row in selected])
        axis.fill_between(force, low, high, color=COLORS[morphology], alpha=0.17)
        axis.plot(force, median, color=COLORS[morphology], label=MORPHOLOGY_LABELS[morphology])
    axis.set(
        xlabel="Actual force [N]",
        ylabel=r"Loading--unloading gap $H(F)$ [DN]",
        xlim=(FORCE_GRID_N[0], FORCE_GRID_N[-1]),
    )
    axis.axvspan(summary_grid[0], summary_grid[-1], color="#eeeeee", zorder=-2)
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#dedede", linewidth=0.5)
    _save(figure, output / "loading_unloading_gap")


def _plot_relative_history(output: Path, rows: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(4.1, 2.9))
    x = np.arange(len(MORPHOLOGY_ORDER))
    ordered = [next(row for row in rows if row["morphology"] == name) for name in MORPHOLOGY_ORDER]
    median = np.asarray([row["H_rel_median"] for row in ordered])
    low = np.asarray([row["H_rel_q25"] for row in ordered])
    high = np.asarray([row["H_rel_q75"] for row in ordered])
    axis.bar(x, median, color=[COLORS[name] for name in MORPHOLOGY_ORDER], width=0.62)
    axis.errorbar(x, median, yerr=(median - low, high - median), fmt="none", color="black", capsize=3)
    axis.set(
        ylabel=r"Signal-normalized history $H_{rel}$",
        xticks=x,
        xticklabels=[MORPHOLOGY_LABELS[name] for name in MORPHOLOGY_ORDER],
    )
    axis.grid(axis="y", color="#dedede", linewidth=0.5)
    _save(figure, output / "relative_history_dependence")


def _plot_cycle_repeatability(
    output: Path, rows: list[dict[str, Any]], summary_grid: np.ndarray
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharex=True, sharey=True)
    for axis, branch in zip(axes, ("loading", "unloading"), strict=True):
        for morphology in MORPHOLOGY_ORDER:
            selected = [
                row
                for row in rows
                if row["morphology"] == morphology
                and row["branch"] == branch
                and np.isfinite(row["W_cycle_dn"])
            ]
            values_by_force = defaultdict(list)
            for row in selected:
                values_by_force[row["actual_force_n"]].append(row["W_cycle_dn"])
            force, median, low, high = [], [], [], []
            for value in summary_grid:
                data = np.asarray(values_by_force[float(value)], dtype=np.float64)
                if not len(data):
                    continue
                force.append(value)
                median.append(np.median(data))
                low.append(np.percentile(data, 25))
                high.append(np.percentile(data, 75))
            axis.fill_between(force, low, high, color=COLORS[morphology], alpha=0.17)
            axis.plot(force, median, color=COLORS[morphology], label=MORPHOLOGY_LABELS[morphology])
        axis.set_title(branch.capitalize())
        axis.set_xlabel("Actual force [N]")
        axis.grid(axis="y", color="#dedede", linewidth=0.5)
    axes[0].set_ylabel(r"Same-contact variation $W_{cycle}(F)$ [DN]")
    axes[1].legend(frameon=False)
    _save(figure, output / "same_contact_repeatability")


def _plot_conditioning(
    output: Path, cycles: list[dict[str, Any]], summary_grid: np.ndarray
) -> None:
    target_force = float(summary_grid[np.argmin(np.abs(summary_grid - 10.0))])
    force_index = int(np.flatnonzero(FORCE_GRID_N == target_force)[0])
    coordinate = np.linspace(0.0, 1.0, PROFILE_BINS)
    target_hole = max(cycle["hole_index"] for cycle in cycles)
    figure, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), sharex=True, sharey=True)
    for axis, morphology in zip(axes, MORPHOLOGY_ORDER, strict=True):
        selected = [
            cycle
            for cycle in cycles
            if cycle["morphology"] == morphology
            and cycle["hole_index"] == target_hole
        ]
        templates = {}
        for cycle_index in range(1, 8):
            profiles = np.asarray(
                [
                    cycle["loading"][force_index]
                    for cycle in selected
                    if cycle["cycle_index"] == cycle_index
                    and np.all(np.isfinite(cycle["loading"][force_index]))
                ]
            )
            if len(profiles):
                templates[cycle_index] = np.median(profiles, axis=0)
        measurement = [templates[index] for index in range(3, 8) if index in templates]
        if not measurement:
            continue
        reference = np.median(measurement, axis=0)
        for cycle_index, profile in templates.items():
            if cycle_index <= 2:
                color = ("#f6a15f", "#c85d2c")[cycle_index - 1]
                label = f"Conditioning {cycle_index}"
                width = 1.1
            else:
                color = plt.cm.viridis(0.35 + 0.12 * (cycle_index - 3))
                label = f"Measurement {cycle_index - 2}"
                width = 0.8
            axis.plot(coordinate, profile - reference, color=color, linewidth=width, label=label)
        axis.axhline(0.0, color="#999999", linewidth=0.6)
        axis.set_title(MORPHOLOGY_LABELS[morphology])
        axis.set_xlabel("Longitudinal coordinate")
    axes[0].set_ylabel(f"Profile residual at {target_force:g} N [DN]")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=7)
    _save(figure, output / "conditioning_cycle_profiles", rect=(0.0, 0.0, 1.0, 0.84))


def _plot_repeat_comparison(
    output: Path,
    cycle_rows: list[dict[str, Any]],
    repeat_rows: list[dict[str, Any]],
) -> None:
    cycle = {row["morphology"]: row["W_cycle_median_dn"] for row in cycle_rows}
    repeat = {row["morphology"]: row["W_repeat_dn_per_n"] for row in repeat_rows}
    x = np.arange(len(MORPHOLOGY_ORDER))
    cycle_ratio = np.asarray([cycle[name] / cycle["baseline"] for name in MORPHOLOGY_ORDER])
    repeat_ratio = np.asarray([repeat[name] / repeat["baseline"] for name in MORPHOLOGY_ORDER])
    figure, axis = plt.subplots(figsize=(4.6, 2.9))
    axis.bar(x - 0.18, repeat_ratio, 0.36, color="#9aa0a6", label=r"Independent contact $W_{repeat}$")
    axis.bar(x + 0.18, cycle_ratio, 0.36, color="#287d8e", label=r"Same contact $W_{cycle}$")
    axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axis.set(
        ylabel="Variation relative to baseline",
        xticks=x,
        xticklabels=[MORPHOLOGY_LABELS[name] for name in MORPHOLOGY_ORDER],
    )
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", color="#dedede", linewidth=0.5)
    _save(figure, output / "independent_vs_same_contact_variation")


def _write_qc_csv(
    path: Path,
    runs: list[dict[str, Any]],
    common_conditions: set[tuple[str, int]],
) -> None:
    rows = []
    for run in runs:
        diagnostics = run["diagnostics"]
        rows.append(
            {
                key: run[key]
                for key in (
                    "specimen_id",
                    "morphology",
                    "run_id",
                    "indenter",
                    "hole_index",
                    "repetition_index",
                    "status",
                    "saved_frame_count",
                    "expected_frame_count",
                    "contact_loss_detected",
                    "contact_loss_event_count",
                    "minimum_force_seen_n",
                    "actual_force_min_n",
                    "actual_force_max_n",
                    "fz_share_median",
                    "fz_share_iqr",
                    "loading_monotonic_fraction_median",
                    "unloading_monotonic_fraction_median",
                    "actual_overlap_low_n_median",
                    "actual_overlap_high_n_median",
                    "primary_valid",
                    "qc_reasons",
                )
            }
            | {
                "condition_shared_by_all_morphologies": run["condition"]
                in common_conditions,
                "dropped_camera_frame_count": diagnostics["dropped_camera_frame_count"],
                "dropped_writer_frame_count": diagnostics["dropped_writer_frame_count"],
                "missed_capture_deadline_count": diagnostics[
                    "missed_capture_deadline_count"
                ],
                "calibration_capture_id": run.get("calibration_capture_id", ""),
            }
        )
    _write_csv(path, rows)


def _write_profile_cache(
    path: Path, rows: list[dict[str, Any]], profiles: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        profiles=np.asarray(profiles, dtype=np.float32),
        longitudinal_coordinate=np.linspace(0.0, 1.0, PROFILE_BINS, dtype=np.float32),
        specimen_id=np.asarray([row["specimen_id"] for row in rows]),
        morphology=np.asarray([row["morphology"] for row in rows]),
        run_id=np.asarray([row["run_id"] for row in rows]),
        hole_index=np.asarray([row["hole_index"] for row in rows], dtype=np.int16),
        repetition_index=np.asarray([row["repetition_index"] for row in rows], dtype=np.int16),
        frame_index=np.asarray([row["frame_index"] for row in rows], dtype=np.int16),
        cycle_index=np.asarray([row["cycle_index"] for row in rows], dtype=np.int8),
        cycle_role=np.asarray([row["cycle_role"] for row in rows]),
        phase=np.asarray([row["phase"] for row in rows]),
        trajectory_elapsed_s=np.asarray([row["trajectory_elapsed_s"] for row in rows]),
        actual_force_n=np.asarray([row["actual_force_n"] for row in rows]),
        target_force_n=np.asarray([row["target_force_n"] for row in rows]),
        calibration_capture_id=np.asarray([row["calibration_capture_id"] for row in rows]),
    )


def _write_report(
    path: Path,
    sessions: list[dict[str, Any]],
    common_conditions: set[tuple[str, int]],
    summary_grid: np.ndarray,
    history: list[dict[str, Any]],
    cycle: list[dict[str, Any]],
    repeat: list[dict[str, Any]],
    material: str,
) -> None:
    history_by_name = {row["morphology"]: row for row in history}
    cycle_by_name = {row["morphology"]: row for row in cycle}
    repeat_by_name = {row["morphology"]: row for row in repeat}
    lines = [
        f"# {material.replace('_', ' ').title()} cyclic contact-history analysis",
        "",
        "## Contract",
        "",
        "- Optical signature: fixed unloaded-calibrated 128-bin longitudinal Green-DN profile.",
        "- Branch matching: linear interpolation by actual measured force; nominal loading/unloading labels only; no extrapolation.",
        f"- Requested force grid: {FORCE_GRID_N[0]:g}--{FORCE_GRID_N[-1]:g} N in 0.5 N steps.",
        f"- Common >=50% cycle-coverage range used for scalar summaries: {summary_grid[0]:g}--{summary_grid[-1]:g} N.",
        "- Primary metrics exclude contact-loss warnings and acquisition/frame-count failures.",
        "- Only indenter/contact-location conditions represented in all three sessions are compared.",
        "",
        "## Data QC",
        "",
        f"- Common conditions: {', '.join(f'{indenter}, hole {hole}' for indenter, hole in sorted(common_conditions))}.",
    ]
    for morphology in MORPHOLOGY_ORDER:
        session = next(item for item in sessions if item["morphology"] == morphology)
        common = [run for run in session["runs"] if run["condition"] in common_conditions]
        valid = [run for run in common if run["primary_valid"]]
        loss = [run for run in common if run["contact_loss_detected"]]
        fz = np.asarray([run["fz_share_median"] for run in valid])
        loading = np.asarray([run["loading_monotonic_fraction_median"] for run in valid])
        unloading = np.asarray([run["unloading_monotonic_fraction_median"] for run in valid])
        lines.append(
            f"- {MORPHOLOGY_LABELS[morphology]}: {len(valid)}/{len(common)} primary runs; "
            f"contact-loss warnings={len(loss)}; median fz_share={np.median(fz):.3f}; "
            f"median monotonic fractions loading/unloading={np.median(loading):.3f}/{np.median(unloading):.3f}."
        )
    lines.extend(
        [
            "",
            "All retained runs contain the expected 150 saved frames and no camera, writer, or capture-deadline drops.",
            "",
            "## History dependence and same-contact repeatability",
            "",
            "| Morphology | eligible cycles | H [DN] | S_span [DN] | H_rel | W_cycle [DN] |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for morphology in MORPHOLOGY_ORDER:
        h = history_by_name[morphology]
        w = cycle_by_name[morphology]
        lines.append(
            f"| {MORPHOLOGY_LABELS[morphology]} | {h['eligible_cycle_count']} | "
            f"{h['H_median_dn']:.4f} | {h['S_span_median_dn']:.4f} | "
            f"{h['H_rel_median']:.4f} | {w['W_cycle_median_dn']:.4f} |"
        )
    if repeat:
        lines.extend(
            [
                "",
                "## Independent re-contact comparison",
                "",
                "The absolute W_repeat [DN/N] and W_cycle [DN] units differ, so only each metric's ratio to its own baseline is compared.",
                "",
                "| Morphology | W_repeat [DN/N] | W_repeat / baseline | W_cycle / baseline |",
                "|---|---:|---:|---:|",
            ]
        )
        for morphology in MORPHOLOGY_ORDER:
            r = repeat_by_name[morphology]["W_repeat_dn_per_n"]
            w = cycle_by_name[morphology]["W_cycle_median_dn"]
            lines.append(
                f"| {MORPHOLOGY_LABELS[morphology]} | {r:.5f} | "
                f"{r / repeat_by_name['baseline']['W_repeat_dn_per_n']:.3f} | "
                f"{w / cycle_by_name['baseline']['W_cycle_median_dn']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These metrics directly measure optical branch dependence and within-contact cycle variation. They do not by themselves identify viscoelasticity, Mullins behavior, or another mechanical mechanism.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def _median_or_nan(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def _save(
    figure: plt.Figure,
    path: Path,
    *,
    rect: tuple[float, float, float, float] | None = None,
) -> None:
    figure.tight_layout(rect=rect)
    figure.savefig(path.with_suffix(".png"), dpi=240)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


__all__ = ["analyze_contact_history"]
