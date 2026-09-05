"""One-pass raw extraction and compact morphology-analysis output."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from experiments.data_collection.contact_dataset import FORMAT_VERSION

from .dataset import SessionIndex, camera_consistency_warnings, index_session
from .metrics import (
    actual_force_magnitude,
    aggregate_run_force_frames,
    fit_load_responses,
    morphology_metrics,
    spatial_metrics,
)
from .optical import (
    GREEN_EXCESS_THRESHOLD_DN,
    INTERIOR_MARGIN_PX,
    PROFILE_BINS,
    calibrate_optical_strip,
    load_rgb,
    longitudinal_green_profile,
    strip_centroid,
    temporal_median_rgb,
)
from .plotting import write_figures
from .run_qc import analyze_run_qc


SUMMARY_SCHEMA_VERSION = 1
CAMERA_GEOMETRY_WARNING_FRACTION = 0.01


@dataclass(frozen=True)
class AnalysisConfig:
    """The small explicit configuration surface of morphology analysis."""

    expected_repetitions: int = 5
    profile_bins: int = PROFILE_BINS
    green_excess_threshold_dn: float = GREEN_EXCESS_THRESHOLD_DN
    optical_interior_margin_px: float = INTERIOR_MARGIN_PX
    hole_spacing_mm: float | None = None

    def __post_init__(self) -> None:
        if self.expected_repetitions < 1:
            raise ValueError("expected_repetitions must be positive")
        if self.profile_bins < 2:
            raise ValueError("profile_bins must be at least two")
        if self.green_excess_threshold_dn <= 0.0:
            raise ValueError("green_excess_threshold_dn must be positive")
        if self.optical_interior_margin_px <= 0.0:
            raise ValueError("optical_interior_margin_px must be positive")
        if self.hole_spacing_mm is not None and self.hole_spacing_mm <= 0.0:
            raise ValueError("hole_spacing_mm must be positive when supplied")


def analyze_morphologies(
    session_paths: list[str | Path],
    output_path: str | Path,
    *,
    config: AnalysisConfig = AnalysisConfig(),
) -> Path:
    """Analyze arbitrary format-v3 sessions and write results plus raw summary."""

    if not session_paths:
        raise ValueError("at least one session path is required")
    indexes = [
        index_session(path, expected_repetitions=config.expected_repetitions)
        for path in session_paths
    ]
    specimen_ids = [index.specimen_id for index in indexes]
    if len(specimen_ids) != len(set(specimen_ids)):
        raise ValueError("input sessions must have unique specimen_id values")

    dataset_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    frame_profiles: list[np.ndarray] = []
    for index in indexes:
        extracted = _extract_session(index, config)
        dataset_rows.append(extracted["dataset_row"])
        frame_rows.extend(extracted["frame_rows"])
        frame_profiles.extend(extracted["frame_profiles"])

    run_force_rows, run_force_profiles = aggregate_run_force_frames(
        frame_rows, np.asarray(frame_profiles, dtype=np.float64)
    )
    run_rows, slope_profiles = fit_load_responses(
        run_force_rows, run_force_profiles
    )
    neighboring_rows, variability_rows = spatial_metrics(
        run_rows, slope_profiles, hole_spacing_mm=config.hole_spacing_mm
    )
    morphology_rows = morphology_metrics(
        run_rows, neighboring_rows, variability_rows
    )
    coverage_rows = [row for index in indexes for row in index.coverage_rows]
    run_qc = analyze_run_qc(run_force_rows, run_force_profiles, coverage_rows)
    camera_warnings = camera_consistency_warnings(indexes)
    qc_rows = _qc_rows(
        indexes,
        dataset_rows,
        run_force_rows,
        camera_warnings,
    )

    output = Path(output_path).resolve()
    results = output / "results"
    figures = output / "figures"
    raw_summary = output / "raw_data_summary"
    for directory in (results, figures, raw_summary):
        directory.mkdir(parents=True, exist_ok=True)

    _write_csv(results / "run_metrics.csv", run_rows)
    _write_csv(results / "morphology_metrics.csv", morphology_rows)
    _write_csv(results / "neighboring_separability.csv", neighboring_rows)
    _write_csv(results / "repeat_variability.csv", variability_rows)

    _write_csv(raw_summary / "dataset_summary.csv", dataset_rows)
    _write_csv(raw_summary / "run_force_summary.csv", run_force_rows)
    _write_csv(raw_summary / "run_load_response.csv", run_rows)
    _write_csv(raw_summary / "qc_summary.csv", qc_rows)
    _write_csv(raw_summary / "suspect_runs.csv", run_qc["rows"])
    _write_profile_npz(
        raw_summary / "longitudinal_profiles.npz",
        "profiles",
        run_force_profiles,
        run_force_rows,
        include_force=True,
    )
    _write_profile_npz(
        raw_summary / "load_response_profiles.npz",
        "slope_profiles",
        slope_profiles,
        run_rows,
        include_force=False,
    )
    _write_slope_profile_csv(
        raw_summary / "load_response_profiles.csv", run_rows, slope_profiles
    )
    _write_readme(
        raw_summary,
        dataset_rows=dataset_rows,
        camera_warnings=camera_warnings,
        profile_bins=config.profile_bins,
    )
    write_figures(
        figures,
        run_rows=run_rows,
        slope_profiles=slope_profiles,
        neighboring_rows=neighboring_rows,
        variability_rows=variability_rows,
        morphology_rows=morphology_rows,
    )
    _write_summary_zip(output, raw_summary)
    return output


def _extract_session(
    index: SessionIndex, config: AnalysisConfig
) -> dict[str, Any]:
    unloaded_records = [frame for frame in index.frames if frame.run is None]
    if not unloaded_records:
        raise RuntimeError(
            f"{index.specimen_id} has no unloaded frame for fixed strip calibration"
        )
    unloaded_images = [load_rgb(frame.rgb_path) for frame in unloaded_records]
    reference = temporal_median_rgb(unloaded_images)
    strip = calibrate_optical_strip(
        reference,
        green_excess_threshold_dn=config.green_excess_threshold_dn,
        interior_margin_px=config.optical_interior_margin_px,
    )

    calibration_warnings: list[str] = []
    capture_centroids = []
    capture_paths = sorted({frame.segment_path for frame in unloaded_records})
    for path in capture_paths:
        images = [
            image
            for frame, image in zip(unloaded_records, unloaded_images, strict=True)
            if frame.segment_path == path
        ]
        try:
            capture_strip = calibrate_optical_strip(
                temporal_median_rgb(images),
                green_excess_threshold_dn=config.green_excess_threshold_dn,
                interior_margin_px=config.optical_interior_margin_px,
            )
            capture_centroids.append(strip_centroid(capture_strip))
        except RuntimeError as error:
            calibration_warnings.append(f"unloaded geometry calibration failed: {error}")
    centroid_span_px = max(
        (
            float(np.linalg.norm(first - second))
            for position, first in enumerate(capture_centroids)
            for second in capture_centroids[position + 1 :]
        ),
        default=0.0,
    )
    image_diagonal = float(
        np.hypot(index.session.camera_width, index.session.camera_height)
    )
    if centroid_span_px > CAMERA_GEOMETRY_WARNING_FRACTION * image_diagonal:
        calibration_warnings.append(
            f"suspicious camera geometry change: unloaded centroid span {centroid_span_px:.3f} px"
        )

    frame_rows: list[dict[str, Any]] = []
    profiles: list[np.ndarray] = []
    for frame in index.frames:
        if frame.run is None or frame.target_force_n is None:
            continue
        image = load_rgb(frame.rgb_path)
        profile, optical_qc = longitudinal_green_profile(
            image, strip, bins=config.profile_bins
        )
        measurements = frame.measurements
        force = actual_force_magnitude(
            float(measurements["Fx_N"]),
            float(measurements["Fy_N"]),
            float(measurements["Fz_N"]),
        )
        row = {
            "specimen_id": index.session.specimen_id,
            "material": index.session.material,
            "morphology": index.session.morphology,
            "run_id": frame.run.run_id,
            "run_status": frame.run.status,
            "indenter": frame.run.indenter,
            "hole_index": frame.run.hole_index,
            "repetition_index": frame.run.repetition_index,
            "target_force_n": frame.target_force_n,
            "frame_index": int(measurements["frame_index"]),
            "expected_frame_count": index.session.force_sequence.expected_record_frame_count,
            "force_tolerance_n": index.session.force_sequence.tolerance_n(
                frame.target_force_n
            ),
            "acquisition_target_forces_n": ";".join(
                f"{value:g}"
                for value in index.session.force_sequence.target_forces_n
            ),
            "actual_force_n": force,
        }
        for name in (
            "camera_bota_time_delta_ms",
            "Fx_N",
            "Fy_N",
            "Fz_N",
            "Mx_Nm",
            "My_Nm",
            "Mz_Nm",
        ):
            row[name] = float(measurements[name])
        row.update(optical_qc)
        frame_rows.append(row)
        profiles.append(profile)

    invalid_coverage = [
        row for row in index.coverage_rows if row["validity"] != "valid"
    ]
    warning_summary = _warning_summary(invalid_coverage, index.issues)
    warning_summary.extend(calibration_warnings)
    dataset_row = {
        "specimen_id": index.session.specimen_id,
        "material": index.session.material,
        "morphology": index.session.morphology,
        "source_dataset_path": str(index.path),
        "dataset_format_version": FORMAT_VERSION,
        "source_git_commit": index.session.git_commit or "",
        "camera_model": index.session.camera_model,
        "camera_serial": index.session.camera_serial_number or "",
        "width": index.session.camera_width,
        "height": index.session.camera_height,
        "fps": index.session.camera_fps,
        "exposure_us": index.session.camera_exposure_us,
        "gain": index.session.camera_gain,
        "white_balance_k": index.session.camera_white_balance_k,
        "number_of_runs": len(index.runs),
        "number_of_loaded_images": len(frame_rows),
        "number_of_unloaded_images": len(unloaded_records),
        "number_of_indenters": len({run.indenter for run in index.runs}),
        "number_of_holes": len({run.hole_index for run in index.runs}),
        "number_of_repetitions": len(
            {run.repetition_index for run in index.runs}
        ),
        "acquisition_target_forces_n": ";".join(
            f"{value:g}" for value in index.session.force_sequence.target_forces_n
        ),
        "expected_frames_per_hold": index.session.force_sequence.expected_record_frame_count,
        "optical_support_fraction": float(np.mean(strip.support_mask)),
        "unloaded_geometry_centroid_span_px": centroid_span_px,
        "coverage_warnings": "; ".join(warning_summary),
    }
    return {
        "dataset_row": dataset_row,
        "frame_rows": frame_rows,
        "frame_profiles": profiles,
    }


def _qc_rows(
    indexes: list[SessionIndex],
    dataset_rows: list[dict[str, Any]],
    run_force_rows: list[dict[str, Any]],
    camera_warnings: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for warning in camera_warnings:
        rows.append(_qc_row("all", "camera_setting_mismatch", warning))
    for index, dataset in zip(indexes, dataset_rows, strict=True):
        for issue in index.issues:
            rows.append(_qc_row(index.specimen_id, "dataset_integrity", issue))
        for coverage in index.coverage_rows:
            if coverage["validity"] == "valid":
                continue
            message = (
                f"{coverage['indenter']}, hole {coverage['hole_index']}, "
                f"repetition {coverage['repetition_index']}, "
                f"{float(coverage['target_force_n']):g} N: {coverage['validity']}"
            )
            rows.append(
                _qc_row(index.specimen_id, str(coverage["validity"]), message)
            )
        geometry_span = float(dataset["unloaded_geometry_centroid_span_px"])
        image_diagonal = float(np.hypot(dataset["width"], dataset["height"]))
        if geometry_span > CAMERA_GEOMETRY_WARNING_FRACTION * image_diagonal:
            rows.append(
                _qc_row(
                    index.specimen_id,
                    "suspicious_camera_geometry_change",
                    f"unloaded optical-region centroid span is {geometry_span:.3f} px",
                )
            )
    for row in run_force_rows:
        for flag in str(row["qc_flags"]).split(";"):
            if not flag:
                continue
            message = (
                f"{row['run_id']} {float(row['target_force_n']):g} N: {flag}"
            )
            rows.append(_qc_row(str(row["specimen_id"]), flag, message))
    if not rows:
        rows.append(_qc_row("all", "none", "no QC warnings"))
    return rows


def _qc_row(specimen_id: str, code: str, message: str) -> dict[str, str]:
    return {"specimen_id": specimen_id, "qc_code": code, "message": message}


def _warning_summary(
    invalid_coverage: list[dict[str, Any]], issues: tuple[str, ...]
) -> list[str]:
    counts: dict[str, int] = {}
    for row in invalid_coverage:
        validity = str(row["validity"])
        counts[validity] = counts.get(validity, 0) + 1
    return [f"{key}={counts[key]}" for key in sorted(counts)] + list(issues)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _write_profile_npz(
    path: Path,
    array_name: str,
    profiles: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    include_force: bool,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SUMMARY_SCHEMA_VERSION, dtype=np.int16),
        "longitudinal_coordinate": np.linspace(
            0.0, 1.0, profiles.shape[1], dtype=np.float32
        ),
        array_name: np.asarray(profiles, dtype=np.float32),
        "specimen_id": _strings(rows, "specimen_id"),
        "material": _strings(rows, "material"),
        "morphology": _strings(rows, "morphology"),
        "run_id": _strings(rows, "run_id"),
        "run_status": _strings(rows, "run_status"),
        "indenter": _strings(rows, "indenter"),
        "hole_index": np.asarray([row["hole_index"] for row in rows], dtype=np.int16),
        "repetition_index": np.asarray(
            [row["repetition_index"] for row in rows], dtype=np.int16
        ),
    }
    if include_force:
        arrays.update(
            {
                "target_force_n": np.asarray(
                    [row["target_force_n"] for row in rows], dtype=np.float32
                ),
                "actual_force_n": np.asarray(
                    [row["actual_force_median_n"] for row in rows], dtype=np.float32
                ),
                "frame_count": np.asarray(
                    [row["frame_count"] for row in rows], dtype=np.int16
                ),
            }
        )
    else:
        arrays["S_load_DN_per_N"] = np.asarray(
            [row["S_load_DN_per_N"] for row in rows], dtype=np.float32
        )
    np.savez_compressed(path, **arrays)


def _strings(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([str(row[key]) for row in rows], dtype=np.str_)


def _write_slope_profile_csv(
    path: Path, rows: list[dict[str, Any]], profiles: np.ndarray
) -> None:
    output = []
    for row, profile in zip(rows, profiles, strict=True):
        identity = {
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
        identity.update(
            {f"bin_{index:03d}": float(value) for index, value in enumerate(profile)}
        )
        output.append(identity)
    _write_csv(path, output)


def _write_readme(
    path: Path,
    *,
    dataset_rows: list[dict[str, Any]],
    camera_warnings: list[str],
    profile_bins: int,
) -> None:
    warning_text = "\n".join(f"- {warning}" for warning in camera_warnings)
    if not warning_text:
        warning_text = "- No cross-session camera-setting mismatch was detected."
    frame_counts = sorted({int(row["expected_frames_per_hold"]) for row in dataset_rows})
    path.joinpath("README.md").write_text(
        f"""# LUMO raw data summary

This directory is the compact, image-free representation of the supplied
format-v3 physical contact sessions. The statistical unit is one independent
run. The {frame_counts} scheduled frames within each force hold are repeated
observations and were reduced by a per-bin median.

`s(v,F)` is the raw Green camera intensity [DN] sampled on one fixed
unloaded-derived interior strip per specimen and stored at {profile_bins}
longitudinal bins. It is not unloaded-subtracted or magnitude-normalized.
Actual force is `sqrt(Fx^2 + Fy^2 + Fz^2)` and the hold median is the fit
coordinate. For each run and bin, `s(v,F)=a(v)+b(v)F` is fit across available
forces. `S_load = RMS_v(b(v))` [DN/N].

For each specimen, indenter, and hole, the hole template is the median slope
profile across independent runs. `D_neighbor` is the median RMS distance
between adjacent hole templates (1-2 through 5-6). Repeat variability `W` is
the median RMS distance from each independent run slope to its same-hole
template. Different indenters are never pooled. No trustworthy mechanical
deformation input is available, so `S_OM` is explicitly unavailable.

## Files

- `dataset_summary.csv`: one row per physical specimen/session and acquisition QC.
- `run_force_summary.csv`: one row per run and target force, including actual-force statistics, synchronized wrench medians, frame count, optical variation, saturation, and QC flags.
- `run_load_response.csv`: one row per run with `S_load`, finite differences, fit diagnostics, and unavailable `S_OM` status.
- `longitudinal_profiles.npz`: `profiles` (`N x {profile_bins}`, float32) plus specimen/run/indenter/hole/repetition/target/actual-force metadata and normalized longitudinal coordinates.
- `load_response_profiles.npz`: `slope_profiles` (`M x {profile_bins}`, float32) plus complete run identity, `S_load`, and normalized longitudinal coordinates.
- `load_response_profiles.csv`: the same slope profiles as `bin_000` through `bin_{profile_bins - 1:03d}`.
- `qc_summary.csv`: coverage, frame-count, force, saturation, camera-setting, and simple unloaded-geometry warnings.
- `suspect_runs.csv`: deterministic optical/metadata ranking for manual review only; it never repairs or deletes a run.

Target force identifies the requested hold. Always use `actual_force_n` for
quantitative fitting. Hole indices retain acquisition order; physical spacing
is reported only when supplied explicitly to the analysis command.

## Cross-session camera QC

{warning_text}
""",
        encoding="utf-8",
    )


def _write_summary_zip(output: Path, raw_summary: Path) -> None:
    archive = output / "raw_data_summary.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for source in sorted(raw_summary.rglob("*")):
            if not source.is_file():
                continue
            if source.suffix.lower() == ".png":
                raise RuntimeError("raw_data_summary must not contain PNG files")
            stream.write(source, Path("raw_data_summary") / source.relative_to(raw_summary))


__all__ = ["AnalysisConfig", "analyze_morphologies"]
