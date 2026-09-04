"""Export compact numerical analysis bundles and diagnostic figures."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.data_collection.contact_dataset import FORMAT_VERSION

from .dataset_index import SessionIndex


def export_analysis_bundle(
    output: Path,
    *,
    indexes: list[SessionIndex],
    frame_rows: list[dict[str, Any]],
    frame_signatures: np.ndarray,
    unloaded_rows: list[dict[str, Any]],
    aggregated: dict[str, Any],
    camera_warnings: list[str],
    extraction_details: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    """Write the requested compact uploadable analysis bundle."""

    bundle = output / "analysis_bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    figures = bundle / "figures"
    figures.mkdir(parents=True)
    coverage = [row for index in indexes for row in index.coverage_rows]
    session_rows = [
        _session_summary(index, details)
        for index, details in zip(indexes, extraction_details, strict=True)
    ]
    run_rows = aggregated["run_rows"]
    run_signatures = aggregated["run_signatures"]
    spatial_rows = []
    for row, signature in zip(run_rows, run_signatures, strict=True):
        spatial = {
            key: row[key]
            for key in (
                "specimen_id",
                "material",
                "morphology",
                "run_id",
                "indenter",
                "hole_index",
                "repetition_index",
                "target_force_n",
                "actual_force_mean_n",
            )
        }
        spatial.update(
            {f"bin_{index:03d}": float(value) for index, value in enumerate(signature)}
        )
        spatial_rows.append(spatial)
    files = {
        "session_summary.csv": session_rows,
        "coverage.csv": coverage,
        "frame_features.csv": frame_rows,
        "run_force_summary.csv": run_rows,
        "condition_summary.csv": aggregated["condition_rows"],
        "spatial_signatures.csv": spatial_rows,
        "pairwise_separability.csv": aggregated["pairwise_rows"],
        "force_response_fits.csv": aggregated["fit_rows"],
        "unloaded_stability.csv": unloaded_rows,
    }
    for filename, rows in files.items():
        _write_csv(bundle / filename, rows)

    issues = {
        index.session_id: list(index.issues)
        + sorted(
            {
                str(row["validity"])
                for row in index.coverage_rows
                if row["validity"] != "valid"
            }
        )
        for index in indexes
    }
    manifest = {
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "analysis_git_commit": _git_commit(),
        "analysis_worktree_dirty": _git_dirty(),
        "source_session_paths": [str(index.path) for index in indexes],
        "specimen_ids": [index.session_id for index in indexes],
        "source_dataset_format_versions": [FORMAT_VERSION for _ in indexes],
        "source_git_commits": [index.session.git_commit for index in indexes],
        "analysis_configuration": config,
        "csv_schemas": {
            filename: list(rows[0]) if rows else [] for filename, rows in files.items()
        },
        "camera_consistency_warnings": camera_warnings,
        "known_coverage_issues": issues,
        "extraction": extraction_details,
        "statistical_unit": "one run-force observation; hold frames are repeated observations",
        "paper_signal_channel": "G",
    }
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (bundle / "README.md").write_text(
        _readme(indexes, issues, camera_warnings, extraction_details, config),
        encoding="utf-8",
    )
    _make_figures(
        figures,
        session_rows=session_rows,
        coverage=coverage,
        frame_rows=frame_rows,
        run_rows=run_rows,
        condition_rows=aggregated["condition_rows"],
        run_signatures=run_signatures,
        pairwise_rows=aggregated["pairwise_rows"],
    )
    archive = output / "analysis_bundle.zip"
    if archive.exists():
        archive.unlink()
    shutil.make_archive(
        str(archive.with_suffix("")), "zip", root_dir=output, base_dir="analysis_bundle"
    )
    return bundle


def _session_summary(index: SessionIndex, extraction: dict[str, Any]) -> dict[str, Any]:
    session = index.session
    loaded = [frame for frame in index.frames if frame.run is not None]
    unloaded = [frame for frame in index.frames if frame.run is None]
    return {
        "source_session_path": str(index.path),
        "source_format_version": FORMAT_VERSION,
        "specimen_id": session.specimen_id,
        "material": session.material,
        "morphology": session.morphology,
        "camera_model": session.camera_model,
        "camera_serial_number": session.camera_serial_number,
        "camera_width": session.camera_width,
        "camera_height": session.camera_height,
        "camera_fps": session.camera_fps,
        "camera_exposure_us": session.camera_exposure_us,
        "camera_gain": session.camera_gain,
        "camera_white_balance_k": session.camera_white_balance_k,
        "auto_exposure": False,
        "auto_white_balance": False,
        "run_count": index.run_count,
        "loaded_frame_count": len(loaded),
        "unloaded_capture_count": index.unloaded_capture_count,
        "unloaded_frame_count": len(unloaded),
        "unloaded_capture_geometry_centroid_span_px": extraction[
            "unloaded_capture_geometry_centroid_span_px"
        ],
        "source_git_commit": session.git_commit,
    }


def _readme(
    indexes: list[SessionIndex],
    issues: dict[str, list[str]],
    camera_warnings: list[str],
    extraction_details: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    identities = "\n".join(
        f"- `{index.session_id}`: material `{index.session.material}`, morphology `{index.session.morphology}`, source `{index.path}`"
        for index in indexes
    )
    issue_lines = []
    for specimen, values in issues.items():
        if values:
            issue_lines.extend(f"- `{specimen}`: {value}" for value in values)
    if not issue_lines:
        issue_lines = ["- None detected."]
    warning_lines = [f"- {value}" for value in camera_warnings] or [
        "- Camera settings match across supplied sessions."
    ]
    pose_lines = [
        f"- `{details['specimen_id']}`: maximum unloaded-capture geometry-centroid separation = {float(details['unloaded_capture_geometry_centroid_span_px']):.3f} px"
        for details in extraction_details
    ]
    return f"""# LUMO physical contact analysis bundle

This compact bundle was derived from format-v3 fixed-camera contact datasets. It contains no raw image sequence and is sufficient to recompute run-, condition-, and spatial-comparison summaries.

## Sessions

{identities}

## Statistical and measurement conventions

- The independent sample is one run. Five frames in one force hold are repeated observations and are reduced by their median.
- Actual force is `sqrt(Fx^2 + Fy^2 + Fz^2)` from the synchronized Rokubi axes. Target force remains a grouping label.
- Each specimen uses the pixelwise temporal median of its own unloaded frames.
- Geometry and the canonical sampling strip are calibrated once from that unloaded reference and reused for every loaded frame.
- Optical differences are signed `loaded - unloaded` camera DN. Green is the paper-facing channel; R/G/B diagnostics are retained.
- MAE is `mean(abs(Delta I))`; RMS is `sqrt(mean(Delta I^2))`; signed mean is `mean(Delta I)`.
- Force-normalized values divide each frame response by its measured force before within-hold aggregation.
- Deformation is image-space motion of the visible contour edge along fixed unloaded normals and is reported only in pixels.
- Each 128-bin spatial signature is the unnormalized signed transverse mean of canonical `Delta G`, resampled longitudinally without high-pass filtering or PCA.
- Hole templates are medians over independent run signatures. Pairwise separation is their RMS distance. Repeat variability is each run's RMS distance from its hole template.
- Ordinary and through-origin force-response fits are descriptive summaries, not physical linearity assumptions.

## Coverage observations

{chr(10).join(issue_lines)}

## Camera comparability

{chr(10).join(warning_lines)}

## Fixed-pose diagnostic

{chr(10).join(pose_lines)}

This value is reported without an acceptance threshold. A large value means the fixed-pose premise should be reviewed before interpreting canonical DN differences.

## Configuration

```json
{json.dumps(config, indent=2)}
```

## Files

- `session_summary.csv`: session identities, fixed camera settings, and counts.
- `coverage.csv`: expected condition grid plus missing, duplicate, incomplete, or unexpected entries.
- `frame_features.csv`: synchronized FT values, RGB response/saturation, and deformation per raw loaded frame.
- `run_force_summary.csv`: one median-reduced independent run-force observation.
- `condition_summary.csv`: robust summaries across independent runs.
- `spatial_signatures.csv`: one signed 128-bin longitudinal signature per run-force observation.
- `pairwise_separability.csv`: pairwise hole-template distances and repeat-variability ratios.
- `force_response_fits.csv`: ordinary and through-origin force-response summaries.
- `unloaded_stability.csv`: deviations of each unloaded frame from its specimen median.
- `figures/`: diagnostic plots only; these are not publication-polished figures.
"""


def _make_figures(
    directory: Path,
    *,
    session_rows: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    run_signatures: np.ndarray,
    pairwise_rows: list[dict[str, Any]],
) -> None:
    labels = {row["specimen_id"]: row["morphology"] for row in session_rows}
    colors = {
        specimen: plt.cm.viridis(value)
        for specimen, value in zip(
            labels, np.linspace(0.15, 0.85, len(labels)), strict=True
        )
    }

    valid_counts = []
    invalid_counts = []
    for specimen in labels:
        rows = [row for row in coverage if row["specimen_id"] == specimen]
        valid_counts.append(sum(row["validity"] == "valid" for row in rows))
        invalid_counts.append(sum(row["validity"] != "valid" for row in rows))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(labels))
    ax.bar(x, valid_counts, color="#2a788e", label="valid")
    ax.bar(
        x,
        invalid_counts,
        bottom=valid_counts,
        color="#d95f02",
        label="nonstandard/missing",
    )
    ax.set(
        xticks=x,
        xticklabels=list(labels.values()),
        ylabel="Expected run-force cells",
        title="Experimental coverage",
    )
    ax.legend()
    _save(fig, directory / "01_experimental_coverage.png")

    _categorical_boxplot(
        directory / "02_actual_force_distributions.png",
        frame_rows,
        "actual_force_n",
        labels,
        "Actual force [N]",
        "Actual force by target and morphology",
    )
    _scatter_by_specimen(
        directory / "03_optical_response_vs_force.png",
        run_rows,
        labels,
        colors,
        "actual_force_mean_n",
        "optical_mae_G_dn",
        "Actual force [N]",
        "Green optical MAE [DN]",
        "Optical response",
    )
    _scatter_by_specimen(
        directory / "04_optical_response_per_n.png",
        run_rows,
        labels,
        colors,
        "actual_force_mean_n",
        "optical_mae_G_dn_per_n",
        "Actual force [N]",
        "Green optical MAE / force [DN/N]",
        "Force-normalized optical response",
    )
    _scatter_by_specimen(
        directory / "05_deformation_vs_force.png",
        run_rows,
        labels,
        colors,
        "actual_force_mean_n",
        "deformation_rms_px",
        "Actual force [N]",
        "Contour displacement RMS [px]",
        "Image deformation",
    )
    _scatter_by_specimen(
        directory / "06_optical_response_vs_deformation.png",
        run_rows,
        labels,
        colors,
        "deformation_rms_px",
        "optical_mae_G_dn",
        "Contour displacement RMS [px]",
        "Green optical MAE [DN]",
        "Optical response vs deformation",
    )
    _scatter_by_specimen(
        directory / "07_optomechanical_response_vs_force.png",
        run_rows,
        labels,
        colors,
        "actual_force_mean_n",
        "optical_mae_G_dn_per_deformation_px",
        "Actual force [N]",
        "Green MAE / contour RMS [DN/px]",
        "Optomechanical response",
    )
    _signature_figure(
        directory / "08_longitudinal_signatures.png",
        run_rows,
        run_signatures,
        labels,
        colors,
    )
    _heatmap_figure(directory / "09_pairwise_hole_distances.png", pairwise_rows, labels)
    neighbors = [row for row in pairwise_rows if int(row["hole_index_separation"]) == 1]
    _pairwise_scatter(
        directory / "10_neighboring_hole_distance.png",
        neighbors,
        labels,
        colors,
        "optical_rms_distance_dn",
        "Neighboring-hole RMS distance [DN]",
        "Neighboring-hole separation",
    )
    _condition_scatter(
        directory / "11_repeat_variability.png",
        condition_rows,
        labels,
        colors,
        "signature_repeat_variability_median_dn",
        "Repeat variability [DN]",
        "Within-location repeat variability",
    )
    _pairwise_scatter(
        directory / "12_distance_variability_ratio.png",
        neighbors,
        labels,
        colors,
        "distance_to_variability_ratio",
        "Distance / repeat variability",
        "Neighboring separation-to-variability ratio",
    )


def _categorical_boxplot(
    path: Path,
    rows: list[dict[str, Any]],
    key: str,
    labels: dict[str, str],
    ylabel: str,
    title: str,
) -> None:
    groups = []
    names = []
    for specimen in labels:
        for target in sorted({float(row["target_force_n"]) for row in rows}):
            values = [
                float(row[key])
                for row in rows
                if row["specimen_id"] == specimen
                and float(row["target_force_n"]) == target
            ]
            if values:
                groups.append(values)
                names.append(f"{labels[specimen]}\n{target:g} N")
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.boxplot(groups, tick_labels=names, showfliers=False)
    ax.set(ylabel=ylabel, title=title)
    _save(fig, path)


def _scatter_by_specimen(
    path: Path,
    rows: list[dict[str, Any]],
    labels: dict[str, str],
    colors: dict[str, Any],
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    plotted = False
    for specimen, label in labels.items():
        selected = [
            (float(row[x_key]), float(row[y_key]))
            for row in rows
            if row["specimen_id"] == specimen and _finite(row[y_key])
        ]
        if selected:
            ax.scatter(
                *np.asarray(selected).T,
                s=13,
                alpha=0.48,
                color=colors[specimen],
                label=label,
            )
            plotted = True
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if plotted:
        ax.legend()
    _save(fig, path)


def _signature_figure(
    path: Path,
    rows: list[dict[str, Any]],
    signatures: np.ndarray,
    labels: dict[str, str],
    colors: dict[str, Any],
) -> None:
    indenters = sorted({str(row["indenter"]) for row in rows})
    fig, axes = plt.subplots(
        1,
        len(indenters),
        figsize=(5.0 * len(indenters), 3.8),
        squeeze=False,
        sharey=True,
    )
    x = np.linspace(0.0, 1.0, signatures.shape[1])
    for ax, indenter in zip(axes[0], indenters, strict=True):
        for specimen, label in labels.items():
            for hole in range(1, 7):
                indices = [
                    i
                    for i, row in enumerate(rows)
                    if row["specimen_id"] == specimen
                    and row["indenter"] == indenter
                    and int(row["hole_index"]) == hole
                ]
                if indices:
                    ax.plot(
                        x,
                        np.median(signatures[indices], axis=0),
                        color=colors[specimen],
                        alpha=0.25 + 0.11 * hole,
                        lw=1.0,
                    )
        ax.set(title=indenter, xlabel="Normalized longitudinal coordinate")
    axes[0, 0].set_ylabel("Signed transverse-mean Delta G [DN]")
    fig.suptitle("Longitudinal signatures (line opacity increases with hole index)")
    _save(fig, path)


def _heatmap_figure(
    path: Path, rows: list[dict[str, Any]], labels: dict[str, str]
) -> None:
    panels = [
        (specimen, indenter)
        for specimen in labels
        for indenter in sorted(
            {str(row["indenter"]) for row in rows if row["specimen_id"] == specimen}
        )
    ]
    if not panels:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.text(
            0.5,
            0.5,
            "At least two observed holes are required",
            ha="center",
            va="center",
        )
        ax.set(title="Pairwise hole distances", xticks=[], yticks=[])
        _save(fig, path)
        return
    targets = sorted({float(row["target_force_n"]) for row in rows})
    fig, axes = plt.subplots(
        len(targets),
        len(panels),
        figsize=(3.4 * len(panels), 2.8 * len(targets)),
        squeeze=False,
        layout="constrained",
    )
    maximum = max((float(row["optical_rms_distance_dn"]) for row in rows), default=1.0)
    image = None
    for row_index, target in enumerate(targets):
        for column_index, (specimen, indenter) in enumerate(panels):
            ax = axes[row_index, column_index]
            matrix = np.zeros((6, 6))
            selected = [
                row
                for row in rows
                if row["specimen_id"] == specimen
                and row["indenter"] == indenter
                and float(row["target_force_n"]) == target
            ]
            for row in selected:
                i, j = int(row["hole_i"]) - 1, int(row["hole_j"]) - 1
                matrix[i, j] = matrix[j, i] = float(row["optical_rms_distance_dn"])
            image = ax.imshow(matrix, vmin=0.0, vmax=maximum, cmap="viridis")
            ax.set(
                title=f"{labels[specimen]} · {indenter}" if row_index == 0 else "",
                xlabel="Hole" if row_index == len(targets) - 1 else "",
                ylabel=f"{target:g} N\nHole" if column_index == 0 else "",
                xticks=range(6),
                yticks=range(6),
                xticklabels=range(1, 7),
                yticklabels=range(1, 7),
            )
    fig.colorbar(image, ax=axes, label="Template RMS distance [DN]", shrink=0.75)
    _save(fig, path, tight=False)


def _pairwise_scatter(
    path: Path,
    rows: list[dict[str, Any]],
    labels: dict[str, str],
    colors: dict[str, Any],
    key: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for specimen, label in labels.items():
        selected = [
            row for row in rows if row["specimen_id"] == specimen and _finite(row[key])
        ]
        ax.scatter(
            [float(row["target_force_n"]) for row in selected],
            [float(row[key]) for row in selected],
            s=18,
            alpha=0.6,
            color=colors[specimen],
            label=label,
        )
    ax.set(xlabel="Target force [N]", ylabel=ylabel, title=title)
    if rows:
        ax.legend()
    _save(fig, path)


def _condition_scatter(
    path: Path,
    rows: list[dict[str, Any]],
    labels: dict[str, str],
    colors: dict[str, Any],
    key: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for specimen, label in labels.items():
        selected = [
            row for row in rows if row["specimen_id"] == specimen and _finite(row[key])
        ]
        ax.scatter(
            [float(row["target_force_n"]) for row in selected],
            [float(row[key]) for row in selected],
            s=18,
            alpha=0.6,
            color=colors[specimen],
            label=label,
        )
    ax.set(xlabel="Target force [N]", ylabel=ylabel, title=title)
    if any(row for row in rows if _finite(row[key])):
        ax.legend()
    _save(fig, path)


def _save(fig: Any, path: Path, *, tight: bool = True) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if tight:
            fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str | None:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def _git_dirty() -> bool:
    result = subprocess.run(
        ("git", "status", "--porcelain"),
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


__all__ = ["export_analysis_bundle"]
