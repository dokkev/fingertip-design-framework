"""Small publication-facing plots for morphology analysis outputs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def write_figures(
    output_directory: str | Path,
    *,
    run_rows: list[dict[str, Any]],
    slope_profiles: np.ndarray,
    neighboring_rows: list[dict[str, Any]],
    variability_rows: list[dict[str, Any]],
    morphology_rows: list[dict[str, Any]],
) -> None:
    """Write the four requested optical morphology-comparison figures."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _plot_load_sensitivity(output, run_rows)
    _plot_slope_profiles(output, run_rows, slope_profiles)
    _plot_neighbor_and_repeat(output, neighboring_rows, variability_rows)
    _plot_separability_ratio(output, morphology_rows)


def _plot_load_sensitivity(output: Path, rows: list[dict[str, Any]]) -> None:
    indenters = sorted({str(row["indenter"]) for row in rows})
    if not indenters:
        return
    figure, axes = plt.subplots(
        1, len(indenters), figsize=(3.4 * len(indenters), 3.0), squeeze=False
    )
    for axis, indenter in zip(axes[0], indenters, strict=True):
        groups = _morphology_groups(rows, indenter)
        for position, (label, values) in enumerate(groups.items()):
            finite = np.asarray(values, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            if not len(finite):
                continue
            offsets = np.linspace(-0.12, 0.12, len(finite)) if len(finite) > 1 else [0]
            axis.scatter(
                position + np.asarray(offsets), finite, s=10, alpha=0.45, color="#31688e"
            )
            median = float(np.median(finite))
            low, high = np.percentile(finite, (25, 75))
            axis.errorbar(
                position,
                median,
                yerr=((median - low,), (high - median,)),
                fmt="o",
                color="black",
                capsize=3,
                markersize=4,
            )
        axis.set(
            title=_indenter_label(indenter),
            ylabel=r"$S_{load}$ [DN/N]",
            xticks=range(len(groups)),
            xticklabels=list(groups),
        )
        axis.tick_params(axis="x", rotation=25)
    _save(figure, output / "optical_load_sensitivity")


def _plot_slope_profiles(
    output: Path,
    rows: list[dict[str, Any]],
    slope_profiles: np.ndarray,
) -> None:
    profiles = np.asarray(slope_profiles, dtype=np.float64)
    if not len(rows):
        return
    conditions = sorted(
        {(str(row["material"]), str(row["morphology"])) for row in rows}
    )
    indenters = sorted({str(row["indenter"]) for row in rows})
    figure, axes = plt.subplots(
        len(indenters),
        len(conditions),
        figsize=(max(5.5, 3.0 * len(conditions)), 2.5 * len(indenters) + 0.5),
        squeeze=False,
        sharex=True,
    )
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, 6))
    coordinate = np.linspace(0.0, 1.0, profiles.shape[1])
    for row_index, indenter in enumerate(indenters):
        for column_index, condition in enumerate(conditions):
            axis = axes[row_index, column_index]
            for hole in range(1, 7):
                indices = [
                    index
                    for index, row in enumerate(rows)
                    if (str(row["material"]), str(row["morphology"])) == condition
                    and str(row["indenter"]) == indenter
                    and int(row["hole_index"]) == hole
                    and np.all(np.isfinite(profiles[index]))
                ]
                if indices:
                    axis.plot(
                        coordinate,
                        np.median(profiles[indices], axis=0),
                        color=colors[hole - 1],
                        label=f"Hole {hole}",
                    )
            axis.set_title(f"{condition[0]} · {condition[1]} · {_indenter_label(indenter)}")
            axis.set_xlabel("Longitudinal coordinate")
            axis.set_ylabel(r"$b(v)$ [DN/N]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(3, len(handles)),
            frameon=False,
            fontsize=8,
        )
    _save(figure, output / "load_response_profiles", rect=(0.0, 0.0, 1.0, 0.9))


def _plot_neighbor_and_repeat(
    output: Path,
    neighboring: list[dict[str, Any]],
    variability: list[dict[str, Any]],
) -> None:
    indenters = sorted(
        {str(row["indenter"]) for row in neighboring + variability}
    )
    if not indenters:
        return
    figure, axes = plt.subplots(
        1, len(indenters), figsize=(3.6 * len(indenters), 3.0), squeeze=False
    )
    for axis, indenter in zip(axes[0], indenters, strict=True):
        labels = sorted(
            {
                _morphology_label(row)
                for row in neighboring + variability
                if str(row["indenter"]) == indenter
            }
        )
        positions = np.arange(len(labels), dtype=np.float64)
        d_values = [
            _median(
                row["D_neighbor_DN_per_N"]
                for row in neighboring
                if str(row["indenter"]) == indenter
                and _morphology_label(row) == label
            )
            for label in labels
        ]
        w_values = [
            _median(
                row["repeat_variability_DN_per_N"]
                for row in variability
                if str(row["indenter"]) == indenter
                and _morphology_label(row) == label
            )
            for label in labels
        ]
        axis.bar(positions - 0.18, d_values, 0.36, label=r"$D_{neighbor}$")
        axis.bar(positions + 0.18, w_values, 0.36, label=r"$W$")
        axis.set(
            title=_indenter_label(indenter),
            ylabel="Profile difference [DN/N]",
            xticks=positions,
            xticklabels=labels,
        )
        axis.tick_params(axis="x", rotation=25)
    axes[0, 0].legend(frameon=False)
    _save(figure, output / "neighboring_separability_and_repeat_variation")


def _plot_separability_ratio(
    output: Path, rows: list[dict[str, Any]]
) -> None:
    if not rows:
        return
    labels = [f"{_morphology_label(row)}\n{_indenter_label(str(row['indenter']))}" for row in rows]
    values = [float(row["D_neighbor_over_W"]) for row in rows]
    figure, axis = plt.subplots(figsize=(max(4.0, 0.8 * len(rows)), 3.0))
    positions = np.arange(len(rows))
    axis.bar(positions, values, color="#35b779")
    axis.set(
        ylabel=r"$D_{neighbor}/W$",
        xticks=positions,
        xticklabels=labels,
    )
    axis.tick_params(axis="x", rotation=25)
    _save(figure, output / "separability_relative_to_repeat_variation")


def _morphology_groups(
    rows: list[dict[str, Any]], indenter: str
) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if str(row["indenter"]) == indenter:
            groups[_morphology_label(row)].append(float(row["S_load_DN_per_N"]))
    return dict(sorted(groups.items()))


def _morphology_label(row: dict[str, Any]) -> str:
    return f"{row['material']} · {row['morphology']}"


def _indenter_label(indenter: str) -> str:
    return indenter.replace("sphere_", "Sphere ").replace("mm", " mm")


def _median(values: Any) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else float("nan")


def _save(
    figure: plt.Figure,
    path: Path,
    *,
    rect: tuple[float, float, float, float] | None = None,
) -> None:
    figure.tight_layout(rect=rect)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


__all__ = ["write_figures"]
