"""Compose the four-panel Figure 6 perception-workload analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

from experiments.analysis.fig5c_decoder import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    PaperFigureConfig,
    load_config,
    read_csv,
    run_analysis,
)
from lumo.visualization import (  # noqa: E402
    DEFAULT_STYLE,
    EDGE_COLOR,
    publication_context,
    save_figure,
)


def _measured(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["status"] == "measured"]


def _condition_labels(config: PaperFigureConfig) -> list[str]:
    return [
        f"{config.material_labels[material]}\n"
        f"{config.indenter_labels[indenter].replace(' sphere', '')}"
        for material in config.materials
        for indenter in config.indenters
    ]


def _lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    return {
        (row["material"], row["indenter"], row["morphology_id"]): row for row in rows
    }


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#777777")
    axis.spines["bottom"].set_color("#777777")
    axis.grid(axis="y", color="#E3E3E3", linewidth=0.45, zorder=0)
    axis.tick_params(labelsize=6.0, length=2.0, pad=1.5)


def _panel_title(axis: plt.Axes, label: str, title: str) -> None:
    axis.set_title(
        f"{label}  {title}",
        loc="left",
        fontsize=7.5,
        fontweight="bold",
        pad=4.0,
    )


def _plot_baseline_relative_changes(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    config: PaperFigureConfig,
) -> None:
    values = _measured(rows)
    annotation_offsets = {
        ("solaris", "sphere_10mm", "flat_opt"): (6, 5),
        ("solaris", "sphere_10mm", "angled_opt"): (6, -8),
        ("solaris", "sphere_30mm", "flat_opt"): (-7, 6),
        ("solaris", "sphere_30mm", "angled_opt"): (-6, 0),
        ("dragon_skin", "sphere_10mm", "flat_opt"): (-6, -7),
        ("dragon_skin", "sphere_10mm", "angled_opt"): (6, 6),
        ("dragon_skin", "sphere_30mm", "flat_opt"): (6, 5),
        ("dragon_skin", "sphere_30mm", "angled_opt"): (6, -8),
    }
    for row in values:
        morphology = row["morphology"]
        x_value = float(row["magnitude_change_percent"])
        y_value = float(row["accuracy_change_pp"])
        material_code = config.material_labels[row["material"]][0].upper()
        indenter_code = config.indenter_labels[row["indenter"]].split()[0]
        offset = annotation_offsets.get(
            (row["material"], row["indenter"], morphology),
            (6, 5) if morphology == "flat_opt" else (6, -8),
        )
        axis.scatter(
            x_value,
            y_value,
            s=25,
            color=config.morphology_colors[morphology],
            edgecolor=EDGE_COLOR,
            linewidth=0.45,
            zorder=3,
        )
        axis.annotate(
            f"{material_code}{indenter_code}",
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=5.3,
            color="#3F3F3F",
            ha="right" if offset[0] < 0 else "left",
            va="center",
        )
    axis.axvline(0.0, color="#777777", linewidth=0.7, linestyle="--", zorder=1)
    axis.axhline(0.0, color="#777777", linewidth=0.7, linestyle="--", zorder=1)
    if values:
        x_values = [float(row["magnitude_change_percent"]) for row in values]
        y_values = [float(row["accuracy_change_pp"]) for row in values]
        x_span = max(x_values + [0.0]) - min(x_values + [0.0])
        y_span = max(y_values + [0.0]) - min(y_values + [0.0])
        x_pad = max(1.0, 0.09 * x_span)
        y_pad = max(1.0, 0.10 * y_span)
        axis.set_xlim(min(x_values + [0.0]) - x_pad, max(x_values + [0.0]) + x_pad)
        axis.set_ylim(min(y_values + [0.0]) - y_pad, max(y_values + [0.0]) + y_pad)
    else:
        axis.set_xlim(-1.0, 1.0)
        axis.set_ylim(-1.0, 1.0)
    axis.set_xlabel("Change in optical magnitude vs. baseline [%]")
    axis.set_ylabel("Change in decoding accuracy vs. baseline [pp]")
    _panel_title(
        axis,
        "(a)",
        "Decoding improves without increased signal magnitude",
    )
    _style_axis(axis)


def _plot_distinguishability(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    config: PaperFigureConfig,
) -> None:
    lookup = _lookup(rows)
    centers = np.arange(len(config.materials) * len(config.indenters), dtype=float)
    offsets = np.linspace(-0.23, 0.23, len(config.morphologies))
    measured_values = []
    condition_index = 0
    for material in config.materials:
        for indenter in config.indenters:
            for morphology_index, morphology in enumerate(config.morphologies):
                row = lookup[(material, indenter, morphology)]
                x = centers[condition_index] + offsets[morphology_index]
                if row["status"] != "measured":
                    axis.text(
                        x,
                        0.1,
                        "n/a",
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=5.0,
                    )
                    continue
                value = float(row["Q_sep"])
                measured_values.append(value)
                axis.bar(
                    x,
                    value,
                    width=0.20,
                    color=config.morphology_colors[morphology],
                    edgecolor=EDGE_COLOR,
                    linewidth=0.4,
                    zorder=2,
                )
            condition_index += 1
    axis.set_ylim(0.0, 1.12 * max(measured_values))
    axis.set_xticks(centers, _condition_labels(config))
    axis.set_ylabel(r"$Q_{\mathrm{sep}}=D_{\mathrm{neighbor}}/W_{\mathrm{contact}}$")
    _panel_title(axis, "(b)", "Separation exceeds re-contact variation")
    _style_axis(axis)
    axis.tick_params(axis="x", length=0.0)


def _plot_scalar_spatial(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    config: PaperFigureConfig,
) -> None:
    lookup = _lookup(rows)
    centers = np.arange(len(config.materials) * len(config.indenters), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(config.morphologies))
    all_values = []
    condition_index = 0
    for material in config.materials:
        for indenter in config.indenters:
            for morphology_index, morphology in enumerate(config.morphologies):
                row = lookup[(material, indenter, morphology)]
                if row["status"] != "measured":
                    continue
                x = centers[condition_index] + offsets[morphology_index]
                scalar = float(row["scalar_only_accuracy_percent"])
                spatial = float(row["spatial_6_region_accuracy_percent"])
                all_values.extend((scalar, spatial))
                color = config.morphology_colors[morphology]
                axis.plot(
                    (x, x), (scalar, spatial), color=color, linewidth=1.0, zorder=1
                )
                axis.scatter(
                    x,
                    scalar,
                    s=19,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.0,
                    zorder=2,
                )
                axis.scatter(
                    x,
                    spatial,
                    s=19,
                    facecolor=color,
                    edgecolor=EDGE_COLOR,
                    linewidth=0.4,
                    zorder=3,
                )
            condition_index += 1
    axis.set_ylim(max(0.0, 5.0 * np.floor((min(all_values) - 5.0) / 5.0)), 102.0)
    axis.set_xticks(centers, _condition_labels(config))
    axis.set_ylabel("Exact location accuracy [%]")
    _panel_title(axis, "(c)", "Spatial structure enables decoding")
    _style_axis(axis)
    axis.tick_params(axis="x", length=0.0)
    axis.legend(
        handles=(
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="white",
                markeredgecolor="#555555",
                linestyle="none",
                label="Scalar only",
            ),
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="#555555",
                markeredgecolor="#555555",
                linestyle="none",
                label="6-region spatial",
            ),
        ),
        loc="lower right",
        frameon=False,
        fontsize=5.5,
        handletextpad=0.35,
        labelspacing=0.25,
    )


def _plot_calibration(
    figure: plt.Figure,
    subplot_spec: Any,
    rows: list[dict[str, str]],
    config: PaperFigureConfig,
) -> tuple[plt.Axes, ...]:
    grid = subplot_spec.subgridspec(
        3,
        2,
        height_ratios=(0.17, 1.0, 1.0),
        hspace=0.36,
        wspace=0.30,
    )
    title_axis = figure.add_subplot(grid[0, :])
    title_axis.axis("off")
    title_axis.text(
        0.0,
        0.55,
        "(d)  Calibration burden",
        fontsize=7.5,
        fontweight="bold",
        va="center",
    )
    measured = _measured(rows)
    global_min = min(float(row["test_accuracy_q25"]) for row in measured)
    y_min = max(0.0, 5.0 * np.floor((global_min - 7.0) / 5.0))
    axes = []
    for condition_index, (material, indenter) in enumerate(
        (
            pair
            for material in config.materials
            for pair in (
                (material, config.indenters[0]),
                (material, config.indenters[1]),
            )
        )
    ):
        axis = figure.add_subplot(grid[1 + condition_index // 2, condition_index % 2])
        for morphology in config.morphologies:
            selected = sorted(
                (
                    row
                    for row in measured
                    if row["material"] == material
                    and row["indenter"] == indenter
                    and row["morphology_id"] == morphology
                ),
                key=lambda row: int(row["calibration_contacts_per_location"]),
            )
            if not selected:
                continue
            x = np.asarray(
                [int(row["calibration_contacts_per_location"]) for row in selected]
            )
            mean = np.asarray([float(row["test_accuracy_mean"]) for row in selected])
            q25 = np.asarray([float(row["test_accuracy_q25"]) for row in selected])
            q75 = np.asarray([float(row["test_accuracy_q75"]) for row in selected])
            color = config.morphology_colors[morphology]
            axis.fill_between(x, q25, q75, color=color, alpha=0.12, linewidth=0.0)
            axis.plot(x, mean, marker="o", markersize=2.8, color=color, linewidth=0.9)
        axis.set_title(
            config.condition_label(material, indenter), fontsize=6.0, pad=2.0
        )
        axis.set_xlim(0.85, config.maximum_calibration_contacts + 0.15)
        axis.set_ylim(y_min, 102.0)
        axis.set_xticks(range(1, config.maximum_calibration_contacts + 1))
        if condition_index // 2 == 1:
            axis.set_xlabel("Calibration contacts / location", fontsize=5.8)
        if condition_index % 2 == 0:
            axis.set_ylabel("Accuracy [%]", fontsize=5.8)
        _style_axis(axis)
        axis.tick_params(labelsize=5.2)
        axes.append(axis)
    return tuple(axes)


def build_figure(config: PaperFigureConfig) -> plt.Figure:
    """Build Figure 6 from the compact machine-readable summaries."""

    output = config.analysis_output_directory
    magnitude = read_csv(output / "fig6a_magnitude_vs_accuracy.csv")
    distinguishability = read_csv(output / "fig6b_spatial_distinguishability.csv")
    scalar_spatial = read_csv(output / "fig6c_scalar_vs_spatial.csv")
    calibration = read_csv(output / "fig6d_calibration_burden.csv")

    figure = plt.figure(figsize=(DEFAULT_STYLE.double_column_width_in, 4.75))
    grid = figure.add_gridspec(
        2,
        2,
        left=0.070,
        right=0.992,
        bottom=0.075,
        top=0.925,
        hspace=0.34,
        wspace=0.26,
    )
    _plot_baseline_relative_changes(figure.add_subplot(grid[0, 0]), magnitude, config)
    _plot_distinguishability(figure.add_subplot(grid[0, 1]), distinguishability, config)
    _plot_scalar_spatial(figure.add_subplot(grid[1, 0]), scalar_spatial, config)
    _plot_calibration(figure, grid[1, 1], calibration, config)
    figure.legend(
        handles=[
            Patch(
                facecolor=config.morphology_colors[morphology],
                edgecolor=EDGE_COLOR,
                linewidth=0.4,
                label=config.morphology_labels[morphology],
            )
            for morphology in config.morphologies
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(config.morphologies),
        frameon=False,
        fontsize=6.5,
        columnspacing=1.2,
        handletextpad=0.45,
    )
    return figure


def save_final(
    config: PaperFigureConfig, *, recompute: bool = False
) -> tuple[Path, ...]:
    """Write the final Figure 6 PDF and PNG."""

    required = (
        "fig6a_magnitude_vs_accuracy.csv",
        "fig6b_spatial_distinguishability.csv",
        "fig6c_scalar_vs_spatial.csv",
        "fig6d_calibration_burden.csv",
    )
    if recompute or any(
        not (config.analysis_output_directory / name).is_file() for name in required
    ):
        run_analysis(config)
    config.figure6_output_directory.mkdir(parents=True, exist_ok=True)
    with publication_context(DEFAULT_STYLE):
        figure = build_figure(config)
        outputs = save_figure(
            figure,
            config.figure6_output_directory / "fig6_final",
            formats=("pdf", "png"),
            bbox_inches=None,
            pad_inches=0.0,
        )
        plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--recompute", action="store_true")
    arguments = parser.parse_args()
    for path in save_final(
        load_config(arguments.config), recompute=arguments.recompute
    ):
        print(path)


if __name__ == "__main__":
    main()
