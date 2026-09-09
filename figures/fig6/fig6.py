"""Compose the canonical double-column Figure 6 analysis."""

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
import numpy as np  # noqa: E402

from experiments.analysis.fig5c_decoder import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    PaperFigureConfig,
    load_config,
    read_csv,
    run_analysis,
)
from figures.fig6.fig6d_force_timeseries import (  # noqa: E402
    DEFAULT_DATASET_ROOT as DEFAULT_FORCE_DATASET_ROOT,
    DEFAULT_OUTPUT_STEM as FORCE_TIMESERIES_OUTPUT_STEM,
    EXPERIMENT_LABEL as FORCE_EXPERIMENT_LABEL,
    build_force_timeseries,
    plot_force_timeseries,
    save_force_timeseries_panel,
    write_force_timeseries_csv,
)
from lumo.visualization import (  # noqa: E402
    DEFAULT_STYLE,
    EDGE_COLOR,
    STRUCTURAL_FONT_FAMILY,
    publication_context,
    save_figure,
)


FIGURE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_OUTPUT_STEM = FIGURE_DIRECTORY / "fig6"
OPTIONAL_CALIBRATION_OUTPUT_STEM = (
    FIGURE_DIRECTORY / "fig6c_optional_calibration_10mm_combined"
)
FIGURE_SIZE_IN = (DEFAULT_STYLE.double_column_width_in, 4.15)
PANEL_A_MORPHOLOGIES = ("flat_opt", "angled_opt")
DEFAULT_CYCLE_SUMMARIES = {
    "solaris": REPOSITORY_ROOT
    / "output"
    / "analysis"
    / "solaris_contact_history"
    / "results"
    / "same_contact_repeatability_summary.csv",
    "dragon_skin": REPOSITORY_ROOT
    / "output"
    / "analysis"
    / "dragon_skin_contact_history"
    / "results"
    / "same_contact_repeatability_summary.csv",
}


def _measured(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["status"] == "measured"]


def _condition_tick_labels(config: PaperFigureConfig) -> list[str]:
    return [
        config.indenter_labels[indenter].replace(" sphere", "")
        for _material in config.materials
        for indenter in config.indenters
    ]


def _lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    return {
        (row["material"], row["indenter"], row["morphology_id"]): row for row in rows
    }


def _cycle_lookup(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    lookup = {row["morphology"]: row for row in rows}
    required = {"baseline", *PANEL_A_MORPHOLOGIES}
    if set(lookup) != required:
        raise RuntimeError(
            f"expected exactly {sorted(required)} in {path}, got {sorted(lookup)}"
        )
    return lookup


def _load_panel_a_variability(
    config: PaperFigureConfig,
    recontact_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Join separated 10 mm maintained-contact and re-contact summaries."""

    recontact = _lookup(recontact_rows)
    output = []
    for material in config.materials:
        cycle = _cycle_lookup(DEFAULT_CYCLE_SUMMARIES[material])
        cycle_baseline = float(cycle["baseline"]["W_cycle_median_dn"])
        recontact_baseline_row = recontact[(material, "sphere_10mm", "baseline")]
        if recontact_baseline_row["status"] != "measured":
            raise RuntimeError(f"missing 10 mm baseline W_recontact for {material}")
        recontact_baseline = float(recontact_baseline_row["W_contact_DN_per_N"])
        if cycle_baseline <= 0.0 or recontact_baseline <= 0.0:
            raise RuntimeError(f"non-positive variability baseline for {material}")
        for morphology in PANEL_A_MORPHOLOGIES:
            cycle_value = float(cycle[morphology]["W_cycle_median_dn"])
            recontact_row = recontact[(material, "sphere_10mm", morphology)]
            if recontact_row["status"] != "measured":
                raise RuntimeError(
                    f"missing 10 mm W_recontact for {material}/{morphology}"
                )
            recontact_value = float(recontact_row["W_contact_DN_per_N"])
            output.append(
                {
                    "material": material,
                    "morphology": morphology,
                    "cycle_variability_change_percent": 100.0
                    * (cycle_value - cycle_baseline)
                    / cycle_baseline,
                    "recontact_variability_change_percent": 100.0
                    * (recontact_value - recontact_baseline)
                    / recontact_baseline,
                }
            )
    return output


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#777777")
    axis.spines["bottom"].set_color("#777777")
    axis.grid(axis="y", color="#E3E3E3", linewidth=0.45, zorder=0)
    axis.tick_params(
        labelsize=DEFAULT_STYLE.tick_font_size_pt,
        length=DEFAULT_STYLE.tick_length_pt,
        width=DEFAULT_STYLE.tick_width_pt,
        pad=1.5,
    )


def _panel_title(axis: plt.Axes, label: str, title: str) -> None:
    axis.text(
        0.0,
        1.025,
        label,
        transform=axis.transAxes,
        fontsize=DEFAULT_STYLE.panel_label_font_size_pt,
        fontweight="normal",
        ha="left",
        va="bottom",
        clip_on=False,
    )
    axis.set_title(
        title,
        loc="center",
        fontsize=DEFAULT_STYLE.panel_title_font_size_pt,
        fontweight="normal",
        pad=4.0,
    )


def _draw_panel_title(
    figure: plt.Figure,
    subplot_spec: Any,
    panel_label: str,
    title: str,
    *,
    y_offset: float = 0.012,
    label_x_offset: float = 0.0,
) -> None:
    """Place aligned panel labels and titles in figure coordinates."""

    bounds = subplot_spec.get_position(figure)
    y = bounds.y1 + y_offset
    figure.text(
        bounds.x0 + label_x_offset,
        y,
        panel_label,
        fontsize=DEFAULT_STYLE.panel_label_font_size_pt,
        fontweight="normal",
        ha="left",
        va="bottom",
    )
    figure.text(
        0.5 * (bounds.x0 + bounds.x1),
        y,
        title,
        fontsize=DEFAULT_STYLE.panel_title_font_size_pt,
        fontweight="normal",
        ha="center",
        va="bottom",
    )


def _plot_variability_compact(
    figure: plt.Figure,
    subplot_spec: Any,
    rows: list[dict[str, object]],
    config: PaperFigureConfig,
) -> None:
    """Plot the existing 10 mm variability comparison in a narrow grid cell."""

    grid = subplot_spec.subgridspec(
        3,
        2,
        height_ratios=(0.12, 0.17, 1.0),
        hspace=0.04,
        wspace=0.22,
    )
    condition_axis = figure.add_subplot(grid[0, :])
    condition_axis.axis("off")
    condition_axis.text(
        0.5,
        0.50,
        "10 mm sphere",
        fontsize=DEFAULT_STYLE.condition_header_font_size_pt,
        color="#555555",
        ha="center",
        va="center",
    )

    values = [
        float(row[key])
        for row in rows
        for key in (
            "cycle_variability_change_percent",
            "recontact_variability_change_percent",
        )
    ]
    limit = 10.0 * np.ceil(1.08 * max(abs(value) for value in values) / 10.0)
    lookup = {(str(row["material"]), str(row["morphology"])): row for row in rows}
    metric_columns = (
        (
            "cycle_variability_change_percent",
            "Maintained contact",
            r"$W_{\mathrm{cycle}}$",
        ),
        (
            "recontact_variability_change_percent",
            "Re-contact",
            r"$W_{\mathrm{recontact}}$",
        ),
    )
    centers = np.arange(len(config.materials), dtype=float)
    offsets = {"flat_opt": -0.09, "angled_opt": 0.09}
    for column, (value_key, header, notation) in enumerate(metric_columns):
        header_axis = figure.add_subplot(grid[1, column])
        header_axis.axis("off")
        header_axis.text(
            0.5,
            0.50,
            f"{header}\n{notation}",
            fontsize=DEFAULT_STYLE.minimum_font_size_pt,
            ha="center",
            va="center",
            linespacing=1.0,
        )
        axis = figure.add_subplot(grid[2, column])
        for material_index, material in enumerate(config.materials):
            for morphology in PANEL_A_MORPHOLOGIES:
                row = lookup[(material, morphology)]
                axis.scatter(
                    centers[material_index] + offsets[morphology],
                    float(row[value_key]),
                    s=21,
                    color=config.morphology_colors[morphology],
                    edgecolor=EDGE_COLOR,
                    linewidth=0.45,
                    zorder=3,
                )
        axis.axhline(0.0, color="#777777", linewidth=0.65, linestyle="--", zorder=1)
        axis.set_xlim(-0.34, len(config.materials) - 0.66)
        axis.set_ylim(-limit, limit)
        axis.set_xticks(centers, ("Solaris", "Dragon\nSkin"))
        _style_axis(axis)
        axis.tick_params(labelsize=DEFAULT_STYLE.minimum_font_size_pt, pad=1.0)
        if column == 0:
            axis.set_ylabel("Variability change [%]", labelpad=1.5)
        else:
            axis.tick_params(axis="y", labelleft=False)


def _plot_calibration_compact(
    figure: plt.Figure,
    subplot_spec: Any,
    rows: list[dict[str, str]],
    config: PaperFigureConfig,
) -> None:
    """Render the production calibration curves in a compact 2-by-2 block."""

    grid = subplot_spec.subgridspec(
        3,
        2,
        height_ratios=(0.13, 1.0, 1.0),
        hspace=0.16,
        wspace=0.18,
    )
    for column, indenter in enumerate(config.indenters):
        header_axis = figure.add_subplot(grid[0, column])
        header_axis.axis("off")
        header_axis.text(
            0.5,
            0.48,
            config.indenter_labels[indenter].replace(" sphere", ""),
            fontsize=DEFAULT_STYLE.condition_header_font_size_pt,
            ha="center",
            va="center",
        )
    measured = _measured(rows)
    global_min = min(float(row["test_accuracy_q25"]) for row in measured)
    y_min = max(0.0, 5.0 * np.floor((global_min - 7.0) / 5.0))
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
            axis.plot(x, mean, marker="o", markersize=2.5, color=color, linewidth=0.85)
        axis.set_xlim(0.85, config.maximum_calibration_contacts + 0.15)
        axis.set_ylim(y_min, 102.0)
        axis.set_xticks(range(1, config.maximum_calibration_contacts + 1))
        _style_axis(axis)
        axis.tick_params(labelsize=DEFAULT_STYLE.minimum_font_size_pt, pad=1.0)
        if condition_index // 2 == 0:
            axis.tick_params(axis="x", labelbottom=False)
        if condition_index % 2 == 1:
            axis.tick_params(axis="y", labelleft=False)
        if condition_index % 2 == 0:
            axis.text(
                0.04,
                0.08,
                config.material_labels[material],
                transform=axis.transAxes,
                fontsize=DEFAULT_STYLE.minimum_font_size_pt,
                fontweight="bold",
                fontfamily=STRUCTURAL_FONT_FAMILY,
                color="#444444",
                ha="left",
                va="bottom",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 0.5,
                },
            )

    bounds = subplot_spec.get_position(figure)
    figure.text(
        bounds.x0 - 0.050,
        0.5 * (bounds.y0 + bounds.y1),
        "Accuracy [%]",
        rotation=90,
        fontsize=DEFAULT_STYLE.axis_label_font_size_pt,
        ha="center",
        va="center",
    )
    figure.text(
        0.5 * (bounds.x0 + bounds.x1),
        bounds.y0 - 0.040,
        "Contacts / location",
        fontsize=DEFAULT_STYLE.axis_label_font_size_pt,
        ha="center",
        va="top",
    )


def _plot_placeholder(axis: plt.Axes, *, description: str) -> None:
    """Render one restrained, explicitly non-data placeholder panel."""

    axis.set_facecolor("#FAFAFA")
    axis.set_xticks(())
    axis.set_yticks(())
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#BFC3C7")
        spine.set_linewidth(DEFAULT_STYLE.spine_width_pt)
    axis.text(
        0.5,
        0.55,
        "Placeholder",
        fontsize=DEFAULT_STYLE.annotation_font_size_pt,
        color="#555555",
        ha="center",
        va="center",
    )
    axis.text(
        0.5,
        0.43,
        description,
        fontsize=DEFAULT_STYLE.minimum_font_size_pt,
        color="#777777",
        ha="center",
        va="center",
        linespacing=1.15,
    )


def _group_condition_ticks(axis: plt.Axes, config: PaperFigureConfig) -> None:
    """Label repeated indenter columns once per material group."""

    centers = np.arange(len(config.materials) * len(config.indenters), dtype=float)
    axis.set_xticks(centers, _condition_tick_labels(config))
    for material_index, material in enumerate(config.materials):
        first = material_index * len(config.indenters)
        center = first + 0.5 * (len(config.indenters) - 1)
        axis.text(
            center,
            -0.145,
            config.material_labels[material],
            transform=axis.get_xaxis_transform(),
            fontsize=DEFAULT_STYLE.group_header_font_size_pt,
            fontweight="bold",
            fontfamily=STRUCTURAL_FONT_FAMILY,
            ha="center",
            va="top",
            clip_on=False,
        )


def _plot_condition_bars(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    config: PaperFigureConfig,
    *,
    value_key: str,
    panel_label: str,
    title: str,
    y_label: str,
    show_title: bool = True,
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
                        fontsize=DEFAULT_STYLE.minimum_font_size_pt,
                    )
                    continue
                value = float(row[value_key])
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
    _group_condition_ticks(axis, config)
    axis.set_ylabel(y_label)
    if show_title:
        _panel_title(axis, panel_label, title)
    _style_axis(axis)
    axis.tick_params(axis="x", length=0.0)


def _plot_recontact_consistency(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    config: PaperFigureConfig,
) -> None:
    _plot_condition_bars(
        axis,
        rows,
        config,
        value_key="W_contact_DN_per_N",
        panel_label="(a)",
        title="Re-contact consistency",
        y_label=r"$W_{\mathrm{recontact}}$ [DN N$^{-1}$]",
    )


def _plot_distinguishability(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    config: PaperFigureConfig,
    *,
    show_title: bool = True,
) -> None:
    _plot_condition_bars(
        axis,
        rows,
        config,
        value_key="Q_sep",
        panel_label="(b)",
        title="Re-contact distinguishability",
        y_label=r"$Q_{\mathrm{recontact}}$",
        show_title=show_title,
    )


def _material_line_style(material: str) -> str:
    return "-" if material == "solaris" else "--"


def _morphology_marker(morphology: str) -> str:
    return {"baseline": "o", "flat_opt": "s", "angled_opt": "D"}[morphology]


def _optional_calibration_legend_handles(
    config: PaperFigureConfig,
) -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            color=config.morphology_colors[morphology],
            linestyle=_material_line_style(material),
            marker=_morphology_marker(morphology),
            markeredgecolor=EDGE_COLOR,
            markeredgewidth=0.4,
            linewidth=1.0,
            markersize=3.5,
            label=(
                f"{config.material_labels[material]} · "
                f"{config.morphology_labels[morphology]}"
            ),
        )
        for material in config.materials
        for morphology in config.morphologies
    ]


def _plot_optional_calibration(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    config: PaperFigureConfig,
    *,
    show_title: bool = True,
    show_legend: bool = False,
) -> None:
    """Plot geometry-prior k=0 and calibrated k=1..4 MAE regimes."""

    measured = [row for row in rows if row["status"] == "measured"]
    plotted_values: list[float] = []
    for material in config.materials:
        for morphology in config.morphologies:
            selected = sorted(
                (
                    row
                    for row in measured
                    if row["material"] == material
                    and row["indenter"] == "sphere_10mm"
                    and row["morphology_id"] == morphology
                ),
                key=lambda row: int(row["calibration_contacts_per_location"]),
            )
            if len(selected) != config.maximum_calibration_contacts + 1:
                raise RuntimeError(
                    f"optional-calibration series is incomplete for "
                    f"{material}/{morphology}"
                )
            x = np.asarray(
                [int(row["calibration_contacts_per_location"]) for row in selected]
            )
            if not np.array_equal(
                x, np.arange(config.maximum_calibration_contacts + 1)
            ):
                raise RuntimeError("optional-calibration x support must be 0 through 4")
            mean = np.asarray(
                [float(row["localization_mae_mean_mm"]) for row in selected]
            )
            plotted_values.extend(mean.tolist())
            color = config.morphology_colors[morphology]
            marker = _morphology_marker(morphology)
            calibrated = selected[1:]
            q25 = np.asarray(
                [float(row["localization_mae_q25_mm"]) for row in calibrated]
            )
            q75 = np.asarray(
                [float(row["localization_mae_q75_mm"]) for row in calibrated]
            )
            axis.plot(
                x[1:],
                mean[1:],
                color=color,
                linestyle=_material_line_style(material),
                linewidth=1.0,
                marker=marker,
                markersize=3.4,
                markeredgecolor=EDGE_COLOR,
                markeredgewidth=0.4,
                zorder=3,
            )
            axis.vlines(
                x[1:],
                q25,
                q75,
                color=color,
                linewidth=0.55,
                zorder=2,
            )
            for position, lower, upper_bound in zip(x[1:], q25, q75, strict=True):
                axis.hlines(
                    (lower, upper_bound),
                    position - 0.035,
                    position + 0.035,
                    color=color,
                    linewidth=0.55,
                    zorder=2,
                )
            axis.plot(
                x[:2],
                mean[:2],
                color=color,
                linestyle=(0, (1.0, 1.6)),
                linewidth=0.9,
                zorder=2,
            )
            axis.scatter(
                x[0],
                mean[0],
                s=18,
                marker=marker,
                color=color,
                edgecolor=EDGE_COLOR,
                linewidth=0.4,
                zorder=4,
            )
    axis.axvline(0.5, color="#888888", linewidth=0.65, linestyle=":", zorder=1)
    upper = max(5.0, 2.0 * np.ceil(1.10 * max(plotted_values) / 2.0))
    axis.set_xlim(-0.18, config.maximum_calibration_contacts + 0.18)
    axis.set_ylim(0.0, upper)
    axis.set_xticks(
        range(config.maximum_calibration_contacts + 1),
        ("0\nNo contact\ncalibration", "1", "2", "3", "4"),
    )
    axis.set_xlabel("Calibration contacts / location", labelpad=1.5)
    axis.set_ylabel("Localization MAE [mm]", labelpad=1.5)
    if show_title:
        _panel_title(axis, "(c)", "Optional calibration")
    _style_axis(axis)
    axis.tick_params(axis="x", length=0.0, pad=1.0)
    if show_legend:
        axis.legend(
            handles=_optional_calibration_legend_handles(config),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=3,
            frameon=False,
            fontsize=DEFAULT_STYLE.minimum_font_size_pt,
            columnspacing=0.8,
            handletextpad=0.35,
        )


def _plot_calibration(
    figure: plt.Figure,
    subplot_spec: Any,
    rows: list[dict[str, str]],
    config: PaperFigureConfig,
) -> tuple[plt.Axes, ...]:
    grid = subplot_spec.subgridspec(
        4,
        4,
        height_ratios=(0.17, 0.10, 1.0, 1.0),
        width_ratios=(0.08, 0.12, 1.0, 1.0),
        hspace=0.18,
        wspace=0.24,
    )
    title_axis = figure.add_subplot(grid[0, :])
    title_axis.axis("off")
    title_axis.text(
        0.0,
        0.55,
        "(d)",
        fontsize=DEFAULT_STYLE.panel_label_font_size_pt,
        fontweight="normal",
        va="center",
    )
    title_axis.text(
        0.065,
        0.55,
        "Calibration-set size",
        fontsize=DEFAULT_STYLE.panel_title_font_size_pt,
        fontweight="normal",
        va="center",
    )
    for column, indenter in enumerate(config.indenters, start=2):
        header_axis = figure.add_subplot(grid[1, column])
        header_axis.axis("off")
        header_axis.text(
            0.5,
            0.5,
            config.indenter_labels[indenter].replace(" sphere", ""),
            fontsize=DEFAULT_STYLE.condition_header_font_size_pt,
            ha="center",
            va="center",
        )
    y_label_axis = figure.add_subplot(grid[2:, 0])
    y_label_axis.axis("off")
    y_label_axis.text(
        0.5,
        0.5,
        "Accuracy [%]",
        rotation=90,
        fontsize=DEFAULT_STYLE.axis_label_font_size_pt,
        ha="center",
        va="center",
    )
    for row_index, material in enumerate(config.materials, start=2):
        label_axis = figure.add_subplot(grid[row_index, 1])
        label_axis.axis("off")
        label_axis.text(
            0.5,
            0.5,
            config.material_labels[material],
            rotation=90,
            fontsize=DEFAULT_STYLE.group_header_font_size_pt,
            fontweight="bold",
            fontfamily=STRUCTURAL_FONT_FAMILY,
            ha="center",
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
        axis = figure.add_subplot(
            grid[2 + condition_index // 2, 2 + condition_index % 2]
        )
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
        axis.set_xlim(0.85, config.maximum_calibration_contacts + 0.15)
        axis.set_ylim(y_min, 102.0)
        axis.set_xticks(range(1, config.maximum_calibration_contacts + 1))
        if condition_index // 2 == 1:
            axis.set_xlabel(
                "Contacts / location",
                fontsize=DEFAULT_STYLE.axis_label_font_size_pt,
            )
        _style_axis(axis)
        if condition_index // 2 == 0:
            axis.tick_params(axis="x", labelbottom=False)
        if condition_index % 2 == 1:
            axis.tick_params(axis="y", labelleft=False)
        axes.append(axis)
    return tuple(axes)


def _load_inputs(
    config: PaperFigureConfig,
    *,
    recompute: bool,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    """Load the fixed Figure 6 summaries without changing their metrics."""

    output = config.analysis_output_directory
    required = (
        "fig6b_spatial_distinguishability.csv",
        "fig6c_optional_calibration_10mm_combined.csv",
    )
    if recompute or any(not (output / name).is_file() for name in required):
        run_analysis(config)

    distinguishability = read_csv(output / required[0])
    optional_calibration = read_csv(output / required[1])
    variability = _load_panel_a_variability(config, distinguishability)
    return variability, distinguishability, optional_calibration


def build_figure(
    config: PaperFigureConfig,
    variability: list[dict[str, object]],
    distinguishability: list[dict[str, str]],
    optional_calibration: list[dict[str, str]],
    force_timeseries: Any,
    force_summary: dict[str, object],
) -> plt.Figure:
    """Build the canonical 2-by-3 Figure 6 at IEEE double-column width."""

    figure = plt.figure(figsize=FIGURE_SIZE_IN)
    grid = figure.add_gridspec(
        2,
        3,
        left=0.072,
        right=0.992,
        bottom=0.075,
        top=0.855,
        height_ratios=(0.72, 1.08),
        hspace=0.58,
        wspace=0.32,
    )

    _plot_variability_compact(figure, grid[0, 0], variability, config)
    _plot_distinguishability(
        figure.add_subplot(grid[0, 1]),
        distinguishability,
        config,
        show_title=False,
    )
    _plot_optional_calibration(
        figure.add_subplot(grid[0, 2]),
        optional_calibration,
        config,
        show_title=False,
        show_legend=False,
    )
    panel_d_spec = grid[1, 1]
    plot_force_timeseries(
        figure.add_subplot(panel_d_spec),
        force_timeseries,
        force_summary,
        show_title=False,
        compact=True,
    )
    panel_d_bounds = panel_d_spec.get_position(figure)
    figure.text(
        0.5 * (panel_d_bounds.x0 + panel_d_bounds.x1),
        panel_d_bounds.y1 + 0.021,
        FORCE_EXPERIMENT_LABEL,
        fontsize=DEFAULT_STYLE.minimum_font_size_pt,
        color="#555555",
        ha="center",
        va="bottom",
    )

    panel_titles = (
        (grid[0, 0], "(a)", "Contact-state variability", 0.012),
        (grid[0, 1], "(b)", "Re-contact distinguishability", 0.012),
        (grid[0, 2], "(c)", "Optional calibration", 0.012),
        (
            panel_d_spec,
            "(d)",
            "Location-aware force estimation",
            0.040,
        ),
    )
    for subplot_spec, panel_label, title, y_offset in panel_titles:
        _draw_panel_title(
            figure,
            subplot_spec,
            panel_label,
            title,
            y_offset=y_offset,
            label_x_offset=-0.015 if panel_label == "(d)" else 0.0,
        )

    figure.legend(
        handles=_optional_calibration_legend_handles(config),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=DEFAULT_STYLE.minimum_font_size_pt,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    return figure


def save_optional_calibration_panel(
    config: PaperFigureConfig,
    rows: list[dict[str, str]],
) -> tuple[Path, ...]:
    """Write the standalone Figure 6(c) PNG and PDF."""

    figure, axis = plt.subplots(figsize=(4.8, 3.15))
    figure.subplots_adjust(left=0.13, right=0.985, bottom=0.20, top=0.73)
    _plot_optional_calibration(
        axis,
        rows,
        config,
        show_title=True,
        show_legend=True,
    )
    outputs = save_figure(
        figure,
        OPTIONAL_CALIBRATION_OUTPUT_STEM,
        formats=("pdf", "png"),
        bbox_inches=None,
        pad_inches=0.0,
    )
    plt.close(figure)
    return outputs


def save_final(
    config: PaperFigureConfig, *, recompute: bool = False
) -> tuple[Path, ...]:
    """Write the standalone panel and canonical Figure 6 PDF/PNG outputs."""

    inputs = _load_inputs(config, recompute=recompute)
    DEFAULT_OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    with publication_context(DEFAULT_STYLE):
        panel_outputs = save_optional_calibration_panel(config, inputs[2])
        force_timeseries, force_summary = build_force_timeseries(
            DEFAULT_FORCE_DATASET_ROOT
        )
        force_csv = write_force_timeseries_csv(
            force_timeseries, FORCE_TIMESERIES_OUTPUT_STEM.with_suffix(".csv")
        )
        force_outputs = save_force_timeseries_panel(
            force_timeseries,
            force_summary,
            FORCE_TIMESERIES_OUTPUT_STEM,
        )
        figure = build_figure(config, *inputs, force_timeseries, force_summary)
        figure_outputs = save_figure(
            figure,
            DEFAULT_OUTPUT_STEM,
            formats=("pdf", "png"),
            bbox_inches=None,
            pad_inches=0.0,
        )
        plt.close(figure)
    return (*panel_outputs, force_csv, *force_outputs, *figure_outputs)


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
