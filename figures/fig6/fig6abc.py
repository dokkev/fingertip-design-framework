"""Build corrected Figure 6(a--c) without touching panels (d) or (e)."""

from __future__ import annotations

import argparse
import csv
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
)
from experiments.analysis.fig6abc import (  # noqa: E402
    OUTPUT_SUBDIRECTORY,
    cached_artifacts_current,
    run_analysis,
)
from lumo.visualization import (  # noqa: E402
    DEFAULT_STYLE,
    EDGE_COLOR,
    STRUCTURAL_FONT_FAMILY,
    publication_context,
    save_figure,
)


FIGURE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_OUTPUT_STEM = FIGURE_DIRECTORY / "fig6abc_review"
FIGURE_SIZE_IN = (DEFAULT_STYLE.double_column_width_in, 2.62)
INDENTER_MARKERS = {"sphere_10mm": "o", "sphere_30mm": "s"}
PANEL_C_TABLE = FIGURE_DIRECTORY / "fig6c_uncalibrated_location_mae.csv"
PANEL_C_REFERENCE = ("solaris", "angled_opt")
PANEL_C_MINIMUM_FIT_LOCATIONS = 2
# Shared x-label drop for panels that own nested subplots, in inches so the
# gap stays constant across the review render and the taller full Figure 6.
SHARED_X_LABEL_OFFSET_IN = 0.21


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#777777")
    axis.spines["bottom"].set_color("#777777")
    axis.grid(axis="y", color="#E4E4E4", linewidth=0.45, zorder=0)
    axis.tick_params(
        labelsize=DEFAULT_STYLE.tick_font_size_pt,
        length=DEFAULT_STYLE.tick_length_pt,
        width=DEFAULT_STYLE.tick_width_pt,
        pad=1.5,
    )


def _panel_title(figure: plt.Figure, spec: Any, label: str, title: str) -> None:
    bounds = spec.get_position(figure)
    y = bounds.y1 + 0.025
    figure.text(
        bounds.x0,
        y,
        label,
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


def _plot_panel_a(
    figure: plt.Figure,
    spec: Any,
    rows: list[dict[str, str]],
    config: PaperFigureConfig,
) -> None:
    grid = spec.subgridspec(1, 2, wspace=0.18)
    measured = [row for row in rows if row["status"] == "measured"]
    x_values = np.asarray([float(row["W_cycle_DN"]) for row in measured])
    y_values = np.asarray([float(row["W_recontact_DN_per_N"]) for row in measured])
    x_upper = 0.05 * np.ceil(1.12 * np.max(x_values) / 0.05)
    y_upper = 0.01 * np.ceil(1.15 * np.max(y_values) / 0.01)
    for column, material in enumerate(config.materials):
        axis = figure.add_subplot(grid[0, column])
        for row in rows:
            if row["material"] != material:
                continue
            if row["status"] != "measured":
                continue
            morphology = row["morphology_id"]
            axis.scatter(
                float(row["W_cycle_DN"]),
                float(row["W_recontact_DN_per_N"]),
                s=27,
                marker=INDENTER_MARKERS[row["indenter"]],
                color=config.morphology_colors[morphology],
                edgecolor=EDGE_COLOR,
                linewidth=0.45,
                zorder=3,
            )
        axis.set_xlim(0.0, x_upper)
        axis.set_ylim(0.0, y_upper)
        axis.text(
            0.5,
            0.985,
            f"{config.material_labels[material]} · 10 mm",
            transform=axis.transAxes,
            fontsize=DEFAULT_STYLE.condition_header_font_size_pt,
            color="#555555",
            ha="center",
            va="bottom",
        )
        _style_axis(axis)
        if column:
            axis.tick_params(axis="y", left=False, labelleft=False)
        else:
            axis.set_ylabel(r"$W_{\mathrm{recontact}}$ [DN N$^{-1}$]", labelpad=1.5)
    bounds = spec.get_position(figure)
    figure.text(
        0.5 * (bounds.x0 + bounds.x1),
        bounds.y0 - SHARED_X_LABEL_OFFSET_IN / figure.get_figheight(),
        r"$W_{\mathrm{cycle}}$ [DN]",
        fontsize=DEFAULT_STYLE.axis_label_font_size_pt,
        ha="center",
        va="top",
    )


def _condition_labels(config: PaperFigureConfig) -> list[str]:
    return [
        config.indenter_labels[indenter].replace(" sphere", "")
        for _material in config.materials
        for indenter in config.indenters
    ]


def _plot_panel_b(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    config: PaperFigureConfig,
) -> None:
    lookup = {
        (row["material"], row["indenter"], row["morphology_id"]): row
        for row in rows
    }
    centers = np.arange(len(config.materials) * len(config.indenters))
    offsets = np.linspace(-0.23, 0.23, len(config.morphologies))
    values = []
    condition = 0
    for material in config.materials:
        for indenter in config.indenters:
            for index, morphology in enumerate(config.morphologies):
                row = lookup[(material, indenter, morphology)]
                x = centers[condition] + offsets[index]
                if row["status"] != "measured":
                    axis.text(
                        x,
                        0.05,
                        "n/a",
                        rotation=90,
                        fontsize=DEFAULT_STYLE.minimum_font_size_pt,
                        ha="center",
                        va="bottom",
                    )
                    continue
                value = float(row["Q_recontact"])
                values.append(value)
                axis.bar(
                    x,
                    value,
                    width=0.20,
                    color=config.morphology_colors[morphology],
                    edgecolor=EDGE_COLOR,
                    linewidth=0.4,
                    zorder=2,
                )
            condition += 1
    axis.set_ylim(0.0, 1.12 * max(values))
    axis.set_xticks(centers, _condition_labels(config))
    axis.set_ylabel(r"$Q_{\mathrm{recontact}}$", labelpad=1.5)
    for index, material in enumerate(config.materials):
        center = index * len(config.indenters) + 0.5
        axis.text(
            center,
            -0.16,
            config.material_labels[material],
            transform=axis.get_xaxis_transform(),
            fontsize=DEFAULT_STYLE.group_header_font_size_pt,
            ha="center",
            va="top",
            clip_on=False,
        )
    axis.axvline(1.5, color="#B8B8B8", linewidth=0.55)
    _style_axis(axis)
    axis.tick_params(axis="x", length=0.0)


def _material_line_style(material: str) -> str:
    return "-" if material == "solaris" else "--"


def _morphology_marker(morphology: str) -> str:
    return {"baseline": "o", "flat_opt": "s", "angled_opt": "D"}[morphology]


def _legend_handles(config: PaperFigureConfig) -> list[Line2D]:
    """Morphology identity plus the uncalibrated prior-only mark."""

    handles = [
        Line2D(
            [],
            [],
            color=config.morphology_colors[morphology],
            marker=_morphology_marker(morphology),
            markeredgecolor=EDGE_COLOR,
            markeredgewidth=0.4,
            linewidth=DEFAULT_STYLE.line_width_pt,
            markersize=4.0,
            label=config.morphology_labels[morphology],
        )
        for morphology in config.morphologies
    ]
    handles.append(
        Line2D(
            [],
            [],
            color="#8A8F94",
            linestyle="none",
            marker="X",
            markeredgecolor=EDGE_COLOR,
            markeredgewidth=0.4,
            markersize=5.2,
            label=r"Prior only ($K$ = 0)",
        )
    )
    return handles


def load_panel_c(path: Path = PANEL_C_TABLE) -> list[dict[str, str]]:
    """Read the checked-in uncalibrated-location MAE table for Figure 6(c)."""

    return _read_csv(path)


def _panel_c_series(
    rows: list[dict[str, str]],
    material: str,
    morphology: str,
) -> list[tuple[int, float, float | None, float | None]]:
    """Return measured (K, MAE, q25, q75) points ordered by calibration effort."""

    selected = [
        row
        for row in rows
        if row["material"] == material
        and row["morphology_id"] == morphology
        and row["status"] == "measured"
    ]
    series: list[tuple[int, float, float | None, float | None]] = []
    for row in sorted(selected, key=lambda item: int(item["calibration_locations"])):
        low = row["mae_q25_mm"].strip()
        high = row["mae_q75_mm"].strip()
        series.append(
            (
                int(row["calibration_locations"]),
                float(row["mae_mm_uncalibrated"]),
                float(low) if low else None,
                float(high) if high else None,
            )
        )
    return series


def _plot_panel_c(
    figure: plt.Figure,
    spec: Any,
    rows: list[dict[str, str]],
    config: PaperFigureConfig,
) -> None:
    """Draw MAE at contact locations absent from calibration, split by material.

    Calibration effort ``K`` is the number of distinct contact locations set up
    on the rig; every evaluated location is one that ``K`` never calibrated, so
    ``K = 6`` is undefined and ``K = 1`` cannot support the affine fit. ``K = 0``
    is the LED geometry prior alone and exists only where that prior resolves.
    """

    measured = [row for row in rows if row["status"] == "measured"]
    if not measured:
        raise RuntimeError(f"{PANEL_C_TABLE} contains no measured rows")
    budgets = sorted({int(row["calibration_locations"]) for row in measured})
    upper = float(np.ceil(1.06 * max(float(row["mae_mm_uncalibrated"]) for row in measured)))
    reference_material, reference_morphology = PANEL_C_REFERENCE
    reference = float(
        next(
            row["mae_mm_uncalibrated"]
            for row in measured
            if row["material"] == reference_material
            and row["morphology_id"] == reference_morphology
            and int(row["calibration_locations"]) == 0
        )
    )
    grid = spec.subgridspec(1, len(config.materials), wspace=0.10)
    shared: plt.Axes | None = None
    for column, material in enumerate(config.materials):
        axis = figure.add_subplot(grid[0, column], sharey=shared)
        shared = shared or axis
        axis.axhline(
            reference,
            color=config.morphology_colors[reference_morphology],
            linewidth=0.8,
            linestyle=(0, (4.0, 2.5)),
            zorder=1,
        )
        axis.axvline(
            1.0,
            color="#B8B8B8",
            linewidth=0.55,
            linestyle=(0, (1.6, 1.6)),
            zorder=1,
        )
        has_prior = False
        for morphology in config.morphologies:
            series = _panel_c_series(rows, material, morphology)
            color = config.morphology_colors[morphology]
            fitted = [
                point for point in series if point[0] >= PANEL_C_MINIMUM_FIT_LOCATIONS
            ]
            if fitted:
                x = [point[0] for point in fitted]
                axis.plot(
                    x,
                    [point[1] for point in fitted],
                    color=color,
                    marker=_morphology_marker(morphology),
                    linewidth=DEFAULT_STYLE.line_width_pt,
                    markersize=4.0,
                    markeredgecolor=EDGE_COLOR,
                    markeredgewidth=0.4,
                    zorder=3,
                )
                if all(point[2] is not None and point[3] is not None for point in fitted):
                    axis.fill_between(
                        x,
                        [point[2] for point in fitted],
                        [point[3] for point in fitted],
                        color=color,
                        alpha=0.14,
                        linewidth=0.0,
                        zorder=2,
                    )
            prior = [point for point in series if point[0] == 0]
            if prior:
                has_prior = True
                axis.plot(
                    [0],
                    [prior[0][1]],
                    linestyle="none",
                    marker="X",
                    markersize=5.6,
                    color=color,
                    markeredgecolor=EDGE_COLOR,
                    markeredgewidth=0.4,
                    zorder=4,
                )
        axis.set_xlim(budgets[0] - 0.6, budgets[-1] + 0.6)
        axis.set_ylim(0.0, upper)
        axis.set_xticks(budgets)
        _style_axis(axis)
        axis.text(
            0.5,
            0.99,
            config.material_labels[material],
            transform=axis.transAxes,
            fontsize=DEFAULT_STYLE.group_header_font_size_pt,
            fontfamily=STRUCTURAL_FONT_FAMILY,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
        if column:
            axis.tick_params(axis="y", left=False, labelleft=False)
        else:
            axis.set_ylabel("Localization MAE [mm]", labelpad=1.5)
            axis.text(
                0.5,
                reference - 0.03 * upper,
                f"{config.material_labels[reference_material]} "
                f"{config.morphology_labels[reference_morphology]} prior",
                fontsize=DEFAULT_STYLE.annotation_font_size_pt,
                color=config.morphology_colors[reference_morphology],
                ha="left",
                va="top",
            )
        if not has_prior:
            axis.text(
                0.0,
                0.90 * upper,
                "No prior\navailable",
                fontsize=DEFAULT_STYLE.annotation_font_size_pt,
                color="#666666",
                ha="center",
                va="top",
                linespacing=1.1,
            )
    bounds = spec.get_position(figure)
    figure.text(
        0.5 * (bounds.x0 + bounds.x1),
        bounds.y0 - SHARED_X_LABEL_OFFSET_IN / figure.get_figheight(),
        r"Calibration locations $K$",
        fontsize=DEFAULT_STYLE.axis_label_font_size_pt,
        ha="center",
        va="top",
    )


def build_review(
    config: PaperFigureConfig,
    panel_a: list[dict[str, str]],
    panel_b: list[dict[str, str]],
    panel_c: list[dict[str, str]],
) -> plt.Figure:
    figure = plt.figure(figsize=FIGURE_SIZE_IN)
    grid = figure.add_gridspec(
        1,
        3,
        left=0.070,
        right=0.992,
        bottom=0.235,
        top=0.760,
        width_ratios=(1.04, 1.10, 1.46),
        wspace=0.30,
    )
    _plot_panel_a(figure, grid[0, 0], panel_a, config)
    _plot_panel_b(figure.add_subplot(grid[0, 1]), panel_b, config)
    _plot_panel_c(figure, grid[0, 2], panel_c, config)
    _panel_title(figure, grid[0, 0], "(a)", "Contact-state variability")
    _panel_title(figure, grid[0, 1], "(b)", "Re-contact distinguishability")
    _panel_title(figure, grid[0, 2], "(c)", "Transfer to uncalibrated locations")
    figure.legend(
        handles=_legend_handles(config),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=DEFAULT_STYLE.legend_font_size_pt,
        columnspacing=1.1,
        handletextpad=0.35,
    )
    return figure


def save_review(
    config: PaperFigureConfig,
    config_path: Path,
    *,
    recompute: bool,
) -> tuple[Path, ...]:
    output = config.analysis_output_directory / OUTPUT_SUBDIRECTORY
    if recompute or not cached_artifacts_current(config, config_path):
        run_analysis(config, config_path)
    a_rows = _read_csv(output / "fig6a_contact_state_variability.csv")
    b_rows = _read_csv(output / "fig6b_distinguishability_values.csv")
    c_rows = load_panel_c()
    with publication_context(DEFAULT_STYLE):
        figure = build_review(config, a_rows, b_rows, c_rows)
        outputs = save_figure(
            figure,
            DEFAULT_OUTPUT_STEM,
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
    parser.add_argument("--analysis-only", action="store_true")
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    config = load_config(config_path)
    if arguments.analysis_only:
        paths = run_analysis(config, config_path)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return
    for path in save_review(config, config_path, recompute=arguments.recompute):
        print(path)


if __name__ == "__main__":
    main()
