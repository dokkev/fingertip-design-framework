"""Alternate IEEE single-column Figure 6 candidate.

Relative to the production ``fig6.py`` composition, this candidate replaces
panel (a) with a direct baseline-relative comparison of maintained-contact
``W_cycle`` and independent re-contact ``W_recontact``. ``W_cycle`` is loaded
only from the explicitly separate 10 mm contact-history summaries;
``W_recontact`` and panels (b)--(d) retain the frozen production Figure 6
support tables. The two absolute metrics have different units and are never
pooled; panel (a) compares only each metric's percent change from its matched
material baseline.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

from experiments.analysis.fig5c_decoder import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    PaperFigureConfig,
    load_config,
    read_csv,
)
from lumo.visualization import DEFAULT_STYLE, EDGE_COLOR, save_figure  # noqa: E402

from . import fig6 as production  # noqa: E402


FIGURE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_OUTPUT_STEM = FIGURE_DIRECTORY / "fig6_singlecol_alt"
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
PANEL_A_MORPHOLOGIES = ("flat_opt", "angled_opt")
ALT_STYLE = replace(
    DEFAULT_STYLE,
    base_font_size_pt=7.0,
    tick_font_size_pt=6.0,
    axis_label_font_size_pt=7.0,
    legend_font_size_pt=6.0,
)


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
    cycle_summaries: dict[str, Path],
) -> list[dict[str, object]]:
    """Join separated 10 mm W_cycle and production W_recontact summaries."""

    recontact = production._lookup(recontact_rows)
    output = []
    for material in config.materials:
        cycle = _cycle_lookup(cycle_summaries[material])
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
                    "material_display": config.material_labels[material],
                    "indenter": "sphere_10mm",
                    "indenter_display": config.indenter_labels["sphere_10mm"],
                    "morphology": morphology,
                    "morphology_display": config.morphology_labels[morphology],
                    "W_cycle_baseline_DN": cycle_baseline,
                    "W_cycle_variant_DN": cycle_value,
                    "W_cycle_baseline_run_count": int(
                        cycle["baseline"]["run_count"]
                    ),
                    "W_cycle_variant_run_count": int(cycle[morphology]["run_count"]),
                    "cycle_variability_change_percent": 100.0
                    * (cycle_value - cycle_baseline)
                    / cycle_baseline,
                    "W_cycle_source": str(cycle_summaries[material]),
                    "W_recontact_baseline_DN_per_N": recontact_baseline,
                    "W_recontact_variant_DN_per_N": recontact_value,
                    "recontact_variability_change_percent": 100.0
                    * (recontact_value - recontact_baseline)
                    / recontact_baseline,
                    "W_recontact_table": str(
                        config.analysis_output_directory
                        / "fig6b_spatial_distinguishability.csv"
                    ),
                    "W_recontact_source": recontact_row["source_path"],
                }
            )
    return output


def _write_panel_a_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_contact_state_variability(
    figure: plt.Figure,
    subplot_spec: object,
    rows: list[dict[str, object]],
    config: PaperFigureConfig,
) -> None:
    grid = subplot_spec.subgridspec(
        3,
        2,
        height_ratios=(0.13, 0.20, 1.0),
        hspace=0.04,
        wspace=0.10,
    )
    title_axis = figure.add_subplot(grid[0, :2])
    title_axis.axis("off")
    title_axis.text(
        0.0,
        0.96,
        r"$\mathbf{(a)}$  Contact-state variability",
        fontsize=7.2,
        fontweight="normal",
        ha="left",
        va="top",
    )
    title_axis.text(
        1.0,
        0.96,
        "10 mm sphere",
        fontsize=4.5,
        color="#555555",
        ha="right",
        va="top",
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
    lookup = {
        (str(row["material"]), str(row["morphology"])): row for row in rows
    }
    metric_columns = (
        ("cycle_variability_change_percent", "Maintained contact", r"$W_{\mathrm{cycle}}$"),
        (
            "recontact_variability_change_percent",
            "Re-contact",
            r"$W_{\mathrm{recontact}}$",
        ),
    )
    centers = np.arange(len(config.materials), dtype=float)
    offsets = {"flat_opt": -0.08, "angled_opt": 0.08}
    for column, (value_key, header, notation) in enumerate(metric_columns):
        header_axis = figure.add_subplot(grid[1, column])
        header_axis.axis("off")
        header_axis.text(
            0.5,
            0.56,
            f"{header}\n{notation}",
            fontsize=5.8,
            fontweight="normal",
            ha="center",
            va="center",
            linespacing=1.05,
        )
        axis = figure.add_subplot(grid[2, column])
        for material_index, material in enumerate(config.materials):
            for morphology in PANEL_A_MORPHOLOGIES:
                row = lookup[(material, morphology)]
                axis.scatter(
                    centers[material_index] + offsets[morphology],
                    float(row[value_key]),
                    s=24,
                    color=config.morphology_colors[morphology],
                    edgecolor=EDGE_COLOR,
                    linewidth=0.45,
                    zorder=3,
                )
        axis.axhline(0.0, color="#777777", linewidth=0.65, linestyle="--", zorder=1)
        axis.set_xlim(-0.32, len(config.materials) - 0.68)
        axis.set_ylim(-limit, limit)
        axis.set_xticks(
            centers,
            [config.material_labels[material] for material in config.materials],
        )
        production._style_axis(axis)
        axis.grid(axis="y", color="#E3E3E3", linewidth=0.4, zorder=0)
        axis.tick_params(labelsize=5.0, pad=1.0)
        if column == 0:
            axis.set_ylabel("Variability change [%]", fontsize=5.7, labelpad=2.0)
        else:
            axis.tick_params(axis="y", labelleft=False)


def build_figure(
    config: PaperFigureConfig,
    panel_a_rows: list[dict[str, object]],
    distinguishability: list[dict[str, str]],
    scalar_spatial: list[dict[str, str]],
    calibration: list[dict[str, str]],
) -> plt.Figure:
    """Build the alternate figure at true IEEE single-column width."""

    figure = plt.figure(figsize=(ALT_STYLE.single_column_width_in, 6.58))
    grid = figure.add_gridspec(
        4,
        1,
        left=0.19,
        right=0.985,
        bottom=0.040,
        top=0.955,
        height_ratios=(0.96, 1.0, 1.0, 1.44),
        hspace=0.35,
    )
    _plot_contact_state_variability(figure, grid[0], panel_a_rows, config)
    axis_b = figure.add_subplot(grid[1])
    production._plot_distinguishability(axis_b, distinguishability, config)
    axis_b.set_title(
        r"$\mathbf{(b)}$  Re-contact distinguishability",
        loc="left",
        fontsize=7.2,
        fontweight="normal",
        pad=2.0,
    )
    axis_c = figure.add_subplot(grid[2])
    production._plot_scalar_spatial(axis_c, scalar_spatial, config)
    axis_c.set_title(
        r"$\mathbf{(c)}$  Spatial vs. scalar decoding",
        loc="left",
        fontsize=7.2,
        fontweight="normal",
        pad=2.0,
    )
    axes_before_d = set(figure.axes)
    production._plot_calibration(figure, grid[3], calibration, config)
    for axis in set(figure.axes) - axes_before_d:
        for text in axis.texts:
            if text.get_text().startswith("(d)"):
                text.set_text(r"$\mathbf{(d)}$  Calibration-set size")
                text.set_fontsize(7.2)
                text.set_fontweight("normal")
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
        fontsize=6.0,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    return figure


def save_alternative(
    config: PaperFigureConfig,
    *,
    output_stem: Path = DEFAULT_OUTPUT_STEM,
) -> tuple[Path, ...]:
    """Render the alternate figure without changing production Figure 6."""

    analysis = config.analysis_output_directory
    distinguishability = read_csv(
        analysis / "fig6b_spatial_distinguishability.csv"
    )
    scalar_spatial = read_csv(analysis / "fig6c_scalar_vs_spatial.csv")
    calibration = read_csv(analysis / "fig6d_calibration_burden.csv")
    panel_a_rows = _load_panel_a_variability(
        config, distinguishability, DEFAULT_CYCLE_SUMMARIES
    )
    _write_panel_a_csv(
        output_stem.with_name(f"{output_stem.name}_panel_a_variability.csv"),
        panel_a_rows,
    )

    rc_params = ALT_STYLE.rc_params()
    rc_params.update(
        {
            # Use the installed regular face explicitly.  The repository also
            # carries Helvetica Light, which Matplotlib may otherwise select
            # even when ``font.weight`` is set to ``normal``.
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
            "font.cursive": ["DejaVu Sans"],
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "mathtext.rm": "DejaVu Sans",
            "mathtext.it": "DejaVu Sans:italic",
            "mathtext.bf": "DejaVu Sans:bold",
            "mathtext.sf": "DejaVu Sans",
            "mathtext.cal": "DejaVu Sans",
        }
    )
    with matplotlib.rc_context(rc=rc_params):
        figure = build_figure(
            config,
            panel_a_rows,
            distinguishability,
            scalar_spatial,
            calibration,
        )
        outputs = save_figure(
            figure,
            output_stem,
            formats=("pdf", "png"),
            style=ALT_STYLE,
            bbox_inches=None,
            pad_inches=0.0,
        )
        plt.close(figure)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    arguments = parser.parse_args()
    for path in save_alternative(
        load_config(arguments.config),
        output_stem=arguments.output_stem,
    ):
        print(path)


if __name__ == "__main__":
    main()
