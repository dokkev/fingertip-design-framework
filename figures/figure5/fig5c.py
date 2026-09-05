"""Figure 5(c): neighboring-contact load-response separation."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.gridspec import SubplotSpec  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from lumo.visualization import DEFAULT_STYLE, publication_context, save_figure  # noqa: E402

from .config import (  # noqa: E402
    ANALYSIS_ROOTS,
    COMPARISON_CONDITIONS,
    COMPARISON_MORPHOLOGIES,
    COMPARISON_TITLES,
    FIGURE_DIRECTORY,
)


MORPHOLOGY_COLORS = {
    "baseline": "#B8BCC2",
    "flat_opt": "#4F7180",
    "angled_opt": "#A87446",
}


def load_spatial_metrics() -> list[dict[str, object]]:
    """Read stored slope-profile separation and add baseline improvements."""

    source_rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for expected_material, root in ANALYSIS_ROOTS.items():
        path = root / "results" / "morphology_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing required {expected_material} analysis artifact: {path}"
            )
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["material"] != expected_material:
                    continue
                key = (row["material"], row["morphology"], row["indenter"])
                if key in source_rows:
                    raise RuntimeError(f"duplicate Figure 5(c) metric row: {key}")
                source_rows[key] = row

    output: list[dict[str, object]] = []
    for material, indenter, _ in COMPARISON_CONDITIONS:
        baseline_key = (material, "baseline", indenter)
        if baseline_key not in source_rows:
            raise RuntimeError(f"missing Figure 5(c) baseline metric: {baseline_key}")
        baseline_value = float(
            source_rows[baseline_key]["D_neighbor_median_DN_per_N"]
        )
        if not np.isfinite(baseline_value) or baseline_value <= 0.0:
            raise ValueError(f"invalid Figure 5(c) baseline: {baseline_key}")

        for morphology in COMPARISON_MORPHOLOGIES:
            key = (material, morphology, indenter)
            source = source_rows.get(key)
            if source is None:
                if material == "dragon_skin" and morphology == "angled_opt":
                    output.append(
                        {
                            "material": material,
                            "morphology": morphology,
                            "indenter": indenter,
                            "D_neighbor_median_DN_per_N": float("nan"),
                            "D_neighbor_IQR_DN_per_N": float("nan"),
                            "baseline_D_neighbor_DN_per_N": baseline_value,
                            "improvement_percent": float("nan"),
                            "status": "pending",
                        }
                    )
                    continue
                raise RuntimeError(f"missing required Figure 5(c) metric: {key}")

            separation = float(source["D_neighbor_median_DN_per_N"])
            iqr = float(source["D_neighbor_IQR_DN_per_N"])
            if not np.isfinite(separation) or separation <= 0.0:
                raise ValueError(f"invalid Figure 5(c) separation: {key}")
            if not np.isfinite(iqr) or iqr < 0.0:
                raise ValueError(f"invalid Figure 5(c) separation IQR: {key}")
            output.append(
                {
                    "material": material,
                    "morphology": morphology,
                    "indenter": indenter,
                    "D_neighbor_median_DN_per_N": separation,
                    "D_neighbor_IQR_DN_per_N": iqr,
                    "baseline_D_neighbor_DN_per_N": baseline_value,
                    "improvement_percent": 100.0
                    * (separation / baseline_value - 1.0),
                    "status": "measured",
                }
            )
    return output


def write_metrics(
    rows: list[dict[str, object]],
    path: Path = FIGURE_DIRECTORY / "fig5c_metrics.csv",
) -> Path:
    """Persist the exact stored bar metric and derived baseline comparison."""

    fields = (
        "material",
        "morphology",
        "indenter",
        "D_neighbor_median_DN_per_N",
        "D_neighbor_IQR_DN_per_N",
        "baseline_D_neighbor_DN_per_N",
        "improvement_percent",
        "status",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _print_metrics(rows: list[dict[str, object]]) -> None:
    print(
        f"{'material':<13} {'indenter':<13} {'morphology':<12} "
        f"{'D_neighbor [DN/N]':>18} {'improvement [%]':>17}"
    )
    for row in rows:
        if row["status"] == "pending":
            separation_text = "pending"
            improvement_text = "pending"
        else:
            separation_text = f"{float(row['D_neighbor_median_DN_per_N']):.4f}"
            improvement_text = f"{float(row['improvement_percent']):+.1f}"
        material = str(row["material"]).replace("dragon_skin", "Dragon Skin")
        print(
            f"{material:<13} {str(row['indenter']):<13} "
            f"{str(row['morphology']):<12} {separation_text:>18} "
            f"{improvement_text:>17}"
        )


def render_panel(
    figure: Figure,
    subplot_spec: SubplotSpec,
    *,
    panel_label: str = "(c)",
    metrics: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Render absolute slope-profile separation with relative annotations."""

    rows = load_spatial_metrics() if metrics is None else metrics
    write_metrics(rows)
    lookup = {
        (
            str(row["material"]),
            str(row["indenter"]),
            str(row["morphology"]),
        ): row
        for row in rows
    }

    grid = subplot_spec.subgridspec(
        3,
        1,
        height_ratios=(0.11, 0.075, 1.0),
        hspace=0.01,
    )
    title_axis = figure.add_subplot(grid[0, 0])
    title_axis.axis("off")
    title_axis.text(
        0.0,
        0.55,
        panel_label,
        fontsize=DEFAULT_STYLE.panel_label_font_size_pt,
        fontweight="bold",
        va="center",
    )
    title_axis.text(
        0.21,
        0.55,
        "Spatial separation across\nhardware conditions",
        fontsize=5.7,
        fontweight="bold",
        va="center",
        linespacing=0.95,
    )

    legend_axis = figure.add_subplot(grid[1, 0])
    legend_axis.axis("off")
    legend_axis.legend(
        handles=[
            Patch(
                facecolor=MORPHOLOGY_COLORS[morphology],
                edgecolor="#4A4A4A",
                linewidth=0.35,
                label=title,
            )
            for morphology, title in zip(
                COMPARISON_MORPHOLOGIES, COMPARISON_TITLES, strict=True
            )
        ],
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=3.9,
        handlelength=0.75,
        handleheight=0.65,
        handletextpad=0.25,
        columnspacing=0.45,
        borderaxespad=0.0,
    )

    axis = figure.add_subplot(grid[2, 0])
    group_centers = np.arange(len(COMPARISON_CONDITIONS), dtype=np.float64)
    offsets = np.asarray((-0.24, 0.0, 0.24), dtype=np.float64)
    bar_width = 0.205
    measured_values = [
        float(row["D_neighbor_median_DN_per_N"])
        for row in rows
        if row["status"] == "measured"
    ]
    y_max = max(measured_values) * 1.22

    for group_index, (material, indenter, _) in enumerate(
        COMPARISON_CONDITIONS
    ):
        for morphology_index, morphology in enumerate(COMPARISON_MORPHOLOGIES):
            row = lookup[(material, indenter, morphology)]
            x = group_centers[group_index] + offsets[morphology_index]
            if row["status"] == "pending":
                axis.text(
                    x,
                    0.025 * y_max,
                    "pending",
                    fontsize=3.9,
                    color="#858585",
                    ha="center",
                    va="bottom",
                    rotation=90,
                )
                continue

            value = float(row["D_neighbor_median_DN_per_N"])
            axis.bar(
                x,
                value,
                width=bar_width,
                color=MORPHOLOGY_COLORS[morphology],
                edgecolor="#4A4A4A",
                linewidth=0.35,
                zorder=2,
            )
            if morphology != "baseline":
                improvement = float(row["improvement_percent"])
                axis.text(
                    x,
                    value + 0.025 * y_max,
                    f"{improvement:+.0f}%",
                    fontsize=4.2,
                    color="#303030",
                    ha="center",
                    va="bottom",
                )

    axis.set_xlim(-0.52, len(COMPARISON_CONDITIONS) - 0.48)
    axis.set_ylim(0.0, y_max)
    axis.set_xticks(group_centers)
    axis.set_xticklabels(
        (
            "Solaris\n10 mm",
            "Solaris\n30 mm",
            "Dragon Skin\n10 mm",
            "Dragon Skin\n30 mm",
        ),
        fontsize=3.95,
        linespacing=0.95,
    )
    axis.text(
        0.015,
        0.52,
        "Spatial separation [camera DN/N]",
        transform=axis.transAxes,
        fontsize=4.8,
        rotation=90,
        ha="left",
        va="center",
    )
    axis.tick_params(axis="x", length=0.0, pad=2.0)
    axis.tick_params(axis="y", labelsize=4.5, length=2.0, pad=1.2)
    axis.grid(axis="y", color="#E2E2E2", linewidth=0.4, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#777777")
    axis.spines["bottom"].set_color("#777777")
    axis.spines["left"].set_linewidth(0.5)
    axis.spines["bottom"].set_linewidth(0.5)

    return {
        "axes": (axis,),
        "metrics": rows,
        "y_limit": y_max,
    }


def main() -> None:
    """Export a standalone debug render and print its source metrics."""

    rows = load_spatial_metrics()
    _print_metrics(rows)
    with publication_context(DEFAULT_STYLE):
        figure = plt.figure(figsize=(3.15, 4.25))
        grid = figure.add_gridspec(
            1, 1, left=0.03, right=0.985, bottom=0.025, top=0.99
        )
        render_panel(figure, grid[0, 0], metrics=rows)
        save_figure(
            figure,
            FIGURE_DIRECTORY / "fig5c",
            formats=("png",),
            bbox_inches=None,
            pad_inches=0.0,
        )
        plt.close(figure)


if __name__ == "__main__":
    main()
