"""Figure 5(c): baseline-relative transfer across hardware conditions."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.gridspec import SubplotSpec  # noqa: E402

from lumo.visualization import DEFAULT_STYLE, publication_context, save_figure  # noqa: E402

from .config import (  # noqa: E402
    ANALYSIS_ROOTS,
    COMPARISON_CONDITIONS,
    COMPARISON_MORPHOLOGIES,
    COMPARISON_TITLES,
    FIGURE_DIRECTORY,
    require_available_inputs,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_transfer_metrics() -> list[dict[str, object]]:
    """Read D, W, R and normalize R by each material/indenter baseline."""

    require_available_inputs()
    source_rows = []
    for material, root in ANALYSIS_ROOTS.items():
        for row in _read_csv(root / "results" / "morphology_metrics.csv"):
            if row["material"] == material:
                source_rows.append(row)

    by_identity = {
        (row["material"], row["morphology"], row["indenter"]): row
        for row in source_rows
    }
    output: list[dict[str, object]] = []
    for material, indenter, _ in COMPARISON_CONDITIONS:
        baseline_key = (material, "baseline", indenter)
        if baseline_key not in by_identity:
            raise RuntimeError(f"missing Figure 5(c) baseline metric: {baseline_key}")
        baseline_r = float(by_identity[baseline_key]["D_neighbor_over_W"])
        if not np.isfinite(baseline_r) or baseline_r <= 0.0:
            raise ValueError(f"invalid baseline R_obs for {baseline_key}")

        for morphology in COMPARISON_MORPHOLOGIES:
            key = (material, morphology, indenter)
            row = by_identity.get(key)
            if row is None:
                if material == "dragon_skin" and morphology == "angled_opt":
                    output.append(
                        {
                            "material": material,
                            "morphology": morphology,
                            "indenter": indenter,
                            "D_neighbor": float("nan"),
                            "W": float("nan"),
                            "R_obs": float("nan"),
                            "G_obs": float("nan"),
                            "status": "pending",
                        }
                    )
                    continue
                raise RuntimeError(f"missing required Figure 5(c) metric: {key}")

            d_neighbor = float(row["D_neighbor_median_DN_per_N"])
            variation = float(row["W_median_DN_per_N"])
            ratio = float(row["D_neighbor_over_W"])
            if not np.all(np.isfinite((d_neighbor, variation, ratio))):
                raise ValueError(f"non-finite Figure 5(c) metric: {key}")
            if not np.isclose(ratio, d_neighbor / variation, rtol=1.0e-10, atol=1.0e-12):
                raise ValueError(f"stored R_obs disagrees with D_neighbor/W: {key}")
            output.append(
                {
                    "material": material,
                    "morphology": morphology,
                    "indenter": indenter,
                    "D_neighbor": d_neighbor,
                    "W": variation,
                    "R_obs": ratio,
                    "G_obs": ratio / baseline_r,
                    "status": "measured",
                }
            )
    return output


def write_metrics(
    rows: list[dict[str, object]],
    path: Path = FIGURE_DIRECTORY / "fig5c_metrics.csv",
) -> Path:
    """Persist measured and explicitly pending transfer entries."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "material",
        "morphology",
        "indenter",
        "D_neighbor",
        "W",
        "R_obs",
        "G_obs",
        "status",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _matrix(rows: list[dict[str, object]]) -> np.ndarray:
    lookup = {
        (str(row["material"]), str(row["indenter"]), str(row["morphology"])): float(
            row["G_obs"]
        )
        for row in rows
    }
    return np.asarray(
        [
            [
                lookup[(material, indenter, morphology)]
                for morphology in COMPARISON_MORPHOLOGIES[1:]
            ]
            for material, indenter, _ in COMPARISON_CONDITIONS
        ],
        dtype=np.float64,
    )


def render_panel(
    figure: Figure,
    subplot_spec: SubplotSpec,
    *,
    panel_label: str = "(c)",
    metrics: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Render the compact 4 x 2 baseline-normalized transfer matrix."""

    rows = load_transfer_metrics() if metrics is None else metrics
    write_metrics(rows)
    values = _matrix(rows)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Figure 5(c) has no measured transfer values")
    lower = min(float(np.min(finite)), 0.75)
    upper = max(float(np.max(finite)), 1.25)
    normalization = TwoSlopeNorm(vmin=lower, vcenter=1.0, vmax=upper)
    colormap = LinearSegmentedColormap.from_list(
        "light_transfer", ("#D9B8B4", "#F4F4F1", "#91B7AA")
    )

    grid = subplot_spec.subgridspec(
        8,
        3,
        height_ratios=(0.34, 0.18, 0.22, 1.0, 1.0, 0.10, 1.0, 1.0),
        width_ratios=(0.92, 1.0, 1.0),
        hspace=0.028,
        wspace=0.045,
    )
    cell_rows = (3, 4, 6, 7)

    title_axis = figure.add_subplot(grid[0, :])
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
        "Transfer across material\nand contact geometry",
        fontsize=5.7,
        fontweight="bold",
        va="center",
        linespacing=0.95,
    )
    note_axis = figure.add_subplot(grid[1, :])
    note_axis.axis("off")
    note_axis.text(
        0.5,
        0.48,
        r"$G_{\mathrm{obs}}=1$: baseline  ·  $>1$: improved",
        fontsize=4.5,
        color="#666666",
        ha="center",
        va="center",
    )

    for column, column_title in enumerate(COMPARISON_TITLES[1:], start=1):
        header_axis = figure.add_subplot(grid[2, column])
        header_axis.axis("off")
        header_axis.text(
            0.5,
            0.50,
            column_title,
            fontsize=5.0,
            ha="center",
            va="center",
        )

    cell_axes = []
    for row, (cell_row, (_, _, row_title)) in enumerate(
        zip(cell_rows, COMPARISON_CONDITIONS, strict=True)
    ):
        label_axis = figure.add_subplot(grid[cell_row, 0])
        label_axis.axis("off")
        material, sphere = row_title.split(" · ", maxsplit=1)
        label_axis.text(
            0.0,
            0.5,
            f"{material}\n{sphere.replace(' sphere', '')}",
            fontsize=4.7,
            ha="left",
            va="center",
            linespacing=1.05,
        )
        for column in range(2):
            axis = figure.add_subplot(grid[cell_row, column + 1])
            axis.set_xticks([])
            axis.set_yticks([])
            value = values[row, column]
            if np.isfinite(value):
                axis.set_facecolor(colormap(normalization(value)))
                text = f"{value:.2f}×"
                color = "#1F2A27"
                weight = "bold"
            else:
                axis.set_facecolor("#EFEFEF")
                text = "pending"
                color = "#888888"
                weight = "normal"
            axis.text(
                0.5,
                0.5,
                text,
                transform=axis.transAxes,
                fontsize=6.4 if np.isfinite(value) else 4.8,
                fontweight=weight,
                color=color,
                ha="center",
                va="center",
            )
            for spine in axis.spines.values():
                spine.set_linewidth(0.45)
                spine.set_color("#9A9A9A")
            cell_axes.append(axis)

    return {
        "axes": tuple(cell_axes),
        "metrics": rows,
        "matrix": values,
        "color_limits": (lower, upper),
    }


def main() -> None:
    """Export a standalone debug render of Figure 5(c)."""

    with publication_context(DEFAULT_STYLE):
        figure = plt.figure(figsize=(3.15, 4.25))
        grid = figure.add_gridspec(1, 1, left=0.03, right=0.985, bottom=0.025, top=0.99)
        render_panel(figure, grid[0, 0])
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
