"""Figure 5(c): measured neighboring-contact spatial separation."""

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
    ALL_HOLES,
    ANALYSIS_ROOTS,
    COMPARISON_CONDITIONS,
    COMPARISON_MORPHOLOGIES,
    COMPARISON_TITLES,
    FIGURE_DIRECTORY,
    HOLE_TO_CONTACT_X_MM,
    require_available_inputs,
)


FORCE_TARGETS_N = (2.0, 5.0, 10.0, 15.0)
MORPHOLOGY_COLORS = {
    "baseline": "#B8BCC2",
    "flat_opt": "#4F7180",
    "angled_opt": "#A87446",
}


def _text_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).astype(str)


def _load_profiles(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        required = {
            "profiles",
            "material",
            "morphology",
            "run_status",
            "indenter",
            "hole_index",
            "repetition_index",
            "target_force_n",
        }
        missing = required.difference(bundle.files)
        if missing:
            raise KeyError(
                f"missing Figure 5(c) profile fields in {path}: {sorted(missing)}"
            )
        return {name: np.asarray(bundle[name]) for name in required}


def load_spatial_separation_metrics(
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Compute all same-force neighbor distances and morphology summaries."""

    require_available_inputs()
    bundles = {
        material: _load_profiles(
            root / "raw_data_summary" / "longitudinal_profiles.npz"
        )
        for material, root in ANALYSIS_ROOTS.items()
    }
    neighbor_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for material, indenter, _ in COMPARISON_CONDITIONS:
        data = bundles[material]
        profiles = np.asarray(data["profiles"], dtype=np.float64)
        if profiles.ndim != 2 or profiles.shape[1] != 128:
            raise ValueError(
                f"Figure 5(c) requires 128-bin profiles, got {profiles.shape}"
            )
        material_values = _text_array(data["material"])
        morphology_values = _text_array(data["morphology"])
        status_values = _text_array(data["run_status"])
        indenter_values = _text_array(data["indenter"])
        hole_values = np.asarray(data["hole_index"], dtype=np.int64)
        repetition_values = np.asarray(data["repetition_index"], dtype=np.int64)
        force_values = np.asarray(data["target_force_n"], dtype=np.float64)

        condition_summaries: list[dict[str, object]] = []
        for morphology in COMPARISON_MORPHOLOGIES:
            if material == "dragon_skin" and morphology == "angled_opt":
                condition_summaries.append(
                    {
                        "material": material,
                        "morphology": morphology,
                        "indenter": indenter,
                        "D_spatial_median_DN": float("nan"),
                        "D_spatial_q1_DN": float("nan"),
                        "D_spatial_q3_DN": float("nan"),
                        "D_spatial_iqr_DN": float("nan"),
                        "baseline_D_spatial_DN": float("nan"),
                        "improvement_percent": float("nan"),
                        "status": "pending",
                    }
                )
                continue

            separations: list[float] = []
            for force_n in FORCE_TARGETS_N:
                templates: dict[int, np.ndarray] = {}
                for hole_index in ALL_HOLES:
                    selected = np.flatnonzero(
                        (material_values == material)
                        & (morphology_values == morphology)
                        & (status_values == "complete")
                        & (indenter_values == indenter)
                        & (hole_values == hole_index)
                        & np.isclose(
                            force_values, force_n, rtol=0.0, atol=1.0e-9
                        )
                    )
                    repetitions = repetition_values[selected]
                    unique_repetitions, counts = np.unique(
                        repetitions, return_counts=True
                    )
                    if (
                        selected.size != 5
                        or unique_repetitions.size != 5
                        or np.any(counts != 1)
                    ):
                        identity = (
                            material,
                            morphology,
                            indenter,
                            force_n,
                            hole_index,
                        )
                        raise RuntimeError(
                            "Figure 5(c) requires exactly five unique complete "
                            f"repetitions for {identity}; found repetitions "
                            f"{repetitions.tolist()}"
                        )
                    selected = selected[np.argsort(repetitions)]
                    templates[hole_index] = np.median(profiles[selected], axis=0)

                for hole_i, hole_j in zip(
                    ALL_HOLES[:-1], ALL_HOLES[1:], strict=True
                ):
                    difference = templates[hole_i] - templates[hole_j]
                    separation = float(np.sqrt(np.mean(np.square(difference))))
                    if not np.isfinite(separation):
                        raise ValueError(
                            "non-finite Figure 5(c) neighboring-contact separation"
                        )
                    separations.append(separation)
                    neighbor_rows.append(
                        {
                            "material": material,
                            "morphology": morphology,
                            "indenter": indenter,
                            "force_n": force_n,
                            "contact_i_mm": HOLE_TO_CONTACT_X_MM[hole_i],
                            "contact_j_mm": HOLE_TO_CONTACT_X_MM[hole_j],
                            "D_neighbor_DN": separation,
                        }
                    )

            if len(separations) != 20:
                raise AssertionError(
                    f"expected 20 Figure 5(c) separations, found {len(separations)}"
                )
            q1, median, q3 = np.quantile(separations, (0.25, 0.5, 0.75))
            condition_summaries.append(
                {
                    "material": material,
                    "morphology": morphology,
                    "indenter": indenter,
                    "D_spatial_median_DN": float(median),
                    "D_spatial_q1_DN": float(q1),
                    "D_spatial_q3_DN": float(q3),
                    "D_spatial_iqr_DN": float(q3 - q1),
                    "baseline_D_spatial_DN": float("nan"),
                    "improvement_percent": float("nan"),
                    "status": "measured",
                }
            )

        baseline = next(
            row
            for row in condition_summaries
            if row["morphology"] == "baseline" and row["status"] == "measured"
        )
        baseline_value = float(baseline["D_spatial_median_DN"])
        if not np.isfinite(baseline_value) or baseline_value <= 0.0:
            raise ValueError(
                f"invalid Figure 5(c) baseline for {(material, indenter)}"
            )
        for row in condition_summaries:
            row["baseline_D_spatial_DN"] = baseline_value
            if row["status"] == "measured":
                value = float(row["D_spatial_median_DN"])
                row["improvement_percent"] = 100.0 * (
                    value / baseline_value - 1.0
                )
        summary_rows.extend(condition_summaries)

    return neighbor_rows, summary_rows


def _write_csv(
    rows: list[dict[str, object]], path: Path, fields: tuple[str, ...]
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_neighbor_separations(
    rows: list[dict[str, object]],
    path: Path = FIGURE_DIRECTORY / "fig5c_neighbor_separations.csv",
) -> Path:
    """Persist all 20 force/location-pair values per measured condition."""

    return _write_csv(
        rows,
        path,
        (
            "material",
            "morphology",
            "indenter",
            "force_n",
            "contact_i_mm",
            "contact_j_mm",
            "D_neighbor_DN",
        ),
    )


def write_metrics(
    rows: list[dict[str, object]],
    path: Path = FIGURE_DIRECTORY / "fig5c_metrics.csv",
) -> Path:
    """Persist the morphology-level bar heights and supporting quartiles."""

    return _write_csv(
        rows,
        path,
        (
            "material",
            "morphology",
            "indenter",
            "D_spatial_median_DN",
            "D_spatial_q1_DN",
            "D_spatial_q3_DN",
            "D_spatial_iqr_DN",
            "baseline_D_spatial_DN",
            "improvement_percent",
            "status",
        ),
    )


def render_panel(
    figure: Figure,
    subplot_spec: SubplotSpec,
    *,
    panel_label: str = "(c)",
    metrics: tuple[list[dict[str, object]], list[dict[str, object]]] | None = None,
) -> dict[str, object]:
    """Render absolute separation bars with baseline-relative annotations."""

    neighbor_rows, summary_rows = (
        load_spatial_separation_metrics() if metrics is None else metrics
    )
    write_neighbor_separations(neighbor_rows)
    write_metrics(summary_rows)
    lookup = {
        (
            str(row["material"]),
            str(row["indenter"]),
            str(row["morphology"]),
        ): row
        for row in summary_rows
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
        float(row["D_spatial_median_DN"])
        for row in summary_rows
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

            value = float(row["D_spatial_median_DN"])
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
        "Spatial separation [camera DN]",
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
        "neighbor_separations": neighbor_rows,
        "metrics": summary_rows,
        "y_limit": y_max,
    }


def main() -> None:
    """Export a standalone debug render of Figure 5(c)."""

    with publication_context(DEFAULT_STYLE):
        figure = plt.figure(figsize=(3.15, 4.25))
        grid = figure.add_gridspec(
            1, 1, left=0.03, right=0.985, bottom=0.025, top=0.99
        )
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
