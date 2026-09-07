"""Figure 5(c): spatial distinguishability relative to repeat variation."""

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

from experiments.analysis.metrics import morphology_metrics, spatial_metrics  # noqa: E402
from lumo.visualization import DEFAULT_STYLE, publication_context, save_figure  # noqa: E402

from .config import (  # noqa: E402
    ANALYSIS_CONDITION_OVERRIDES,
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

# run_0045 failed the production large-force-spread QC at the 2 N state.
# Keep the raw dataset intact and make the Figure 5(c)-only exclusion explicit.
FIGURE_5C_EXCLUDED_RUN_IDS = {
    ("dragon_skin", "angled_opt", "sphere_30mm"): ("run_0045",),
}


def _metrics_with_run_exclusions(
    key: tuple[str, str, str],
    excluded_run_ids: tuple[str, ...],
) -> dict[str, object]:
    """Re-evaluate one stored condition with exact production metric functions."""

    material, morphology, indenter = key
    root = ANALYSIS_CONDITION_OVERRIDES.get(key, ANALYSIS_ROOTS[material])
    rows_path = root / "results" / "run_metrics.csv"
    profiles_path = root / "raw_data_summary" / "load_response_profiles.npz"

    with rows_path.open(newline="", encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    with np.load(profiles_path, allow_pickle=False) as archive:
        slope_profiles = np.asarray(archive["slope_profiles"], dtype=np.float64)
        profile_keys = list(
            zip(
                archive["material"].astype(str),
                archive["morphology"].astype(str),
                archive["indenter"].astype(str),
                archive["run_id"].astype(str),
                strict=True,
            )
        )

    profile_lookup = {
        profile_key: slope_profiles[index]
        for index, profile_key in enumerate(profile_keys)
    }
    excluded = set(excluded_run_ids)
    found_excluded: set[str] = set()
    selected_rows: list[dict[str, object]] = []
    selected_profiles: list[np.ndarray] = []
    for row in all_rows:
        row_key = (row["material"], row["morphology"], row["indenter"])
        if row_key != key:
            continue
        run_id = row["run_id"]
        if run_id in excluded:
            found_excluded.add(run_id)
            continue
        profile_key = (*key, run_id)
        if profile_key not in profile_lookup:
            raise RuntimeError(f"missing Figure 5(c) slope profile: {profile_key}")
        selected_rows.append(row)
        selected_profiles.append(profile_lookup[profile_key])

    missing = excluded - found_excluded
    if missing:
        raise RuntimeError(
            f"Figure 5(c) exclusions were not found for {key}: {sorted(missing)}"
        )
    neighboring, variability = spatial_metrics(
        selected_rows,
        np.asarray(selected_profiles, dtype=np.float64),
        hole_spacing_mm=10.0,
    )
    summary = morphology_metrics(selected_rows, neighboring, variability)
    if len(summary) != 1:
        raise RuntimeError(f"expected one filtered Figure 5(c) metric row for {key}")
    return summary[0] | {"excluded_run_ids": ";".join(excluded_run_ids)}


def load_spatial_metrics() -> list[dict[str, object]]:
    """Read stored D/W metrics and add matching-baseline improvements."""

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

    for override_key, root in ANALYSIS_CONDITION_OVERRIDES.items():
        path = root / "results" / "morphology_metrics.csv"
        replacement = None
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (row["material"], row["morphology"], row["indenter"])
                if key == override_key:
                    replacement = row
                    break
        if replacement is None:
            raise RuntimeError(f"missing Figure 5(c) override metric: {override_key}")
        source_rows[override_key] = replacement

    for key, excluded_run_ids in FIGURE_5C_EXCLUDED_RUN_IDS.items():
        source_rows[key] = _metrics_with_run_exclusions(key, excluded_run_ids)

    output: list[dict[str, object]] = []
    for material, indenter, _ in COMPARISON_CONDITIONS:
        baseline_key = (material, "baseline", indenter)
        if baseline_key not in source_rows:
            raise RuntimeError(f"missing Figure 5(c) baseline metric: {baseline_key}")
        baseline_ratio = float(source_rows[baseline_key]["D_neighbor_over_W"])
        if not np.isfinite(baseline_ratio) or baseline_ratio <= 0.0:
            raise ValueError(f"invalid Figure 5(c) baseline: {baseline_key}")

        for morphology in COMPARISON_MORPHOLOGIES:
            key = (material, morphology, indenter)
            source = source_rows.get(key)
            if source is None:
                raise RuntimeError(f"missing required Figure 5(c) metric: {key}")

            separation = float(source["D_neighbor_median_DN_per_N"])
            separation_iqr = float(source["D_neighbor_IQR_DN_per_N"])
            repeat_variation = float(source["W_median_DN_per_N"])
            repeat_variation_iqr = float(source["W_IQR_DN_per_N"])
            ratio = float(source["D_neighbor_over_W"])
            if not np.isfinite(separation) or separation <= 0.0:
                raise ValueError(f"invalid Figure 5(c) separation: {key}")
            if not np.isfinite(separation_iqr) or separation_iqr < 0.0:
                raise ValueError(f"invalid Figure 5(c) separation IQR: {key}")
            if not np.isfinite(repeat_variation) or repeat_variation <= 0.0:
                raise ValueError(f"invalid Figure 5(c) repeat variation: {key}")
            if not np.isfinite(repeat_variation_iqr) or repeat_variation_iqr < 0.0:
                raise ValueError(f"invalid Figure 5(c) repeat-variation IQR: {key}")
            if not np.isfinite(ratio) or ratio <= 0.0:
                raise ValueError(f"invalid Figure 5(c) D/W ratio: {key}")
            expected_ratio = separation / repeat_variation
            if not np.isclose(ratio, expected_ratio, rtol=1.0e-9, atol=1.0e-12):
                raise ValueError(
                    "inconsistent stored Figure 5(c) D/W ratio for "
                    f"{key}: stored={ratio:.12g}, D/W={expected_ratio:.12g}"
                )
            output.append(
                {
                    "material": material,
                    "morphology": morphology,
                    "indenter": indenter,
                    "D_neighbor_median_DN_per_N": separation,
                    "D_neighbor_IQR_DN_per_N": separation_iqr,
                    "W_median_DN_per_N": repeat_variation,
                    "W_IQR_DN_per_N": repeat_variation_iqr,
                    "D_neighbor_over_W": ratio,
                    "baseline_D_neighbor_over_W": baseline_ratio,
                    "improvement_percent": 100.0 * (ratio / baseline_ratio - 1.0),
                    "excluded_run_ids": source.get("excluded_run_ids", ""),
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
        "W_median_DN_per_N",
        "W_IQR_DN_per_N",
        "D_neighbor_over_W",
        "baseline_D_neighbor_over_W",
        "improvement_percent",
        "excluded_run_ids",
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
        f"{'D_neighbor':>12} {'W_repeat':>12} {'D/W':>10} "
        f"{'improvement [%]':>17}"
    )
    for row in rows:
        if row["status"] == "pending":
            separation_text = "pending"
            repeat_text = "pending"
            ratio_text = "pending"
            improvement_text = "pending"
        else:
            separation_text = f"{float(row['D_neighbor_median_DN_per_N']):.4f}"
            repeat_text = f"{float(row['W_median_DN_per_N']):.4f}"
            ratio_text = f"{float(row['D_neighbor_over_W']):.4f}"
            improvement_text = f"{float(row['improvement_percent']):+.1f}"
        material = str(row["material"]).replace("dragon_skin", "Dragon Skin")
        print(
            f"{material:<13} {str(row['indenter']):<13} "
            f"{str(row['morphology']):<12} {separation_text:>12} "
            f"{repeat_text:>12} {ratio_text:>10} "
            f"{improvement_text:>17}"
        )


def render_panel(
    figure: Figure,
    subplot_spec: SubplotSpec,
    *,
    panel_label: str = "(c)",
    metrics: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Render absolute stored D/W values with baseline-relative annotations."""

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
        "Spatial distinguishability\nrelative to repeat variation",
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
        float(row["D_neighbor_over_W"])
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

            value = float(row["D_neighbor_over_W"])
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
        r"$D_{\mathrm{neighbor}} / W_{\mathrm{repeat}}$",
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
            1, 1, left=0.065, right=0.985, bottom=0.025, top=0.99
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
