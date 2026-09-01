"""Compose paper Figure 3 from paired ablations and completed BO campaigns."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from lumo.visualization import (  # noqa: E402
    DEFAULT_STYLE,
    add_figure_box,
    plot_carrier_identity_comparison,
    plot_pareto_small_multiple,
    plot_structural_ablation_schematic,
    plot_void_coupled_response,
    publication_context,
    save_figure,
)


_ROOT = Path(__file__).resolve().parents[1]
_ABLATION_PATH = (
    _ROOT
    / "output"
    / "validation"
    / "multi_design_void_ablation"
    / "paired_effects.csv"
)
_REPORT_PATH = (
    _ROOT
    / "output"
    / "validation"
    / "multi_design_void_ablation"
    / "figure3_validation.md"
)
_OUTPUT_STEM = Path(__file__).resolve().with_suffix("")
_VOID_CMAP = LinearSegmentedColormap.from_list(
    "viridis_truncated",
    plt.get_cmap("viridis")(np.linspace(0.05, 0.90, 256)),
)
_CAMPAIGNS = (
    (r"DragonSkin · $\theta=0^\circ$", "mobo_fingertip_contact_1_2_5_10_05mm"),
    (
        "DragonSkin · 5 angles",
        "mobo_fingertip_orientation_robust_1_2_5_10_05mm",
    ),
    (
        r"Solaris · $\theta=0^\circ$",
        "mobo_fingertip_contact_1_2_5_10_05mm_solaris_nominal",
    ),
    (
        "Solaris · 5 angles",
        "mobo_fingertip_orientation_robust_1_2_5_10_05mm_solaris_nominal",
    ),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _read_paired_results(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; complete multi_design_void_ablation.py --all first"
        )
    rows = _read_csv(path)
    if not rows:
        raise ValueError("paired ablation table is empty")

    fields = {
        "soft_contact": "soft_J_contact",
        "carrier_contact": "no_void_J_contact",
        "carrier_delta": "carrier_delta_fixed_J_contact",
        "void_delta_contact": "void_delta_fixed_J_contact",
        "void_delta_d_1n": "void_delta_D_1N",
        "void_width_mm": "lumo_void_width_mm",
    }
    arrays = {
        name: np.asarray([float(row[column]) for row in rows], dtype=np.float64)
        for name, column in fields.items()
    }
    if not all(np.all(np.isfinite(values)) for values in arrays.values()):
        raise ValueError("paired ablation table contains missing or non-finite values")
    if not np.allclose(
        arrays["carrier_contact"] - arrays["soft_contact"],
        arrays["carrier_delta"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("carrier paired effects do not match the stored objectives")

    zero = np.isclose(arrays["void_width_mm"], 0.0, atol=1.0e-12)
    if not np.any(zero):
        raise ValueError("expected at least one zero-void identity sample")
    if not np.allclose(arrays["void_delta_contact"][zero], 0.0, atol=1.0e-12):
        raise ValueError("zero-void mechanical identity checks are not zero")
    if not np.allclose(arrays["void_delta_d_1n"][zero], 0.0, atol=1.0e-12):
        raise ValueError("zero-void optical identity checks are not zero")
    return arrays


def _empirical_pareto_mask(j_contact: np.ndarray, j_obs: np.ndarray) -> np.ndarray:
    """Return non-dominated observations for two maximized objectives."""

    mask = np.ones(j_contact.size, dtype=np.bool_)
    for index, (contact, observation) in enumerate(zip(j_contact, j_obs, strict=True)):
        dominates = (
            (j_contact >= contact)
            & (j_obs >= observation)
            & ((j_contact > contact) | (j_obs > observation))
        )
        mask[index] = not np.any(dominates)
    return mask


def _read_optimization_dataset(label: str, directory: str) -> dict[str, object]:
    root = _ROOT / "output" / "optimization" / directory
    config = json.loads((root / "run_config.json").read_text())[
        "scientific_contract"
    ]
    objectives = config["objectives"]
    if objectives["names"] != ["J_contact", "J_obs"]:
        raise ValueError(f"unexpected objectives in {root}")
    if objectives["directions"] != ["maximize", "maximize"]:
        raise ValueError(f"Figure 3 requires two maximized objectives: {root}")

    valid_rows = []
    for row in _read_csv(root / "trials.csv"):
        try:
            objective_values = (float(row["J_contact"]), float(row["J_obs"]))
        except (TypeError, ValueError):
            continue
        if (
            row["status"] == "COMPLETED"
            and row["analytically_valid"] == "True"
            and not row["failure"]
            and np.all(np.isfinite(objective_values))
        ):
            valid_rows.append(row)
    if not valid_rows:
        raise RuntimeError(f"campaign has no valid completed observations: {root}")

    trial_id = np.asarray(
        [int(row["ax_trial_index"]) for row in valid_rows], dtype=np.int64
    )
    if np.unique(trial_id).size != trial_id.size:
        raise ValueError(f"duplicate completed trial IDs in {root}")
    contact = np.asarray(
        [float(row["J_contact"]) for row in valid_rows], dtype=np.float64
    )
    observation = np.asarray(
        [float(row["J_obs"]) for row in valid_rows], dtype=np.float64
    )
    void_width = np.asarray(
        [float(row["geometry.void_width_mm"]) for row in valid_rows],
        dtype=np.float64,
    )
    pareto = _empirical_pareto_mask(contact, observation)
    stored_pareto = np.asarray(
        [row["is_pareto"] == "True" for row in valid_rows], dtype=np.bool_
    )
    if not np.array_equal(pareto, stored_pareto):
        raise ValueError(f"stored and recomputed Pareto flags disagree: {root}")
    pareto_csv_ids = {
        int(row["ax_trial_index"]) for row in _read_csv(root / "pareto.csv")
    }
    if pareto_csv_ids != set(trial_id[pareto]):
        raise ValueError(f"pareto.csv disagrees with recomputed non-dominance: {root}")

    relative_score = np.minimum(
        contact / np.max(contact), observation / np.max(observation)
    )
    best_score = float(np.max(relative_score))
    tied = np.flatnonzero(
        np.isclose(relative_score, best_score, rtol=0.0, atol=1.0e-15)
    )
    balanced_index = int(tied[np.argmin(trial_id[tied])])
    if not pareto[balanced_index]:
        raise ValueError(f"balanced trial is not Pareto optimal: {root}")

    return {
        "label": label,
        "directory": directory,
        "trial_id": trial_id,
        "j_contact": contact,
        "j_obs": observation,
        "void_width_mm": void_width,
        "pareto": pareto,
        "balanced_index": balanced_index,
        "objective_definition": objectives["definition"],
        "scenarios": config["scenarios"],
        "mechanics_preset": config["mechanics_preset"],
        "optical_preset": config["optical_preset"],
        "void_bounds_mm": tuple(
            config["design_space"]["decoded_physical_bounds_mm"][
                "geometry.void_width_mm"
            ]
        ),
        "emitted_power": float(config["optics"]["simultaneous_emitted_power"]),
    }


def _check_optimization_contracts(
    datasets: tuple[dict[str, object], ...],
) -> tuple[float, float]:
    bounds = {tuple(dataset["void_bounds_mm"]) for dataset in datasets}
    emitted_power = {float(dataset["emitted_power"]) for dataset in datasets}
    if len(bounds) != 1:
        raise ValueError("optimization campaigns use different void-width bounds")
    if emitted_power != {5.0}:
        raise ValueError("optimization campaigns do not share P_emit=5 normalization")

    definitions = {str(dataset["objective_definition"]) for dataset in datasets}
    scenario_contracts = {
        json.dumps(dataset["scenarios"], sort_keys=True) for dataset in datasets
    }
    material_contracts = {
        (str(dataset["mechanics_preset"]), str(dataset["optical_preset"]))
        for dataset in datasets
    }
    if len(definitions) == len(scenario_contracts) == len(material_contracts) == 1:
        raise ValueError(
            "all optimization contracts are now directly comparable; reconsider the "
            "separate Figure 3 Pareto panels"
        )
    return next(iter(bounds))


def _write_validation_report(
    ablation: dict[str, np.ndarray],
    datasets: tuple[dict[str, object], ...],
) -> None:
    carrier = ablation["carrier_delta"]
    finite = ablation["void_width_mm"] > 0.0
    carrier_q = np.quantile(carrier, (0.25, 0.5, 0.75))
    finite_contact_q = np.quantile(
        ablation["void_delta_contact"][finite], (0.25, 0.5, 0.75)
    )
    finite_optical_q = np.quantile(
        ablation["void_delta_d_1n"][finite], (0.25, 0.5, 0.75)
    )
    balanced_finite = sum(
        float(dataset["void_width_mm"][dataset["balanced_index"]]) > 0.0
        for dataset in datasets
    )
    rows = []
    for dataset in datasets:
        balanced = int(dataset["balanced_index"])
        rows.append(
            "| "
            f"{dataset['label']} | {len(dataset['trial_id'])} | "
            f"{np.count_nonzero(dataset['pareto'])} | "
            f"{int(dataset['trial_id'][balanced])} | "
            f"{float(dataset['void_width_mm'][balanced]):.1f} |"
        )

    report = f"""# Figure 3 validation

## Structural ablation

- Complete paired morphologies: {carrier.size}.
- Exact zero-void LUMO morphologies: {np.count_nonzero(~finite)}.
- Carrier effect: median $\\Delta J_{{contact}}={carrier_q[1]:+.6f}$, IQR [{carrier_q[0]:+.6f}, {carrier_q[2]:+.6f}], positive for {np.count_nonzero(carrier > 0.0)}/{carrier.size} morphologies.
- Finite-void contact effect: median {finite_contact_q[1]:+.6f}, IQR [{finite_contact_q[0]:+.6f}, {finite_contact_q[2]:+.6f}].
- Finite-void low-load optical effect: median $\\Delta D(1\\,\\mathrm{{N}})={finite_optical_q[1]:+.6e}$, IQR [{finite_optical_q[0]:+.6e}, {finite_optical_q[2]:+.6e}].
- The carrier effect remains uniformly positive; both signs occur for the finite-void mechanical and optical effects.

## Optimization data integrity

| Dataset | Valid evaluated | Empirical Pareto | Balanced trial | Balanced $w_v$ [mm] |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

- Stored Pareto flags and `pareto.csv` agree exactly with recomputed empirical non-dominance in all four datasets.
- All campaigns maximize both $J_{{contact}}$ and $J_{{obs}}$ and use emitted-power normalization $P_{{emit}}=5$.
- A combined quantitative Pareto cloud is not valid: the $\\theta=0$ campaigns use the v3 seven-location objective, while the orientation campaigns use the v4 five-angle worst-case objective; Dragon and Solaris also use different mechanics and optical presets.
- Panel (d) therefore uses four separately scaled empirical Pareto small multiples.
- Balanced designs retaining finite void: {balanced_finite}/{len(datasets)}. The optimizer can retain or discard the lateral-void degree of freedom depending on the complete morphology and evaluation contract.

## Suggested caption

**Figure 3. Design-space justification and multi-objective morphology optimization.** (a) Each sampled morphology is evaluated through paired structural counterfactuals consisting of a Soft-only pad, a No-void carrier configuration, and the corresponding LUMO morphology. (b) Restoring the rigid carrier consistently improves the controlled fixed-scenario contact objective across sampled morphologies. (c) Restoring the lateral void produces morphology-dependent changes in both contact mechanics and the low-load optical state-change diagnostic, showing that $w_v$ acts as a coupled internal design degree of freedom rather than a monotonic performance parameter. (d) Empirical Pareto fronts from the four completed multi-objective Bayesian optimization datasets show the search over morphologies balancing contact mechanics and optical contact observation. Separate axes are used because the evaluation contracts are not directly comparable.
"""
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(report)


def _print_validation_summary(
    ablation: dict[str, np.ndarray],
    datasets: tuple[dict[str, object], ...],
) -> None:
    carrier = ablation["carrier_delta"]
    finite = ablation["void_width_mm"] > 0.0
    carrier_q = np.quantile(carrier, (0.25, 0.5, 0.75))
    print("Figure 3 validation summary")
    print("---------------------------")
    print(f"ablation samples: {carrier.size}")
    print(
        "carrier delta J_contact: "
        f"median={carrier_q[1]:+.6f}, "
        f"IQR=[{carrier_q[0]:+.6f}, {carrier_q[2]:+.6f}], "
        f"positive={np.count_nonzero(carrier > 0.0)}/{carrier.size}"
    )
    print(
        "finite-void medians: "
        f"delta J_contact={np.median(ablation['void_delta_contact'][finite]):+.6f}, "
        f"delta D(1 N)={np.median(ablation['void_delta_d_1n'][finite]):+.6e}"
    )
    for dataset in datasets:
        balanced = int(dataset["balanced_index"])
        print(
            f"{dataset['label']}: valid={len(dataset['trial_id'])}, "
            f"Pareto={np.count_nonzero(dataset['pareto'])}, "
            f"balanced trial={int(dataset['trial_id'][balanced])}, "
            f"w_v={float(dataset['void_width_mm'][balanced]):.1f} mm"
        )
    print("combined Pareto cloud: no; objective/evaluation contracts differ")


def main() -> None:
    ablation = _read_paired_results(_ABLATION_PATH)
    datasets = tuple(
        _read_optimization_dataset(label, directory)
        for label, directory in _CAMPAIGNS
    )
    void_bounds = _check_optimization_contracts(datasets)
    void_normalization = Normalize(*void_bounds)

    with publication_context():
        figure = plt.figure(figsize=(7.14, 4.05), constrained_layout=False)
        outer = figure.add_gridspec(
            1,
            2,
            width_ratios=(0.18, 0.82),
            left=0.012,
            right=0.995,
            bottom=0.060,
            top=0.930,
            wspace=0.10,
        )
        structural_axes = figure.add_subplot(outer[0, 0])

        board = outer[0, 1].subgridspec(
            5,
            3,
            height_ratios=(0.12, 1.0, 0.28, 1.0, 0.05),
            width_ratios=(1.18, 1.0, 1.0),
            hspace=0.10,
            wspace=0.32,
        )
        carrier_axes = figure.add_subplot(board[1, 0])
        void_axes = figure.add_subplot(board[3, 0])

        plot_structural_ablation_schematic(
            structural_axes,
            sample_count=ablation["carrier_delta"].size,
        )
        plot_carrier_identity_comparison(
            carrier_axes,
            ablation["soft_contact"],
            ablation["carrier_contact"],
        )
        void_mappable = plot_void_coupled_response(
            void_axes,
            ablation["void_delta_contact"],
            ablation["void_delta_d_1n"],
            ablation["void_width_mm"],
            colormap=_VOID_CMAP,
            normalization=void_normalization,
        )
        void_axes.set_box_aspect(1.0)
        void_axes.set_anchor("N")

        structure_slot = outer[0, 0].get_position(figure)
        structure_frame_bounds = (
            structure_slot.x0,
            void_axes.get_position().y0,
            structure_slot.x1,
            carrier_axes.get_position().y1,
        )
        structural_axes.set_position(
            [
                structure_frame_bounds[0],
                structure_frame_bounds[1],
                structure_frame_bounds[2] - structure_frame_bounds[0],
                structure_frame_bounds[3] - structure_frame_bounds[1],
            ]
        )
        structural_axes.set_anchor("C")
        structural_frame_axes = figure.add_axes(
            (
                structure_frame_bounds[0],
                structure_frame_bounds[1],
                structure_frame_bounds[2] - structure_frame_bounds[0],
                structure_frame_bounds[3] - structure_frame_bounds[1],
            ),
            facecolor="none",
        )
        structural_frame_axes.set_xticks(())
        structural_frame_axes.set_yticks(())
        structural_frame_axes.tick_params(length=0.0)
        for spine in structural_frame_axes.spines.values():
            spine.set_color("#5F5F5F")
            spine.set_linewidth(0.8)

        optimization_header_axes = figure.add_subplot(board[0, 1:])
        optimization_header_axes.axis("off")

        pareto_cells = ((1, 1), (1, 2), (3, 1), (3, 2))
        pareto_axes = tuple(
            figure.add_subplot(board[row, column]) for row, column in pareto_cells
        )
        for index, (axes, dataset) in enumerate(
            zip(pareto_axes, datasets, strict=True)
        ):
            plot_pareto_small_multiple(
                axes,
                dataset["j_contact"],
                dataset["j_obs"],
                dataset["void_width_mm"],
                dataset["pareto"],
                int(dataset["balanced_index"]),
                colormap=_VOID_CMAP,
                normalization=void_normalization,
            )
            axes.set_box_aspect(1.0)
            axes.set_anchor("N")
            axes.set_title(str(dataset["label"]), fontsize=6.2, pad=1.5)
            if index >= 2:
                axes.set_xlabel(r"$J_{contact}$", fontsize=6.2, labelpad=1.0)
            if index % 2 == 0:
                axes.set_ylabel(
                    r"$J_{obs}$ [$\times 10^{-3}$]",
                    fontsize=6.2,
                    labelpad=0.0,
                )
            axes.tick_params(labelsize=5.6, pad=1.0)

        horizontal_gap = (
            pareto_axes[1].get_position().x0 - pareto_axes[0].get_position().x1
        )
        vertical_gap = (
            pareto_axes[0].get_position().y0 - pareto_axes[2].get_position().y1
        )
        horizontal_shift = 0.275 * horizontal_gap
        vertical_shift = 0.10 * vertical_gap
        for index, axes in enumerate(pareto_axes):
            position = axes.get_position()
            x_shift = horizontal_shift if index % 2 == 0 else -horizontal_shift
            y_shift = -vertical_shift if index < 2 else vertical_shift
            axes.set_position(
                [
                    position.x0 + x_shift,
                    position.y0 + y_shift,
                    position.width,
                    position.height,
                ]
            )

        legend_handles = (
            Line2D(
                (),
                (),
                marker="o",
                linestyle="none",
                markerfacecolor="#C8C8C8",
                markeredgecolor="none",
                markersize=3.5,
                label="Evaluated",
            ),
            Line2D(
                (),
                (),
                marker="o",
                linestyle="-",
                color="#555555",
                markerfacecolor="#7560A8",
                markeredgecolor="#333333",
                linewidth=0.55,
                markersize=3.8,
                label="Pareto",
            ),
            Line2D(
                (),
                (),
                marker="*",
                linestyle="none",
                markerfacecolor=DEFAULT_STYLE.colors.mechanical,
                markeredgecolor="#2F2F2F",
                markersize=6.5,
                label="Balanced",
            ),
        )
        optimization_header_axes.legend(
            handles=legend_handles,
            loc="center",
            ncol=3,
            frameon=False,
            handletextpad=0.25,
            columnspacing=0.55,
            fontsize=5.8,
        )

        void_position = void_axes.get_position()
        void_width_colorbar_axes = figure.add_axes(
            [
                void_position.x1 + 0.006,
                void_position.y0 + 0.12 * void_position.height,
                0.008,
                0.76 * void_position.height,
            ]
        )
        void_width_colorbar = figure.colorbar(
            void_mappable,
            cax=void_width_colorbar_axes,
            orientation="vertical",
            ticks=(0.0, 2.5, 5.0, 7.5),
        )
        void_width_colorbar.set_label(
            r"$w_v$ [mm]",
            fontsize=6.0,
            labelpad=1.5,
        )
        void_width_colorbar.ax.tick_params(labelsize=5.6, length=1.4, pad=0.7)
        void_width_colorbar.outline.set_linewidth(0.4)

        structure_position = outer[0, 0].get_position(figure)
        optimization_position = board[:, 1:].get_position(figure)
        block_bottom = 0.035
        block_top = 0.985
        left_bounds = (
            structure_position.x0 - 0.004,
            void_width_colorbar_axes.get_position().x1 + 0.032,
        )
        right_bounds = (
            optimization_position.x0 - 0.015,
            optimization_position.x1 + 0.004,
        )
        for left, right in (left_bounds, right_bounds):
            add_figure_box(
                figure,
                (left, block_bottom, right, block_top),
            )
        super_title_y = 0.967
        top_panel_title_y = carrier_axes.get_position().y1 + 0.006
        bottom_panel_title_y = void_axes.get_position().y1 + 0.006
        figure.text(
            0.5 * (right_bounds[0] + right_bounds[1]),
            super_title_y,
            "(d)  Morphology Optimization",
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
        )
        panel_title_style = {
            "va": "bottom",
            "fontsize": 6.6,
            "fontweight": "bold",
        }
        figure.text(
            0.5 * (structure_frame_bounds[0] + structure_frame_bounds[2]),
            top_panel_title_y,
            "(a)  Structure",
            ha="center",
            **panel_title_style,
        )
        figure.text(
            0.5
            * (
                carrier_axes.get_position().x0
                + carrier_axes.get_position().x1
            ),
            top_panel_title_y,
            "(b)  Carrier Contribution",
            ha="center",
            **panel_title_style,
        )
        figure.text(
            0.5 * (void_axes.get_position().x0 + void_axes.get_position().x1),
            bottom_panel_title_y,
            "(c)  Void Effect",
            ha="center",
            **panel_title_style,
        )
        flow_axes = figure.add_axes(
            [
                left_bounds[1],
                0.940,
                right_bounds[0] - left_bounds[1],
                0.025,
            ],
            frameon=False,
        )
        flow_axes.set_axis_off()
        flow_axes.annotate(
            "",
            xy=(0.9, 0.5),
            xytext=(0.1, 0.5),
            arrowprops={"arrowstyle": "-|>", "color": "#8A8A8A", "lw": 0.65},
        )

        outputs = save_figure(
            figure,
            _OUTPUT_STEM,
            formats=("pdf", "svg", "png"),
        )
        plt.close(figure)

    _write_validation_report(ablation, datasets)
    for output in outputs:
        print(f"wrote {output}")
    print(f"wrote {_REPORT_PATH}")
    _print_validation_summary(ablation, datasets)


if __name__ == "__main__":
    main()
