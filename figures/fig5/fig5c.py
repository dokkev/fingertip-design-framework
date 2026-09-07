"""Panel tools for Figure 5(c) location-resolved confusion matrices."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec

from experiments.analysis.fig5c_decoder import PaperFigureConfig
from lumo.visualization import DEFAULT_STYLE, STRUCTURAL_FONT_FAMILY

from .config import (
    COMPARISON_CONDITIONS,
    COMPARISON_MORPHOLOGIES,
    MATERIAL_SEPARATOR_COLOR,
    MATERIAL_SEPARATOR_LINEWIDTH_PT,
    MORPHOLOGY_TABLE_HEIGHT_RATIOS,
    MORPHOLOGY_TABLE_HSPACE,
    MORPHOLOGY_TABLE_ROW_SLOTS,
)

EXPECTED_CONTACT_POSITIONS_MM = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
CONFUSION_CMAP = "Blues"
CONFUSION_ANNOTATION_THRESHOLD_PERCENT = 0.0
CONFUSION_LIGHT_TEXT_THRESHOLD_PERCENT = 60.0
PLOT_COLUMNS = (2, 3, 5, 6)


def build_confusion_matrices(
    config: PaperFigureConfig,
    predictions: list[dict[str, str]],
) -> dict[tuple[str, str, str], np.ndarray]:
    """Build one row-normalized 6-by-6 matrix per physical condition."""

    holes = tuple(sorted(config.contact_positions_mm))
    configured_positions = tuple(config.contact_positions_mm[hole] for hole in holes)
    if configured_positions != EXPECTED_CONTACT_POSITIONS_MM:
        raise ValueError(
            "Figure 5(c) requires the corrected 0--50 mm contact mapping, "
            f"got {configured_positions}"
        )
    hole_to_index = {hole: index for index, hole in enumerate(holes)}
    expected_conditions = {
        (material, indenter, morphology)
        for material in config.materials
        for indenter in config.indenters
        for morphology in config.morphologies
    }
    counts = {
        condition: np.zeros((len(holes), len(holes)), dtype=np.int64)
        for condition in expected_conditions
    }

    for row in predictions:
        condition = (row["material"], row["indenter"], row["morphology_id"])
        if condition not in counts:
            raise ValueError(f"unexpected decoder condition: {condition}")
        true_hole = int(row["true_hole_index"])
        predicted_hole = int(row["predicted_hole_spatial"])
        if true_hole not in hole_to_index or predicted_hole not in hole_to_index:
            raise ValueError(
                "decoder prediction contains an unmapped contact index: "
                f"true={true_hole}, predicted={predicted_hole}"
            )
        counts[condition][
            hole_to_index[true_hole], hole_to_index[predicted_hole]
        ] += 1

    matrices: dict[tuple[str, str, str], np.ndarray] = {}
    for condition, matrix in counts.items():
        row_totals = matrix.sum(axis=1)
        if np.any(row_totals == 0):
            missing_positions = [
                position
                for position, total in zip(
                    EXPECTED_CONTACT_POSITIONS_MM, row_totals, strict=True
                )
                if total == 0
            ]
            raise ValueError(
                f"condition {condition} has no held-out samples at "
                f"{missing_positions} mm"
            )
        matrices[condition] = matrix / row_totals[:, np.newaxis] * 100.0
    return matrices


def _style_matrix_axis(
    axis: Any,
    *,
    show_x_tick_labels: bool,
    show_y_tick_labels: bool,
    font_scale: float,
) -> None:
    tick_positions = np.arange(len(EXPECTED_CONTACT_POSITIONS_MM))
    tick_labels = [f"{position:.0f}" for position in EXPECTED_CONTACT_POSITIONS_MM]
    axis.set_xticks(tick_positions)
    axis.set_yticks(tick_positions)
    axis.set_xticklabels(tick_labels if show_x_tick_labels else ())
    axis.set_yticklabels(tick_labels if show_y_tick_labels else ())
    axis.tick_params(
        axis="both",
        # A 6-by-6 matrix is only a few tenths of an inch wide in the final
        # three-panel composition. Keep this localized dense-grid exception
        # above 4 pt rather than shrinking the shared figure typography.
        labelsize=3.8 * font_scale,
        length=1.1,
        pad=0.45,
    )
    axis.set_xticks(np.arange(-0.5, 6.0, 1.0), minor=True)
    axis.set_yticks(np.arange(-0.5, 6.0, 1.0), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.25)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_color("#737373")
        spine.set_linewidth(0.34)


def render_panel(
    figure: Figure,
    subplot_spec: SubplotSpec,
    *,
    config: PaperFigureConfig,
    matrices: dict[tuple[str, str, str], np.ndarray],
    panel_label: str = "(c)",
    font_scale: float = 1.0,
    show_row_labels: bool = False,
) -> dict[str, object]:
    """Render morphology rows by material-and-indenter columns."""

    morphology_label_width = 0.13 if show_row_labels else 0.012
    grid = subplot_spec.subgridspec(
        6,
        8,
        height_ratios=MORPHOLOGY_TABLE_HEIGHT_RATIOS,
        width_ratios=(
            0.20,
            morphology_label_width,
            1,
            1,
            0.04,
            1,
            1,
            0.02,
        ),
        hspace=MORPHOLOGY_TABLE_HSPACE,
        wspace=0.02,
    )
    title_axis = figure.add_subplot(grid[0, :])
    title_axis.axis("off")
    title_axis.text(
        0.0,
        0.55,
        panel_label,
        fontsize=DEFAULT_STYLE.panel_label_font_size_pt * font_scale,
        fontweight="normal",
        va="center",
    )
    title_axis.text(
        0.170,
        0.55,
        "Contact-location decoding",
        fontsize=DEFAULT_STYLE.panel_title_font_size_pt * font_scale,
        fontweight="normal",
        va="center",
    )

    for material, column_slice in (
        ("solaris", slice(2, 4)),
        ("dragon_skin", slice(5, 7)),
    ):
        material_axis = figure.add_subplot(grid[1, column_slice])
        material_axis.axis("off")
        material_axis.text(
            0.5,
            0.55,
            config.material_labels[material],
            fontsize=DEFAULT_STYLE.group_header_font_size_pt * font_scale,
            fontweight="bold",
            fontfamily=STRUCTURAL_FONT_FAMILY,
            ha="center",
            va="center",
        )

    for column, (_, indenter, _) in zip(
        PLOT_COLUMNS, COMPARISON_CONDITIONS, strict=True
    ):
        column_axis = figure.add_subplot(grid[2, column])
        column_axis.axis("off")
        column_axis.text(
            0.5,
            0.52,
            f"Ø{config.indenter_labels[indenter]}".replace(" sphere", "\nsphere"),
            fontsize=DEFAULT_STYLE.minimum_font_size_pt * font_scale,
            linespacing=0.92,
            ha="center",
            va="center",
        )

    axes: list[Any] = []
    image = None
    shared_y_axis = figure.add_subplot(grid[3:, 0])
    shared_y_axis.axis("off")
    shared_y_axis.text(
        0.0,
        0.5,
        "True contact location [mm]",
        fontsize=DEFAULT_STYLE.minimum_font_size_pt * font_scale,
        rotation=90,
        ha="center",
        va="center",
    )
    separator_axis = figure.add_subplot(grid[1:, 4])
    separator_axis.axis("off")
    separator_axis.plot(
        (0.5, 0.5),
        (0.0, 1.0),
        color=MATERIAL_SEPARATOR_COLOR,
        linewidth=MATERIAL_SEPARATOR_LINEWIDTH_PT,
        transform=separator_axis.transAxes,
        clip_on=False,
    )

    for row_index, (morphology, row_slot) in enumerate(
        zip(COMPARISON_MORPHOLOGIES, MORPHOLOGY_TABLE_ROW_SLOTS, strict=True)
    ):
        if show_row_labels:
            row_label_axis = figure.add_subplot(grid[row_slot, 1])
            row_label_axis.axis("off")
            row_label_axis.text(
                0.05,
                0.5,
                config.morphology_labels[morphology],
                fontsize=DEFAULT_STYLE.minimum_font_size_pt * font_scale,
                ha="left",
                va="center",
            )

        for plot_index, ((material, indenter, _), column) in enumerate(
            zip(COMPARISON_CONDITIONS, PLOT_COLUMNS, strict=True)
        ):
            key = (material, indenter, morphology)
            if key not in matrices:
                raise ValueError(f"missing Figure 5(c) confusion matrix: {key}")
            axis = figure.add_subplot(grid[row_slot, column])
            image = axis.imshow(
                matrices[key],
                cmap=CONFUSION_CMAP,
                vmin=0.0,
                vmax=100.0,
                interpolation="nearest",
                aspect="equal",
            )
            for true_index, predicted_index in np.ndindex(matrices[key].shape):
                percentage = matrices[key][true_index, predicted_index]
                if percentage <= CONFUSION_ANNOTATION_THRESHOLD_PERCENT:
                    continue
                text_color = (
                    "white"
                    if percentage >= CONFUSION_LIGHT_TEXT_THRESHOLD_PERCENT
                    else "#252525"
                )
                axis.text(
                    predicted_index,
                    true_index,
                    f"{percentage / 100.0:.1f}",
                    color=text_color,
                    fontsize=3.8 * font_scale,
                    ha="center",
                    va="center",
                )
            _style_matrix_axis(
                axis,
                show_x_tick_labels=(
                    row_index == len(COMPARISON_MORPHOLOGIES) - 1
                    and plot_index in (0, 2)
                ),
                show_y_tick_labels=(plot_index == 0),
                font_scale=font_scale,
            )
            axis.set_anchor("C")
            axes.append(axis)

    if image is None:
        raise ValueError("no Figure 5(c) confusion matrices were rendered")

    body_position = grid[3:, 2:7].get_position(figure)
    figure.text(
        0.5 * (body_position.x0 + body_position.x1),
        body_position.y0 - 0.014,
        "Predicted contact location [mm]",
        fontsize=DEFAULT_STYLE.annotation_font_size_pt * font_scale,
        ha="center",
        va="top",
    )
    return {"axes": tuple(axes), "matrices": matrices}
