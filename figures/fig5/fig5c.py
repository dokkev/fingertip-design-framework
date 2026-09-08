"""Panel tools for Figure 5(c) location-resolved confusion matrices."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec

from experiments.analysis.fig5c_decoder import PaperFigureConfig
from lumo.visualization import DEFAULT_STYLE, EDGE_COLOR

from .config import (
    COMPARISON_MORPHOLOGIES,
)
from .table_layout import (
    HEADER_ROW,
    MATERIAL_BLOCKS,
    TABLE_HEIGHT_RATIOS,
    TABLE_HSPACE,
    add_panel_title,
)

EXPECTED_CONTACT_POSITIONS_MM = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
EXPECTED_SAMPLES_PER_LOCATION = 5
CONFUSION_CMAP = "Blues"
CONFUSION_ANNOTATION_THRESHOLD_PERCENT = 0.0
CONFUSION_LIGHT_TEXT_THRESHOLD_PERCENT = 60.0
INDENTER_COLUMNS = (
    ("sphere_10mm", "Ø10 mm\nsphere"),
    ("sphere_30mm", "Ø30 mm\nsphere"),
)
PLOT_COLUMNS = (2, 3)


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
        if np.any(row_totals != EXPECTED_SAMPLES_PER_LOCATION):
            raise ValueError(
                f"condition {condition} requires "
                f"n={EXPECTED_SAMPLES_PER_LOCATION} held-out samples per "
                f"location, got {row_totals.tolist()}"
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
    displayed_locations_mm = (0.0, 20.0, 40.0)
    tick_positions = [
        EXPECTED_CONTACT_POSITIONS_MM.index(location)
        for location in displayed_locations_mm
    ]
    tick_labels = [f"{position:.0f}" for position in displayed_locations_mm]
    axis.set_xticks(tick_positions)
    axis.set_yticks(tick_positions)
    axis.set_xticklabels(tick_labels if show_x_tick_labels else ())
    axis.set_yticklabels(tick_labels if show_y_tick_labels else ())
    axis.tick_params(
        axis="both",
        labelsize=DEFAULT_STYLE.minimum_font_size_pt * font_scale,
        width=DEFAULT_STYLE.tick_width_pt,
        length=DEFAULT_STYLE.tick_length_pt,
        pad=0.45,
    )
    axis.set_xticks(np.arange(-0.5, 6.0, 1.0), minor=True)
    axis.set_yticks(np.arange(-0.5, 6.0, 1.0), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.25)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_color(EDGE_COLOR)
        spine.set_linewidth(DEFAULT_STYLE.spine_width_pt)


def render_panel(
    figure: Figure,
    subplot_spec: SubplotSpec,
    *,
    config: PaperFigureConfig,
    matrices: dict[tuple[str, str, str], np.ndarray],
    panel_label: str = "(c)",
    font_scale: float = 1.0,
) -> dict[str, object]:
    """Render six shared specimen rows by two indenter columns."""

    grid = subplot_spec.subgridspec(
        len(TABLE_HEIGHT_RATIOS),
        4,
        height_ratios=TABLE_HEIGHT_RATIOS,
        width_ratios=(0.24, 0.28, 1, 1),
        hspace=TABLE_HSPACE,
        wspace=0.06,
    )
    add_panel_title(
        figure,
        grid,
        panel_label=panel_label,
        title="Contact-location decoding",
        sample_note="n = 30 held-out predictions / matrix (5 / location)",
        font_scale=font_scale,
    )

    for column, (_, title) in zip(
        PLOT_COLUMNS, INDENTER_COLUMNS, strict=True
    ):
        column_axis = figure.add_subplot(grid[HEADER_ROW, column])
        column_axis.axis("off")
        column_axis.text(
            0.5,
            0.60,
            title,
            fontsize=DEFAULT_STYLE.condition_header_font_size_pt * font_scale,
            linespacing=0.92,
            ha="center",
            va="center",
        )

    axes: list[Any] = []
    image = None
    shared_y_axis = figure.add_subplot(grid[2:, 0])
    shared_y_axis.axis("off")
    shared_y_axis.text(
        1.0,
        0.5,
        "True contact location [mm]",
        fontsize=DEFAULT_STYLE.axis_label_font_size_pt * font_scale,
        rotation=90,
        ha="center",
        va="center",
        transform=shared_y_axis.transAxes,
        clip_on=False,
        zorder=2,
    )

    for material, morphology_rows in MATERIAL_BLOCKS:
        for morphology_index, (morphology, row_slot) in enumerate(
            zip(COMPARISON_MORPHOLOGIES, morphology_rows, strict=True)
        ):
            for plot_index, ((indenter, _), column) in enumerate(
                zip(INDENTER_COLUMNS, PLOT_COLUMNS, strict=True)
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
                    aspect="auto",
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
                        fontsize=DEFAULT_STYLE.minimum_font_size_pt * font_scale,
                        ha="center",
                        va="center",
                    )
                _style_matrix_axis(
                    axis,
                    show_x_tick_labels=(
                        material == "dragon_skin"
                        and morphology_index == len(COMPARISON_MORPHOLOGIES) - 1
                    ),
                    show_y_tick_labels=(plot_index == 0),
                    font_scale=font_scale,
                )
                axes.append(axis)

    if image is None:
        raise ValueError("no Figure 5(c) confusion matrices were rendered")

    body_position = grid[2:, 2:4].get_position(figure)
    figure.text(
        0.5 * (body_position.x0 + body_position.x1),
        body_position.y0 - 0.045,
        "Predicted contact location [mm]",
        fontsize=DEFAULT_STYLE.axis_label_font_size_pt * font_scale,
        ha="center",
        va="top",
    )
    return {"axes": tuple(axes), "matrices": matrices}
