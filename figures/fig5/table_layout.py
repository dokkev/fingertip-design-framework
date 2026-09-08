"""Shared table grammar for the three Figure 5 panels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from matplotlib.figure import Figure
from lumo.visualization import DEFAULT_STYLE, EDGE_COLOR, STRUCTURAL_FONT_FAMILY

from .config import COMPARISON_MORPHOLOGIES


TITLE_ROW = 0
HEADER_ROW = 1
TABLE_HEIGHT_RATIOS = (0.48, 0.80, 1.0, 1.0, 1.0, 0.30, 1.0, 1.0, 1.0)
TABLE_HSPACE = 0.025
MATERIAL_BLOCKS = (
    ("solaris", (2, 3, 4)),
    ("dragon_skin", (6, 7, 8)),
)


def add_panel_title(
    figure: Figure,
    grid: Any,
    *,
    panel_label: str,
    title: str,
    sample_note: str | None = None,
    font_scale: float = 1.0,
) -> None:
    """Add one aligned panel label, centered title, and optional sample note."""

    axis = figure.add_subplot(grid[TITLE_ROW, :])
    axis.axis("off")
    axis.text(
        0.0,
        0.68,
        panel_label,
        fontsize=DEFAULT_STYLE.panel_label_font_size_pt * font_scale,
        fontweight="normal",
        ha="left",
        va="center",
    )
    axis.text(
        0.5,
        0.68,
        title,
        fontsize=DEFAULT_STYLE.panel_title_font_size_pt * font_scale,
        fontweight="normal",
        ha="center",
        va="center",
    )
    if sample_note is not None:
        axis.text(
            0.5,
            0.08,
            sample_note,
            fontsize=DEFAULT_STYLE.minimum_font_size_pt * font_scale,
            color=EDGE_COLOR,
            ha="center",
            va="bottom",
        )


def add_material_labels(
    figure: Figure,
    grid: Any,
    *,
    column: int,
    labels: Mapping[str, str],
    font_scale: float = 1.0,
) -> None:
    """Render each material as one narrow merged-cell label."""

    for material, rows in MATERIAL_BLOCKS:
        axis = figure.add_subplot(grid[rows[0] : rows[-1] + 1, column])
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            labels[material],
            rotation=90,
            fontsize=DEFAULT_STYLE.group_header_font_size_pt * font_scale,
            fontweight="bold",
            fontfamily=STRUCTURAL_FONT_FAMILY,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
        )


def add_morphology_labels(
    figure: Figure,
    grid: Any,
    *,
    column: int,
    labels: Mapping[str, str],
    colors: Mapping[str, str],
    font_scale: float = 1.0,
) -> None:
    """Render morphology labels with a short paper-color identity bar."""

    for _, rows in MATERIAL_BLOCKS:
        for morphology, row in zip(COMPARISON_MORPHOLOGIES, rows, strict=True):
            label = labels[morphology]
            if label.startswith("Opt-"):
                label = label.replace("-", "-\n", 1)
            axis = figure.add_subplot(grid[row, column])
            axis.axis("off")
            if morphology == "baseline":
                axis.plot(
                    (0.04, 0.04),
                    (0.28, 0.72),
                    color=EDGE_COLOR,
                    linewidth=3.6 * font_scale,
                    solid_capstyle="butt",
                    transform=axis.transAxes,
                    clip_on=False,
                )
            axis.plot(
                (0.04, 0.04),
                (0.28, 0.72),
                color=colors[morphology],
                linewidth=2.6 * font_scale,
                solid_capstyle="butt",
                transform=axis.transAxes,
                clip_on=False,
            )
            axis.text(
                0.14,
                0.5,
                label,
                fontsize=DEFAULT_STYLE.annotation_font_size_pt * font_scale,
                fontweight="normal",
                linespacing=0.90,
                ha="left",
                va="center",
                transform=axis.transAxes,
                clip_on=False,
            )
__all__ = [
    "HEADER_ROW",
    "MATERIAL_BLOCKS",
    "TABLE_HEIGHT_RATIOS",
    "TABLE_HSPACE",
    "add_material_labels",
    "add_morphology_labels",
    "add_panel_title",
]
