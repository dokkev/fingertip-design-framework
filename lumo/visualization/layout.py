"""Figure creation, composition, labeling, and export helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from string import ascii_lowercase

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text

from .style import DEFAULT_STYLE, FigureWidth, PublicationStyle, publication_context


def _figure_size(
    rows: int,
    cols: int,
    width: FigureWidth,
    panel_aspect: float,
    style: PublicationStyle,
) -> tuple[float, float]:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if panel_aspect <= 0.0:
        raise ValueError("panel_aspect must be positive")
    width_in = style.figure_width_in(width)
    return width_in, width_in * panel_aspect * rows / cols


def create_figure(
    rows: int,
    cols: int,
    *,
    width: FigureWidth = "double",
    panel_aspect: float = 0.75,
    sharex: bool = False,
    sharey: bool = False,
    style: PublicationStyle = DEFAULT_STYLE,
) -> tuple[Figure, np.ndarray]:
    """Create a uniform publication layout with a stable two-dimensional axes array."""

    figure_size = _figure_size(rows, cols, width, panel_aspect, style)
    with publication_context(style):
        figure, axes = plt.subplots(
            rows,
            cols,
            figsize=figure_size,
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
            constrained_layout=True,
        )
    return figure, np.asarray(axes, dtype=object)


def create_gridspec(
    rows: int,
    cols: int,
    *,
    width: FigureWidth = "double",
    panel_aspect: float = 0.75,
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
    style: PublicationStyle = DEFAULT_STYLE,
) -> tuple[Figure, GridSpec]:
    """Create a publication figure and GridSpec for a non-uniform layout."""

    figure_size = _figure_size(rows, cols, width, panel_aspect, style)
    with publication_context(style):
        figure = plt.figure(figsize=figure_size, constrained_layout=True)
        grid = figure.add_gridspec(
            rows,
            cols,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )
    return figure, grid


def render_panel(
    panel: Callable[..., object],
    *args: object,
    width: FigureWidth = "single",
    panel_aspect: float = 0.75,
    style: PublicationStyle = DEFAULT_STYLE,
    **kwargs: object,
) -> tuple[Figure, Axes, object]:
    """Render one axes-owned panel in a standalone publication figure."""

    figure_size = _figure_size(1, 1, width, panel_aspect, style)
    with publication_context(style):
        figure, axes = plt.subplots(figsize=figure_size, constrained_layout=True)
        artist = panel(axes, *args, style=style, **kwargs)
    return figure, axes, artist


def _alphabetic_label(index: int) -> str:
    label = ""
    value = index
    while True:
        value, remainder = divmod(value, len(ascii_lowercase))
        label = ascii_lowercase[remainder] + label
        if value == 0:
            return label
        value -= 1


def add_panel_labels(
    axes: Iterable[Axes] | np.ndarray,
    labels: Sequence[str] | None = None,
    *,
    x: float = -0.14,
    y: float = 1.05,
    style: PublicationStyle = DEFAULT_STYLE,
) -> tuple[Text, ...]:
    """Add bold panel labels in axes coordinates at composition time."""

    axes_list = (
        list(axes.flat)
        if isinstance(axes, np.ndarray)
        else list(axes)
    )
    if labels is None:
        labels = tuple(f"({_alphabetic_label(index)})" for index in range(len(axes_list)))
    if len(labels) != len(axes_list):
        raise ValueError("labels must have one entry per axes")
    return tuple(
        axes_item.text(
            x,
            y,
            label,
            transform=axes_item.transAxes,
            fontweight="bold",
            fontsize=style.panel_label_font_size_pt,
            ha="left",
            va="bottom",
        )
        for axes_item, label in zip(axes_list, labels, strict=True)
    )


def add_figure_box(
    figure: Figure,
    bounds: tuple[float, float, float, float],
    *,
    zorder: float = -10.0,
) -> FancyBboxPatch:
    """Add the shared rounded publication box in figure coordinates."""

    x0, y0, x1, y1 = bounds
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("figure-box bounds must be ordered inside [0, 1]")
    box = FancyBboxPatch(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        boxstyle="round,pad=0.002,rounding_size=0.004",
        transform=figure.transFigure,
        facecolor="#FFFFFF",
        edgecolor="#C7C7C7",
        linewidth=0.32,
        clip_on=False,
        zorder=zorder,
    )
    figure.add_artist(box)
    return box


def save_figure(
    figure: Figure,
    output_stem: str | Path,
    *,
    formats: Sequence[str] = ("pdf", "png"),
    style: PublicationStyle = DEFAULT_STYLE,
    transparent: bool = False,
) -> tuple[Path, ...]:
    """Save one assembled figure as vector and/or high-resolution raster files."""

    stem = Path(output_stem)
    if stem.suffix:
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    normalized_formats = tuple(file_format.lower().lstrip(".") for file_format in formats)
    unsupported = set(normalized_formats) - {"pdf", "svg", "png"}
    if unsupported:
        raise ValueError(f"unsupported publication formats: {sorted(unsupported)}")
    if not normalized_formats:
        raise ValueError("at least one output format is required")

    output_paths: list[Path] = []
    for file_format in normalized_formats:
        output_path = stem.with_suffix(f".{file_format}")
        save_options: dict[str, object] = {
            "bbox_inches": "tight",
            "pad_inches": 0.02,
            "transparent": transparent,
            # Preserve high-resolution raster panels inside PDF/SVG while
            # keeping text, paths, and annotations vector-based.
            "dpi": style.png_dpi,
        }
        figure.savefig(output_path, **save_options)
        output_paths.append(output_path)
    return tuple(output_paths)
