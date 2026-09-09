"""Central publication styling for LUMO figures."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import matplotlib as mpl


FigureWidth = Literal["single", "double"]


PAPER_COLORS = {
    "baseline": "#BFC3C7",
    "flat_opt": "#2C758E",
    "angled_opt": "#D97707",
}

PAPER_LABELS = {
    "baseline": "Baseline",
    "flat_opt": "Flat-Opt",
    "angled_opt": "Curved-Opt",
}

MATERIAL_LABELS = {
    "solaris": "Sol.",
    "dragon_skin": "DS.",
    "dragonskin": "DS.",
}

EDGE_COLOR = "#4C5055"
STRUCTURAL_FONT_FAMILY = "Liberation Sans"

SEMANTIC_COLORS = {
    "rigid": "#59616B",
    "pad": "#F2F1ED",
    "led": "#009E73",
    "force": "#D62728",
}


@dataclass(frozen=True)
class SemanticColors:
    """Colors with stable meanings across LUMO publication figures."""

    silicone: str = SEMANTIC_COLORS["pad"]
    carrier: str = SEMANTIC_COLORS["rigid"]
    optical: str = SEMANTIC_COLORS["led"]
    mechanical: str = SEMANTIC_COLORS["force"]
    optimization: str = "#D62728"
    neutral: str = "#777777"
    dragon_skin: str = "#8A4F9E"
    solaris: str = "#2F6B9A"
    grid: str = "#D8D8D8"


@dataclass(frozen=True)
class PublicationStyle:
    """One adjustable source for publication dimensions and typography."""

    single_column_width_in: float = 3.5
    double_column_width_in: float = 7.16
    base_font_size_pt: float = 8.0
    tick_font_size_pt: float = 7.0
    axis_label_font_size_pt: float = 8.0
    legend_font_size_pt: float = 7.0
    panel_label_font_size_pt: float = 9.0
    panel_title_font_size_pt: float = 8.0
    group_header_font_size_pt: float = 7.0
    condition_header_font_size_pt: float = 7.0
    annotation_font_size_pt: float = 6.5
    minimum_font_size_pt: float = 6.0
    line_width_pt: float = 1.2
    marker_size_pt: float = 4.5
    spine_width_pt: float = 0.7
    tick_width_pt: float = 0.7
    tick_length_pt: float = 2.5
    grid_width_pt: float = 0.5
    png_dpi: int = 600
    colors: SemanticColors = SemanticColors()

    def figure_width_in(self, width: FigureWidth) -> float:
        """Return the final paper width for one IEEE column or two columns."""

        if width == "single":
            return self.single_column_width_in
        if width == "double":
            return self.double_column_width_in
        raise ValueError(f"unsupported figure width: {width!r}")

    def rc_params(self) -> dict[str, object]:
        """Return Matplotlib parameters shared by all publication figures."""

        return {
            "font.family": "Helvetica Light",
            "font.sans-serif": [
                "Helvetica Light",
                "Helvetica",
                "Arial",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "font.size": self.base_font_size_pt,
            # The selected family is itself the Helvetica Light face. Request
            # its regular style instead of an unavailable synthetic weight.
            "font.weight": "normal",
            "axes.labelsize": self.axis_label_font_size_pt,
            "axes.labelweight": "normal",
            "axes.linewidth": self.spine_width_pt,
            "axes.titlesize": self.axis_label_font_size_pt,
            "axes.titleweight": "normal",
            "xtick.labelsize": self.tick_font_size_pt,
            "ytick.labelsize": self.tick_font_size_pt,
            "xtick.major.width": self.tick_width_pt,
            "ytick.major.width": self.tick_width_pt,
            "xtick.major.size": self.tick_length_pt,
            "ytick.major.size": self.tick_length_pt,
            "legend.fontsize": self.legend_font_size_pt,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Helvetica Light",
            "mathtext.it": "Liberation Sans:italic",
            "mathtext.bf": "Liberation Sans:bold",
            "mathtext.sf": "Helvetica Light",
            "mathtext.cal": "Liberation Sans",
            "lines.linewidth": self.line_width_pt,
            "lines.markersize": self.marker_size_pt,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }


DEFAULT_STYLE = PublicationStyle()

STATUS_MARKERS = {
    "candidate": "o",
    "optimized": "o",
    "nominal": "s",
    "suboptimal": "^",
}


def material_color(material: str, style: PublicationStyle = DEFAULT_STYLE) -> str:
    """Return the non-green color assigned to a silicone material."""

    normalized = material.lower().replace(" ", "_")
    if normalized in {"dragon_skin", "dragon_skin_10_nv"}:
        return style.colors.dragon_skin
    if normalized == "solaris":
        return style.colors.solaris
    raise ValueError(f"unsupported material: {material!r}")


@contextmanager
def publication_context(
    style: PublicationStyle = DEFAULT_STYLE,
) -> Iterator[None]:
    """Apply the LUMO publication typography without changing global defaults."""

    with mpl.rc_context(rc=style.rc_params()):
        yield
