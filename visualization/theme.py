"""Publication figure themes and independent scale policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization.data import ScientificFigureError


@dataclass(frozen=True)
class FigureTheme:
    """Central publication style with journal and blog presets."""

    name: str
    font_family: str
    base_font_size: float
    title_size: float
    axis_label_size: float
    panel_label_size: float
    line_width: float
    marker_size: float
    signed_colormap: str
    magnitude_colormap: str
    invalid_color: str
    mesh_color: str
    observation_colors: Mapping[str, str]
    raster_dpi: int
    pdf_font_type: int

    @classmethod
    def preset(cls, name: str) -> "FigureTheme":
        if name == "journal":
            return cls(
                name="journal",
                font_family="DejaVu Sans",
                base_font_size=8.5,
                title_size=11.0,
                axis_label_size=8.5,
                panel_label_size=11.0,
                line_width=1.4,
                marker_size=4.0,
                signed_colormap="RdBu_r",
                magnitude_colormap="viridis",
                invalid_color="#BDBDBD",
                mesh_color="#7F8C8D",
                observation_colors={"right": "#2166AC", "left": "#B2182B"},
                raster_dpi=300,
                pdf_font_type=42,
            )
        if name == "blog":
            return cls(
                name="blog",
                font_family="DejaVu Sans",
                base_font_size=10.0,
                title_size=14.0,
                axis_label_size=10.0,
                panel_label_size=13.0,
                line_width=2.0,
                marker_size=5.5,
                signed_colormap="RdBu_r",
                magnitude_colormap="viridis",
                invalid_color="#BDBDBD",
                mesh_color="#87939A",
                observation_colors={"right": "#287D91", "left": "#D95F02"},
                raster_dpi=240,
                pdf_font_type=42,
            )
        raise ScientificFigureError(f"unknown theme preset {name!r}")

    def apply(self) -> None:
        plt.rcParams.update(
            {
                "font.family": self.font_family,
                "font.size": self.base_font_size,
                "figure.titlesize": self.title_size,
                "axes.labelsize": self.axis_label_size,
                "pdf.fonttype": self.pdf_font_type,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "savefig.bbox": "tight",
            }
        )


@dataclass(frozen=True)
class ScalePolicy:
    """Independent geometry, vector, and color scaling."""

    deformation_scale: float = 1.0
    arrow_scale: float = 1.0
    arrow_minimum_mm: float = 0.0
    color_limits: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.deformation_scale)
            or self.deformation_scale <= 0.0
            or not math.isfinite(self.arrow_scale)
            or self.arrow_scale <= 0.0
            or not math.isfinite(self.arrow_minimum_mm)
            or self.arrow_minimum_mm < 0.0
        ):
            raise ScientificFigureError("scale policy values are invalid")
        if self.color_limits is not None:
            low, high = self.color_limits
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ScientificFigureError("color limits are invalid")
