"""Compose paper Figure 5 from raw and compact physical measurements."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from lumo.visualization import DEFAULT_STYLE, publication_context, save_figure  # noqa: E402

from .config import FIGURE_DIRECTORY  # noqa: E402
from .fig5a import render_panel as render_panel_a  # noqa: E402
from .fig5b import render_panel as render_panel_b  # noqa: E402
from .fig5c import render_panel as render_panel_c  # noqa: E402


FIGURE_SIZE_IN = (DEFAULT_STYLE.double_column_width_in, 4.25)


def build_figure() -> plt.Figure:
    """Build the complete Figure 5 at exact IEEE double-column width."""

    figure = plt.figure(figsize=FIGURE_SIZE_IN)
    outer = figure.add_gridspec(
        1,
        3,
        left=0.008,
        right=0.992,
        bottom=0.065,
        top=0.994,
        width_ratios=(0.37, 0.41, 0.22),
        wspace=0.025,
    )
    render_panel_a(figure, outer[0], panel_label="(a)")
    render_panel_b(figure, outer[1], panel_label="(b)")
    render_panel_c(figure, outer[2], panel_label="(c)")
    return figure


def main() -> None:
    """Render standalone panels and the final vector-native composition."""

    from .fig5a import main as render_a
    from .fig5b import main as render_b
    from .fig5c import main as render_c

    render_a()
    render_b()
    render_c()
    with publication_context(DEFAULT_STYLE):
        figure = build_figure()
        save_figure(
            figure,
            FIGURE_DIRECTORY / "fig5",
            formats=("pdf", "png"),
            bbox_inches=None,
            pad_inches=0.0,
        )
        plt.close(figure)


if __name__ == "__main__":
    main()
