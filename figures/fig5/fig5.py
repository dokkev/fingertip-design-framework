"""Compose paper Figure 5 from raw and compact physical measurements."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from experiments.analysis.fig5c_decoder import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    PaperFigureConfig,
    load_config,
    read_csv,
    run_analysis,
)
from lumo.visualization import DEFAULT_STYLE, publication_context, save_figure  # noqa: E402

from .fig5a import render_panel as render_panel_a  # noqa: E402
from .fig5b import render_panel as render_panel_b  # noqa: E402
from .fig5c import build_confusion_matrices  # noqa: E402
from .fig5c import render_panel as render_panel_c  # noqa: E402
from .config import FIG5_MATERIAL_LABELS  # noqa: E402


FIGURE_SIZE_IN = (DEFAULT_STYLE.double_column_width_in, 4.35)


def build_figure(
    config: PaperFigureConfig,
    confusion_matrices: dict[tuple[str, str, str], object],
) -> plt.Figure:
    """Build the complete Figure 5 at exact IEEE double-column width."""

    figure = plt.figure(figsize=FIGURE_SIZE_IN)
    outer = figure.add_gridspec(
        1,
        5,
        left=0.008,
        right=0.992,
        bottom=0.090,
        top=0.996,
        width_ratios=(0.380, 0.010, 0.290, 0.010, 0.330),
        wspace=0.0,
    )
    render_panel_a(figure, outer[0], config=config, panel_label="(a)")
    render_panel_b(
        figure,
        outer[2],
        panel_label="(b)",
    )
    render_panel_c(
        figure,
        outer[4],
        config=config,
        matrices=confusion_matrices,
        panel_label="(c)",
    )
    return figure


def _save_composition(
    config: PaperFigureConfig,
    *,
    output_stem: str,
    recompute: bool = False,
) -> tuple[Path, ...]:
    """Render one vector-native Figure 5 composition."""

    predictions_path = (
        config.analysis_output_directory / "fig5c_per_sample_predictions.csv"
    )
    if recompute or not predictions_path.is_file():
        run_analysis(config)
    decoder_predictions = read_csv(predictions_path)
    confusion_matrices = build_confusion_matrices(config, decoder_predictions)
    config.figure5_output_directory.mkdir(parents=True, exist_ok=True)
    figure_config = replace(config, material_labels=FIG5_MATERIAL_LABELS)

    with publication_context(DEFAULT_STYLE):
        figure = build_figure(figure_config, confusion_matrices)
        outputs = save_figure(
            figure,
            config.figure5_output_directory / output_stem,
            formats=("pdf", "png"),
            bbox_inches=None,
            pad_inches=0.0,
        )
        plt.close(figure)
    return outputs


def save_final(
    config: PaperFigureConfig,
    *,
    recompute: bool = False,
) -> tuple[Path, ...]:
    """Render the canonical final Figure 5 PDF and PNG."""

    return _save_composition(config, output_stem="fig5", recompute=recompute)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--recompute", action="store_true")
    arguments = parser.parse_args()
    for path in save_final(
        load_config(arguments.config), recompute=arguments.recompute
    ):
        print(path)


if __name__ == "__main__":
    main()
