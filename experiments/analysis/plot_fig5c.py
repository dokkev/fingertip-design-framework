"""Render the standalone Figure 5(c) condition table."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from experiments.analysis.fig5c_decoder import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    PaperFigureConfig,
    load_config,
    read_csv,
    run_analysis,
)
from figures.fig5.fig5c import (  # noqa: E402
    build_confusion_matrices,
    render_panel,
)
from lumo.visualization import (  # noqa: E402
    DEFAULT_STYLE,
    publication_context,
    save_figure,
)


def save_panel(
    config: PaperFigureConfig,
    *,
    recompute: bool = False,
    output_directory: Path | None = None,
) -> tuple[Path, ...]:
    """Save the standalone condition-grouped confusion panel."""

    predictions_path = (
        config.analysis_output_directory / "fig5c_per_sample_predictions.csv"
    )
    if recompute or not predictions_path.is_file():
        run_analysis(config)
    matrices = build_confusion_matrices(config, read_csv(predictions_path))
    destination = output_directory or config.figure5_output_directory
    destination.mkdir(parents=True, exist_ok=True)
    with publication_context(DEFAULT_STYLE):
        figure = plt.figure(
            figsize=(DEFAULT_STYLE.double_column_width_in, 4.25),
            facecolor="white",
        )
        grid = figure.add_gridspec(
            1,
            1,
            left=0.075,
            right=0.965,
            bottom=0.055,
            top=0.995,
        )
        render_panel(
            figure,
            grid[0, 0],
            config=config,
            matrices=matrices,
            font_scale=1.15,
            show_row_labels=True,
        )
        outputs = save_figure(
            figure,
            destination / "fig5c_confusion_2x2",
            formats=("pdf", "png"),
            bbox_inches=None,
            pad_inches=0.0,
        )
        plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--recompute", action="store_true")
    arguments = parser.parse_args()
    for path in save_panel(
        load_config(arguments.config),
        recompute=arguments.recompute,
        output_directory=(
            arguments.output_directory.resolve()
            if arguments.output_directory is not None
            else None
        ),
    ):
        print(path)


if __name__ == "__main__":
    main()
