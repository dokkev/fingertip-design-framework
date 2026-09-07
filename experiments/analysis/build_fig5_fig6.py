"""Build the shared analysis and render final Figures 5 and 6."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.analysis.fig5c_decoder import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config,
    run_analysis,
)
from experiments.analysis.plot_fig5c import save_panel  # noqa: E402
from figures.fig5.fig5 import save_final as save_figure5  # noqa: E402
from figures.fig6.fig6 import save_final as save_figure6  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--recompute", action="store_true")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    summary = config.analysis_output_directory / "fig5c_accuracy_summary.csv"
    if arguments.recompute or not summary.is_file():
        for path in run_analysis(config).values():
            print(path)
    for path in save_panel(config):
        print(path)
    for path in save_figure5(config):
        print(path)
    for path in save_figure6(config):
        print(path)


if __name__ == "__main__":
    main()
