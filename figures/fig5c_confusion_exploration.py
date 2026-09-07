"""Render the condition-grouped Figure 5(c) confusion-matrix exploration."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.analysis.fig5c_decoder import DEFAULT_CONFIG_PATH, load_config
from experiments.analysis.plot_fig5c import save_panel


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
