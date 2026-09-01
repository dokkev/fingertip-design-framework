"""Export the current parametric LUMO fingertip cross-section."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from lumo.fingertip import Fingertip  # noqa: E402
from lumo.visualization import (  # noqa: E402
    plot_fingertip_parameterization,
    render_panel,
    save_figure,
)


_OUTPUT_STEM = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "publication"
    / "fingertip_parameterization"
)


def main() -> None:
    figure, _, _ = render_panel(
        plot_fingertip_parameterization,
        Fingertip(),
        width="double",
        panel_aspect=0.72,
    )
    output_paths = save_figure(
        figure,
        _OUTPUT_STEM,
        formats=("pdf", "svg", "png"),
    )
    plt.close(figure)
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
