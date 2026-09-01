"""Render standalone and composed panels with the publication toolkit."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

from lumo.visualization import (  # noqa: E402
    DEFAULT_STYLE,
    add_panel_labels,
    create_figure,
    plot_contact_area,
    plot_force_displacement,
    plot_image,
    plot_incremental_stiffness,
    plot_optical_response,
    plot_pareto,
    publication_context,
    render_panel,
    save_figure,
)


_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "validation"
    / "publication_toolkit_demo"
)


def _schematic_image() -> np.ndarray:
    image = np.ones((100, 180, 3), dtype=np.float64)
    image[18:78, 10:170] = to_rgb(DEFAULT_STYLE.colors.silicone)
    image[18:31, 10:170] = to_rgb(DEFAULT_STYLE.colors.carrier)
    for center in (25, 58, 90, 122, 155):
        image[36:42, center - 4 : center + 5] = to_rgb(DEFAULT_STYLE.colors.optical)
    return image


def main() -> None:
    force_n = np.array((1.0, 2.0, 5.0, 10.0))
    indentation_mm = np.array((0.35, 0.58, 1.12, 1.68))
    contact_area_mm2 = np.array((7.0, 15.0, 21.0, 24.0))
    stiffness_n_mm = np.array((4.3, 5.2, 8.9, 13.4))
    optical_response = np.column_stack(
        (
            (0.012, 0.019, 0.031, 0.044),
            (0.009, 0.017, 0.029, 0.041),
        )
    )

    standalone, _, _ = render_panel(
        plot_force_displacement,
        indentation_mm,
        force_n,
        label="Nominal",
    )
    standalone_paths = save_figure(
        standalone,
        _OUTPUT_DIRECTORY / "standalone_force_displacement",
    )
    plt.close(standalone)

    with publication_context():
        figure, axes = create_figure(2, 3, width="double", panel_aspect=0.72)
        plot_pareto(
            axes[0, 0],
            (0.39, 0.43, 0.47),
            (0.00008, 0.00015, 0.00011),
            material="dragon_skin",
            status=("nominal", "optimized", "suboptimal"),
            pareto_mask=(False, True, True),
            label="Dragon Skin",
        )
        plot_pareto(
            axes[0, 0],
            (0.41, 0.46, 0.48),
            (0.00012, 0.00026, 0.00017),
            material="solaris",
            status=("nominal", "optimized", "suboptimal"),
            pareto_mask=(False, True, True),
            label="Solaris",
        )
        plot_force_displacement(axes[0, 1], indentation_mm, force_n)
        plot_contact_area(axes[0, 2], force_n, contact_area_mm2)
        plot_incremental_stiffness(axes[1, 0], force_n, stiffness_n_mm)
        plot_optical_response(
            axes[1, 1],
            force_n,
            optical_response,
            labels=("left ROI", "right ROI"),
        )
        plot_image(axes[1, 2], _schematic_image(), crop_whitespace=True)
        add_panel_labels(axes)

    composed_paths = save_figure(
        figure,
        _OUTPUT_DIRECTORY / "composed_panels",
        formats=("pdf", "svg", "png"),
    )
    plt.close(figure)

    for output_path in (*standalone_paths, *composed_paths):
        print(output_path)


if __name__ == "__main__":
    main()
