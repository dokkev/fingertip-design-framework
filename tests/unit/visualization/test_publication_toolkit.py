"""Functional checks for standalone and composed publication rendering."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lumo.visualization import (  # noqa: E402
    add_panel_labels,
    create_figure,
    create_gridspec,
    plot_force_displacement,
    plot_image,
    plot_optical_response,
    plot_pareto_small_multiple,
    render_panel,
    save_figure,
)
from matplotlib.colors import Normalize  # noqa: E402


def test_standalone_panel_and_uniform_composition_export(tmp_path) -> None:
    force_n = np.array((1.0, 2.0, 5.0, 10.0))
    indentation_mm = np.array((0.2, 0.4, 0.9, 1.4))

    standalone, axes, artists = render_panel(
        plot_force_displacement,
        indentation_mm,
        force_n,
    )
    assert axes.figure is standalone
    assert len(artists) == 1
    standalone_paths = save_figure(
        standalone,
        tmp_path / "standalone",
        formats=("pdf", "png"),
    )
    plt.close(standalone)

    figure, axes_grid = create_figure(2, 2, width="double")
    assert axes_grid.shape == (2, 2)
    for axes_item in axes_grid.flat:
        plot_optical_response(axes_item, force_n, force_n[:, None] / 10.0)
    labels = add_panel_labels(axes_grid)
    assert [label.get_text() for label in labels] == ["(a)", "(b)", "(c)", "(d)"]
    composed_paths = save_figure(
        figure,
        tmp_path / "composed",
        formats=("pdf", "png"),
    )
    plt.close(figure)

    for output_path in (*standalone_paths, *composed_paths):
        assert output_path.is_file()
        assert output_path.stat().st_size > 0


def test_nonuniform_gridspec_and_cropped_image() -> None:
    figure, grid = create_gridspec(
        2,
        2,
        width="double",
        width_ratios=(2.0, 1.0),
    )
    wide_axes = figure.add_subplot(grid[:, 0])
    image_axes = figure.add_subplot(grid[0, 1])
    image = np.ones((20, 30, 3), dtype=np.float64)
    image[5:15, 8:22] = 0.25

    plot_force_displacement(wide_axes, (0.0, 1.0), (0.0, 2.0))
    artist = plot_image(image_axes, image, crop_whitespace=True)

    assert artist.get_array().shape == (10, 14, 3)
    assert not image_axes.axison
    plt.close(figure)


def test_empirical_pareto_panel_marks_front_and_balanced_design() -> None:
    figure, axes = plt.subplots()
    points = plot_pareto_small_multiple(
        axes,
        j_contact=(0.30, 0.40, 0.45, 0.35),
        j_obs=(0.50e-3, 0.45e-3, 0.30e-3, 0.35e-3),
        void_width_mm=(0.0, 2.0, 5.0, 1.0),
        pareto_mask=(True, True, True, False),
        balanced_index=1,
        colormap=plt.get_cmap("BuPu"),
        normalization=Normalize(0.0, 7.5),
    )

    assert points.get_offsets().shape == (3, 2)
    assert len(axes.collections) == 3
    assert len(axes.lines) == 1
    plt.close(figure)
