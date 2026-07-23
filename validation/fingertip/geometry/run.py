"""Generate the limiting cases and parameter sweep for LIT pad clearance."""

from __future__ import annotations

import argparse
from dataclasses import replace
from math import isclose
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters
from visualization.geometry import plot_fingertip


def _four_cases() -> list[tuple[str, FingertipParameters]]:
    base = FingertipParameters()
    side_gap = 2.5
    bottom_gap = 3.0
    return [
        (r"Zero-clearance fit: $w_v=0, h_v=0$", base),
        (
            rf"Side clearance: $w_v={side_gap:g}, h_v=0$",
            replace(base, void_width=side_gap),
        ),
        (
            rf"Bottom clearance: $w_v=0, h_v={bottom_gap:g}$",
            replace(base, void_height=bottom_gap),
        ),
        (
            rf"U-clearance: $w_v={side_gap:g}, h_v={bottom_gap:g}$",
            replace(base, void_width=side_gap, void_height=bottom_gap),
        ),
    ]


def run_sanity_checks() -> None:
    """Verify the four limiting clearance areas before plotting."""
    base = FingertipParameters()
    side_gap = 2.5
    bottom_gap = 3.0
    expected_areas = [
        0.0,
        2.0 * side_gap * base.stem_height,
        base.stem_width * bottom_gap,
        (base.stem_width + 2.0 * side_gap) * (base.stem_height + bottom_gap)
        - base.stem_width * base.stem_height,
    ]

    headings = ("w_v", "h_v", "classification", "void area", "bond length")
    print("  ".join(f"{heading:>16}" for heading in headings))
    print("  ".join("-" * 16 for _ in headings))
    for (_, parameters), expected_area in zip(
        _four_cases(), expected_areas, strict=True
    ):
        model = FingertipModel(parameters)
        model.validate_geometry()
        actual_area = 0.0 if model.void_geometry is None else model.void_geometry.area
        if not isclose(actual_area, expected_area, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(
                f"void area mismatch: got {actual_area}, expected {expected_area}"
            )
        print(
            f"{parameters.void_width:16g}  {parameters.void_height:16g}  "
            f"{model.classify_void():>16}  {actual_area:16g}  "
            f"{model.pad_link_connection_length():16g}"
        )


def make_four_case_figure(output_directory: Path) -> Path:
    """Save the four limiting combinations of side and bottom clearance."""
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), constrained_layout=True)
    for axis, (title, parameters) in zip(axes.flat, _four_cases(), strict=True):
        plot_fingertip(
            FingertipModel(parameters),
            ax=axis,
            show_dimensions=True,
            show_axes=False,
            show_legend=False,
            title=title,
        )

    handles, labels = axes.flat[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "LIT pad void parameterization — four limiting cases",
        fontsize=15,
        weight="bold",
    )
    path = output_directory / "lit_pad_void_four_cases.png"
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def make_parameter_grid(output_directory: Path) -> Path:
    """Save a 3x3 sweep of side and bottom clearance dimensions."""
    base = FingertipParameters()
    widths = [0.0, 1.5, 3.0]
    heights = [0.0, 2.0, 4.0]
    figure, axes = plt.subplots(3, 3, figsize=(12.5, 10.5), constrained_layout=True)

    for row, void_height in enumerate(heights):
        for column, void_width in enumerate(widths):
            parameters = replace(
                base,
                void_width=void_width,
                void_height=void_height,
            )
            plot_fingertip(
                FingertipModel(parameters),
                ax=axes[row, column],
                show_dimensions=True,
                show_axes=False,
                show_legend=False,
                title=rf"$w_v={void_width:g},\ h_v={void_height:g}$",
            )

    figure.suptitle(
        r"Geometry sweep: columns vary $w_v$, rows vary $h_v$",
        fontsize=15,
        weight="bold",
    )
    path = output_directory / "lit_pad_void_parameter_grid.png"
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("output/validation/fingertip/geometry"),
    )
    return parser.parse_args()


def main() -> None:
    """Run analytic checks and generate both reference figures under output/."""
    arguments = parse_arguments()
    run_sanity_checks()
    output_directory = arguments.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    for path in (
        make_four_case_figure(output_directory),
        make_parameter_grid(output_directory),
    ):
        print(path)


if __name__ == "__main__":
    main()
