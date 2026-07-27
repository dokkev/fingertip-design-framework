"""Render the parametric fingertip geometry with Matplotlib.

The script can be run from any working directory:

    /home/dk/miniconda3/envs/lit/bin/python \
      /home/dk/workspace/lit_ws/examples/fingertip_visualize.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

if __package__:
    from .bootstrap import ensure_repository_root
else:
    from bootstrap import ensure_repository_root

ensure_repository_root()

from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters
from visualization.geometry import plot_fingertip


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one parametric LIT Hand fingertip cross-section."
    )
    parser.add_argument(
        "--void-width",
        type=float,
        default=0.0,
        help="Clearance on each side of the stem in mm.",
    )
    parser.add_argument(
        "--void-height",
        type=float,
        default=0.0,
        help="Clearance below the stem in mm.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    parameters = FingertipParameters(
        void_width=arguments.void_width,
        void_height=arguments.void_height,
    )
    model = FingertipModel(parameters)

    figure, axis = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    plot_fingertip(
        model,
        ax=axis,
        show_void=True,
        show_interface=True,
        show_contact_boundaries=True,
        title=(
            "Parametric fingertip"
            f"  ($w_v={parameters.void_width:g}$ mm,"
            f" $h_v={parameters.void_height:g}$ mm)"
        ),
    )

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
