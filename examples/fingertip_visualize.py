"""Render the parametric fingertip geometry with Matplotlib.

The script can be run from any working directory:

    /home/dk/miniconda3/envs/lit/bin/python \
      /home/dk/workspace/lit_ws/examples/fingertip_visualize.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt

if __package__:
    from .bootstrap import ensure_repository_root
else:
    from bootstrap import ensure_repository_root

ensure_repository_root()

from model import Fingertip, FingertipParameters
from visualization import plot_fingertip


def main() -> int:
    parameters = FingertipParameters(
        vertical_pad_width=20.0,
        vertical_pad_height=3.0,
        semielliptical_pad_width=20.0,
        semielliptical_pad_height=7.0,
        link_thickness=3.5,
        stem_width=7.6,
        stem_height=6.0,
        void_width=0.0,
        void_height=0.0,
        bonded=True,
        arc_resolution=128,
        geometry_tolerance=1e-9,
    )
    tip = Fingertip(parameters)

    figure, axis = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    plot_fingertip(
        tip,
        ax=axis,
        show_void=True,
        show_interface=True,
        show_contact_boundaries=True,
 
    )

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
