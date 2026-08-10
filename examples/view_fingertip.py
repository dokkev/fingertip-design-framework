"""Define and inspect a fingertip design."""

from __future__ import annotations

import matplotlib.pyplot as plt

from bootstrap import ensure_repository_root

ensure_repository_root()

from model import Fingertip, FingertipParameters
from visualization import plot_fingertip


def main() -> int:
    # Change these dimensions to explore the fingertip geometry.
    parameters = FingertipParameters(
        semielliptical_pad_width=20.0,
        semielliptical_pad_height=7.0,
        stem_width=7.6,
        stem_height=6.0,
    )
    tip = Fingertip(parameters)

    plot_fingertip(tip)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
