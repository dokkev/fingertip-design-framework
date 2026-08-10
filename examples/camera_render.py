"""Show the reference state with the optional Mitsuba camera validator."""

from __future__ import annotations

import matplotlib.pyplot as plt

if __package__:
    from .bootstrap import ensure_repository_root
else:
    from bootstrap import ensure_repository_root

ensure_repository_root()

from model import Fingertip, FingertipParameters
from optics.mitsuba import MitsubaRenderer
from visualization import plot_camera


def main() -> int:
    tip = Fingertip(FingertipParameters())
    mesh = tip.mesh()
    renderer = MitsubaRenderer(tip, mesh, depth_mm=10.0)
    image = renderer.render()
    plot_camera(image)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
