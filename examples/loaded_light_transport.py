"""Show transport through a small synthetic pad deformation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .bootstrap import ensure_repository_root
else:
    from bootstrap import ensure_repository_root

ensure_repository_root()

from model import Fingertip, FingertipParameters
from optics import TraceSettings, trace
from visualization import plot_transport


def main() -> int:
    tip = Fingertip(FingertipParameters())
    mesh = tip.mesh()
    displacement = np.zeros_like(mesh.coordinates)
    cutout_bottom = mesh.boundary_node_indices_for("pad_cutout_bottom")
    displacement[cutout_bottom, 1] = -0.05

    loaded = trace(
        tip,
        mesh.deformed(displacement),
        TraceSettings(ray_count=81, grid_width=120, grid_height=120),
    )
    plot_transport(loaded, title="Synthetic loaded light transport")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
