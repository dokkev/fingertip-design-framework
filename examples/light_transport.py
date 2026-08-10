"""Show deterministic reference light transport in the fingertip."""

from __future__ import annotations

import matplotlib.pyplot as plt

if __package__:
    from .bootstrap import ensure_repository_root
else:
    from bootstrap import ensure_repository_root

ensure_repository_root()

from model import Fingertip, FingertipParameters
from optics import trace
from visualization import plot_transport


def main() -> int:
    tip = Fingertip(FingertipParameters(void_width=1.0, void_height=2.0))
    mesh = tip.mesh()
    result = trace(tip, mesh)
    plot_transport(result, title="Reference qualitative light transport")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
