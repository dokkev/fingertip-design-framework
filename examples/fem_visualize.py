"""Solve one indentation and show its qualitative optical response."""

from __future__ import annotations

import matplotlib.pyplot as plt

if __package__:
    from .bootstrap import ensure_repository_root
else:
    from bootstrap import ensure_repository_root

ensure_repository_root()

from fem import solve
from model import Fingertip, FingertipParameters
from optics import evaluate, trace
from visualization import plot_transport


def main() -> int:
    tip = Fingertip(
        FingertipParameters(void_width=1.0, void_height=2.0)
    )
    mesh = tip.mesh()
    fea = solve(tip, mesh, indentation=1.5)
    if not fea.converged:
        reason = fea.details.get("failure_reason", "unknown failure")
        raise RuntimeError(f"indentation solve did not converge: {reason}")

    reference = trace(tip, mesh)
    loaded = trace(tip, fea.deformed_mesh)
    metrics = evaluate(reference, loaded)

    print(f"reaction_force_n={fea.reaction_force}")
    for name, value in metrics.items():
        print(f"{name}={value:.8g}")
    plot_transport(loaded, title="Loaded qualitative light transport")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
