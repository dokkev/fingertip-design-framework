"""Show how mechanical deformation changes deterministic light transport."""

from __future__ import annotations

import matplotlib.pyplot as plt

from bootstrap import ensure_repository_root

ensure_repository_root()

from fem import IndenterSettings, solve
from model import Fingertip, FingertipParameters
from optics import evaluate, trace
from visualization import plot_displacement, plot_transport


OBJECT_DIAMETER_MM = 8.0
INDENTATION_MM = 1.5


def main() -> int:
    tip = Fingertip(
        FingertipParameters(
            void_width=1.0,
            void_height=2.0,
            young_modulus_mpa=1.0,
            poisson_ratio=0.49,
        )
    )
    mesh = tip.mesh()

    fea = solve(
        tip,
        mesh,
        indentation=INDENTATION_MM,
        indenter=IndenterSettings(radius_mm=OBJECT_DIAMETER_MM / 2.0),
    )
    if not fea.converged or fea.displacement is None:
        reason = fea.details.get("failure_reason", "unknown failure")
        raise RuntimeError(f"indentation solve did not converge: {reason}")

    reference = trace(tip, mesh)
    loaded = trace(tip, fea.deformed_mesh)
    metrics = evaluate(reference, loaded)

    print("Optical transport change:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4g}")

    shared_light_max = max(
        float(reference.density.max()),
        float(loaded.density.max()),
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(19.0, 6.0),
        constrained_layout=True,
    )
    plot_displacement(
        mesh,
        fea.displacement,
        ax=axes[0],
        title="FEA deformation",
    )
    plot_transport(
        reference,
        ax=axes[1],
        normalization_max=shared_light_max,
        title="Reference light",
    )
    plot_transport(
        loaded,
        ax=axes[2],
        normalization_max=shared_light_max,
        title="Loaded light",
    )
    figure.suptitle("Mechanical deformation → optical transport")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
