"""Show how mechanical deformation changes deterministic light transport."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from bootstrap import ensure_repository_root

ensure_repository_root()

from fem import IndenterSettings, solve
from model import Fingertip, FingertipParameters
from optics.transport3d import Transport3DSettings, trace_3d
from visualization import plot_fea, plot_transport
from visualization.optics import shared_optical_normalization


OBJECT_DIAMETER_MM = 8.0
INDENTATION_MM = 1.5


def main() -> int:
    tip = Fingertip(
        FingertipParameters(
            void_width=1.0,
            # Production morphology is the bonded zero-clearance case.
            void_height=0.0,
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

    settings = Transport3DSettings(mode="planar", retain_projected_segments=True)
    reference = trace_3d(tip, mesh, settings=settings)
    loaded = trace_3d(
        tip,
        fea.deformed_mesh,
        reference_mesh=mesh,
        settings=settings,
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(19.0, 6.0),
        constrained_layout=True,
    )
    optical_norm, optical_cmap = shared_optical_normalization((reference, loaded))
    plot_fea(
        mesh,
        fea.displacement,
        ax=axes[0],
        title="FEA deformation",
    )
    plot_transport(
        reference,
        ax=axes[1],
        norm=optical_norm,
        title="Reference light",
    )
    plot_transport(
        loaded,
        ax=axes[2],
        norm=optical_norm,
        title="Loaded light",
    )
    optical_mappable = ScalarMappable(norm=optical_norm, cmap=optical_cmap)
    optical_mappable.set_array([])
    figure.colorbar(
        optical_mappable,
        ax=axes[1:].tolist(),
        fraction=0.046,
        pad=0.04,
    ).set_label("Weighted optical path density")
    figure.suptitle("Mechanical deformation → optical transport")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
