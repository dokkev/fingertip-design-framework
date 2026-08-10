"""Compare fingertip deformation for three circular object diameters."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from bootstrap import ensure_repository_root

ensure_repository_root()

from fem import IndenterSettings, solve
from model import Fingertip, FingertipParameters
from visualization import plot_displacement


DIAMETERS_MM = (4.0, 8.0, 12.0)
INDENTATION_MM = 1.5


def main() -> int:
    tip = Fingertip(
        FingertipParameters(void_width=1.0, void_height=2.0)
    )
    mesh = tip.mesh()

    results = []
    for diameter in DIAMETERS_MM:
        result = solve(
            tip,
            mesh,
            indentation=INDENTATION_MM,
            indenter=IndenterSettings(radius_mm=diameter / 2.0),
        )
        if not result.converged or result.displacement is None:
            reason = result.details.get("failure_reason", "unknown failure")
            raise RuntimeError(
                f"{diameter:g} mm indentation solve did not converge: {reason}"
            )
        results.append(result)

    shared_max = max(
        float(np.linalg.norm(result.displacement, axis=1).max())
        for result in results
    )
    figure, axes = plt.subplots(
        1,
        len(DIAMETERS_MM),
        figsize=(18.0, 6.0),
        constrained_layout=True,
    )

    for axis, diameter, result in zip(axes, DIAMETERS_MM, results):
        reaction = (
            "n/a"
            if result.reaction_force is None
            else f"{result.reaction_force:.3g} N"
        )
        plot_displacement(
            mesh,
            result.displacement,
            ax=axis,
            normalization_max=shared_max,
            title=f"Ø {diameter:g} mm\nReaction = {reaction}",
        )

    figure.suptitle("Fingertip deformation under circular objects")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
