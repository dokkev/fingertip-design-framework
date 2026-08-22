"""Show the analytic carrier-silicone bond in the XZ cross-section."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from lumo.fingertip import Fingertip, FingertipParameters


def _silicone_boundary(fingertip: Fingertip) -> np.ndarray:
    silicone = fingertip.silicone
    carrier_boundary_left = (
        (silicone.cavity_left_x_mm, 0.0),
        (silicone.bond_left_inner_x_mm, 0.0),
        (silicone.bond_left_inner_x_mm, silicone.bond_top_z_mm),
        (-silicone.half_width_mm, silicone.bond_top_z_mm),
    )
    carrier_boundary_right = (
        (silicone.half_width_mm, silicone.bond_top_z_mm),
        (silicone.bond_right_inner_x_mm, silicone.bond_top_z_mm),
        (silicone.bond_right_inner_x_mm, 0.0),
        (silicone.cavity_right_x_mm, 0.0),
    )
    ellipse_angle = np.linspace(0.0, np.pi, 181)
    ellipse = np.column_stack(
        (
            silicone.ellipse_radius_x_mm * np.cos(ellipse_angle),
            silicone.ellipse_center_z_mm
            - silicone.ellipse_radius_z_mm * np.sin(ellipse_angle),
        )
    )

    return np.vstack(
        (
            silicone.void_left,
            silicone.void_bottom[1:],
            silicone.void_right[1:],
            carrier_boundary_right[::-1][1:],
            silicone.outer_right[:1],
            ellipse[1:],
            silicone.outer_left[:1],
            carrier_boundary_left[::-1][1:],
        )
    )


def main() -> None:
    fingertip = Fingertip(FingertipParameters())
    silicone_boundary = _silicone_boundary(fingertip)
    carrier_boundary = np.asarray(
        fingertip.carrier.cross_section,
        dtype=np.float64,
    )

    _, axes = plt.subplots(figsize=(10.0, 6.0), constrained_layout=True)
    axes.add_patch(
        Polygon(
            silicone_boundary,
            closed=True,
            facecolor="#8ecae6",
            edgecolor="#126782",
            linewidth=1.5,
            label="silicone",
        )
    )
    axes.add_patch(
        Polygon(
            carrier_boundary,
            closed=True,
            facecolor="#adb5bd",
            edgecolor="#343a40",
            linewidth=1.5,
            label="carrier",
        )
    )

    for index, bond in enumerate(
        (
            fingertip.bonding_interface.left,
            fingertip.bonding_interface.right,
        )
    ):
        bond_points = np.asarray(bond, dtype=np.float64)
        axes.plot(
            bond_points[:, 0],
            bond_points[:, 1],
            color="#d00000",
            linewidth=5.0,
            solid_capstyle="round",
            label="perfect kinematic bond" if index == 0 else None,
            zorder=3,
        )

    axes.set_title("LUMO carrier-silicone bond")
    axes.set_xlabel("X [mm]")
    axes.set_ylabel("Z [mm]")
    axes.set_aspect("equal", adjustable="box")
    axes.autoscale_view()
    axes.margins(0.08)
    axes.grid(alpha=0.25)
    axes.legend(loc="lower left")
    plt.show()


if __name__ == "__main__":
    main()
