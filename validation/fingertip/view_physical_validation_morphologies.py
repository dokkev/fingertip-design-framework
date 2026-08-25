"""Compare the six selected physical-validation fingertip morphologies."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

from lumo.fingertip import Fingertip, FingertipGeometry, FingertipParameters


_OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "validation"
    / "physical_validation_morphologies.png"
)
_SILICONE_FACE_COLOR = "#8ecae6"
_SILICONE_EDGE_COLOR = "#126782"
_CARRIER_FACE_COLOR = "#adb5bd"
_CARRIER_EDGE_COLOR = "#343a40"
_BOND_COLOR = "#d00000"


def _geometry(
    flat_pad_height_mm: float,
    semiellipse_height_mm: float,
    stem_width_mm: float,
    stem_height_mm: float,
    void_width_mm: float,
    void_height_mm: float,
) -> FingertipGeometry:
    return replace(
        FingertipGeometry(),
        flat_pad_height_mm=flat_pad_height_mm,
        semiellipse_height_mm=semiellipse_height_mm,
        stem_width_mm=stem_width_mm,
        stem_height_mm=stem_height_mm,
        void_width_mm=void_width_mm,
        void_height_mm=void_height_mm,
    )


_MORPHOLOGIES = (
    (
        "Dragon Skin",
        (
            (
                "Optimized",
                "Trial 123",
                _geometry(13.5, 14.0, 7.0, 4.0, 3.0, 5.0),
            ),
            (
                "Jointly suboptimal",
                "Trial 126",
                _geometry(14.0, 13.5, 7.0, 4.0, 3.5, 5.0),
            ),
            (
                "Nominal",
                "Hand-designed baseline",
                _geometry(5.0, 9.0, 7.6, 6.0, 2.0, 0.0),
            ),
        ),
    ),
    (
        "Solaris",
        (
            (
                "Optimized",
                "Trial 107",
                _geometry(20.0, 5.5, 13.0, 2.5, 0.5, 4.5),
            ),
            (
                "Jointly suboptimal",
                "Trial 94",
                _geometry(20.0, 6.0, 13.0, 2.0, 0.5, 4.5),
            ),
            (
                "Nominal",
                "Hand-designed baseline",
                _geometry(5.0, 9.0, 7.6, 6.0, 2.0, 0.0),
            ),
        ),
    ),
)


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


def _draw_fingertip(
    axes: plt.Axes,
    fingertip: Fingertip,
    *,
    selection: str,
    source: str,
) -> None:
    axes.add_patch(
        Polygon(
            _silicone_boundary(fingertip),
            closed=True,
            facecolor=_SILICONE_FACE_COLOR,
            edgecolor=_SILICONE_EDGE_COLOR,
            linewidth=1.25,
        )
    )
    axes.add_patch(
        Polygon(
            np.asarray(fingertip.carrier.cross_section, dtype=np.float64),
            closed=True,
            facecolor=_CARRIER_FACE_COLOR,
            edgecolor=_CARRIER_EDGE_COLOR,
            linewidth=1.25,
        )
    )
    for bond in (
        fingertip.bonding_interface.left,
        fingertip.bonding_interface.right,
    ):
        bond_points = np.asarray(bond, dtype=np.float64)
        axes.plot(
            bond_points[:, 0],
            bond_points[:, 1],
            color=_BOND_COLOR,
            linewidth=3.0,
            solid_capstyle="round",
            zorder=3,
        )

    geometry = fingertip.parameters.geometry
    geometry_vector = (
        geometry.flat_pad_height_mm,
        geometry.semiellipse_height_mm,
        geometry.stem_width_mm,
        geometry.stem_height_mm,
        geometry.void_width_mm,
        geometry.void_height_mm,
    )
    axes.set_title(f"{selection}\n{source}", fontsize=11)
    axes.text(
        0.5,
        0.025,
        f"{list(geometry_vector)} mm",
        transform=axes.transAxes,
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    axes.set_aspect("equal", adjustable="box")
    axes.grid(alpha=0.2)


def main() -> None:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14.0, 10.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    geometries = []
    for row_index, (material, selections) in enumerate(_MORPHOLOGIES):
        for column_index, (selection, source, geometry) in enumerate(selections):
            geometries.append(geometry)
            fingertip = Fingertip(FingertipParameters(geometry=geometry))
            _draw_fingertip(
                axes[row_index, column_index],
                fingertip,
                selection=selection,
                source=source,
            )
        axes[row_index, 0].set_ylabel(f"{material}\nZ [mm]")

    half_width_limit = max(
        0.5 * geometry.flat_pad_width_mm for geometry in geometries
    )
    minimum_z = min(
        -geometry.flat_pad_height_mm - geometry.semiellipse_height_mm
        for geometry in geometries
    )
    maximum_z = max(geometry.link_thickness_mm for geometry in geometries)
    for subplot in axes.flat:
        subplot.set_xlim(-half_width_limit - 1.0, half_width_limit + 1.0)
        subplot.set_ylim(minimum_z - 1.0, maximum_z + 1.0)
    for subplot in axes[-1]:
        subplot.set_xlabel("X [mm]")

    figure.suptitle(
        "Selected physical-validation fingertip morphologies",
        fontsize=15,
    )
    figure.legend(
        handles=(
            Patch(
                facecolor=_SILICONE_FACE_COLOR,
                edgecolor=_SILICONE_EDGE_COLOR,
                label="silicone",
            ),
            Patch(
                facecolor=_CARRIER_FACE_COLOR,
                edgecolor=_CARRIER_EDGE_COLOR,
                label="carrier",
            ),
            Line2D(
                (0,),
                (0,),
                color=_BOND_COLOR,
                linewidth=3.0,
                label="perfect kinematic bond",
            ),
        ),
        loc="outside lower center",
        ncols=3,
    )

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(_OUTPUT_PATH, dpi=200)
    print(f"saved: {_OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
