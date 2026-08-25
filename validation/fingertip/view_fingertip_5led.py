"""Visualize the full five-LED mesh and its simplified material layout."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lumo.fingertip import Fingertip
from lumo.mesh import (
    MAIN_Y_BOUNDS_MM,
    TOTAL_Y_BOUNDS_MM,
    make_fingertip_5led_mesh,
    make_fingertip_mesh,
)


_OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "validation"
    / "fingertip_5led_mesh.png"
)
_SILICONE_COLOR = "#8ecae6"
_SILICONE_EDGE = "#126782"
_CARRIER_COLOR = "#6c757d"
_VOID_COLOR = "#fff3bf"
_LED_COLOR = "#38b000"


def _silicone_boundary(fingertip: Fingertip) -> np.ndarray:
    silicone = fingertip.silicone
    right_bond = (
        (silicone.cavity_right_x_mm, 0.0),
        (silicone.bond_right_inner_x_mm, 0.0),
        (silicone.bond_right_inner_x_mm, silicone.bond_top_z_mm),
        (silicone.half_width_mm, silicone.bond_top_z_mm),
    )
    left_bond = (
        (-silicone.half_width_mm, silicone.bond_top_z_mm),
        (silicone.bond_left_inner_x_mm, silicone.bond_top_z_mm),
        (silicone.bond_left_inner_x_mm, 0.0),
        (silicone.cavity_left_x_mm, 0.0),
    )
    angle = np.linspace(0.0, np.pi, 181)
    ellipse = np.column_stack(
        (
            silicone.ellipse_radius_x_mm * np.cos(angle),
            silicone.ellipse_center_z_mm
            - silicone.ellipse_radius_z_mm * np.sin(angle),
        )
    )
    return np.vstack(
        (
            silicone.void_left,
            silicone.void_bottom[1:],
            silicone.void_right[1:],
            right_bond[1:],
            silicone.outer_right[:1],
            ellipse[1:],
            silicone.outer_left[:1],
            left_bond[1:],
        )
    )


def _draw_longitudinal(axes: plt.Axes, mesh) -> None:
    fingertip = mesh.fingertip
    silicone = fingertip.silicone
    geometry = fingertip.parameters.geometry
    proximal_y, main_distal_y = MAIN_Y_BOUNDS_MM
    total_distal_y = TOTAL_Y_BOUNDS_MM[1]
    tip_z = silicone.ellipse_center_z_mm - silicone.ellipse_radius_z_mm
    cavity_bottom_z = silicone.cavity_bottom_z_mm

    axes.add_patch(
        Rectangle(
            (proximal_y, tip_z),
            main_distal_y - proximal_y,
            cavity_bottom_z - tip_z,
            facecolor=_SILICONE_COLOR,
            edgecolor=_SILICONE_EDGE,
            label="continuous silicone",
        )
    )
    axes.add_patch(
        Rectangle(
            (proximal_y, cavity_bottom_z),
            main_distal_y - proximal_y,
            -cavity_bottom_z,
            facecolor=_VOID_COLOR,
            edgecolor="#e67700",
            hatch="//",
            label="continuous side void",
        )
    )
    axes.add_patch(
        Rectangle(
            (proximal_y, 0.0),
            main_distal_y - proximal_y,
            geometry.link_thickness_mm,
            facecolor=_CARRIER_COLOR,
            edgecolor="#343a40",
            label="continuous carrier",
        )
    )
    axes.add_patch(
        Rectangle(
            (proximal_y, -geometry.stem_height_mm),
            main_distal_y - proximal_y,
            geometry.stem_height_mm,
            facecolor="#495057",
            edgecolor="#212529",
            label="continuous stem rail",
        )
    )
    axes.add_patch(
        Rectangle(
            (main_distal_y, tip_z),
            total_distal_y - main_distal_y,
            silicone.bond_top_z_mm - tip_z,
            facecolor=_SILICONE_COLOR,
            edgecolor=_SILICONE_EDGE,
            linewidth=1.5,
            label="solid distal end-cap",
        )
    )
    axes.add_patch(
        Rectangle(
            (main_distal_y, silicone.bond_top_z_mm),
            total_distal_y - main_distal_y,
            geometry.link_thickness_mm - silicone.bond_top_z_mm,
            facecolor=_CARRIER_COLOR,
            edgecolor="#343a40",
            linewidth=1.5,
            label="distal dorsal reinforcement",
        )
    )
    centers_mm = 1.0e3 * mesh.led_centers_m
    axes.scatter(
        centers_mm[:, 1],
        centers_mm[:, 2],
        color=_LED_COLOR,
        edgecolor="#1b4332",
        s=55.0,
        zorder=5,
        label="LED references",
    )
    for index, center in enumerate(centers_mm, start=1):
        axes.annotate(
            str(index),
            (center[1], center[2]),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes.axvline(main_distal_y, color="#d00000", linestyle="--", linewidth=1.2)
    axes.text(
        main_distal_y + 0.2,
        geometry.link_thickness_mm + 0.5,
        "solid closure begins",
        color="#d00000",
        fontsize=8,
        rotation=90,
        va="bottom",
    )
    axes.set_title("Longitudinal material layout")
    axes.set_xlabel("Y [mm]  (proximal → distal)")
    axes.set_ylabel("Z [mm]")
    axes.set_xlim(proximal_y - 2.0, total_distal_y + 2.0)
    axes.set_ylim(tip_z - 1.5, geometry.link_thickness_mm + 2.0)
    axes.grid(alpha=0.2)
    axes.legend(fontsize=7, loc="lower left")


def _draw_cross_section(axes: plt.Axes, fingertip: Fingertip) -> None:
    axes.add_patch(
        Polygon(
            _silicone_boundary(fingertip),
            closed=True,
            facecolor=_SILICONE_COLOR,
            edgecolor=_SILICONE_EDGE,
            linewidth=1.3,
        )
    )
    axes.add_patch(
        Polygon(
            np.asarray(fingertip.carrier.cross_section),
            closed=True,
            facecolor=_CARRIER_COLOR,
            edgecolor="#343a40",
            linewidth=1.3,
        )
    )
    axes.scatter(
        [0.0],
        [-fingertip.parameters.geometry.stem_height_mm],
        color=_LED_COLOR,
        edgecolor="#1b4332",
        s=55.0,
        zorder=5,
        label="LED reference",
    )
    axes.set_title("Local XZ morphology at a LED section")
    axes.set_xlabel("X [mm]")
    axes.set_ylabel("Z [mm]")
    axes.set_aspect("equal", adjustable="box")
    axes.autoscale_view()
    axes.margins(0.08)
    axes.grid(alpha=0.2)
    axes.legend(fontsize=8, loc="lower left")


def _draw_material_mesh(axes, mesh) -> None:
    silicone_vertices = 1.0e3 * np.asarray(mesh.silicone.vertices)
    silicone_triangles = np.asarray(mesh.silicone.surface_tri_indices).reshape(-1, 3)
    carrier_vertices = 1.0e3 * np.asarray(mesh.carrier.vertices)
    carrier_triangles = np.asarray(mesh.carrier.indices).reshape(-1, 3)

    axes.add_collection3d(
        Poly3DCollection(
            silicone_vertices[silicone_triangles],
            facecolor=_SILICONE_COLOR,
            edgecolor="none",
            alpha=0.12,
        )
    )
    axes.add_collection3d(
        Poly3DCollection(
            carrier_vertices[carrier_triangles],
            facecolor=_CARRIER_COLOR,
            edgecolor="#343a40",
            linewidth=0.15,
            alpha=0.75,
        )
    )
    centers_mm = 1.0e3 * mesh.led_centers_m
    axes.scatter(
        centers_mm[:, 0],
        centers_mm[:, 1],
        centers_mm[:, 2],
        color=_LED_COLOR,
        edgecolor="#1b4332",
        s=28.0,
        depthshade=False,
    )
    axes.set_xlim(-16.0, 16.0)
    axes.set_ylim(TOTAL_Y_BOUNDS_MM[0] - 2.0, TOTAL_Y_BOUNDS_MM[1] + 2.0)
    axes.set_zlim(
        silicone_vertices[:, 2].min() - 1.0,
        carrier_vertices[:, 2].max() + 1.0,
    )
    axes.set_box_aspect((32.0, 64.0, 26.0))
    axes.view_init(elev=22.0, azim=-38.0)
    axes.set_title("Actual tetrahedral/surface material mesh")
    axes.set_xlabel("X [mm]")
    axes.set_ylabel("Y [mm]")
    axes.set_zlabel("Z [mm]")


def main() -> None:
    fingertip = Fingertip()
    full_mesh = make_fingertip_5led_mesh(fingertip, element_size_mm=1.0)
    cell_mesh = make_fingertip_mesh(
        fingertip,
        extrusion_depth_mm=11.0,
        element_size_mm=1.0,
    )

    figure = plt.figure(figsize=(18.0, 6.5), constrained_layout=True)
    longitudinal_axes = figure.add_subplot(1, 3, 1)
    cross_section_axes = figure.add_subplot(1, 3, 2)
    material_axes = figure.add_subplot(1, 3, 3, projection="3d")
    _draw_longitudinal(longitudinal_axes, full_mesh)
    _draw_cross_section(cross_section_axes, fingertip)
    _draw_material_mesh(material_axes, full_mesh)
    figure.suptitle("Full five-LED LUMO fingertip mesh", fontsize=15)

    full_counts = (
        full_mesh.silicone.vertex_count,
        full_mesh.silicone.tet_count,
        len(full_mesh.silicone.surface_tri_indices) // 3,
    )
    cell_counts = (
        cell_mesh.silicone.vertex_count,
        cell_mesh.silicone.tet_count,
        len(cell_mesh.silicone.surface_tri_indices) // 3,
    )
    labels = ("vertices", "tetrahedra", "surface triangles")
    print("five-LED mesh")
    for label, full_count, cell_count in zip(
        labels,
        full_counts,
        cell_counts,
        strict=True,
    ):
        print(f"{label}: {full_count} (single cell {cell_count}, {full_count / cell_count:.2f}x)")
    print(
        "LED centers Y [mm]: "
        f"{(1.0e3 * full_mesh.led_centers_m[:, 1]).tolist()}"
    )

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(_OUTPUT_PATH, dpi=180)
    print(f"saved: {_OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
