"""Matplotlib rendering for solver-independent fingertip meshes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D

from fem.mesh_types import FingertipMesh

PAD_FACE = "#9ED7E5"
CARRIER_FACE = "#747B84"
MESH_EDGE = "#56616A"

BOUNDARY_STYLE = {
    "pad_bond_left": ("#111111", "-", 2.4),
    "pad_bond_right": ("#111111", "-", 2.4),
    "pad_cutout_left": ("#D95F02", "-", 2.7),
    "pad_cutout_right": ("#E67E22", "-", 2.7),
    "pad_cutout_bottom": ("#F39C12", "-", 2.7),
    "stem_left": ("#542788", "--", 2.2),
    "stem_right": ("#8073AC", "--", 2.2),
    "stem_bottom": ("#B2ABD2", "--", 2.2),
    "pad_outer_arc": ("#287D91", "-", 2.0),
    "pad_void_unpaired": ("#C9473D", ":", 2.5),
    "rigid_link_outer": ("#2D3339", "-", 1.8),
    "rigid_bond_interface": ("#4D4D4D", ":", 2.5),
}


def save_mesh_figure(mesh: FingertipMesh, output_path: str | Path) -> Path:
    """Write a domain- and semantic-tagged T3 mesh PNG."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11.0, 8.5))

    for elements, face_color, label in (
        (mesh.pad_elements, PAD_FACE, "Deformable pad T3"),
        (mesh.carrier_elements, CARRIER_FACE, "Rigid carrier T3"),
    ):
        polygons = [
            [
                (mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm)
                for node_id in element.node_ids
            ]
            for element in elements
        ]
        axis.add_collection(
            PolyCollection(
                polygons,
                facecolors=face_color,
                edgecolors=MESH_EDGE,
                linewidths=0.16,
                alpha=0.68,
                label=label,
                zorder=1,
            )
        )

    legend_handles: list[Line2D] = [
        Line2D([0], [0], color=PAD_FACE, linewidth=8, label="Deformable pad T3"),
        Line2D([0], [0], color=CARRIER_FACE, linewidth=8, label="Rigid carrier T3"),
    ]
    for tag, edges in mesh.boundary_edges.items():
        color, linestyle, linewidth = BOUNDARY_STYLE[tag]
        if edges:
            segments = [
                [
                    (mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm)
                    for node_id in edge.node_ids
                ]
                for edge in edges
            ]
            axis.add_collection(
                LineCollection(
                    segments,
                    colors=color,
                    linestyles=linestyle,
                    linewidths=linewidth,
                    zorder=5,
                )
            )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=tag,
            )
        )

    coordinates = [(node.x_mm, node.y_mm) for node in mesh.nodes.values()]
    minimum_x = min(point[0] for point in coordinates)
    maximum_x = max(point[0] for point in coordinates)
    minimum_y = min(point[1] for point in coordinates)
    maximum_y = max(point[1] for point in coordinates)
    padding = 0.04 * max(maximum_x - minimum_x, maximum_y - minimum_y)
    axis.set_xlim(minimum_x - padding, maximum_x + padding)
    axis.set_ylim(minimum_y - padding, maximum_y + padding)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.set_title(
        f"Phase 4M {mesh.settings.level} mesh — "
        f"{mesh.quality.node_count} nodes, "
        f"{mesh.quality.t3_element_count} T3, "
        f"min angle {mesh.quality.minimum_triangle_angle_degrees:.2f}°"
    )
    axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=4,
        fontsize=7,
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path
