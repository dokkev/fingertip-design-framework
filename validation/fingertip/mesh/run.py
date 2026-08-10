"""Generate, validate, and visualize the Phase 4M fingertip meshes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D

from mesh.fingertip import generate_fingertip_mesh
from mesh.types import mesh_settings_for_level
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


_BOUNDARY_STYLE = {
    "pad_bond_left": ("#111111", "-", 2.4),
    "pad_bond_right": ("#111111", "-", 2.4),
    "pad_outer_left": ("#287D91", "-", 2.0),
    "pad_outer_right": ("#287D91", "-", 2.0),
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


def _save_mesh_figure(mesh: Any, output_path: Path) -> Path:
    """Save the full carrier-and-pad validation figure locally to validation."""
    figure, axis = plt.subplots(figsize=(11.0, 8.5))
    for elements, face_color, label in (
        (mesh.pad_elements, "#9ED7E5", "Deformable pad T3"),
        (mesh.carrier_elements, "#747B84", "Rigid carrier T3"),
    ):
        polygons = [
            [(mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm) for node_id in element.node_ids]
            for element in elements
        ]
        axis.add_collection(
            PolyCollection(
                polygons,
                facecolors=face_color,
                edgecolors="#56616A",
                linewidths=0.16,
                alpha=0.68,
                label=label,
                zorder=1,
            )
        )
    handles = [
        Line2D([0], [0], color="#9ED7E5", linewidth=8, label="Deformable pad T3"),
        Line2D([0], [0], color="#747B84", linewidth=8, label="Rigid carrier T3"),
    ]
    for tag, edges in mesh.boundary_edges.items():
        color, linestyle, linewidth = _BOUNDARY_STYLE[tag]
        segments = [
            [(mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm) for node_id in edge.node_ids]
            for edge in edges
        ]
        if segments:
            axis.add_collection(
                LineCollection(
                    segments,
                    colors=color,
                    linestyles=linestyle,
                    linewidths=linewidth,
                    zorder=5,
                )
            )
        handles.append(
            Line2D([0], [0], color=color, linestyle=linestyle, linewidth=linewidth, label=tag)
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
        f"Phase 4M {mesh.settings.level} mesh — {mesh.quality.node_count} nodes, "
        f"{mesh.quality.t3_element_count} T3, min angle "
        f"{mesh.quality.minimum_triangle_angle_degrees:.2f}°"
    )
    axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4, fontsize=7, frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=("medium", "fine"),
        default=("medium", "fine"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("output/validation/fingertip/mesh"),
    )
    return parser.parse_args()


def _mesh_summary(mesh: Any, image_path: Path) -> dict[str, Any]:
    return {
        "image": str(image_path),
        "gmsh_version": mesh.gmsh_version,
        "settings": asdict(mesh.settings),
        "quality": asdict(mesh.quality),
        "validation": asdict(mesh.validation),
        "boundary_counts": {
            tag: {
                "edges": len(edges),
                "nodes": len(
                    {node_id for edge in edges for node_id in edge.node_ids}
                ),
            }
            for tag, edges in mesh.boundary_edges.items()
        },
        "contact_pairs": [asdict(pair) for pair in mesh.contact_pairs],
    }


def main() -> int:
    arguments = _parse_arguments()
    output_directory = arguments.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    model = FingertipModel(FingertipParameters())
    summaries: dict[str, Any] = {}
    all_passed = True
    for level in arguments.levels:
        mesh = generate_fingertip_mesh(model, mesh_settings_for_level(level))
        image_path = _save_mesh_figure(mesh, output_directory / f"{level}_mesh.png")
        summaries[level] = _mesh_summary(mesh, image_path)
        all_passed = all_passed and mesh.validation.passed
    metrics = {
        "phase": "4M",
        "geometry_source": "FingertipModel Shapely geometries and semantics",
        "parameters": asdict(model.parameters),
        "levels": summaries,
        "all_meshes_pass": all_passed,
    }
    metrics_path = output_directory / "mesh_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for level, summary in summaries.items():
        quality = summary["quality"]
        print(
            f"{level}: {quality['node_count']} nodes, "
            f"{quality['t3_element_count']} T3, "
            f"min angle={quality['minimum_triangle_angle_degrees']:.3f} deg, "
            f"validation={'PASS' if summary['validation']['passed'] else 'FAIL'}"
        )
    print(metrics_path)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
