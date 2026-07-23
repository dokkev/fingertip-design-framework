"""Generate, validate, and visualize the Phase 4M fingertip meshes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from mesh.fingertip import generate_fingertip_mesh
from mesh.types import mesh_settings_for_level
from visualization.mesh import save_mesh_figure
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


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
        image_path = save_mesh_figure(
            mesh, output_directory / f"{level}_mesh.png"
        )
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
