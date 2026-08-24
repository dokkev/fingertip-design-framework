"""Load triangle-mesh files for Newton."""

from __future__ import annotations

from pathlib import Path

import newton
import numpy as np

from lumo.util.scalar_validation import require_positive


def load_mesh(
    mesh_path: str | Path,
    *,
    scale_m_per_unit: float,
) -> newton.Mesh:
    """Load an OBJ or STL triangle surface as a Newton mesh in SI units."""
    mesh_path = Path(mesh_path)
    if mesh_path.suffix.lower() not in {".obj", ".stl"}:
        raise ValueError(f"mesh_path must be an OBJ or STL file; got {mesh_path}")
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    require_positive("scale_m_per_unit", scale_m_per_unit)

    import meshio

    source_mesh = meshio.read(mesh_path)
    unsupported_cell_types = sorted(
        {
            cell_block.type
            for cell_block in source_mesh.cells
            if cell_block.type not in {"triangle", "quad"}
        }
    )
    if unsupported_cell_types:
        raise ValueError(
            "surface mesh contains unsupported cell types: "
            + ", ".join(unsupported_cell_types)
        )

    triangles = source_mesh.get_cells_type("triangle")
    quads = source_mesh.get_cells_type("quad")
    if quads.size:
        quad_triangles = np.concatenate(
            (quads[:, (0, 1, 2)], quads[:, (0, 2, 3)]),
            axis=0,
        )
        triangles = np.concatenate((triangles, quad_triangles), axis=0)
    if not triangles.size:
        raise ValueError("surface mesh must contain triangle or quad faces")

    vertices = np.asarray(source_mesh.points, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("surface mesh vertices must have three coordinates")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("surface mesh vertices must be finite")

    return newton.Mesh(
        vertices=vertices * scale_m_per_unit,
        indices=np.asarray(triangles, dtype=np.int32).reshape(-1),
        compute_inertia=False,
    )


__all__ = ["load_mesh"]
