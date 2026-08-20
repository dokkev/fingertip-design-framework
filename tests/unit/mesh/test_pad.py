from __future__ import annotations

import numpy as np

from mesh import PadMesh


def _square_mesh() -> PadMesh:
    return PadMesh.from_arrays(
        node_ids=np.asarray([30, 10, 40, 20], dtype=np.int64),
        reference_coordinates_mm=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=float,
        ),
        element_connectivity_node_ids=np.asarray(
            [[30, 40, 10], [30, 20, 40]],
            dtype=np.int64,
        ),
        boundary_edge_node_ids_by_tag={
            "bottom": np.asarray([[30, 10]], dtype=np.int64),
            "right": np.asarray([[10, 40]], dtype=np.int64),
            "top": np.asarray([[40, 20]], dtype=np.int64),
            "left": np.asarray([[20, 30]], dtype=np.int64),
        },
    )


def test_pad_mesh_maps_ids_normalizes_winding_and_preserves_semantics() -> None:
    mesh = _square_mesh()

    np.testing.assert_array_equal(
        mesh.triangles,
        np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )
    assert {tuple(edge) for edge in mesh.boundary_edges} == {
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
    }
    assert mesh.semantic_boundary_tags == ("bottom", "left", "right", "top")
    np.testing.assert_array_equal(mesh.boundary_edges_for("bottom"), [[0, 1]])
    np.testing.assert_array_equal(mesh.boundary_edges_for("left"), [[3, 0]])
    np.testing.assert_array_equal(mesh.boundary_edges_for("right"), [[1, 2]])
    np.testing.assert_array_equal(mesh.boundary_edges_for("top"), [[2, 3]])
    assert mesh.node_ids.tolist() == [30, 10, 40, 20]


def test_pad_mesh_exposes_reference_topology_only() -> None:
    mesh = _square_mesh()

    np.testing.assert_array_equal(mesh.coordinates, mesh.reference_coordinates_mm)
    assert mesh.boundaries is mesh.boundary_edges_by_tag
    assert not hasattr(mesh, "deformed")
    assert not hasattr(mesh, "reference_mesh")
    assert not hasattr(mesh, "displacement")
    assert not hasattr(mesh, "metadata")
