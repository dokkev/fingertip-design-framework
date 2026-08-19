"""Dependency-light contracts for neutral rigid triangle meshes."""

from __future__ import annotations

import numpy as np
import pytest

from mesh.rigid_object import (
    RigidObjectMesh,
    make_box_mesh,
    make_cube_mesh,
    make_cylinder_mesh,
    make_sphere_mesh,
)


def _signed_volume(mesh: RigidObjectMesh) -> float:
    points = mesh.vertices_mm[mesh.faces]
    return float(np.sum(np.einsum("ij,ij->i", points[:, 0], np.cross(points[:, 1], points[:, 2]))) / 6.0)


def _edge_directions(mesh: RigidObjectMesh) -> dict[tuple[int, int], list[int]]:
    result: dict[tuple[int, int], list[int]] = {}
    for face in mesh.faces:
        for start, end in ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))):
            edge = min(start, end), max(start, end)
            result.setdefault(edge, []).append(1 if (start, end) == edge else -1)
    return result


@pytest.mark.parametrize(
    "factory",
    (
        lambda: make_sphere_mesh(4.0, subdivisions=1),
        lambda: make_cylinder_mesh(3.0, 8.0, radial_segments=12),
        lambda: make_box_mesh(2.0, 3.0, 4.0),
        lambda: make_cube_mesh(2.0),
    ),
)
def test_primitives_are_closed_outward_and_deterministic(factory) -> None:
    first = factory()
    second = factory()

    assert first.vertices_mm.flags.writeable is False
    assert first.faces.flags.writeable is False
    np.testing.assert_array_equal(first.vertices_mm, second.vertices_mm)
    np.testing.assert_array_equal(first.faces, second.faces)
    assert _signed_volume(first) > 0.0
    assert all(sorted(directions) == [-1, 1] for directions in _edge_directions(first).values())
    assert all(len(directions) == 2 for directions in _edge_directions(first).values())


def test_primitive_geometry_conventions() -> None:
    sphere = make_sphere_mesh(4.0, subdivisions=1)
    np.testing.assert_allclose(np.linalg.norm(sphere.vertices_mm, axis=1), 4.0, atol=1.0e-12, rtol=0.0)

    cylinder = make_cylinder_mesh(3.0, 8.0, radial_segments=12)
    np.testing.assert_allclose(np.min(cylinder.vertices_mm[:, 2]), -4.0, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(np.max(cylinder.vertices_mm[:, 2]), 4.0, atol=1.0e-12, rtol=0.0)

    box = make_box_mesh(2.0, 3.0, 4.0)
    np.testing.assert_allclose(box.bounds_mm[0], (-1.0, -1.5, -2.0), atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(box.bounds_mm[1], (1.0, 1.5, 2.0), atol=1.0e-12, rtol=0.0)
    assert box.vertices_mm.shape == (8, 3)
    assert box.faces.shape == (12, 3)


def test_mesh_contract_copies_arrays_and_rejects_open_or_degenerate_geometry() -> None:
    box = make_cube_mesh(2.0)
    vertices = box.vertices_mm.copy()
    faces = box.faces.copy()
    mesh = RigidObjectMesh(vertices, faces, name="copy")
    vertices[0] = 100.0
    faces[0] = faces[0, ::-1]
    assert mesh.vertices_mm[0, 0] != 100.0
    assert not np.array_equal(mesh.faces[0], faces[0])
    with pytest.raises(ValueError, match="closed manifold"):
        RigidObjectMesh(box.vertices_mm, box.faces[:-1])
    with pytest.raises(ValueError, match="three distinct"):
        bad_faces = box.faces.copy()
        bad_faces[0] = (0, 0, 1)
        RigidObjectMesh(box.vertices_mm, bad_faces)


def test_mesh_rejects_nonfinite_vertices() -> None:
    box = make_cube_mesh(2.0)
    vertices = box.vertices_mm.copy()
    vertices[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        RigidObjectMesh(vertices, box.faces)
