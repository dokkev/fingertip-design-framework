"""Dependency-light contracts for the render-only distal phalanx mesh."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from mesh.rigid.carrier import make_distal_phalanx_mesh
from finger.fingertip_geometry import FingertipModel
from finger.fingertip_parameters import FingertipParameters
from finger.extrusion import FingertipSolid, build_fingertip_solid


def _solid():
    return build_fingertip_solid(FingertipModel(FingertipParameters()))


def _signed_volume(mesh) -> float:
    points = mesh.vertices_mm[mesh.faces]
    return float(
        np.sum(np.einsum("ij,ij->i", points[:, 0], np.cross(points[:, 1], points[:, 2])))
        / 6.0
    )


def _edge_directions(mesh) -> dict[tuple[int, int], list[int]]:
    result: dict[tuple[int, int], list[int]] = {}
    for face in mesh.faces:
        for start, end in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            edge = min(start, end), max(start, end)
            result.setdefault(edge, []).append(1 if (start, end) == edge else -1)
    return result


def test_distal_phalanx_mesh_is_closed_deterministic_and_outward() -> None:
    first = make_distal_phalanx_mesh(_solid())
    second = make_distal_phalanx_mesh(_solid())
    first_surface = first.surface_mesh
    second_surface = second.surface_mesh

    np.testing.assert_array_equal(first_surface.vertices_mm, second_surface.vertices_mm)
    np.testing.assert_array_equal(first_surface.faces, second_surface.faces)
    assert np.all(np.isfinite(first_surface.vertices_mm))
    assert np.all(np.isfinite(first_surface.faces))
    triangle_points = first_surface.vertices_mm[first_surface.faces]
    assert np.all(
        np.linalg.norm(
            np.cross(triangle_points[:, 1] - triangle_points[:, 0], triangle_points[:, 2] - triangle_points[:, 0]),
            axis=1,
        )
        > 1.0e-12
    )
    assert _signed_volume(first_surface) > 0.0
    assert all(len(directions) == 2 for directions in _edge_directions(first_surface).values())
    assert all(sorted(directions) == [-1, 1] for directions in _edge_directions(first_surface).values())
    np.testing.assert_allclose(
        first_surface.bounds_mm,
        ((-15.0, -6.0, -5.5), (15.0, 3.5, 5.5)),
        atol=1.0e-12,
        rtol=0.0,
    )
    assert first_surface.metadata["source_geometry"] == "FingertipSolid.rigid_geometry"
    assert first.z_min_mm == -5.5
    assert first.z_max_mm == 5.5
    assert first.morphology_fingerprint == _solid().morphology_fingerprint
    assert set(first.lateral_face_indices).isdisjoint(
        first.longitudinal_end_face_indices
    )
    assert set(first.lateral_face_indices) | set(
        first.longitudinal_end_face_indices
    ) == set(range(len(first_surface.faces)))
    lateral_triangles = first_surface.vertices_mm[
        first_surface.faces[np.asarray(first.lateral_face_indices, dtype=np.int64)]
    ]
    assert not np.any(
        np.all(
            np.isclose(
                lateral_triangles[:, :, 2],
                lateral_triangles[:, :1, 2],
                atol=1.0e-12,
            ),
            axis=1,
        )
    )


def test_distal_phalanx_cross_section_matches_authoritative_rigid_geometry() -> None:
    solid = _solid()
    mesh = make_distal_phalanx_mesh(solid)
    top = mesh.surface_mesh.vertices_mm[mesh.surface_mesh.faces]
    top_faces = top[np.all(np.isclose(top[:, :, 2], 5.5, atol=1.0e-12), axis=1)]
    reconstructed = unary_union(
        [Polygon(face[:, :2]) for face in top_faces]
    )

    assert reconstructed.symmetric_difference(solid.rigid_geometry).area <= 1.0e-9


def test_distal_phalanx_mesh_preserves_authoritative_holes() -> None:
    parameters = FingertipParameters()
    pad_geometry = box(-1.0, -1.0, 1.0, 1.0)
    rigid_geometry = Polygon(
        [(3.0, -2.0), (7.0, -2.0), (7.0, 2.0), (3.0, 2.0)],
        [[(4.0, -1.0), (4.0, 1.0), (6.0, 1.0), (6.0, -1.0)]],
    )
    fingerprint = FingertipSolid._fingerprint(
        parameters,
        pad_geometry,
        rigid_geometry,
        pad_geometry,
        -5.5,
        5.5,
    )
    solid = FingertipSolid(
        parameters,
        pad_geometry,
        rigid_geometry,
        pad_geometry,
        -5.5,
        5.5,
        (),
        fingerprint,
    )

    mesh = make_distal_phalanx_mesh(solid)
    top = mesh.surface_mesh.vertices_mm[mesh.surface_mesh.faces]
    top_faces = top[np.all(np.isclose(top[:, :, 2], 5.5, atol=1.0e-12), axis=1)]
    reconstructed = unary_union([Polygon(face[:, :2]) for face in top_faces])

    assert reconstructed.symmetric_difference(rigid_geometry).area <= 1.0e-9
