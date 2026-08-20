from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("gmsh")

from mesh import FingertipVolumeState, volume_mesh_settings_for_tier
from mesh.fingertip.geometry import generate_fingertip_mesh
from mesh.fingertip.contracts import mesh_settings_for_level
from mesh.volume.mesh import generate_volume_mesh
from model import Fingertip
from optics.transport3d import build_fingertip_volume_state_geometry


def _relabel_with_surface_nodes_outside_the_canonical_prefix(volume_mesh):
    lateral_node_ids = {
        int(node_id)
        for tag, triangles in volume_mesh.surface_triangles.items()
        if not tag.startswith("longitudinal_end_")
        for triangle in triangles
        for node_id in triangle.node_ids
    }
    original_node_ids = tuple(sorted(volume_mesh.nodes))
    interior_first = [node_id for node_id in original_node_ids if node_id not in lateral_node_ids]
    lateral_last = [node_id for node_id in original_node_ids if node_id in lateral_node_ids]
    relabel = {
        old_id: new_id
        for new_id, old_id in enumerate((*interior_first, *lateral_last), start=1)
    }
    nodes = {
        relabel[old_id]: replace(node, id=relabel[old_id])
        for old_id, node in volume_mesh.nodes.items()
    }
    tetrahedra = tuple(
        replace(
            tetrahedron,
            node_ids=tuple(relabel[node_id] for node_id in tetrahedron.node_ids),
        )
        for tetrahedron in volume_mesh.tetrahedra
    )
    surface_triangles = {
        tag: tuple(
            replace(
                triangle,
                node_ids=tuple(relabel[node_id] for node_id in triangle.node_ids),
            )
            for triangle in triangles
        )
        for tag, triangles in volume_mesh.surface_triangles.items()
    }
    return replace(
        volume_mesh,
        nodes=nodes,
        tetrahedra=tetrahedra,
        surface_triangles=surface_triangles,
    )


def _surface_node_ids(volume_mesh) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(node_id)
                for tag, triangles in volume_mesh.surface_triangles.items()
                if not tag.startswith("longitudinal_end_")
                for triangle in triangles
                for node_id in triangle.node_ids
            }
        )
    )


def _tetra_by_boundary_face(state: FingertipVolumeState) -> dict[tuple[int, int, int], tuple[int, ...]]:
    result: dict[tuple[int, int, int], list[tuple[int, ...]]] = {}
    for tetrahedron in state.tetrahedra:
        node_ids = tuple(int(node_id) for node_id in tetrahedron.node_ids)
        for face in (
            (node_ids[0], node_ids[1], node_ids[2]),
            (node_ids[0], node_ids[1], node_ids[3]),
            (node_ids[0], node_ids[2], node_ids[3]),
            (node_ids[1], node_ids[2], node_ids[3]),
        ):
            result.setdefault(tuple(sorted(face)), []).append(node_ids)
    return {
        key: values[0]
        for key, values in result.items()
        if len(values) == 1
    }


def test_volume_state_direct_adapter_builds_full3d_geometry_without_fea_artifact() -> None:
    tip = Fingertip()
    volume_mesh = generate_volume_mesh(
        tip.solid(),
        volume_mesh_settings_for_tier("search"),
    )
    state = FingertipVolumeState.reference(volume_mesh)

    geometry = build_fingertip_volume_state_geometry(
        tip,
        state,
        reference_mesh=generate_fingertip_mesh(
            tip.geometry,
            mesh_settings_for_level("medium"),
        ),
        full3d_surface_provenance="actual_reference_3d_volume_state",
    )

    assert geometry.metadata["geometry_mode"] == "full3d_surface"
    assert geometry.depth_mm == pytest.approx(11.0)
    assert geometry.z_min_mm == pytest.approx(-5.5)
    assert geometry.z_max_mm == pytest.approx(5.5)
    assert geometry.metadata["full3d_surface_provenance"] == "actual_reference_3d_volume_state"
    assert set(geometry.silicone.semantic_tags or ()) == set(volume_mesh.surface_triangles) - {
        "longitudinal_end_minus",
        "longitudinal_end_plus",
    }
    assert geometry.silicone.external_surface is not None
    assert np.all(geometry.silicone.external_surface == np.asarray([
        tag.startswith("outer_compliant_")
        for tag in geometry.silicone.semantic_tags or ()
    ]))
    geometric_normals = np.cross(
        geometry.silicone.vertices[geometry.silicone.faces[:, 1]]
        - geometry.silicone.vertices[geometry.silicone.faces[:, 0]],
        geometry.silicone.vertices[geometry.silicone.faces[:, 2]]
        - geometry.silicone.vertices[geometry.silicone.faces[:, 0]],
    )
    alignment = np.sum(
        geometric_normals * geometry.silicone.normals,
        axis=1,
    )
    assert np.all(alignment > 0.0)


def test_surface_coordinates_use_canonical_ids_when_surface_nodes_are_not_a_prefix() -> None:
    tip = Fingertip()
    original_mesh = generate_volume_mesh(
        tip.solid(),
        volume_mesh_settings_for_tier("search"),
    )
    volume_mesh = _relabel_with_surface_nodes_outside_the_canonical_prefix(original_mesh)
    state = FingertipVolumeState.reference(volume_mesh)
    geometry = build_fingertip_volume_state_geometry(
        tip,
        state,
        reference_mesh=generate_fingertip_mesh(
            tip.geometry,
            mesh_settings_for_level("medium"),
        ),
        full3d_surface_provenance="actual_reference_3d_volume_state",
    )

    surface_node_ids = _surface_node_ids(volume_mesh)
    assert surface_node_ids != tuple(range(1, len(surface_node_ids) + 1))
    canonical_index = {node_id: index for index, node_id in enumerate(state.source_node_ids)}
    expected = np.asarray(
        [state.deformed_coordinates_mm[canonical_index[node_id]] for node_id in surface_node_ids],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(geometry.silicone.vertices, expected)


def test_external_surface_orientation_is_outward_from_canonical_tetrahedra() -> None:
    tip = Fingertip()
    volume_mesh = generate_volume_mesh(
        tip.solid(),
        volume_mesh_settings_for_tier("search"),
    )
    state = FingertipVolumeState.reference(volume_mesh)
    geometry = build_fingertip_volume_state_geometry(
        tip,
        state,
        reference_mesh=generate_fingertip_mesh(
            tip.geometry,
            mesh_settings_for_level("medium"),
        ),
        full3d_surface_provenance="actual_reference_3d_volume_state",
    )
    surface_node_ids = _surface_node_ids(volume_mesh)
    canonical_index = {node_id: index for index, node_id in enumerate(state.source_node_ids)}
    tetra_by_face = _tetra_by_boundary_face(state)

    external_surface = np.asarray(geometry.silicone.external_surface, dtype=bool)
    for face, tag, is_external in zip(
        geometry.silicone.faces,
        geometry.silicone.semantic_tags or (),
        external_surface,
    ):
        if not is_external:
            continue
        source_ids = tuple(surface_node_ids[int(index)] for index in face)
        tetrahedron = tetra_by_face[tuple(sorted(source_ids))]
        points = state.reference_coordinates_mm[
            np.asarray([canonical_index[node_id] for node_id in source_ids], dtype=np.int64)
        ]
        tetra_points = state.reference_coordinates_mm[
            np.asarray([canonical_index[node_id] for node_id in tetrahedron], dtype=np.int64)
        ]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        offset = np.mean(points, axis=0) - np.mean(tetra_points, axis=0)
        assert float(np.dot(normal, offset)) > 0.0, tag


def test_u_material_coordinates_are_reference_based_and_backend_independent() -> None:
    tip = Fingertip()
    volume_mesh = generate_volume_mesh(
        tip.solid(),
        volume_mesh_settings_for_tier("search"),
    )
    state = FingertipVolumeState.reference(volume_mesh)
    deformed = state.reference_coordinates_mm.copy()
    deformed[:, 1] += 0.05
    deformed_state = FingertipVolumeState.from_deformed_coordinates(volume_mesh, deformed)
    reference_geometry = build_fingertip_volume_state_geometry(
        tip,
        state,
        reference_mesh=generate_fingertip_mesh(
            tip.geometry,
            mesh_settings_for_level("medium"),
        ),
        full3d_surface_provenance="actual_reference_3d_volume_state",
    )
    deformed_geometry = build_fingertip_volume_state_geometry(
        tip,
        deformed_state,
        reference_mesh=generate_fingertip_mesh(
            tip.geometry,
            mesh_settings_for_level("medium"),
        ),
        full3d_surface_provenance="actual_deformed_3d_volume_state",
    )

    np.testing.assert_array_equal(
        reference_geometry.silicone.u_start,
        deformed_geometry.silicone.u_start,
    )
    np.testing.assert_array_equal(
        reference_geometry.silicone.u_end,
        deformed_geometry.silicone.u_end,
    )
    external = np.asarray(deformed_geometry.silicone.external_surface, dtype=bool)
    external_u = np.concatenate(
        (
            np.asarray(deformed_geometry.silicone.u_start)[external],
            np.asarray(deformed_geometry.silicone.u_end)[external],
        )
    )
    assert np.all(np.isfinite(external_u))
    assert float(np.ptp(external_u)) > 0.0
