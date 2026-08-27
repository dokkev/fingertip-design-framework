"""Regression tests for the complete fingertip mesh."""

from __future__ import annotations

import numpy as np
import pytest

from lumo.fingertip import (
    ACTIVE_Y_BOUNDS_MM,
    DISTAL_END_CAP_LENGTH_MM,
    LED_CENTERS_Y_MM,
    LED_RECESS_DEPTH_MM,
    LED_RECESS_WIDTH_MM,
    TOTAL_Y_BOUNDS_MM,
    Fingertip,
)
from lumo.mesh import FingertipMesh, make_fingertip_mesh


@pytest.fixture(scope="module")
def full_mesh():
    return make_fingertip_mesh(Fingertip(), element_size_mm=1.0)


def _connected_vertex_components(
    element_indices: np.ndarray,
    vertex_count: int,
) -> int:
    parent = np.arange(vertex_count, dtype=np.int32)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for element in element_indices:
        first_root = root(int(element[0]))
        for vertex in element[1:]:
            other_root = root(int(vertex))
            if first_root != other_root:
                parent[other_root] = first_root

    used = np.unique(element_indices)
    return len({root(int(vertex)) for vertex in used})


def test_full_silicone_is_one_valid_tet_body(full_mesh) -> None:
    vertices = np.asarray(full_mesh.silicone.vertices, dtype=np.float64)
    tetrahedra = np.asarray(full_mesh.silicone.tet_indices).reshape(-1, 4)

    assert np.all(np.isfinite(vertices))
    assert np.isclose(
        vertices[:, 1].min(),
        1.0e-3 * TOTAL_Y_BOUNDS_MM[0],
        atol=1.0e-8,
    )
    assert np.isclose(
        vertices[:, 1].max(),
        1.0e-3 * TOTAL_Y_BOUNDS_MM[1],
        atol=1.0e-8,
    )
    assert np.unique(np.sort(tetrahedra, axis=1), axis=0).shape == tetrahedra.shape
    assert np.unique(tetrahedra).size == vertices.shape[0]
    assert _connected_vertex_components(tetrahedra, len(vertices)) == 1

    points = vertices[tetrahedra]
    signed_volumes = np.einsum(
        "ij,ij->i",
        points[:, 1] - points[:, 0],
        np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 0]),
    ) / 6.0
    assert np.all(np.isfinite(signed_volumes))
    assert np.all(np.abs(signed_volumes) > 0.0)
    assert np.all(np.sign(signed_volumes) == np.sign(signed_volumes[0]))


def test_distal_end_cap_fills_the_main_cavity(full_mesh) -> None:
    vertices = np.asarray(full_mesh.silicone.vertices, dtype=np.float64)
    tetrahedra = np.asarray(full_mesh.silicone.tet_indices).reshape(-1, 4)
    centroids_mm = 1.0e3 * vertices[tetrahedra].mean(axis=1)
    fingertip = full_mesh.fingertip
    silicone = fingertip.silicone
    geometry = fingertip.parameters.geometry
    stem_right_x_mm = 0.5 * geometry.stem_width_mm
    lateral_void_width_mm = silicone.cavity_right_x_mm - stem_right_x_mm
    cavity_height_mm = -silicone.cavity_bottom_z_mm
    in_lateral_void = (
        (
            centroids_mm[:, 0]
            > stem_right_x_mm + 0.2 * lateral_void_width_mm
        )
        & (
            centroids_mm[:, 0]
            < stem_right_x_mm + 0.8 * lateral_void_width_mm
        )
        & (
            centroids_mm[:, 2]
            > silicone.cavity_bottom_z_mm + 0.2 * cavity_height_mm
        )
        & (
            centroids_mm[:, 2]
            < silicone.cavity_bottom_z_mm + 0.8 * cavity_height_mm
        )
    )
    active_half_length_mm = ACTIVE_Y_BOUNDS_MM[1]
    distal_start_mm = ACTIVE_Y_BOUNDS_MM[1]
    distal_probe_offset_mm = 0.1 * DISTAL_END_CAP_LENGTH_MM
    main_cavity = in_lateral_void & (
        np.abs(centroids_mm[:, 1]) < 0.75 * active_half_length_mm
    )
    distal_solid = in_lateral_void & (
        centroids_mm[:, 1] > distal_start_mm + distal_probe_offset_mm
    )
    bond_height_mm = silicone.bond_top_z_mm
    distal_stem_fill = (
        (np.abs(centroids_mm[:, 0]) < 0.25 * geometry.stem_width_mm)
        & (
            centroids_mm[:, 1]
            > distal_start_mm + distal_probe_offset_mm
        )
        & (centroids_mm[:, 2] > 0.2 * bond_height_mm)
        & (centroids_mm[:, 2] < 0.8 * bond_height_mm)
    )

    assert not np.any(main_cavity)
    assert np.any(distal_solid)
    assert np.any(distal_stem_fill)
    bonded_y_m = vertices[full_mesh.bonded_vertex_indices, 1]
    assert bonded_y_m.min() >= 1.0e-3 * TOTAL_Y_BOUNDS_MM[0] - 1.0e-8
    assert np.isclose(
        bonded_y_m.max(),
        1.0e-3 * TOTAL_Y_BOUNDS_MM[1],
        atol=1.0e-8,
    )


def test_carrier_has_55_mm_stem_and_distal_dorsal_reinforcement(full_mesh) -> None:
    vertices = np.asarray(full_mesh.carrier.vertices, dtype=np.float64)
    triangles = np.asarray(full_mesh.carrier.indices).reshape(-1, 3)

    assert np.isclose(
        vertices[:, 1].min(),
        1.0e-3 * TOTAL_Y_BOUNDS_MM[0],
        atol=1.0e-8,
    )
    assert np.isclose(
        vertices[:, 1].max(),
        1.0e-3 * TOTAL_Y_BOUNDS_MM[1],
        atol=1.0e-8,
    )
    assert _connected_vertex_components(triangles, len(vertices)) == 1
    edges = np.sort(
        np.concatenate(
            (
                triangles[:, (0, 1)],
                triangles[:, (1, 2)],
                triangles[:, (2, 0)],
            )
        ),
        axis=1,
    )
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(edge_counts == 2)
    distal_vertices = vertices[vertices[:, 1] > 27.5e-3 + 1.0e-8]
    assert len(distal_vertices) > 0
    dorsal_bottom_m = 1.0e-3 * full_mesh.fingertip.silicone.bond_top_z_mm
    assert distal_vertices[:, 2].min() >= dorsal_bottom_m - 1.0e-8


def test_each_led_has_explicit_stem_recess_and_air_gap(full_mesh) -> None:
    stem_bottom_z_mm = min(
        z_mm for _, z_mm in full_mesh.fingertip.carrier.cross_section
    )
    recess_floor_z_mm = stem_bottom_z_mm + LED_RECESS_DEPTH_MM
    silicone_surface_z_mm = full_mesh.fingertip.silicone.cavity_bottom_z_mm
    led_source_centers_m = np.asarray(
        full_mesh.fingertip.led_source_centers_m
    )

    assert np.allclose(
        1.0e3 * led_source_centers_m[:, 2],
        recess_floor_z_mm,
        rtol=0.0,
        atol=1.0e-6,
    )
    assert np.allclose(
        1.0e3 * led_source_centers_m[:, 2] - silicone_surface_z_mm,
        LED_RECESS_DEPTH_MM,
        rtol=0.0,
        atol=1.0e-6,
    )

    expected_edges_mm = np.sort(
        np.concatenate(
            (
                np.asarray(LED_CENTERS_Y_MM) - 0.5 * LED_RECESS_WIDTH_MM,
                np.asarray(LED_CENTERS_Y_MM) + 0.5 * LED_RECESS_WIDTH_MM,
            )
        )
    )
    for carrier_mesh in (full_mesh.carrier, full_mesh.carrier_collision):
        vertices_mm = 1.0e3 * np.asarray(carrier_mesh.vertices)
        triangles = np.asarray(carrier_mesh.indices).reshape(-1, 3)
        points_mm = vertices_mm[triangles]
        is_recess_floor = np.all(
            np.isclose(
                points_mm[:, :, 2],
                recess_floor_z_mm,
                rtol=0.0,
                atol=1.0e-5,
            ),
            axis=1,
        ) & (np.ptp(points_mm[:, :, 1], axis=1) > 1.0)
        recess_floor_points_mm = points_mm[is_recess_floor]
        assert len(recess_floor_points_mm) == 2 * len(LED_CENTERS_Y_MM)
        actual_edges_mm = np.unique(
            np.round(recess_floor_points_mm[:, :, 1], decimals=5)
        )
        assert np.allclose(
            actual_edges_mm,
            expected_edges_mm,
            rtol=0.0,
            atol=1.0e-5,
        )

        edges = np.sort(
            np.concatenate(
                (
                    triangles[:, (0, 1)],
                    triangles[:, (1, 2)],
                    triangles[:, (2, 0)],
                )
            ),
            axis=1,
        )
        _, edge_counts = np.unique(edges, axis=0, return_counts=True)
        assert np.all(edge_counts == 2)


def test_mesh_owns_bonded_index_validity(full_mesh) -> None:
    common = {
        "fingertip": full_mesh.fingertip,
        "silicone": full_mesh.silicone,
        "carrier": full_mesh.carrier,
        "carrier_collision": full_mesh.carrier_collision,
    }
    with pytest.raises(ValueError, match="must not be empty"):
        FingertipMesh(
            **common,
            bonded_vertex_indices=np.array([], dtype=np.int32),
        )
    with pytest.raises(ValueError, match="exceeds silicone vertex count"):
        FingertipMesh(
            **common,
            bonded_vertex_indices=np.array(
                [full_mesh.silicone.vertex_count],
                dtype=np.int32,
            ),
        )
