"""Regression tests for the full five-LED fingertip mesh."""

from __future__ import annotations

import numpy as np
import pytest

from lumo.fingertip import Fingertip
from lumo.mesh import (
    LED_PITCH_MM,
    MAIN_Y_BOUNDS_MM,
    NUM_LEDS,
    TOTAL_Y_BOUNDS_MM,
    led_centers_y_mm,
    make_fingertip_5led_mesh,
)


@pytest.fixture(scope="module")
def full_mesh():
    return make_fingertip_5led_mesh(Fingertip(), element_size_mm=1.0)


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


def test_five_led_longitudinal_layout() -> None:
    centers = np.asarray(led_centers_y_mm())
    assert centers.shape == (NUM_LEDS,)
    assert NUM_LEDS == 5
    assert np.array_equal(centers, np.array((-22.0, -11.0, 0.0, 11.0, 22.0)))
    assert np.allclose(np.diff(centers), LED_PITCH_MM, rtol=0.0, atol=1.0e-12)
    assert centers[-1] - centers[0] == 44.0
    assert MAIN_Y_BOUNDS_MM == (-27.5, 27.5)
    assert TOTAL_Y_BOUNDS_MM == (-27.5, 32.5)


def test_full_silicone_is_one_valid_tet_body(full_mesh) -> None:
    vertices = np.asarray(full_mesh.silicone.vertices, dtype=np.float64)
    tetrahedra = np.asarray(full_mesh.silicone.tet_indices).reshape(-1, 4)

    assert np.all(np.isfinite(vertices))
    assert np.isclose(vertices[:, 1].min(), -27.5e-3, atol=1.0e-8)
    assert np.isclose(vertices[:, 1].max(), 32.5e-3, atol=1.0e-8)
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
    in_lateral_void = (
        (centroids_mm[:, 0] > 4.2)
        & (centroids_mm[:, 0] < 5.4)
        & (centroids_mm[:, 2] > -5.0)
        & (centroids_mm[:, 2] < -1.0)
    )
    main_cavity = in_lateral_void & (np.abs(centroids_mm[:, 1]) < 20.0)
    distal_solid = in_lateral_void & (centroids_mm[:, 1] > 28.0)
    distal_stem_fill = (
        (np.abs(centroids_mm[:, 0]) < 2.0)
        & (centroids_mm[:, 1] > 28.0)
        & (centroids_mm[:, 2] > 1.0)
        & (centroids_mm[:, 2] < 7.0)
    )

    assert not np.any(main_cavity)
    assert np.any(distal_solid)
    assert np.any(distal_stem_fill)
    bonded_y_m = vertices[full_mesh.bonded_vertex_indices, 1]
    assert bonded_y_m.min() >= -27.5e-3 - 1.0e-8
    assert np.isclose(bonded_y_m.max(), 32.5e-3, atol=1.0e-8)


def test_carrier_has_55_mm_stem_and_distal_dorsal_reinforcement(full_mesh) -> None:
    vertices = np.asarray(full_mesh.carrier.vertices, dtype=np.float64)
    triangles = np.asarray(full_mesh.carrier.indices).reshape(-1, 3)

    assert np.isclose(vertices[:, 1].min(), -27.5e-3, atol=1.0e-8)
    assert np.isclose(vertices[:, 1].max(), 32.5e-3, atol=1.0e-8)
    assert _connected_vertex_components(triangles, len(vertices)) == 1
    distal_vertices = vertices[vertices[:, 1] > 27.5e-3 + 1.0e-8]
    assert len(distal_vertices) > 0
    dorsal_bottom_m = 1.0e-3 * full_mesh.fingertip.silicone.bond_top_z_mm
    assert distal_vertices[:, 2].min() >= dorsal_bottom_m - 1.0e-8
    assert full_mesh.led_centers_m.shape == (5, 3)
    assert np.allclose(
        full_mesh.led_centers_m[:, 1],
        1.0e-3 * np.asarray(led_centers_y_mm()),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert full_mesh.inter_led_midpoints_m.shape == (4, 3)


def test_full_mesh_preserves_local_height_contract(full_mesh) -> None:
    assert full_mesh.fingertip.full_height_mm == 24.0
    assert full_mesh.fingertip.full_height_mm <= 30.0
