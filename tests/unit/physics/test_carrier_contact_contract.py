"""Geometry contracts for positive void-height carrier-contact validation."""

from __future__ import annotations

import pytest

pytest.importorskip("gmsh")

from physics import prepare_fingertip_mesh
from mesh.rigid.carrier import make_distal_phalanx_mesh
from mesh.volume.mesh import generate_volume_mesh
from mesh.volume.contracts import volume_mesh_settings_for_tier
from finger import Fingertip, FingertipParameters


def test_positive_void_height_preserves_support_bonds_and_free_void_bottom() -> None:
    fingertip = Fingertip(FingertipParameters(void_height=1.0))
    volume_mesh = generate_volume_mesh(
        fingertip.solid(),
        volume_mesh_settings_for_tier("search"),
    )
    prepared = prepare_fingertip_mesh(volume_mesh)

    support = set(prepared.support_vertex_indices)
    left = set(prepared.surface_triangles["support_bond_left"].reshape(-1).tolist())
    right = set(prepared.surface_triangles["support_bond_right"].reshape(-1).tolist())
    void_bottom = set(prepared.surface_triangles["void_bottom"].reshape(-1).tolist())

    assert left
    assert right
    assert left | right <= support
    assert void_bottom
    assert void_bottom.isdisjoint(support)
    assert all(
        prepared.tet_mesh.vertices[index, 1]
        == pytest.approx(
            -(fingertip.parameters.stem_height + fingertip.parameters.void_height)
        )
        for index in void_bottom
    )


def test_carrier_mesh_records_authoritative_cross_section_and_depth() -> None:
    fingertip = Fingertip(FingertipParameters(void_height=1.0))
    carrier = make_distal_phalanx_mesh(fingertip.solid())

    assert carrier.surface_mesh.name == "distal_phalanx_carrier"
    assert carrier.cross_section.equals(fingertip.solid().rigid_geometry)
    assert carrier.z_min_mm == pytest.approx(-5.5)
    assert carrier.z_max_mm == pytest.approx(5.5)
