"""GPU smoke contract for explicit compliant-pad/distal-carrier contact."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely import wkt
from shapely.geometry import Point

pytest.importorskip("gmsh")
pytest.importorskip("warp")
pytest.importorskip("newton")

import warp as wp

from contact import (
    FirstContactSettings,
    canonical_sphere_alignment,
    find_first_contact,
    make_outer_compliant_surface,
)
from physics import (
    IndentationSettings,
    NewtonSettings,
    RigidIndenter3D,
    prepare_fingertip_mesh,
    solve_fingertip_indentation,
)
from mesh.rigid_carrier import make_distal_phalanx_mesh
from mesh.rigid_object import make_sphere_mesh
from mesh.volume3d import generate_volume_mesh
from mesh.volume_types import volume_mesh_settings_for_tier
from model import Fingertip, FingertipParameters


def _six_volumes(vertices: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    points = vertices[tetrahedra]
    return np.einsum(
        "ij,ij->i",
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
        points[:, 3] - points[:, 0],
    )


def _void_bottom_clearance_mm(vertices: np.ndarray, prepared, carrier) -> float:
    polygon = wkt.loads(carrier.metadata["cross_section_wkt"])
    z_min = float(carrier.metadata["z_min_mm"])
    z_max = float(carrier.metadata["z_max_mm"])
    indices = np.unique(prepared.surface_triangles["void_bottom"].reshape(-1))
    clearances = []
    for x_mm, y_mm, z_mm in vertices[indices]:
        if z_min <= z_mm <= z_max:
            point = Point(float(x_mm), float(y_mm))
            distance = float(point.distance(polygon.boundary))
            clearances.append(-distance if polygon.covers(point) else distance)
    return float(min(clearances))


@pytest.mark.smoke
@pytest.mark.physics
def test_positive_void_height_collision_off_vs_on() -> None:
    if not wp.is_device_available("cuda:0"):
        pytest.skip("carrier-contact smoke requires cuda:0")

    fingertip = Fingertip(FingertipParameters(void_height=1.0))
    volume_mesh = generate_volume_mesh(
        fingertip.solid(),
        volume_mesh_settings_for_tier("search"),
    )
    prepared = prepare_fingertip_mesh(volume_mesh)
    carrier = make_distal_phalanx_mesh(volume_mesh.solid)
    sphere = make_sphere_mesh(5.0, subdivisions=3)
    alignment = canonical_sphere_alignment(
        fingertip.geometry,
        sphere,
        initial_gap_mm=0.25,
    )
    first_contact = find_first_contact(
        make_outer_compliant_surface(volume_mesh.solid),
        sphere,
        alignment.nominal_pose,
        alignment.approach_direction,
        FirstContactSettings(
            coarse_step_mm=0.25,
            tolerance_mm=1.0e-3,
            spawn_clearance_mm=0.05,
            max_travel_mm=20.0,
        ),
    )
    indenter = RigidIndenter3D(
        sphere,
        alignment.nominal_pose,
        alignment.approach_direction,
    )
    mechanics_settings = NewtonSettings(
        device="cuda:0",
        gravity=0.0,
        dt=1.0e-3,
        steps=1,
        iterations=10,
        fixed_vertex_indices=prepared.support_vertex_indices,
    )
    indentation_settings = IndentationSettings(
        travel_mm=3.0,
        load_steps=math.ceil(3.0 / 0.05),
        soft_contact_margin_mm=0.02,
        soft_contact_ke=1.0e3,
        soft_contact_kd=10.0,
    )

    collision_off = solve_fingertip_indentation(
        prepared,
        indenter,
        mechanics_settings,
        indentation_settings,
        first_contact=first_contact,
        visual_carrier_mesh=carrier,
    )
    collision_on = solve_fingertip_indentation(
        prepared,
        indenter,
        mechanics_settings,
        indentation_settings,
        first_contact=first_contact,
        rigid_carrier_mesh=carrier,
    )

    off_gap = _void_bottom_clearance_mm(
        collision_off.mechanics_result.deformed_vertices,
        prepared,
        carrier,
    )
    six_volumes = _six_volumes(
        collision_on.mechanics_result.deformed_vertices,
        collision_on.mechanics_result.tetrahedra,
    )
    print(
        "carrier OFF/ON: "
        f"off_gap_mm={off_gap:.6g}, "
        f"on_gap_mm={collision_on.diagnostics['min_carrier_clearance_mm']:.6g}, "
        f"on_contacts={collision_on.diagnostics['max_void_bottom_carrier_contact_count']}, "
        f"on_first_step={collision_on.diagnostics['first_carrier_contact_step']}"
    )
    assert collision_on.diagnostics["carrier_collision_enabled"] is True
    assert collision_off.diagnostics["carrier_collision_enabled"] is False
    assert collision_on.diagnostics["carrier_contact_active"] is True
    assert int(collision_on.diagnostics["max_void_bottom_carrier_contact_count"]) > 0
    assert int(collision_on.diagnostics["max_sphere_carrier_rigid_contact_count"]) == 0
    assert int(collision_on.diagnostics["max_soft_contact_overflow"]) == 0
    assert int(collision_on.diagnostics["max_rigid_contact_overflow"]) == 0
    assert off_gap < -0.25
    assert float(collision_on.diagnostics["max_carrier_penetration_mm"]) <= 0.5 * 0.125
    assert np.all(np.isfinite(collision_on.mechanics_result.deformed_vertices))
    assert float(np.min(six_volumes)) > 0.0
