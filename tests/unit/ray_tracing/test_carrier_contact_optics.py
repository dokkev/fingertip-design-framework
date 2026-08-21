from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gmsh")

from lumo.mesh import FingertipVolumeState, volume_mesh_settings_for_tier
from lumo.mesh.rigid.carrier import make_distal_phalanx_mesh
from lumo.mesh.volume.mesh import generate_volume_mesh
from lumo.finger import Fingertip
from lumo.ray_tracing.contracts.objects import CarrierOptics, IndenterOptics
from lumo.ray_tracing.optical_mechanics import build_fingertip_volume_state_geometry
from lumo.ray_tracing.optical_mechanics.geometry import CARRIER_CONTACT_INTERFACE


@pytest.fixture(scope="module")
def reference_state():
    tip = Fingertip()
    volume_mesh = generate_volume_mesh(
        tip.solid(),
        volume_mesh_settings_for_tier("search"),
    )
    return tip, volume_mesh, FingertipVolumeState.reference(volume_mesh)


def test_open_gap_keeps_all_void_triangles_as_air_or_internal(reference_state) -> None:
    tip, volume_mesh, state = reference_state
    geometry = build_fingertip_volume_state_geometry(
        tip,
        state,
        carrier_mesh=make_distal_phalanx_mesh(volume_mesh.solid),
        full3d_surface_provenance="actual_reference_3d_volume_state",
        carrier_optics=CarrierOptics("absorber"),
    )

    assert np.count_nonzero(
        np.asarray(geometry.silicone.interface_tags, dtype=object)
        == CARRIER_CONTACT_INTERFACE
    ) == 0
    assert CARRIER_CONTACT_INTERFACE not in (geometry.silicone.interface_tags or ())
    assert geometry.carrier_optics == IndenterOptics("absorber")
    assert geometry.carrier_contact_active is False


def test_exact_contact_vertices_map_only_semantic_void_triangles(reference_state) -> None:
    tip, volume_mesh, state = reference_state
    triangle = volume_mesh.surface_triangles["void_bottom"][0]
    contacted_node = int(triangle.node_ids[0])
    geometry = build_fingertip_volume_state_geometry(
        tip,
        state,
        carrier_mesh=make_distal_phalanx_mesh(volume_mesh.solid),
        full3d_surface_provenance="actual_reference_3d_volume_state",
        carrier_contact_source_node_ids={contacted_node},
        carrier_optics=CarrierOptics("absorber"),
        carrier_mapping_tolerance_mm=0.0625,
    )

    tags = np.asarray(geometry.silicone.interface_tags, dtype=object)
    semantic = np.asarray(geometry.silicone.semantic_tags, dtype=object)
    carrier = semantic == "void_bottom"
    assert int(np.count_nonzero(tags == CARRIER_CONTACT_INTERFACE)) > 0
    assert np.all(semantic[tags == CARRIER_CONTACT_INTERFACE] == "void_bottom")
    assert np.count_nonzero(tags == CARRIER_CONTACT_INTERFACE) <= np.count_nonzero(carrier)
    assert geometry.carrier_contact_active is True


@pytest.mark.parametrize("source", ("unknown", "void_left"))
def test_carrier_contact_rejects_non_void_bottom_source_nodes(
    reference_state,
    source: str,
) -> None:
    tip, volume_mesh, state = reference_state
    if source == "unknown":
        contacted_node = max(volume_mesh.nodes) + 1
    else:
        bottom_nodes = {
            int(node_id)
            for triangle in volume_mesh.surface_triangles["void_bottom"]
            for node_id in triangle.node_ids
        }
        contacted_node = next(
            int(node_id)
            for triangle in volume_mesh.surface_triangles["void_left"]
            for node_id in triangle.node_ids
            if int(node_id) not in bottom_nodes
        )

    with pytest.raises(ValueError, match="void-bottom"):
        build_fingertip_volume_state_geometry(
            tip,
            state,
            carrier_mesh=make_distal_phalanx_mesh(volume_mesh.solid),
            full3d_surface_provenance="actual_reference_3d_volume_state",
            carrier_contact_source_node_ids={contacted_node},
            carrier_optics=CarrierOptics("absorber"),
            carrier_mapping_tolerance_mm=0.0625,
        )


def test_contact_triangles_require_explicit_carrier_optics(reference_state) -> None:
    tip, volume_mesh, state = reference_state
    contacted_node = int(volume_mesh.surface_triangles["void_bottom"][0].node_ids[0])
    with pytest.raises(ValueError, match="carrier contact triangles"):
        build_fingertip_volume_state_geometry(
            tip,
            state,
            carrier_mesh=make_distal_phalanx_mesh(volume_mesh.solid),
            full3d_surface_provenance="actual_reference_3d_volume_state",
            carrier_contact_source_node_ids={contacted_node},
            carrier_mapping_tolerance_mm=0.0625,
        )


def test_carrier_mapping_tolerance_is_required_for_contact(reference_state) -> None:
    tip, volume_mesh, state = reference_state
    contacted_node = int(volume_mesh.surface_triangles["void_bottom"][0].node_ids[0])
    with pytest.raises(ValueError, match="explicit mapping tolerance"):
        build_fingertip_volume_state_geometry(
            tip,
            state,
            carrier_mesh=make_distal_phalanx_mesh(volume_mesh.solid),
            full3d_surface_provenance="actual_reference_3d_volume_state",
            carrier_contact_source_node_ids={contacted_node},
            carrier_optics=CarrierOptics("absorber"),
        )


def test_surface_provenance_is_an_explicit_geometry_argument(reference_state) -> None:
    tip, volume_mesh, state = reference_state
    with pytest.raises(TypeError, match="full3d_surface_provenance"):
        build_fingertip_volume_state_geometry(
            tip,
            state,
            carrier_mesh=make_distal_phalanx_mesh(volume_mesh.solid),
        )


def test_geometry_has_no_metadata_override_path(reference_state) -> None:
    tip, volume_mesh, state = reference_state
    geometry = build_fingertip_volume_state_geometry(
        tip,
        state,
        carrier_mesh=make_distal_phalanx_mesh(volume_mesh.solid),
        full3d_surface_provenance="actual_reference_3d_volume_state",
    )
    assert not hasattr(geometry, "metadata")


def test_carrier_contact_requires_explicit_mapping_tolerance(reference_state) -> None:
    tip, volume_mesh, state = reference_state
    contacted_node = int(volume_mesh.surface_triangles["void_bottom"][0].node_ids[0])
    with pytest.raises(ValueError, match="explicit mapping tolerance"):
        build_fingertip_volume_state_geometry(
            tip,
            state,
            carrier_mesh=make_distal_phalanx_mesh(volume_mesh.solid),
            full3d_surface_provenance="actual_reference_3d_volume_state",
            carrier_contact_source_node_ids={contacted_node},
            carrier_optics=CarrierOptics("absorber"),
        )
