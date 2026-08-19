from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gmsh")

from mesh import FingertipVolumeState, volume_mesh_settings_for_tier
from model import Fingertip
from optics.contact_object import CarrierOptics, IndenterOptics
from optics.transport3d import build_fingertip_volume_state_geometry
from optics.transport3d.geometry import CARRIER_CONTACT_INTERFACE


def _reference_state():
    tip = Fingertip()
    volume_mesh = tip.volume_mesh(volume_mesh_settings_for_tier("search"))
    return tip, volume_mesh, FingertipVolumeState.reference(volume_mesh)


def test_open_gap_keeps_all_void_triangles_as_air_or_internal() -> None:
    tip, volume_mesh, state = _reference_state()
    geometry = build_fingertip_volume_state_geometry(
        tip,
        state,
        reference_mesh=tip.mesh(),
        carrier_optics=CarrierOptics("absorber"),
    )

    assert geometry.metadata["carrier_optical_contact_triangle_count"] == 0
    assert CARRIER_CONTACT_INTERFACE not in (geometry.silicone.interface_tags or ())
    assert geometry.carrier_optics == IndenterOptics("absorber")
    assert geometry.metadata["carrier_contact_active"] is False


def test_exact_contact_vertices_map_only_semantic_void_triangles() -> None:
    tip, volume_mesh, state = _reference_state()
    triangle = volume_mesh.surface_triangles["void_bottom"][0]
    contacted_node = int(triangle.node_ids[0])
    geometry = build_fingertip_volume_state_geometry(
        tip,
        state,
        reference_mesh=tip.mesh(),
        carrier_contact_source_node_ids={contacted_node},
        carrier_optics=CarrierOptics("absorber"),
    )

    tags = np.asarray(geometry.silicone.interface_tags, dtype=object)
    semantic = np.asarray(geometry.silicone.semantic_tags, dtype=object)
    carrier = semantic == "void_bottom"
    assert int(np.count_nonzero(tags == CARRIER_CONTACT_INTERFACE)) > 0
    assert np.all(semantic[tags == CARRIER_CONTACT_INTERFACE] == "void_bottom")
    assert np.count_nonzero(tags == CARRIER_CONTACT_INTERFACE) <= np.count_nonzero(carrier)
    assert geometry.metadata["carrier_contact_active"] is True


def test_contact_triangles_require_explicit_carrier_optics() -> None:
    tip, volume_mesh, state = _reference_state()
    contacted_node = int(volume_mesh.surface_triangles["void_bottom"][0].node_ids[0])
    with pytest.raises(ValueError, match="carrier contact triangles"):
        build_fingertip_volume_state_geometry(
            tip,
            state,
            reference_mesh=tip.mesh(),
            carrier_contact_source_node_ids={contacted_node},
        )

