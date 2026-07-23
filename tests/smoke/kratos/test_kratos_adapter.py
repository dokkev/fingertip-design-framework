"""Kratos integration contract for the Phase 4M initialization smoke model."""

from __future__ import annotations

import pytest

pytest.importorskip("KratosMultiphysics")

from mesh.fingertip import generate_fingertip_mesh
from fem.kratos_adapter import run_initialization_smoke
from fem.kratos_settings import CARRIER_ELEMENT, MIXED_PAD_ELEMENT
from mesh.types import mesh_settings_for_level
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


@pytest.fixture(scope="module")
def smoke_result():
    model = FingertipModel(FingertipParameters())
    mesh = generate_fingertip_mesh(model, mesh_settings_for_level("medium"))
    return run_initialization_smoke(mesh)


def test_kratos_initialization_and_element_contract(smoke_result) -> None:
    assert smoke_result["initialization_succeeded"]
    assert smoke_result["status"] == "PASS"
    contract = smoke_result["element_runtime_contract"]
    assert contract["pad_registered_creation_name"] == MIXED_PAD_ELEMENT
    assert contract["carrier_registered_creation_name"] == CARRIER_ELEMENT
    assert contract["strategy_check_return_value"] == 0


def test_internal_contact_runtime_roles_and_nodal_h(smoke_result) -> None:
    runtime = smoke_result["runtime_contact_contract"]
    assert runtime["checks"]["all_pad_contact_conditions_are_slave"]
    assert runtime["checks"]["all_stem_contact_conditions_are_master"]
    assert runtime["checks"]["all_contact_nodal_h_finite_and_positive"]
    assert runtime["checks"]["runtime_normals_match_mesh_outward_normals"]
    assert runtime["zero_clearance_contact_node_ids_distinct"]
    for name in (
        "PadCutoutLeft",
        "PadCutoutRight",
        "PadCutoutBottom",
        "StemLeft",
        "StemRight",
        "StemBottom",
    ):
        assert runtime["surfaces"][name]["condition_count"] > 0


def test_required_submodel_parts_have_topology(smoke_result) -> None:
    parts = smoke_result["submodel_parts"]
    assert parts["PadDomain"]["elements"] > 0
    assert parts["RigidCarrier"]["elements"] > 0
    assert parts["RigidMotion"]["nodes"] == parts["RigidCarrier"]["nodes"]
    for name in (
        "PadBondLeft",
        "PadBondRight",
        "PadOuterArc",
        "PadCutoutLeft",
        "PadCutoutRight",
        "PadCutoutBottom",
        "StemLeft",
        "StemRight",
        "StemBottom",
    ):
        assert parts[name]["nodes"] > 0
        assert parts[name]["conditions"] > 0
