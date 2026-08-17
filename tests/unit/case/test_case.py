from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

import case.core as case_module
from case import ContactState, FingertipCase, load_case, run_case, save_case
from fem import FEAResult
from mesh.indenter import IndenterSettings, pose_from_fixture, build_normal_indenter_fixture_at_x
from mesh.types import (
    BoundaryEdge,
    FingertipMesh,
    MeshQualityStatistics,
    MeshValidationReport,
    MeshedContactPair,
    MeshNode,
    T3Element,
    mesh_settings_for_level,
)
from model import Fingertip, FingertipParameters
from model.solid import build_fingertip_solid
from optics.transport3d import UnifiedTransportResult


def _mesh(parameters: FingertipParameters) -> FingertipMesh:
    nodes = {
        1: MeshNode(1, 0.0, 0.0, "pad"),
        2: MeshNode(2, 1.0, 0.0, "pad"),
        3: MeshNode(3, 0.0, 1.0, "pad"),
        4: MeshNode(4, 0.0, 0.0, "rigid_carrier"),
        5: MeshNode(5, 1.0, 0.0, "rigid_carrier"),
        6: MeshNode(6, 0.0, 1.0, "rigid_carrier"),
    }
    pad_edges = {
        "pad_outer_left": (BoundaryEdge((1, 2), "pad"),),
        "pad_outer_arc": (BoundaryEdge((2, 3), "pad"),),
        "pad_outer_right": (BoundaryEdge((3, 1), "pad"),),
    }
    quality = MeshQualityStatistics(
        node_count=6,
        pad_node_count=3,
        carrier_node_count=3,
        t3_element_count=2,
        pad_t3_element_count=1,
        carrier_t3_element_count=1,
        minimum_triangle_angle_degrees=45.0,
        minimum_triangle_angle_element_id=1,
        minimum_triangle_angle_centroid_mm=(1.0 / 3.0, 1.0 / 3.0),
        maximum_edge_length_mm=1.0,
        pad_mesh_area_mm2=0.5,
        pad_geometry_area_mm2=0.5,
        pad_area_relative_error=0.0,
        carrier_mesh_area_mm2=0.5,
        carrier_geometry_area_mm2=0.5,
        carrier_area_relative_error=0.0,
        orphan_node_count=0,
        duplicate_element_count=0,
        nonpositive_area_element_count=0,
    )
    return FingertipMesh(
        nodes=nodes,
        pad_elements=(T3Element(1, (1, 2, 3), "pad"),),
        carrier_elements=(T3Element(2, (4, 5, 6), "rigid_carrier"),),
        domain_node_ids={"pad": (1, 2, 3), "rigid_carrier": (4, 5, 6)},
        domain_element_ids={"pad": (1,), "rigid_carrier": (2,)},
        boundary_edges=pad_edges,
        contact_pairs=(
            MeshedContactPair("synthetic", "pad_outer_arc", "stem_bottom", 0.0, 0.0),
        ),
        parameters=parameters,
        settings=mesh_settings_for_level("medium"),
        quality=quality,
        validation=MeshValidationReport(True, {"synthetic": True}, ()),
        gmsh_version="synthetic",
    )


def _case_parts(parameters: FingertipParameters | None = None):
    parameters = FingertipParameters() if parameters is None else parameters
    tip = Fingertip(parameters)
    indenter = IndenterSettings()
    contact = ContactState(0.0, 0.5, indenter.radius_mm)
    mesh = _mesh(parameters)
    displacement = np.asarray(
        [[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]],
        dtype=float,
    )
    fixture = build_normal_indenter_fixture_at_x(
        tip.geometry, contact.location_x_mm, indenter
    )
    pose = pose_from_fixture(fixture, contact.indentation_mm)
    fea = FEAResult(
        mesh=mesh,
        displacement=displacement,
        reaction_force=0.5,
        contact={"external_pad_indenter": {"active": True}},
        converged=True,
        details={"solver": "synthetic", "contact_state": asdict(contact)},
        indenter_pose=pose,
    )
    morphology = build_fingertip_solid(tip.geometry).morphology_fingerprint
    contact_payload = {
        "contact_mode": "explicit_contact_2d",
        "location_x_mm": contact.location_x_mm,
        "indentation_mm": contact.indentation_mm,
        "indenter_radius_mm": contact.indenter_radius_mm,
        "initial_gap_mm": indenter.initial_gap_mm,
    }
    raytrace = UnifiedTransportResult(
        morphology_id="synthetic",
        morphology_fingerprint=morphology,
        mechanics_source="explicit_contact_fea",
        mechanics_dimension="2D",
        contact_state={
            **contact_payload,
            "contact_state_fingerprint": case_module.fingerprint_mapping(
                contact_payload
            ),
        },
        optical_mode="PLANAR_2D",
        ray_count=3,
        transport_configuration_fingerprint="synthetic-config",
        field=np.ones((2, 2), dtype=float),
        field_axes=(np.asarray([0.0, 1.0, 2.0]), np.asarray([0.0, 1.0, 2.0])),
        total_transport=1.0,
        launched_weight=1.0,
        escaped_weight=0.5,
        absorbed_weight=0.5,
        terminated_weight=0.0,
        valid_ray_count=1,
        terminated_ray_count=2,
        energy_balance_error=0.0,
        path_diagnostics={"source": "synthetic"},
    )
    return parameters, indenter, contact, fea, raytrace


def _case(**kwargs) -> FingertipCase:
    parameters, indenter, contact, fea, raytrace = _case_parts(
        kwargs.pop("parameters", None)
    )
    return FingertipCase(
        fingertip_parameters=kwargs.pop("fingertip_parameters", parameters),
        indenter_parameters=kwargs.pop("indenter_parameters", indenter),
        contact_state=kwargs.pop("contact_state", contact),
        fea=kwargs.pop("fea", fea),
        raytrace=kwargs.pop("raytrace", raytrace),
        provenance=kwargs.pop("provenance", {"test": "synthetic"}),
        **kwargs,
    )


def test_case_stores_neutral_inputs_and_delegates_results() -> None:
    parameters, indenter, contact, fea, raytrace = _case_parts()
    case = FingertipCase(
        fingertip_parameters=parameters,
        indenter_parameters=indenter,
        contact_state=contact,
        fea=fea,
        raytrace=raytrace,
        provenance={"test": "synthetic"},
    )
    assert case.fingertip_parameters is parameters
    assert case.indenter_parameters is indenter
    assert case.fea is fea
    assert case.raytrace is raytrace
    assert case.displacement is fea.displacement
    assert case.optical_field is raytrace.field
    assert case.reaction_force == fea.reaction_force
    assert case.escaped_weight == raytrace.escaped_weight
    assert case.indenter_pose is fea.indenter_pose


def test_case_rejects_mismatched_fea_parameters() -> None:
    parameters, _, _, fea, raytrace = _case_parts()
    mismatched_mesh = _mesh(FingertipParameters(void_width=1.2))
    mismatched_fea = FEAResult(
        mesh=mismatched_mesh,
        displacement=fea.displacement,
        reaction_force=fea.reaction_force,
        contact=fea.contact,
        converged=True,
        details=fea.details,
        indenter_pose=fea.indenter_pose,
    )
    with pytest.raises(ValueError, match="FEA mesh parameters"):
        FingertipCase(parameters, IndenterSettings(), ContactState(0.0, 0.5, 4.0), mismatched_fea, raytrace)


def test_case_rejects_mismatched_raytrace_morphology() -> None:
    parameters, indenter, contact, fea, raytrace = _case_parts()
    bad = UnifiedTransportResult(
        **{
            name: getattr(raytrace, name)
            for name in raytrace.__dataclass_fields__
            if name != "morphology_fingerprint"
        },
        morphology_fingerprint="wrong",
    )
    with pytest.raises(ValueError, match="morphology fingerprint"):
        FingertipCase(parameters, indenter, contact, fea, bad)


def test_case_rejects_mismatched_contact_provenance() -> None:
    parameters, indenter, contact, fea, raytrace = _case_parts()
    payload = dict(raytrace.contact_state)
    payload["indentation_mm"] = 0.75
    bad = UnifiedTransportResult(
        **{
            name: getattr(raytrace, name)
            for name in raytrace.__dataclass_fields__
            if name != "contact_state"
        },
        contact_state=payload,
    )
    with pytest.raises(ValueError, match="contact provenance"):
        FingertipCase(parameters, indenter, contact, fea, bad)


def test_case_id_is_deterministic() -> None:
    first = _case()
    second = _case()
    assert first.case_id == second.case_id


def test_case_manifest_round_trip_and_references(tmp_path) -> None:
    original = _case()
    manifest = save_case(original, tmp_path / "cases")
    loaded = load_case(manifest)
    assert manifest == tmp_path / "cases" / original.case_id / "case.json"
    assert loaded.case_id == original.case_id
    assert loaded.fingertip_parameters == original.fingertip_parameters
    assert loaded.indenter_parameters == original.indenter_parameters
    assert loaded.contact_state == original.contact_state
    np.testing.assert_allclose(loaded.displacement, original.displacement)
    np.testing.assert_allclose(loaded.optical_field, original.optical_field)
    assert dict(loaded.provenance) == dict(original.provenance)


def test_case_manifest_rejects_mismatched_optics_artifact(tmp_path) -> None:
    original = _case()
    manifest_path = save_case(original, tmp_path / "cases")
    optical_path = manifest_path.parent / "optics" / "raytrace.json"
    optical = json.loads(optical_path.read_text(encoding="utf-8"))
    optical["result"]["contact_state"]["indentation_mm"] = 0.75
    optical_path.write_text(json.dumps(optical, indent=2, sort_keys=True) + "\n")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["optical"]["artifact_sha256"] = hashlib.sha256(
        optical_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="contact provenance"):
        load_case(manifest_path)


def test_pose_tracks_surface_location_and_radius() -> None:
    tip = Fingertip(FingertipParameters())
    first = build_normal_indenter_fixture_at_x(
        tip.geometry, -3.0, IndenterSettings(radius_mm=4.0)
    )
    second = build_normal_indenter_fixture_at_x(
        tip.geometry, 3.0, IndenterSettings(radius_mm=5.0)
    )
    assert first.frame.point_mm[0] == pytest.approx(-3.0)
    assert second.frame.point_mm[0] == pytest.approx(3.0)
    assert first.settings.radius_mm != second.settings.radius_mm
    assert first.center_mm != second.center_mm


def test_run_case_routes_to_existing_explicit_contact_solver(monkeypatch) -> None:
    parameters, indenter, contact, fea, raytrace = _case_parts()
    real_tip = Fingertip(parameters)
    synthetic_mesh = fea.mesh
    calls = {}

    fake_tip = SimpleNamespace(
        geometry=real_tip.geometry,
        led=real_tip.led,
        optical=real_tip.optical,
        mesh=lambda settings: synthetic_mesh,
    )

    def fake_solve(tip, mesh, **kwargs):
        calls["tip"] = tip
        calls["mesh"] = mesh
        calls["kwargs"] = kwargs
        return fea

    class FakeOptiX:
        def trace(self, tip, geometry, **kwargs):
            calls["trace"] = kwargs
            return raytrace

    monkeypatch.setattr(case_module, "Fingertip", lambda *args, **kwargs: fake_tip)
    monkeypatch.setattr(case_module, "solve", fake_solve)
    monkeypatch.setattr(case_module, "build_transport_geometry", lambda *args, **kwargs: object())
    monkeypatch.setattr(case_module, "OptiXTransport", FakeOptiX)

    result = run_case(
        fingertip_parameters=parameters,
        indenter_parameters=indenter,
        contact_state=contact,
        fem_steps=3,
    )
    assert result.fea is fea
    assert calls["kwargs"]["indentation"] == contact.indentation_mm
    assert calls["kwargs"]["surface_x_mm"] == contact.location_x_mm
    assert calls["kwargs"]["indenter"] is indenter
    assert calls["kwargs"]["steps"] == 3
    assert calls["kwargs"]["internal_contact"] == "three_pairs"
    assert result.provenance["localized_load_used"] is False


def test_pose_is_neutral_and_contains_no_kratos_object() -> None:
    case = _case()
    pose = case.indenter_pose
    assert pose.prescribed_travel_mm == pytest.approx(0.5)
    assert "Kratos" not in repr(pose)
    assert "Kratos" not in repr(case)
