from __future__ import annotations

from dataclasses import asdict, replace
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
from model import (
    Fingertip,
    FingertipParameters,
    LED,
    OpticalMaterial,
    fingertip_parameters_fingerprint,
)
from optics.transport3d import (
    Transport3DResult,
    Transport3DSettings,
    UnifiedTransportResult,
)


def _mesh(
    parameters: FingertipParameters,
    settings=None,
) -> FingertipMesh:
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
        settings=(
            mesh_settings_for_level("medium") if settings is None else settings
        ),
        quality=quality,
        validation=MeshValidationReport(True, {"synthetic": True}, ()),
        gmsh_version="synthetic",
    )


def _case_parts(
    parameters: FingertipParameters | None = None,
    mesh_settings=None,
):
    parameters = FingertipParameters() if parameters is None else parameters
    tip = Fingertip(parameters)
    indenter = IndenterSettings()
    contact = ContactState(0.0, 0.5, indenter.radius_mm)
    mesh = _mesh(parameters, mesh_settings)
    displacement = np.asarray(
        [[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]],
        dtype=float,
    )
    fixture = build_normal_indenter_fixture_at_x(
        tip.geometry, contact.location_x_mm, indenter
    )
    pose = pose_from_fixture(fixture, contact.indentation_mm)
    fea = FEAResult(
        mesh=mesh.pad,
        displacement=displacement,
        reaction_force=0.5,
        contact={"external_pad_indenter": {"active": True}},
        converged=True,
        details={"solver": "synthetic", "contact_state": asdict(contact)},
        indenter_pose=pose,
        reference_mesh=mesh,
    )
    morphology = fingertip_parameters_fingerprint(parameters)
    contact_payload = {
        "contact_mode": "explicit_contact_2d",
        "location_x_mm": contact.location_x_mm,
        "indentation_mm": contact.indentation_mm,
        "indenter_radius_mm": contact.indenter_radius_mm,
        "initial_gap_mm": indenter.initial_gap_mm,
    }
    raytrace = Transport3DResult(
        source_position_mm=(0.0, -6.0, 0.0),
        source_mode="planar",
        extrusion_depth_mm=11.0,
        launched_ray_count=3,
        launched_weight=1.0,
        escaped_weight=0.5,
        absorbed_weight=0.5,
        terminated_weight=0.0,
        outgoing_surface_weight=0.5,
        surface_u_edges=np.asarray([0.0, 1.0, 2.0]),
        surface_z_edges=np.asarray([-1.0, 1.0]),
        outgoing_surface_field=np.ones((1, 2), dtype=float),
        escape_positions_mm=np.asarray([[0.5, 0.0, 0.0]]),
        escape_directions=np.asarray([[0.0, 1.0, 0.0]]),
        escape_surface_normals=np.asarray([[0.0, -1.0, 0.0]]),
        escape_surface_u=np.asarray([0.5]),
        escape_surface_z=np.asarray([0.0]),
        escape_surface_tags=("pad_outer_arc",),
        escape_surface_primitive_indices=np.asarray([0]),
        escape_weights=np.asarray([0.5]),
        escape_primary_ray_indices=np.asarray([0]),
        escape_path_lengths_mm=np.asarray([1.0]),
        escape_interaction_counts=np.asarray([1]),
        energy_balance_error=0.0,
        energy_balance_tolerance=1.0e-6,
        projected_x_edges_mm=np.asarray([0.0, 1.0, 2.0]),
        projected_y_edges_mm=np.asarray([0.0, 1.0]),
        projected_weighted_path_density=np.asarray([[1.0, 2.0]], dtype=float),
        geometry_metadata={"source": "synthetic"},
    )
    trace_settings = Transport3DSettings(
        mode="planar",
        ray_count=3,
        surface_u_bins=4,
        surface_z_bins=2,
        projected_grid_width=16,
        projected_grid_height=16,
        internal_z_bins=2,
        retain_projected_segments=True,
    )
    configuration = case_module.transport_configuration(
        trace_settings,
        material=case_module._optical_material_mapping(OpticalMaterial()),
        source={"led": asdict(LED())},
    )
    optics = UnifiedTransportResult.from_transport_result(
        raytrace,
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
        transport_configuration_fingerprint=case_module.fingerprint_mapping(
            configuration
        ),
    )
    return parameters, indenter, contact, fea, raytrace, optics


def _case(**kwargs) -> FingertipCase:
    requested_mesh_settings = kwargs.get("mesh_settings")
    parameters, indenter, contact, fea, raytrace, optics = _case_parts(
        kwargs.pop("parameters", None),
        requested_mesh_settings,
    )
    led = kwargs.pop("led", LED())
    optical = kwargs.pop("optical", OpticalMaterial())
    mesh_settings = kwargs.pop("mesh_settings", fea.reference_mesh.settings)
    fem_steps = kwargs.pop("fem_steps", 48)
    internal_contact = kwargs.pop("internal_contact", "three_pairs")
    trace_settings = kwargs.pop(
        "trace_settings",
        Transport3DSettings(
            mode="planar",
            ray_count=3,
            surface_u_bins=4,
            surface_z_bins=2,
            projected_grid_width=16,
            projected_grid_height=16,
            internal_z_bins=2,
            retain_projected_segments=True,
        ),
    )
    supplied_optics = kwargs.pop("optics", None)
    if supplied_optics is None:
        configuration = case_module.transport_configuration(
            trace_settings,
            material=case_module._optical_material_mapping(optical),
            source={"led": asdict(led)},
        )
        optics = replace(
            optics,
            transport_configuration_fingerprint=case_module.fingerprint_mapping(
                configuration
            ),
        )
    else:
        optics = supplied_optics
    return FingertipCase(
        fingertip_parameters=kwargs.pop("fingertip_parameters", parameters),
        indenter_parameters=kwargs.pop("indenter_parameters", indenter),
        contact_state=kwargs.pop("contact_state", contact),
        fea=kwargs.pop("fea", fea),
        raytrace=kwargs.pop("raytrace", raytrace),
        optics=optics,
        led=led,
        optical=optical,
        mesh_settings=mesh_settings,
        fem_steps=fem_steps,
        internal_contact=internal_contact,
        trace_settings=trace_settings,
        provenance=kwargs.pop("provenance", {"test": "synthetic"}),
        **kwargs,
    )


def test_case_stores_neutral_inputs_and_delegates_results() -> None:
    case = _case()
    parameters = case.fingertip_parameters
    indenter = case.indenter_parameters
    fea = case.fea
    raytrace = case.raytrace
    optics = case.optics
    assert case.fingertip_parameters is parameters
    assert case.indenter_parameters is indenter
    assert case.fea is fea
    assert case.raytrace is raytrace
    assert case.optics is optics
    assert case.displacement is fea.displacement
    assert case.optical_field is optics.field
    assert case.reaction_force == fea.reaction_force
    assert case.escaped_weight == raytrace.escaped_weight
    assert case.indenter_pose is fea.indenter_pose


def test_case_rejects_mismatched_fea_parameters() -> None:
    base = _case()
    parameters = base.fingertip_parameters
    fea = base.fea
    mismatched_mesh = _mesh(FingertipParameters(void_width=1.2))
    mismatched_fea = FEAResult(
        mesh=mismatched_mesh.pad,
        displacement=fea.displacement,
        reaction_force=fea.reaction_force,
        contact=fea.contact,
        converged=True,
        details=fea.details,
        indenter_pose=fea.indenter_pose,
        reference_mesh=mismatched_mesh,
    )
    with pytest.raises(ValueError, match="FEA reference_mesh parameters"):
        FingertipCase(
            fingertip_parameters=parameters,
            indenter_parameters=base.indenter_parameters,
            contact_state=base.contact_state,
            fea=mismatched_fea,
            raytrace=base.raytrace,
            optics=base.optics,
            led=base.led,
            optical=base.optical,
            mesh_settings=base.mesh_settings,
            fem_steps=base.fem_steps,
            internal_contact=base.internal_contact,
            trace_settings=base.trace_settings,
        )


def test_case_rejects_mismatched_raytrace_morphology() -> None:
    base = _case()
    parameters, indenter, contact, fea, raytrace, optics = (
        base.fingertip_parameters,
        base.indenter_parameters,
        base.contact_state,
        base.fea,
        base.raytrace,
        base.optics,
    )
    bad = UnifiedTransportResult(
        **{
            name: getattr(optics, name)
            for name in optics.__dataclass_fields__
            if name != "morphology_fingerprint"
        },
        morphology_fingerprint="wrong",
    )
    with pytest.raises(ValueError, match="morphology fingerprint"):
        _case(optics=bad)


def test_case_rejects_mismatched_contact_provenance() -> None:
    base = _case()
    parameters, indenter, contact, fea, raytrace, optics = (
        base.fingertip_parameters,
        base.indenter_parameters,
        base.contact_state,
        base.fea,
        base.raytrace,
        base.optics,
    )
    payload = dict(optics.contact_state)
    payload["indentation_mm"] = 0.75
    bad = UnifiedTransportResult(
        **{
            name: getattr(optics, name)
            for name in optics.__dataclass_fields__
            if name != "contact_state"
        },
        contact_state=payload,
    )
    with pytest.raises(ValueError, match="contact provenance"):
        _case(optics=bad)


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
    assert loaded.led == original.led
    assert loaded.optical == original.optical
    assert loaded.mesh_settings == original.mesh_settings
    assert loaded.fem_steps == original.fem_steps
    assert loaded.internal_contact == original.internal_contact
    assert loaded.trace_settings == original.trace_settings
    np.testing.assert_allclose(loaded.displacement, original.displacement)
    np.testing.assert_allclose(loaded.optical_field, original.optical_field)
    np.testing.assert_allclose(
        loaded.raytrace.escape_positions_mm,
        original.raytrace.escape_positions_mm,
    )
    np.testing.assert_allclose(
        loaded.raytrace.escape_directions,
        original.raytrace.escape_directions,
    )
    np.testing.assert_allclose(
        loaded.raytrace.escape_weights,
        original.raytrace.escape_weights,
    )
    assert loaded.fea.reference_mesh == original.fea.reference_mesh
    assert loaded.fea.mesh is loaded.fea.reference_mesh.pad
    assert dict(loaded.provenance) == dict(original.provenance)


def test_case_id_excludes_provenance_notes() -> None:
    first = _case(provenance={"test": "synthetic", "note": "first"})
    second = _case(provenance={"test": "synthetic", "note": "second"})
    assert first.case_id == second.case_id


def test_case_id_includes_run_and_optical_configuration() -> None:
    base = _case()
    custom_led = _case(led=LED(relative_radiant_power=2.0))
    assert base.case_id != _case(fem_steps=96).case_id
    assert base.case_id != _case(internal_contact="continuous_u").case_id
    assert base.case_id != _case(
        mesh_settings=mesh_settings_for_level("fine"),
    ).case_id
    assert base.case_id != custom_led.case_id
    assert (
        base.optics.transport_configuration_fingerprint
        != custom_led.optics.transport_configuration_fingerprint
    )
    assert base.case_id != _case(
        optical=OpticalMaterial(absorption_per_mm=0.03),
    ).case_id
    assert base.case_id != _case(
        trace_settings=Transport3DSettings(
            mode="planar",
            ray_count=5,
            surface_u_bins=4,
            surface_z_bins=2,
            projected_grid_width=16,
            projected_grid_height=16,
            internal_z_bins=2,
            retain_projected_segments=True,
        ),
    ).case_id


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
    parameters, indenter, contact, fea, raytrace, _ = _case_parts()
    real_tip = Fingertip(parameters)
    synthetic_mesh = fea.reference_mesh
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

    monkeypatch.setattr(case_module, "Fingertip", lambda *args, **kwargs: fake_tip)
    monkeypatch.setattr(case_module, "solve", fake_solve)
    monkeypatch.setattr(case_module, "build_transport_geometry", lambda *args, **kwargs: object())

    def fake_trace_geometry(tip, geometry, **kwargs):
        calls["trace"] = kwargs
        return raytrace

    monkeypatch.setattr(case_module, "trace_geometry", fake_trace_geometry)

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
