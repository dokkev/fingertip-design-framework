"""Neutral aggregate and orchestration for one 2D contact case."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from types import MappingProxyType
from typing import Any, Mapping

from fem import FEAResult, solve
from mesh import MeshSettings, mesh_settings_for_level
from mesh.indenter import IndenterPose2D, IndenterSettings
from model import Fingertip, FingertipParameters, LED, OpticalMaterial
from optimization.scenarios import ContactScenario
from optics.transport3d import (
    OptiXTransport,
    Transport3DSettings,
    UnifiedTransportResult,
    transport_configuration,
    fingerprint_mapping,
)
from optics.transport3d.geometry import build_transport_geometry


CASE_SCHEMA = "fingertip-case-v1"
ContactState = ContactScenario


class CaseConstructionError(RuntimeError):
    """Raised when one complete case cannot satisfy its neutral contract."""


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def contact_state_contract(
    contact_state: ContactState,
    indenter_parameters: IndenterSettings,
) -> dict[str, Any]:
    """Return the canonical state identity shared by mechanics and optics."""
    if not isinstance(contact_state, ContactScenario):
        raise TypeError("contact_state must be a ContactScenario")
    if not isinstance(indenter_parameters, IndenterSettings):
        raise TypeError("indenter_parameters must be IndenterSettings")
    if not math.isclose(
        contact_state.indenter_radius_mm,
        indenter_parameters.radius_mm,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "contact_state.indenter_radius_mm must match "
            "indenter_parameters.radius_mm"
        )
    return {
        "contact_mode": "explicit_contact_2d",
        "location_x_mm": float(contact_state.location_x_mm),
        "indentation_mm": float(contact_state.indentation_mm),
        "indenter_radius_mm": float(contact_state.indenter_radius_mm),
        "initial_gap_mm": float(indenter_parameters.initial_gap_mm),
    }


def _expected_morphology_fingerprint(parameters: FingertipParameters) -> str:
    from model.solid import build_fingertip_solid

    return build_fingertip_solid(Fingertip(parameters).geometry).morphology_fingerprint


def _case_identity_payload(case: "FingertipCase") -> dict[str, Any]:
    return {
        "schema": CASE_SCHEMA,
        "fingertip_parameters": asdict(case.fingertip_parameters),
        "indenter_parameters": (
            None
            if case.indenter_parameters is None
            else asdict(case.indenter_parameters)
        ),
        "contact_state": asdict(case.contact_state),
        "mechanics": {
            "mesh_level": case.fea.mesh.settings.level,
            "morphology_fingerprint": _expected_morphology_fingerprint(
                case.fingertip_parameters
            ),
            "indenter_pose": case.indenter_pose.to_dict(),
        },
        "raytrace": {
            "morphology_fingerprint": case.raytrace.morphology_fingerprint,
            "mechanics_source": case.raytrace.mechanics_source,
            "mechanics_dimension": case.raytrace.mechanics_dimension,
            "optical_mode": case.raytrace.optical_mode,
            "ray_count": case.raytrace.ray_count,
            "transport_configuration_fingerprint": (
                case.raytrace.transport_configuration_fingerprint
            ),
        },
        "provenance": dict(case.provenance),
    }


def case_id_for(case: "FingertipCase") -> str:
    """Return a deterministic ID from physical inputs and result provenance."""
    if not isinstance(case, FingertipCase):
        raise TypeError("case must be a FingertipCase")
    return f"case-{fingerprint_mapping(_case_identity_payload(case))[:24]}"


@dataclass(frozen=True)
class FingertipCase:
    """One immutable morphology/contact state and its neutral results."""

    fingertip_parameters: FingertipParameters
    indenter_parameters: IndenterSettings | None
    contact_state: ContactState
    fea: FEAResult
    raytrace: UnifiedTransportResult
    case_id: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.fingertip_parameters, FingertipParameters):
            raise TypeError("fingertip_parameters must be FingertipParameters")
        if self.indenter_parameters is not None and not isinstance(
            self.indenter_parameters, IndenterSettings
        ):
            raise TypeError("indenter_parameters must be IndenterSettings or None")
        if self.indenter_parameters is None:
            raise ValueError("a contact case requires indenter_parameters")
        if not isinstance(self.contact_state, ContactScenario):
            raise TypeError("contact_state must be a ContactScenario")
        if not isinstance(self.fea, FEAResult):
            raise TypeError("fea must be an FEAResult")
        if not isinstance(self.raytrace, UnifiedTransportResult):
            raise TypeError("raytrace must be a UnifiedTransportResult")
        if not self.fea.converged or self.fea.indenter_pose is None:
            raise ValueError("a FingertipCase requires converged FEA with indenter_pose")
        fea_parameters = getattr(self.fea.mesh, "parameters", None)
        if fea_parameters != self.fingertip_parameters:
            raise ValueError("FEA mesh parameters do not match fingertip_parameters")

        expected_morphology = _expected_morphology_fingerprint(
            self.fingertip_parameters
        )
        if self.raytrace.morphology_fingerprint != expected_morphology:
            raise ValueError("raytrace morphology fingerprint does not match the case")
        if self.raytrace.optical_mode != "PLANAR_2D":
            raise ValueError("a FingertipCase requires a PLANAR_2D optical result")
        if self.raytrace.mechanics_dimension != "2D":
            raise ValueError("PLANAR_2D case optics must be paired with 2D mechanics")
        if self.raytrace.mechanics_source != "explicit_contact_fea":
            raise ValueError(
                "FingertipCase requires the explicit-contact mechanics provenance"
            )

        pose = self.fea.indenter_pose
        assert pose is not None
        if pose.fixture.settings != self.indenter_parameters:
            raise ValueError("indenter pose fixture does not match case indenter parameters")
        if not math.isclose(
            pose.fixture.frame.point_mm[0],
            self.contact_state.location_x_mm,
            rel_tol=0.0,
            abs_tol=max(1.0e-8, 10.0 * self.fingertip_parameters.geometry_tolerance),
        ):
            raise ValueError("indenter pose location does not match contact_state")
        if not math.isclose(
            pose.prescribed_travel_mm,
            self.contact_state.indentation_mm,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("indenter pose travel does not match contact_state")
        expected_contact = contact_state_contract(
            self.contact_state,
            self.indenter_parameters,
        )
        observed_contact = dict(self.raytrace.contact_state)
        for key, expected in expected_contact.items():
            observed = observed_contact.get(key)
            if isinstance(expected, float):
                if observed is None or not math.isclose(
                    float(observed), expected, rel_tol=0.0, abs_tol=1.0e-12
                ):
                    raise ValueError(f"raytrace contact provenance mismatch: {key}")
            elif observed != expected:
                raise ValueError(f"raytrace contact provenance mismatch: {key}")
        expected_contact_fingerprint = fingerprint_mapping(expected_contact)
        if observed_contact.get("contact_state_fingerprint") not in (
            None,
            expected_contact_fingerprint,
        ):
            raise ValueError("raytrace contact-state fingerprint does not match the case")

        provenance = self.provenance if isinstance(self.provenance, Mapping) else None
        if provenance is None:
            raise TypeError("provenance must be a mapping")
        object.__setattr__(self, "provenance", _freeze_mapping(provenance))
        expected_case_id = case_id_for(self)
        if self.case_id and self.case_id != expected_case_id:
            raise ValueError("case_id is not deterministic for the supplied case")
        object.__setattr__(self, "case_id", expected_case_id)

    @property
    def parameters(self) -> FingertipParameters:
        return self.fingertip_parameters

    @property
    def indenter(self) -> IndenterSettings:
        assert self.indenter_parameters is not None
        return self.indenter_parameters

    @property
    def displacement(self):
        return self.fea.displacement

    @property
    def deformed_mesh(self):
        return self.fea.deformed_mesh

    @property
    def reaction_force(self) -> float | None:
        return self.fea.reaction_force

    @property
    def contact(self) -> Mapping[str, Any]:
        return self.fea.contact

    @property
    def indenter_pose(self) -> IndenterPose2D:
        assert self.fea.indenter_pose is not None
        return self.fea.indenter_pose

    @property
    def optical_field(self):
        return self.raytrace.field

    @property
    def escaped_weight(self) -> float:
        return self.raytrace.escaped_weight


def _optical_material_mapping(material: OpticalMaterial) -> dict[str, float]:
    return {
        "refractive_index_air": material.refractive_index_air,
        "refractive_index_silicone": material.refractive_index_silicone,
        "absorption_per_mm": material.absorption_per_mm,
        "scattering_per_mm": material.scattering_per_mm,
        "anisotropy_g": material.anisotropy_g,
    }


def run_case(
    *,
    fingertip_parameters: FingertipParameters,
    indenter_parameters: IndenterSettings,
    contact_state: ContactState,
    mesh_settings: MeshSettings | None = None,
    trace_settings: Transport3DSettings | None = None,
    led: LED | None = None,
    optical: OpticalMaterial | None = None,
    fem_steps: int = 48,
    internal_contact: str = "three_pairs",
    provenance: Mapping[str, Any] | None = None,
    optix_runtime: Any | None = None,
) -> FingertipCase:
    """Run explicit-contact 2D FEA followed by the public PLANAR_2D OptiX path."""
    if not isinstance(fingertip_parameters, FingertipParameters):
        raise TypeError("fingertip_parameters must be FingertipParameters")
    if not isinstance(indenter_parameters, IndenterSettings):
        raise TypeError("indenter_parameters must be IndenterSettings")
    if not isinstance(contact_state, ContactScenario):
        raise TypeError("contact_state must be a ContactScenario")
    if mesh_settings is None:
        mesh_settings = mesh_settings_for_level("medium")
    if not isinstance(mesh_settings, MeshSettings):
        raise TypeError("mesh_settings must be MeshSettings")
    if trace_settings is None:
        trace_settings = Transport3DSettings(mode="planar")
    if not isinstance(trace_settings, Transport3DSettings):
        raise TypeError("trace_settings must be Transport3DSettings")
    if trace_settings.mode != "planar":
        raise ValueError("run_case is the 2D production path and requires mode='planar'")
    if not trace_settings.retain_projected_segments:
        # PLANAR_2D's authoritative neutral result is the native P2 field.
        # Retaining it is an output contract, not a change to ray physics.
        trace_settings = replace(trace_settings, retain_projected_segments=True)
    state = contact_state_contract(contact_state, indenter_parameters)
    tip = Fingertip(
        fingertip_parameters,
        led=LED() if led is None else led,
        optical=OpticalMaterial() if optical is None else optical,
    )
    mesh = tip.mesh(mesh_settings)

    # This is the existing authoritative explicit-contact solver.  The
    # localized-load surrogate is intentionally not reachable from this path.
    fea = solve(
        tip,
        mesh,
        indentation=contact_state.indentation_mm,
        surface_x_mm=contact_state.location_x_mm,
        steps=fem_steps,
        indenter=indenter_parameters,
        internal_contact=internal_contact,
    )
    if not fea.converged or fea.indenter_pose is None:
        raise CaseConstructionError(
            "explicit-contact FEA did not produce a converged pose: "
            f"{fea.details.get('failure_reason', 'unknown failure')}"
        )

    geometry = build_transport_geometry(
        tip,
        fea.deformed_mesh,
        mesh,
        depth_mm=trace_settings.extrusion_depth_mm,
        source_epsilon_mm=trace_settings.source_epsilon_mm,
    )
    configuration = transport_configuration(
        trace_settings,
        material=_optical_material_mapping(tip.optical),
    )
    contact_provenance = {
        **state,
        "contact_state_fingerprint": fingerprint_mapping(state),
    }
    raytrace = OptiXTransport().trace(
        tip,
        geometry,
        settings=trace_settings,
        morphology_id="custom",
        morphology_fingerprint=_expected_morphology_fingerprint(
            fingertip_parameters
        ),
        mechanics_source="explicit_contact_fea",
        mechanics_dimension="2D",
        contact_state=contact_provenance,
        transport_configuration=configuration,
        runtime=optix_runtime,
    )
    case_provenance = {
        "mechanics_path": "fem.solve.solve -> fem.indentation.run_indentation_case",
        "mechanics_mode": "explicit_contact_2d",
        "localized_load_used": False,
        "mesh_level": mesh_settings.level,
        "fem_steps": fem_steps,
        "internal_contact": internal_contact,
        "optical_backend": "OptiXTransport",
        "optical_mode": "PLANAR_2D",
        "indenter_optically_active": False,
        **dict(provenance or {}),
    }
    return FingertipCase(
        fingertip_parameters=fingertip_parameters,
        indenter_parameters=indenter_parameters,
        contact_state=contact_state,
        fea=fea,
        raytrace=raytrace,
        provenance=case_provenance,
    )


__all__ = [
    "CASE_SCHEMA",
    "CaseConstructionError",
    "ContactState",
    "FingertipCase",
    "case_id_for",
    "contact_state_contract",
    "run_case",
]
