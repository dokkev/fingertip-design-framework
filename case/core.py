"""Neutral aggregate and orchestration for one 2D contact case."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fem import FEAResult, solve
from case.state import ContactState
from mesh import MeshSettings, mesh_settings_for_level
from mesh.indenter import IndenterPose2D, IndenterSettings
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
    transport_configuration,
    fingerprint_mapping,
    trace_geometry,
)
from optics.transport3d.geometry import build_transport_geometry


CASE_SCHEMA = "fingertip-case-v2"


class CaseConstructionError(RuntimeError):
    """Raised when one complete case cannot satisfy its neutral contract."""


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def contact_state_contract(
    contact_state: ContactState,
    indenter_parameters: IndenterSettings,
) -> dict[str, Any]:
    """Return the canonical state identity shared by mechanics and optics."""
    if not isinstance(contact_state, ContactState):
        raise TypeError("contact_state must be a ContactState")
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
    return fingertip_parameters_fingerprint(parameters)


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
            "mesh_settings": asdict(case.mesh_settings),
            "fem_steps": case.fem_steps,
            "internal_contact": case.internal_contact,
            "morphology_fingerprint": _expected_morphology_fingerprint(
                case.fingertip_parameters
            ),
        },
        "optics": {
            "led": asdict(case.led),
            "optical_material": asdict(case.optical),
            "trace_settings": asdict(case.trace_settings),
            "morphology_fingerprint": case.optics.morphology_fingerprint,
            "mechanics_source": case.optics.mechanics_source,
            "mechanics_dimension": case.optics.mechanics_dimension,
            "optical_mode": case.optics.optical_mode,
            "ray_count": case.optics.ray_count,
            "transport_configuration_fingerprint": (
                case.optics.transport_configuration_fingerprint
            ),
        },
    }


def case_id_for(case: "FingertipCase") -> str:
    """Return a deterministic ID from physical and numerical inputs."""
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
    raytrace: Transport3DResult
    optics: UnifiedTransportResult
    led: LED
    optical: OpticalMaterial
    mesh_settings: MeshSettings
    fem_steps: int
    internal_contact: str
    trace_settings: Transport3DSettings
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
        if not isinstance(self.contact_state, ContactState):
            raise TypeError("contact_state must be a ContactState")
        if not isinstance(self.fea, FEAResult):
            raise TypeError("fea must be an FEAResult")
        if not isinstance(self.raytrace, Transport3DResult):
            raise TypeError("raytrace must be a Transport3DResult")
        if not isinstance(self.optics, UnifiedTransportResult):
            raise TypeError("optics must be a UnifiedTransportResult")
        if not isinstance(self.led, LED):
            raise TypeError("led must be an LED")
        if not isinstance(self.optical, OpticalMaterial):
            raise TypeError("optical must be an OpticalMaterial")
        if not isinstance(self.mesh_settings, MeshSettings):
            raise TypeError("mesh_settings must be MeshSettings")
        if (
            not isinstance(self.fem_steps, int)
            or isinstance(self.fem_steps, bool)
            or self.fem_steps <= 0
        ):
            raise ValueError("fem_steps must be a positive integer")
        if not isinstance(self.internal_contact, str) or not self.internal_contact:
            raise ValueError("internal_contact must be a nonempty string")
        if not isinstance(self.trace_settings, Transport3DSettings):
            raise TypeError("trace_settings must be Transport3DSettings")
        if self.trace_settings.mode != "planar":
            raise ValueError("FingertipCase requires planar trace_settings")
        if not self.fea.converged or self.fea.indenter_pose is None:
            raise ValueError("a FingertipCase requires converged FEA with indenter_pose")
        if self.fea.reference_mesh is None:
            raise ValueError("FingertipCase requires FEAResult.reference_mesh")
        if self.fea.reference_mesh.parameters != self.fingertip_parameters:
            raise ValueError("FEA reference_mesh parameters do not match fingertip_parameters")
        if self.fea.reference_mesh.settings != self.mesh_settings:
            raise ValueError("FEA reference_mesh settings do not match mesh_settings")
        observed_steps = self.fea.details.get("requested_increments")
        if observed_steps is not None and int(observed_steps) != self.fem_steps:
            raise ValueError("FEA requested increments do not match fem_steps")
        observed_contact = self.fea.details.get("configuration", {}).get(
            "internal_contact_configuration"
        )
        if observed_contact is not None and observed_contact != self.internal_contact:
            raise ValueError("FEA internal contact does not match internal_contact")

        expected_morphology = _expected_morphology_fingerprint(
            self.fingertip_parameters
        )
        if self.optics.morphology_fingerprint != expected_morphology:
            raise ValueError("optics morphology fingerprint does not match the case")
        if self.raytrace.source_mode != "planar":
            raise ValueError("a FingertipCase requires raw PLANAR_2D ray tracing")
        if (
            self.raytrace.projected_x_edges_mm is None
            or self.raytrace.projected_y_edges_mm is None
            or self.raytrace.projected_weighted_path_density is None
        ):
            raise ValueError("raw PLANAR_2D ray tracing must retain its native P2 field")
        if self.optics.optical_mode != "PLANAR_2D":
            raise ValueError("a FingertipCase requires a PLANAR_2D optical result")
        if self.optics.mechanics_dimension != "2D":
            raise ValueError("PLANAR_2D case optics must be paired with 2D mechanics")
        if self.optics.mechanics_source != "explicit_contact_fea":
            raise ValueError(
                "FingertipCase requires the explicit-contact mechanics provenance"
            )
        expected_transport_configuration = transport_configuration(
            self.trace_settings,
            material=_optical_material_mapping(self.optical),
            source={"led": asdict(self.led)},
        )
        if self.optics.transport_configuration_fingerprint != fingerprint_mapping(
            expected_transport_configuration
        ):
            raise ValueError("optics transport configuration does not match the case")
        for name in (
            "launched_weight",
            "escaped_weight",
            "absorbed_weight",
            "terminated_weight",
            "energy_balance_error",
        ):
            if not math.isclose(
                float(getattr(self.raytrace, name)),
                float(getattr(self.optics, name)),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(f"raw raytrace and optics summary mismatch: {name}")
        if self.raytrace.launched_ray_count != self.optics.ray_count:
            raise ValueError("raw raytrace and optics summary mismatch: ray_count")
        if not np.array_equal(
            self.raytrace.projected_weighted_path_density.T,
            self.optics.field,
        ):
            raise ValueError("raw P2 field and optics summary field mismatch")
        expected_contact = contact_state_contract(
            self.contact_state,
            self.indenter_parameters,
        )
        observed_contact = dict(self.optics.contact_state)

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
        return self.optics.field

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
    if not isinstance(contact_state, ContactState):
        raise TypeError("contact_state must be a ContactState")
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
        source={"led": asdict(tip.led)},
    )
    contact_provenance = {
        **state,
        "contact_state_fingerprint": fingerprint_mapping(state),
    }
    raytrace = trace_geometry(
        tip,
        geometry,
        settings=trace_settings,
        runtime=optix_runtime,
    )
    optics = UnifiedTransportResult.from_transport_result(
        raytrace,
        morphology_id="custom",
        morphology_fingerprint=_expected_morphology_fingerprint(
            fingertip_parameters
        ),
        mechanics_source="explicit_contact_fea",
        mechanics_dimension="2D",
        contact_state=contact_provenance,
        transport_configuration_fingerprint=fingerprint_mapping(configuration),
    )
    case_provenance = {
        "mechanics_path": "fem.solve.solve -> fem.indentation.run_indentation_case",
        "mechanics_mode": "explicit_contact_2d",
        "localized_load_used": False,
        "mesh_level": mesh_settings.level,
        "fem_steps": fem_steps,
        "internal_contact": internal_contact,
        "optical_backend": "trace_geometry (shared OptiX backend)",
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
        optics=optics,
        led=tip.led,
        optical=tip.optical,
        mesh_settings=mesh_settings,
        fem_steps=fem_steps,
        internal_contact=internal_contact,
        trace_settings=trace_settings,
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
