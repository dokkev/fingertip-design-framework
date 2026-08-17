"""Public orchestration for one explicit-contact 2D fingertip case."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from case.fea2d import FEA2D
from case.raytracing2d import RayTracing2D
from case.state import ContactState
from fem import FEAResult
from mesh import MeshSettings, mesh_settings_for_level
from mesh.indenter import IndenterPose2D, IndenterSettings
from model import Fingertip, FingertipParameters, LED, OpticalMaterial, fingertip_parameters_fingerprint
from optics.contact_object import IndenterOptics
from optics.transport3d import Transport3DResult, Transport3DSettings, UnifiedTransportResult, fingerprint_mapping


CASE_SCHEMA = "fingertip-case-v3"


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
    """Return only physical and numerical inputs, never free-form provenance."""
    return {
        "fingertip": {
            "parameters": asdict(case.fingertip.parameters),
            "led": asdict(case.fingertip.led),
            "optical": asdict(case.fingertip.optical),
        },
        "fea": {
            "mesh_settings": asdict(case.fea.mesh_settings),
            "indenter": asdict(case.fea.indenter),
            "contact": asdict(case.fea.contact),
            "steps": case.fea.steps,
            "internal_contact": case.fea.internal_contact,
        },
        "raytracing": {
            "settings": asdict(case.raytracing.settings),
            "indenter_optics": (
                None
                if case.raytracing.indenter_optics is None
                else asdict(case.raytracing.indenter_optics)
            ),
        },
    }


def case_id_for(case: "FingertipCase") -> str:
    """Return a deterministic ID from physical and numerical inputs."""
    if not isinstance(case, FingertipCase):
        raise TypeError("case must be a FingertipCase")
    return f"case-{fingerprint_mapping(_case_identity_payload(case))[:24]}"


def _require_results(case: "FingertipCase") -> tuple[FEAResult, Transport3DResult, UnifiedTransportResult]:
    fea_result = case.fea.result
    raw = case.raytracing.raw
    summary = case.raytracing.summary
    if fea_result is None or raw is None or summary is None:
        raise CaseConstructionError("the case has not completed FEA and PLANAR_2D tracing")
    return fea_result, raw, summary


def _validate_completed_case(case: "FingertipCase") -> None:
    fea_result, raw, summary = _require_results(case)
    if not fea_result.converged or fea_result.indenter_pose is None:
        raise CaseConstructionError("a completed case requires a converged FEA pose")
    if fea_result.reference_mesh is None:
        raise CaseConstructionError("a completed case requires FEAResult.reference_mesh")
    if fea_result.reference_mesh.parameters != case.fingertip.parameters:
        raise ValueError("FEA reference_mesh parameters do not match fingertip")
    if fea_result.reference_mesh.settings != case.fea.mesh_settings:
        raise ValueError("FEA reference_mesh settings do not match FEA2D")
    observed_steps = fea_result.details.get("requested_increments")
    if observed_steps is not None and int(observed_steps) != case.fea.steps:
        raise ValueError("FEA requested increments do not match FEA2D.steps")
    observed_internal_contact = fea_result.details.get("configuration", {}).get(
        "internal_contact_configuration"
    )
    if (
        observed_internal_contact is not None
        and observed_internal_contact != case.fea.internal_contact
    ):
        raise ValueError("FEA internal contact does not match FEA2D.internal_contact")

    expected_morphology = _expected_morphology_fingerprint(case.fingertip.parameters)
    if summary.morphology_fingerprint != expected_morphology:
        raise ValueError("optics morphology fingerprint does not match fingertip")
    if raw.source_mode != "planar":
        raise ValueError("a FingertipCase requires raw PLANAR_2D ray tracing")
    if (
        raw.projected_x_edges_mm is None
        or raw.projected_y_edges_mm is None
        or raw.projected_weighted_path_density is None
    ):
        raise ValueError("raw PLANAR_2D tracing must retain its native P2 field")
    if summary.optical_mode != "PLANAR_2D":
        raise ValueError("a FingertipCase requires a PLANAR_2D optical result")
    if summary.mechanics_dimension != "2D":
        raise ValueError("PLANAR_2D optics must be paired with 2D mechanics")
    if summary.mechanics_source != "explicit_contact_fea":
        raise ValueError("optics must identify explicit-contact FEA as its source")
    if summary.transport_configuration_fingerprint != fingerprint_mapping(
        case.raytracing.configuration(case.fingertip)
    ):
        raise ValueError("optics transport configuration does not match the case")
    for name in (
        "launched_weight",
        "escaped_weight",
        "absorbed_weight",
        "terminated_weight",
        "object_absorbed_weight",
        "object_transmitted_weight",
        "object_interface_incident_weight",
        "object_reflected_weight",
        "energy_balance_error",
    ):
        if not math.isclose(
            float(getattr(raw, name)),
            float(getattr(summary, name)),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"raw raytrace and summary mismatch: {name}")
    if raw.launched_ray_count != summary.ray_count:
        raise ValueError("raw raytrace and summary mismatch: ray_count")
    if not np.array_equal(raw.projected_weighted_path_density.T, summary.field):
        raise ValueError("raw P2 field and summary field mismatch")

    expected_contact = contact_state_contract(case.fea.contact, case.fea.indenter)
    observed_contact = dict(summary.contact_state)
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
        raise ValueError("raytrace contact-state fingerprint does not match case")

    pose = fea_result.indenter_pose
    assert pose is not None
    if pose.fixture.settings != case.fea.indenter:
        raise ValueError("indenter pose fixture does not match FEA2D")
    if not math.isclose(
        pose.fixture.frame.point_mm[0],
        case.fea.contact.location_x_mm,
        rel_tol=0.0,
        abs_tol=max(1.0e-8, 10.0 * case.fingertip.parameters.geometry_tolerance),
    ):
        raise ValueError("indenter pose location does not match contact")
    if not math.isclose(
        pose.prescribed_travel_mm,
        case.fea.contact.indentation_mm,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("indenter pose travel does not match contact")


@dataclass
class FingertipCase:
    """Configuration aggregate for one fingertip, FEA2D, and RayTracing2D run."""

    fingertip: Fingertip
    fea: FEA2D
    raytracing: RayTracing2D
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.fingertip, Fingertip):
            raise TypeError("fingertip must be Fingertip")
        if not isinstance(self.fea, FEA2D):
            raise TypeError("fea must be FEA2D")
        if not isinstance(self.raytracing, RayTracing2D):
            raise TypeError("raytracing must be RayTracing2D")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        self.provenance = _freeze_mapping(self.provenance)
        if (
            self.fea.result is not None
            and self.raytracing.raw is not None
            and self.raytracing.summary is not None
        ):
            _validate_completed_case(self)

    @property
    def case_id(self) -> str:
        return case_id_for(self)

    def solve(self) -> FEAResult:
        """Run only the configured 2D mechanics experiment."""
        result = self.fea.solve(self.fingertip)
        if not result.converged or result.indenter_pose is None:
            raise CaseConstructionError(
                "explicit-contact FEA did not produce a converged pose: "
                f"{result.details.get('failure_reason', 'unknown failure')}"
            )
        return result

    def trace(self, *, runtime: Any | None = None) -> UnifiedTransportResult:
        """Run PLANAR_2D optics from the exact FEA result."""
        if self.fea.result is None:
            raise CaseConstructionError("solve the case before tracing optics")
        contact = contact_state_contract(self.fea.contact, self.fea.indenter)
        contact["contact_state_fingerprint"] = fingerprint_mapping(contact)
        summary = self.raytracing.trace(
            self.fingertip,
            self.fea.result,
            contact_state=self.fea.contact,
            indenter=self.fea.indenter,
            morphology_id="custom",
            contact_provenance=contact,
            runtime=runtime,
        )
        _validate_completed_case(self)
        return summary

    def run(self, *, runtime: Any | None = None) -> "FingertipCase":
        """Run mechanics followed by optics and return this case."""
        self.solve()
        self.trace(runtime=runtime)
        return self

    @property
    def parameters(self) -> FingertipParameters:
        return self.fingertip.parameters

    @property
    def contact_state(self) -> ContactState:
        return self.fea.contact

    @property
    def indenter(self) -> IndenterSettings:
        return self.fea.indenter

    @property
    def displacement(self) -> np.ndarray | None:
        return None if self.fea.result is None else self.fea.result.displacement

    @property
    def deformed_mesh(self) -> Any:
        if self.fea.result is None:
            raise RuntimeError("the FEA result is unavailable")
        return self.fea.result.deformed_mesh

    @property
    def reaction_force(self) -> float | None:
        return None if self.fea.result is None else self.fea.result.reaction_force

    @property
    def contact(self) -> Mapping[str, Any]:
        return {} if self.fea.result is None else self.fea.result.contact

    @property
    def indenter_pose(self) -> IndenterPose2D:
        if self.fea.result is None or self.fea.result.indenter_pose is None:
            raise RuntimeError("the FEA indenter pose is unavailable")
        return self.fea.result.indenter_pose

    @property
    def optical_field(self) -> np.ndarray:
        if self.raytracing.summary is None:
            raise RuntimeError("the optical summary is unavailable")
        return self.raytracing.summary.field

    @property
    def escaped_weight(self) -> float:
        if self.raytracing.raw is None:
            raise RuntimeError("the raw optical result is unavailable")
        return self.raytracing.raw.escaped_weight


def run_case(
    *,
    fingertip_parameters: FingertipParameters,
    indenter_parameters: IndenterSettings,
    contact_state: ContactState,
    mesh_settings: MeshSettings | None = None,
    trace_settings: Transport3DSettings | None = None,
    led: LED | None = None,
    optical: OpticalMaterial | None = None,
    indenter_optics: IndenterOptics | None = None,
    fem_steps: int = 48,
    internal_contact: str = "three_pairs",
    provenance: Mapping[str, Any] | None = None,
    optix_runtime: Any | None = None,
) -> FingertipCase:
    """Convenience API for constructing and running the three-object case."""
    if not isinstance(fingertip_parameters, FingertipParameters):
        raise TypeError("fingertip_parameters must be FingertipParameters")
    if mesh_settings is None:
        mesh_settings = mesh_settings_for_level("medium")
    if trace_settings is None:
        trace_settings = Transport3DSettings(mode="planar")
    if not trace_settings.retain_projected_segments:
        trace_settings = Transport3DSettings(
            **{**asdict(trace_settings), "retain_projected_segments": True}
        )
    tip = Fingertip(
        fingertip_parameters,
        led=LED() if led is None else led,
        optical=OpticalMaterial() if optical is None else optical,
    )
    case = FingertipCase(
        fingertip=tip,
        fea=FEA2D(
            indenter=indenter_parameters,
            contact=contact_state,
            mesh_settings=mesh_settings,
            steps=fem_steps,
            internal_contact=internal_contact,
        ),
        raytracing=RayTracing2D(
            settings=trace_settings,
            indenter_optics=indenter_optics,
        ),
        provenance={
            "mechanics_mode": "explicit_contact_2d",
            "localized_load_used": False,
            "optical_mode": "PLANAR_2D",
            "indenter_optically_active": indenter_optics is not None,
            **dict(provenance or {}),
        },
    )
    return case.run(runtime=optix_runtime)


__all__ = [
    "CASE_SCHEMA",
    "CaseConstructionError",
    "ContactState",
    "FEA2D",
    "FingertipCase",
    "RayTracing2D",
    "case_id_for",
    "contact_state_contract",
    "run_case",
]
