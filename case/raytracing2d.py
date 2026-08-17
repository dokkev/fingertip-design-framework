"""Configuration and result ownership for PLANAR_2D OptiX transport."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from fem import FEAResult
from model import Fingertip, fingertip_parameters_fingerprint
from optics.contact_object import IndenterOptics
from optics.transport3d import (
    Transport3DResult,
    Transport3DSettings,
    UnifiedTransportResult,
    fingerprint_mapping,
    trace_geometry,
    transport_configuration,
)
from optics.transport3d.geometry import build_transport_geometry

from case.state import ContactState
from mesh.indenter import IndenterSettings


def _optical_material_mapping(fingertip: Fingertip) -> dict[str, float]:
    material = fingertip.optical
    return {
        "refractive_index_air": material.refractive_index_air,
        "refractive_index_silicone": material.refractive_index_silicone,
        "absorption_per_mm": material.absorption_per_mm,
        "scattering_per_mm": material.scattering_per_mm,
        "anisotropy_g": material.anisotropy_g,
    }


def _source_mapping(
    fingertip: Fingertip,
    indenter_optics: IndenterOptics | None,
) -> dict[str, Any]:
    return {
        "led": asdict(fingertip.led),
        "indenter_optics": (
            None if indenter_optics is None else asdict(indenter_optics)
        ),
    }


@dataclass
class RayTracing2D:
    """One PLANAR_2D optical experiment and its optional results."""

    settings: Transport3DSettings = field(
        default_factory=lambda: Transport3DSettings(mode="planar")
    )
    indenter_optics: IndenterOptics | None = None
    raw: Transport3DResult | None = field(default=None, init=False)
    summary: UnifiedTransportResult | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.settings, Transport3DSettings):
            raise TypeError("settings must be Transport3DSettings")
        if self.settings.mode != "planar":
            raise ValueError("RayTracing2D requires settings.mode='planar'")
        if self.indenter_optics is not None and not isinstance(
            self.indenter_optics, IndenterOptics
        ):
            raise TypeError("indenter_optics must be IndenterOptics or None")

    def configuration(self, fingertip: Fingertip) -> Mapping[str, Any]:
        """Return the complete result-changing transport configuration."""
        if not isinstance(fingertip, Fingertip):
            raise TypeError("fingertip must be Fingertip")
        return transport_configuration(
            self.settings,
            material=_optical_material_mapping(fingertip),
            source=_source_mapping(fingertip, self.indenter_optics),
        )

    def trace(
        self,
        fingertip: Fingertip,
        fea_result: FEAResult,
        *,
        contact_state: ContactState,
        indenter: IndenterSettings,
        morphology_id: str = "custom",
        contact_provenance: Mapping[str, Any] | None = None,
        runtime: Any | None = None,
    ) -> UnifiedTransportResult:
        """Trace the exact solved FEA pose without reconstructing mechanics."""
        if not isinstance(fingertip, Fingertip):
            raise TypeError("fingertip must be Fingertip")
        if not isinstance(fea_result, FEAResult):
            raise TypeError("fea_result must be FEAResult")
        if not fea_result.converged or fea_result.indenter_pose is None:
            raise ValueError("RayTracing2D requires a converged FEAResult pose")
        if fea_result.reference_mesh is None:
            raise ValueError("RayTracing2D requires FEAResult.reference_mesh")
        if not isinstance(contact_state, ContactState):
            raise TypeError("contact_state must be ContactState")
        if not isinstance(indenter, IndenterSettings):
            raise TypeError("indenter must be IndenterSettings")
        if contact_state.indenter_radius_mm != indenter.radius_mm:
            raise ValueError("contact indenter radius must match indenter settings")
        settings = self.settings
        if not settings.retain_projected_segments:
            settings = replace(settings, retain_projected_segments=True)
            self.settings = settings
        geometry = build_transport_geometry(
            fingertip,
            fea_result.deformed_mesh,
            fea_result.reference_mesh,
            depth_mm=settings.extrusion_depth_mm,
            source_epsilon_mm=settings.source_epsilon_mm,
            indenter_pose=(
                fea_result.indenter_pose
                if self.indenter_optics is not None
                else None
            ),
            indenter_optics=self.indenter_optics,
        )
        state = dict(contact_provenance or {})
        raytrace = trace_geometry(
            fingertip,
            geometry,
            settings=settings,
            runtime=runtime,
        )
        summary = UnifiedTransportResult.from_transport_result(
            raytrace,
            morphology_id=morphology_id,
            morphology_fingerprint=fingertip_parameters_fingerprint(
                fingertip.parameters
            ),
            mechanics_source="explicit_contact_fea",
            mechanics_dimension="2D",
            contact_state=state,
            transport_configuration_fingerprint=fingerprint_mapping(
                self.configuration(fingertip)
            ),
        )
        self.raw = raytrace
        self.summary = summary
        return summary


__all__ = ["RayTracing2D"]
