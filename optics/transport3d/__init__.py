"""Deterministic camera-independent OptiX 3D transport."""

from optics.transport3d.result import Transport3DResult, Transport3DResultError
from optics.transport3d.artifact import (
    FULL3D_SURFACE_SCHEMA,
    NATIVE_3D_FEA_STATE_SCHEMA,
    Full3DSurfaceArtifact,
    load_full3d_surface_artifact,
)
from optics.transport3d.geometry import (
    build_full3d_transport_geometry,
    build_fixed_transport_surfaces,
    build_transport_geometry,
)
from optics.transport3d.fingertip import build_fingertip_volume_state_geometry
from optics.transport3d.settings import Transport3DSettings
from optics.transport3d.transport import (
    Transport3DDependencyError,
    Transport3DGeometryError,
    Transport3DPhysicsError,
    Transport3DTraceError,
    trace_geometry,
    trace_3d,
)
from optics.transport3d.unified import (
    OptiXTransport,
    LEGACY_UNIFIED_ARTIFACT_SCHEMA,
    LEGACY_UNIFIED_ARTIFACT_SCHEMA_V2,
    UnifiedTransportResult,
    fingerprint_mapping,
    load_case_artifact,
    native_field_separability,
    save_case_artifact,
    transport_configuration,
)

__all__ = [
    "Transport3DDependencyError",
    "Transport3DGeometryError",
    "Transport3DPhysicsError",
    "Transport3DResult",
    "Transport3DResultError",
    "Transport3DSettings",
    "Transport3DTraceError",
    "FULL3D_SURFACE_SCHEMA",
    "NATIVE_3D_FEA_STATE_SCHEMA",
    "Full3DSurfaceArtifact",
    "OptiXTransport",
    "LEGACY_UNIFIED_ARTIFACT_SCHEMA",
    "LEGACY_UNIFIED_ARTIFACT_SCHEMA_V2",
    "UnifiedTransportResult",
    "build_full3d_transport_geometry",
    "build_fixed_transport_surfaces",
    "build_fingertip_volume_state_geometry",
    "build_transport_geometry",
    "fingerprint_mapping",
    "load_case_artifact",
    "load_full3d_surface_artifact",
    "native_field_separability",
    "save_case_artifact",
    "trace_geometry",
    "trace_3d",
    "transport_configuration",
]
