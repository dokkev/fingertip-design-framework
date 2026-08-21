"""Deterministic camera-independent 3D optical mechanics."""

from lumo.ray_tracing.optical_mechanics.result import Transport3DResult, Transport3DResultError
from lumo.ray_tracing.optical_mechanics.geometry import (
    CARRIER_CONTACT_INTERFACE,
    Full3DSurfaceProvenance,
    build_full3d_transport_geometry,
)
from lumo.ray_tracing.optical_mechanics.state_adapter import build_fingertip_volume_state_geometry
from lumo.ray_tracing.optical_mechanics.settings import Transport3DSettings
from lumo.ray_tracing.optical_mechanics.transport import (
    Transport3DDependencyError,
    Transport3DGeometryError,
    Transport3DPhysicsError,
    Transport3DTraceError,
    trace_geometry,
)

__all__ = [
    "Transport3DDependencyError",
    "Transport3DGeometryError",
    "Transport3DPhysicsError",
    "Transport3DResult",
    "Transport3DResultError",
    "Transport3DSettings",
    "Transport3DTraceError",
    "build_full3d_transport_geometry",
    "CARRIER_CONTACT_INTERFACE",
    "Full3DSurfaceProvenance",
    "build_fingertip_volume_state_geometry",
    "trace_geometry",
]
