"""Deterministic camera-independent OptiX 3D transport."""

from optics.transport3d.result import Transport3DResult, Transport3DResultError
from optics.transport3d.geometry import (
    CARRIER_CONTACT_INTERFACE,
    Full3DSurfaceProvenance,
    build_full3d_transport_geometry,
    build_fixed_transport_surfaces,
)
from optics.transport3d.fingertip import build_fingertip_volume_state_geometry
from optics.transport3d.settings import Transport3DSettings
from optics.transport3d.transport import (
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
    "build_fixed_transport_surfaces",
    "build_fingertip_volume_state_geometry",
    "trace_geometry",
]
