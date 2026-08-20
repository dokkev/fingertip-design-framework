"""Neutral volume-mesh generation and deformed-state records."""

from .contracts import (
    FingertipVolumeMesh,
    SurfaceTriangle,
    Tetrahedron,
    VolumeMeshQuality,
    VolumeMeshSettings,
    VolumeMeshValidation,
    VolumeNode,
    volume_mesh_settings_for_tier,
)
from .mesh import VolumeMeshDependencyError, VolumeMeshingError, generate_volume_mesh
from .state import FingertipVolumeState, make_fingertip_volume_state

__all__ = [
    "FingertipVolumeState",
    "FingertipVolumeMesh",
    "SurfaceTriangle",
    "Tetrahedron",
    "VolumeMeshDependencyError",
    "VolumeMeshQuality",
    "VolumeMeshSettings",
    "VolumeMeshValidation",
    "VolumeMeshingError",
    "VolumeNode",
    "generate_volume_mesh",
    "make_fingertip_volume_state",
    "volume_mesh_settings_for_tier",
]
