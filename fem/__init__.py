"""Solver-independent FEM mesh interfaces for the LIT fingertip."""

from fem.fingertip_mesher import (
    GmshDependencyError,
    generate_fingertip_mesh,
    validate_fingertip_mesh,
)
from fem.mesh_types import (
    FingertipMesh,
    InvalidMeshSettings,
    MeshSettings,
    mesh_settings_for_level,
)

__all__ = [
    "FingertipMesh",
    "GmshDependencyError",
    "InvalidMeshSettings",
    "MeshSettings",
    "generate_fingertip_mesh",
    "mesh_settings_for_level",
    "validate_fingertip_mesh",
]
