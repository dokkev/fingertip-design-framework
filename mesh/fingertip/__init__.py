"""Fingertip contracts, geometry generation, and neutral surface views."""

from .contracts import (
    BoundaryEdge,
    FingertipMesh,
    InvalidMeshSettings,
    MeshedContactPair,
    MeshDomain,
    MeshNode,
    MeshSettings,
    MeshValidationReport,
    T3Element,
    mesh_settings_for_level,
)
from .geometry import FingertipMeshingError, GmshDependencyError, generate_fingertip_mesh
from .surface import InvalidPadMesh, PadMesh
from .validation import mesh_quality_statistics, validate_fingertip_mesh

__all__ = [
    "BoundaryEdge",
    "FingertipMesh",
    "FingertipMeshingError",
    "GmshDependencyError",
    "InvalidPadMesh",
    "InvalidMeshSettings",
    "MeshedContactPair",
    "MeshDomain",
    "MeshNode",
    "MeshSettings",
    "MeshValidationReport",
    "PadMesh",
    "T3Element",
    "generate_fingertip_mesh",
    "mesh_quality_statistics",
    "mesh_settings_for_level",
    "validate_fingertip_mesh",
]
