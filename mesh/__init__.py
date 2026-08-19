"""Public solver-independent mesh data contracts."""

from mesh.pad import InvalidPadMesh, PadMesh
from mesh.indenter import IndenterPose2D
from mesh.types import (
    FingertipMesh,
    InvalidMeshSettings,
    MeshSettings,
    mesh_settings_for_level,
)
from mesh.volume3d import (
    VolumeMeshDependencyError,
    VolumeMeshingError,
    generate_volume_mesh,
)
from mesh.volume_types import (
    FingertipVolumeMesh,
    SurfaceTriangle,
    Tetrahedron,
    VolumeMeshQuality,
    VolumeMeshSettings,
    VolumeMeshValidation,
    VolumeNode,
    volume_mesh_settings_for_tier,
)
from mesh.volume_state import FingertipVolumeState, make_fingertip_volume_state
from mesh.rigid_object import (
    RigidObjectMesh,
    make_box_mesh,
    make_cube_mesh,
    make_cylinder_mesh,
    make_sphere_mesh,
)

__all__ = [
    "FingertipMesh",
    "InvalidMeshSettings",
    "InvalidPadMesh",
    "IndenterPose2D",
    "MeshSettings",
    "PadMesh",
    "mesh_settings_for_level",
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
    "volume_mesh_settings_for_tier",
    "FingertipVolumeState",
    "make_fingertip_volume_state",
    "RigidObjectMesh",
    "make_box_mesh",
    "make_cube_mesh",
    "make_cylinder_mesh",
    "make_sphere_mesh",
]
