"""Public solver-independent mesh data contracts."""

from mesh.fingertip.surface import InvalidPadMesh, PadMesh
from mesh.rigid import IndenterPose2D
from mesh.fingertip.contracts import (
    FingertipMesh,
    InvalidMeshSettings,
    MeshSettings,
    mesh_settings_for_level,
)
from mesh.fingertip.geometry import (
    FingertipMeshingError,
    GmshDependencyError,
)
from mesh.volume.mesh import (
    VolumeMeshDependencyError,
    VolumeMeshingError,
    generate_volume_mesh,
)
from mesh.volume.contracts import (
    FingertipVolumeMesh,
    SurfaceTriangle,
    Tetrahedron,
    VolumeMeshQuality,
    VolumeMeshSettings,
    VolumeMeshValidation,
    VolumeNode,
    volume_mesh_settings_for_tier,
)
from mesh.volume.state import FingertipVolumeState, make_fingertip_volume_state
from mesh.rigid.object import (
    RigidPose3D,
    RigidObjectMesh,
    make_box_mesh,
    make_cube_mesh,
    make_cylinder_mesh,
    make_sphere_mesh,
)
from mesh.rigid.carrier import make_distal_phalanx_mesh
from mesh.io.obj import (
    RigidMeshAssetError,
    load_obj,
    load_obj_directory,
    save_obj,
)

__all__ = [
    "FingertipMesh",
    "FingertipMeshingError",
    "GmshDependencyError",
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
    "RigidPose3D",
    "make_box_mesh",
    "make_cube_mesh",
    "make_cylinder_mesh",
    "make_distal_phalanx_mesh",
    "make_sphere_mesh",
    "RigidMeshAssetError",
    "load_obj",
    "load_obj_directory",
    "save_obj",
]
