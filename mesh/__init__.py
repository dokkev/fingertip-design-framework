"""Public solver-independent mesh data contracts."""

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
from mesh.volume.state import (
    FingertipVolumeState,
    InvalidDeformedFingertipState,
    make_fingertip_volume_state,
)
from mesh.rigid.object import (
    RigidPose3D,
    RigidObjectMesh,
    make_box_mesh,
    make_cube_mesh,
    make_cylinder_mesh,
    make_sphere_mesh,
)
from mesh.rigid.carrier import RigidCarrierMesh, make_distal_phalanx_mesh
from mesh.io.obj import (
    RigidMeshAssetError,
    load_obj,
    load_obj_directory,
    save_obj,
)

__all__ = [
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
    "InvalidDeformedFingertipState",
    "make_fingertip_volume_state",
    "RigidObjectMesh",
    "RigidCarrierMesh",
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
