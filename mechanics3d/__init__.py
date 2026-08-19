"""Optional GPU mechanics surrogate with a neutral NumPy boundary."""

from .fingertip import (
    FingertipMechanicsMesh,
    InvalidFingertipMechanicsMesh,
    PrescribedVertexDisplacement,
    outer_compliant_timing_patch,
    make_fingertip_volume_state,
    prepare_fingertip_mechanics_mesh,
    solve_prescribed_indentation,
)
from .load import ParticleLoad
from .session import Mechanics3DSession
from .solve import Mechanics3DSettings, solve
from .types import Mechanics3DResult, TetMeshData
from .indentation import (
    IndentationResult,
    IndentationSettings,
    RigidIndenter3D,
    RigidPose3D,
    solve_fingertip_indentation,
)

__all__ = [
    "Mechanics3DResult",
    "Mechanics3DSettings",
    "Mechanics3DSession",
    "IndentationResult",
    "IndentationSettings",
    "FingertipMechanicsMesh",
    "InvalidFingertipMechanicsMesh",
    "PrescribedVertexDisplacement",
    "ParticleLoad",
    "RigidIndenter3D",
    "RigidPose3D",
    "TetMeshData",
    "outer_compliant_timing_patch",
    "make_fingertip_volume_state",
    "prepare_fingertip_mechanics_mesh",
    "solve_prescribed_indentation",
    "solve_fingertip_indentation",
    "solve",
]
