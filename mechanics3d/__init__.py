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
    IndentationCheckpoint,
    IndentationResult,
    IndentationSettings,
    IndentationTrajectoryResult,
    RigidIndenter3D,
    RigidPose3D,
    checkpoint_step_schedule,
    solve_fingertip_indentation,
    solve_fingertip_indentation_trajectory,
)

__all__ = [
    "Mechanics3DResult",
    "Mechanics3DSettings",
    "Mechanics3DSession",
    "IndentationResult",
    "IndentationCheckpoint",
    "IndentationSettings",
    "IndentationTrajectoryResult",
    "FingertipMechanicsMesh",
    "InvalidFingertipMechanicsMesh",
    "PrescribedVertexDisplacement",
    "ParticleLoad",
    "RigidIndenter3D",
    "RigidPose3D",
    "checkpoint_step_schedule",
    "TetMeshData",
    "outer_compliant_timing_patch",
    "make_fingertip_volume_state",
    "prepare_fingertip_mechanics_mesh",
    "solve_prescribed_indentation",
    "solve_fingertip_indentation",
    "solve_fingertip_indentation_trajectory",
    "solve",
]
