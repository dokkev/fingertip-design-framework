"""Optional GPU mechanics surrogate with a neutral NumPy boundary."""

from .fingertip import (
    PreparedFingertipMesh,
    InvalidFingertipMesh,
    PrescribedVertexDisplacement,
    outer_compliant_timing_patch,
    make_fingertip_volume_state,
    prepare_fingertip_mesh,
    solve_prescribed_indentation,
)
from .load import ParticleLoad
from .session import NewtonSession
from .solve import NewtonSettings, solve
from .types import NewtonResult, TetMeshData
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
    "NewtonResult",
    "NewtonSettings",
    "NewtonSession",
    "IndentationResult",
    "IndentationCheckpoint",
    "IndentationSettings",
    "IndentationTrajectoryResult",
    "PreparedFingertipMesh",
    "InvalidFingertipMesh",
    "PrescribedVertexDisplacement",
    "ParticleLoad",
    "RigidIndenter3D",
    "RigidPose3D",
    "checkpoint_step_schedule",
    "TetMeshData",
    "outer_compliant_timing_patch",
    "make_fingertip_volume_state",
    "prepare_fingertip_mesh",
    "solve_prescribed_indentation",
    "solve_fingertip_indentation",
    "solve_fingertip_indentation_trajectory",
    "solve",
]
