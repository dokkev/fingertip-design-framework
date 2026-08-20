"""Neutral mechanics contracts and the canonical fingertip indentation API.

Production evaluation enters through ``prepare_fingertip_mesh`` and
``solve_fingertip_indentation_trajectory``.  The generic ``NewtonSession``,
``ParticleLoad``, and ``solve`` boundaries remain available from their owning
modules for validation and smoke workflows, but are intentionally not part of
the production-facing package export.
"""

from .trajectory.fingertip import (
    PreparedFingertipMesh,
    InvalidFingertipMesh,
    PrescribedVertexDisplacement,
    outer_compliant_timing_patch,
    make_fingertip_volume_state,
    prepare_fingertip_mesh,
    solve_prescribed_indentation,
)
from .newton.solve import NewtonSettings, PhysicsDependencyError
from .contracts.types import NewtonResult, TetMeshData
from .trajectory.indentation import (
    IndentationCheckpoint,
    IndentationResult,
    IndentationSettings,
    IndentationTrajectoryResult,
    RigidIndenter3D,
    checkpoint_step_schedule,
    solve_fingertip_indentation,
    solve_fingertip_indentation_trajectory,
)

__all__ = [
    "NewtonResult",
    "NewtonSettings",
    "PhysicsDependencyError",
    "IndentationResult",
    "IndentationCheckpoint",
    "IndentationSettings",
    "IndentationTrajectoryResult",
    "PreparedFingertipMesh",
    "InvalidFingertipMesh",
    "PrescribedVertexDisplacement",
    "RigidIndenter3D",
    "checkpoint_step_schedule",
    "TetMeshData",
    "outer_compliant_timing_patch",
    "make_fingertip_volume_state",
    "prepare_fingertip_mesh",
    "solve_prescribed_indentation",
    "solve_fingertip_indentation",
    "solve_fingertip_indentation_trajectory",
]
