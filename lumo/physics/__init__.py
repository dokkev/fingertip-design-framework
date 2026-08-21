"""Neutral mechanics contracts and the canonical fingertip indentation API.

Production evaluation enters through ``prepare_fingertip_mesh`` and
``solve_fingertip_indentation_trajectory``.  The generic ``NewtonSession``,
``ParticleLoad``, and ``solve`` boundaries remain available from their owning
modules for validation and smoke workflows, but are intentionally not part of
the production-facing package export.
"""

from .trajectory.fingertip_adapter import (
    PreparedFingertipMesh,
    InvalidFingertipMesh,
    make_fingertip_volume_state,
    prepare_fingertip_mesh,
)
from .newton.solve import NewtonSettings, PhysicsDependencyError
from .contracts.types import NewtonResult, TetMeshData
from .trajectory.indentation import (
    CandidateMechanicsError,
    CheckpointStep,
    IndentationCheckpoint,
    IndentationResult,
    IndentationSettings,
    IndentationTrajectoryResult,
    MechanicsCheckpointState,
    RigidIndenter3D,
    checkpoint_step_schedule,
    solve_fingertip_indentation,
    solve_fingertip_indentation_trajectory,
)

__all__ = [
    "CandidateMechanicsError",
    "CheckpointStep",
    "NewtonResult",
    "NewtonSettings",
    "PhysicsDependencyError",
    "IndentationResult",
    "MechanicsCheckpointState",
    "IndentationCheckpoint",
    "IndentationSettings",
    "IndentationTrajectoryResult",
    "PreparedFingertipMesh",
    "InvalidFingertipMesh",
    "RigidIndenter3D",
    "checkpoint_step_schedule",
    "TetMeshData",
    "make_fingertip_volume_state",
    "prepare_fingertip_mesh",
    "solve_fingertip_indentation",
    "solve_fingertip_indentation_trajectory",
]
