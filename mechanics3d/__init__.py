"""Optional GPU mechanics surrogate with a neutral NumPy boundary."""

from .fingertip import (
    FingertipMechanicsMesh,
    PrescribedVertexDisplacement,
    outer_compliant_timing_patch,
    prepare_fingertip_mechanics_mesh,
    solve_prescribed_indentation,
)
from .load import ParticleLoad
from .session import Mechanics3DSession
from .solve import Mechanics3DSettings, solve
from .types import Mechanics3DResult, TetMeshData

__all__ = [
    "Mechanics3DResult",
    "Mechanics3DSettings",
    "Mechanics3DSession",
    "FingertipMechanicsMesh",
    "PrescribedVertexDisplacement",
    "ParticleLoad",
    "TetMeshData",
    "outer_compliant_timing_patch",
    "prepare_fingertip_mechanics_mesh",
    "solve_prescribed_indentation",
    "solve",
]
