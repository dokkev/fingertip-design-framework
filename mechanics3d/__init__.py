"""Optional GPU mechanics surrogate with a neutral NumPy boundary."""

from .fingertip import FingertipMechanicsMesh, prepare_fingertip_mechanics_mesh
from .solve import Mechanics3DSettings, solve
from .types import Mechanics3DResult, TetMeshData

__all__ = [
    "Mechanics3DResult",
    "Mechanics3DSettings",
    "FingertipMechanicsMesh",
    "TetMeshData",
    "prepare_fingertip_mechanics_mesh",
    "solve",
]
