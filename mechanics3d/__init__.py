"""Optional GPU mechanics surrogate with a neutral NumPy boundary."""

from .solve import Mechanics3DSettings, solve
from .types import Mechanics3DResult, TetMeshData

__all__ = [
    "Mechanics3DResult",
    "Mechanics3DSettings",
    "TetMeshData",
    "solve",
]
