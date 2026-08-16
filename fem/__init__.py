"""Kratos backend for LIT fingertip finite-element analyses.

Kratos is loaded lazily by the concrete adapter and solve modules so importing
this package does not require the external Kratos environment.
"""

from fem.solve import FEAResult, solve
from fem.solid3d import SolidFEAError, SolidFEAResult, SolidFEASettings, solve_solid_3d
from mesh.indenter import IndenterSettings

__all__ = [
    "FEAResult",
    "IndenterSettings",
    "SolidFEAError",
    "SolidFEAResult",
    "SolidFEASettings",
    "solve",
    "solve_solid_3d",
]
