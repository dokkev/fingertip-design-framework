"""Kratos backend for LIT fingertip finite-element analyses.

Kratos is loaded lazily by the concrete adapter and solve modules so importing
this package does not require the external Kratos environment.
"""

from fem.solve import FEAResult, solve
from mesh.indenter import IndenterSettings

__all__ = ["FEAResult", "IndenterSettings", "solve"]
