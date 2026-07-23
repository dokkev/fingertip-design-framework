"""Solver-agnostic parametric fingertip geometry package."""

from model.fingertip_model import (
    BoundarySegment,
    ContactPair,
    FingertipBoundaries,
    FingertipModel,
    InterfaceDefinition,
    InvalidFingertipGeometry,
)
from model.fingertip_parameters import FingertipParameters, InvalidFingertipParameters

__all__ = [
    "BoundarySegment",
    "ContactPair",
    "FingertipBoundaries",
    "FingertipModel",
    "FingertipParameters",
    "InterfaceDefinition",
    "InvalidFingertipGeometry",
    "InvalidFingertipParameters",
]
