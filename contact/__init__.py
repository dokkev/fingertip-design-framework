"""Solver-neutral geometric contact initialization contracts."""

from .first_contact import (
    FirstContactResult,
    FirstContactSettings,
    FingertipContactSurface,
    find_first_contact,
    make_outer_compliant_surface,
    intersects,
)
from .sphere_alignment import SphereAlignment, canonical_sphere_alignment

__all__ = [
    "FirstContactResult",
    "FirstContactSettings",
    "FingertipContactSurface",
    "SphereAlignment",
    "canonical_sphere_alignment",
    "find_first_contact",
    "intersects",
    "make_outer_compliant_surface",
]
