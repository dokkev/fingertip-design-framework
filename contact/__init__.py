"""Solver-neutral geometric contact initialization contracts."""

from .first_contact import (
    FirstContactResult,
    FirstContactSettings,
    FingertipContactSurface,
    find_first_contact,
    make_outer_compliant_surface,
    intersects,
)
from .sphere_alignment import (
    SphereAlignment,
    canonical_sphere_alignment,
    sphere_alignment_at_normalized_location,
)

__all__ = [
    "FirstContactResult",
    "FirstContactSettings",
    "FingertipContactSurface",
    "SphereAlignment",
    "canonical_sphere_alignment",
    "sphere_alignment_at_normalized_location",
    "find_first_contact",
    "intersects",
    "make_outer_compliant_surface",
]
