"""Solver-neutral geometric contact initialization contracts."""

from .first_contact import (
    CandidateContactError,
    DEFAULT_FIRST_CONTACT_SETTINGS,
    FirstContactResult,
    FirstContactSettings,
    FingertipContactSurface,
    find_first_contact,
    make_outer_compliant_surface,
    intersects,
    unintended_boundary_clearance_mm,
)
from .sphere_alignment import (
    SphereAlignment,
    canonical_sphere_alignment,
    sphere_alignment_at_normalized_location,
)
from .surface_frame import (
    CrownFrame,
    InvalidSurfaceFrame,
    crown_frame_from_model,
    surface_frame_from_normalized_location,
)

__all__ = [
    "CandidateContactError",
    "DEFAULT_FIRST_CONTACT_SETTINGS",
    "FirstContactResult",
    "FirstContactSettings",
    "FingertipContactSurface",
    "SphereAlignment",
    "CrownFrame",
    "InvalidSurfaceFrame",
    "canonical_sphere_alignment",
    "sphere_alignment_at_normalized_location",
    "find_first_contact",
    "intersects",
    "make_outer_compliant_surface",
    "unintended_boundary_clearance_mm",
    "crown_frame_from_model",
    "surface_frame_from_normalized_location",
]
