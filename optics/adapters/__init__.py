"""Adapters that convert mechanical artifacts to neutral optical state."""

from optics.adapters.fem_field import (
    OpticalFieldAdapterError,
    build_pad_field_from_arrays,
    build_pad_field_from_mesh_and_displacements,
    load_pad_field_npz,
)
from optics.adapters.shapely_preview import build_preview_pad_mesh_template
from optics.adapters.semantic_boundaries import (
    OpticalBoundaryClassificationError,
    tag_reference_pad_boundaries,
)

__all__ = [
    "OpticalFieldAdapterError",
    "OpticalBoundaryClassificationError",
    "build_pad_field_from_arrays",
    "build_pad_field_from_mesh_and_displacements",
    "build_preview_pad_mesh_template",
    "load_pad_field_npz",
    "tag_reference_pad_boundaries",
]
