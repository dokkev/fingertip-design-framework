"""Discretized fingertip geometry for Newton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lumo.fingertip import (
    ACTIVE_Y_BOUNDS_MM,
    DISTAL_END_CAP_LENGTH_MM,
    LED_CENTERS_Y_MM,
    LED_RECESS_DEPTH_MM,
    LED_RECESS_WIDTH_MM,
    Fingertip,
)
from lumo.util.scalar_validation import require_positive

from .carrier_mesh import (
    _make_carrier_collision_mesh,
    _make_carrier_mesh,
)
from .silicone_mesh import _make_silicone_mesh

if TYPE_CHECKING:
    import newton


@dataclass(frozen=True)
class FingertipMesh:
    """Newton meshes produced from one analytic fingertip assembly."""

    fingertip: Fingertip
    silicone: "newton.TetMesh"
    carrier: "newton.Mesh"
    carrier_collision: "newton.Mesh"
    bonded_vertex_indices: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(
            self.bonded_vertex_indices,
            dtype=np.int32,
        )
        if indices.ndim != 1:
            raise ValueError("bonded_vertex_indices must be one-dimensional")
        if indices.size == 0:
            raise ValueError("bonded_vertex_indices must not be empty")
        if np.any(indices < 0):
            raise ValueError("bonded_vertex_indices must be non-negative")

        indices = np.unique(indices)
        if indices[-1] >= self.silicone.vertex_count:
            raise ValueError(
                "bonded vertex index exceeds silicone vertex count"
            )
        indices.setflags(write=False)
        object.__setattr__(self, "bonded_vertex_indices", indices)


def make_fingertip_mesh(
    fingertip: Fingertip,
    *,
    element_size_mm: float = 1.0,
) -> FingertipMesh:
    """Discretize the complete current LUMO fingertip hardware."""
    if not isinstance(fingertip, Fingertip):
        raise TypeError("fingertip must be a Fingertip")
    require_positive("element_size_mm", element_size_mm)

    silicone, bonded_vertex_indices = _make_silicone_mesh(
        fingertip.silicone,
        fingertip.bonding_interface,
        active_y_bounds_mm=ACTIVE_Y_BOUNDS_MM,
        distal_end_cap_length_mm=DISTAL_END_CAP_LENGTH_MM,
        element_size_mm=element_size_mm,
    )
    active_length_mm = ACTIVE_Y_BOUNDS_MM[1] - ACTIVE_Y_BOUNDS_MM[0]
    carrier = _make_carrier_mesh(
        fingertip.carrier,
        active_length_mm=active_length_mm,
        distal_end_cap_length_mm=DISTAL_END_CAP_LENGTH_MM,
        led_centers_y_mm=LED_CENTERS_Y_MM,
        led_recess_width_mm=LED_RECESS_WIDTH_MM,
        led_recess_depth_mm=LED_RECESS_DEPTH_MM,
    )
    carrier_collision = _make_carrier_collision_mesh(
        fingertip.carrier,
        fingertip.silicone,
        active_length_mm=active_length_mm,
        led_centers_y_mm=LED_CENTERS_Y_MM,
        led_recess_width_mm=LED_RECESS_WIDTH_MM,
        led_recess_depth_mm=LED_RECESS_DEPTH_MM,
    )

    return FingertipMesh(
        fingertip=fingertip,
        silicone=silicone,
        carrier=carrier,
        carrier_collision=carrier_collision,
        bonded_vertex_indices=bonded_vertex_indices,
    )


__all__ = [
    "FingertipMesh",
    "make_fingertip_mesh",
]
