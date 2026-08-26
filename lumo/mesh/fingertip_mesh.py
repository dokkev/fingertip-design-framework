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


_MM_TO_M = 1.0e-3


@dataclass(frozen=True)
class FingertipMesh:
    """Newton meshes produced from one analytic fingertip assembly."""

    fingertip: Fingertip
    silicone: "newton.TetMesh"
    carrier: "newton.Mesh"
    carrier_collision: "newton.Mesh"
    bonded_vertex_indices: np.ndarray
    led_centers_m: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(
            self.bonded_vertex_indices,
            dtype=np.int32,
        )
        if indices.ndim != 1:
            raise ValueError("bonded_vertex_indices must be one-dimensional")
        if np.any(indices < 0):
            raise ValueError("bonded_vertex_indices must be non-negative")

        indices = np.unique(indices)
        indices.setflags(write=False)
        object.__setattr__(self, "bonded_vertex_indices", indices)
        centers = np.asarray(self.led_centers_m, dtype=np.float64)
        expected_shape = (len(LED_CENTERS_Y_MM), 3)
        if centers.shape != expected_shape:
            raise ValueError(
                f"led_centers_m must have shape {expected_shape}"
            )
        if not np.all(np.isfinite(centers)):
            raise ValueError("led_centers_m must be finite")
        if not np.allclose(
            centers[:, 1],
            _MM_TO_M * np.asarray(LED_CENTERS_Y_MM),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("LED centers do not match the hardware layout")

        centers = np.ascontiguousarray(centers)
        centers.setflags(write=False)
        object.__setattr__(self, "led_centers_m", centers)


def make_fingertip_mesh(
    fingertip: Fingertip,
    *,
    element_size_mm: float = 1.0,
) -> FingertipMesh:
    """Discretize the complete current LUMO fingertip hardware."""
    if not isinstance(fingertip, Fingertip):
        raise TypeError("fingertip must be a Fingertip")
    require_positive("element_size_mm", element_size_mm)
    if fingertip.parameters.geometry.void_height_mm != 0.0:
        raise ValueError(
            "the production geometry requires "
            "void_height_mm=0; LED clearance comes from the stem recesses"
        )

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

    led_top_z_m = _MM_TO_M * (
        min(z_mm for _, z_mm in fingertip.carrier.cross_section)
        + LED_RECESS_DEPTH_MM
    )
    centers_m = np.asarray(
        [
            (0.0, _MM_TO_M * y_mm, led_top_z_m)
            for y_mm in LED_CENTERS_Y_MM
        ],
        dtype=np.float64,
    )
    return FingertipMesh(
        fingertip=fingertip,
        silicone=silicone,
        carrier=carrier,
        carrier_collision=carrier_collision,
        bonded_vertex_indices=bonded_vertex_indices,
        led_centers_m=centers_m,
    )


__all__ = [
    "FingertipMesh",
    "make_fingertip_mesh",
]
