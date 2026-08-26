"""Discretized fingertip geometry for Newton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lumo.fingertip.fingertip import Fingertip
from lumo.util.scalar_validation import require_positive

from .carrier_mesh import (
    _make_carrier_5led_collision_mesh,
    _make_carrier_5led_mesh,
    _make_carrier_collision_mesh,
    _make_carrier_mesh,
)
from .silicone_mesh import _make_silicone_5led_mesh, _make_silicone_mesh

if TYPE_CHECKING:
    import newton


NUM_LEDS = 5
LED_PITCH_MM = 11.0
LED_RECESS_WIDTH_MM = 5.1
LED_RECESS_DEPTH_MM = 0.19
MAIN_LENGTH_MM = NUM_LEDS * LED_PITCH_MM
DISTAL_END_CAP_LENGTH_MM = 5.0
TOTAL_LENGTH_MM = MAIN_LENGTH_MM + DISTAL_END_CAP_LENGTH_MM
MAIN_Y_BOUNDS_MM = (-0.5 * MAIN_LENGTH_MM, 0.5 * MAIN_LENGTH_MM)
TOTAL_Y_BOUNDS_MM = (
    MAIN_Y_BOUNDS_MM[0],
    MAIN_Y_BOUNDS_MM[1] + DISTAL_END_CAP_LENGTH_MM,
)
_MM_TO_M = 1.0e-3


def led_centers_y_mm() -> tuple[float, ...]:
    """Return the five longitudinal LED centers from proximal to distal."""
    center_index = 0.5 * (NUM_LEDS - 1)
    return tuple(
        LED_PITCH_MM * (index - center_index)
        for index in range(NUM_LEDS)
    )


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
        if np.any(indices < 0):
            raise ValueError("bonded_vertex_indices must be non-negative")

        indices = np.unique(indices)
        indices.setflags(write=False)
        object.__setattr__(self, "bonded_vertex_indices", indices)


@dataclass(frozen=True)
class Fingertip5LEDMesh(FingertipMesh):
    """Full 60 mm mesh and longitudinal metadata for five LED references."""

    led_centers_m: np.ndarray
    main_y_bounds_m: tuple[float, float]
    total_y_bounds_m: tuple[float, float]

    def __post_init__(self) -> None:
        super().__post_init__()
        centers = np.asarray(self.led_centers_m, dtype=np.float64)
        if centers.shape != (NUM_LEDS, 3):
            raise ValueError(f"led_centers_m must have shape ({NUM_LEDS}, 3)")
        if not np.all(np.isfinite(centers)):
            raise ValueError("led_centers_m must be finite")
        if not np.allclose(
            np.diff(centers[:, 1]),
            LED_PITCH_MM * _MM_TO_M,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("adjacent LED centers must use the 11 mm pitch")

        main_bounds = tuple(float(value) for value in self.main_y_bounds_m)
        total_bounds = tuple(float(value) for value in self.total_y_bounds_m)
        if main_bounds != tuple(value * _MM_TO_M for value in MAIN_Y_BOUNDS_MM):
            raise ValueError("main_y_bounds_m do not match the 55 mm section")
        if total_bounds != tuple(value * _MM_TO_M for value in TOTAL_Y_BOUNDS_MM):
            raise ValueError("total_y_bounds_m do not match the 60 mm fingertip")

        centers = np.ascontiguousarray(centers)
        centers.setflags(write=False)
        object.__setattr__(self, "led_centers_m", centers)
        object.__setattr__(self, "main_y_bounds_m", main_bounds)
        object.__setattr__(self, "total_y_bounds_m", total_bounds)

    @property
    def inter_led_midpoints_m(self) -> np.ndarray:
        """Return the four world-frame midpoints between adjacent LEDs."""
        midpoints = 0.5 * (self.led_centers_m[:-1] + self.led_centers_m[1:])
        midpoints.setflags(write=False)
        return midpoints


def make_fingertip_mesh(
    fingertip: Fingertip,
    *,
    extrusion_depth_mm: float = 11.0,
    element_size_mm: float = 1.0,
) -> FingertipMesh:
    """Discretize the silicone and carrier of one fingertip assembly."""
    if not isinstance(fingertip, Fingertip):
        raise TypeError("fingertip must be a Fingertip")

    require_positive("extrusion_depth_mm", extrusion_depth_mm)
    require_positive("element_size_mm", element_size_mm)

    silicone, bonded_vertex_indices = _make_silicone_mesh(
        fingertip.silicone,
        fingertip.bonding_interface,
        extrusion_depth_mm=extrusion_depth_mm,
        element_size_mm=element_size_mm,
    )
    carrier = _make_carrier_mesh(
        fingertip.carrier,
        extrusion_depth_mm=extrusion_depth_mm,
    )
    carrier_collision = _make_carrier_collision_mesh(
        fingertip.carrier,
        fingertip.silicone,
        extrusion_depth_mm=extrusion_depth_mm,
    )

    return FingertipMesh(
        fingertip=fingertip,
        silicone=silicone,
        carrier=carrier,
        carrier_collision=carrier_collision,
        bonded_vertex_indices=bonded_vertex_indices,
    )


def make_fingertip_5led_mesh(
    fingertip: Fingertip,
    *,
    element_size_mm: float = 1.0,
) -> Fingertip5LEDMesh:
    """Build the full five-LED mesh without changing the local morphology."""
    if not isinstance(fingertip, Fingertip):
        raise TypeError("fingertip must be a Fingertip")
    require_positive("element_size_mm", element_size_mm)
    if fingertip.parameters.geometry.void_height_mm != 0.0:
        raise ValueError(
            "the full five-LED production geometry requires "
            "void_height_mm=0; LED clearance comes from the stem recesses"
        )

    silicone, bonded_vertex_indices = _make_silicone_5led_mesh(
        fingertip.silicone,
        fingertip.bonding_interface,
        main_y_bounds_mm=MAIN_Y_BOUNDS_MM,
        distal_end_cap_length_mm=DISTAL_END_CAP_LENGTH_MM,
        element_size_mm=element_size_mm,
    )
    carrier = _make_carrier_5led_mesh(
        fingertip.carrier,
        main_length_mm=MAIN_LENGTH_MM,
        distal_end_cap_length_mm=DISTAL_END_CAP_LENGTH_MM,
        led_centers_y_mm=led_centers_y_mm(),
        led_recess_width_mm=LED_RECESS_WIDTH_MM,
        led_recess_depth_mm=LED_RECESS_DEPTH_MM,
    )
    carrier_collision = _make_carrier_5led_collision_mesh(
        fingertip.carrier,
        fingertip.silicone,
        main_length_mm=MAIN_LENGTH_MM,
        led_centers_y_mm=led_centers_y_mm(),
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
            for y_mm in led_centers_y_mm()
        ],
        dtype=np.float64,
    )
    return Fingertip5LEDMesh(
        fingertip=fingertip,
        silicone=silicone,
        carrier=carrier,
        carrier_collision=carrier_collision,
        bonded_vertex_indices=bonded_vertex_indices,
        led_centers_m=centers_m,
        main_y_bounds_m=tuple(
            _MM_TO_M * value for value in MAIN_Y_BOUNDS_MM
        ),
        total_y_bounds_m=tuple(
            _MM_TO_M * value for value in TOTAL_Y_BOUNDS_MM
        ),
    )


__all__ = [
    "DISTAL_END_CAP_LENGTH_MM",
    "Fingertip5LEDMesh",
    "FingertipMesh",
    "LED_PITCH_MM",
    "LED_RECESS_DEPTH_MM",
    "LED_RECESS_WIDTH_MM",
    "MAIN_LENGTH_MM",
    "MAIN_Y_BOUNDS_MM",
    "NUM_LEDS",
    "TOTAL_LENGTH_MM",
    "TOTAL_Y_BOUNDS_MM",
    "led_centers_y_mm",
    "make_fingertip_5led_mesh",
    "make_fingertip_mesh",
]
