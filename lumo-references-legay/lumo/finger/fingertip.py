"""Public physical fingertip facade."""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import Polygon, box

from lumo.finger.fingertip_geometry import FingertipModel
from lumo.finger.fingertip_parameters import FingertipParameters, OpticalParameters
from lumo.finger.led import LED
from lumo.finger.extrusion import FingertipSolid, build_fingertip_solid


class InvalidFingertip(ValueError):
    """Raised when physical fingertip components do not fit together."""


@dataclass(frozen=True)
class Fingertip:
    """Physical fingertip geometry, LED, and parameterized optical material."""

    parameters: FingertipParameters = field(default_factory=FingertipParameters)
    led: LED = field(default_factory=LED)
    geometry: FingertipModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        geometry = FingertipModel(self.parameters)
        tolerance = self.parameters.geometry_length_tolerance_mm
        if self.led.width_mm > self.parameters.stem_width + tolerance:
            raise InvalidFingertip(
                "the LED package width exceeds the rigid stem width"
            )
        if self.led.height_mm > self.parameters.stem_height + tolerance:
            raise InvalidFingertip(
                "the LED package height exceeds the rigid stem height"
            )
        object.__setattr__(self, "geometry", geometry)

    @property
    def optical(self) -> OpticalParameters:
        """Return the optical material owned by the fingertip parameters."""
        return self.parameters.optical

    @property
    def led_source(self) -> tuple[float, float]:
        """Return the source at the center of the LED's distal edge [mm]."""
        return 0.0, -self.parameters.stem_height

    @property
    def led_package_geometry(self) -> Polygon:
        """Return LED package placement metadata for optics and plotting."""
        stem_tip_y = -self.parameters.stem_height
        return box(
            -self.led.width_mm / 2.0,
            stem_tip_y,
            self.led.width_mm / 2.0,
            stem_tip_y + self.led.height_mm,
        )

    @property
    def emission_axis(self) -> tuple[float, float]:
        """Return the distal unit emission axis in model coordinates."""
        return 0.0, -1.0

    def solid(self, extrusion_depth_mm: float = 11.0) -> FingertipSolid:
        """Build the independent semantic 3D representative cell."""
        return build_fingertip_solid(
            self.geometry,
            extrusion_depth_mm=extrusion_depth_mm,
        )
