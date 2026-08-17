"""Public physical fingertip facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

from shapely.geometry import Polygon, box

from model.fingertip_model import FingertipBoundaries, FingertipModel
from model.fingertip_parameters import FingertipParameters
from model.optical import LED, OpticalMaterial


class InvalidFingertip(ValueError):
    """Raised when physical fingertip components do not fit together."""


@dataclass(frozen=True)
class Fingertip:
    """Physical fingertip geometry, LED, and bulk optical material."""

    parameters: FingertipParameters = field(default_factory=FingertipParameters)
    led: LED = field(default_factory=LED)
    optical: OpticalMaterial = field(default_factory=OpticalMaterial)
    geometry: FingertipModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        geometry = FingertipModel(self.parameters)
        tolerance = self.parameters.geometry_tolerance
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
    def boundaries(self) -> FingertipBoundaries:
        """Return the physical geometry's named boundary semantics."""
        return self.geometry.boundaries

    @property
    def led_source(self) -> tuple[float, float]:
        """Return the source at the center of the LED's distal edge [mm]."""
        return 0.0, self.parameters.stem_tip_y

    @property
    def led_package_geometry(self) -> Polygon:
        """Return LED package placement metadata for optics and plotting."""
        stem_tip_y = self.parameters.stem_tip_y
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

    def mesh(self, settings: Any | None = None) -> Any:
        """Generate one discrete mesh; repeated calls may use other settings."""
        generator = import_module("mesh.fingertip").generate_fingertip_mesh
        settings_for_level = import_module(
            "mesh.types"
        ).mesh_settings_for_level
        selected = settings or settings_for_level("medium")
        return generator(self.geometry, selected)

    def solid(self, extrusion_depth_mm: float = 11.0) -> Any:
        """Build the independent semantic 3D representative cell."""
        builder = import_module("model.solid").build_fingertip_solid
        return builder(self.geometry, extrusion_depth_mm=extrusion_depth_mm)
