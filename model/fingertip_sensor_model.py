"""Physical fingertip geometry together with optical components."""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import Polygon, box

from model.fingertip_model import FingertipModel
from model.led_parameters import LEDParameters
from model.optical_material_parameters import OpticalMaterialParameters


class InvalidFingertipSensorModel(ValueError):
    """Raised when physical sensor metadata does not fit the assembly."""


@dataclass(frozen=True)
class FingertipSensorModel:
    """Physical fingertip geometry together with optical components."""

    geometry: FingertipModel
    led: LEDParameters = field(default_factory=LEDParameters)
    optical_material: OpticalMaterialParameters = field(
        default_factory=OpticalMaterialParameters
    )

    def __post_init__(self) -> None:
        """Validate sensor metadata without changing mechanical geometry."""
        if not isinstance(self.geometry, FingertipModel):
            raise InvalidFingertipSensorModel(
                "geometry must be a FingertipModel"
            )
        tolerance = self.geometry.parameters.geometry_tolerance
        if self.led.width_mm > self.geometry.parameters.stem_width + tolerance:
            raise InvalidFingertipSensorModel(
                "the LED package width exceeds the rigid stem width"
            )
        if self.led.height_mm > self.geometry.parameters.stem_height + tolerance:
            raise InvalidFingertipSensorModel(
                "the LED package height exceeds the rigid stem height"
            )
        min_x, min_y, max_x, _ = self.led_package_geometry.bounds
        expected_source = (0.5 * (min_x + max_x), min_y)
        source = self.led_source_position_2d
        if (
            abs(source[0] - expected_source[0]) > tolerance
            or abs(source[1] - expected_source[1]) > tolerance
        ):
            raise InvalidFingertipSensorModel(
                "the LED source is not centered on the lower emitting edge"
            )

    @classmethod
    def from_geometry(
        cls,
        geometry: FingertipModel,
        *,
        led: LEDParameters | None = None,
        optical_material: OpticalMaterialParameters | None = None,
    ) -> FingertipSensorModel:
        """Wrap mechanical geometry with default or supplied optical metadata."""
        return cls(
            geometry=geometry,
            led=led or LEDParameters(),
            optical_material=optical_material or OpticalMaterialParameters(),
        )

    @property
    def led_package_geometry(self) -> Polygon:
        """Return the embedded LED package as non-mechanical metadata."""
        stem_tip_y = self.geometry.parameters.stem_tip_y
        return box(
            -self.led.width_mm / 2.0,
            stem_tip_y,
            self.led.width_mm / 2.0,
            stem_tip_y + self.led.height_mm,
        )

    @property
    def led_source_position_2d(self) -> tuple[float, float]:
        """Return the center of the distal emitting edge in millimeters."""
        return 0.0, self.geometry.parameters.stem_tip_y

    @property
    def led_source_position_3d(self) -> tuple[float, float, float]:
        """Return the undeformed source position on the z=0 center plane."""
        x, y = self.led_source_position_2d
        return x, y, 0.0

    @property
    def led_emission_axis_2d(self) -> tuple[float, float]:
        """Return the unit distal emission axis in fingertip coordinates."""
        return 0.0, -1.0
