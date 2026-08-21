"""Complete physical parameter set for one LUMO fingertip."""

from __future__ import annotations

from dataclasses import dataclass, field

from .geometric_param import FingertipGeometry, InvalidFingertipParameters
from .optical_param import LEDParameters, OpticalParameters
from .viscoelastic_param import ViscoelasticParameters


@dataclass(frozen=True)
class FingertipParameters:
    """Physical parameters defining one complete fingertip."""

    geometry: FingertipGeometry = field(default_factory=FingertipGeometry)
    viscoelastic: ViscoelasticParameters = field(
        default_factory=ViscoelasticParameters
    )
    optical: OpticalParameters = field(default_factory=OpticalParameters)
    led: LEDParameters = field(default_factory=LEDParameters)

    def __post_init__(self) -> None:
        self._validate_led_fit()

    def _validate_led_fit(self) -> None:
        if self.led.width_mm > self.geometry.stem_width_mm:
            raise InvalidFingertipParameters(
                "LED width must not exceed stem width: "
                f"led={self.led.width_mm:g} mm, "
                f"stem={self.geometry.stem_width_mm:g} mm"
            )

        if self.led.height_mm > self.geometry.stem_height_mm:
            raise InvalidFingertipParameters(
                "LED height must not exceed stem height: "
                f"led={self.led.height_mm:g} mm, "
                f"stem={self.geometry.stem_height_mm:g} mm"
            )


__all__ = ["FingertipParameters"]