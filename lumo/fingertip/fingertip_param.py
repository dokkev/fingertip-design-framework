"""Complete physical parameter set for one LUMO fingertip."""

from __future__ import annotations

from dataclasses import dataclass, field

from .geometric_param import FingertipGeometry
from .mechanical_param import SiliconeMechanics
from .optical_param import (
    DRAGON_SKIN_10_NV_OPTICS_NOMINAL,
    LEDParameters,
    SiliconeOptics,
)


@dataclass(frozen=True)
class FingertipParameters:
    """Physical parameters defining one complete fingertip."""

    geometry: FingertipGeometry = field(default_factory=FingertipGeometry)
    mechanics: SiliconeMechanics = field(default_factory=SiliconeMechanics)
    optics: SiliconeOptics = DRAGON_SKIN_10_NV_OPTICS_NOMINAL
    led: LEDParameters = field(default_factory=LEDParameters)

    def __post_init__(self) -> None:
        self._validate_led_fit()

    def _validate_led_fit(self) -> None:
        if self.led.width_mm > self.geometry.stem_width_mm:
            raise ValueError(
                "LED width must not exceed stem width: "
                f"led={self.led.width_mm:g} mm, "
                f"stem={self.geometry.stem_width_mm:g} mm"
            )

        if self.led.height_mm > self.geometry.stem_height_mm:
            raise ValueError(
                "LED height must not exceed stem height: "
                f"led={self.led.height_mm:g} mm, "
                f"stem={self.geometry.stem_height_mm:g} mm"
            )


__all__ = ["FingertipParameters"]
