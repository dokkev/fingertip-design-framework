"""Fingertip domain ownership for the new LUMO implementation."""

from .bonding_interface import BondingInterface
from .fingertip import Carrier, Fingertip, Silicone
from .fingertip_param import FingertipParameters
from .geometric_param import FingertipGeometry
from .layout import (
    ACTIVE_Y_BOUNDS_MM,
    DISTAL_END_CAP_LENGTH_MM,
    LED_CENTERS_Y_MM,
    LED_RECESS_DEPTH_MM,
    LED_RECESS_WIDTH_MM,
    TOTAL_Y_BOUNDS_MM,
)
from .mechanical_param import (
    MECHANICS_PRESETS,
    SILICONE_MECHANICS,
    SOLARIS_MECHANICS,
    SiliconeMechanics,
)
from .optical_param import (
    DRAGON_SKIN_10_NV_OPTICS_HIGH,
    DRAGON_SKIN_10_NV_OPTICS_LOW,
    DRAGON_SKIN_10_NV_OPTICS_NOMINAL,
    SOLARIS_OPTICS_HIGH,
    SOLARIS_OPTICS_LOW,
    SOLARIS_OPTICS_NOMINAL,
    LEDParameters,
    OPTICAL_PRESETS,
    SiliconeOptics,
)


__all__ = [
    "ACTIVE_Y_BOUNDS_MM",
    "DISTAL_END_CAP_LENGTH_MM",
    "Fingertip",
    "Silicone",
    "Carrier",
    "BondingInterface",
    "SILICONE_MECHANICS",
    "SOLARIS_MECHANICS",
    "FingertipParameters",
    "FingertipGeometry",
    "LED_CENTERS_Y_MM",
    "LED_RECESS_DEPTH_MM",
    "LED_RECESS_WIDTH_MM",
    "MECHANICS_PRESETS",
    "DRAGON_SKIN_10_NV_OPTICS_HIGH",
    "DRAGON_SKIN_10_NV_OPTICS_LOW",
    "DRAGON_SKIN_10_NV_OPTICS_NOMINAL",
    "LEDParameters",
    "OPTICAL_PRESETS",
    "SOLARIS_OPTICS_HIGH",
    "SOLARIS_OPTICS_LOW",
    "SOLARIS_OPTICS_NOMINAL",
    "SiliconeOptics",
    "SiliconeMechanics",
    "TOTAL_Y_BOUNDS_MM",
]
