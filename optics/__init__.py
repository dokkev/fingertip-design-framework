"""Deformation-aware optical geometry, transport, and rendering contracts."""

from optics.geometry import (
    ExtrudedOpticalMeshTemplate,
    PadDeformationState2D,
    PadField2D,
    PadMeshTemplate2D,
)
from optics.source import led_source_position_2d, led_source_position_3d

__all__ = [
    "ExtrudedOpticalMeshTemplate",
    "PadDeformationState2D",
    "PadField2D",
    "PadMeshTemplate2D",
    "led_source_position_2d",
    "led_source_position_3d",
]
