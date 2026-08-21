"""Public physical fingertip model."""

from lumo.finger.fingertip import Fingertip, InvalidFingertip
from lumo.finger.fingertip_parameters import (
    FingertipParameters,
    InvalidFingertipParameters,
    KinematicParameters,
    MAX_TOTAL_PAD_DEPTH_MM,
    OpticalParameters,
    PRODUCTION_MINIMUM_SILICONE_THICKNESS_MM,
    ViscoelasticParameters,
    ellipse_depth_at_cutout_mm,
    fingertip_parameters_fingerprint,
)
from lumo.finger.fingertip_geometry import (
    SiliconeThicknessMeasures,
    silicone_thickness_measures,
    validate_minimum_silicone_thickness,
)
from lumo.finger.led import LED
from lumo.finger.extrusion import (
    DEFAULT_EXTRUSION_DEPTH_MM,
    FingertipSolid,
    SolidSurfaceDefinition,
    build_fingertip_solid,
)

__all__ = [
    "Fingertip",
    "FingertipParameters",
    "KinematicParameters",
    "InvalidFingertip",
    "InvalidFingertipParameters",
    "MAX_TOTAL_PAD_DEPTH_MM",
    "OpticalParameters",
    "PRODUCTION_MINIMUM_SILICONE_THICKNESS_MM",
    "LED",
    "ViscoelasticParameters",
    "ellipse_depth_at_cutout_mm",
    "SiliconeThicknessMeasures",
    "silicone_thickness_measures",
    "validate_minimum_silicone_thickness",
    "fingertip_parameters_fingerprint",
    "DEFAULT_EXTRUSION_DEPTH_MM",
    "FingertipSolid",
    "SolidSurfaceDefinition",
    "build_fingertip_solid",
]
