"""Optical properties for opaque or dielectric rigid boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


class InvalidObjectBoundaryOptics(ValueError):
    """Raised when a rigid optical-boundary contract is invalid."""


@dataclass(frozen=True)
class ObjectBoundaryOptics:
    """Small explicit contract shared by external objects and carriers."""

    boundary_model: Literal["absorber", "dielectric"]
    refractive_index: float | None = None

    def __post_init__(self) -> None:
        if self.boundary_model not in ("absorber", "dielectric"):
            raise InvalidObjectBoundaryOptics(
                "boundary_model must be 'absorber' or 'dielectric'"
            )
        if self.boundary_model == "absorber":
            if self.refractive_index is not None:
                raise InvalidObjectBoundaryOptics(
                    "absorber object-boundary optics must not specify refractive_index"
                )
            return
        if self.refractive_index is None or not math.isfinite(
            float(self.refractive_index)
        ) or self.refractive_index <= 0.0:
            raise InvalidObjectBoundaryOptics(
                "dielectric object-boundary optics require a finite positive "
                "refractive_index"
            )


# Keep the established public name for external indenter callers while making
# the shared contract explicit for the FULL_3D carrier path.
IndenterOptics = ObjectBoundaryOptics
CarrierOptics = ObjectBoundaryOptics
InvalidIndenterOptics = InvalidObjectBoundaryOptics


__all__ = [
    "CarrierOptics",
    "IndenterOptics",
    "InvalidIndenterOptics",
    "InvalidObjectBoundaryOptics",
    "ObjectBoundaryOptics",
]
