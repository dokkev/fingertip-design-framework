"""Optical properties for an external mechanical contact object."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


class InvalidIndenterOptics(ValueError):
    """Raised when an external indenter optical contract is invalid."""


@dataclass(frozen=True)
class IndenterOptics:
    """Small explicit boundary contract for the external indenter."""

    boundary_model: Literal["absorber", "dielectric"]
    refractive_index: float | None = None

    def __post_init__(self) -> None:
        if self.boundary_model not in ("absorber", "dielectric"):
            raise InvalidIndenterOptics(
                "boundary_model must be 'absorber' or 'dielectric'"
            )
        if self.boundary_model == "absorber":
            if self.refractive_index is not None:
                raise InvalidIndenterOptics(
                    "absorber indenter optics must not specify refractive_index"
                )
            return
        if self.refractive_index is None or not math.isfinite(
            float(self.refractive_index)
        ) or self.refractive_index <= 0.0:
            raise InvalidIndenterOptics(
                "dielectric indenter optics require a finite positive "
                "refractive_index"
            )


__all__ = ["IndenterOptics", "InvalidIndenterOptics"]
