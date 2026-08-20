"""Validated dimensions for the parameterized LIT Hand fingertip pad."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite, sqrt


class InvalidFingertipParameters(ValueError):
    """Raised when fingertip dimensions cannot define a valid LIT pad."""


PRODUCTION_MINIMUM_SILICONE_THICKNESS_MM = 5.0
MAX_TOTAL_PAD_DEPTH_MM = 30.0


def ellipse_depth_at_cutout_mm(
    *,
    flat_pad_width: float,
    semielliptical_pad_height: float,
    stem_width: float,
    void_width: float,
) -> float:
    """Return the analytic lower-envelope depth at the cutout side."""
    half_width = float(flat_pad_width) / 2.0
    cutout_half_width = float(stem_width) / 2.0 + float(void_width)
    normalized_x = cutout_half_width / half_width
    if not 0.0 <= normalized_x < 1.0:
        raise ValueError("cutout must lie strictly inside the pad half-width")
    return float(semielliptical_pad_height) * sqrt(1.0 - normalized_x**2)


@dataclass(frozen=True)
class FingertipParameters:
    """Fingertip geometry and compliant-pad mechanical parameters.

    All dimensions are in millimeters. The flat-pad, semi-ellipse, and rigid
    link share one full width: ``flat_pad_width``. The link-pad interface is
    ``y = 0`` and the distal direction is negative ``y``. ``void_width`` is
    the one-sided clearance beside the stem; ``void_height`` is the
    additional clearance below the stem tip. Production zero-height
    morphology uses a bonded basal stem/pad interface; positive height
    remains a finite-clearance diagnostic geometry.
    """

    flat_pad_width: float = 30.0
    flat_pad_height: float = 5.0
    semielliptical_pad_height: float = 9.0
    link_thickness: float = 3.5
    bond_extension_width: float = 4.0
    bond_extension_height: float = 2.0
    stem_width: float = 7.6
    stem_height: float = 6.0
    void_width: float = 1.0
    void_height: float = 0.0
    arc_resolution: int = 128
    geometry_tolerance: float = 1e-9
    young_modulus_mpa: float = 0.55
    poisson_ratio: float = 0.49

    def __post_init__(self) -> None:
        """Validate values immediately so every instance is usable."""
        self.validate()

    def validate(self) -> None:
        """Raise ``InvalidFingertipParameters`` for inconsistent dimensions."""
        if not isfinite(self.flat_pad_width):
            raise InvalidFingertipParameters("flat_pad_width must be finite")
        if not isfinite(self.flat_pad_height):
            raise InvalidFingertipParameters("flat_pad_height must be finite")
        if not isfinite(self.semielliptical_pad_height):
            raise InvalidFingertipParameters(
                "semielliptical_pad_height must be finite"
            )
        if not isfinite(self.link_thickness):
            raise InvalidFingertipParameters("link_thickness must be finite")
        if not isfinite(self.bond_extension_width):
            raise InvalidFingertipParameters(
                "bond_extension_width must be finite"
            )
        if not isfinite(self.bond_extension_height):
            raise InvalidFingertipParameters(
                "bond_extension_height must be finite"
            )
        if not isfinite(self.stem_width):
            raise InvalidFingertipParameters("stem_width must be finite")
        if not isfinite(self.stem_height):
            raise InvalidFingertipParameters("stem_height must be finite")
        if not isfinite(self.void_width):
            raise InvalidFingertipParameters("void_width must be finite")
        if not isfinite(self.void_height):
            raise InvalidFingertipParameters("void_height must be finite")
        if not isfinite(self.geometry_tolerance):
            raise InvalidFingertipParameters("geometry_tolerance must be finite")
        if not isfinite(self.young_modulus_mpa):
            raise InvalidFingertipParameters("young_modulus_mpa must be finite")
        if not isfinite(self.poisson_ratio):
            raise InvalidFingertipParameters("poisson_ratio must be finite")

        if self.flat_pad_width <= 0.0:
            raise InvalidFingertipParameters(
                "flat_pad_width must be greater than zero"
            )
        if self.flat_pad_height <= 0.0:
            raise InvalidFingertipParameters(
                "flat_pad_height must be greater than zero"
            )
        if self.semielliptical_pad_height <= 0.0:
            raise InvalidFingertipParameters(
                "semielliptical_pad_height must be greater than zero"
            )
        if self.link_thickness <= 0.0:
            raise InvalidFingertipParameters(
                "link_thickness must be greater than zero"
            )
        if self.bond_extension_width <= 0.0:
            raise InvalidFingertipParameters(
                "bond_extension_width must be greater than zero"
            )
        if self.bond_extension_height <= 0.0:
            raise InvalidFingertipParameters(
                "bond_extension_height must be greater than zero"
            )
        if self.stem_width <= 0.0:
            raise InvalidFingertipParameters("stem_width must be greater than zero")
        if self.stem_height <= 0.0:
            raise InvalidFingertipParameters(
                "stem_height must be greater than zero"
            )

        if self.void_width < 0.0 or self.void_height < 0.0:
            raise InvalidFingertipParameters(
                "void_width and void_height must be nonnegative"
            )
        total_pad_depth = self.flat_pad_height + self.semielliptical_pad_height
        cutout_width = self.stem_width + 2.0 * self.void_width
        cutout_half_width = cutout_width / 2.0
        cutout_height = self.stem_height + self.void_height
        if total_pad_depth > MAX_TOTAL_PAD_DEPTH_MM + self.geometry_tolerance:
            raise InvalidFingertipParameters(
                "total pad depth must not exceed 30 mm: "
                f"flat_pad_height={self.flat_pad_height:g}, "
                "semielliptical_pad_height="
                f"{self.semielliptical_pad_height:g}, "
                f"total={total_pad_depth:g}"
            )
        if self.young_modulus_mpa <= 0.0:
            raise InvalidFingertipParameters(
                "young_modulus_mpa must be greater than zero"
            )
        if not -1.0 < self.poisson_ratio < 0.5:
            raise InvalidFingertipParameters(
                "poisson_ratio must lie strictly between -1 and 0.5"
            )
        if self.geometry_tolerance <= 0.0:
            raise InvalidFingertipParameters(
                "geometry_tolerance must be greater than zero"
            )
        if (
            not isinstance(self.arc_resolution, int)
            or isinstance(self.arc_resolution, bool)
            or self.arc_resolution < 16
        ):
            raise InvalidFingertipParameters(
                "arc_resolution must be an integer of at least 16"
            )
        if self.bond_extension_height >= self.link_thickness:
            raise InvalidFingertipParameters(
                "bond_extension_height must be smaller than link_thickness"
            )
        if (
            2.0 * self.bond_extension_width + cutout_width
            >= self.flat_pad_width
        ):
            raise InvalidFingertipParameters(
                "2*bond_extension_width + cutout_width must be smaller than "
                "flat_pad_width: "
                f"bond_extension_width={self.bond_extension_width:g}, "
                f"cutout_width={cutout_width:g}, "
                f"flat_pad_width={self.flat_pad_width:g}"
            )

        half_width = self.flat_pad_width / 2.0
        if cutout_half_width >= half_width - self.geometry_tolerance:
            raise InvalidFingertipParameters(
                "cutout must remain strictly inside the external half-width: "
                f"cutout_half_width={cutout_half_width:g}, "
                f"available_half_width={half_width - self.geometry_tolerance:g}, "
                f"geometry_tolerance={self.geometry_tolerance:g}"
            )

        penetration_depth = max(
            0.0,
            cutout_height - self.flat_pad_height,
        )
        if penetration_depth > 0.0:
            available_ellipse_depth = ellipse_depth_at_cutout_mm(
                flat_pad_width=self.flat_pad_width,
                semielliptical_pad_height=self.semielliptical_pad_height,
                stem_width=self.stem_width,
                void_width=self.void_width,
            )
            if penetration_depth >= available_ellipse_depth - self.geometry_tolerance:
                raise InvalidFingertipParameters(
                    "cutout bottom must remain strictly inside the "
                    "semielliptical envelope: "
                    f"penetration_depth={penetration_depth:g}, "
                    f"available_depth={available_ellipse_depth:g}, "
                    f"geometry_tolerance={self.geometry_tolerance:g}"
                )


def fingertip_parameters_fingerprint(parameters: FingertipParameters) -> str:
    """Return a fingerprint for the physical morphology parameters.

    Sampling resolution, geometry tolerance, and material properties are
    representation/numerical/material settings rather than morphology
    identity.  The explicit payload is serialized only at this boundary so
    the same physical morphology has one stable identity across adapters.
    """
    if not isinstance(parameters, FingertipParameters):
        raise TypeError("parameters must be FingertipParameters")
    payload = json.dumps(
        {
            "flat_pad_width": parameters.flat_pad_width,
            "flat_pad_height": parameters.flat_pad_height,
            "semielliptical_pad_height": parameters.semielliptical_pad_height,
            "link_thickness": parameters.link_thickness,
            "bond_extension_width": parameters.bond_extension_width,
            "bond_extension_height": parameters.bond_extension_height,
            "stem_width": parameters.stem_width,
            "stem_height": parameters.stem_height,
            "void_width": parameters.void_width,
            "void_height": parameters.void_height,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
