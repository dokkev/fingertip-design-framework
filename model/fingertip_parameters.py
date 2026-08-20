"""Validated physical and representation inputs for fingertip geometry."""

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


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise InvalidFingertipParameters(f"{name} must be finite")


def _require_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise InvalidFingertipParameters(f"{name} must be greater than zero")


@dataclass(frozen=True)
class FingertipParameters:
    """Geometry inputs for one fingertip morphology.

    All dimensions are in millimeters. Numerical mechanics settings are owned
    separately by :class:`lumo.mechanics_contract.MechanicsContract`.
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

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``InvalidFingertipParameters`` for inconsistent dimensions."""
        _require_finite("flat_pad_width", self.flat_pad_width)
        _require_finite("flat_pad_height", self.flat_pad_height)
        _require_finite(
            "semielliptical_pad_height", self.semielliptical_pad_height
        )
        _require_finite("link_thickness", self.link_thickness)
        _require_finite("bond_extension_width", self.bond_extension_width)
        _require_finite("bond_extension_height", self.bond_extension_height)
        _require_finite("stem_width", self.stem_width)
        _require_finite("stem_height", self.stem_height)
        _require_finite("void_width", self.void_width)
        _require_finite("void_height", self.void_height)
        _require_finite("geometry_tolerance", self.geometry_tolerance)

        _require_positive("flat_pad_width", self.flat_pad_width)
        _require_positive("flat_pad_height", self.flat_pad_height)
        _require_positive(
            "semielliptical_pad_height", self.semielliptical_pad_height
        )
        _require_positive("link_thickness", self.link_thickness)
        _require_positive("bond_extension_width", self.bond_extension_width)
        _require_positive("bond_extension_height", self.bond_extension_height)
        _require_positive("stem_width", self.stem_width)
        _require_positive("stem_height", self.stem_height)
        _require_positive("geometry_tolerance", self.geometry_tolerance)

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
        if 2.0 * self.bond_extension_width + cutout_width >= self.flat_pad_width:
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

        penetration_depth = max(0.0, cutout_height - self.flat_pad_height)
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
    """Return the stable physical-morphology fingerprint."""
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


__all__ = [
    "FingertipParameters",
    "InvalidFingertipParameters",
    "MAX_TOTAL_PAD_DEPTH_MM",
    "PRODUCTION_MINIMUM_SILICONE_THICKNESS_MM",
    "ellipse_depth_at_cutout_mm",
    "fingertip_parameters_fingerprint",
]
