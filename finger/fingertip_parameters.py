"""Validated physical and representation inputs for fingertip geometry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from math import sqrt

from util.validation import require_nonnegative, require_positive


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
class KinematicParameters:
    """Geometry and representation inputs for one fingertip morphology.

    All dimensions are in millimeters. The fingertip's material law is owned
    separately by :class:`ViscoelasticParameters`.
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
    geometry_length_tolerance_mm: float = 1e-9
    geometry_area_tolerance_mm2: float = 1e-9

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``InvalidFingertipParameters`` for inconsistent dimensions."""
        error_type = InvalidFingertipParameters
        for name, value in (
            ("flat_pad_width", self.flat_pad_width),
            ("flat_pad_height", self.flat_pad_height),
            ("semielliptical_pad_height", self.semielliptical_pad_height),
            ("link_thickness", self.link_thickness),
            ("bond_extension_width", self.bond_extension_width),
            ("bond_extension_height", self.bond_extension_height),
            ("stem_width", self.stem_width),
            ("stem_height", self.stem_height),
            ("geometry_length_tolerance_mm", self.geometry_length_tolerance_mm),
            ("geometry_area_tolerance_mm2", self.geometry_area_tolerance_mm2),
        ):
            require_positive(name, value, error_type=error_type)

        require_nonnegative(
            "void_width",
            self.void_width,
            error_type=error_type,
        )
        require_nonnegative(
            "void_height",
            self.void_height,
            error_type=error_type,
        )
        total_pad_depth = self.flat_pad_height + self.semielliptical_pad_height
        cutout_width = self.stem_width + 2.0 * self.void_width
        cutout_half_width = cutout_width / 2.0
        cutout_height = self.stem_height + self.void_height
        if total_pad_depth > MAX_TOTAL_PAD_DEPTH_MM + self.geometry_length_tolerance_mm:
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
        if cutout_half_width >= half_width - self.geometry_length_tolerance_mm:
            raise InvalidFingertipParameters(
                "cutout must remain strictly inside the external half-width: "
                f"cutout_half_width={cutout_half_width:g}, "
                "available_half_width="
                f"{half_width - self.geometry_length_tolerance_mm:g}, "
                "geometry_length_tolerance_mm="
                f"{self.geometry_length_tolerance_mm:g}"
            )

        penetration_depth = max(0.0, cutout_height - self.flat_pad_height)
        if penetration_depth > 0.0:
            available_ellipse_depth = ellipse_depth_at_cutout_mm(
                flat_pad_width=self.flat_pad_width,
                semielliptical_pad_height=self.semielliptical_pad_height,
                stem_width=self.stem_width,
                void_width=self.void_width,
            )
            if (
                penetration_depth
                >= available_ellipse_depth - self.geometry_length_tolerance_mm
            ):
                raise InvalidFingertipParameters(
                    "cutout bottom must remain strictly inside the "
                    "semielliptical envelope: "
                    f"penetration_depth={penetration_depth:g}, "
                    f"available_depth={available_ellipse_depth:g}, "
                    "geometry_length_tolerance_mm="
                    f"{self.geometry_length_tolerance_mm:g}"
                )


@dataclass(frozen=True)
class ViscoelasticParameters:
    """Material coefficients consumed by the Newton soft-body model.

    ``k_mu_pa`` and ``k_lambda_pa`` are Newton's constitutive coefficients;
    they are intentionally not presented as an inferred Young's modulus and
    Poisson ratio. ``density_kg_m3`` is kept with the material law because it
    is the companion inertial input for the same soft-body model.
    """

    density_kg_m3: float = 1.0e3
    k_mu_pa: float = 1.0e5
    k_lambda_pa: float = 1.0e5
    k_damp: float = 10.0

    def __post_init__(self) -> None:
        error_type = InvalidFingertipParameters
        for name, value in (
            ("density_kg_m3", self.density_kg_m3),
            ("k_mu_pa", self.k_mu_pa),
            ("k_lambda_pa", self.k_lambda_pa),
        ):
            require_positive(name, value, error_type=error_type)
        require_nonnegative("k_damp", self.k_damp, error_type=error_type)


@dataclass(frozen=True)
class OpticalParameters:
    """Bulk optical inputs consumed by the FULL_3D transport model."""

    refractive_index_air: float = 1.0
    refractive_index_silicone: float = 1.41
    absorption_per_mm: float = 0.02

    def __post_init__(self) -> None:
        error_type = InvalidFingertipParameters
        require_positive(
            "refractive_index_air",
            self.refractive_index_air,
            error_type=error_type,
        )
        require_positive(
            "refractive_index_silicone",
            self.refractive_index_silicone,
            error_type=error_type,
        )
        require_nonnegative(
            "absorption_per_mm",
            self.absorption_per_mm,
            error_type=error_type,
        )


@dataclass(frozen=True)
class FingertipParameters(KinematicParameters):
    """Complete fingertip inputs: kinematics, mechanics, and optics."""

    viscoelastic: ViscoelasticParameters = field(
        default_factory=ViscoelasticParameters
    )
    optical: OpticalParameters = field(default_factory=OpticalParameters)

    def __post_init__(self) -> None:
        # Accept nested JSON/dict payloads at this boundary while keeping the
        # in-memory domain representation typed.
        if isinstance(self.viscoelastic, Mapping):
            object.__setattr__(
                self,
                "viscoelastic",
                ViscoelasticParameters(**dict(self.viscoelastic)),
            )
        if isinstance(self.optical, Mapping):
            object.__setattr__(
                self,
                "optical",
                OpticalParameters(**dict(self.optical)),
            )
        if not isinstance(self.viscoelastic, ViscoelasticParameters):
            raise TypeError("viscoelastic must be ViscoelasticParameters")
        if not isinstance(self.optical, OpticalParameters):
            raise TypeError("optical must be OpticalParameters")
        self.validate()

    def validate(self) -> None:
        """Validate the kinematic group; nested groups validate on construction."""
        KinematicParameters.validate(self)


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
    "KinematicParameters",
    "FingertipParameters",
    "InvalidFingertipParameters",
    "MAX_TOTAL_PAD_DEPTH_MM",
    "PRODUCTION_MINIMUM_SILICONE_THICKNESS_MM",
    "OpticalParameters",
    "ViscoelasticParameters",
    "ellipse_depth_at_cutout_mm",
    "fingertip_parameters_fingerprint",
]
