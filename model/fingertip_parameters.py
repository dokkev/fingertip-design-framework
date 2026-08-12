"""Validated dimensions for the parameterized LIT Hand fingertip pad."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


class InvalidFingertipParameters(ValueError):
    """Raised when fingertip dimensions cannot define a valid LIT pad."""


@dataclass(frozen=True)
class FingertipParameters:
    """Fingertip geometry and compliant-pad mechanical parameters.

    All dimensions are in millimeters. The link-pad interface is ``y = 0``
    and the distal direction is negative ``y``. ``void_width`` is the
    one-sided clearance beside the stem; ``void_height`` is the additional
    clearance below the stem tip. ``young_modulus_mpa`` is in MPa and
    ``poisson_ratio`` is dimensionless.
    """

    flat_pad_width: float = 20.0
    flat_pad_height: float = 3.0
    semielliptical_pad_height: float = 7.0
    link_width: float = 12.0
    link_thickness: float = 3.5
    bond_extension_height: float = 2.0
    stem_width: float = 7.6
    stem_height: float = 6.0
    void_width: float = 0.0
    void_height: float = 0.0
    arc_resolution: int = 128
    geometry_tolerance: float = 1e-9
    young_modulus_mpa: float = 1.0
    poisson_ratio: float = 0.49

    def __post_init__(self) -> None:
        """Validate values immediately so every instance is usable."""
        self.validate()

    @property
    def ellipse_start_y(self) -> float:
        """Vertical coordinate where the lower semi-ellipse begins."""
        return -self.flat_pad_height

    @property
    def stem_tip_y(self) -> float:
        """Vertical coordinate of the rigid stem tip."""
        return -self.stem_height

    @property
    def void_bottom_y(self) -> float:
        """Vertical coordinate of the complete internal cutout bottom."""
        return -(self.stem_height + self.void_height)

    @property
    def pad_tip_y(self) -> float:
        """Distal-most coordinate of the complete external pad envelope."""
        return -(self.flat_pad_height + self.semielliptical_pad_height)

    @property
    def total_pad_depth(self) -> float:
        """Total external depth from the interface to the distal pad tip."""
        return self.flat_pad_height + self.semielliptical_pad_height

    @property
    def cutout_width(self) -> float:
        """Total centered cutout width around the rigid stem."""
        return self.stem_width + 2.0 * self.void_width

    @property
    def cutout_half_width(self) -> float:
        """Distance from the symmetry axis to either cutout side."""
        return self.cutout_width / 2.0

    @property
    def cutout_height(self) -> float:
        """Total cutout depth from ``y = 0`` into the pad."""
        return self.stem_height + self.void_height

    @property
    def cutout_depth(self) -> float:
        """Depth alias retained for boundary and mesh construction."""
        return self.cutout_height

    @property
    def bond_extension_width(self) -> float:
        """Width of either compliant extension beside the rigid link."""
        return (self.flat_pad_width - self.link_width) / 2.0

    @property
    def bonded_segment_length(self) -> float:
        """Length of either L-shaped bonded pad-to-link segment."""
        return self.bond_extension_height + (
            self.link_width - self.cutout_width
        ) / 2.0

    @property
    def void_area(self) -> float:
        """Area of clearance left after the rigid stem fills the cutout."""
        return (
            self.cutout_width * self.cutout_height
            - self.stem_width * self.stem_height
        )

    def validate(self) -> None:
        """Raise ``InvalidFingertipParameters`` for inconsistent dimensions."""
        dimensions = {
            "flat_pad_width": self.flat_pad_width,
            "flat_pad_height": self.flat_pad_height,
            "semielliptical_pad_height": self.semielliptical_pad_height,
            "link_width": self.link_width,
            "link_thickness": self.link_thickness,
            "bond_extension_height": self.bond_extension_height,
            "stem_width": self.stem_width,
            "stem_height": self.stem_height,
            "void_width": self.void_width,
            "void_height": self.void_height,
            "young_modulus_mpa": self.young_modulus_mpa,
            "poisson_ratio": self.poisson_ratio,
            "geometry_tolerance": self.geometry_tolerance,
        }
        for name, value in dimensions.items():
            if not isfinite(value):
                raise InvalidFingertipParameters(f"{name} must be finite")

        for name in (
            "flat_pad_width",
            "flat_pad_height",
            "semielliptical_pad_height",
            "link_width",
            "link_thickness",
            "bond_extension_height",
            "stem_width",
            "stem_height",
        ):
            if dimensions[name] <= 0.0:
                raise InvalidFingertipParameters(f"{name} must be greater than zero")

        if self.void_width < 0.0 or self.void_height < 0.0:
            raise InvalidFingertipParameters(
                "void_width and void_height must be nonnegative"
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
        if self.flat_pad_width <= self.link_width:
            raise InvalidFingertipParameters(
                "flat_pad_width must be greater than link_width"
            )
        if self.cutout_width >= self.link_width:
            raise InvalidFingertipParameters(
                "cutout_width must be smaller than link_width: "
                f"cutout_width={self.cutout_width:g}, "
                f"link_width={self.link_width:g}"
            )
        if self.bond_extension_height > self.link_thickness:
            raise InvalidFingertipParameters(
                "bond_extension_height must not exceed link_thickness"
            )
