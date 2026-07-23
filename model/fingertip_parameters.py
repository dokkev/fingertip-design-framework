"""Validated dimensions for the parameterized LIT Hand fingertip pad."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from warnings import warn


class InvalidFingertipParameters(ValueError):
    """Raised when fingertip dimensions cannot define a valid LIT pad."""


@dataclass(frozen=True)
class FingertipParameters:
    """Dimensions for a rigid link inserted into a compliant half-ellipse pad.

    All values are in millimeters. The flat link-pad plane is ``y = 0``. The
    compliant pad occupies ``y <= 0`` and the rigid link plate occupies
    ``y >= 0``. ``void_width`` is the clearance on each side of the stem, while
    ``void_height`` is the clearance below the stem tip. ``bonded`` is retained
    only for source compatibility; the upper link-pad interface is always
    bonded regardless of its value.
    """

    pad_width: float = 30.0
    pad_height: float = 18.0
    link_thickness: float = 3.5
    stem_width: float = 7.0
    stem_height: float = 7.0
    void_width: float = 0.0
    void_height: float = 0.0
    bonded: bool = True
    arc_resolution: int = 128
    geometry_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        """Validate values immediately so every instance is usable."""
        self.validate()
        if not self.bonded:
            warn(
                "bonded=False is deprecated and ignored; the upper link-pad "
                "interface is always bonded",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def link_width(self) -> float:
        """Width of the top rigid plate, equal to the pad width."""
        return self.pad_width

    @property
    def cutout_width(self) -> float:
        """Total centered cutout width around the rigid stem."""
        return self.stem_width + 2.0 * self.void_width

    @property
    def cutout_half_width(self) -> float:
        """Distance from the symmetry axis to either cutout side."""
        return self.stem_width / 2.0 + self.void_width

    @property
    def cutout_height(self) -> float:
        """Total cutout depth from ``y = 0`` into the pad."""
        return self.stem_height + self.void_height

    @property
    def cutout_depth(self) -> float:
        """Alias for the cutout depth used by boundary construction."""
        return self.cutout_height

    @property
    def bonded_segment_length(self) -> float:
        """Length of either upper bonded segment outside the cutout."""
        return self.pad_width / 2.0 - self.cutout_half_width

    @property
    def void_area(self) -> float:
        """Area of clearance left after the rigid stem fills the cutout."""
        return (
            self.cutout_width * self.cutout_height - self.stem_width * self.stem_height
        )

    def validate(self) -> None:
        """Raise ``InvalidFingertipParameters`` for inconsistent dimensions."""
        dimensions = {
            "pad_width": self.pad_width,
            "pad_height": self.pad_height,
            "link_thickness": self.link_thickness,
            "stem_width": self.stem_width,
            "stem_height": self.stem_height,
            "void_width": self.void_width,
            "void_height": self.void_height,
            "geometry_tolerance": self.geometry_tolerance,
        }
        for name, value in dimensions.items():
            if not isfinite(value):
                raise InvalidFingertipParameters(f"{name} must be finite")

        for name in (
            "pad_width",
            "pad_height",
            "link_thickness",
            "stem_width",
            "stem_height",
        ):
            if dimensions[name] <= 0.0:
                raise InvalidFingertipParameters(f"{name} must be greater than zero")

        if self.void_width < 0.0 or self.void_height < 0.0:
            raise InvalidFingertipParameters(
                "void_width and void_height must be nonnegative"
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
        if not isinstance(self.bonded, bool):
            raise InvalidFingertipParameters("bonded must be a boolean")

        if self.stem_width > self.pad_width + self.geometry_tolerance:
            raise InvalidFingertipParameters(
                "stem_width must not exceed pad_width: "
                f"stem_width={self.stem_width:g}, pad_width={self.pad_width:g}"
            )

        if self.bonded_segment_length <= self.geometry_tolerance:
            raise InvalidFingertipParameters(
                "the cutout must leave a positive bonded segment on both sides: "
                f"pad_width={self.pad_width:g}, cutout_width={self.cutout_width:g}, "
                f"geometry_tolerance={self.geometry_tolerance:g}"
            )

        normalized_corner_radius = (
            self.cutout_half_width / (self.pad_width / 2.0)
        ) ** 2 + (self.cutout_depth / self.pad_height) ** 2
        normalized_tolerance = self.geometry_tolerance / min(
            self.pad_width / 2.0, self.pad_height
        )
        if normalized_corner_radius > 1.0 + normalized_tolerance:
            raise InvalidFingertipParameters(
                "the cutout lower corners lie outside the half-ellipse: "
                f"cutout_half_width={self.cutout_half_width:g}, "
                f"cutout_depth={self.cutout_depth:g}, "
                f"normalized_ellipse_value={normalized_corner_radius:.6g}"
            )
