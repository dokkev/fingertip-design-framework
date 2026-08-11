"""Validated dimensions for the parameterized LIT Hand fingertip pad."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping
from warnings import warn


class InvalidFingertipParameters(ValueError):
    """Raised when fingertip dimensions cannot define a valid LIT pad."""


@dataclass(frozen=True)
class FingertipParameters:
    """Fingertip geometry and compliant-pad mechanical parameters.

    All dimensions are in millimeters. The link-pad interface is ``y = 0`` and
    the distal direction is negative ``y``. ``void_width`` is the one-sided
    clearance beside the stem; ``void_height`` is the additional clearance
    below the stem tip. ``young_modulus_mpa`` is in MPa and ``poisson_ratio``
    is dimensionless.
    """

    vertical_pad_width: float = 20.0
    vertical_pad_height: float = 3.0
    semielliptical_pad_width: float = 20.0
    semielliptical_pad_height: float = 7.0
    link_thickness: float = 3.5
    stem_width: float = 7.6
    stem_height: float = 6.0
    void_width: float = 0.0
    void_height: float = 0.0
    bonded: bool = True
    arc_resolution: int = 128
    geometry_tolerance: float = 1e-9
    young_modulus_mpa: float = 1.0
    poisson_ratio: float = 0.49

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

    @classmethod
    def from_legacy_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        vertical_pad_height: float,
    ) -> FingertipParameters:
        """Migrate an old ``pad_width/pad_height`` mapping explicitly.

        The former ``pad_width`` supplies both component widths because the new
        geometry requires width continuity. The former ``pad_height`` was the
        semi-axis and therefore maps only to ``semielliptical_pad_height``.
        ``vertical_pad_height`` has no legacy equivalent and must be supplied.
        """
        migrated = dict(values)
        if "pad_width" not in migrated or "pad_height" not in migrated:
            raise InvalidFingertipParameters(
                "legacy migration requires pad_width and pad_height"
            )
        new_names = {
            "vertical_pad_width",
            "vertical_pad_height",
            "semielliptical_pad_width",
            "semielliptical_pad_height",
        }
        conflicts = sorted(new_names.intersection(migrated))
        if conflicts:
            raise InvalidFingertipParameters(
                "legacy migration cannot mix old and new pad parameters: "
                + ", ".join(conflicts)
            )
        legacy_width = migrated.pop("pad_width")
        legacy_semi_axis = migrated.pop("pad_height")
        return cls(
            vertical_pad_width=legacy_width,
            vertical_pad_height=vertical_pad_height,
            semielliptical_pad_width=legacy_width,
            semielliptical_pad_height=legacy_semi_axis,
            **migrated,
        )

    @property
    def link_width(self) -> float:
        """Width of the top rigid plate, equal to the vertical-pad width."""
        return self.vertical_pad_width

    @property
    def ellipse_start_y(self) -> float:
        """Vertical coordinate where the lower semi-ellipse begins."""
        return -self.vertical_pad_height

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
        return -(self.vertical_pad_height + self.semielliptical_pad_height)

    @property
    def total_pad_depth(self) -> float:
        """Total external depth from the interface to the distal pad tip."""
        return self.vertical_pad_height + self.semielliptical_pad_height

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
    def bonded_segment_length(self) -> float:
        """Length of either upper bonded segment outside the cutout."""
        return self.vertical_pad_width / 2.0 - self.cutout_half_width

    @property
    def void_area(self) -> float:
        """Area of clearance left after the rigid stem fills the cutout."""
        return (
            self.cutout_width * self.cutout_height - self.stem_width * self.stem_height
        )

    def validate(self) -> None:
        """Raise ``InvalidFingertipParameters`` for inconsistent dimensions."""
        dimensions = {
            "vertical_pad_width": self.vertical_pad_width,
            "vertical_pad_height": self.vertical_pad_height,
            "semielliptical_pad_width": self.semielliptical_pad_width,
            "semielliptical_pad_height": self.semielliptical_pad_height,
            "link_thickness": self.link_thickness,
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
            "vertical_pad_width",
            "vertical_pad_height",
            "semielliptical_pad_width",
            "semielliptical_pad_height",
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
        if not isinstance(self.bonded, bool):
            raise InvalidFingertipParameters("bonded must be a boolean")

        if (
            abs(self.vertical_pad_width - self.semielliptical_pad_width)
            > self.geometry_tolerance
        ):
            raise InvalidFingertipParameters(
                "vertical_pad_width and semielliptical_pad_width must be equal "
                "within geometry_tolerance to avoid a shoulder"
            )
        if self.cutout_width >= self.vertical_pad_width - self.geometry_tolerance:
            raise InvalidFingertipParameters(
                "cutout_width must be smaller than vertical_pad_width: "
                f"cutout_width={self.cutout_width:g}, "
                f"vertical_pad_width={self.vertical_pad_width:g}"
            )
