"""Physical geometry parameters for the LUMO fingertip."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from lumo.util.scalar_validation import (
    require_finite,
    require_nonnegative,
    require_positive,
)


@dataclass(frozen=True)
class FingertipGeometry:
    """Physical dimensions defining one fingertip morphology.

    All dimensions are expressed in millimeters.

    Numerical representation settings such as mesh resolution and geometry
    tolerances are intentionally excluded from this class.
    """

    flat_pad_width_mm: float = 30.0
    flat_pad_height_mm: float = 5.0
    semiellipse_height_mm: float = 9.0

    link_thickness_mm: float = 10
    bond_extension_width_mm: float = 5.0
    bond_extension_height_mm: float = 8.0

    stem_width_mm: float = 7.6
    stem_height_mm: float = 6.0

    void_width_mm: float = 2.0

    def __post_init__(self) -> None:
        self._validate_positive_dimensions()
        self._validate_clearance()
        self._validate_link_geometry()
        self._validate_cutout_geometry()

    @property
    def total_pad_depth_mm(self) -> float:
        """Return the total flat-plus-semielliptical pad depth."""
        return self.flat_pad_height_mm + self.semiellipse_height_mm

    @property
    def cutout_width_mm(self) -> float:
        """Return the total internal cutout width."""
        return self.stem_width_mm + 2.0 * self.void_width_mm

    def _validate_positive_dimensions(self) -> None:
        for name in (
            "flat_pad_width_mm",
            "flat_pad_height_mm",
            "semiellipse_height_mm",
            "link_thickness_mm",
            "bond_extension_width_mm",
            "bond_extension_height_mm",
            "stem_width_mm",
            "stem_height_mm",
        ):
            require_positive(
                name,
                getattr(self, name),
            )

    def _validate_clearance(self) -> None:
        require_nonnegative("void_width_mm", self.void_width_mm)

    def _validate_link_geometry(self) -> None:
        if self.bond_extension_height_mm >= self.link_thickness_mm:
            raise ValueError(
                "bond_extension_height_mm must be smaller than "
                "link_thickness_mm"
            )

        required_width = (
            2.0 * self.bond_extension_width_mm
            + self.cutout_width_mm
        )

        if required_width >= self.flat_pad_width_mm:
            raise ValueError(
                "bond extensions and the internal cutout must leave a "
                "nonzero bonded region: "
                f"required_width={required_width:g} mm, "
                f"flat_pad_width_mm={self.flat_pad_width_mm:g} mm"
            )

    def _validate_cutout_geometry(self) -> None:
        half_pad_width = 0.5 * self.flat_pad_width_mm
        half_cutout_width = 0.5 * self.cutout_width_mm

        if half_cutout_width >= half_pad_width:
            raise ValueError(
                "the internal cutout must remain inside the pad width: "
                f"half_cutout_width={half_cutout_width:g} mm, "
                f"half_pad_width={half_pad_width:g} mm"
            )

        penetration_into_semiellipse = max(
            0.0,
            self.stem_height_mm - self.flat_pad_height_mm,
        )

        if penetration_into_semiellipse == 0.0:
            return

        available_depth = semiellipse_depth_at_x_mm(
            half_width_mm=half_pad_width,
            height_mm=self.semiellipse_height_mm,
            x_mm=half_cutout_width,
        )

        if penetration_into_semiellipse >= available_depth:
            raise ValueError(
                "the internal cutout exits the semielliptical pad envelope: "
                f"penetration={penetration_into_semiellipse:g} mm, "
                f"available_depth={available_depth:g} mm"
            )


def semiellipse_depth_at_x_mm(
    *,
    half_width_mm: float,
    height_mm: float,
    x_mm: float,
) -> float:
    """Return the semiellipse depth at horizontal position ``x_mm``.

    The lower semiellipse is defined by

        (x / a)^2 + (y / b)^2 = 1,

    where ``a`` is the half-width and ``b`` is the semiellipse height.
    """

    for name, value in (
        ("half_width_mm", half_width_mm),
        ("height_mm", height_mm),
        ("x_mm", x_mm),
    ):
        require_finite(name, value)

    require_positive("half_width_mm", half_width_mm)
    require_positive("height_mm", height_mm)

    if not 0.0 <= abs(x_mm) < half_width_mm:
        raise ValueError(
            "abs(x_mm) must be smaller than half_width_mm"
        )

    normalized_x = x_mm / half_width_mm

    return height_mm * sqrt(1.0 - normalized_x**2)


__all__ = [
    "FingertipGeometry",
    "semiellipse_depth_at_x_mm",
]
