"""Exact-enough geometry checks for minimum silicone wall thickness."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, pi, sin

from model.fingertip_parameters import (
    FingertipParameters,
    InvalidFingertipParameters,
    MINIMUM_SILICONE_LIGAMENT_MM,
)


@dataclass(frozen=True)
class SiliconeThicknessMeasures:
    """Geometric wall-thickness diagnostics in millimetres."""

    side_ligament_mm: float
    diagonal_ellipse_ligament_mm: float
    minimum_silicone_thickness_mm: float


def _corner_to_semiellipse_distance_mm(parameters: FingertipParameters) -> float:
    """Return minimum Euclidean distance from the void corner to the lower semiellipse.

    The lower outer envelope is parameterized on the right half as
    ``x=a*cos(theta)``, ``y=-h_fp-h_ep*sin(theta)``, theta in [0, pi/2].
    Symmetry makes the right internal corner authoritative.  A deterministic
    coarse global scan is followed by golden-section refinement inside the
    winning interval, so the production result does not depend on arc_resolution.
    """

    a = 0.5 * parameters.flat_pad_width
    b = parameters.semielliptical_pad_height
    px = parameters.cutout_half_width
    py = -parameters.cutout_height
    h_fp = parameters.flat_pad_height

    def squared(theta: float) -> float:
        x = a * cos(theta)
        y = -h_fp - b * sin(theta)
        return (x - px) ** 2 + (y - py) ** 2

    sample_count = 257
    step = 0.5 * pi / (sample_count - 1)
    values = [squared(index * step) for index in range(sample_count)]
    best = min(range(sample_count), key=values.__getitem__)
    left = max(0.0, (best - 1) * step)
    right = min(0.5 * pi, (best + 1) * step)

    if right > left:
        ratio = (5.0**0.5 - 1.0) / 2.0
        c = right - ratio * (right - left)
        d = left + ratio * (right - left)
        fc = squared(c)
        fd = squared(d)
        for _ in range(64):
            if fc <= fd:
                right, d, fd = d, c, fc
                c = right - ratio * (right - left)
                fc = squared(c)
            else:
                left, c, fc = c, d, fd
                d = left + ratio * (right - left)
                fd = squared(d)
        refined = squared(0.5 * (left + right))
    else:
        refined = values[best]

    return refined**0.5


def silicone_thickness_measures(
    parameters: FingertipParameters,
) -> SiliconeThicknessMeasures:
    """Return the minimum side/diagonal wall thickness for the current morphology."""

    if not isinstance(parameters, FingertipParameters):
        raise TypeError("parameters must be FingertipParameters")
    side = 0.5 * parameters.flat_pad_width - parameters.cutout_half_width
    diagonal = _corner_to_semiellipse_distance_mm(parameters)
    return SiliconeThicknessMeasures(
        side_ligament_mm=float(side),
        diagonal_ellipse_ligament_mm=float(diagonal),
        minimum_silicone_thickness_mm=float(min(side, diagonal)),
    )


def validate_minimum_silicone_thickness(
    parameters: FingertipParameters,
    *,
    minimum_mm: float = MINIMUM_SILICONE_LIGAMENT_MM,
) -> SiliconeThicknessMeasures:
    """Reject morphologies thinner than the production silicone-wall constraint."""

    measures = silicone_thickness_measures(parameters)
    if measures.minimum_silicone_thickness_mm < float(minimum_mm):
        raise InvalidFingertipParameters(
            "minimum silicone thickness must be at least "
            f"{float(minimum_mm):g} mm: side={measures.side_ligament_mm:g} mm, "
            "diagonal_ellipse="
            f"{measures.diagonal_ellipse_ligament_mm:g} mm"
        )
    return measures


__all__ = [
    "SiliconeThicknessMeasures",
    "silicone_thickness_measures",
    "validate_minimum_silicone_thickness",
]
