"""Deterministic production minimum silicone wall-thickness checks.

The production thickness is the minimum Euclidean distance between the
internal cutout boundary and the external compliant-pad envelope.  The full
pad-facing cutout rectangle is included, including the lower side shelves;
bonded pad/link interfaces and rigid stem boundaries are intentionally
excluded.
The semi-ellipse is evaluated from its analytic parameterization rather than
the sampled model arc, so the result is independent of ``arc_resolution`` and
of any mechanics or optical mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, pi, sin

from shapely.geometry import LineString
from shapely.ops import nearest_points

from model.fingertip_parameters import (
    FingertipParameters,
    InvalidFingertipParameters,
)

PRODUCTION_MINIMUM_SILICONE_THICKNESS_MM = 5.0


@dataclass(frozen=True)
class SiliconeThicknessMeasures:
    """Geometric wall-thickness diagnostics in millimetres."""

    side_ligament_mm: float
    diagonal_ellipse_ligament_mm: float
    minimum_silicone_thickness_mm: float
    shortest_boundary_pair: str = ""
    shortest_segment_start_mm: tuple[float, float] = (0.0, 0.0)
    shortest_segment_end_mm: tuple[float, float] = (0.0, 0.0)


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


def _closest_point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator == 0.0:
        return start
    parameter = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / denominator
    parameter = min(1.0, max(0.0, parameter))
    return (start[0] + parameter * dx, start[1] + parameter * dy)


def _refine_bounded_minimum(function, left: float, right: float) -> tuple[float, float]:
    """Refine one deterministic bracket with golden-section minimization."""
    if right <= left:
        return function(left), left
    ratio = (5.0**0.5 - 1.0) / 2.0
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc = function(c)
    fd = function(d)
    for _ in range(64):
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = function(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = function(d)
    theta = 0.5 * (left + right)
    return function(theta), theta


def _ellipse_to_segment_minimum(
    parameters: FingertipParameters,
    segment_start: tuple[float, float],
    segment_end: tuple[float, float],
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """Find the global bounded minimum from an analytic ellipse to a segment."""
    a = 0.5 * parameters.flat_pad_width
    b = parameters.semielliptical_pad_height
    h_fp = parameters.flat_pad_height

    def points(theta: float) -> tuple[tuple[float, float], tuple[float, float]]:
        ellipse_point = (a * cos(theta), -h_fp - b * sin(theta))
        segment_point = _closest_point_on_segment(
            ellipse_point, segment_start, segment_end
        )
        return ellipse_point, segment_point

    def squared(theta: float) -> float:
        left, right = points(theta)
        return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2

    # The geometry is smooth and low-dimensional.  A fixed dense partition
    # brackets every observed local basin; refinement is not tied to the
    # polygonal arc used later for meshing.
    sample_count = 4097
    step = pi / (sample_count - 1)
    values = [squared(index * step) for index in range(sample_count)]
    candidates = {0, sample_count - 1, min(range(sample_count), key=values.__getitem__)}
    for index in range(1, sample_count - 1):
        if values[index] <= values[index - 1] and values[index] <= values[index + 1]:
            candidates.add(index)
    best_theta = 0.0
    best_value = float("inf")
    for index in sorted(candidates):
        left = max(0.0, (index - 1) * step)
        right = min(pi, (index + 1) * step)
        value, theta = _refine_bounded_minimum(squared, left, right)
        if value < best_value:
            best_value = value
            best_theta = theta
    ellipse_point, segment_point = points(best_theta)
    return hypot(
        ellipse_point[0] - segment_point[0],
        ellipse_point[1] - segment_point[1],
    ), segment_point, ellipse_point


def _straight_segment_minimum(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    first = LineString([first_start, first_end])
    second = LineString([second_start, second_end])
    first_point, second_point = nearest_points(first, second)
    return (
        float(first.distance(second)),
        (float(first_point.x), float(first_point.y)),
        (float(second_point.x), float(second_point.y)),
    )


def silicone_thickness_measures(
    parameters: FingertipParameters,
) -> SiliconeThicknessMeasures:
    """Return legacy diagnostics plus the true relevant-boundary minimum."""

    if not isinstance(parameters, FingertipParameters):
        raise TypeError("parameters must be FingertipParameters")
    side = 0.5 * parameters.flat_pad_width - parameters.cutout_half_width
    diagonal = _corner_to_semiellipse_distance_mm(parameters)
    half_width = 0.5 * parameters.flat_pad_width
    cutout = parameters.cutout_half_width
    cutout_bottom = -parameters.cutout_height
    outer_bottom = -parameters.flat_pad_height
    internal_segments = (
        ("pad_cutout_left", (-cutout, 0.0), (-cutout, cutout_bottom)),
        ("pad_cutout_right", (cutout, 0.0), (cutout, cutout_bottom)),
        (
            "pad_cutout_bottom",
            (-cutout, cutout_bottom),
            (cutout, cutout_bottom),
        ),
    )
    external_segments = (
        ("pad_outer_left", (-half_width, parameters.bond_extension_height), (-half_width, outer_bottom)),
        ("pad_outer_right", (half_width, outer_bottom), (half_width, parameters.bond_extension_height)),
    )
    best_distance = float("inf")
    best_pair = ""
    best_start = (0.0, 0.0)
    best_end = (0.0, 0.0)

    def consider(
        pair: str,
        result: tuple[float, tuple[float, float], tuple[float, float]],
    ) -> None:
        nonlocal best_distance, best_pair, best_start, best_end
        distance, start, end = result
        if distance < best_distance:
            best_distance = distance
            best_pair = pair
            best_start = start
            best_end = end

    for internal_name, internal_start, internal_end in internal_segments:
        for external_name, external_start, external_end in external_segments:
            consider(
                f"{internal_name}__{external_name}",
                _straight_segment_minimum(
                    internal_start,
                    internal_end,
                    external_start,
                    external_end,
                ),
            )
        ellipse_distance, internal_point, ellipse_point = _ellipse_to_segment_minimum(
            parameters,
            internal_start,
            internal_end,
        )
        consider(
            f"{internal_name}__pad_outer_arc",
            (ellipse_distance, internal_point, ellipse_point),
        )
    if abs(best_distance - diagonal) <= 1.0e-12:
        best_distance = diagonal
    return SiliconeThicknessMeasures(
        side_ligament_mm=float(side),
        diagonal_ellipse_ligament_mm=float(diagonal),
        minimum_silicone_thickness_mm=float(best_distance),
        shortest_boundary_pair=best_pair,
        shortest_segment_start_mm=best_start,
        shortest_segment_end_mm=best_end,
    )


def validate_minimum_silicone_thickness(
    parameters: FingertipParameters,
    *,
    minimum_mm: float = PRODUCTION_MINIMUM_SILICONE_THICKNESS_MM,
) -> SiliconeThicknessMeasures:
    """Reject morphologies below the production silicone-wall design margin."""

    measures = silicone_thickness_measures(parameters)
    if measures.minimum_silicone_thickness_mm < float(minimum_mm):
        raise InvalidFingertipParameters(
            "minimum silicone thickness must be at least "
            f"{float(minimum_mm):g} mm: side={measures.side_ligament_mm:g} mm, "
            f"global_d_min={measures.minimum_silicone_thickness_mm:g} mm, "
            f"boundary_pair={measures.shortest_boundary_pair}"
        )
    return measures


__all__ = [
    "SiliconeThicknessMeasures",
    "PRODUCTION_MINIMUM_SILICONE_THICKNESS_MM",
    "silicone_thickness_measures",
    "validate_minimum_silicone_thickness",
]
