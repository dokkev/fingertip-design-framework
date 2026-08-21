"""Parametric 2D cross-section of the LUMO fingertip."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import cos, pi, sin, sqrt

from .fingertip_param import FingertipParameters


Point2D = tuple[float, float]
LineSegment2D = tuple[Point2D, Point2D]


@dataclass(frozen=True)
class Fingertip:
    """Physical 2D cross-section defined by fingertip parameters.

    The fingertip owns the analytic cross-section geometry only. Meshing,
    extrusion, mechanics, and ray tracing are handled by downstream packages.

    Coordinates are expressed in millimeters. The cross-section is symmetric
    about x = 0, with the flat pad surface at y = 0 and the compliant pad
    extending in the negative y direction.
    """

    parameters: FingertipParameters = field(default_factory=FingertipParameters)

    @property
    def half_width_mm(self) -> float:
        return 0.5 * self.parameters.geometry.flat_pad_width_mm

    @property
    def cutout_half_width_mm(self) -> float:
        geometry = self.parameters.geometry
        return 0.5 * geometry.stem_width_mm + geometry.void_width_mm

    @property
    def ellipse_center_y_mm(self) -> float:
        """Return the y coordinate of the semiellipse center line."""
        return -self.parameters.geometry.flat_pad_height_mm

    # ------------------------------------------------------------------
    # Analytic compliant-pad boundary
    # ------------------------------------------------------------------

    @property
    def semiellipse_axes_mm(self) -> tuple[float, float]:
        """Return semiellipse axes (horizontal, vertical)."""
        return (
            self.half_width_mm,
            self.parameters.geometry.semiellipse_height_mm,
        )

    @property
    def semiellipse_endpoints(self) -> tuple[Point2D, Point2D]:
        """Return left and right endpoints of the lower semiellipse."""
        y = self.ellipse_center_y_mm
        return (
            (-self.half_width_mm, y),
            (self.half_width_mm, y),
        )

    @property
    def outer_left(self) -> LineSegment2D:
        """Return the straight left outer pad boundary."""
        geometry = self.parameters.geometry
        return (
            (
                -self.half_width_mm,
                geometry.bond_extension_height_mm,
            ),
            (
                -self.half_width_mm,
                self.ellipse_center_y_mm,
            ),
        )

    @property
    def outer_right(self) -> LineSegment2D:
        """Return the straight right outer pad boundary."""
        geometry = self.parameters.geometry
        return (
            (
                self.half_width_mm,
                self.ellipse_center_y_mm,
            ),
            (
                self.half_width_mm,
                geometry.bond_extension_height_mm,
            ),
        )

    # ------------------------------------------------------------------
    # Internal void boundary
    # ------------------------------------------------------------------

    @property
    def void_left(self) -> LineSegment2D:
        geometry = self.parameters.geometry
        cutout_bottom_y = -(
            geometry.stem_height_mm + geometry.void_height_mm
        )
        return (
            (-self.cutout_half_width_mm, 0.0),
            (-self.cutout_half_width_mm, cutout_bottom_y),
        )

    @property
    def void_right(self) -> LineSegment2D:
        geometry = self.parameters.geometry
        cutout_bottom_y = -(
            geometry.stem_height_mm + geometry.void_height_mm
        )
        return (
            (self.cutout_half_width_mm, cutout_bottom_y),
            (self.cutout_half_width_mm, 0.0),
        )

    @property
    def void_bottom(self) -> LineSegment2D:
        geometry = self.parameters.geometry
        cutout_bottom_y = -(
            geometry.stem_height_mm + geometry.void_height_mm
        )
        return (
            (-self.cutout_half_width_mm, cutout_bottom_y),
            (self.cutout_half_width_mm, cutout_bottom_y),
        )

    # ------------------------------------------------------------------
    # Rigid stem
    # ------------------------------------------------------------------

    @property
    def stem_left(self) -> LineSegment2D:
        geometry = self.parameters.geometry
        half_width = 0.5 * geometry.stem_width_mm
        stem_bottom_y = -geometry.stem_height_mm
        return (
            (-half_width, 0.0),
            (-half_width, stem_bottom_y),
        )

    @property
    def stem_right(self) -> LineSegment2D:
        geometry = self.parameters.geometry
        half_width = 0.5 * geometry.stem_width_mm
        stem_bottom_y = -geometry.stem_height_mm
        return (
            (half_width, stem_bottom_y),
            (half_width, 0.0),
        )

    @property
    def stem_bottom(self) -> LineSegment2D:
        geometry = self.parameters.geometry
        half_width = 0.5 * geometry.stem_width_mm
        stem_bottom_y = -geometry.stem_height_mm
        return (
            (-half_width, stem_bottom_y),
            (half_width, stem_bottom_y),
        )

    # ------------------------------------------------------------------
    # Bonded pad-link interface
    # ------------------------------------------------------------------

    @property
    def bond_left(self) -> tuple[Point2D, ...]:
        """Return the left compliant-to-rigid bonded boundary."""
        geometry = self.parameters.geometry
        inner_x = -self.half_width_mm + geometry.bond_extension_width_mm

        return (
            (-self.cutout_half_width_mm, 0.0),
            (inner_x, 0.0),
            (inner_x, geometry.bond_extension_height_mm),
            (-self.half_width_mm, geometry.bond_extension_height_mm),
        )

    @property
    def bond_right(self) -> tuple[Point2D, ...]:
        """Return the right compliant-to-rigid bonded boundary."""
        geometry = self.parameters.geometry
        inner_x = self.half_width_mm - geometry.bond_extension_width_mm

        return (
            (self.half_width_mm, geometry.bond_extension_height_mm),
            (inner_x, geometry.bond_extension_height_mm),
            (inner_x, 0.0),
            (self.cutout_half_width_mm, 0.0),
        )

    @property
    def minimum_silicone_thickness_mm(self) -> float:
        """Return the minimum distance between void and outer pad boundaries."""
        side_clearance = self.half_width_mm - self.cutout_half_width_mm
        horizontal_axis, vertical_axis = self.semiellipse_axes_mm
        ellipse_clearances = tuple(
            _minimum_ellipse_segment_distance_mm(
                horizontal_axis_mm=horizontal_axis,
                vertical_axis_mm=vertical_axis,
                center_y_mm=self.ellipse_center_y_mm,
                segment=segment,
            )
            for segment in (self.void_left, self.void_right, self.void_bottom)
        )
        return min(side_clearance, *ellipse_clearances)


def _closest_point_on_segment(
    point: Point2D,
    start: Point2D,
    end: Point2D,
) -> Point2D:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        return start

    parameter = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / length_squared
    parameter = min(1.0, max(0.0, parameter))
    return (start[0] + parameter * dx, start[1] + parameter * dy)


def _ellipse_segment_distance_squared(
    theta: float,
    *,
    horizontal_axis_mm: float,
    vertical_axis_mm: float,
    center_y_mm: float,
    segment: LineSegment2D,
) -> float:
    ellipse_point = (
        horizontal_axis_mm * cos(theta),
        center_y_mm - vertical_axis_mm * sin(theta),
    )
    segment_point = _closest_point_on_segment(
        ellipse_point,
        segment[0],
        segment[1],
    )
    return (
        (ellipse_point[0] - segment_point[0]) ** 2
        + (ellipse_point[1] - segment_point[1]) ** 2
    )


def _refine_bounded_minimum(
    function: Callable[[float], float],
    left: float,
    right: float,
) -> float:
    if right <= left:
        return function(left)

    ratio = (5.0**0.5 - 1.0) / 2.0
    first = right - ratio * (right - left)
    second = left + ratio * (right - left)
    first_value = function(first)
    second_value = function(second)

    for _ in range(64):
        if first_value <= second_value:
            right, second, second_value = second, first, first_value
            first = right - ratio * (right - left)
            first_value = function(first)
        else:
            left, first, first_value = first, second, second_value
            second = left + ratio * (right - left)
            second_value = function(second)

    return function(0.5 * (left + right))


def _minimum_ellipse_segment_distance_mm(
    *,
    horizontal_axis_mm: float,
    vertical_axis_mm: float,
    center_y_mm: float,
    segment: LineSegment2D,
) -> float:
    """Return the deterministic minimum distance to the lower semiellipse."""
    sample_count = 4097
    step = pi / (sample_count - 1)
    values = [
        _ellipse_segment_distance_squared(
            index * step,
            horizontal_axis_mm=horizontal_axis_mm,
            vertical_axis_mm=vertical_axis_mm,
            center_y_mm=center_y_mm,
            segment=segment,
        )
        for index in range(sample_count)
    ]

    candidate_indices = {0, sample_count - 1}
    candidate_indices.add(min(range(sample_count), key=values.__getitem__))
    candidate_indices.update(
        index
        for index in range(1, sample_count - 1)
        if values[index] <= values[index - 1]
        and values[index] <= values[index + 1]
    )

    best_squared_distance = float("inf")
    for index in sorted(candidate_indices):
        left = max(0.0, (index - 1) * step)
        right = min(pi, (index + 1) * step)
        best_squared_distance = min(
            best_squared_distance,
            _refine_bounded_minimum(
                lambda theta: _ellipse_segment_distance_squared(
                    theta,
                    horizontal_axis_mm=horizontal_axis_mm,
                    vertical_axis_mm=vertical_axis_mm,
                    center_y_mm=center_y_mm,
                    segment=segment,
                ),
                left,
                right,
            ),
        )

    return sqrt(best_squared_distance)


__all__ = ["Fingertip"]
