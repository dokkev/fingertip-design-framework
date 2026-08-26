"""Parametric 2D assembly of the LUMO fingertip."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import cos, isfinite, pi, sin, sqrt

from .bonding_interface import BondingInterface, Point2D
from .fingertip_param import FingertipParameters


LineSegment2D = tuple[Point2D, Point2D]


@dataclass(frozen=True)
class Silicone:
    """Constructed analytic 2D geometry of the compliant silicone region."""

    half_width_mm: float
    ellipse_center_z_mm: float
    ellipse_radius_x_mm: float
    ellipse_radius_z_mm: float
    cavity_left_x_mm: float
    cavity_right_x_mm: float
    cavity_bottom_z_mm: float
    bond_left_inner_x_mm: float
    bond_right_inner_x_mm: float
    bond_top_z_mm: float

    @property
    def semiellipse_axes_mm(self) -> tuple[float, float]:
        """Return the semiellipse horizontal and vertical radii."""
        return self.ellipse_radius_x_mm, self.ellipse_radius_z_mm

    @property
    def semiellipse_endpoints(self) -> tuple[Point2D, Point2D]:
        """Return the left and right endpoints of the lower semiellipse."""
        return (
            (-self.half_width_mm, self.ellipse_center_z_mm),
            (self.half_width_mm, self.ellipse_center_z_mm),
        )

    @property
    def outer_left(self) -> LineSegment2D:
        """Return the straight left outer silicone boundary."""
        return (
            (-self.half_width_mm, self.bond_top_z_mm),
            (-self.half_width_mm, self.ellipse_center_z_mm),
        )

    @property
    def outer_right(self) -> LineSegment2D:
        """Return the straight right outer silicone boundary."""
        return (
            (self.half_width_mm, self.ellipse_center_z_mm),
            (self.half_width_mm, self.bond_top_z_mm),
        )

    @property
    def void_left(self) -> LineSegment2D:
        """Return the left internal cavity boundary."""
        return (
            (self.cavity_left_x_mm, 0.0),
            (self.cavity_left_x_mm, self.cavity_bottom_z_mm),
        )

    @property
    def void_right(self) -> LineSegment2D:
        """Return the right internal cavity boundary."""
        return (
            (self.cavity_right_x_mm, self.cavity_bottom_z_mm),
            (self.cavity_right_x_mm, 0.0),
        )

    @property
    def void_bottom(self) -> LineSegment2D:
        """Return the bottom internal cavity boundary."""
        return (
            (self.cavity_left_x_mm, self.cavity_bottom_z_mm),
            (self.cavity_right_x_mm, self.cavity_bottom_z_mm),
        )

    @property
    def bond_extension_left(self) -> tuple[Point2D, ...]:
        """Return the left axis-aligned silicone bond extension."""
        return (
            (-self.half_width_mm, 0.0),
            (self.bond_left_inner_x_mm, 0.0),
            (self.bond_left_inner_x_mm, self.bond_top_z_mm),
            (-self.half_width_mm, self.bond_top_z_mm),
        )

    @property
    def bond_extension_right(self) -> tuple[Point2D, ...]:
        """Return the right axis-aligned silicone bond extension."""
        return (
            (self.bond_right_inner_x_mm, 0.0),
            (self.half_width_mm, 0.0),
            (self.half_width_mm, self.bond_top_z_mm),
            (self.bond_right_inner_x_mm, self.bond_top_z_mm),
        )

    @property
    def minimum_silicone_thickness_mm(self) -> float:
        """Return the minimum distance between cavity and outer boundaries."""
        side_clearance = min(
            self.cavity_left_x_mm + self.half_width_mm,
            self.half_width_mm - self.cavity_right_x_mm,
        )
        ellipse_clearances = tuple(
            _minimum_ellipse_segment_distance_mm(
                horizontal_axis_mm=self.ellipse_radius_x_mm,
                vertical_axis_mm=self.ellipse_radius_z_mm,
                center_z_mm=self.ellipse_center_z_mm,
                segment=segment,
            )
            for segment in (self.void_left, self.void_right, self.void_bottom)
        )
        return min(side_clearance, *ellipse_clearances)


@dataclass(frozen=True)
class Carrier:
    """Constructed analytic 2D geometry of the rigid fingertip carrier."""

    cross_section: tuple[Point2D, ...]

    def __post_init__(self) -> None:
        boundary = tuple(
            (float(x_mm), float(z_mm))
            for x_mm, z_mm in self.cross_section
        )
        if len(boundary) < 3:
            raise ValueError("carrier cross-section needs at least three points")
        if any(
            not isfinite(value)
            for point in boundary
            for value in point
        ):
            raise ValueError("carrier cross-section must contain finite points")
        object.__setattr__(self, "cross_section", boundary)


@dataclass(frozen=True)
class Fingertip:
    """Constructed analytic 2D fingertip assembly.

    ``parameters`` owns the physical inputs. ``silicone``, ``carrier``, and
    ``bonding_interface`` are derived analytic data consumed by downstream
    mesh and physics packages.
    """

    parameters: FingertipParameters = field(default_factory=FingertipParameters)
    silicone: Silicone = field(init=False)
    carrier: Carrier = field(init=False)
    bonding_interface: BondingInterface = field(init=False)

    @property
    def tip_z_m(self) -> float:
        """Return the reference silicone tip Z coordinate in metres."""
        return 1.0e-3 * (
            self.silicone.ellipse_center_z_mm
            - self.silicone.ellipse_radius_z_mm
        )

    @property
    def full_height_mm(self) -> float:
        """Return the complete carrier-to-silicone Z extent in millimetres."""
        carrier_z = tuple(z_mm for _, z_mm in self.carrier.cross_section)
        top_z_mm = max(self.silicone.bond_top_z_mm, *carrier_z)
        bottom_z_mm = min(
            self.silicone.ellipse_center_z_mm
            - self.silicone.ellipse_radius_z_mm,
            *carrier_z,
        )
        return top_z_mm - bottom_z_mm

    def __post_init__(self) -> None:
        geometry = self.parameters.geometry
        half_width = 0.5 * geometry.flat_pad_width_mm
        cavity_half_width = (
            0.5 * geometry.stem_width_mm + geometry.void_width_mm
        )
        ellipse_center_z = -geometry.flat_pad_height_mm
        cavity_bottom_z = -geometry.stem_height_mm
        bond_left_inner_x = (
            -half_width + geometry.bond_extension_width_mm
        )
        bond_right_inner_x = (
            half_width - geometry.bond_extension_width_mm
        )

        silicone = Silicone(
            half_width_mm=half_width,
            ellipse_center_z_mm=ellipse_center_z,
            ellipse_radius_x_mm=half_width,
            ellipse_radius_z_mm=geometry.semiellipse_height_mm,
            cavity_left_x_mm=-cavity_half_width,
            cavity_right_x_mm=cavity_half_width,
            cavity_bottom_z_mm=cavity_bottom_z,
            bond_left_inner_x_mm=bond_left_inner_x,
            bond_right_inner_x_mm=bond_right_inner_x,
            bond_top_z_mm=geometry.bond_extension_height_mm,
        )

        carrier = Carrier(
            cross_section=(
                (-half_width, geometry.bond_extension_height_mm),
                (bond_left_inner_x, geometry.bond_extension_height_mm),
                (bond_left_inner_x, 0.0),
                (-0.5 * geometry.stem_width_mm, 0.0),
                (
                    -0.5 * geometry.stem_width_mm,
                    -geometry.stem_height_mm,
                ),
                (
                    0.5 * geometry.stem_width_mm,
                    -geometry.stem_height_mm,
                ),
                (0.5 * geometry.stem_width_mm, 0.0),
                (bond_right_inner_x, 0.0),
                (bond_right_inner_x, geometry.bond_extension_height_mm),
                (half_width, geometry.bond_extension_height_mm),
                (half_width, geometry.link_thickness_mm),
                (-half_width, geometry.link_thickness_mm),
            )
        )

        bonding_interface = BondingInterface(
            left=(
                (silicone.cavity_left_x_mm, 0.0),
                (silicone.bond_left_inner_x_mm, 0.0),
                (
                    silicone.bond_left_inner_x_mm,
                    silicone.bond_top_z_mm,
                ),
                (-silicone.half_width_mm, silicone.bond_top_z_mm),
            ),
            right=(
                (silicone.half_width_mm, silicone.bond_top_z_mm),
                (
                    silicone.bond_right_inner_x_mm,
                    silicone.bond_top_z_mm,
                ),
                (silicone.bond_right_inner_x_mm, 0.0),
                (silicone.cavity_right_x_mm, 0.0),
            ),
        )
        object.__setattr__(self, "silicone", silicone)
        object.__setattr__(self, "carrier", carrier)
        object.__setattr__(self, "bonding_interface", bonding_interface)


def _closest_point_on_segment(
    point: Point2D,
    start: Point2D,
    end: Point2D,
) -> Point2D:
    dx = end[0] - start[0]
    dz = end[1] - start[1]
    length_squared = dx * dx + dz * dz
    if length_squared == 0.0:
        return start

    parameter = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dz
    ) / length_squared
    parameter = min(1.0, max(0.0, parameter))
    return (start[0] + parameter * dx, start[1] + parameter * dz)


def _ellipse_segment_distance_squared(
    theta: float,
    *,
    horizontal_axis_mm: float,
    vertical_axis_mm: float,
    center_z_mm: float,
    segment: LineSegment2D,
) -> float:
    ellipse_point = (
        horizontal_axis_mm * cos(theta),
        center_z_mm - vertical_axis_mm * sin(theta),
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
    center_z_mm: float,
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
            center_z_mm=center_z_mm,
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
                    center_z_mm=center_z_mm,
                    segment=segment,
                ),
                left,
                right,
            ),
        )

    return sqrt(best_squared_distance)


__all__ = ["Carrier", "Fingertip", "Silicone"]
