"""Solver-independent fingertip geometry and wall-thickness diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, pi, sin
from typing import Literal

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points

from finger.fingertip_parameters import (
    FingertipParameters,
    InvalidFingertipParameters,
    PRODUCTION_MINIMUM_SILICONE_THICKNESS_MM,
)
from util.validation import require_nonnegative

VoidClassification = Literal[
    "zero_clearance_fit", "side_clearance", "bottom_clearance", "u_clearance"
]
InterfaceType = Literal["bonded"]
PolygonalGeometry = Polygon | MultiPolygon


class InvalidFingertipGeometry(ValueError):
    """Raised when constructed material fails a geometric validity check."""


@dataclass(frozen=True)
class BoundarySegment:
    """Named analytic boundary segment for later mesh tagging."""

    name: str
    geometry: LineString


@dataclass(frozen=True)
class ContactPair:
    """Potential stem-pad contact surfaces and their initial normal gap."""

    name: str
    stem_boundary: BoundarySegment
    pad_boundary: BoundarySegment
    initial_normal_gap: float


@dataclass(frozen=True)
class FingertipBoundaries:
    """Complete boundary metadata needed by a future mesh/contact adapter."""

    pad_bond_left: BoundarySegment
    pad_bond_right: BoundarySegment
    pad_outer_left: BoundarySegment
    pad_outer_right: BoundarySegment
    pad_cutout_left: BoundarySegment
    pad_cutout_right: BoundarySegment
    pad_cutout_bottom: BoundarySegment
    stem_left: BoundarySegment
    stem_right: BoundarySegment
    stem_bottom: BoundarySegment
    pad_outer_arc: BoundarySegment
    contact_pairs: tuple[ContactPair, ContactPair, ContactPair]

    @property
    def segments(self) -> dict[str, BoundarySegment]:
        """Return all named segments keyed by their stable boundary tag."""
        boundary_segments = (
            self.pad_bond_left,
            self.pad_bond_right,
            self.pad_outer_left,
            self.pad_outer_right,
            self.pad_cutout_left,
            self.pad_cutout_right,
            self.pad_cutout_bottom,
            self.stem_left,
            self.stem_right,
            self.stem_bottom,
            self.pad_outer_arc,
        )
        return {segment.name: segment for segment in boundary_segments}


@dataclass(frozen=True)
class InterfaceDefinition:
    """Metadata for the always-bonded upper link-pad interface."""

    name: str
    geometry: MultiLineString
    interface_type: InterfaceType


class FingertipModel:
    """Build the symmetric compliant pad, rigid link/stem, and clearance.

    The outer pad is a flat rectangle from ``y = 0`` to
    ``-flat_pad_height`` joined directly to a lower semi-ellipse. Two
    compliant extensions occupy lower-corner recesses in a full-width rigid
    plate. The centered stem extends downward into the completed envelope.
    """

    def __init__(self, parameters: FingertipParameters):
        self.parameters = parameters
        self.flat_pad_geometry = self._build_flat_pad()
        self.left_bond_extension_geometry = self._build_bond_extension("left")
        self.right_bond_extension_geometry = self._build_bond_extension("right")
        self._pad_outer_arc_geometry = self._build_pad_outer_arc()
        self.semielliptical_pad_geometry = self._build_semielliptical_pad()
        self.outer_pad_geometry = self._build_outer_pad()
        self.stem_geometry = self._build_stem()
        self.cutout_geometry = self._build_cutout()
        self._validate_internal_geometry()
        self.link_plate_geometry = self._build_link_plate()
        self.link_geometry = self._build_rigid_link()
        self.pad_material_geometry = self._validated_polygonal_geometry(
            self.outer_pad_geometry.difference(self.cutout_geometry),
            "compliant pad material",
        )
        self.void_geometry = self._build_void_geometry()
        self.raw_material_geometry = self._validated_polygonal_geometry(
            self.outer_pad_geometry.union(self.link_geometry),
            "raw assembly material",
        )
        self.material_geometry = self._validated_polygonal_geometry(
            self.pad_material_geometry.union(self.link_geometry),
            "assembly material",
        )
        self.boundaries = self._build_boundaries()
        self.pad_link_interface = MultiLineString(
            [
                list(self.boundaries.pad_bond_left.geometry.coords),
                list(self.boundaries.pad_bond_right.geometry.coords),
            ]
        )
        self.symmetry_axis = LineString(
            [
                (0.0, -(parameters.flat_pad_height + parameters.semielliptical_pad_height)),
                (0.0, parameters.link_thickness),
            ]
        )
        self.interface_definition = InterfaceDefinition(
            name="pad_link_interface",
            geometry=self.pad_link_interface,
            interface_type="bonded",
        )
        self.validate_geometry()

    @property
    def contact_pairs(self) -> tuple[ContactPair, ContactPair, ContactPair]:
        """Return left, right, and bottom stem-pad potential contact pairs."""
        return self.boundaries.contact_pairs

    def classify_void(self) -> VoidClassification:
        """Describe which independent stem-clearance dimensions are nonzero."""
        if self.parameters.void_width == 0.0 and self.parameters.void_height == 0.0:
            return "zero_clearance_fit"
        if self.parameters.void_width > 0.0 and self.parameters.void_height == 0.0:
            return "side_clearance"
        if self.parameters.void_width == 0.0 and self.parameters.void_height > 0.0:
            return "bottom_clearance"
        return "u_clearance"

    def pad_link_connection_length(self) -> float:
        """Return the total link-pad interface length outside the cutout."""
        return float(self.pad_link_interface.length)

    def is_material_connected(self) -> bool:
        """Return whether the complete rigid/compliant assembly is connected."""
        if self.material_geometry.is_empty:
            return False
        component_count = (
            len(self.material_geometry.geoms)
            if isinstance(self.material_geometry, MultiPolygon)
            else 1
        )
        return (
            component_count == 1
            and self.pad_link_connection_length()
            > self.parameters.geometry_length_tolerance_mm
        )

    def validate_geometry(self) -> None:
        """Raise if a material domain is invalid or the bonded interface vanishes."""
        if self.outer_pad_geometry.is_empty:
            raise InvalidFingertipGeometry("outer pad geometry is empty")
        if not self.outer_pad_geometry.is_valid:
            raise InvalidFingertipGeometry("outer pad geometry is invalid")
        if self.pad_material_geometry.is_empty:
            raise InvalidFingertipGeometry("compliant pad geometry is empty")
        if not self.pad_material_geometry.is_valid:
            raise InvalidFingertipGeometry("compliant pad geometry is invalid")
        if self.link_geometry.is_empty:
            raise InvalidFingertipGeometry("rigid link geometry is empty")
        if not self.link_geometry.is_valid:
            raise InvalidFingertipGeometry("rigid link geometry is invalid")
        if self.material_geometry.is_empty:
            raise InvalidFingertipGeometry("assembly geometry is empty")
        if not self.material_geometry.is_valid:
            raise InvalidFingertipGeometry("assembly geometry is invalid")
        if isinstance(self.pad_material_geometry, MultiPolygon):
            raise InvalidFingertipGeometry(
                "the cutout creates disconnected compliant-pad fragments"
            )
        overlap_area = self.pad_material_geometry.intersection(
            self.link_geometry
        ).area
        if overlap_area > self.parameters.geometry_area_tolerance_mm2:
            raise InvalidFingertipGeometry(
                "compliant pad and rigid link materials overlap: "
                f"overlap_area={overlap_area:g}"
            )
        if (
            self.pad_link_connection_length()
            <= self.parameters.geometry_length_tolerance_mm
        ):
            raise InvalidFingertipGeometry(
                "the always-bonded upper interface has zero effective length"
            )

    def _build_flat_pad(self) -> Polygon:
        parameters = self.parameters
        return orient(
            box(
                -parameters.flat_pad_width / 2.0,
                -parameters.flat_pad_height,
                parameters.flat_pad_width / 2.0,
                0.0,
            ),
            sign=1.0,
        )

    def _build_bond_extension(self, side: Literal["left", "right"]) -> Polygon:
        parameters = self.parameters
        half_flat_width = parameters.flat_pad_width / 2.0
        if side == "left":
            bounds = (
                -half_flat_width,
                0.0,
                -half_flat_width + parameters.bond_extension_width,
                parameters.bond_extension_height,
            )
        else:
            bounds = (
                half_flat_width - parameters.bond_extension_width,
                0.0,
                half_flat_width,
                parameters.bond_extension_height,
            )
        return orient(box(*bounds), sign=1.0)

    def _build_pad_outer_arc(self) -> LineString:
        # Snap the sampled arc to the flat-pad endpoints so there is no
        # numerical shoulder at the rectangle/ellipse join.
        half_width = self.parameters.flat_pad_width / 2.0
        semi_axis = self.parameters.semielliptical_pad_height
        ellipse_start_y = -self.parameters.flat_pad_height
        arc_segments = (
            self.parameters.arc_resolution
            if self.parameters.arc_resolution % 2 == 0
            else self.parameters.arc_resolution + 1
        )
        angles = np.linspace(0.0, np.pi, arc_segments + 1)
        coordinates = [
            (
                float(half_width * np.cos(angle)),
                float(ellipse_start_y - semi_axis * np.sin(angle)),
            )
            for angle in angles
        ]
        coordinates[0] = (half_width, ellipse_start_y)
        coordinates[-1] = (-half_width, ellipse_start_y)
        return LineString(coordinates)

    def _build_semielliptical_pad(self) -> Polygon:
        semiellipse = orient(Polygon(self._pad_outer_arc_geometry.coords), sign=1.0)
        if semiellipse.is_empty or not semiellipse.is_valid:
            raise InvalidFingertipGeometry(
                "semi-elliptical pad construction failed"
            )
        return semiellipse

    def _build_outer_pad(self) -> Polygon:
        outer_pad = self._validated_polygonal_geometry(
            self.flat_pad_geometry.union(
                self.semielliptical_pad_geometry
            ).union(self.left_bond_extension_geometry).union(
                self.right_bond_extension_geometry
            ),
            "outer pad",
        )
        if not isinstance(outer_pad, Polygon):
            raise InvalidFingertipGeometry(
                "flat pad, bond extensions, and semi-elliptical pad do not "
                "form one envelope"
            )
        return outer_pad

    def _build_link_plate(self) -> Polygon:
        parameters = self.parameters
        half_width = parameters.flat_pad_width / 2.0
        full_link_plate = box(
            -half_width,
            0.0,
            half_width,
            parameters.link_thickness,
        )
        left_recess = box(
            -half_width,
            0.0,
            -half_width + parameters.bond_extension_width,
            parameters.bond_extension_height,
        )
        right_recess = box(
            half_width - parameters.bond_extension_width,
            0.0,
            half_width,
            parameters.bond_extension_height,
        )
        link_plate = self._validated_polygonal_geometry(
            full_link_plate.difference(left_recess.union(right_recess)),
            "rigid link plate",
        )
        if not isinstance(link_plate, Polygon):
            raise InvalidFingertipGeometry(
                "rigid link plate recesses do not form one polygon"
            )
        return orient(link_plate, sign=1.0)

    def _build_stem(self) -> Polygon:
        parameters = self.parameters
        return orient(
            box(
                -parameters.stem_width / 2.0,
                -parameters.stem_height,
                parameters.stem_width / 2.0,
                0.0,
            ),
            sign=1.0,
        )

    def _build_rigid_link(self) -> Polygon:
        rigid_link = self._validated_polygonal_geometry(
            self.link_plate_geometry.union(self.stem_geometry), "rigid link"
        )
        if not isinstance(rigid_link, Polygon):
            raise InvalidFingertipGeometry(
                "rigid link construction is not a single polygon"
            )
        return rigid_link

    def _build_cutout(self) -> Polygon:
        parameters = self.parameters
        cutout_half_width = (
            0.5 * parameters.stem_width + parameters.void_width
        )
        cutout_bottom_y = -(parameters.stem_height + parameters.void_height)
        return box(
            -cutout_half_width,
            cutout_bottom_y,
            cutout_half_width,
            0.0,
        )

    def _validate_internal_geometry(self) -> None:
        tolerance = self.parameters.geometry_length_tolerance_mm
        if not self.outer_pad_geometry.buffer(tolerance).covers(
            self.cutout_geometry
        ):
            cutout_half_width = (
                0.5 * self.parameters.stem_width + self.parameters.void_width
            )
            cutout_bottom_y = -(
                self.parameters.stem_height + self.parameters.void_height
            )
            ellipse_start_y = -self.parameters.flat_pad_height
            pad_tip_y = -(
                self.parameters.flat_pad_height
                + self.parameters.semielliptical_pad_height
            )
            raise InvalidFingertipGeometry(
                "the internal cutout exits the completed outer pad envelope: "
                f"cutout_half_width={cutout_half_width:g}, "
                f"cutout_bottom_y={cutout_bottom_y:g}, "
                f"ellipse_start_y={ellipse_start_y:g}, "
                f"pad_tip_y={pad_tip_y:g}"
            )
        if not self.cutout_geometry.buffer(tolerance).covers(self.stem_geometry):
            raise InvalidFingertipGeometry(
                "the rigid stem is not fully contained by the internal cutout"
            )

    def _build_void_geometry(self) -> PolygonalGeometry | None:
        clearance = self.cutout_geometry.difference(self.stem_geometry)
        if clearance.is_empty:
            return None
        return self._validated_polygonal_geometry(clearance, "void")

    def _build_boundaries(self) -> FingertipBoundaries:
        parameters = self.parameters
        flat_pad_edge = parameters.flat_pad_width / 2.0
        recess_inner_left = -flat_pad_edge + parameters.bond_extension_width
        recess_inner_right = flat_pad_edge - parameters.bond_extension_width
        cutout_edge = 0.5 * parameters.stem_width + parameters.void_width
        stem_edge = parameters.stem_width / 2.0
        ellipse_start_y = -parameters.flat_pad_height
        stem_bottom_y = -parameters.stem_height
        cutout_bottom_y = -(parameters.stem_height + parameters.void_height)
        bond_height = parameters.bond_extension_height

        pad_bond_left = BoundarySegment(
            "pad_bond_left",
            LineString(
                [
                    (-cutout_edge, 0.0),
                    (recess_inner_left, 0.0),
                    (recess_inner_left, bond_height),
                    (-flat_pad_edge, bond_height),
                ]
            ),
        )
        pad_bond_right = BoundarySegment(
            "pad_bond_right",
            LineString(
                [
                    (flat_pad_edge, bond_height),
                    (recess_inner_right, bond_height),
                    (recess_inner_right, 0.0),
                    (cutout_edge, 0.0),
                ]
            ),
        )
        pad_outer_left = BoundarySegment(
            "pad_outer_left",
            LineString(
                [
                    (-flat_pad_edge, bond_height),
                    (-flat_pad_edge, ellipse_start_y),
                ]
            ),
        )
        pad_outer_right = BoundarySegment(
            "pad_outer_right",
            LineString(
                [
                    (flat_pad_edge, ellipse_start_y),
                    (flat_pad_edge, bond_height),
                ]
            ),
        )
        pad_cutout_left = BoundarySegment(
            "pad_cutout_left",
            LineString([(-cutout_edge, 0.0), (-cutout_edge, stem_bottom_y)]),
        )
        pad_cutout_right = BoundarySegment(
            "pad_cutout_right",
            LineString([(cutout_edge, 0.0), (cutout_edge, stem_bottom_y)]),
        )
        pad_cutout_bottom = BoundarySegment(
            "pad_cutout_bottom",
            LineString([(-stem_edge, cutout_bottom_y), (stem_edge, cutout_bottom_y)]),
        )
        stem_left = BoundarySegment(
            "stem_left", LineString([(-stem_edge, 0.0), (-stem_edge, stem_bottom_y)])
        )
        stem_right = BoundarySegment(
            "stem_right", LineString([(stem_edge, 0.0), (stem_edge, stem_bottom_y)])
        )
        stem_bottom = BoundarySegment(
            "stem_bottom",
            LineString([(-stem_edge, stem_bottom_y), (stem_edge, stem_bottom_y)]),
        )
        pad_outer_arc = BoundarySegment(
            "pad_outer_arc", self._pad_outer_arc_geometry
        )

        contact_pairs = (
            ContactPair(
                "left_contact",
                stem_boundary=stem_left,
                pad_boundary=pad_cutout_left,
                initial_normal_gap=parameters.void_width,
            ),
            ContactPair(
                "right_contact",
                stem_boundary=stem_right,
                pad_boundary=pad_cutout_right,
                initial_normal_gap=parameters.void_width,
            ),
            ContactPair(
                "bottom_contact",
                stem_boundary=stem_bottom,
                pad_boundary=pad_cutout_bottom,
                initial_normal_gap=parameters.void_height,
            ),
        )
        return FingertipBoundaries(
            pad_bond_left=pad_bond_left,
            pad_bond_right=pad_bond_right,
            pad_outer_left=pad_outer_left,
            pad_outer_right=pad_outer_right,
            pad_cutout_left=pad_cutout_left,
            pad_cutout_right=pad_cutout_right,
            pad_cutout_bottom=pad_cutout_bottom,
            stem_left=stem_left,
            stem_right=stem_right,
            stem_bottom=stem_bottom,
            pad_outer_arc=pad_outer_arc,
            contact_pairs=contact_pairs,
        )

    @staticmethod
    def _validated_polygonal_geometry(
        geometry: BaseGeometry, name: str
    ) -> PolygonalGeometry:
        if geometry.is_empty:
            raise InvalidFingertipGeometry(f"{name} geometry is empty")

        candidate = geometry
        if not candidate.is_valid:
            candidate = candidate.buffer(0)
        if candidate.is_empty or not candidate.is_valid:
            raise InvalidFingertipGeometry(f"{name} geometry is invalid")
        if not isinstance(candidate, (Polygon, MultiPolygon)):
            raise InvalidFingertipGeometry(f"{name} geometry is not polygonal")

        if isinstance(candidate, Polygon):
            return orient(candidate, sign=1.0)
        return MultiPolygon([orient(polygon, sign=1.0) for polygon in candidate.geoms])


@dataclass(frozen=True)
class SiliconeThicknessMeasures:
    """Geometric wall-thickness diagnostics in millimetres."""

    side_ligament_mm: float
    diagonal_ellipse_ligament_mm: float
    minimum_silicone_thickness_mm: float
    shortest_boundary_pair: str = ""
    shortest_segment_start_mm: tuple[float, float] = (0.0, 0.0)
    shortest_segment_end_mm: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class _ThicknessBoundarySegment:
    """Named boundary segment used by the thickness calculation."""

    name: str
    start: tuple[float, float]
    end: tuple[float, float]


def _corner_to_semiellipse_distance_mm(parameters: FingertipParameters) -> float:
    """Return the minimum distance from a void corner to the lower ellipse."""
    a = 0.5 * parameters.flat_pad_width
    b = parameters.semielliptical_pad_height
    px = 0.5 * parameters.stem_width + parameters.void_width
    py = -(parameters.stem_height + parameters.void_height)
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
    """Find the global bounded minimum from the ellipse to a segment."""
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

    sample_count = 4097
    step = pi / (sample_count - 1)
    values = [squared(index * step) for index in range(sample_count)]
    candidates = {
        0,
        sample_count - 1,
        min(range(sample_count), key=values.__getitem__),
    }
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
    """Return the minimum distance and witness points for two segments."""
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
    """Return relevant-boundary thickness diagnostics and global d_min."""
    if not isinstance(parameters, FingertipParameters):
        raise TypeError("parameters must be FingertipParameters")
    cutout_half_width = 0.5 * parameters.stem_width + parameters.void_width
    cutout_height = parameters.stem_height + parameters.void_height
    side = 0.5 * parameters.flat_pad_width - cutout_half_width
    diagonal = _corner_to_semiellipse_distance_mm(parameters)
    half_width = 0.5 * parameters.flat_pad_width
    cutout = cutout_half_width
    cutout_bottom = -cutout_height
    outer_bottom = -parameters.flat_pad_height
    internal_segments = (
        _ThicknessBoundarySegment(
            "pad_cutout_left", (-cutout, 0.0), (-cutout, cutout_bottom)
        ),
        _ThicknessBoundarySegment(
            "pad_cutout_right", (cutout, 0.0), (cutout, cutout_bottom)
        ),
        _ThicknessBoundarySegment(
            "pad_cutout_bottom",
            (-cutout, cutout_bottom),
            (cutout, cutout_bottom),
        ),
    )
    external_segments = (
        _ThicknessBoundarySegment(
            "pad_outer_left",
            (-half_width, parameters.bond_extension_height),
            (-half_width, outer_bottom),
        ),
        _ThicknessBoundarySegment(
            "pad_outer_right",
            (half_width, outer_bottom),
            (half_width, parameters.bond_extension_height),
        ),
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

    for internal in internal_segments:
        for external in external_segments:
            consider(
                f"{internal.name}__{external.name}",
                _straight_segment_minimum(
                    internal.start,
                    internal.end,
                    external.start,
                    external.end,
                ),
            )
        ellipse_distance, internal_point, ellipse_point = _ellipse_to_segment_minimum(
            parameters,
            internal.start,
            internal.end,
        )
        consider(
            f"{internal.name}__pad_outer_arc",
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
    minimum = float(minimum_mm)
    require_nonnegative("minimum_mm", minimum)
    measures = silicone_thickness_measures(parameters)
    if measures.minimum_silicone_thickness_mm < minimum:
        raise InvalidFingertipParameters(
            "minimum silicone thickness must be at least "
            f"{minimum:g} mm: side={measures.side_ligament_mm:g} mm, "
            f"global_d_min={measures.minimum_silicone_thickness_mm:g} mm, "
            f"boundary_pair={measures.shortest_boundary_pair}"
        )
    return measures


__all__ = [
    "BoundarySegment",
    "ContactPair",
    "FingertipBoundaries",
    "FingertipModel",
    "InterfaceDefinition",
    "InvalidFingertipGeometry",
    "PolygonalGeometry",
    "SiliconeThicknessMeasures",
    "silicone_thickness_measures",
    "validate_minimum_silicone_thickness",
]
