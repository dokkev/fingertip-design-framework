"""Solver-independent construction of the parameterized LIT pad geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

from model.fingertip_parameters import FingertipParameters

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

    The compliant pad is the lower half of an ellipse whose top diameter lies
    on ``y = 0``. A rigid plate sits above that line and its centered stem
    extends downward into the pad. The rectangular cutout around the stem has
    width ``stem_width + 2 * void_width`` and depth
    ``stem_height + void_height``.
    """

    def __init__(self, parameters: FingertipParameters):
        parameters.validate()
        self._parameters = parameters
        self._pad_outer_arc_geometry = self._build_pad_outer_arc()
        self._outer_pad_geometry = self._build_outer_pad()
        self._link_plate_geometry = self._build_link_plate()
        self._stem_geometry = self._build_stem()
        self._link_geometry = self._build_rigid_link()
        self._cutout_geometry = self._build_cutout()
        self._pad_material_geometry = self._validated_polygonal_geometry(
            self._outer_pad_geometry.difference(self._cutout_geometry),
            "compliant pad material",
        )
        self._void_geometry = self._build_void_geometry()
        self._raw_material_geometry = self._validated_polygonal_geometry(
            self._outer_pad_geometry.union(self._link_geometry),
            "raw assembly material",
        )
        self._material_geometry = self._validated_polygonal_geometry(
            self._pad_material_geometry.union(self._link_geometry),
            "assembly material",
        )
        self._boundaries = self._build_boundaries()
        self._pad_link_interface = MultiLineString(
            [
                list(self._boundaries.pad_bond_left.geometry.coords),
                list(self._boundaries.pad_bond_right.geometry.coords),
            ]
        )
        self._symmetry_axis = LineString(
            [(0.0, -parameters.pad_height), (0.0, parameters.link_thickness)]
        )
        self._interface_definition = InterfaceDefinition(
            name="pad_link_interface",
            geometry=self._pad_link_interface,
            interface_type="bonded",
        )
        self.validate_geometry()

    @property
    def parameters(self) -> FingertipParameters:
        """Return the immutable dimensions used to construct the model."""
        return self._parameters

    @property
    def outer_pad_geometry(self) -> Polygon:
        """Return the complete half-ellipse before the stem cutout."""
        return self._outer_pad_geometry

    @property
    def pad_material_geometry(self) -> PolygonalGeometry:
        """Return only the compliant pad after removing the full cutout."""
        return self._pad_material_geometry

    @property
    def link_geometry(self) -> Polygon:
        """Return the rigid top plate and centered stem as one polygon."""
        return self._link_geometry

    @property
    def link_plate_geometry(self) -> Polygon:
        """Return only the rigid plate above ``y = 0``."""
        return self._link_plate_geometry

    @property
    def stem_geometry(self) -> Polygon:
        """Return only the rigid stem inserted into the pad cutout."""
        return self._stem_geometry

    @property
    def cutout_geometry(self) -> Polygon:
        """Return the full rectangular region reserved around the stem."""
        return self._cutout_geometry

    @property
    def raw_material_geometry(self) -> PolygonalGeometry:
        """Return the exterior assembly before clearance removal."""
        return self._raw_material_geometry

    @property
    def void_geometry(self) -> PolygonalGeometry | None:
        """Return visible clearance, or ``None`` for a zero-clearance fit."""
        return self._void_geometry

    @property
    def material_geometry(self) -> PolygonalGeometry:
        """Return the union of compliant and rigid material after clearance."""
        return self._material_geometry

    @property
    def pad_link_interface(self) -> MultiLineString:
        """Return the two interface segments outside the centered cutout."""
        return self._pad_link_interface

    @property
    def interface_definition(self) -> InterfaceDefinition:
        """Return metadata for the always-bonded upper interface."""
        return self._interface_definition

    @property
    def boundaries(self) -> FingertipBoundaries:
        """Return analytic boundary segments and potential contact pairs."""
        return self._boundaries

    @property
    def contact_pairs(self) -> tuple[ContactPair, ContactPair, ContactPair]:
        """Return left, right, and bottom stem-pad potential contact pairs."""
        return self._boundaries.contact_pairs

    @property
    def symmetry_axis(self) -> LineString:
        """Return the vertical axis spanning the pad and rigid link plate."""
        return self._symmetry_axis

    def classify_void(self) -> VoidClassification:
        """Describe which independent stem-clearance dimensions are nonzero."""
        if self._parameters.void_width == 0.0 and self._parameters.void_height == 0.0:
            return "zero_clearance_fit"
        if self._parameters.void_width > 0.0 and self._parameters.void_height == 0.0:
            return "side_clearance"
        if self._parameters.void_width == 0.0 and self._parameters.void_height > 0.0:
            return "bottom_clearance"
        return "u_clearance"

    def pad_link_connection_length(self) -> float:
        """Return the total link-pad interface length outside the cutout."""
        return float(self._pad_link_interface.length)

    def is_material_connected(self) -> bool:
        """Return whether the complete rigid/compliant assembly is connected."""
        if self._material_geometry.is_empty:
            return False
        component_count = (
            len(self._material_geometry.geoms)
            if isinstance(self._material_geometry, MultiPolygon)
            else 1
        )
        return (
            component_count == 1
            and self.pad_link_connection_length() > self._parameters.geometry_tolerance
        )

    def validate_geometry(self) -> None:
        """Raise if a material domain is invalid or the bonded interface vanishes."""
        named_geometries = {
            "outer pad": self._outer_pad_geometry,
            "compliant pad": self._pad_material_geometry,
            "rigid link": self._link_geometry,
            "assembly": self._material_geometry,
        }
        for name, geometry in named_geometries.items():
            if geometry.is_empty:
                raise InvalidFingertipGeometry(f"{name} geometry is empty")
            if not geometry.is_valid:
                raise InvalidFingertipGeometry(f"{name} geometry is invalid")
        if self.pad_link_connection_length() <= self._parameters.geometry_tolerance:
            raise InvalidFingertipGeometry(
                "the always-bonded upper interface has zero effective length"
            )

    def summary(self) -> dict[str, object]:
        """Return parameter values and derived geometry diagnostics."""
        void_area = 0.0 if self._void_geometry is None else self._void_geometry.area
        return {
            "parameters": asdict(self._parameters),
            "void_classification": self.classify_void(),
            "interface_type": self._interface_definition.interface_type,
            "boundary_tags": tuple(self._boundaries.segments),
            "contact_gaps": {
                pair.name: pair.initial_normal_gap for pair in self.contact_pairs
            },
            "cutout_width": self._parameters.cutout_width,
            "cutout_height": self._parameters.cutout_height,
            "material_area": float(self._material_geometry.area),
            "raw_material_area": float(self._raw_material_geometry.area),
            "outer_pad_area": float(self._outer_pad_geometry.area),
            "pad_area": float(self._pad_material_geometry.area),
            "link_area": float(self._link_geometry.area),
            "void_area": float(void_area),
            "removed_material_area": float(
                self._raw_material_geometry.area - self._material_geometry.area
            ),
            "material_connected": self.is_material_connected(),
            "pad_link_connection_length": self.pad_link_connection_length(),
            "geometry_valid": self._material_geometry.is_valid,
            "bounds": tuple(float(value) for value in self._material_geometry.bounds),
        }

    def _build_pad_outer_arc(self) -> LineString:
        half_width = self._parameters.pad_width / 2.0
        pad_depth = self._parameters.pad_height
        arc_segments = (
            self._parameters.arc_resolution
            if self._parameters.arc_resolution % 2 == 0
            else self._parameters.arc_resolution + 1
        )
        angles = np.linspace(0.0, np.pi, arc_segments + 1)
        return LineString(
            [
                (
                    float(half_width * np.cos(angle)),
                    float(-pad_depth * np.sin(angle)),
                )
                for angle in angles
            ]
        )

    def _build_outer_pad(self) -> Polygon:
        half_width = self._parameters.pad_width / 2.0
        pad = orient(
            Polygon([(-half_width, 0.0), *self._pad_outer_arc_geometry.coords]),
            sign=1.0,
        )
        if pad.is_empty or not pad.is_valid:
            raise InvalidFingertipGeometry("half-ellipse pad construction failed")
        return pad

    def _build_link_plate(self) -> Polygon:
        parameters = self._parameters
        return orient(
            box(
                -parameters.link_width / 2.0,
                0.0,
                parameters.link_width / 2.0,
                parameters.link_thickness,
            ),
            sign=1.0,
        )

    def _build_stem(self) -> Polygon:
        parameters = self._parameters
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
            self._link_plate_geometry.union(self._stem_geometry), "rigid link"
        )
        if not isinstance(rigid_link, Polygon):
            raise InvalidFingertipGeometry(
                "rigid link construction is not a single polygon"
            )
        return rigid_link

    def _build_cutout(self) -> Polygon:
        parameters = self._parameters
        return box(
            -parameters.cutout_half_width,
            -parameters.cutout_depth,
            parameters.cutout_half_width,
            0.0,
        )

    def _build_void_geometry(self) -> PolygonalGeometry | None:
        clearance = self._cutout_geometry.difference(self._stem_geometry)
        if clearance.is_empty:
            return None
        return self._validated_polygonal_geometry(clearance, "void")

    def _build_boundaries(self) -> FingertipBoundaries:
        parameters = self._parameters
        pad_edge = parameters.pad_width / 2.0
        cutout_edge = parameters.cutout_half_width
        stem_edge = parameters.stem_width / 2.0
        stem_bottom_y = -parameters.stem_height
        cutout_bottom_y = -parameters.cutout_depth

        pad_bond_left = BoundarySegment(
            "pad_bond_left", LineString([(-pad_edge, 0.0), (-cutout_edge, 0.0)])
        )
        pad_bond_right = BoundarySegment(
            "pad_bond_right", LineString([(cutout_edge, 0.0), (pad_edge, 0.0)])
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
