"""Validated fingertip parameters and solver-independent pad geometry."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import sqrt
from typing import Literal

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

from util import require_finite, require_positive


class InvalidFingertipParameters(ValueError):
    """Raised when fingertip dimensions cannot define a valid LIT pad."""


PRODUCTION_MINIMUM_SILICONE_THICKNESS_MM = 5.0
MAX_TOTAL_PAD_DEPTH_MM = 30.0


def ellipse_depth_at_cutout_mm(
    *,
    flat_pad_width: float,
    semielliptical_pad_height: float,
    stem_width: float,
    void_width: float,
) -> float:
    """Return the analytic lower-envelope depth at the cutout side."""
    half_width = float(flat_pad_width) / 2.0
    cutout_half_width = float(stem_width) / 2.0 + float(void_width)
    normalized_x = cutout_half_width / half_width
    if not 0.0 <= normalized_x < 1.0:
        raise ValueError("cutout must lie strictly inside the pad half-width")
    return float(semielliptical_pad_height) * sqrt(1.0 - normalized_x**2)


def _require_finite(name: str, value: float) -> None:
    require_finite(
        name,
        value,
        error_type=InvalidFingertipParameters,
    )


def _require_positive(name: str, value: float) -> None:
    require_positive(
        name,
        value,
        error_type=InvalidFingertipParameters,
    )


@dataclass(frozen=True)
class FingertipParameters:
    """Fingertip geometry and compliant-pad mechanical parameters.

    All dimensions are in millimeters. The flat-pad, semi-ellipse, and rigid
    link share one full width: ``flat_pad_width``. The link-pad interface is
    ``y = 0`` and the distal direction is negative ``y``. ``void_width`` is
    the one-sided clearance beside the stem; ``void_height`` is the
    additional clearance below the stem tip. Production zero-height
    morphology uses a bonded basal stem/pad interface; positive height
    remains a finite-clearance diagnostic geometry.
    """

    flat_pad_width: float = 30.0
    flat_pad_height: float = 5.0
    semielliptical_pad_height: float = 9.0
    link_thickness: float = 3.5
    bond_extension_width: float = 4.0
    bond_extension_height: float = 2.0
    stem_width: float = 7.6
    stem_height: float = 6.0
    void_width: float = 1.0
    void_height: float = 0.0
    arc_resolution: int = 128
    geometry_tolerance: float = 1e-9
    young_modulus_mpa: float = 0.55
    poisson_ratio: float = 0.49

    def __post_init__(self) -> None:
        """Validate values immediately so every instance is usable."""
        self.validate()

    def validate(self) -> None:
        """Raise ``InvalidFingertipParameters`` for inconsistent dimensions."""
        _require_finite("flat_pad_width", self.flat_pad_width)
        _require_finite("flat_pad_height", self.flat_pad_height)
        _require_finite("semielliptical_pad_height", self.semielliptical_pad_height)
        _require_finite("link_thickness", self.link_thickness)
        _require_finite("bond_extension_width", self.bond_extension_width)
        _require_finite("bond_extension_height", self.bond_extension_height)
        _require_finite("stem_width", self.stem_width)
        _require_finite("stem_height", self.stem_height)
        _require_finite("void_width", self.void_width)
        _require_finite("void_height", self.void_height)
        _require_finite("geometry_tolerance", self.geometry_tolerance)
        _require_finite("young_modulus_mpa", self.young_modulus_mpa)
        _require_finite("poisson_ratio", self.poisson_ratio)

        for name in (
            "flat_pad_width",
            "flat_pad_height",
            "semielliptical_pad_height",
            "link_thickness",
            "bond_extension_width",
            "bond_extension_height",
            "stem_width",
            "stem_height",
        ):
            _require_positive(name, getattr(self, name))

        if self.void_width < 0.0 or self.void_height < 0.0:
            raise InvalidFingertipParameters(
                "void_width and void_height must be nonnegative"
            )
        total_pad_depth = self.flat_pad_height + self.semielliptical_pad_height
        cutout_width = self.stem_width + 2.0 * self.void_width
        cutout_half_width = cutout_width / 2.0
        cutout_height = self.stem_height + self.void_height
        if total_pad_depth > MAX_TOTAL_PAD_DEPTH_MM + self.geometry_tolerance:
            raise InvalidFingertipParameters(
                "total pad depth must not exceed 30 mm: "
                f"flat_pad_height={self.flat_pad_height:g}, "
                "semielliptical_pad_height="
                f"{self.semielliptical_pad_height:g}, "
                f"total={total_pad_depth:g}"
            )
        _require_positive("young_modulus_mpa", self.young_modulus_mpa)
        if not -1.0 < self.poisson_ratio < 0.5:
            raise InvalidFingertipParameters(
                "poisson_ratio must lie strictly between -1 and 0.5"
            )
        _require_positive("geometry_tolerance", self.geometry_tolerance)
        if (
            not isinstance(self.arc_resolution, int)
            or isinstance(self.arc_resolution, bool)
            or self.arc_resolution < 16
        ):
            raise InvalidFingertipParameters(
                "arc_resolution must be an integer of at least 16"
            )
        if self.bond_extension_height >= self.link_thickness:
            raise InvalidFingertipParameters(
                "bond_extension_height must be smaller than link_thickness"
            )
        if (
            2.0 * self.bond_extension_width + cutout_width
            >= self.flat_pad_width
        ):
            raise InvalidFingertipParameters(
                "2*bond_extension_width + cutout_width must be smaller than "
                "flat_pad_width: "
                f"bond_extension_width={self.bond_extension_width:g}, "
                f"cutout_width={cutout_width:g}, "
                f"flat_pad_width={self.flat_pad_width:g}"
            )

        half_width = self.flat_pad_width / 2.0
        if cutout_half_width >= half_width - self.geometry_tolerance:
            raise InvalidFingertipParameters(
                "cutout must remain strictly inside the external half-width: "
                f"cutout_half_width={cutout_half_width:g}, "
                f"available_half_width={half_width - self.geometry_tolerance:g}, "
                f"geometry_tolerance={self.geometry_tolerance:g}"
            )

        penetration_depth = max(
            0.0,
            cutout_height - self.flat_pad_height,
        )
        if penetration_depth > 0.0:
            available_ellipse_depth = ellipse_depth_at_cutout_mm(
                flat_pad_width=self.flat_pad_width,
                semielliptical_pad_height=self.semielliptical_pad_height,
                stem_width=self.stem_width,
                void_width=self.void_width,
            )
            if penetration_depth >= available_ellipse_depth - self.geometry_tolerance:
                raise InvalidFingertipParameters(
                    "cutout bottom must remain strictly inside the "
                    "semielliptical envelope: "
                    f"penetration_depth={penetration_depth:g}, "
                    f"available_depth={available_ellipse_depth:g}, "
                    f"geometry_tolerance={self.geometry_tolerance:g}"
                )


def fingertip_parameters_fingerprint(parameters: FingertipParameters) -> str:
    """Return a fingerprint for the physical morphology parameters.

    Sampling resolution, geometry tolerance, and material properties are
    representation/numerical/material settings rather than morphology
    identity.  The explicit payload is serialized only at this boundary so
    the same physical morphology has one stable identity across adapters.
    """
    if not isinstance(parameters, FingertipParameters):
        raise TypeError("parameters must be FingertipParameters")
    payload = json.dumps(
        {
            "flat_pad_width": parameters.flat_pad_width,
            "flat_pad_height": parameters.flat_pad_height,
            "semielliptical_pad_height": parameters.semielliptical_pad_height,
            "link_thickness": parameters.link_thickness,
            "bond_extension_width": parameters.bond_extension_width,
            "bond_extension_height": parameters.bond_extension_height,
            "stem_width": parameters.stem_width,
            "stem_height": parameters.stem_height,
            "void_width": parameters.void_width,
            "void_height": parameters.void_height,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

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
            and self.pad_link_connection_length() > self.parameters.geometry_tolerance
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
        if overlap_area > self.parameters.geometry_tolerance:
            raise InvalidFingertipGeometry(
                "compliant pad and rigid link materials overlap: "
                f"overlap_area={overlap_area:g}"
            )
        if self.pad_link_connection_length() <= self.parameters.geometry_tolerance:
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
        tolerance = self.parameters.geometry_tolerance
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
