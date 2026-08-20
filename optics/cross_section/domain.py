"""Neutral geometry for two-dimensional fingertip optical transport."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import linemerge, unary_union

from model.fingertip_model import PolygonalGeometry
from model.fingertip import Fingertip
from mesh.pad import PadMesh
from mesh.indenter import IndenterPose2D
from optics.contact_object import IndenterOptics


class CrossSectionOpticsError(RuntimeError):
    """Raised when a 2D optical domain or trace is invalid."""


@dataclass(frozen=True)
class _OpticalDomain:
    """FEA-independent regions used by the cross-sectional ray tracer."""

    outer_envelope: PolygonalGeometry
    silicone_region: PolygonalGeometry
    rigid_region: PolygonalGeometry
    accessible_region: PolygonalGeometry
    source_position_mm: tuple[float, float]
    source_emission_axis_2d: tuple[float, float]
    geometry_tolerance_mm: float


    indenter_region: Polygon | None = None
    contact_patch: LineString | MultiLineString | None = None
    indenter_optics: IndenterOptics | None = None
    indenter_center_mm: tuple[float, float] | None = None


OpticalDomain = _OpticalDomain


_OUTER_ENVELOPE_TAGS = (
    "pad_bond_left",
    "pad_outer_left",
    "pad_outer_arc",
    "pad_outer_right",
    "pad_bond_right",
)


def _build_deformed_outer_envelope(
    template: PadMesh,
    coordinates_mm: np.ndarray,
    silicone_region: Polygon,
    *,
    tolerance_mm: float,
) -> Polygon:
    """Close the tagged deformed external shell across the cutout mouth."""
    missing_tags = sorted(
        set(_OUTER_ENVELOPE_TAGS).difference(
            template.semantic_boundary_tags
        )
    )
    if missing_tags:
        available = ", ".join(template.semantic_boundary_tags) or "<none>"
        raise CrossSectionOpticsError(
            "loaded optical geometry requires a semantically tagged pad "
            f"mesh; missing outer-envelope tags: {missing_tags}; available "
            f"semantic tags: {available}"
        )

    shell_segments = [
        LineString([coordinates_mm[int(first)], coordinates_mm[int(second)]])
        for tag in _OUTER_ENVELOPE_TAGS
        for first, second in template.boundary_edges_for(tag)
    ]
    merged_geometry = unary_union(shell_segments)
    if merged_geometry.is_empty:
        raise CrossSectionOpticsError("deformed external pad shell is empty")
    if isinstance(merged_geometry, LineString):
        outer_shell_chain = merged_geometry
    else:
        outer_shell_chain = linemerge(merged_geometry)
    if isinstance(outer_shell_chain, MultiLineString):
        raise CrossSectionOpticsError(
            "deformed external pad shell is disconnected"
        )
    if not isinstance(outer_shell_chain, LineString):
        raise CrossSectionOpticsError(
            "deformed external pad shell did not form one line chain"
        )
    if outer_shell_chain.is_empty or not outer_shell_chain.is_simple:
        raise CrossSectionOpticsError(
            "deformed external pad shell must be nonempty and simple"
        )
    if outer_shell_chain.is_ring:
        raise CrossSectionOpticsError(
            "deformed external pad shell must remain open at the cutout mouth"
        )
    coordinates = list(outer_shell_chain.coords)
    if len(coordinates) < 3:
        raise CrossSectionOpticsError(
            "deformed external pad shell has too few coordinates"
        )
    endpoint_distance = hypot(
        coordinates[-1][0] - coordinates[0][0],
        coordinates[-1][1] - coordinates[0][1],
    )
    if endpoint_distance <= tolerance_mm:
        raise CrossSectionOpticsError(
            "deformed external pad shell needs two distinct mouth endpoints"
        )

    outer_envelope = Polygon(coordinates)
    if outer_envelope.is_empty or not outer_envelope.is_valid:
        raise CrossSectionOpticsError(
            "virtual closure produced an invalid deformed outer envelope"
        )
    if outer_envelope.area <= 0.0:
        raise CrossSectionOpticsError(
            "deformed outer envelope must have positive area"
        )
    if outer_envelope.interiors:
        raise CrossSectionOpticsError(
            "deformed outer envelope must not contain interior holes"
        )
    if outer_envelope.area < silicone_region.area:
        raise CrossSectionOpticsError(
            "deformed outer envelope area is smaller than silicone area"
        )
    if not outer_envelope.buffer(tolerance_mm).covers(silicone_region):
        raise CrossSectionOpticsError(
            "deformed outer envelope does not cover the silicone region"
        )
    return outer_envelope


def _validate_domain(
    tip: Fingertip,
    *,
    outer_envelope: PolygonalGeometry,
    silicone_region: PolygonalGeometry,
    indenter_pose: IndenterPose2D | None = None,
    indenter_optics: IndenterOptics | None = None,
) -> _OpticalDomain:
    rigid_region = tip.geometry.link_geometry
    if (indenter_pose is None) != (indenter_optics is None):
        raise CrossSectionOpticsError(
            "indenter_pose and indenter_optics must be supplied together"
        )
    indenter_region = None if indenter_pose is None else indenter_pose.carrier_geometry
    contact_patch = None if indenter_pose is None else indenter_pose.contact_patch
    indenter_center = None if indenter_pose is None else indenter_pose.center_mm
    if indenter_pose is not None:
        if contact_patch is None or contact_patch.is_empty:
            raise CrossSectionOpticsError(
                "BLOCKED_CONTACT_INTERFACE_MAPPING: requested indenter optics "
                "requires a nonempty mechanical contact patch"
            )
    occupied_region = rigid_region
    # The current indenter optical contract models only the exact
    # silicone-to-object contact patch.  The planar OptiX scene does not
    # contain an exposed indenter GAS, so its carrier must not be invented as
    # an air-side blocker or P2 occupancy mask here.
    accessible_region = outer_envelope.difference(occupied_region)
    if not isinstance(accessible_region, Polygon | MultiPolygon):
        raise CrossSectionOpticsError(
            "outer envelope minus rigid region is not polygonal"
        )
    for name, geometry in (
        ("outer envelope", outer_envelope),
        ("silicone region", silicone_region),
        ("rigid region", rigid_region),
        ("accessible region", accessible_region),
    ):
        if geometry.is_empty:
            raise CrossSectionOpticsError(f"{name} is empty")
        if not geometry.is_valid:
            raise CrossSectionOpticsError(f"{name} is invalid")
    if indenter_region is not None and (
        indenter_region.is_empty or not indenter_region.is_valid
    ):
        raise CrossSectionOpticsError("posed indenter region is invalid")

    tolerance = tip.geometry.parameters.geometry_tolerance
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise CrossSectionOpticsError(
            "geometry_tolerance_mm must be finite and greater than zero"
        )
    source_position = tip.led_source
    emission_axis = tip.emission_axis
    emission_axis_norm = hypot(*emission_axis)
    if (
        not isfinite(emission_axis_norm)
        or abs(emission_axis_norm - 1.0) > tolerance
    ):
        raise CrossSectionOpticsError("the LED emission axis must be a unit vector")

    led_min_x, led_min_y, led_max_x, _ = (
        tip.led_package_geometry.bounds
    )
    expected_source = (0.5 * (led_min_x + led_max_x), led_min_y)
    if (
        abs(source_position[0] - expected_source[0]) > tolerance
        or abs(source_position[1] - expected_source[1]) > tolerance
    ):
        raise CrossSectionOpticsError(
            "the LED source is not centered on its lower emitting edge"
        )
    probe_distance = max(10.0 * tolerance, 1.0e-7)
    distal_probe = Point(
        source_position[0] + probe_distance * emission_axis[0],
        source_position[1] + probe_distance * emission_axis[1],
    )
    if not outer_envelope.covers(distal_probe):
        raise CrossSectionOpticsError(
            "a distal step from the LED source leaves the outer envelope"
        )
    if not accessible_region.covers(distal_probe):
        raise CrossSectionOpticsError(
            "a distal step from the LED source does not enter the optical region"
        )
    return _OpticalDomain(
        outer_envelope=outer_envelope,
        silicone_region=silicone_region,
        rigid_region=rigid_region,
        accessible_region=accessible_region,
        indenter_region=indenter_region,
        contact_patch=contact_patch,
        indenter_optics=indenter_optics,
        indenter_center_mm=indenter_center,
        source_position_mm=source_position,
        source_emission_axis_2d=emission_axis,
        geometry_tolerance_mm=tolerance,
    )


def build_no_load_domain(
    tip: Fingertip,
    *,
    indenter_pose: IndenterPose2D | None = None,
    indenter_optics: IndenterOptics | None = None,
) -> _OpticalDomain:
    """Build a neutral optical domain from the analytic undeformed geometry."""
    return _validate_domain(
        tip,
        outer_envelope=tip.geometry.outer_pad_geometry,
        silicone_region=tip.geometry.pad_material_geometry,
        indenter_pose=indenter_pose,
        indenter_optics=indenter_optics,
    )


def build_mesh_domain(
    tip: Fingertip,
    mesh: PadMesh,
    *,
    indenter_pose: IndenterPose2D | None = None,
    indenter_optics: IndenterOptics | None = None,
) -> _OpticalDomain:
    """Build an optical domain from a reference or deformed pad mesh view."""
    coordinates = mesh.coordinates
    triangle_polygons = [
        Polygon(coordinates[triangle]) for triangle in mesh.triangles
    ]
    silicone_region = unary_union(triangle_polygons)
    if not isinstance(silicone_region, Polygon | MultiPolygon):
        raise CrossSectionOpticsError(
            "deformed pad triangles did not produce polygonal geometry"
        )
    if silicone_region.is_empty or not silicone_region.is_valid:
        raise CrossSectionOpticsError("deformed silicone region is invalid")
    if isinstance(silicone_region, MultiPolygon):
        raise CrossSectionOpticsError(
            "deformed silicone triangles form a disconnected pad"
        )
    outer_envelope = _build_deformed_outer_envelope(
        mesh,
        coordinates,
        silicone_region,
        tolerance_mm=tip.geometry.parameters.geometry_tolerance,
    )
    return _validate_domain(
        tip,
        outer_envelope=outer_envelope,
        silicone_region=silicone_region,
        indenter_pose=indenter_pose,
        indenter_optics=indenter_optics,
    )


# Retained for validation-only imports while production owners use the public
# name above.
_build_mesh_domain = build_mesh_domain
_build_no_load_domain = build_no_load_domain
