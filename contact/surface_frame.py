"""Surface frames for registering a spherical contact on the fingertip arc."""

from __future__ import annotations

from dataclasses import dataclass
import math

from shapely.geometry import Point

from model.fingertip_model import FingertipModel

Vector2 = tuple[float, float]


class InvalidSurfaceFrame(ValueError):
    """Raised when the fingertip surface cannot provide a usable frame."""


@dataclass(frozen=True)
class CrownFrame:
    """Local orthonormal frame derived from the actual pad outer boundary."""

    point_mm: Vector2
    tangent: Vector2
    pad_outward_normal: Vector2
    loading_direction: Vector2
    arc_distance_mm: float


def _normalized(vector: Vector2) -> Vector2:
    length = math.hypot(*vector)
    if not math.isfinite(length) or length <= 0.0:
        raise InvalidSurfaceFrame("cannot normalize a zero-length surface tangent")
    return vector[0] / length, vector[1] / length


def crown_frame_from_model(model: FingertipModel) -> CrownFrame:
    """Derive the central crown, tangent, and outward normal from Shapely."""

    if not isinstance(model, FingertipModel):
        raise TypeError("model must be a FingertipModel")
    arc = model.boundaries.segments["pad_outer_arc"].geometry
    intersection = arc.intersection(model.symmetry_axis)
    if isinstance(intersection, Point):
        points = [intersection]
    elif hasattr(intersection, "geoms"):
        points = [geometry for geometry in intersection.geoms if isinstance(geometry, Point)]
    else:
        points = []
    if not points:
        raise InvalidSurfaceFrame(
            "pad_outer_arc does not intersect the model symmetry axis"
        )
    crown = min(points, key=lambda point: (point.y, abs(point.x)))
    distance = float(arc.project(crown))
    sample_distance = max(
        1.0e-6 * arc.length,
        100.0 * model.parameters.geometry_tolerance,
    )
    before = arc.interpolate(max(0.0, distance - sample_distance))
    after = arc.interpolate(min(arc.length, distance + sample_distance))
    tangent = _normalized((after.x - before.x, after.y - before.y))
    normal_candidates = ((-tangent[1], tangent[0]), (tangent[1], -tangent[0]))
    probe_distance = max(1.0e-4, 1000.0 * model.parameters.geometry_tolerance)
    outside_candidates = [
        candidate
        for candidate in normal_candidates
        if not model.pad_material_geometry.covers(
            Point(
                crown.x + probe_distance * candidate[0],
                crown.y + probe_distance * candidate[1],
            )
        )
    ]
    if len(outside_candidates) != 1:
        raise InvalidSurfaceFrame(
            "the pad outward normal is ambiguous at the central crown"
        )
    outward = _normalized(outside_candidates[0])
    return CrownFrame(
        point_mm=(float(crown.x), float(crown.y)),
        tangent=tangent,
        pad_outward_normal=outward,
        loading_direction=(-outward[0], -outward[1]),
        arc_distance_mm=distance,
    )


def surface_frame_from_normalized_location(
    model: FingertipModel,
    normalized_location: float,
) -> CrownFrame:
    """Return the local frame at one normalized point on ``pad_outer_arc``.

    Zero is the right bonded endpoint, one half is the crown, and one is the
    left bonded endpoint.  The local outward normal defines contact approach;
    the loading direction is the central crown direction for compatibility
    with the current sphere-contact contract.
    """

    if not isinstance(model, FingertipModel):
        raise TypeError("model must be a FingertipModel")
    location = float(normalized_location)
    if not math.isfinite(location) or not 0.0 <= location <= 1.0:
        raise InvalidSurfaceFrame(
            "normalized contact location must be finite and lie in [0, 1]"
        )
    arc = model.boundaries.segments["pad_outer_arc"].geometry
    distance = location * float(arc.length)
    point = arc.interpolate(distance)
    sample_distance = max(
        1.0e-6 * arc.length,
        100.0 * model.parameters.geometry_tolerance,
    )
    before = arc.interpolate(max(0.0, distance - sample_distance))
    after = arc.interpolate(min(arc.length, distance + sample_distance))
    tangent = _normalized((after.x - before.x, after.y - before.y))
    candidates = ((-tangent[1], tangent[0]), (tangent[1], -tangent[0]))
    probe_distance = max(1.0e-4, 1000.0 * model.parameters.geometry_tolerance)
    outside = [
        candidate
        for candidate in candidates
        if not model.pad_material_geometry.covers(
            Point(
                point.x + probe_distance * candidate[0],
                point.y + probe_distance * candidate[1],
            )
        )
    ]
    if len(outside) != 1:
        interior = model.pad_material_geometry.representative_point()
        radial = (point.x - interior.x, point.y - interior.y)
        outward = max(
            candidates,
            key=lambda candidate: (
                candidate[0] * radial[0] + candidate[1] * radial[1]
            ),
        )
    else:
        outward = outside[0]
    central = crown_frame_from_model(model)
    return CrownFrame(
        point_mm=(float(point.x), float(point.y)),
        tangent=tangent,
        pad_outward_normal=_normalized(outward),
        loading_direction=central.loading_direction,
        arc_distance_mm=distance,
    )


__all__ = [
    "CrownFrame",
    "InvalidSurfaceFrame",
    "crown_frame_from_model",
    "surface_frame_from_normalized_location",
]
