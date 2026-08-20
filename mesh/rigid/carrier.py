"""Neutral rigid-carrier mesh construction from the fingertip solid."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import triangulate

from model.solid import FingertipSolid

from .rigid_object import RigidObjectMesh


_CARRIER_Z_MIN_MM = -5.5
_CARRIER_Z_MAX_MM = 5.5
_XY_KEY_DIGITS = 12


def _iter_polygons(geometry: Polygon | MultiPolygon) -> tuple[Polygon, ...]:
    if isinstance(geometry, Polygon):
        return (geometry,)
    if isinstance(geometry, MultiPolygon):
        return tuple(sorted(geometry.geoms, key=lambda polygon: polygon.wkb_hex))
    raise TypeError(
        "FingertipSolid.rigid_geometry must be a Polygon or MultiPolygon, "
        f"got {type(geometry).__name__}"
    )


def _xy_key(x: float, y: float) -> tuple[float, float]:
    return round(float(x), _XY_KEY_DIGITS), round(float(y), _XY_KEY_DIGITS)


def _signed_area(coordinates: Iterable[tuple[float, float]]) -> float:
    points = tuple(coordinates)
    return 0.5 * sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )


def _oriented_ring(ring, *, hole: bool) -> tuple[tuple[float, float], ...]:
    coordinates = tuple((float(x), float(y)) for x, y, *_ in ring.coords[:-1])
    if len(coordinates) < 3 or abs(_signed_area(coordinates)) <= 1.0e-12:
        raise ValueError("distal phalanx contains a degenerate boundary ring")

    # Outward side-wall winding uses CCW exterior rings and CW hole rings.
    wants_ccw = not hole
    is_ccw = _signed_area(coordinates) > 0.0
    if is_ccw != wants_ccw:
        coordinates = tuple(reversed(coordinates))
    return coordinates


def _triangle_coordinates(triangle: Polygon) -> tuple[tuple[float, float], ...]:
    coordinates = tuple((float(x), float(y)) for x, y, *_ in triangle.exterior.coords[:-1])
    if len(coordinates) != 3 or abs(_signed_area(coordinates)) <= 1.0e-12:
        raise ValueError("distal phalanx cap triangulation produced a degenerate triangle")
    if _signed_area(coordinates) < 0.0:
        coordinates = (coordinates[0], coordinates[2], coordinates[1])
    return coordinates


def make_distal_phalanx_mesh(solid: FingertipSolid) -> RigidObjectMesh:
    """Create the closed 11 mm carrier surface from ``solid.rigid_geometry``.

    The helper is neutral geometry only.  It preserves the authoritative XY
    carrier polygon and does not recenter, rescale, or introduce a volume/TET
    representation.
    """

    if not isinstance(solid, FingertipSolid):
        raise TypeError("solid must be FingertipSolid")
    if not math.isclose(solid.z_min_mm, _CARRIER_Z_MIN_MM, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("distal phalanx carrier requires z_min_mm=-5.5")
    if not math.isclose(solid.z_max_mm, _CARRIER_Z_MAX_MM, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("distal phalanx carrier requires z_max_mm=+5.5")

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    vertex_indices: dict[tuple[tuple[float, float], float], int] = {}

    def vertex_index(x: float, y: float, z: float) -> int:
        key = (_xy_key(x, y), float(z))
        if key not in vertex_indices:
            vertex_indices[key] = len(vertices)
            vertices.append((float(x), float(y), float(z)))
        return vertex_indices[key]

    for polygon in _iter_polygons(solid.rigid_geometry):
        if polygon.is_empty or not polygon.is_valid:
            raise ValueError("distal phalanx source geometry must be valid and non-empty")

        rings = ((polygon.exterior, False), *((ring, True) for ring in polygon.interiors))
        for ring, is_hole in rings:
            coordinates = _oriented_ring(ring, hole=is_hole)
            bottom = [vertex_index(x, y, _CARRIER_Z_MIN_MM) for x, y in coordinates]
            top = [vertex_index(x, y, _CARRIER_Z_MAX_MM) for x, y in coordinates]
            for index, next_index in enumerate(range(1, len(coordinates) + 1)):
                next_index %= len(coordinates)
                faces.extend(
                    (
                        (bottom[index], bottom[next_index], top[next_index]),
                        (bottom[index], top[next_index], top[index]),
                    )
                )

        cap_triangles = tuple(triangulate(polygon))
        kept_triangles = tuple(
            triangle
            for triangle in cap_triangles
            if triangle.area > 1.0e-12 and polygon.covers(triangle)
        )
        covered_area = float(sum(triangle.area for triangle in kept_triangles))
        if not math.isclose(covered_area, polygon.area, rel_tol=1.0e-9, abs_tol=1.0e-10):
            raise ValueError(
                "distal phalanx cap triangulation does not cover the authoritative polygon"
            )

        for triangle in kept_triangles:
            coordinates = _triangle_coordinates(triangle)
            top_face = tuple(
                vertex_index(x, y, _CARRIER_Z_MAX_MM) for x, y in coordinates
            )
            bottom_face = tuple(
                vertex_index(x, y, _CARRIER_Z_MIN_MM) for x, y in coordinates
            )
            bottom_face = (bottom_face[0], bottom_face[2], bottom_face[1])
            faces.extend((bottom_face, top_face))

    return RigidObjectMesh(
        vertices_mm=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        name="distal_phalanx_carrier",
        metadata={
            "source_geometry": "FingertipSolid.rigid_geometry",
            "extrusion_depth_mm": _CARRIER_Z_MAX_MM - _CARRIER_Z_MIN_MM,
            "cross_section_wkt": solid.rigid_geometry.wkt,
            "z_min_mm": _CARRIER_Z_MIN_MM,
            "z_max_mm": _CARRIER_Z_MAX_MM,
        },
    )


__all__ = ["make_distal_phalanx_mesh"]
