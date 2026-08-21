"""Neutral rigid-carrier mesh construction from the fingertip solid."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import triangulate

from finger.extrusion import FingertipSolid

from .object import RigidObjectMesh


_CARRIER_Z_MIN_MM = -5.5
_CARRIER_Z_MAX_MM = 5.5
_XY_KEY_DIGITS = 12


@dataclass(frozen=True)
class RigidCarrierMesh:
    """Rigid carrier surface with semantic lateral/end face ownership.

    Newton consumes the closed ``surface_mesh`` while periodic ray tracing
    consumes only ``lateral_face_indices``.  The longitudinal end faces are
    numerical cell caps, not physical optical boundaries.
    """

    surface_mesh: RigidObjectMesh
    cross_section: Polygon | MultiPolygon
    z_min_mm: float
    z_max_mm: float
    lateral_face_indices: tuple[int, ...]
    longitudinal_end_face_indices: tuple[int, ...]
    morphology_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.surface_mesh, RigidObjectMesh):
            raise TypeError("surface_mesh must be a RigidObjectMesh")
        if not isinstance(self.cross_section, (Polygon, MultiPolygon)):
            raise TypeError("cross_section must be a Polygon or MultiPolygon")
        if self.cross_section.is_empty or not self.cross_section.is_valid:
            raise ValueError("cross_section must be valid and non-empty")
        z_min = float(self.z_min_mm)
        z_max = float(self.z_max_mm)
        if not math.isfinite(z_min) or not math.isfinite(z_max) or z_min >= z_max:
            raise ValueError("carrier z bounds must be finite with z_min_mm < z_max_mm")
        face_count = len(self.surface_mesh.faces)

        def normalized_indices(
            values: tuple[int, ...],
            *,
            name: str,
        ) -> tuple[int, ...]:
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a tuple of face indices")
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in values
            ):
                raise TypeError(f"{name} must contain integer face indices")
            if not values:
                raise ValueError(f"{name} must be non-empty")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicate face indices")
            if any(value < 0 or value >= face_count for value in values):
                raise ValueError(f"{name} contains an out-of-range face index")
            return tuple(int(value) for value in values)

        lateral = normalized_indices(
            self.lateral_face_indices,
            name="lateral_face_indices",
        )
        longitudinal_ends = normalized_indices(
            self.longitudinal_end_face_indices,
            name="longitudinal_end_face_indices",
        )
        if set(lateral) & set(longitudinal_ends):
            raise ValueError("carrier lateral and longitudinal end faces must be disjoint")
        if set(lateral) | set(longitudinal_ends) != set(range(face_count)):
            raise ValueError("carrier semantic face groups must cover the closed surface mesh")
        if (
            not isinstance(self.morphology_fingerprint, str)
            or not self.morphology_fingerprint
        ):
            raise ValueError("morphology_fingerprint must be a non-empty string")
        object.__setattr__(self, "z_min_mm", z_min)
        object.__setattr__(self, "z_max_mm", z_max)
        object.__setattr__(self, "lateral_face_indices", lateral)
        object.__setattr__(
            self,
            "longitudinal_end_face_indices",
            longitudinal_ends,
        )


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


def make_distal_phalanx_mesh(solid: FingertipSolid) -> RigidCarrierMesh:
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
    lateral_face_indices: list[int] = []
    longitudinal_end_face_indices: list[int] = []
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
                first_face_index = len(faces)
                faces.extend(
                    (
                        (bottom[index], bottom[next_index], top[next_index]),
                        (bottom[index], top[next_index], top[index]),
                    )
                )
                lateral_face_indices.extend(
                    (first_face_index, first_face_index + 1)
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
            first_face_index = len(faces)
            faces.extend((bottom_face, top_face))
            longitudinal_end_face_indices.extend(
                (first_face_index, first_face_index + 1)
            )

    surface_mesh = RigidObjectMesh(
        vertices_mm=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        name="distal_phalanx_carrier",
    )
    return RigidCarrierMesh(
        surface_mesh=surface_mesh,
        cross_section=solid.rigid_geometry,
        z_min_mm=_CARRIER_Z_MIN_MM,
        z_max_mm=_CARRIER_Z_MAX_MM,
        lateral_face_indices=tuple(lateral_face_indices),
        longitudinal_end_face_indices=tuple(longitudinal_end_face_indices),
        morphology_fingerprint=solid.morphology_fingerprint,
    )


__all__ = ["RigidCarrierMesh", "make_distal_phalanx_mesh"]
