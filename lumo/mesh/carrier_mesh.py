"""Generate a rigid carrier surface mesh from a fingertip assembly."""

from __future__ import annotations

from math import isclose
from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import triangulate

from lumo.fingertip.fingertip import Carrier, Silicone

if TYPE_CHECKING:
    import newton


_MM_TO_M = 1.0e-3


def _signed_area(points: tuple[tuple[float, float], ...]) -> float:
    return 0.5 * sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )


def _extrude_closed_polygon(
    boundary: tuple[tuple[float, float], ...],
    *,
    extrusion_depth_mm: float,
    compute_inertia: bool,
) -> "newton.Mesh":
    """Extrude one counter-clockwise XZ polygon along Y."""
    if _signed_area(boundary) <= 0.0:
        raise ValueError("extrusion boundary must be counter-clockwise")

    polygon = Polygon(boundary)
    if polygon.is_empty or not polygon.is_valid:
        raise ValueError("extrusion boundary must define a valid polygon")

    cap_triangles = tuple(
        triangle
        for triangle in triangulate(polygon)
        if polygon.covers(triangle)
    )
    covered_area = sum(triangle.area for triangle in cap_triangles)
    if not cap_triangles or not isclose(
        covered_area,
        polygon.area,
        rel_tol=1.0e-9,
        abs_tol=1.0e-10,
    ):
        raise ValueError(
            "cap triangulation does not cover the extrusion boundary"
        )

    half_depth_mm = 0.5 * extrusion_depth_mm
    vertices_mm: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    vertex_indices: dict[tuple[float, float, float], int] = {}

    def vertex_index(x_mm: float, y_mm: float, z_mm: float) -> int:
        key = (
            round(float(x_mm), 12),
            round(float(y_mm), 12),
            round(float(z_mm), 12),
        )
        if key not in vertex_indices:
            vertex_indices[key] = len(vertices_mm)
            vertices_mm.append(key)
        return vertex_indices[key]

    bottom = [
        vertex_index(x_mm, -half_depth_mm, z_mm)
        for x_mm, z_mm in boundary
    ]
    top = [
        vertex_index(x_mm, half_depth_mm, z_mm)
        for x_mm, z_mm in boundary
    ]

    for index, next_index in enumerate(range(1, len(boundary) + 1)):
        next_index %= len(boundary)
        faces.extend(
            (
                (bottom[index], top[index], top[next_index]),
                (bottom[index], top[next_index], bottom[next_index]),
            )
        )

    for triangle in cap_triangles:
        coordinates = tuple(
            (float(x_mm), float(z_mm))
            for x_mm, z_mm, *_ in triangle.exterior.coords[:-1]
        )
        if len(coordinates) != 3:
            raise ValueError("cap triangulation produced a non-triangle")
        if _signed_area(coordinates) < 0.0:
            coordinates = (
                coordinates[0],
                coordinates[2],
                coordinates[1],
            )

        bottom_triangle = tuple(
            vertex_index(x_mm, -half_depth_mm, z_mm)
            for x_mm, z_mm in coordinates
        )
        top_triangle = tuple(
            vertex_index(x_mm, half_depth_mm, z_mm)
            for x_mm, z_mm in coordinates
        )
        faces.extend(
            (
                bottom_triangle,
                (top_triangle[0], top_triangle[2], top_triangle[1]),
            )
        )

    vertices_m = np.asarray(vertices_mm, dtype=np.float32) * _MM_TO_M
    indices = np.asarray(faces, dtype=np.int32).reshape(-1)

    try:
        import newton
    except ImportError as exc:
        raise RuntimeError("carrier meshing requires newton") from exc

    return newton.Mesh(
        vertices=vertices_m,
        indices=indices,
        compute_inertia=compute_inertia,
        is_solid=True,
    )


def _make_carrier_mesh(
    carrier: Carrier,
    *,
    extrusion_depth_mm: float = 11.0,
) -> "newton.Mesh":
    """Extrude analytic carrier geometry into a Newton surface mesh."""
    if not isinstance(carrier, Carrier):
        raise TypeError("carrier must be a Carrier geometry")

    boundary = tuple(
        (float(x_mm), float(z_mm))
        for x_mm, z_mm in carrier.cross_section
    )
    return _extrude_closed_polygon(
        boundary,
        extrusion_depth_mm=extrusion_depth_mm,
        compute_inertia=True,
    )


def _make_carrier_collision_mesh(
    carrier: Carrier,
    silicone: Silicone,
    *,
    extrusion_depth_mm: float = 11.0,
) -> "newton.Mesh":
    """Build a closed proxy whose reachable boundary faces the cavity."""
    if not isinstance(carrier, Carrier):
        raise TypeError("carrier must be a Carrier geometry")
    if not isinstance(silicone, Silicone):
        raise TypeError("silicone must be a Silicone geometry")

    carrier_boundary = tuple(
        (float(x_mm), float(z_mm))
        for x_mm, z_mm in carrier.cross_section
    )
    if _signed_area(carrier_boundary) <= 0.0:
        raise ValueError("fingertip carrier boundary must be counter-clockwise")
    carrier_polygon = Polygon(carrier_boundary)
    if carrier_polygon.is_empty or not carrier_polygon.is_valid:
        raise ValueError("fingertip carrier cross-section must be valid")

    stem_bottom_z_mm = min(z_mm for _, z_mm in carrier_boundary)
    stem_bottom_x_mm = sorted(
        x_mm
        for x_mm, z_mm in carrier_boundary
        if isclose(z_mm, stem_bottom_z_mm, abs_tol=1.0e-12)
    )
    if len(stem_bottom_x_mm) != 2:
        raise ValueError("carrier must have one horizontal stem-bottom segment")

    stem_left_x_mm, stem_right_x_mm = stem_bottom_x_mm
    if not (
        silicone.cavity_left_x_mm <= stem_left_x_mm
        < stem_right_x_mm <= silicone.cavity_right_x_mm
    ):
        raise ValueError("silicone cavity must contain the carrier stem")
    if stem_bottom_z_mm < silicone.cavity_bottom_z_mm:
        raise ValueError("silicone cavity must contain the carrier stem depth")

    cavity_top_z_mm = float(silicone.void_left[0][1])
    if not isclose(
        cavity_top_z_mm,
        silicone.void_right[1][1],
        abs_tol=1.0e-12,
    ):
        raise ValueError("silicone cavity sides must share one top height")

    # Follow the counter-clockwise carrier boundary through the cavity-facing
    # lip and stem. Close the cross-section through the carrier interior because
    # Newton's particle-mesh contact requires a reliable signed mesh query.
    boundary = (
        (silicone.cavity_left_x_mm, cavity_top_z_mm),
        (stem_left_x_mm, cavity_top_z_mm),
        (stem_left_x_mm, stem_bottom_z_mm),
        (stem_right_x_mm, stem_bottom_z_mm),
        (stem_right_x_mm, cavity_top_z_mm),
        (silicone.cavity_right_x_mm, cavity_top_z_mm),
        (silicone.cavity_right_x_mm, silicone.bond_top_z_mm),
        (silicone.cavity_left_x_mm, silicone.bond_top_z_mm),
    )
    boundary = tuple(
        point
        for index, point in enumerate(boundary)
        if index == 0 or point != boundary[index - 1]
    )

    polygon = Polygon(boundary)
    if polygon.is_empty or not polygon.is_valid:
        raise ValueError("carrier collision cross-section must be valid")
    if not carrier_polygon.covers(polygon):
        raise ValueError(
            "carrier collision closure must remain inside the carrier"
        )

    # Put the signed-query closure caps one silicone half-depth beyond the
    # silicone mesh on each side. Only the cavity-facing side wall remains
    # reachable within the physical silicone extrusion.
    return _extrude_closed_polygon(
        boundary,
        extrusion_depth_mm=2.0 * extrusion_depth_mm,
        compute_inertia=False,
    )


__all__ = []
