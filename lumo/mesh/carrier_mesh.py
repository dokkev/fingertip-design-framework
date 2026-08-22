"""Generate a rigid carrier surface mesh from a fingertip assembly."""

from __future__ import annotations

from math import isclose
from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import triangulate

from lumo.fingertip.fingertip import Carrier

if TYPE_CHECKING:
    import newton


_MM_TO_M = 1.0e-3


def _signed_area(points: tuple[tuple[float, float], ...]) -> float:
    return 0.5 * sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
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
    if _signed_area(boundary) <= 0.0:
        raise ValueError("fingertip carrier boundary must be counter-clockwise")

    polygon = Polygon(boundary)
    if polygon.is_empty or not polygon.is_valid:
        raise ValueError("fingertip carrier cross-section must be valid")

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
            "carrier cap triangulation does not cover the analytic boundary"
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
            raise ValueError("carrier cap triangulation produced a non-triangle")
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
        raise RuntimeError(
            "carrier meshing requires newton"
        ) from exc

    return newton.Mesh(
        vertices=vertices_m,
        indices=indices,
        is_solid=True,
    )


__all__ = []
