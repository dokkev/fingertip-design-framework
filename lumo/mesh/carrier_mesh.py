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


def _make_carrier_5led_mesh(
    carrier: Carrier,
    *,
    main_length_mm: float,
    distal_end_cap_length_mm: float,
    led_centers_y_mm: tuple[float, ...],
    led_recess_width_mm: float,
    led_recess_depth_mm: float,
) -> "newton.Mesh":
    """Build the recessed main rail and distal dorsal reinforcement."""
    if not isinstance(carrier, Carrier):
        raise TypeError("carrier must be a Carrier geometry")

    boundary = tuple(
        (float(x_mm), float(z_mm))
        for x_mm, z_mm in carrier.cross_section
    )
    half_width_mm = max(abs(x_mm) for x_mm, _ in boundary)
    outer_edge_z_mm = tuple(
        z_mm
        for x_mm, z_mm in boundary
        if isclose(abs(x_mm), half_width_mm, abs_tol=1.0e-12)
    )
    if len(set(outer_edge_z_mm)) != 2:
        raise ValueError("carrier must have one dorsal plate at its outer edge")
    dorsal_bottom_z_mm = min(outer_edge_z_mm)
    dorsal_top_z_mm = max(outer_edge_z_mm)
    dorsal_bottom_x_mm = sorted(
        x_mm
        for x_mm, z_mm in boundary
        if isclose(z_mm, dorsal_bottom_z_mm, abs_tol=1.0e-12)
    )
    dorsal_boundary = (
        *((x_mm, dorsal_bottom_z_mm) for x_mm in dorsal_bottom_x_mm),
        (half_width_mm, dorsal_top_z_mm),
        (-half_width_mm, dorsal_top_z_mm),
    )

    return _extrude_recessed_rail(
        boundary,
        main_length_mm=main_length_mm,
        led_centers_y_mm=led_centers_y_mm,
        led_recess_width_mm=led_recess_width_mm,
        led_recess_depth_mm=led_recess_depth_mm,
        distal_extension_boundary=dorsal_boundary,
        distal_extension_length_mm=distal_end_cap_length_mm,
    )


def _extrude_recessed_rail(
    boundary: tuple[tuple[float, float], ...],
    *,
    main_length_mm: float,
    led_centers_y_mm: tuple[float, ...],
    led_recess_width_mm: float,
    led_recess_depth_mm: float,
    distal_extension_boundary: tuple[tuple[float, float], ...] | None = None,
    distal_extension_length_mm: float = 0.0,
) -> "newton.Mesh":
    """Extrude one rail with a shallow bottom recess at every LED station."""
    if _signed_area(boundary) <= 0.0:
        raise ValueError("carrier rail boundary must be counter-clockwise")
    if main_length_mm <= 0.0:
        raise ValueError("main_length_mm must be positive")
    if led_recess_width_mm <= 0.0 or led_recess_depth_mm <= 0.0:
        raise ValueError("LED recess dimensions must be positive")

    stem_bottom_z_mm = min(z_mm for _, z_mm in boundary)
    stem_bottom_x_mm = sorted(
        x_mm
        for x_mm, z_mm in boundary
        if isclose(z_mm, stem_bottom_z_mm, abs_tol=1.0e-12)
    )
    if len(stem_bottom_x_mm) != 2:
        raise ValueError("carrier rail must have one horizontal stem bottom")
    stem_left_x_mm, stem_right_x_mm = stem_bottom_x_mm
    recess_floor_z_mm = stem_bottom_z_mm + led_recess_depth_mm

    # Split the stem-side edges at the recess floor so each transition cap
    # shares exact edges with the neighboring longitudinal side surfaces.
    split_boundary: list[tuple[float, float]] = []
    for index, start in enumerate(boundary):
        end = boundary[(index + 1) % len(boundary)]
        split_boundary.append(start)
        if (
            isclose(start[0], end[0], abs_tol=1.0e-12)
            and any(
                isclose(start[0], stem_x_mm, abs_tol=1.0e-12)
                for stem_x_mm in (stem_left_x_mm, stem_right_x_mm)
            )
            and min(start[1], end[1])
            < recess_floor_z_mm
            < max(start[1], end[1])
        ):
            split_boundary.append((start[0], recess_floor_z_mm))
    boundary = tuple(split_boundary)

    rail_polygon = Polygon(boundary)
    recess_cutout = Polygon(
        (
            (stem_left_x_mm, stem_bottom_z_mm - led_recess_depth_mm),
            (stem_right_x_mm, stem_bottom_z_mm - led_recess_depth_mm),
            (stem_right_x_mm, recess_floor_z_mm),
            (stem_left_x_mm, recess_floor_z_mm),
        )
    )
    recessed_polygon = rail_polygon.difference(recess_cutout)
    removed_polygon = rail_polygon.intersection(recess_cutout)
    if not isinstance(recessed_polygon, Polygon) or not isinstance(
        removed_polygon,
        Polygon,
    ):
        raise ValueError("LED recess must preserve one connected carrier rail")
    if recessed_polygon.is_empty or not recessed_polygon.is_valid:
        raise ValueError("LED recess produced an invalid carrier section")

    def polygon_boundary(polygon: Polygon) -> tuple[tuple[float, float], ...]:
        points = tuple(
            (float(x_mm), float(z_mm))
            for x_mm, z_mm, *_ in polygon.exterior.coords[:-1]
        )
        if _signed_area(points) < 0.0:
            points = tuple(reversed(points))
        return points

    recessed_boundary = polygon_boundary(recessed_polygon)
    proximal_y_mm = -0.5 * main_length_mm
    rail_end_y_mm = 0.5 * main_length_mm
    half_recess_width_mm = 0.5 * led_recess_width_mm
    recess_intervals = tuple(
        (center_y_mm - half_recess_width_mm, center_y_mm + half_recess_width_mm)
        for center_y_mm in sorted(led_centers_y_mm)
    )
    previous_upper_y_mm = proximal_y_mm
    for lower_y_mm, upper_y_mm in recess_intervals:
        if lower_y_mm < proximal_y_mm or upper_y_mm > rail_end_y_mm:
            raise ValueError("LED recess lies outside the main carrier rail")
        if lower_y_mm < previous_upper_y_mm:
            raise ValueError("LED recesses must not overlap")
        previous_upper_y_mm = upper_y_mm

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

    def add_sides(
        side_boundary: tuple[tuple[float, float], ...],
        lower_y_mm: float,
        upper_y_mm: float,
    ) -> None:
        lower = [
            vertex_index(x_mm, lower_y_mm, z_mm)
            for x_mm, z_mm in side_boundary
        ]
        upper = [
            vertex_index(x_mm, upper_y_mm, z_mm)
            for x_mm, z_mm in side_boundary
        ]
        for index in range(len(side_boundary)):
            next_index = (index + 1) % len(side_boundary)
            faces.extend(
                (
                    (lower[index], upper[index], upper[next_index]),
                    (lower[index], upper[next_index], lower[next_index]),
                )
            )

    def add_cap(polygon: Polygon, y_mm: float, *, positive_y: bool) -> None:
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
            raise ValueError("carrier cap triangulation is incomplete")

        for triangle in cap_triangles:
            coordinates = tuple(
                (float(x_mm), float(z_mm))
                for x_mm, z_mm, *_ in triangle.exterior.coords[:-1]
            )
            if _signed_area(coordinates) < 0.0:
                coordinates = (
                    coordinates[0],
                    coordinates[2],
                    coordinates[1],
                )
            face = tuple(
                vertex_index(x_mm, y_mm, z_mm)
                for x_mm, z_mm in coordinates
            )
            if positive_y:
                face = (face[0], face[2], face[1])
            faces.append(face)

    add_cap(rail_polygon, proximal_y_mm, positive_y=False)
    cursor_y_mm = proximal_y_mm
    for lower_y_mm, upper_y_mm in recess_intervals:
        if lower_y_mm > cursor_y_mm:
            add_sides(boundary, cursor_y_mm, lower_y_mm)
        add_cap(removed_polygon, lower_y_mm, positive_y=True)
        add_sides(recessed_boundary, lower_y_mm, upper_y_mm)
        add_cap(removed_polygon, upper_y_mm, positive_y=False)
        cursor_y_mm = upper_y_mm
    if cursor_y_mm < rail_end_y_mm:
        add_sides(boundary, cursor_y_mm, rail_end_y_mm)

    if distal_extension_boundary is None:
        if distal_extension_length_mm != 0.0:
            raise ValueError(
                "distal extension length requires an extension boundary"
            )
        add_cap(rail_polygon, rail_end_y_mm, positive_y=True)
    else:
        if distal_extension_length_mm <= 0.0:
            raise ValueError("distal extension length must be positive")
        extension_polygon = Polygon(distal_extension_boundary)
        if not rail_polygon.covers(extension_polygon):
            raise ValueError("distal extension must remain inside the rail")
        distal_y_mm = rail_end_y_mm + distal_extension_length_mm
        add_sides(
            distal_extension_boundary,
            rail_end_y_mm,
            distal_y_mm,
        )
        rail_end_geometry = rail_polygon.difference(extension_polygon)
        rail_end_polygons = (
            (rail_end_geometry,)
            if isinstance(rail_end_geometry, Polygon)
            else tuple(
                geometry
                for geometry in rail_end_geometry.geoms
                if isinstance(geometry, Polygon)
            )
        )
        if not rail_end_polygons:
            raise ValueError("carrier rail end must contain a closed surface")
        for rail_end_polygon in rail_end_polygons:
            add_cap(rail_end_polygon, rail_end_y_mm, positive_y=True)
        add_cap(extension_polygon, distal_y_mm, positive_y=True)

    try:
        import newton
    except ImportError as exc:
        raise RuntimeError("carrier meshing requires newton") from exc

    return newton.Mesh(
        vertices=np.asarray(vertices_mm, dtype=np.float32) * _MM_TO_M,
        indices=np.asarray(faces, dtype=np.int32).reshape(-1),
        compute_inertia=False,
        is_solid=True,
    )


def _make_carrier_collision_mesh(
    carrier: Carrier,
    silicone: Silicone,
    *,
    extrusion_depth_mm: float = 11.0,
) -> "newton.Mesh":
    """Build a closed proxy whose reachable boundary faces the cavity."""
    boundary = _carrier_collision_boundary(carrier, silicone)

    # Put the signed-query closure caps one silicone half-depth beyond the
    # silicone mesh on each side. Only the cavity-facing side wall remains
    # reachable within the representative single-section extrusion.
    return _extrude_closed_polygon(
        boundary,
        extrusion_depth_mm=2.0 * extrusion_depth_mm,
        compute_inertia=False,
    )


def _make_carrier_5led_collision_mesh(
    carrier: Carrier,
    silicone: Silicone,
    *,
    main_length_mm: float,
    led_centers_y_mm: tuple[float, ...],
    led_recess_width_mm: float,
    led_recess_depth_mm: float,
) -> "newton.Mesh":
    """Build the recessed 55 mm rail collision proxy."""
    boundary = _carrier_collision_boundary(carrier, silicone)
    return _extrude_recessed_rail(
        boundary,
        main_length_mm=main_length_mm,
        led_centers_y_mm=led_centers_y_mm,
        led_recess_width_mm=led_recess_width_mm,
        led_recess_depth_mm=led_recess_depth_mm,
    )


def _carrier_collision_boundary(
    carrier: Carrier,
    silicone: Silicone,
) -> tuple[tuple[float, float], ...]:
    """Return the closed XZ proxy section shared by both mesh paths."""
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

    return boundary


__all__ = []
