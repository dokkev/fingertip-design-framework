"""Gmsh TET4 meshing of the authoritative compliant-pad solid."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable

import numpy as np
from shapely.geometry import LineString, Polygon

from lumo.mesh.volume.contracts import (
    FingertipVolumeMesh,
    SurfaceTriangle,
    Tetrahedron,
    VolumeMeshQuality,
    VolumeMeshSettings,
    VolumeMeshValidation,
    VolumeNode,
)
from lumo.finger.extrusion import FingertipSolid


class VolumeMeshDependencyError(RuntimeError):
    """Raised when the Gmsh Python API cannot be imported."""


class VolumeMeshingError(RuntimeError):
    """Raised when the authoritative solid cannot be meshed safely."""


def _import_gmsh() -> Any:
    try:
        import gmsh
    except (ImportError, OSError) as exception:
        raise VolumeMeshDependencyError(
            "M2 requires the Gmsh Python API; install gmsh in the active interpreter"
        ) from exception
    return gmsh


def _iter_polygons(geometry: Any) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        return (geometry,)
    if hasattr(geometry, "geoms"):
        return tuple(geometry.geoms)
    raise VolumeMeshingError(f"expected polygonal geometry, got {type(geometry).__name__}")


def _add_ring(gmsh: Any, coordinates: Iterable[tuple[float, float]], z_mm: float) -> int:
    points = [(float(x), float(y)) for x, y in coordinates]
    if points and points[0] == points[-1]:
        points.pop()
    if len(points) < 3:
        raise VolumeMeshingError("a solid ring needs at least three points")
    point_tags = [gmsh.model.occ.addPoint(x, y, z_mm) for x, y in points]
    curves = [
        gmsh.model.occ.addLine(point_tags[index], point_tags[(index + 1) % len(point_tags)])
        for index in range(len(point_tags))
    ]
    return gmsh.model.occ.addCurveLoop(curves)


def _add_planar_domain(gmsh: Any, geometry: Any, z_mm: float) -> tuple[int, ...]:
    surfaces: list[int] = []
    for polygon in _iter_polygons(geometry):
        loops = [_add_ring(gmsh, polygon.exterior.coords, z_mm)]
        loops.extend(_add_ring(gmsh, ring.coords, z_mm) for ring in polygon.interiors)
        surfaces.append(gmsh.model.occ.addPlaneSurface(loops))
    return tuple(surfaces)


def _signed_tetra_volume(points: np.ndarray) -> float:
    first, second, third, fourth = points
    return float(np.linalg.det(np.vstack((second - first, third - first, fourth - first))) / 6.0)


def _tetra_quality(points: np.ndarray, volume: float) -> float:
    edge_sum = sum(
        float(np.sum((points[i] - points[j]) ** 2))
        for i in range(4)
        for j in range(i + 1, 4)
    )
    if edge_sum <= 0.0 or volume <= 0.0:
        return 0.0
    return float(12.0 * (3.0 * volume) ** (2.0 / 3.0) / edge_sum)


def _projected_line(points: np.ndarray) -> LineString | None:
    unique: list[tuple[float, float]] = []
    for point in points:
        xy = (float(point[0]), float(point[1]))
        if not unique or math.dist(unique[-1], xy) > 1.0e-10:
            unique.append(xy)
    if len(unique) >= 2 and unique[0] == unique[-1]:
        unique.pop()
    if len(unique) < 2:
        return None
    if len(unique) > 2:
        pair = max(
            ((unique[i], unique[j]) for i in range(len(unique)) for j in range(i + 1, len(unique))),
            key=lambda value: math.dist(*value),
        )
        return LineString(pair)
    return LineString(unique)


def _surface_tag(solid: FingertipSolid, points: np.ndarray, *, z_tolerance: float) -> str:
    if np.all(np.abs(points[:, 2] - solid.z_min_mm) <= z_tolerance):
        return "longitudinal_end_minus"
    if np.all(np.abs(points[:, 2] - solid.z_max_mm) <= z_tolerance):
        return "longitudinal_end_plus"
    line = _projected_line(points)
    if line is None:
        raise VolumeMeshingError("could not project a lateral surface triangle")
    tolerance = max(
        1.0e-7,
        100.0 * solid.parameters.geometry_length_tolerance_mm,
    )
    definitions = [definition for definition in solid.surfaces if definition.source_geometry is not None]
    matches = [
        definition.name
        for definition in definitions
        if definition.source_geometry.buffer(tolerance, cap_style=2).covers(line)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        midpoint = line.interpolate(0.5, normalized=True)
        narrowed = [
            definition.name
            for definition in definitions
            if definition.name in matches
            and definition.source_geometry.distance(midpoint) <= tolerance
        ]
        if len(narrowed) == 1:
            return narrowed[0]
        raise VolumeMeshingError(f"ambiguous semantic surface provenance: {matches!r}")
    return "outer_compliant_other"


def _tetrahedra_by_face(
    tetrahedra: tuple[Tetrahedron, ...] | list[Tetrahedron],
) -> dict[tuple[int, int, int], list[Tetrahedron]]:
    result: dict[tuple[int, int, int], list[Tetrahedron]] = defaultdict(list)
    for tetrahedron in tetrahedra:
        first, second, third, fourth = tetrahedron.node_ids
        for face in (
            (first, second, third),
            (first, second, fourth),
            (first, third, fourth),
            (second, third, fourth),
        ):
            result[tuple(sorted(face))].append(tetrahedron)
    return result


def _surface_orientation(
    node_ids: tuple[int, int, int],
    *,
    nodes: dict[int, VolumeNode],
    tetrahedra_by_face: dict[tuple[int, int, int], list[Tetrahedron]],
) -> float:
    adjacent = tetrahedra_by_face.get(tuple(sorted(node_ids)), [])
    if len(adjacent) != 1:
        raise VolumeMeshingError(
            "surface triangle is not a unique tetrahedral boundary face"
        )
    points = np.asarray(
        [
            [nodes[node_id].x_mm, nodes[node_id].y_mm, nodes[node_id].z_mm]
            for node_id in node_ids
        ],
        dtype=float,
    )
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    opposite_node_id = next(
        node_id for node_id in adjacent[0].node_ids if node_id not in node_ids
    )
    opposite = nodes[opposite_node_id]
    interior_direction = np.asarray(
        [opposite.x_mm, opposite.y_mm, opposite.z_mm],
        dtype=float,
    ) - np.mean(points, axis=0)
    return float(np.dot(normal, interior_direction))


def _surface_quality(
    nodes: dict[int, VolumeNode],
    surface_triangles: dict[str, list[SurfaceTriangle]],
    tetrahedra: list[Tetrahedron],
) -> tuple[int, int, int]:
    tetrahedra_by_face = _tetrahedra_by_face(tetrahedra)
    edge_counts: dict[tuple[str, tuple[int, int]], int] = defaultdict(int)
    degenerate = 0
    orientation_failures = 0
    for triangles in surface_triangles.values():
        for triangle in triangles:
            points = np.asarray(
                [[nodes[node_id].x_mm, nodes[node_id].y_mm, nodes[node_id].z_mm]
                for node_id in triangle.node_ids
            ])
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            if (
                len(set(triangle.node_ids)) != 3
                or not np.all(np.isfinite(points))
                or not np.all(np.isfinite(normal))
                or float(np.linalg.norm(normal)) <= 1.0e-12
            ):
                degenerate += 1
                continue
            if _surface_orientation(
                triangle.node_ids,
                nodes=nodes,
                tetrahedra_by_face=tetrahedra_by_face,
            ) >= -1.0e-12:
                orientation_failures += 1
            for first, second in (
                (triangle.node_ids[0], triangle.node_ids[1]),
                (triangle.node_ids[1], triangle.node_ids[2]),
                (triangle.node_ids[2], triangle.node_ids[0]),
            ):
                edge_counts[(triangle.domain, tuple(sorted((first, second))))] += 1
    return degenerate, orientation_failures, sum(count != 2 for count in edge_counts.values())


def _bonded_interface_quality(
    solid: FingertipSolid,
    nodes: dict[int, VolumeNode],
    surface_triangles: dict[str, list[SurfaceTriangle]],
) -> tuple[int, float, float, float]:
    """Check authoritative bonded boundaries after the exact z extrusion."""
    bonded_tags = tuple(
        definition.name
        for definition in solid.surfaces
        if definition.kind == "support" and definition.source_geometry is not None
    )
    expected = sum(
        float(definition.source_geometry.length * solid.extrusion_depth_mm)
        for definition in solid.surfaces
        if definition.name in bonded_tags and definition.source_geometry is not None
    )
    actual = 0.0
    triangle_count = 0
    for tag in bonded_tags:
        for triangle in surface_triangles.get(tag, ()):
            points = np.asarray(
                [[nodes[node_id].x_mm, nodes[node_id].y_mm, nodes[node_id].z_mm]
                for node_id in triangle.node_ids
            ])
            actual += 0.5 * float(
                np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))
            )
            triangle_count += 1
    relative_error = abs(actual - expected) / expected if expected > 0.0 else math.inf
    return triangle_count, actual, expected, relative_error


def _configure(gmsh: Any, settings: VolumeMeshSettings) -> None:
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.MaxNumThreads1D", 1)
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1)
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", 1)
    gmsh.option.setNumber("Mesh.RandomFactor", 0.0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", settings.target_size_mm)
    gmsh.option.setNumber("Mesh.MeshSizeMax", settings.target_size_mm)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)


def generate_volume_mesh(solid: FingertipSolid, settings: VolumeMeshSettings) -> FingertipVolumeMesh:
    """Generate a true 3D tetrahedral mesh from the compliant pad solid."""
    if not isinstance(solid, FingertipSolid):
        raise TypeError("solid must be FingertipSolid")
    if not isinstance(settings, VolumeMeshSettings):
        raise TypeError("settings must be VolumeMeshSettings")
    if not solid.watertight:
        raise VolumeMeshingError("refusing a semantic solid that failed its closed-volume gate")
    gmsh = _import_gmsh()
    try:
        gmsh.initialize()
    except Exception as exc:
        raise VolumeMeshDependencyError(
            f"Gmsh runtime could not initialize: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        gmsh.model.add("fingertip_pad_3d")
        _configure(gmsh, settings)
        planar_surfaces = _add_planar_domain(gmsh, solid.pad_geometry, solid.z_min_mm)
        gmsh.model.occ.synchronize()
        extrusion = gmsh.model.occ.extrude(
            [(2, tag) for tag in planar_surfaces], 0.0, 0.0, solid.extrusion_depth_mm
        )
        gmsh.model.occ.synchronize()
        volume_entity = next((entity for entity in extrusion if entity[0] == 3), None)
        if volume_entity is None:
            raise VolumeMeshingError("OCC extrusion did not create the compliant-pad volume")
        volumes = gmsh.model.getEntities(3)
        if len(volumes) != 1 or volumes[0][1] != volume_entity[1]:
            raise VolumeMeshingError(f"expected one compliant-pad volume, got {volumes!r}")
        volume_tag = int(volume_entity[1])
        gmsh.model.mesh.generate(3)

        node_tags, flat_coordinates, _ = gmsh.model.mesh.getNodes()
        nodes = {
            int(tag): VolumeNode(
                int(tag),
                float(flat_coordinates[3 * index]),
                float(flat_coordinates[3 * index + 1]),
                float(flat_coordinates[3 * index + 2]),
            )
            for index, tag in enumerate(node_tags)
        }
        if not nodes:
            raise VolumeMeshingError("Gmsh returned no volume nodes")
        tetrahedra: list[Tetrahedron] = []
        volume_element_ids = {"pad": []}
        quality_values: list[float] = []
        mesh_volume = 0.0
        max_edge = 0.0
        inverted = 0
        element_types, element_groups, connectivity_groups = gmsh.model.mesh.getElements(3, volume_tag)
        for element_type, element_tags, connectivity in zip(element_types, element_groups, connectivity_groups):
            name, dimension, order, number_of_nodes, _, _ = gmsh.model.mesh.getElementProperties(element_type)
            if dimension != 3 or order != 1 or number_of_nodes != 4:
                raise VolumeMeshingError(f"3D volume mesh must contain only Tetrahedron4; got {name}")
            for index, element_tag in enumerate(element_tags):
                raw_ids = tuple(int(value) for value in connectivity[index * 4 : index * 4 + 4])
                points = np.asarray(
                    [[nodes[node_id].x_mm, nodes[node_id].y_mm, nodes[node_id].z_mm] for node_id in raw_ids]
                )
                signed_volume = _signed_tetra_volume(points)
                if signed_volume < 0.0:
                    raise VolumeMeshingError(
                        "Gmsh returned an inverted tetrahedron"
                    )
                if signed_volume <= 0.0:
                    raise VolumeMeshingError("Gmsh returned a zero-volume tetrahedron")
                mesh_volume += signed_volume
                quality_values.append(_tetra_quality(points, signed_volume))
                for i in range(4):
                    for j in range(i + 1, 4):
                        max_edge = max(max_edge, float(np.linalg.norm(points[i] - points[j])))
                tetrahedra.append(Tetrahedron(int(element_tag), raw_ids, "pad"))
                volume_element_ids["pad"].append(int(element_tag))
        if not tetrahedra:
            raise VolumeMeshingError("Gmsh returned no tetrahedral elements")
        tetrahedra_by_face = _tetrahedra_by_face(tetrahedra)

        surface_triangles: dict[str, list[SurfaceTriangle]] = {}
        surface_id = 1
        for _, surface_tag in gmsh.model.getBoundary([(3, volume_tag)], oriented=False, recursive=False):
            element_types, element_groups, connectivity_groups = gmsh.model.mesh.getElements(2, surface_tag)
            for element_type, _, connectivity in zip(element_types, element_groups, connectivity_groups):
                name, dimension, order, number_of_nodes, _, _ = gmsh.model.mesh.getElementProperties(element_type)
                if dimension != 2 or order != 1 or number_of_nodes != 3:
                    raise VolumeMeshingError(f"surface mesh must contain only Triangle3; got {name}")
                for offset in range(0, len(connectivity), 3):
                    ids = tuple(int(value) for value in connectivity[offset : offset + 3])
                    points = np.asarray(
                        [[nodes[node_id].x_mm, nodes[node_id].y_mm, nodes[node_id].z_mm] for node_id in ids]
                    )
                    orientation = _surface_orientation(
                        ids,
                        nodes=nodes,
                        tetrahedra_by_face=tetrahedra_by_face,
                    )
                    if abs(orientation) <= 1.0e-12:
                        raise VolumeMeshingError(
                            "surface triangle orientation is geometrically ambiguous"
                        )
                    if orientation > 0.0:
                        ids = (ids[0], ids[2], ids[1])
                    tag = _surface_tag(
                        solid,
                        points,
                        z_tolerance=max(
                            1.0e-7,
                            solid.parameters.geometry_length_tolerance_mm * 100.0,
                        ),
                    )
                    surface_triangles.setdefault(tag, []).append(SurfaceTriangle(surface_id, ids, tag, "pad"))
                    surface_id += 1

        expected_surface_names = set(solid.surface_names)
        missing = expected_surface_names - set(surface_triangles)
        if missing:
            raise VolumeMeshingError(f"semantic surface families disappeared: {sorted(missing)!r}")
        degenerate, orientation_failures, closed_edge_failures = _surface_quality(
            nodes, surface_triangles, tetrahedra
        )
        bonded_count, bonded_area, expected_bonded_area, bonded_area_error = _bonded_interface_quality(
            solid, nodes, surface_triangles
        )
        geometry_volume = solid.volume_mm3
        relative_error = abs(mesh_volume - geometry_volume) / geometry_volume
        quality = VolumeMeshQuality(
            node_count=len(nodes),
            tetrahedron_count=len(tetrahedra),
            surface_triangle_count=sum(len(values) for values in surface_triangles.values()),
            minimum_scaled_jacobian=float(min(quality_values)),
            maximum_edge_length_mm=max_edge,
            mesh_volume_mm3=mesh_volume,
            geometry_volume_mm3=geometry_volume,
            volume_relative_error=relative_error,
            inverted_tetrahedron_count=inverted,
            semantic_surface_tags=tuple(sorted(surface_triangles)),
            surface_triangle_degenerate_count=degenerate,
            surface_orientation_failure_count=orientation_failures,
            closed_surface_edge_failure_count=closed_edge_failures,
            bonded_surface_triangle_count=bonded_count,
            bonded_surface_area_mm2=bonded_area,
            bonded_surface_expected_area_mm2=expected_bonded_area,
            bonded_surface_area_relative_error=bonded_area_error,
        )
        all_tetra_node_ids = {node_id for tetra in tetrahedra for node_id in tetra.node_ids}
        checks = {
            "positive_volume": quality.mesh_volume_mm3 > 0.0,
            "all_tetrahedra_positive": quality.inverted_tetrahedron_count == 0,
            "minimum_quality": quality.minimum_scaled_jacobian >= settings.minimum_quality,
            "volume_consistency": quality.volume_relative_error <= 5.0e-3,
            "semantic_surface_coverage": expected_surface_names.issubset(surface_triangles),
            "surface_triangles_nondegenerate": quality.surface_triangle_degenerate_count == 0,
            "surface_orientation_consistent": quality.surface_orientation_failure_count == 0,
            "closed_volume_surface": quality.closed_surface_edge_failure_count == 0,
            "no_orphan_elements": all_tetra_node_ids.issubset(nodes) and bool(all_tetra_node_ids),
            "bonded_surface_present": quality.bonded_surface_triangle_count > 0,
            "bonded_surface_area_consistency": (
                math.isfinite(quality.bonded_surface_area_relative_error)
                and quality.bonded_surface_area_relative_error <= 5.0e-3
            ),
            "morphology_fingerprint_preserved": bool(solid.morphology_fingerprint),
        }
        errors = tuple(name for name, passed in checks.items() if not passed)
        return FingertipVolumeMesh(
            solid=solid,
            nodes=nodes,
            tetrahedra=tuple(sorted(tetrahedra, key=lambda value: value.id)),
            surface_triangles={tag: tuple(values) for tag, values in sorted(surface_triangles.items())},
            volume_element_ids={tag: tuple(sorted(values)) for tag, values in volume_element_ids.items()},
            settings=settings,
            quality=quality,
            validation=VolumeMeshValidation(not errors, checks, errors),
            gmsh_version=str(gmsh.option.getString("General.Version")),
        )
    except VolumeMeshingError:
        raise
    except RuntimeError as exc:
        raise VolumeMeshingError(
            f"Gmsh could not mesh this fingertip solid: {exc}"
        ) from exc
    finally:
        gmsh.finalize()


__all__ = ["VolumeMeshDependencyError", "VolumeMeshingError", "generate_volume_mesh"]
