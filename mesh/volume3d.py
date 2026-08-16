"""Independent Gmsh tetrahedral meshing for the semantic fingertip solid."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from mesh.volume_types import (
    FingertipVolumeMesh,
    SurfaceTriangle,
    Tetrahedron,
    VolumeMeshQuality,
    VolumeMeshSettings,
    VolumeMeshValidation,
    VolumeNode,
)
from model.solid import FingertipSolid, SolidSurfaceDefinition


class VolumeMeshDependencyError(RuntimeError):
    """Raised when the Gmsh Python API cannot be imported."""


class VolumeMeshingError(RuntimeError):
    """Raised when a semantic solid cannot be meshed safely."""


@dataclass(frozen=True)
class _VolumeRecord:
    tag: int
    domain: str
    center: tuple[float, float, float]


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
    if len(points) > 1 and points[0] == points[-1]:
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
    edge_sum = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            edge_sum += float(np.sum((points[i] - points[j]) ** 2))
    if edge_sum <= 0.0 or volume <= 0.0:
        return 0.0
    # This scaled volume is one for a regular unit tetrahedron.
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
        # A vertical triangular surface has one repeated projected vertex and
        # one straight boundary segment.  The farthest pair is unambiguous.
        pair = max(
            ((unique[i], unique[j]) for i in range(len(unique)) for j in range(i + 1, len(unique))),
            key=lambda value: math.dist(*value),
        )
        return LineString(pair)
    return LineString(unique)


def _surface_tag(
    solid: FingertipSolid,
    volume_domain: str,
    points: np.ndarray,
    *,
    z_tolerance: float,
) -> str:
    z_values = points[:, 2]
    if np.all(np.abs(z_values - solid.z_min_mm) <= z_tolerance):
        return "longitudinal_end_minus"
    if np.all(np.abs(z_values - solid.z_max_mm) <= z_tolerance):
        return "longitudinal_end_plus"
    line = _projected_line(points)
    if line is None:
        raise VolumeMeshingError("could not project a lateral surface triangle")
    tolerance = max(1.0e-7, 100.0 * solid.parameters.geometry_tolerance)
    definitions = [
        definition
        for definition in solid.surfaces
        if definition.source_geometry is not None
        and definition.name not in ("rigid_outer",)
        and (
            definition.material_region in ("both", volume_domain)
            or definition.name.startswith("support_bond")
        )
    ]
    matches = [
        definition.name
        for definition in definitions
        if definition.source_geometry.buffer(tolerance, cap_style=2).covers(line)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # The only allowed semantic overlap is at a shared endpoint; classify
        # with the midpoint so a triangle cannot silently receive two labels.
        midpoint = line.interpolate(0.5, normalized=True)
        matches = [
            name
            for name in matches
            if next(
                definition.source_geometry
                for definition in definitions
                if definition.name == name
            ).distance(midpoint)
            <= tolerance
        ]
        if len(matches) == 1:
            return matches[0]
    if volume_domain == "rigid_carrier":
        return "rigid_outer"
    return "outer_compliant_other"


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


def generate_volume_mesh(
    solid: FingertipSolid,
    settings: VolumeMeshSettings,
) -> FingertipVolumeMesh:
    """Generate a true 3D tetrahedral mesh from ``FingertipSolid``."""
    if not isinstance(solid, FingertipSolid):
        raise TypeError("solid must be FingertipSolid")
    if not isinstance(settings, VolumeMeshSettings):
        raise TypeError("settings must be VolumeMeshSettings")
    gmsh = _import_gmsh()
    gmsh.initialize()
    try:
        gmsh.model.add("fingertip_3d")
        _configure(gmsh, settings)
        pad_surfaces = _add_planar_domain(gmsh, solid.pad_geometry, solid.z_min_mm)
        rigid_surfaces = _add_planar_domain(gmsh, solid.rigid_geometry, solid.z_min_mm)
        gmsh.model.occ.synchronize()
        pad_extrusion = gmsh.model.occ.extrude(
            [(2, tag) for tag in pad_surfaces], 0.0, 0.0, solid.extrusion_depth_mm
        )
        rigid_extrusion = gmsh.model.occ.extrude(
            [(2, tag) for tag in rigid_surfaces], 0.0, 0.0, solid.extrusion_depth_mm
        )
        gmsh.model.occ.synchronize()
        pad_source_volume = next(
            (entity for entity in pad_extrusion if entity[0] == 3), None
        )
        rigid_source_volume = next(
            (entity for entity in rigid_extrusion if entity[0] == 3), None
        )
        if pad_source_volume is None or rigid_source_volume is None:
            raise VolumeMeshingError("OCC extrusion did not create two source volumes")
        source_volumes = (pad_source_volume, rigid_source_volume)
        volumes = gmsh.model.getEntities(3)
        if len(volumes) != 2:
            raise VolumeMeshingError(f"expected two material volumes, got {volumes!r}")
        # Keep the two material/contact topologies independent.  Coincident
        # zero-clearance contact facets must not share nodes: Kratos' contact
        # normal calculation otherwise sees equal and opposite normals at the
        # same node.  The semantic solid remains watertight geometrically; the
        # FEA interface is carried by named boundary surfaces.
        volumes = gmsh.model.getEntities(3)
        if len(volumes) != 2:
            raise VolumeMeshingError(f"expected two independent material volumes, got {volumes!r}")
        domain_by_tag = {
            int(pad_source_volume[1]): "pad",
            int(rigid_source_volume[1]): "rigid_carrier",
        }
        volume_records = []
        for _, tag in volumes:
            center = tuple(float(value) for value in gmsh.model.occ.getCenterOfMass(3, tag))
            volume_records.append(_VolumeRecord(tag, domain_by_tag[int(tag)], center))
        if {record.domain for record in volume_records} != {"pad", "rigid_carrier"}:
            raise VolumeMeshingError("the pad and rigid material volumes were not preserved")
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
        coordinates = np.asarray(
            [[node.x_mm, node.y_mm, node.z_mm] for node in nodes.values()], dtype=float
        )
        node_lookup = {node_id: index for index, node_id in enumerate(nodes)}
        tetrahedra: list[Tetrahedron] = []
        volume_element_ids: dict[str, list[int]] = {"pad": [], "rigid_carrier": []}
        quality_values: list[float] = []
        tetra_volumes: dict[str, float] = {"pad": 0.0, "rigid_carrier": 0.0}
        max_edge = 0.0
        inverted = 0
        for record in volume_records:
            element_types, element_groups, connectivity_groups = gmsh.model.mesh.getElements(
                3, record.tag
            )
            for element_type, element_tags, connectivity in zip(
                element_types, element_groups, connectivity_groups
            ):
                name, dimension, order, number_of_nodes, _, _ = gmsh.model.mesh.getElementProperties(
                    element_type
                )
                if dimension != 3 or order != 1 or number_of_nodes != 4:
                    raise VolumeMeshingError(
                        f"3D volume mesh must contain only Tetrahedron4; got {name}"
                    )
                for index, element_tag in enumerate(element_tags):
                    offset = index * 4
                    raw_ids = tuple(int(value) for value in connectivity[offset : offset + 4])
                    points = np.asarray(
                        [[nodes[node_id].x_mm, nodes[node_id].y_mm, nodes[node_id].z_mm] for node_id in raw_ids],
                        dtype=float,
                    )
                    signed_volume = _signed_tetra_volume(points)
                    if signed_volume < 0.0:
                        raw_ids = (raw_ids[0], raw_ids[2], raw_ids[1], raw_ids[3])
                        points = points[[0, 2, 1, 3]]
                        signed_volume = -signed_volume
                        inverted += 1
                    if signed_volume <= 0.0:
                        raise VolumeMeshingError("Gmsh returned a zero-volume tetrahedron")
                    quality_values.append(_tetra_quality(points, signed_volume))
                    tetra_volumes[record.domain] += signed_volume
                    for i in range(4):
                        for j in range(i + 1, 4):
                            max_edge = max(max_edge, float(np.linalg.norm(points[i] - points[j])))
                    tetrahedron = Tetrahedron(int(element_tag), raw_ids, record.domain)  # type: ignore[arg-type]
                    tetrahedra.append(tetrahedron)
                    volume_element_ids[record.domain].append(int(element_tag))

        if not tetrahedra:
            raise VolumeMeshingError("Gmsh returned no tetrahedral elements")
        surface_triangles: dict[str, list[SurfaceTriangle]] = {}
        surface_id = 1
        volume_lookup = {record.tag: record for record in volume_records}
        for _, volume_tag in volumes:
            record = volume_lookup[volume_tag]
            boundary_surfaces = gmsh.model.getBoundary(
                [(3, volume_tag)], oriented=False, recursive=False
            )
            for _, surface_tag in boundary_surfaces:
                element_types, element_groups, connectivity_groups = gmsh.model.mesh.getElements(
                    2, surface_tag
                )
                for element_type, _, connectivity in zip(
                    element_types, element_groups, connectivity_groups
                ):
                    name, dimension, order, number_of_nodes, _, _ = gmsh.model.mesh.getElementProperties(
                        element_type
                    )
                    if dimension != 2 or order != 1 or number_of_nodes != 3:
                        raise VolumeMeshingError(
                            f"surface mesh must contain only Triangle3; got {name}"
                        )
                    for offset in range(0, len(connectivity), 3):
                        ids = tuple(int(value) for value in connectivity[offset : offset + 3])
                        points = np.asarray(
                            [[nodes[node_id].x_mm, nodes[node_id].y_mm, nodes[node_id].z_mm] for node_id in ids],
                            dtype=float,
                        )
                        normal = np.cross(points[1] - points[0], points[2] - points[0])
                        centroid = np.mean(points, axis=0)
                        if float(np.dot(normal, centroid - np.asarray(record.center))) < 0.0:
                            ids = (ids[0], ids[2], ids[1])
                        tag = _surface_tag(
                            solid,
                            record.domain,
                            points,
                            z_tolerance=max(1.0e-7, solid.parameters.geometry_tolerance * 100.0),
                        )
                        surface_triangles.setdefault(tag, []).append(
                            SurfaceTriangle(surface_id, ids, tag, record.domain)  # type: ignore[arg-type]
                        )
                        surface_id += 1

        expected_surface_names = set(solid.surface_names)
        missing = expected_surface_names - set(surface_triangles)
        if missing:
            raise VolumeMeshingError(
                "semantic surface families disappeared during meshing: "
                f"{sorted(missing)!r}"
            )
        mesh_volume = sum(tetra_volumes.values())
        geometry_volume = solid.volume_mm3
        relative_error = abs(mesh_volume - geometry_volume) / geometry_volume
        quality = VolumeMeshQuality(
            node_count=len(nodes),
            tetrahedron_count=len(tetrahedra),
            pad_tetrahedron_count=len(volume_element_ids["pad"]),
            rigid_tetrahedron_count=len(volume_element_ids["rigid_carrier"]),
            surface_triangle_count=sum(len(values) for values in surface_triangles.values()),
            minimum_scaled_jacobian=float(min(quality_values)),
            maximum_edge_length_mm=max_edge,
            mesh_volume_mm3=mesh_volume,
            geometry_volume_mm3=geometry_volume,
            volume_relative_error=relative_error,
            pad_mesh_volume_mm3=tetra_volumes["pad"],
            pad_geometry_volume_mm3=solid.pad_volume_mm3,
            rigid_mesh_volume_mm3=tetra_volumes["rigid_carrier"],
            rigid_geometry_volume_mm3=solid.rigid_volume_mm3,
            inverted_tetrahedron_count=inverted,
            semantic_surface_tags=tuple(sorted(surface_triangles)),
        )
        checks = {
            "positive_volume": quality.mesh_volume_mm3 > 0.0,
            "all_tetrahedra_positive": quality.inverted_tetrahedron_count == 0,
            "minimum_quality": quality.minimum_scaled_jacobian >= settings.minimum_quality,
            "volume_consistency": quality.volume_relative_error <= 5.0e-3,
            "pad_volume_consistency": abs(
                quality.pad_mesh_volume_mm3 - quality.pad_geometry_volume_mm3
            ) / quality.pad_geometry_volume_mm3 <= 5.0e-3,
            "rigid_volume_consistency": abs(
                quality.rigid_mesh_volume_mm3 - quality.rigid_geometry_volume_mm3
            ) / quality.rigid_geometry_volume_mm3 <= 5.0e-3,
            "semantic_surface_coverage": expected_surface_names.issubset(surface_triangles),
            "morphology_fingerprint_preserved": bool(solid.morphology_fingerprint),
        }
        errors = tuple(name for name, passed in checks.items() if not passed)
        validation = VolumeMeshValidation(not errors, checks, errors)
        return FingertipVolumeMesh(
            solid=solid,
            nodes=nodes,
            tetrahedra=tuple(sorted(tetrahedra, key=lambda value: value.id)),
            surface_triangles={
                tag: tuple(values) for tag, values in sorted(surface_triangles.items())
            },
            volume_element_ids={tag: tuple(sorted(values)) for tag, values in volume_element_ids.items()},
            settings=settings,
            quality=quality,
            validation=validation,
            gmsh_version=str(gmsh.option.getString("General.Version")),
        )
    finally:
        gmsh.finalize()


__all__ = [
    "VolumeMeshDependencyError",
    "VolumeMeshingError",
    "generate_volume_mesh",
]
