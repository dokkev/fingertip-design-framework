"""Geometry-conforming Gmsh meshing for ``FingertipModel`` domains."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterable

from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry

from mesh.types import (
    BoundaryEdge,
    FingertipMesh,
    MeshedContactPair,
    MeshDomain,
    MeshNode,
    MeshSettings,
    MeshValidationReport,
    T3Element,
)
from mesh.validation import mesh_quality_statistics, validate_fingertip_mesh
from model.fingertip_model import FingertipModel, PolygonalGeometry


class GmshDependencyError(RuntimeError):
    """Raised when the required Gmsh Python API cannot be imported."""


class FingertipMeshingError(RuntimeError):
    """Raised when Gmsh cannot preserve the fingertip topology contract."""


@dataclass(frozen=True)
class _CurveRecord:
    tag: int
    domain: MeshDomain
    start: tuple[float, float]
    end: tuple[float, float]


@dataclass(frozen=True)
class _DomainEntities:
    surface_tags: tuple[int, ...]
    curves: tuple[_CurveRecord, ...]


def _import_gmsh() -> Any:
    try:
        import gmsh
    except (ImportError, OSError) as exception:
        raise GmshDependencyError(
            "Phase 4M requires the Gmsh Python API; install 'gmsh' in the "
            "active interpreter. No fallback mesher is used."
        ) from exception
    return gmsh


def _iter_polygons(geometry: PolygonalGeometry) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        return (geometry,)
    if isinstance(geometry, MultiPolygon):
        return geometry.geoms
    raise FingertipMeshingError(
        f"expected Polygon or MultiPolygon, got {type(geometry).__name__}"
    )


def _add_ring(
    gmsh: Any,
    coordinates: Iterable[tuple[float, float]],
    domain: MeshDomain,
    boundary_size_mm: float,
    coordinate_tolerance_mm: float,
    split_points: tuple[tuple[float, float], ...],
) -> tuple[int, tuple[_CurveRecord, ...]]:
    points = [(float(x), float(y)) for x, y in coordinates]
    if len(points) >= 2 and math.dist(points[0], points[-1]) <= coordinate_tolerance_mm:
        points.pop()
    filtered_points: list[tuple[float, float]] = []
    for point in points:
        if (
            not filtered_points
            or math.dist(filtered_points[-1], point) > coordinate_tolerance_mm
        ):
            filtered_points.append(point)
    points = filtered_points
    expanded_points: list[tuple[float, float]] = []
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        segment = LineString([start, end])
        length = segment.length
        candidates: list[tuple[float, tuple[float, float]]] = []
        if length > coordinate_tolerance_mm:
            for split_point in split_points:
                point = Point(split_point)
                distance_along = segment.project(point)
                if (
                    segment.distance(point) <= coordinate_tolerance_mm
                    and coordinate_tolerance_mm < distance_along
                    < length - coordinate_tolerance_mm
                ):
                    candidates.append((distance_along, split_point))
        expanded_points.append(start)
        expanded_points.extend(
            point for _, point in sorted(candidates, key=lambda item: item[0])
        )
    points = expanded_points
    if len(points) < 3:
        raise FingertipMeshingError("a polygon ring needs at least three points")
    point_tags = [
        gmsh.model.geo.addPoint(x, y, 0.0, boundary_size_mm)
        for x, y in points
    ]
    curves: list[_CurveRecord] = []
    for index, start_tag in enumerate(point_tags):
        end_index = (index + 1) % len(point_tags)
        end_tag = point_tags[end_index]
        curve_tag = gmsh.model.geo.addLine(start_tag, end_tag)
        curves.append(
            _CurveRecord(
                tag=curve_tag,
                domain=domain,
                start=points[index],
                end=points[end_index],
            )
        )
    loop_tag = gmsh.model.geo.addCurveLoop([curve.tag for curve in curves])
    return loop_tag, tuple(curves)


def _add_domain(
    gmsh: Any,
    geometry: PolygonalGeometry,
    domain: MeshDomain,
    boundary_size_mm: float,
    coordinate_tolerance_mm: float,
    split_points: tuple[tuple[float, float], ...],
) -> _DomainEntities:
    surfaces: list[int] = []
    curves: list[_CurveRecord] = []
    for polygon in _iter_polygons(geometry):
        exterior_loop, exterior_curves = _add_ring(
            gmsh,
            polygon.exterior.coords,
            domain,
            boundary_size_mm,
            coordinate_tolerance_mm,
            split_points,
        )
        loop_tags = [exterior_loop]
        curves.extend(exterior_curves)
        for interior in polygon.interiors:
            interior_loop, interior_curves = _add_ring(
                gmsh,
                interior.coords,
                domain,
                boundary_size_mm,
                coordinate_tolerance_mm,
                split_points,
            )
            loop_tags.append(interior_loop)
            curves.extend(interior_curves)
        surfaces.append(gmsh.model.geo.addPlaneSurface(loop_tags))
    return _DomainEntities(tuple(surfaces), tuple(curves))


def _line_is_on(source: BaseGeometry, candidate: LineString, tolerance: float) -> bool:
    return bool(source.buffer(tolerance, cap_style=1).covers(candidate))


def _contact_source_geometries(model: FingertipModel) -> tuple[BaseGeometry, ...]:
    return tuple(
        geometry
        for pair in model.contact_pairs
        for geometry in (pair.pad_boundary.geometry, pair.stem_boundary.geometry)
    )


def _semantic_split_points(
    model: FingertipModel, domain: MeshDomain
) -> tuple[tuple[float, float], ...]:
    if domain == "pad":
        tags = (
            "pad_bond_left",
            "pad_bond_right",
            "pad_cutout_left",
            "pad_cutout_right",
            "pad_cutout_bottom",
            "pad_outer_arc",
        )
        geometries: tuple[BaseGeometry, ...] = tuple(
            model.boundaries.segments[tag].geometry for tag in tags
        ) + (model.interface_definition.geometry,)
    else:
        tags = ("stem_left", "stem_right", "stem_bottom")
        geometries = tuple(
            model.boundaries.segments[tag].geometry for tag in tags
        ) + (model.interface_definition.geometry,)
    points: set[tuple[float, float]] = set()
    for geometry in geometries:
        lines = geometry.geoms if isinstance(geometry, MultiLineString) else (geometry,)
        for line in lines:
            points.update((float(x), float(y)) for x, y in line.coords)
    return tuple(sorted(points))


def _configure_gmsh(
    gmsh: Any,
    model: FingertipModel,
    settings: MeshSettings,
    domain_entities: tuple[_DomainEntities, _DomainEntities],
) -> None:
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.MaxNumThreads1D", 1)
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1)
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", 1)
    gmsh.option.setNumber("Mesh.RandomFactor", 0.0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", settings.contact_boundary_target_size_mm)
    gmsh.option.setNumber("Mesh.MeshSizeMax", settings.bulk_target_size_mm)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)

    tolerance = settings.classification_tolerance_mm
    contact_sources = _contact_source_geometries(model)
    contact_curve_tags = [
        curve.tag
        for entities in domain_entities
        for curve in entities.curves
        if any(
            _line_is_on(source, LineString([curve.start, curve.end]), tolerance)
            for source in contact_sources
        )
    ]
    if not contact_curve_tags:
        raise FingertipMeshingError("no contact curves were found for local refinement")
    for source in contact_sources:
        matching_curves = [
            curve
            for entities in domain_entities
            for curve in entities.curves
            if _line_is_on(
                source, LineString([curve.start, curve.end]), tolerance
            )
        ]
        if not matching_curves:
            raise FingertipMeshingError(
                "a contact source segment has no matching Gmsh curve"
            )
        for curve in matching_curves:
            curve_length = math.dist(curve.start, curve.end)
            number_of_nodes = (
                math.ceil(
                    curve_length / settings.contact_boundary_target_size_mm
                )
                + 1
            )
            gmsh.model.mesh.setTransfiniteCurve(curve.tag, number_of_nodes)
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(
        distance_field, "CurvesList", sorted(contact_curve_tags)
    )
    gmsh.model.mesh.field.setNumber(distance_field, "Sampling", 100)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(
        threshold_field, "InField", distance_field
    )
    gmsh.model.mesh.field.setNumber(
        threshold_field,
        "SizeMin",
        settings.contact_boundary_target_size_mm,
    )
    gmsh.model.mesh.field.setNumber(
        threshold_field, "SizeMax", settings.bulk_target_size_mm
    )
    gmsh.model.mesh.field.setNumber(
        threshold_field,
        "DistMin",
        settings.contact_boundary_target_size_mm,
    )
    gmsh.model.mesh.field.setNumber(
        threshold_field,
        "DistMax",
        settings.contact_refinement_distance_mm,
    )
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)


def _node_coordinates(gmsh: Any) -> dict[int, tuple[float, float]]:
    node_tags, flattened_coordinates, _ = gmsh.model.mesh.getNodes()
    return {
        int(node_tag): (
            float(flattened_coordinates[3 * index]),
            float(flattened_coordinates[3 * index + 1]),
        )
        for index, node_tag in enumerate(node_tags)
    }


def _extract_domain_elements(
    gmsh: Any,
    entities: _DomainEntities,
    domain: MeshDomain,
    coordinates: dict[int, tuple[float, float]],
) -> tuple[T3Element, ...]:
    elements: list[T3Element] = []
    for surface_tag in entities.surface_tags:
        element_types, element_tag_groups, connectivity_groups = (
            gmsh.model.mesh.getElements(2, surface_tag)
        )
        for element_type, element_tags, connectivity in zip(
            element_types, element_tag_groups, connectivity_groups
        ):
            name, dimension, order, number_of_nodes, _, _ = (
                gmsh.model.mesh.getElementProperties(element_type)
            )
            if dimension != 2 or order != 1 or number_of_nodes != 3:
                raise FingertipMeshingError(
                    "final topology must contain only linear T3 elements; "
                    f"Gmsh returned {name}"
                )
            for index, element_tag in enumerate(element_tags):
                offset = index * number_of_nodes
                node_ids = tuple(
                    int(node_id)
                    for node_id in connectivity[offset : offset + number_of_nodes]
                )
                first, second, third = (coordinates[node_id] for node_id in node_ids)
                signed_double_area = _signed_double_area(first, second, third)
                if signed_double_area < 0.0:
                    node_ids = (node_ids[0], node_ids[2], node_ids[1])
                elements.append(T3Element(int(element_tag), node_ids, domain))
    return tuple(sorted(elements, key=lambda element: element.id))


def _extract_curve_edges(
    gmsh: Any,
    curve: _CurveRecord,
    coordinates: dict[int, tuple[float, float]],
) -> tuple[BoundaryEdge, ...]:
    element_types, _, connectivity_groups = gmsh.model.mesh.getElements(1, curve.tag)
    edges: list[tuple[float, BoundaryEdge]] = []
    curve_dx = curve.end[0] - curve.start[0]
    curve_dy = curve.end[1] - curve.start[1]
    curve_length_squared = curve_dx * curve_dx + curve_dy * curve_dy
    if curve_length_squared <= 0.0:
        raise FingertipMeshingError(f"Gmsh curve {curve.tag} has zero length")
    for element_type, connectivity in zip(element_types, connectivity_groups):
        name, dimension, order, number_of_nodes, _, _ = (
            gmsh.model.mesh.getElementProperties(element_type)
        )
        if dimension != 1 or order != 1 or number_of_nodes != 2:
            raise FingertipMeshingError(
                f"boundary topology must contain Line2 edges; Gmsh returned {name}"
            )
        for offset in range(0, len(connectivity), number_of_nodes):
            first_id, second_id = (
                int(connectivity[offset]),
                int(connectivity[offset + 1]),
            )
            first = coordinates[first_id]
            second = coordinates[second_id]
            edge_dx = second[0] - first[0]
            edge_dy = second[1] - first[1]
            if edge_dx * curve_dx + edge_dy * curve_dy < 0.0:
                first_id, second_id = second_id, first_id
                first, second = second, first
            midpoint = ((first[0] + second[0]) * 0.5, (first[1] + second[1]) * 0.5)
            parameter = (
                (midpoint[0] - curve.start[0]) * curve_dx
                + (midpoint[1] - curve.start[1]) * curve_dy
            ) / curve_length_squared
            edges.append(
                (parameter, BoundaryEdge((first_id, second_id), curve.domain))
            )
    return tuple(edge for _, edge in sorted(edges, key=lambda item: item[0]))


def _edge_line(
    edge: BoundaryEdge, coordinates: dict[int, tuple[float, float]]
) -> LineString:
    return LineString([coordinates[node_id] for node_id in edge.node_ids])


def _classify_boundaries(
    gmsh: Any,
    model: FingertipModel,
    domain_entities: tuple[_DomainEntities, _DomainEntities],
    coordinates: dict[int, tuple[float, float]],
    tolerance: float,
) -> dict[str, tuple[BoundaryEdge, ...]]:
    stable = model.boundaries.segments
    pad_tags = {
        "pad_bond_left",
        "pad_bond_right",
        "pad_cutout_left",
        "pad_cutout_right",
        "pad_cutout_bottom",
        "pad_outer_arc",
    }
    carrier_tags = {"stem_left", "stem_right", "stem_bottom"}
    classified: dict[str, list[BoundaryEdge]] = {
        **{tag: [] for tag in stable},
        "pad_void_unpaired": [],
        "rigid_link_outer": [],
        "rigid_bond_interface": [],
    }
    for entities in domain_entities:
        candidate_tags = pad_tags if entities.curves[0].domain == "pad" else carrier_tags
        for curve in entities.curves:
            for edge in _extract_curve_edges(gmsh, curve, coordinates):
                line = _edge_line(edge, coordinates)
                matches = [
                    tag
                    for tag in candidate_tags
                    if _line_is_on(stable[tag].geometry, line, tolerance)
                ]
                if len(matches) > 1:
                    raise FingertipMeshingError(
                        f"boundary edge {edge.node_ids} matches multiple semantic "
                        f"tags: {sorted(matches)}"
                    )
                if matches:
                    classified[matches[0]].append(edge)
                elif edge.domain == "pad":
                    classified["pad_void_unpaired"].append(edge)
                elif _line_is_on(
                    model.interface_definition.geometry, line, tolerance
                ):
                    classified["rigid_bond_interface"].append(edge)
                else:
                    classified["rigid_link_outer"].append(edge)
    return {tag: tuple(edges) for tag, edges in classified.items()}


def _boundary_geometry(
    edges: tuple[BoundaryEdge, ...], coordinates: dict[int, tuple[float, float]]
) -> BaseGeometry:
    if not edges:
        return MultiLineString([])
    return MultiLineString(
        [[coordinates[node_id] for node_id in edge.node_ids] for edge in edges]
    )


def _signed_double_area(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        third[0] - first[0]
    ) * (second[1] - first[1])




def generate_fingertip_mesh(
    model: FingertipModel, settings: MeshSettings
) -> FingertipMesh:
    """Mesh the model's Shapely pad/link domains with distinct Gmsh topology."""
    gmsh = _import_gmsh()
    gmsh.initialize(["phase4m"])
    try:
        gmsh.model.add(f"lit_fingertip_{settings.level}")
        pad_entities = _add_domain(
            gmsh,
            model.pad_material_geometry,
            "pad",
            settings.contact_boundary_target_size_mm,
            model.parameters.geometry_tolerance,
            _semantic_split_points(model, "pad"),
        )
        carrier_entities = _add_domain(
            gmsh,
            model.link_geometry,
            "rigid_carrier",
            settings.contact_boundary_target_size_mm,
            model.parameters.geometry_tolerance,
            _semantic_split_points(model, "rigid_carrier"),
        )
        gmsh.model.geo.synchronize()
        _configure_gmsh(
            gmsh, model, settings, (pad_entities, carrier_entities)
        )
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen")
        coordinates = _node_coordinates(gmsh)
        pad_elements = _extract_domain_elements(
            gmsh, pad_entities, "pad", coordinates
        )
        carrier_elements = _extract_domain_elements(
            gmsh, carrier_entities, "rigid_carrier", coordinates
        )
        pad_node_ids = {
            node_id for element in pad_elements for node_id in element.node_ids
        }
        carrier_node_ids = {
            node_id for element in carrier_elements for node_id in element.node_ids
        }
        if not pad_node_ids.isdisjoint(carrier_node_ids):
            shared = sorted(pad_node_ids.intersection(carrier_node_ids))
            raise FingertipMeshingError(
                "Gmsh merged pad and carrier topology; shared node IDs: "
                f"{shared[:20]}"
            )
        nodes = {
            node_id: MeshNode(
                node_id,
                coordinates[node_id][0],
                coordinates[node_id][1],
                "pad" if node_id in pad_node_ids else "rigid_carrier",
            )
            for node_id in sorted(pad_node_ids | carrier_node_ids)
        }
        boundaries = _classify_boundaries(
            gmsh,
            model,
            (pad_entities, carrier_entities),
            coordinates,
            settings.classification_tolerance_mm,
        )
        contacts = tuple(
            MeshedContactPair(
                name=pair.name,
                pad_boundary_tag=pair.pad_boundary.name,
                stem_boundary_tag=pair.stem_boundary.name,
                initial_normal_gap_mm=float(pair.initial_normal_gap),
                measured_mesh_gap_mm=float(
                    _boundary_geometry(
                        boundaries[pair.pad_boundary.name], coordinates
                    ).distance(
                        _boundary_geometry(
                            boundaries[pair.stem_boundary.name], coordinates
                        )
                    )
                ),
            )
            for pair in model.contact_pairs
        )
        quality = mesh_quality_statistics(
            nodes, pad_elements, carrier_elements, model
        )
        placeholder_validation = MeshValidationReport(False, {}, ())
        mesh = FingertipMesh(
            nodes=nodes,
            pad_elements=pad_elements,
            carrier_elements=carrier_elements,
            domain_node_ids={
                "pad": tuple(sorted(pad_node_ids)),
                "rigid_carrier": tuple(sorted(carrier_node_ids)),
            },
            domain_element_ids={
                "pad": tuple(element.id for element in pad_elements),
                "rigid_carrier": tuple(
                    element.id for element in carrier_elements
                ),
            },
            boundary_edges=boundaries,
            contact_pairs=contacts,
            parameters=model.parameters,
            settings=settings,
            quality=quality,
            validation=placeholder_validation,
            gmsh_version=str(gmsh.option.getString("General.Version")),
        )
        report = validate_fingertip_mesh(mesh, model)
        return replace(mesh, validation=report)
    finally:
        gmsh.finalize()
