"""Mesh quality and source-geometry validation."""

from __future__ import annotations

import math

from shapely.geometry import LineString, MultiLineString, MultiPoint, Point
from shapely.geometry.base import BaseGeometry

from mesh.types import (
    BoundaryEdge,
    FingertipMesh,
    MeshNode,
    MeshQualityStatistics,
    MeshValidationReport,
    T3Element,
)
from model.fingertip_model import FingertipModel, PolygonalGeometry


def _line_is_on(source: BaseGeometry, candidate: LineString, tolerance: float) -> bool:
    return bool(source.buffer(tolerance, cap_style=1).covers(candidate))


def _edge_line(
    edge: BoundaryEdge, coordinates: dict[int, tuple[float, float]]
) -> LineString:
    return LineString([coordinates[node_id] for node_id in edge.node_ids])


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


def _triangle_angles_and_lengths(
    points: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    lengths = tuple(
        math.dist(points[(index + 1) % 3], points[(index + 2) % 3])
        for index in range(3)
    )
    angles: list[float] = []
    for index in range(3):
        adjacent_first = lengths[(index + 1) % 3]
        adjacent_second = lengths[(index + 2) % 3]
        opposite = lengths[index]
        denominator = 2.0 * adjacent_first * adjacent_second
        cosine = (
            adjacent_first**2 + adjacent_second**2 - opposite**2
        ) / denominator
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    return (angles[0], angles[1], angles[2]), lengths


def mesh_quality_statistics(
    nodes: dict[int, MeshNode],
    pad_elements: tuple[T3Element, ...],
    carrier_elements: tuple[T3Element, ...],
    model: FingertipModel,
) -> MeshQualityStatistics:
    elements = (*pad_elements, *carrier_elements)
    used_node_ids: set[int] = set()
    signatures: set[tuple[int, int, int]] = set()
    duplicate_count = 0
    nonpositive_count = 0
    pad_area = 0.0
    carrier_area = 0.0
    minimum_angle = math.inf
    minimum_angle_element_id = -1
    minimum_angle_centroid = (math.nan, math.nan)
    maximum_edge_length = 0.0
    for element in elements:
        used_node_ids.update(element.node_ids)
        signature = tuple(sorted(element.node_ids))
        if signature in signatures:
            duplicate_count += 1
        signatures.add(signature)
        points = tuple(
            (nodes[node_id].x_mm, nodes[node_id].y_mm)
            for node_id in element.node_ids
        )
        double_area = _signed_double_area(*points)
        if double_area <= 0.0:
            nonpositive_count += 1
        area = 0.5 * double_area
        if element.domain == "pad":
            pad_area += area
        else:
            carrier_area += area
        angles, lengths = _triangle_angles_and_lengths(points)
        element_minimum_angle = min(angles)
        if element_minimum_angle < minimum_angle:
            minimum_angle = element_minimum_angle
            minimum_angle_element_id = element.id
            minimum_angle_centroid = (
                sum(point[0] for point in points) / 3.0,
                sum(point[1] for point in points) / 3.0,
            )
        maximum_edge_length = max(maximum_edge_length, *lengths)
    pad_geometry_area = float(model.pad_material_geometry.area)
    carrier_geometry_area = float(model.link_geometry.area)
    pad_nodes = {node_id for element in pad_elements for node_id in element.node_ids}
    carrier_nodes = {
        node_id for element in carrier_elements for node_id in element.node_ids
    }
    return MeshQualityStatistics(
        node_count=len(nodes),
        pad_node_count=len(pad_nodes),
        carrier_node_count=len(carrier_nodes),
        t3_element_count=len(elements),
        pad_t3_element_count=len(pad_elements),
        carrier_t3_element_count=len(carrier_elements),
        minimum_triangle_angle_degrees=minimum_angle,
        minimum_triangle_angle_element_id=minimum_angle_element_id,
        minimum_triangle_angle_centroid_mm=minimum_angle_centroid,
        maximum_edge_length_mm=maximum_edge_length,
        pad_mesh_area_mm2=pad_area,
        pad_geometry_area_mm2=pad_geometry_area,
        pad_area_relative_error=abs(pad_area - pad_geometry_area)
        / pad_geometry_area,
        carrier_mesh_area_mm2=carrier_area,
        carrier_geometry_area_mm2=carrier_geometry_area,
        carrier_area_relative_error=abs(carrier_area - carrier_geometry_area)
        / carrier_geometry_area,
        orphan_node_count=len(set(nodes) - used_node_ids),
        duplicate_element_count=duplicate_count,
        nonpositive_area_element_count=nonpositive_count,
    )


def _edge_orientation_is_outward(
    edge: BoundaryEdge,
    coordinates: dict[int, tuple[float, float]],
    domain_geometry: PolygonalGeometry,
    tolerance: float,
) -> bool:
    first, second = (coordinates[node_id] for node_id in edge.node_ids)
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    length = math.hypot(dx, dy)
    if length <= tolerance:
        return False
    midpoint = ((first[0] + second[0]) * 0.5, (first[1] + second[1]) * 0.5)
    probe = min(1.0e-4, max(100.0 * tolerance, 1.0e-4 * length))
    left = Point(
        midpoint[0] - dy / length * probe,
        midpoint[1] + dx / length * probe,
    )
    right = Point(
        midpoint[0] + dy / length * probe,
        midpoint[1] - dx / length * probe,
    )
    return bool(domain_geometry.covers(left) and not domain_geometry.contains(right))


def validate_fingertip_mesh(
    mesh: FingertipMesh, model: FingertipModel
) -> MeshValidationReport:
    """Validate topology, source-geometry membership, gaps, and mesh quality."""
    tolerance = mesh.settings.classification_tolerance_mm
    quality = mesh.quality
    checks: dict[str, bool] = {
        "positive_t3_reference_area": quality.nonpositive_area_element_count == 0,
        "no_orphan_nodes": quality.orphan_node_count == 0,
        "no_duplicate_elements": quality.duplicate_element_count == 0,
        "pad_area_relative_error_below_0_1_percent": (
            quality.pad_area_relative_error < 1.0e-3
        ),
        "carrier_area_relative_error_below_0_1_percent": (
            quality.carrier_area_relative_error < 1.0e-3
        ),
        "minimum_angle_at_least_target": (
            quality.minimum_triangle_angle_degrees
            >= mesh.settings.minimum_angle_target_degrees
        ),
        "pad_and_carrier_node_ids_disjoint": set(
            mesh.domain_node_ids["pad"]
        ).isdisjoint(mesh.domain_node_ids["rigid_carrier"]),
    }

    stable_segments = model.boundaries.segments
    semantic_membership = True
    semantic_coverage = True
    semantic_nonempty = True
    no_semantic_overlap = True
    outward_orientation = True
    pad_semantic_tags = {
        "pad_bond_left",
        "pad_bond_right",
        "pad_cutout_left",
        "pad_cutout_right",
        "pad_cutout_bottom",
        "pad_outer_arc",
    }
    carrier_semantic_tags = {"stem_left", "stem_right", "stem_bottom"}
    coordinates = {
        node_id: (node.x_mm, node.y_mm) for node_id, node in mesh.nodes.items()
    }
    for tag, source in stable_segments.items():
        edges = mesh.boundary_edges[tag]
        semantic_nonempty = semantic_nonempty and bool(edges)
        semantic_membership = semantic_membership and all(
            _line_is_on(source.geometry, _edge_line(edge, coordinates), tolerance)
            for edge in edges
        )
        semantic_coverage = semantic_coverage and (
            _boundary_geometry(edges, coordinates).hausdorff_distance(source.geometry)
            <= tolerance
        )
    for tag, edges in mesh.boundary_edges.items():
        for edge in edges:
            candidate_tags = (
                pad_semantic_tags if edge.domain == "pad" else carrier_semantic_tags
            )
            matches = [
                candidate
                for candidate in candidate_tags
                if _line_is_on(
                    stable_segments[candidate].geometry,
                    _edge_line(edge, coordinates),
                    tolerance,
                )
            ]
            no_semantic_overlap = no_semantic_overlap and len(matches) <= 1
            domain_geometry = (
                model.pad_material_geometry
                if edge.domain == "pad"
                else model.link_geometry
            )
            outward_orientation = outward_orientation and _edge_orientation_is_outward(
                edge, coordinates, domain_geometry, tolerance
            )
    checks["all_semantic_boundary_tags_nonempty"] = semantic_nonempty
    checks["semantic_edges_lie_on_source_segments"] = semantic_membership
    checks["semantic_mesh_covers_complete_source_segments"] = semantic_coverage
    checks["no_edge_has_multiple_semantic_tags"] = no_semantic_overlap
    checks["boundary_edges_have_outward_orientation"] = outward_orientation

    contact_gap_ok = True
    zero_clearance_distinct = True
    zero_clearance_coincident = True
    zero_clearance_node_coordinates_coincident = True
    for contact in mesh.contact_pairs:
        contact_gap_ok = contact_gap_ok and abs(
            contact.measured_mesh_gap_mm - contact.initial_normal_gap_mm
        ) <= tolerance
        pad_edges = mesh.boundary_edges[contact.pad_boundary_tag]
        stem_edges = mesh.boundary_edges[contact.stem_boundary_tag]
        pad_ids = {node_id for edge in pad_edges for node_id in edge.node_ids}
        stem_ids = {node_id for edge in stem_edges for node_id in edge.node_ids}
        zero_clearance_distinct = zero_clearance_distinct and pad_ids.isdisjoint(
            stem_ids
        )
        if contact.initial_normal_gap_mm <= tolerance:
            pad_geometry = _boundary_geometry(pad_edges, coordinates)
            stem_geometry = _boundary_geometry(stem_edges, coordinates)
            zero_clearance_coincident = zero_clearance_coincident and (
                pad_geometry.hausdorff_distance(stem_geometry) <= tolerance
            )
            pad_points = MultiPoint([coordinates[node_id] for node_id in pad_ids])
            stem_points = MultiPoint([coordinates[node_id] for node_id in stem_ids])
            zero_clearance_node_coordinates_coincident = (
                zero_clearance_node_coordinates_coincident
                and len(pad_ids) == len(stem_ids)
                and pad_points.hausdorff_distance(stem_points)
                <= model.parameters.geometry_tolerance
            )
    checks["contact_gap_matches_model"] = contact_gap_ok
    checks["contact_node_ids_are_distinct"] = zero_clearance_distinct
    checks["zero_clearance_mesh_surfaces_are_coincident"] = (
        zero_clearance_coincident
    )
    checks["zero_clearance_node_coordinates_are_coincident"] = (
        zero_clearance_node_coordinates_coincident
    )

    errors = tuple(name for name, passed in checks.items() if not passed)
    return MeshValidationReport(not errors, checks, errors)
