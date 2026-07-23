"""Solver-independent rigid circular indenter fixture for Phase 4I.

The indenter is deliberately owned by the FEM layer.  The fingertip geometry
continues to come exclusively from :class:`model.fingertip_model.FingertipModel`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from shapely.geometry import LineString, MultiLineString, Point, Polygon

from model.fingertip_model import FingertipModel

Vector2 = tuple[float, float]


class InvalidIndenterSettings(ValueError):
    """Raised when the rigid fixture settings are not physically usable."""


class IndenterMeshingError(RuntimeError):
    """Raised when Gmsh cannot produce the required rigid carrier topology."""


@dataclass(frozen=True)
class IndenterSettings:
    """Rigid circular fixture settings, with all lengths in millimeters."""

    radius_mm: float = 4.0
    thickness_mm: float = 1.0
    initial_gap_mm: float = 0.0
    contact_half_angle_degrees: float = 60.0
    geometry_resolution: int = 128

    def __post_init__(self) -> None:
        dimensions = {
            "radius_mm": self.radius_mm,
            "thickness_mm": self.thickness_mm,
            "initial_gap_mm": self.initial_gap_mm,
            "contact_half_angle_degrees": self.contact_half_angle_degrees,
        }
        if not all(math.isfinite(value) for value in dimensions.values()):
            raise InvalidIndenterSettings("all indenter settings must be finite")
        if self.radius_mm <= 0.0:
            raise InvalidIndenterSettings("radius_mm must be positive")
        if self.thickness_mm <= 0.0:
            raise InvalidIndenterSettings("thickness_mm must be positive")
        if self.initial_gap_mm < 0.0:
            raise InvalidIndenterSettings("initial_gap_mm must be nonnegative")
        if not 0.0 < self.contact_half_angle_degrees < 90.0:
            raise InvalidIndenterSettings(
                "contact_half_angle_degrees must lie between 0 and 90"
            )
        if (
            not isinstance(self.geometry_resolution, int)
            or isinstance(self.geometry_resolution, bool)
            or self.geometry_resolution < 32
        ):
            raise InvalidIndenterSettings(
                "geometry_resolution must be an integer of at least 32"
            )


@dataclass(frozen=True)
class CrownFrame:
    """Local orthonormal frame derived from the actual pad outer boundary."""

    point_mm: Vector2
    tangent: Vector2
    pad_outward_normal: Vector2
    loading_direction: Vector2
    arc_distance_mm: float


@dataclass(frozen=True)
class IndenterFixture:
    """Analytic fixture geometry positioned from a :class:`FingertipModel`."""

    settings: IndenterSettings
    frame: CrownFrame
    center_mm: Vector2
    contact_direction: Vector2
    carrier_geometry: Polygon
    contact_arc: LineString
    outer_remainder: MultiLineString

    def displacement_for_travel(self, travel_mm: float) -> Vector2:
        """Return the prescribed rigid translation for a positive travel."""
        if not math.isfinite(travel_mm) or travel_mm < 0.0:
            raise InvalidIndenterSettings(
                "indentation travel must be finite and nonnegative"
            )
        return (
            travel_mm * self.frame.loading_direction[0],
            travel_mm * self.frame.loading_direction[1],
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the geometry contract without Gmsh or Kratos objects."""
        return {
            "settings": asdict(self.settings),
            "crown_point_mm": list(self.frame.point_mm),
            "crown_arc_distance_mm": self.frame.arc_distance_mm,
            "crown_tangent": list(self.frame.tangent),
            "pad_outward_normal": list(self.frame.pad_outward_normal),
            "loading_direction": list(self.frame.loading_direction),
            "contact_direction": list(self.contact_direction),
            "center_mm": list(self.center_mm),
            "pad_contact_arc_minimum_distance_mm": self.contact_arc.distance(
                Point(self.frame.point_mm)
            ),
        }


@dataclass(frozen=True)
class IndenterMeshNode:
    """One local node in the separately meshed indenter carrier."""

    id: int
    x_mm: float
    y_mm: float


@dataclass(frozen=True)
class IndenterT3:
    """One counter-clockwise T3 in the rigid circular carrier."""

    id: int
    node_ids: tuple[int, int, int]


@dataclass(frozen=True)
class IndenterBoundaryEdge:
    """A CCW circle edge, so material is left and its right normal is outward."""

    node_ids: tuple[int, int]


@dataclass(frozen=True)
class IndenterMesh:
    """Gmsh topology for the independent rigid fixture domain."""

    nodes: dict[int, IndenterMeshNode]
    elements: tuple[IndenterT3, ...]
    contact_edges: tuple[IndenterBoundaryEdge, ...]
    remainder_edges: tuple[IndenterBoundaryEdge, ...]
    gmsh_version: str
    minimum_triangle_angle_degrees: float
    maximum_contact_edge_length_mm: float
    target_size_mm: float

    @property
    def contact_node_ids(self) -> tuple[int, ...]:
        return tuple(
            sorted({node_id for edge in self.contact_edges for node_id in edge.node_ids})
        )

    @property
    def node_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.nodes))


def _normalized(vector: Vector2) -> Vector2:
    length = math.hypot(*vector)
    if not math.isfinite(length) or length <= 0.0:
        raise InvalidIndenterSettings("cannot normalize a zero-length vector")
    return vector[0] / length, vector[1] / length


def crown_frame_from_model(model: FingertipModel) -> CrownFrame:
    """Derive the central crown, tangent, and outward normal from Shapely."""
    arc = model.boundaries.segments["pad_outer_arc"].geometry
    intersection = arc.intersection(model.symmetry_axis)
    points: list[Point]
    if isinstance(intersection, Point):
        points = [intersection]
    elif hasattr(intersection, "geoms"):
        points = [geometry for geometry in intersection.geoms if isinstance(geometry, Point)]
    else:
        points = []
    if not points:
        raise InvalidIndenterSettings(
            "pad_outer_arc does not intersect the model symmetry axis"
        )
    crown = min(points, key=lambda point: (point.y, abs(point.x)))
    distance = float(arc.project(crown))
    sample_distance = max(1.0e-6 * arc.length, 100.0 * model.parameters.geometry_tolerance)
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
        raise InvalidIndenterSettings(
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
    """Return the local surface frame while retaining the global load direction.

    ``normalized_location`` follows the native ``pad_outer_arc`` orientation:
    zero is the right bonded endpoint, one half is the crown, and one is the
    left bonded endpoint.  Only the placement frame rotates with location.
    The loading direction is always copied from the central Phase 4J frame.
    """
    if (
        not math.isfinite(normalized_location)
        or normalized_location < 0.0
        or normalized_location > 1.0
    ):
        raise InvalidIndenterSettings(
            "normalized contact location must be finite and lie in [0, 1]"
        )
    arc = model.boundaries.segments["pad_outer_arc"].geometry
    distance = normalized_location * float(arc.length)
    point = arc.interpolate(distance)
    sample_distance = max(
        1.0e-6 * arc.length,
        100.0 * model.parameters.geometry_tolerance,
    )
    before = arc.interpolate(max(0.0, distance - sample_distance))
    after = arc.interpolate(min(arc.length, distance + sample_distance))
    tangent = _normalized((after.x - before.x, after.y - before.y))
    candidates = ((-tangent[1], tangent[0]), (tangent[1], -tangent[0]))
    probe_distance = max(
        1.0e-4, 1000.0 * model.parameters.geometry_tolerance
    )
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


def _sample_circle_arc(
    center: Vector2,
    radius: float,
    start_angle: float,
    end_angle: float,
    resolution: int,
) -> LineString:
    estimated = int(
        math.ceil(resolution * (end_angle - start_angle) / (2.0 * math.pi))
    )
    # An even number keeps the arc midpoint as an exact sampled vertex.  The
    # contact arc therefore contains the crown-aligned tangency point instead
    # of approximating it with a chord.
    count = max(2, 2 * int(math.ceil(estimated / 2.0)))
    return LineString(
        [
            (
                center[0] + radius * math.cos(start_angle + (end_angle - start_angle) * index / count),
                center[1] + radius * math.sin(start_angle + (end_angle - start_angle) * index / count),
            )
            for index in range(count + 1)
        ]
    )


def build_indenter_fixture(
    model: FingertipModel,
    settings: IndenterSettings | None = None,
) -> IndenterFixture:
    """Position a circular rigid fixture from the model's actual crown frame."""
    resolved = settings or IndenterSettings()
    frame = crown_frame_from_model(model)
    center = (
        frame.point_mm[0] + (resolved.radius_mm + resolved.initial_gap_mm) * frame.pad_outward_normal[0],
        frame.point_mm[1] + (resolved.radius_mm + resolved.initial_gap_mm) * frame.pad_outward_normal[1],
    )
    contact_direction_angle = math.atan2(
        frame.loading_direction[1], frame.loading_direction[0]
    )
    half_angle = math.radians(resolved.contact_half_angle_degrees)
    contact_arc = _sample_circle_arc(
        center,
        resolved.radius_mm,
        contact_direction_angle - half_angle,
        contact_direction_angle + half_angle,
        resolved.geometry_resolution,
    )
    remainder_arc = _sample_circle_arc(
        center,
        resolved.radius_mm,
        contact_direction_angle + half_angle,
        contact_direction_angle + 2.0 * math.pi - half_angle,
        resolved.geometry_resolution,
    )
    carrier = Point(center).buffer(
        resolved.radius_mm, quad_segs=resolved.geometry_resolution
    )
    if not isinstance(carrier, Polygon) or not carrier.is_valid:
        raise InvalidIndenterSettings("failed to construct the circular carrier")
    fixture = IndenterFixture(
        settings=resolved,
        frame=frame,
        center_mm=center,
        contact_direction=frame.loading_direction,
        carrier_geometry=carrier,
        contact_arc=contact_arc,
        outer_remainder=MultiLineString([remainder_arc.coords]),
    )
    expected_center = (
        frame.point_mm[0] + (resolved.radius_mm + resolved.initial_gap_mm) * frame.pad_outward_normal[0],
        frame.point_mm[1] + (resolved.radius_mm + resolved.initial_gap_mm) * frame.pad_outward_normal[1],
    )
    if math.dist(center, expected_center) > model.parameters.geometry_tolerance:
        raise InvalidIndenterSettings("indenter center does not satisfy the crown contract")
    geometric_gap = model.boundaries.segments["pad_outer_arc"].geometry.distance(
        contact_arc
    )
    if abs(geometric_gap - resolved.initial_gap_mm) > max(
        1.0e-8, 10.0 * model.parameters.geometry_tolerance
    ):
        raise InvalidIndenterSettings(
            f"indenter geometric gap {geometric_gap:g} does not reproduce "
            f"{resolved.initial_gap_mm:g} mm"
        )
    return fixture


def build_indenter_fixture_at_location(
    model: FingertipModel,
    normalized_location: float,
    settings: IndenterSettings | None = None,
) -> IndenterFixture:
    """Place the fixture at one reference-arc location with global loading.

    The circle is tangent to the local undeformed pad surface, but prescribed
    travel remains parallel to the central Phase 4J loading direction.
    """
    resolved = settings or IndenterSettings()
    frame = surface_frame_from_normalized_location(model, normalized_location)
    center = (
        frame.point_mm[0]
        + (resolved.radius_mm + resolved.initial_gap_mm)
        * frame.pad_outward_normal[0],
        frame.point_mm[1]
        + (resolved.radius_mm + resolved.initial_gap_mm)
        * frame.pad_outward_normal[1],
    )
    contact_direction = (
        -frame.pad_outward_normal[0],
        -frame.pad_outward_normal[1],
    )
    contact_direction_angle = math.atan2(
        contact_direction[1], contact_direction[0]
    )
    half_angle = math.radians(resolved.contact_half_angle_degrees)
    contact_arc = _sample_circle_arc(
        center,
        resolved.radius_mm,
        contact_direction_angle - half_angle,
        contact_direction_angle + half_angle,
        resolved.geometry_resolution,
    )
    remainder_arc = _sample_circle_arc(
        center,
        resolved.radius_mm,
        contact_direction_angle + half_angle,
        contact_direction_angle + 2.0 * math.pi - half_angle,
        resolved.geometry_resolution,
    )
    carrier = Point(center).buffer(
        resolved.radius_mm, quad_segs=resolved.geometry_resolution
    )
    if not isinstance(carrier, Polygon) or not carrier.is_valid:
        raise InvalidIndenterSettings("failed to construct the circular carrier")
    fixture = IndenterFixture(
        settings=resolved,
        frame=frame,
        center_mm=center,
        contact_direction=contact_direction,
        carrier_geometry=carrier,
        contact_arc=contact_arc,
        outer_remainder=MultiLineString([remainder_arc.coords]),
    )
    target = Point(frame.point_mm)
    target_gap = fixture.contact_arc.distance(target)
    if abs(target_gap - resolved.initial_gap_mm) > max(
        1.0e-8, 10.0 * model.parameters.geometry_tolerance
    ):
        raise InvalidIndenterSettings(
            f"target-point gap {target_gap:g} does not reproduce "
            f"{resolved.initial_gap_mm:g} mm"
        )
    if resolved.initial_gap_mm == 0.0 and not fixture.contact_arc.distance(target) <= max(
        1.0e-8, 10.0 * model.parameters.geometry_tolerance
    ):
        raise InvalidIndenterSettings("contact arc does not contain its target point")
    return fixture


def _import_gmsh() -> Any:
    try:
        import gmsh
    except (ImportError, OSError) as exception:
        raise IndenterMeshingError(
            "Phase 4I indenter meshing requires the Gmsh Python API"
        ) from exception
    return gmsh


def _triangle_minimum_angle(points: tuple[Vector2, Vector2, Vector2]) -> float:
    lengths = [
        math.dist(points[(index + 1) % 3], points[(index + 2) % 3])
        for index in range(3)
    ]
    angles = []
    for index, opposite in enumerate(lengths):
        first = lengths[(index + 1) % 3]
        second = lengths[(index + 2) % 3]
        cosine = (first * first + second * second - opposite * opposite) / (
            2.0 * first * second
        )
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    return min(angles)


def generate_indenter_mesh(
    fixture: IndenterFixture,
    target_size_mm: float,
) -> IndenterMesh:
    """Mesh the circular carrier with Gmsh and preserve two boundary groups."""
    if not math.isfinite(target_size_mm) or target_size_mm <= 0.0:
        raise IndenterMeshingError("target_size_mm must be finite and positive")
    gmsh = _import_gmsh()
    gmsh.initialize(["phase4i_indenter"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", 1)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1)
        gmsh.option.setNumber("Mesh.RandomFactor", 0.0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", target_size_mm)
        gmsh.option.setNumber("Mesh.MeshSizeMax", target_size_mm)
        gmsh.model.add("phase4i_rigid_indenter")

        center = fixture.center_mm
        contact_angle = math.atan2(
            fixture.contact_direction[1], fixture.contact_direction[0]
        )
        half_angle = math.radians(fixture.settings.contact_half_angle_degrees)
        angles = (
            contact_angle - half_angle,
            contact_angle,
            contact_angle + half_angle,
            contact_angle + math.pi,
        )
        center_tag = gmsh.model.geo.addPoint(center[0], center[1], 0.0, target_size_mm)
        point_tags = [
            gmsh.model.geo.addPoint(
                center[0] + fixture.settings.radius_mm * math.cos(angle),
                center[1] + fixture.settings.radius_mm * math.sin(angle),
                0.0,
                target_size_mm,
            )
            for angle in angles
        ]
        curve_tags = (
            gmsh.model.geo.addCircleArc(point_tags[0], center_tag, point_tags[1]),
            gmsh.model.geo.addCircleArc(point_tags[1], center_tag, point_tags[2]),
            gmsh.model.geo.addCircleArc(point_tags[2], center_tag, point_tags[3]),
            gmsh.model.geo.addCircleArc(point_tags[3], center_tag, point_tags[0]),
        )
        loop = gmsh.model.geo.addCurveLoop(list(curve_tags))
        surface = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
        arc_spans = (half_angle, half_angle, math.pi - half_angle, math.pi - half_angle)
        for curve_tag, span in zip(curve_tags, arc_spans):
            count = int(math.ceil(fixture.settings.radius_mm * span / target_size_mm)) + 1
            gmsh.model.mesh.setTransfiniteCurve(curve_tag, max(2, count))
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen")

        node_tags, flattened, _ = gmsh.model.mesh.getNodes()
        coordinates = {
            int(tag): (float(flattened[3 * index]), float(flattened[3 * index + 1]))
            for index, tag in enumerate(node_tags)
        }
        nodes = {
            node_id: IndenterMeshNode(node_id, point[0], point[1])
            for node_id, point in coordinates.items()
        }
        elements: list[IndenterT3] = []
        element_types, element_tag_groups, connectivity_groups = gmsh.model.mesh.getElements(2, surface)
        minimum_angle = math.inf
        for element_type, tags, connectivity in zip(
            element_types, element_tag_groups, connectivity_groups
        ):
            name, dimension, order, number_of_nodes, _, _ = gmsh.model.mesh.getElementProperties(element_type)
            if dimension != 2 or order != 1 or number_of_nodes != 3:
                raise IndenterMeshingError(f"expected T3 elements, Gmsh returned {name}")
            for index, element_tag in enumerate(tags):
                offset = index * 3
                node_ids = tuple(int(value) for value in connectivity[offset : offset + 3])
                points = tuple(coordinates[node_id] for node_id in node_ids)
                signed_area = (
                    (points[1][0] - points[0][0]) * (points[2][1] - points[0][1])
                    - (points[2][0] - points[0][0]) * (points[1][1] - points[0][1])
                )
                if signed_area < 0.0:
                    node_ids = (node_ids[0], node_ids[2], node_ids[1])
                    points = (points[0], points[2], points[1])
                if signed_area == 0.0:
                    raise IndenterMeshingError("indenter mesh contains a zero-area T3")
                minimum_angle = min(minimum_angle, _triangle_minimum_angle(points))
                elements.append(IndenterT3(int(element_tag), node_ids))

        def extract_edges(tags: tuple[int, ...]) -> tuple[IndenterBoundaryEdge, ...]:
            records: list[tuple[float, IndenterBoundaryEdge]] = []
            start = contact_angle - half_angle
            for curve_tag in tags:
                edge_types, _, groups = gmsh.model.mesh.getElements(1, curve_tag)
                for edge_type, connectivity in zip(edge_types, groups):
                    name, dimension, order, count, _, _ = gmsh.model.mesh.getElementProperties(edge_type)
                    if dimension != 1 or order != 1 or count != 2:
                        raise IndenterMeshingError(f"expected Line2 edges, Gmsh returned {name}")
                    for offset in range(0, len(connectivity), 2):
                        first_id, second_id = int(connectivity[offset]), int(connectivity[offset + 1])
                        first, second = coordinates[first_id], coordinates[second_id]
                        midpoint = ((first[0] + second[0]) * 0.5, (first[1] + second[1]) * 0.5)
                        dx, dy = second[0] - first[0], second[1] - first[1]
                        radial = (midpoint[0] - center[0], midpoint[1] - center[1])
                        if dy * radial[0] - dx * radial[1] < 0.0:
                            first_id, second_id = second_id, first_id
                        angle = math.atan2(radial[1], radial[0])
                        while angle < start:
                            angle += 2.0 * math.pi
                        records.append((angle, IndenterBoundaryEdge((first_id, second_id))))
            return tuple(edge for _, edge in sorted(records, key=lambda item: item[0]))

        contact_edges = extract_edges(curve_tags[:2])
        remainder_edges = extract_edges(curve_tags[2:])
        maximum_contact_edge = max(
            math.dist(
                coordinates[edge.node_ids[0]], coordinates[edge.node_ids[1]]
            )
            for edge in contact_edges
        )
        outward_average = [0.0, 0.0]
        for edge in contact_edges:
            first, second = (coordinates[node_id] for node_id in edge.node_ids)
            outward_average[0] += second[1] - first[1]
            outward_average[1] -= second[0] - first[0]
        outward_average_vector = _normalized((outward_average[0], outward_average[1]))
        normal_dot = sum(
            outward_average_vector[index] * fixture.contact_direction[index]
            for index in range(2)
        )
        if normal_dot <= 0.7:
            raise IndenterMeshingError(
                "IndenterContactArc does not have an outward normal toward the pad"
            )
        return IndenterMesh(
            nodes=nodes,
            elements=tuple(sorted(elements, key=lambda item: item.id)),
            contact_edges=contact_edges,
            remainder_edges=remainder_edges,
            gmsh_version=str(gmsh.option.getString("General.Version")),
            minimum_triangle_angle_degrees=minimum_angle,
            maximum_contact_edge_length_mm=maximum_contact_edge,
            target_size_mm=target_size_mm,
        )
    finally:
        gmsh.finalize()
