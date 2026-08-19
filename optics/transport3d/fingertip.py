"""Production adapter from a fingertip volume state to FULL_3D OptiX."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from shapely.geometry import LineString, Point

from mesh.volume_state import FingertipVolumeState
from mesh.types import FingertipMesh
from model.fingertip import Fingertip

from .geometry import (
    AIR_INTERFACE,
    INTERNAL_INTERFACE,
    TriangleSurface,
    Transport3DGeometryError,
    build_fixed_transport_surfaces,
    build_full3d_transport_geometry,
    _surface_normals,
)


_LATERAL_SURFACE_PREFIX = "longitudinal_end_"
_EXTERNAL_VOLUME_TAGS = {
    "outer_compliant_left",
    "outer_compliant_arc",
    "outer_compliant_other",
    "outer_compliant_right",
}


def _external_reference_path(tip: Fingertip) -> LineString:
    boundaries = tip.geometry.boundaries
    left = list(boundaries.pad_outer_left.geometry.coords)
    arc = list(boundaries.pad_outer_arc.geometry.coords)[::-1]
    right = list(boundaries.pad_outer_right.geometry.coords)
    coordinates: list[tuple[float, float]] = []
    for segment in (left, arc, right):
        for coordinate in segment:
            point = (float(coordinate[0]), float(coordinate[1]))
            if not coordinates or point != coordinates[-1]:
                coordinates.append(point)
    path = LineString(coordinates)
    if path.is_empty or path.length <= 0.0:
        raise Transport3DGeometryError("authoritative fingertip external boundary has zero length")
    return path


def _surface_u_values(
    tip: Fingertip,
    state: FingertipVolumeState,
    node_ids: tuple[int, ...],
) -> dict[int, float]:
    path = _external_reference_path(tip)
    node_index = {node_id: index for index, node_id in enumerate(state.source_node_ids)}
    result: dict[int, float] = {}
    tolerance = max(1.0e-6, 100.0 * tip.parameters.geometry_tolerance)
    for node_id in node_ids:
        coordinate = state.reference_coordinates_mm[node_index[node_id], :2]
        point = Point(float(coordinate[0]), float(coordinate[1]))
        if path.distance(point) <= tolerance:
            result[node_id] = float(path.project(point, normalized=True))
    return result


def _oriented_surface_faces(
    state: FingertipVolumeState,
    rows: list[tuple[str, Any]],
) -> np.ndarray:
    """Orient semantic lateral faces consistently without changing topology."""
    node_ids = tuple(
        sorted({int(node_id) for _, triangle in rows for node_id in triangle.node_ids})
    )
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    faces = np.asarray(
        [
            [node_index[int(node_id)] for node_id in triangle.node_ids]
            for _, triangle in rows
        ],
        dtype=np.int64,
    )
    edge_records: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_index, face in enumerate(faces):
        for first, second in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            key = min(first, second), max(first, second)
            direction = 1 if (first, second) == key else -1
            edge_records.setdefault(key, []).append((face_index, direction))
    adjacency: list[list[tuple[int, int, int]]] = [[] for _ in faces]
    for values in edge_records.values():
        if len(values) > 2:
            raise Transport3DGeometryError("semantic lateral surface has a non-manifold edge")
        if len(values) == 2:
            (first, first_direction), (second, second_direction) = values
            adjacency[first].append((second, first_direction, second_direction))
            adjacency[second].append((first, second_direction, first_direction))

    reference = state.reference_coordinates_mm
    reference_center = np.mean(reference, axis=0)
    flips = np.full(len(faces), -1, dtype=np.int8)
    for start in range(len(faces)):
        if flips[start] >= 0:
            continue
        flips[start] = 0
        component: list[int] = []
        pending = [start]
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbor, current_direction, neighbor_direction in adjacency[current]:
                expected = int(flips[current]) ^ int(current_direction == neighbor_direction)
                if flips[neighbor] < 0:
                    flips[neighbor] = expected
                    pending.append(neighbor)
                elif int(flips[neighbor]) != expected:
                    raise Transport3DGeometryError(
                        "semantic lateral surface cannot be consistently oriented"
                    )
        component_faces = faces[component].copy()
        component_faces[flips[component] == 1, 1], component_faces[flips[component] == 1, 2] = (
            component_faces[flips[component] == 1, 2],
            component_faces[flips[component] == 1, 1].copy(),
        )
        reference_surface = reference[
            np.asarray([node_index[node_id] for node_id in node_ids], dtype=np.int64)
        ]
        points = reference_surface[component_faces]
        normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        if np.any(np.linalg.norm(normals, axis=1) <= 1.0e-12):
            raise Transport3DGeometryError("semantic lateral surface contains a degenerate triangle")
        outward_votes = np.sum(
            np.sum(normals * (np.mean(points, axis=1) - reference_center), axis=1) < 0.0
        )
        if int(outward_votes) > len(component) // 2:
            flips[component] ^= 1

    faces[flips == 1, 1], faces[flips == 1, 2] = (
        faces[flips == 1, 2],
        faces[flips == 1, 1].copy(),
    )
    return faces.astype(np.uint32)


def _silicone_surface(tip: Fingertip, state: FingertipVolumeState) -> TriangleSurface:
    node_index = {node_id: index for index, node_id in enumerate(state.source_node_ids)}
    rows = [
        (tag, triangle)
        for tag, triangles in sorted(state.surface_triangles.items())
        if not tag.startswith(_LATERAL_SURFACE_PREFIX)
        for triangle in triangles
    ]
    if not rows:
        raise Transport3DGeometryError("FingertipVolumeState has no lateral optical surface triangles")
    surface_node_ids = tuple(
        sorted({int(node_id) for _, triangle in rows for node_id in triangle.node_ids})
    )
    faces = _oriented_surface_faces(state, rows)
    vertices = np.asarray(
        [state.deformed_coordinates_mm[node_index[node_id]] for node_id in surface_node_ids],
        dtype=np.float32,
    )
    semantic_tags = tuple(str(tag) for tag, _ in rows)
    external = np.asarray(
        [tag in _EXTERNAL_VOLUME_TAGS for tag in semantic_tags],
        dtype=bool,
    )
    external_node_ids = tuple(
        sorted(
            {
                int(node_id)
                for tag, triangle in rows
                if tag in _EXTERNAL_VOLUME_TAGS
                for node_id in triangle.node_ids
            }
        )
    )
    u_by_node = _surface_u_values(tip, state, external_node_ids)
    u_start = np.asarray(
        [min(u_by_node.get(int(node_id), 0.0) for node_id in triangle.node_ids) for _, triangle in rows],
        dtype=float,
    )
    u_end = np.asarray(
        [max(u_by_node.get(int(node_id), 0.0) for node_id in triangle.node_ids) for _, triangle in rows],
        dtype=float,
    )
    interface_tags = tuple(AIR_INTERFACE if value else INTERNAL_INTERFACE for value in external)
    return TriangleSurface(
        vertices=vertices,
        faces=faces,
        normals=_surface_normals(vertices, faces),
        external_surface=external,
        u_start=u_start,
        u_end=u_end,
        semantic_tags=semantic_tags,
        interface_tags=interface_tags,
    )


def _source_state(tip: Fingertip, *, source_epsilon_mm: float) -> tuple[tuple[float, float, float], int]:
    source_xy = np.asarray(tip.led_source, dtype=float)
    axis = np.asarray(tip.emission_axis, dtype=float)
    source_probe = source_xy + float(source_epsilon_mm) * axis
    source_medium = int(tip.geometry.pad_material_geometry.covers(Point(*source_probe)))
    return (float(source_xy[0]), float(source_xy[1]), 0.0), source_medium


def build_fingertip_volume_state_geometry(
    tip: Fingertip,
    state: FingertipVolumeState,
    *,
    reference_mesh: FingertipMesh | None = None,
    source_epsilon_mm: float = 1.0e-5,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Build direct ``full3d_surface`` geometry without FEA artifacts.

    ``reference_mesh`` is only the fixed-carrier/envelope view used by the
    shared geometry builder.  The compliant optical surface always comes from
    the canonical volume state's deformed coordinates and semantic triangles.
    """
    if not isinstance(tip, Fingertip):
        raise TypeError("tip must be Fingertip")
    if not isinstance(state, FingertipVolumeState):
        raise TypeError("state must be FingertipVolumeState")
    if state.morphology_fingerprint != tip.solid().morphology_fingerprint:
        raise Transport3DGeometryError("volume state morphology does not match Fingertip")
    if reference_mesh is None:
        reference_mesh = tip.mesh()
    rigid, envelope = build_fixed_transport_surfaces(reference_mesh, depth_mm=11.0)
    source_position, source_medium = _source_state(
        tip,
        source_epsilon_mm=source_epsilon_mm,
    )
    silicone = _silicone_surface(tip, state)
    geometry_metadata = {
        "morphology_fingerprint": state.morphology_fingerprint,
        "mechanics_source": "solver_neutral.FingertipVolumeState",
        "volume_mesh_tier": state.settings.tier,
        "volume_state_source_node_count": len(state.source_node_ids),
        "full3d_surface_provenance": "actual_deformed_3d_volume_state",
        "rigid_geometry_source": "shared_authoritative_fingertip_geometry",
    }
    if metadata is not None:
        geometry_metadata.update(dict(metadata))
    return build_full3d_transport_geometry(
        tip,
        silicone=silicone,
        rigid=rigid,
        envelope=envelope,
        source_position_mm=source_position,
        source_medium=source_medium,
        metadata=geometry_metadata,
    )


__all__ = ["build_fingertip_volume_state_geometry"]
