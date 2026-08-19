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
    CARRIER_CONTACT_INTERFACE,
    INTERNAL_INTERFACE,
    TriangleSurface,
    Transport3DGeometryError,
    build_fixed_transport_surfaces,
    build_full3d_transport_geometry,
    _surface_normals,
)
from optics.contact_object import CarrierOptics


_SURFACE_ORIENTATION_TOLERANCE_MM = 1.0e-9


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
        if path.distance(point) > tolerance:
            raise Transport3DGeometryError(
                f"external compliant node {node_id} is not on the authoritative reference boundary"
            )
        result[node_id] = float(path.project(point, normalized=True))
    return result


def _oriented_surface_faces(
    state: FingertipVolumeState,
    rows: list[tuple[str, Any]],
) -> np.ndarray:
    """Orient semantic faces outward using canonical tetra boundary topology.

    Face rows retain their exact source node IDs and ordering in ``rows``. Only
    the local face winding used by the optical surface is changed. The
    reference orientation is determined from the unique adjacent tetrahedron,
    so it does not rely on global coordinate thresholds or a guessed center.
    """
    surface_node_ids = tuple(
        sorted({int(node_id) for _, triangle in rows for node_id in triangle.node_ids})
    )
    canonical_index = {
        int(node_id): index for index, node_id in enumerate(state.source_node_ids)
    }
    surface_index = {
        node_id: index for index, node_id in enumerate(surface_node_ids)
    }
    if any(node_id not in canonical_index for node_id in surface_node_ids):
        raise Transport3DGeometryError(
            "semantic optical surface references an unknown canonical volume node"
        )
    faces = np.asarray(
        [
            [surface_index[int(node_id)] for node_id in triangle.node_ids]
            for _, triangle in rows
        ],
        dtype=np.int64,
    )
    reference = state.reference_coordinates_mm
    reference_surface = reference[
        np.asarray([canonical_index[node_id] for node_id in surface_node_ids], dtype=np.int64)
    ]

    tetra_by_face: dict[tuple[int, int, int], list[tuple[int, ...]]] = {}
    for tetrahedron in state.tetrahedra:
        tetra_node_ids = tuple(int(node_id) for node_id in tetrahedron.node_ids)
        for face in (
            (tetra_node_ids[0], tetra_node_ids[1], tetra_node_ids[2]),
            (tetra_node_ids[0], tetra_node_ids[1], tetra_node_ids[3]),
            (tetra_node_ids[0], tetra_node_ids[2], tetra_node_ids[3]),
            (tetra_node_ids[1], tetra_node_ids[2], tetra_node_ids[3]),
        ):
            tetra_by_face.setdefault(tuple(sorted(face)), []).append(tetra_node_ids)
    row_face_keys = [
        tuple(sorted(int(node_id) for node_id in triangle.node_ids))
        for _, triangle in rows
    ]
    if len(set(row_face_keys)) != len(row_face_keys):
        raise Transport3DGeometryError("semantic optical surface contains duplicate triangle faces")
    adjacent_tetrahedra: list[tuple[int, ...]] = []
    for key in row_face_keys:
        candidates = tetra_by_face.get(key, [])
        if len(candidates) != 1:
            raise Transport3DGeometryError(
                "semantic optical surface triangle is not a unique tetrahedral boundary face"
            )
        adjacent_tetrahedra.append(candidates[0])

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
        def orientation_signs() -> np.ndarray:
            signs: list[float] = []
            for face_index in component:
                face = faces[face_index].copy()
                if flips[face_index] == 1:
                    face[1], face[2] = face[2], face[1]
                points = reference_surface[face]
                cross = np.cross(points[1] - points[0], points[2] - points[0])
                norm = float(np.linalg.norm(cross))
                if not np.isfinite(norm) or norm <= 1.0e-12:
                    raise Transport3DGeometryError(
                        "semantic lateral surface contains a degenerate triangle"
                    )
                unit_normal = cross / norm
                tetra_points = reference[
                    np.asarray(
                        [canonical_index[node_id] for node_id in adjacent_tetrahedra[face_index]],
                        dtype=np.int64,
                    )
                ]
                offset = np.mean(points, axis=0) - np.mean(tetra_points, axis=0)
                sign = float(np.dot(unit_normal, offset))
                if not np.isfinite(sign) or abs(sign) <= _SURFACE_ORIENTATION_TOLERANCE_MM:
                    raise Transport3DGeometryError(
                        "semantic lateral surface orientation is geometrically ambiguous"
                    )
                signs.append(sign)
            return np.asarray(signs, dtype=float)

        signs = orientation_signs()
        if np.all(signs < 0.0):
            flips[component] ^= 1
        elif not np.all(signs > 0.0):
            raise Transport3DGeometryError(
                "semantic lateral surface orientation is inconsistent with tetrahedral interior"
            )

    faces[flips == 1, 1], faces[flips == 1, 2] = (
        faces[flips == 1, 2],
        faces[flips == 1, 1].copy(),
    )
    return faces.astype(np.uint32)


def _silicone_surface(
    tip: Fingertip,
    state: FingertipVolumeState,
    *,
    carrier_contact_source_node_ids: frozenset[int] = frozenset(),
) -> TriangleSurface:
    canonical_index = {
        int(node_id): index for index, node_id in enumerate(state.source_node_ids)
    }
    surface_definitions = {
        definition.name: definition for definition in state.volume_mesh.solid.surfaces
    }
    # The two longitudinal end caps close the finite extrusion in z, but they
    # are not exposed silicone interfaces for the lateral transport scene.
    # Only semantic lateral families enter the optical surface.
    rows = [
        (tag, triangle)
        for tag, triangles in sorted(state.surface_triangles.items())
        if tag in surface_definitions and surface_definitions[tag].kind != "longitudinal_end"
        for triangle in triangles
    ]
    if not rows:
        raise Transport3DGeometryError("FingertipVolumeState has no lateral optical surface triangles")
    row_tags = {tag for tag, _ in rows}
    unknown_tags = set(state.surface_triangles) - set(surface_definitions)
    if unknown_tags:
        raise Transport3DGeometryError(
            f"FingertipVolumeState contains unknown semantic surface families: {sorted(unknown_tags)!r}"
        )
    unsupported_tags = {
        tag
        for tag in row_tags
        if surface_definitions[tag].kind not in {"outer_compliant", "support", "void"}
    }
    if unsupported_tags:
        raise Transport3DGeometryError(
            f"FingertipVolumeState contains unsupported optical surface families: {sorted(unsupported_tags)!r}"
        )
    expected_tags = {
        definition.name
        for definition in state.volume_mesh.solid.surfaces
        if definition.kind in {"outer_compliant", "support", "void"}
    }
    if row_tags != expected_tags:
        raise Transport3DGeometryError(
            "FingertipVolumeState semantic optical surface families do not match the authoritative solid"
        )
    surface_node_ids = tuple(
        sorted({int(node_id) for _, triangle in rows for node_id in triangle.node_ids})
    )
    faces = _oriented_surface_faces(state, rows)
    vertices = np.asarray(
        [state.deformed_coordinates_mm[canonical_index[node_id]] for node_id in surface_node_ids],
        dtype=np.float32,
    )
    semantic_tags = tuple(str(tag) for tag, _ in rows)
    external = np.asarray(
        [surface_definitions[tag].kind == "outer_compliant" for tag in semantic_tags],
        dtype=bool,
    )
    external_node_ids = tuple(
        sorted(
            {
                int(node_id)
                for tag, triangle in rows
                if (
                    surface_definitions[tag].kind == "outer_compliant"
                    and surface_definitions[tag].source_geometry is not None
                )
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
    # The mechanics result supplies exact canonical source-node provenance.
    # Only semantic void triangles participating in that contact are changed;
    # an open void remains an ordinary silicone-air boundary.
    carrier_contact_families = {
        "void_left",
        "void_right",
        "void_bottom",
    }
    interface_tags = tuple(
        CARRIER_CONTACT_INTERFACE
        if tag in carrier_contact_families
        and any(
            int(node_id) in carrier_contact_source_node_ids
            for node_id in triangle.node_ids
        )
        else AIR_INTERFACE if value else INTERNAL_INTERFACE
        for (tag, triangle), value in zip(rows, external)
    )
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
    carrier_contact_source_node_ids: frozenset[int] | set[int] | tuple[int, ...] = frozenset(),
    carrier_optics: CarrierOptics | None = None,
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
    contact_node_ids = frozenset(int(node_id) for node_id in carrier_contact_source_node_ids)
    silicone = _silicone_surface(
        tip,
        state,
        carrier_contact_source_node_ids=contact_node_ids,
    )
    carrier_triangle_mask = np.asarray(
        [tag == CARRIER_CONTACT_INTERFACE for tag in silicone.interface_tags or ()],
        dtype=bool,
    )
    geometry_metadata = {
        "morphology_fingerprint": state.morphology_fingerprint,
        "mechanics_source": "solver_neutral.FingertipVolumeState",
        "volume_mesh_tier": state.settings.tier,
        "volume_state_source_node_count": len(state.source_node_ids),
        "full3d_surface_provenance": "actual_deformed_3d_volume_state",
        "rigid_geometry_source": "shared_authoritative_fingertip_geometry",
        "carrier_contact_source_node_ids": sorted(contact_node_ids),
        "carrier_optical_contact_triangle_count": int(np.count_nonzero(carrier_triangle_mask)),
        "carrier_optics_enabled": carrier_optics is not None,
        "carrier_boundary_model": (
            None if carrier_optics is None else carrier_optics.boundary_model
        ),
        "carrier_mapping_method": "exact_semantic_surface_triangle_any_contact_vertex",
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
        carrier_optics=carrier_optics,
    )


__all__ = ["build_fingertip_volume_state_geometry"]
