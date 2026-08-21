"""Production adapter from a fingertip volume state to FULL_3D OptiX."""

from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import linemerge

from lumo.mesh.rigid.carrier import RigidCarrierMesh
from lumo.mesh.volume.state import FingertipVolumeState
from lumo.finger.fingertip import Fingertip

from .geometry import (
    AIR_INTERFACE,
    CARRIER_CONTACT_INTERFACE,
    TransportGeometry,
    Full3DSurfaceProvenance,
    INTERNAL_INTERFACE,
    TriangleSurface,
    Transport3DGeometryError,
    build_full3d_transport_geometry,
    _surface_normals,
)
from lumo.ray_tracing.contracts.objects import CarrierOptics


_SURFACE_ORIENTATION_TOLERANCE_MM = 1.0e-9
_ENVELOPE_CHAIN_SURFACE_NAMES = (
    "support_bond_left",
    "outer_compliant_left",
    "outer_compliant_arc",
    "outer_compliant_right",
    "support_bond_right",
)


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
    tolerance = max(
        1.0e-6,
        100.0 * tip.parameters.geometry_length_tolerance_mm,
    )
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
    """Validate the neutral mesh's outward winding against adjacent tetrahedra."""

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
    for (_, triangle), key in zip(rows, row_face_keys, strict=True):
        candidates = tetra_by_face.get(key, [])
        if len(candidates) != 1:
            raise Transport3DGeometryError(
                "semantic optical surface triangle is not a unique tetrahedral boundary face"
            )
        points = reference[
            np.asarray(
                [canonical_index[int(node_id)] for node_id in triangle.node_ids],
                dtype=np.int64,
            )
        ]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        norm = float(np.linalg.norm(normal))
        tetra_points = reference[
            np.asarray(
                [canonical_index[node_id] for node_id in candidates[0]],
                dtype=np.int64,
            )
        ]
        outward = np.mean(points, axis=0) - np.mean(tetra_points, axis=0)
        orientation = float(np.dot(normal / max(norm, 1.0e-30), outward))
        if (
            not np.isfinite(norm)
            or norm <= 1.0e-12
            or not np.isfinite(orientation)
            or orientation <= _SURFACE_ORIENTATION_TOLERANCE_MM
        ):
            raise Transport3DGeometryError(
                "neutral semantic surface is not outward-oriented from its tetrahedral interior"
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
    interface_tags = tuple(
        CARRIER_CONTACT_INTERFACE
        if tag == "void_bottom"
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


def _carrier_surface(carrier_mesh: RigidCarrierMesh) -> TriangleSurface:
    """Adapt the shared neutral carrier's physical lateral faces for OptiX."""

    surface_mesh = carrier_mesh.surface_mesh
    face_indices = np.asarray(
        carrier_mesh.lateral_face_indices,
        dtype=np.int64,
    )
    vertices = np.asarray(surface_mesh.vertices_mm, dtype=np.float32)
    faces = np.asarray(surface_mesh.faces[face_indices], dtype=np.uint32)
    return TriangleSurface(
        vertices=vertices,
        faces=faces,
        normals=_surface_normals(vertices, faces),
    )


def _virtual_escape_surface(state: FingertipVolumeState) -> TriangleSurface:
    """Build the virtual reference envelope without a second discretization.

    The physical silicone shell is supplied by the Newton volume state and the
    carrier comes from the shared neutral rigid mesh.  The envelope reuses the
    volume mesh's reference outer/support triangles and adds only the two
    cutout-closure triangles needed to count air escape.
    """

    solid = state.volume_mesh.solid
    definitions = {definition.name: definition for definition in solid.surfaces}
    missing = set(_ENVELOPE_CHAIN_SURFACE_NAMES) - set(definitions)
    if missing:
        raise Transport3DGeometryError(
            "fingertip solid is missing virtual-envelope surface families: "
            f"{sorted(missing)!r}"
        )
    source_lines = []
    for name in _ENVELOPE_CHAIN_SURFACE_NAMES:
        source = definitions[name].source_geometry
        if source is None or source.geom_type != "LineString" or source.is_empty:
            raise Transport3DGeometryError(
                f"virtual-envelope source {name!r} must be a non-empty LineString"
            )
        source_lines.append(source)
    chain = linemerge(source_lines)
    if chain.geom_type != "LineString" or chain.is_empty:
        raise Transport3DGeometryError(
            "authoritative outer/support surfaces do not form one envelope chain"
        )
    coordinates = tuple(chain.coords)
    if len(coordinates) < 2:
        raise Transport3DGeometryError("virtual-envelope chain has no endpoints")
    first_xy = tuple(float(value) for value in coordinates[0][:2])
    second_xy = tuple(float(value) for value in coordinates[-1][:2])
    if np.linalg.norm(np.asarray(first_xy) - np.asarray(second_xy)) <= 1.0e-12:
        raise Transport3DGeometryError("virtual-envelope closure is degenerate")
    rows = [
        (name, triangle)
        for name in _ENVELOPE_CHAIN_SURFACE_NAMES
        for triangle in state.surface_triangles.get(name, ())
    ]
    if any(not state.surface_triangles.get(name) for name in _ENVELOPE_CHAIN_SURFACE_NAMES):
        raise Transport3DGeometryError(
            "volume state is missing virtual-envelope semantic triangles"
        )
    source_node_ids = tuple(
        sorted(
            {
                int(node_id)
                for _, triangle in rows
                for node_id in triangle.node_ids
            }
        )
    )
    canonical_index = {
        int(node_id): index for index, node_id in enumerate(state.source_node_ids)
    }
    outer_vertices = np.asarray(
        [
            state.reference_coordinates_mm[canonical_index[node_id]]
            for node_id in source_node_ids
        ],
        dtype=np.float32,
    )
    outer_faces = _oriented_surface_faces(state, rows)
    closure_vertices = np.asarray(
        (
            (*first_xy, solid.z_min_mm),
            (*second_xy, solid.z_min_mm),
            (*second_xy, solid.z_max_mm),
            (*first_xy, solid.z_max_mm),
        ),
        dtype=np.float32,
    )
    closure_offset = len(outer_vertices)
    closure_faces = np.asarray(
        ((0, 1, 2), (0, 2, 3)),
        dtype=np.uint32,
    ) + closure_offset
    vertices = np.vstack((outer_vertices, closure_vertices))
    faces = np.vstack((outer_faces, closure_faces))
    return TriangleSurface(
        vertices=vertices,
        faces=faces,
        normals=_surface_normals(vertices, faces),
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
    carrier_mesh: RigidCarrierMesh,
    source_epsilon_mm: float = 1.0e-5,
    carrier_contact_source_node_ids: frozenset[int] | set[int] | tuple[int, ...] = frozenset(),
    carrier_optics: CarrierOptics | None = None,
    carrier_mapping_tolerance_mm: float | None = None,
    full3d_surface_provenance: Full3DSurfaceProvenance,
) -> TransportGeometry:
    """Adapt one Newton-compatible volume state directly into OptiX surfaces.

    No optical fingertip mesh is generated.  The compliant surface comes from
    the state's deformed coordinates and canonical semantic triangles; the
    rigid surface is the same neutral carrier mesh supplied to Newton.
    """
    if not isinstance(tip, Fingertip):
        raise TypeError("tip must be Fingertip")
    if not isinstance(state, FingertipVolumeState):
        raise TypeError("state must be FingertipVolumeState")
    if not isinstance(carrier_mesh, RigidCarrierMesh):
        raise TypeError("carrier_mesh must be RigidCarrierMesh")
    if state.morphology_fingerprint != tip.solid().morphology_fingerprint:
        raise Transport3DGeometryError("volume state morphology does not match Fingertip")
    if carrier_mesh.morphology_fingerprint != state.morphology_fingerprint:
        raise Transport3DGeometryError(
            "rigid carrier morphology does not match the volume state"
        )
    solid = state.volume_mesh.solid
    if not np.isclose(carrier_mesh.z_min_mm, solid.z_min_mm) or not np.isclose(
        carrier_mesh.z_max_mm,
        solid.z_max_mm,
    ):
        raise Transport3DGeometryError(
            "rigid carrier longitudinal bounds do not match the volume state"
        )
    rigid = _carrier_surface(carrier_mesh)
    envelope = _virtual_escape_surface(state)
    source_position, source_medium = _source_state(
        tip,
        source_epsilon_mm=source_epsilon_mm,
    )
    contact_node_ids = frozenset(int(node_id) for node_id in carrier_contact_source_node_ids)
    void_bottom_node_ids = frozenset(
        int(node_id)
        for triangle in state.surface_triangles.get("void_bottom", ())
        for node_id in triangle.node_ids
    )
    invalid_contact_node_ids = contact_node_ids - void_bottom_node_ids
    if invalid_contact_node_ids:
        raise Transport3DGeometryError(
            "carrier contact provenance must contain only canonical void-bottom "
            f"source nodes: {sorted(invalid_contact_node_ids)!r}"
        )
    silicone = _silicone_surface(
        tip,
        state,
        carrier_contact_source_node_ids=contact_node_ids,
    )
    return build_full3d_transport_geometry(
        tip,
        silicone=silicone,
        rigid=rigid,
        envelope=envelope,
        source_position_mm=source_position,
        source_medium=source_medium,
        full3d_surface_provenance=full3d_surface_provenance,
        carrier_optics=carrier_optics,
        carrier_mapping_tolerance_mm=carrier_mapping_tolerance_mm,
    )


__all__ = ["build_fingertip_volume_state_geometry"]
