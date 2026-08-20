"""Neutral 11 mm periodic geometry for deterministic 3D transport."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from mesh.fingertip.contracts import FingertipMesh
from mesh.fingertip.surface import PadMesh
from model.fingertip import Fingertip
from optics.contracts.objects import ObjectBoundaryOptics
from optics.geometry.extrusion import (
    InvalidExtrudedOpticalMesh,
    _ExtrudedMesh,
)


class Transport3DGeometryError(ValueError):
    """Raised when a periodic transport scene cannot be built safely."""


EXTERNAL_SURFACE_TAGS = (
    "pad_outer_left",
    "pad_outer_arc",
    "pad_outer_right",
)

AIR_INTERFACE = "AIR_INTERFACE"
OBJECT_CONTACT_INTERFACE = "OBJECT_CONTACT_INTERFACE"
CARRIER_CONTACT_INTERFACE = "CARRIER_CONTACT_INTERFACE"
INTERNAL_INTERFACE = "INTERNAL_INTERFACE"


@dataclass(frozen=True)
class TriangleSurface:
    """One OptiX-ready triangle set and its semantic metadata."""

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    boundary_edge_indices: np.ndarray | None = None
    external_surface: np.ndarray | None = None
    u_start: np.ndarray | None = None
    u_end: np.ndarray | None = None
    semantic_tags: tuple[str, ...] | None = None
    interface_tags: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        vertices = np.array(self.vertices, dtype=np.float32, copy=True)
        faces = np.array(self.faces, dtype=np.uint32, copy=True)
        normals = np.array(self.normals, dtype=np.float32, copy=True)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
            raise Transport3DGeometryError("surface vertices must have shape (N, 3)")
        if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
            raise Transport3DGeometryError("surface faces must have shape (F, 3)")
        if np.any(faces >= len(vertices)):
            raise Transport3DGeometryError("surface face index is out of range")
        if normals.shape != faces.shape:
            raise Transport3DGeometryError("surface normals must have shape (F, 3)")
        if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(normals)):
            raise Transport3DGeometryError("surface geometry must be finite")
        if np.any(np.linalg.norm(normals, axis=1) <= 0.0):
            raise Transport3DGeometryError("surface normals must be nonzero")
        geometric_cross = np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        )
        geometric_lengths = np.linalg.norm(geometric_cross, axis=1)
        if np.any(geometric_lengths <= 1.0e-12):
            raise Transport3DGeometryError("surface contains a degenerate triangle")
        geometric_normals = geometric_cross / geometric_lengths[:, None]
        supplied_lengths = np.linalg.norm(normals, axis=1)
        orientation_alignment = np.sum(
            geometric_normals * normals / supplied_lengths[:, None],
            axis=1,
        )
        if np.any(orientation_alignment <= 1.0 - 1.0e-5):
            raise Transport3DGeometryError(
                "surface triangle normals are not consistently oriented"
            )
        edge_directions: dict[tuple[int, int], list[int]] = {}
        for triangle in faces:
            for first, second in (
                (int(triangle[0]), int(triangle[1])),
                (int(triangle[1]), int(triangle[2])),
                (int(triangle[2]), int(triangle[0])),
            ):
                key = min(first, second), max(first, second)
                edge_directions.setdefault(key, []).append(
                    1 if (first, second) == key else -1
                )
        if any(
            len(directions) > 2 or (len(directions) == 2 and sum(directions) != 0)
            for directions in edge_directions.values()
        ):
            raise Transport3DGeometryError(
                "surface triangle orientation is inconsistent across shared edges"
            )
        arrays = [vertices, faces, normals]
        optional = []
        for name, value, dtype in (
            ("boundary_edge_indices", self.boundary_edge_indices, np.int64),
            ("external_surface", self.external_surface, bool),
            ("u_start", self.u_start, float),
            ("u_end", self.u_end, float),
        ):
            if value is None:
                optional.append((name, None))
                continue
            array = np.array(value, dtype=dtype, copy=True)
            if array.ndim != 1 or len(array) != len(faces):
                raise Transport3DGeometryError(
                    f"{name} must have one value per triangle"
                )
            if dtype is not bool and not np.all(np.isfinite(array)):
                raise Transport3DGeometryError(f"{name} must be finite")
            array.setflags(write=False)
            optional.append((name, array))
        semantic_tags = None
        if self.semantic_tags is not None:
            semantic_tags = tuple(str(tag) for tag in self.semantic_tags)
            if len(semantic_tags) != len(faces):
                raise Transport3DGeometryError(
                    "semantic_tags must have one value per triangle"
                )
        interface_tags = None
        if self.interface_tags is not None:
            interface_tags = tuple(str(tag) for tag in self.interface_tags)
            if len(interface_tags) != len(faces):
                raise Transport3DGeometryError(
                    "interface_tags must have one value per triangle"
                )
        for array in arrays:
            array.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "normals", normals)
        for name, value in optional:
            object.__setattr__(self, name, value)
        object.__setattr__(self, "semantic_tags", semantic_tags)
        object.__setattr__(self, "interface_tags", interface_tags)


@dataclass(frozen=True)
class ExtrudedTransportGeometry:
    """All neutral surfaces and metadata for one direct FULL_3D state."""

    silicone: TriangleSurface
    rigid: TriangleSurface
    envelope: TriangleSurface
    depth_mm: float
    z_min_mm: float
    z_max_mm: float
    source_position_mm: tuple[float, float, float]
    source_medium: int
    metadata: Mapping[str, Any]
    indenter_optics: ObjectBoundaryOptics | None = None
    carrier_optics: ObjectBoundaryOptics | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.depth_mm) or self.depth_mm != 11.0:
            raise Transport3DGeometryError("the transport cell depth must be exactly 11 mm")
        if self.z_min_mm != -5.5 or self.z_max_mm != 5.5:
            raise Transport3DGeometryError("the transport cell must span exactly +/-5.5 mm")
        source = tuple(float(value) for value in self.source_position_mm)
        if source[2] != 0.0 or not np.all(np.isfinite(source)):
            raise Transport3DGeometryError("the single source must be at z=0")
        if self.source_medium not in (0, 1):
            raise Transport3DGeometryError("source_medium must be air=0 or silicone=1")
        object.__setattr__(self, "source_position_mm", source)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _surface_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    first = vertices[faces[:, 0]]
    second = vertices[faces[:, 1]]
    third = vertices[faces[:, 2]]
    normals = np.cross(second - first, third - first)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1.0e-12):
        raise Transport3DGeometryError("triangle normal is degenerate")
    return normals / lengths[:, None]


def _ordered_chain(edges: np.ndarray, coordinates: np.ndarray) -> tuple[int, ...]:
    adjacency: dict[int, list[int]] = {}
    for first, second in edges:
        i, j = int(first), int(second)
        adjacency.setdefault(i, []).append(j)
        adjacency.setdefault(j, []).append(i)
    if not adjacency or any(len(neighbors) > 2 for neighbors in adjacency.values()):
        raise Transport3DGeometryError("external semantic boundary is not a chain")
    endpoints = [node for node, neighbors in adjacency.items() if len(neighbors) == 1]
    if len(endpoints) != 2:
        raise Transport3DGeometryError("external semantic boundary must have two endpoints")
    start = min(endpoints, key=lambda node: (float(coordinates[node, 0]), float(coordinates[node, 1]), node))
    chain = [start]
    previous = None
    current = start
    while True:
        choices = sorted(node for node in adjacency[current] if node != previous)
        if not choices:
            break
        next_node = choices[0]
        if next_node in chain:
            raise Transport3DGeometryError("external semantic boundary chain loops")
        chain.append(next_node)
        previous, current = current, next_node
    if len(chain) != len(adjacency):
        raise Transport3DGeometryError("external semantic boundary chain is disconnected")
    return tuple(chain)


def _rigid_pad_mesh(mesh: FingertipMesh) -> PadMesh:
    node_ids = sorted(
        {
            int(node_id)
            for element in mesh.carrier_elements
            for node_id in element.node_ids
        }
    )
    if not node_ids:
        raise Transport3DGeometryError("the rigid carrier has no elements")
    id_to_local = {node_id: index for index, node_id in enumerate(node_ids)}
    coordinates = np.asarray(
        [[mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm] for node_id in node_ids],
        dtype=float,
    )
    triangles = np.asarray(
        [[id_to_local[int(node_id)] for node_id in element.node_ids] for element in mesh.carrier_elements],
        dtype=np.int64,
    )
    boundaries: dict[str, np.ndarray] = {}
    node_set = set(node_ids)
    for tag, edges in mesh.boundary_edges.items():
        selected = [
            edge.node_ids
            for edge in edges
            if edge.domain == "rigid_carrier" and set(edge.node_ids).issubset(node_set)
        ]
        if selected:
            boundaries[tag] = np.asarray(
                [[id_to_local[int(first)], id_to_local[int(second)]] for first, second in selected],
                dtype=np.int64,
            )
    return PadMesh.from_arrays(
        node_ids=np.asarray(node_ids, dtype=np.int64),
        reference_coordinates_mm=coordinates,
        element_connectivity_node_ids=np.asarray(
            [element.node_ids for element in mesh.carrier_elements],
            dtype=np.int64,
        ),
        boundary_edge_node_ids_by_tag={
            tag: np.asarray(
                [[node_ids[first], node_ids[second]] for first, second in edges],
                dtype=np.int64,
            )
            for tag, edges in boundaries.items()
        },
    )


def _surface_from_extrusion(
    extrusion: _ExtrudedMesh,
    coordinates: np.ndarray,
    faces: np.ndarray,
    **metadata: Any,
) -> TriangleSurface:
    vertices = extrusion.vertices_for_coordinates(coordinates)
    return TriangleSurface(
        vertices=vertices,
        faces=faces,
        normals=_surface_normals(vertices, faces),
        **metadata,
    )


def build_fixed_transport_surfaces(
    reference_mesh: FingertipMesh,
    *,
    depth_mm: float = 11.0,
    envelope_coordinates: np.ndarray | None = None,
) -> tuple[TriangleSurface, TriangleSurface]:
    """Build the fixed rigid-carrier and virtual-envelope surfaces.

    The fixed surfaces are independent of the mechanics backend.  Compliant
    silicone triangles are supplied separately by the direct
    ``FingertipVolumeState`` adapter.
    """
    if not isinstance(reference_mesh, FingertipMesh):
        raise TypeError("reference_mesh must be FingertipMesh")
    reference_pad = reference_mesh.pad
    try:
        rigid_pad = _rigid_pad_mesh(reference_mesh)
        rigid_extrusion = _ExtrudedMesh.from_pad_mesh(rigid_pad, depth_mm=depth_mm)
        rigid_faces = rigid_extrusion.faces_3d[2 * len(rigid_pad.triangles):]
        rigid = _surface_from_extrusion(
            rigid_extrusion,
            rigid_pad.coordinates,
            rigid_faces,
        )

        # The envelope is a virtual air boundary only at the cutout closure.
        # It shares the semantic pad boundary vertices and does not invent a
        # coordinate-based surface classifier.
        full_outer_edges = np.asarray(
            [
                edge
                for tag in (
                    "pad_bond_left",
                    *EXTERNAL_SURFACE_TAGS,
                    "pad_bond_right",
                )
                for edge in reference_pad.boundary_edges_for(tag)
            ],
            dtype=np.int64,
        )
        full_chain = _ordered_chain(
            full_outer_edges,
            reference_pad.reference_coordinates_mm,
        )
        closure = np.asarray([[full_chain[-1], full_chain[0]]], dtype=np.int64)
        envelope_extrusion = _ExtrudedMesh.from_pad_mesh(reference_pad, depth_mm=depth_mm)
        envelope = _surface_from_extrusion(
            envelope_extrusion,
            np.asarray(
                reference_pad.coordinates if envelope_coordinates is None else envelope_coordinates,
                dtype=float,
            ),
            envelope_extrusion.side_faces_for_edges(
                np.vstack((full_outer_edges, closure))
            ),
        )
    except InvalidExtrudedOpticalMesh as exc:
        raise Transport3DGeometryError(str(exc)) from exc
    return rigid, envelope


def build_full3d_transport_geometry(
    tip: Fingertip,
    *,
    silicone: TriangleSurface,
    rigid: TriangleSurface,
    envelope: TriangleSurface,
    source_position_mm: tuple[float, float, float],
    source_medium: int,
    metadata: Mapping[str, Any],
    carrier_optics: ObjectBoundaryOptics | None = None,
    depth_mm: float = 11.0,
) -> ExtrudedTransportGeometry:
    """Build transport geometry from a direct deformed 3D surface artifact.

    This constructor intentionally accepts triangles directly and does not
    accept a ``PadMesh``.  The distinction is the provenance guard against
    accidentally labelling an extrusion of 2D deformation as FULL_3D.  The
    provenance guard accepts only direct FEA or direct VBD surface states.
    Full 3D field accumulation is performed from retained native path
    segments; no collapsed 2D optical domain is accepted here.
    """
    if not isinstance(tip, Fingertip):
        raise TypeError("tip must be a Fingertip")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    source = tuple(float(value) for value in source_position_mm)
    if len(source) != 3 or not np.all(np.isfinite(source)):
        raise Transport3DGeometryError("full 3D source position must be finite")
    if abs(source[2]) > 1.0e-9:
        raise Transport3DGeometryError("the representative full 3D source must be at z=0")
    if source_medium not in (0, 1):
        raise Transport3DGeometryError("source_medium must be air=0 or silicone=1")
    for name, surface in (
        ("silicone", silicone),
        ("rigid", rigid),
        ("envelope", envelope),
    ):
        if not isinstance(surface, TriangleSurface):
            raise TypeError(f"{name} must be a TriangleSurface")
        if name == "silicone" and surface.semantic_tags is None:
            raise Transport3DGeometryError(
                "full 3D silicone surface must preserve semantic surface tags"
            )
        if not np.all(np.isfinite(surface.vertices[:, 2])):
            raise Transport3DGeometryError(f"{name} surface has a non-finite longitudinal coordinate")
    enriched_metadata = dict(metadata)
    enriched_metadata["geometry_mode"] = "full3d_surface"
    provenance = str(
        enriched_metadata.get(
            "full3d_surface_provenance",
            "actual_deformed_3d_fea_surface",
        )
    )
    if provenance not in {
        "actual_reference_3d_volume_state",
        "actual_deformed_3d_fea_surface",
        "actual_deformed_3d_vbd_surface",
        "actual_deformed_3d_volume_state",
    }:
        raise Transport3DGeometryError(
            "full 3D surface provenance must identify a direct FEA or VBD "
            "deformed surface"
        )
    enriched_metadata["full3d_surface_provenance"] = provenance
    enriched_metadata["reference_periodic_z_planes_mm"] = [-depth_mm / 2.0, depth_mm / 2.0]
    enriched_metadata["deformed_surface_z_extent_mm"] = [
        float(np.min(silicone.vertices[:, 2])),
        float(np.max(silicone.vertices[:, 2])),
    ]
    enriched_metadata["deformed_surface_exceeds_reference_z_planes"] = bool(
        np.min(silicone.vertices[:, 2]) < -depth_mm / 2.0 - 1.0e-9
        or np.max(silicone.vertices[:, 2]) > depth_mm / 2.0 + 1.0e-9
    )
    has_carrier_contact = any(
        tag == CARRIER_CONTACT_INTERFACE
        for tag in (silicone.interface_tags or ())
    )
    if has_carrier_contact and carrier_optics is None:
        raise Transport3DGeometryError(
            "carrier contact triangles require an explicit carrier optical boundary"
        )
    enriched_metadata["carrier_contact_active"] = has_carrier_contact
    enriched_metadata["carrier_boundary_model"] = (
        None if carrier_optics is None else carrier_optics.boundary_model
    )
    return ExtrudedTransportGeometry(
        silicone=silicone,
        rigid=rigid,
        envelope=envelope,
        depth_mm=depth_mm,
        z_min_mm=-depth_mm / 2.0,
        z_max_mm=depth_mm / 2.0,
        source_position_mm=source,
        source_medium=source_medium,
        metadata=enriched_metadata,
        carrier_optics=carrier_optics,
    )


__all__ = [
    "AIR_INTERFACE",
    "CARRIER_CONTACT_INTERFACE",
    "EXTERNAL_SURFACE_TAGS",
    "ExtrudedTransportGeometry",
    "INTERNAL_INTERFACE",
    "OBJECT_CONTACT_INTERFACE",
    "TriangleSurface",
    "Transport3DGeometryError",
    "build_full3d_transport_geometry",
    "build_fixed_transport_surfaces",
]
