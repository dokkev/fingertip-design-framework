"""Adapter from the authoritative 3D fingertip volume mesh to mechanics3d."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from mesh.volume_types import FingertipVolumeMesh

from .types import TetMeshData


def _readonly_array(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _positive_tet_orientation(vertices: np.ndarray, tetrahedra: np.ndarray) -> None:
    points = vertices[tetrahedra]
    six_volumes = np.einsum(
        "ij,ij->i",
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
        points[:, 3] - points[:, 0],
    )
    if np.any(~np.isfinite(six_volumes)) or np.any(six_volumes <= 1.0e-12):
        raise ValueError("volume mesh tetrahedra must retain positive orientation")


@dataclass(frozen=True)
class FingertipMechanicsMesh:
    """Neutral fingertip topology plus source-node and surface provenance."""

    tet_mesh: TetMeshData
    source_node_ids: np.ndarray
    support_vertex_indices: tuple[int, ...]
    surface_triangles: Mapping[str, np.ndarray]
    morphology_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.tet_mesh, TetMeshData):
            raise TypeError("tet_mesh must be TetMeshData")

        source_node_ids = np.asarray(self.source_node_ids)
        if source_node_ids.ndim != 1 or source_node_ids.shape[0] != self.tet_mesh.vertices.shape[0]:
            raise ValueError("source_node_ids must have one entry per local vertex")
        if not np.issubdtype(source_node_ids.dtype, np.integer):
            raise ValueError("source_node_ids must contain integer Gmsh node IDs")
        source_node_ids = np.asarray(source_node_ids, dtype=np.int64)
        if len(set(source_node_ids.tolist())) != source_node_ids.shape[0]:
            raise ValueError("source_node_ids must be unique")

        vertex_count = self.tet_mesh.vertices.shape[0]
        support = tuple(int(index) for index in self.support_vertex_indices)
        if len(set(support)) != len(support):
            raise ValueError("support_vertex_indices must be unique")
        if any(index < 0 or index >= vertex_count for index in support):
            raise ValueError("support_vertex_indices contain an out-of-range local index")

        if not isinstance(self.surface_triangles, Mapping):
            raise TypeError("surface_triangles must be a mapping")
        surfaces: dict[str, np.ndarray] = {}
        for tag, triangles in self.surface_triangles.items():
            if not isinstance(tag, str) or not tag:
                raise ValueError("surface tags must be non-empty strings")
            raw_triangles = np.asarray(triangles)
            if raw_triangles.ndim != 2 or raw_triangles.shape[1] != 3:
                raise ValueError(f"surface {tag!r} must have shape (F, 3)")
            if not np.issubdtype(raw_triangles.dtype, np.integer):
                raise ValueError(f"surface {tag!r} must contain integer local indices")
            local_triangles = np.asarray(raw_triangles, dtype=np.int32)
            if np.any(local_triangles < 0) or np.any(local_triangles >= vertex_count):
                raise ValueError(f"surface {tag!r} contains an out-of-range local index")
            if np.any(
                (local_triangles[:, 0] == local_triangles[:, 1])
                | (local_triangles[:, 1] == local_triangles[:, 2])
                | (local_triangles[:, 0] == local_triangles[:, 2])
            ):
                raise ValueError(f"surface {tag!r} contains a degenerate triangle")
            surfaces[tag] = _readonly_array(local_triangles, dtype=np.int32)

        if not isinstance(self.morphology_fingerprint, str) or not self.morphology_fingerprint:
            raise ValueError("morphology_fingerprint must be a non-empty string")

        _positive_tet_orientation(self.tet_mesh.vertices, self.tet_mesh.tetrahedra)
        object.__setattr__(self, "source_node_ids", _readonly_array(source_node_ids, dtype=np.int64))
        object.__setattr__(self, "support_vertex_indices", support)
        object.__setattr__(self, "surface_triangles", MappingProxyType(surfaces))


def _support_surface_tags(volume_mesh: FingertipVolumeMesh) -> tuple[str, ...]:
    tags = tuple(
        definition.name
        for definition in volume_mesh.solid.surfaces
        if definition.kind == "support" and definition.source_geometry is not None
    )
    if set(tags) != {"support_bond_left", "support_bond_right"}:
        raise ValueError(f"unexpected authoritative support surface family: {tags!r}")
    return tags


def prepare_fingertip_mechanics_mesh(
    volume_mesh: FingertipVolumeMesh,
) -> FingertipMechanicsMesh:
    """Convert an existing validated ``FingertipVolumeMesh`` without remeshing."""

    if not isinstance(volume_mesh, FingertipVolumeMesh):
        raise TypeError("volume_mesh must be FingertipVolumeMesh")
    if not volume_mesh.validation.passed:
        raise ValueError(
            "refusing invalid FingertipVolumeMesh: "
            + ", ".join(volume_mesh.validation.errors)
        )
    if not volume_mesh.nodes or not volume_mesh.tetrahedra:
        raise ValueError("FingertipVolumeMesh must contain nodes and tetrahedra")
    if not volume_mesh.surface_triangles:
        raise ValueError("FingertipVolumeMesh must contain semantic surface triangles")

    source_node_ids = tuple(sorted(volume_mesh.nodes))
    if any(volume_mesh.nodes[node_id].id != node_id for node_id in source_node_ids):
        raise ValueError("volume mesh node dictionary keys must match VolumeNode.id")
    coordinates = np.asarray(
        [
            [
                volume_mesh.nodes[node_id].x_mm,
                volume_mesh.nodes[node_id].y_mm,
                volume_mesh.nodes[node_id].z_mm,
            ]
            for node_id in source_node_ids
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("volume mesh node coordinates must be finite")

    local_index = {node_id: index for index, node_id in enumerate(source_node_ids)}

    def translate_node_ids(node_ids: tuple[int, ...], *, owner: str) -> tuple[int, ...]:
        try:
            return tuple(local_index[node_id] for node_id in node_ids)
        except KeyError as exception:
            raise ValueError(f"{owner} references an unknown source node") from exception

    local_tetrahedra = np.asarray(
        [
            translate_node_ids(tetrahedron.node_ids, owner=f"tetrahedron {tetrahedron.id}")
            for tetrahedron in volume_mesh.tetrahedra
        ],
        dtype=np.int32,
    )
    _positive_tet_orientation(coordinates, local_tetrahedra)

    local_surfaces: dict[str, np.ndarray] = {}
    for tag, triangles in sorted(volume_mesh.surface_triangles.items()):
        local_surfaces[tag] = np.asarray(
            [
                translate_node_ids(triangle.node_ids, owner=f"surface {tag} triangle {triangle.id}")
                for triangle in triangles
            ],
            dtype=np.int32,
        )
        if not local_surfaces[tag].size:
            raise ValueError(f"semantic surface {tag!r} has no triangles")
        if any(triangle.semantic_tag != tag for triangle in triangles):
            raise ValueError(f"surface triangle semantic tag mismatch for {tag!r}")

    support_source_ids = {
        node_id
        for tag in _support_surface_tags(volume_mesh)
        for triangle in volume_mesh.surface_triangles.get(tag, ())
        for node_id in triangle.node_ids
    }
    if not support_source_ids:
        raise ValueError("authoritative support surfaces contain no nodes")
    support_vertex_indices = tuple(sorted(local_index[node_id] for node_id in support_source_ids))

    return FingertipMechanicsMesh(
        tet_mesh=TetMeshData(coordinates, local_tetrahedra),
        source_node_ids=np.asarray(source_node_ids, dtype=np.int64),
        support_vertex_indices=support_vertex_indices,
        surface_triangles=local_surfaces,
        morphology_fingerprint=volume_mesh.morphology_fingerprint,
    )


__all__ = ["FingertipMechanicsMesh", "prepare_fingertip_mechanics_mesh"]
