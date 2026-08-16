"""Stable 2D-to-3D extrusion topology for optical rendering."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from mesh import PadMesh


class InvalidExtrudedOpticalMesh(ValueError):
    """Raised when an optical extrusion is malformed."""


@dataclass(frozen=True)
class _ExtrudedMesh:
    """Fixed face topology for a pad extruded through a constant depth."""

    depth_mm: float
    node_count_2d: int
    faces_3d: np.ndarray

    def __post_init__(self) -> None:
        """Own and validate the immutable face connectivity."""
        if not isfinite(self.depth_mm) or self.depth_mm <= 0.0:
            raise InvalidExtrudedOpticalMesh(
                "depth_mm must be finite and greater than zero"
            )
        if (
            not isinstance(self.node_count_2d, int)
            or isinstance(self.node_count_2d, bool)
            or self.node_count_2d < 3
        ):
            raise InvalidExtrudedOpticalMesh(
                "node_count_2d must be an integer of at least three"
            )
        raw_faces = np.asarray(self.faces_3d)
        if not np.issubdtype(raw_faces.dtype, np.integer):
            try:
                numeric = np.asarray(raw_faces, dtype=float)
            except (TypeError, ValueError) as exc:
                raise InvalidExtrudedOpticalMesh(
                    "faces_3d must contain integer-valued indices"
                ) from exc
            if (
                not np.all(np.isfinite(numeric))
                or not np.all(numeric == np.floor(numeric))
            ):
                raise InvalidExtrudedOpticalMesh(
                    "faces_3d must contain integer-valued indices"
                )
        faces = np.array(raw_faces, dtype=np.int64, copy=True)
        if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
            raise InvalidExtrudedOpticalMesh(
                "faces_3d must have nonempty shape (F, 3)"
            )
        if np.any(faces < 0) or np.any(faces >= 2 * self.node_count_2d):
            raise InvalidExtrudedOpticalMesh(
                "faces_3d indices must lie within [0, 2N)"
            )
        if np.any(
            (faces[:, 0] == faces[:, 1])
            | (faces[:, 1] == faces[:, 2])
            | (faces[:, 2] == faces[:, 0])
        ):
            raise InvalidExtrudedOpticalMesh("an extrusion face repeats a node")
        edge_counts: dict[tuple[int, int], int] = {}
        for triangle in faces:
            for first, second in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ):
                edge = (min(int(first), int(second)), max(int(first), int(second)))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        if any(count != 2 for count in edge_counts.values()):
            raise InvalidExtrudedOpticalMesh(
                "faces_3d must form a watertight two-manifold"
            )
        faces.setflags(write=False)
        object.__setattr__(self, "faces_3d", faces)

    @classmethod
    def from_pad_mesh(
        cls,
        mesh: PadMesh,
        *,
        depth_mm: float,
    ) -> _ExtrudedMesh:
        """Create fixed caps and side faces from oriented 2D topology."""
        node_count = len(mesh.node_ids)
        rear_faces = mesh.triangles[:, [0, 2, 1]]
        front_faces = mesh.triangles + node_count
        side_faces: list[tuple[int, int, int]] = []
        for first, second in mesh.boundary_edges:
            i = int(first)
            j = int(second)
            side_faces.extend(
                (
                    (i, j, j + node_count),
                    (i, j + node_count, i + node_count),
                )
            )
        faces = np.vstack(
            (
                rear_faces,
                front_faces,
                np.asarray(side_faces, dtype=np.int64),
            )
        )
        return cls(
            depth_mm=depth_mm,
            node_count_2d=node_count,
            faces_3d=faces,
        )

    def vertices_for_coordinates(
        self,
        coordinates_mm: np.ndarray,
    ) -> np.ndarray:
        """Extrude x-y coordinates into the stable rear/front vertex layout."""
        coordinates = np.array(coordinates_mm, dtype=float, copy=True)
        if coordinates.shape != (self.node_count_2d, 2):
            raise InvalidExtrudedOpticalMesh(
                "coordinates_mm must have shape (node_count_2d, 2)"
            )
        if not np.all(np.isfinite(coordinates)):
            raise InvalidExtrudedOpticalMesh(
                "coordinates_mm must contain only finite values"
            )
        half_depth = self.depth_mm / 2.0
        vertices = np.empty((2 * self.node_count_2d, 3), dtype=float)
        vertices[: self.node_count_2d, :2] = coordinates
        vertices[: self.node_count_2d, 2] = -half_depth
        vertices[self.node_count_2d :, :2] = coordinates
        vertices[self.node_count_2d :, 2] = half_depth
        vertices.setflags(write=False)
        return vertices

    def vertices_for_mesh(
        self,
        mesh: Any,
    ) -> np.ndarray:
        """Extrude one mesh view without changing face topology."""
        if len(mesh.node_ids) != self.node_count_2d:
            raise InvalidExtrudedOpticalMesh(
                "mesh node count does not match the extrusion"
            )
        return self.vertices_for_coordinates(mesh.coordinates)

    def side_faces_for_edges(
        self,
        edges: np.ndarray,
    ) -> np.ndarray:
        """Return the two longitudinal side triangles for each 2D edge.

        The face convention is the same one used by :meth:`from_pad_mesh`.
        Keeping this small selector here lets periodic transport reuse the
        established extrusion vertex layout without treating the numerical
        z-caps as physical surfaces.
        """
        raw_edges = np.asarray(edges)
        if raw_edges.ndim != 2 or raw_edges.shape[1:] != (2,):
            raise InvalidExtrudedOpticalMesh(
                "edges must have shape (K, 2)"
            )
        if len(raw_edges) == 0:
            raise InvalidExtrudedOpticalMesh("edges must be nonempty")
        if not np.issubdtype(raw_edges.dtype, np.integer):
            try:
                numeric = np.asarray(raw_edges, dtype=float)
            except (TypeError, ValueError) as exc:
                raise InvalidExtrudedOpticalMesh(
                    "edges must contain integer-valued indices"
                ) from exc
            if (
                not np.all(np.isfinite(numeric))
                or not np.all(numeric == np.floor(numeric))
            ):
                raise InvalidExtrudedOpticalMesh(
                    "edges must contain integer-valued indices"
                )
        selected = np.asarray(raw_edges, dtype=np.int64)
        if np.any(selected < 0) or np.any(selected >= self.node_count_2d):
            raise InvalidExtrudedOpticalMesh(
                "edges must reference the 2D extrusion node range"
            )
        if np.any(selected[:, 0] == selected[:, 1]):
            raise InvalidExtrudedOpticalMesh("edges must not repeat a node")
        offset = self.node_count_2d
        faces = np.asarray(
            [
                face
                for first, second in selected
                for face in (
                    (first, second, second + offset),
                    (first, second + offset, first + offset),
                )
            ],
            dtype=np.int64,
        )
        faces.setflags(write=False)
        return faces
