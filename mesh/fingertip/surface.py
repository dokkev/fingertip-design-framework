"""Neutral compliant-pad mesh and deformation view."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


class InvalidPadMesh(ValueError):
    """Raised when pad topology or a displacement field is invalid."""


def _integer_array(values: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype == np.bool_:
        raise InvalidPadMesh(f"{name} must contain integer indices")
    if np.issubdtype(raw.dtype, np.integer):
        return np.array(raw, dtype=np.int64, copy=True)
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidPadMesh(f"{name} must contain integer-valued entries") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(
        numeric == np.floor(numeric)
    ):
        raise InvalidPadMesh(f"{name} must contain integer-valued entries")
    return np.array(numeric, dtype=np.int64, copy=True)


def _signed_double_areas(
    coordinates: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    first = coordinates[triangles[:, 0]]
    second = coordinates[triangles[:, 1]]
    third = coordinates[triangles[:, 2]]
    return (
        (second[:, 0] - first[:, 0]) * (third[:, 1] - first[:, 1])
        - (second[:, 1] - first[:, 1]) * (third[:, 0] - first[:, 0])
    )


def _area_tolerance(coordinates: np.ndarray) -> float:
    spans = np.ptp(coordinates, axis=0)
    length_scale = max(1.0, float(np.max(spans)))
    return 64.0 * np.finfo(float).eps * length_scale**2


def _normalized_triangles(
    coordinates: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    normalized = np.array(triangles, dtype=np.int64, copy=True)
    areas = _signed_double_areas(coordinates, normalized)
    if np.any(np.abs(areas) <= _area_tolerance(coordinates)):
        raise InvalidPadMesh("mesh contains a degenerate triangle")
    clockwise = areas < 0.0
    normalized[clockwise, 1], normalized[clockwise, 2] = (
        normalized[clockwise, 2].copy(),
        normalized[clockwise, 1].copy(),
    )
    return normalized


def _boundary_edges(triangles: np.ndarray) -> np.ndarray:
    records: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for triangle in triangles:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            directed = int(first), int(second)
            key = min(directed), max(directed)
            count, inherited = records.get(key, (0, directed))
            if count == 2:
                raise InvalidPadMesh(
                    f"non-manifold edge {key} belongs to more than two triangles"
                )
            records[key] = count + 1, inherited
    boundary = [
        directed
        for _, (count, directed) in sorted(records.items())
        if count == 1
    ]
    if not boundary:
        raise InvalidPadMesh("mesh has no boundary edges")
    return np.asarray(boundary, dtype=np.int64)


def _immutable_boundaries(
    boundary_edges: np.ndarray,
    boundaries: Mapping[str, np.ndarray],
    *,
    node_count: int,
) -> Mapping[str, np.ndarray]:
    supplied = dict(boundaries)
    if not supplied:
        return MappingProxyType({})
    if any(not isinstance(tag, str) or not tag for tag in supplied):
        raise InvalidPadMesh("boundary tags must be nonempty strings")
    canonical = {
        (min(int(first), int(second)), max(int(first), int(second))): (
            int(first),
            int(second),
        )
        for first, second in boundary_edges
    }
    owners: dict[tuple[int, int], str] = {}
    result: dict[str, np.ndarray] = {}
    for tag in sorted(supplied):
        edges = _integer_array(supplied[tag], name=f"boundaries[{tag!r}]")
        if edges.ndim != 2 or edges.shape[1:] != (2,) or len(edges) == 0:
            raise InvalidPadMesh(
                f"boundary tag {tag!r} must have nonempty shape (K, 2)"
            )
        if np.any(edges < 0) or np.any(edges >= node_count):
            raise InvalidPadMesh(f"boundary tag {tag!r} has an invalid node index")
        canonical_edges: list[tuple[int, int]] = []
        within_tag: set[tuple[int, int]] = set()
        for first, second in edges:
            key = min(int(first), int(second)), max(int(first), int(second))
            if key not in canonical:
                raise InvalidPadMesh(
                    f"boundary tag {tag!r} contains non-boundary edge {key}"
                )
            if key in within_tag:
                raise InvalidPadMesh(
                    f"boundary tag {tag!r} contains duplicate edge {key}"
                )
            if key in owners:
                raise InvalidPadMesh(
                    f"boundary edge {key} belongs to both {owners[key]!r} "
                    f"and {tag!r}"
                )
            within_tag.add(key)
            owners[key] = tag
            canonical_edges.append(canonical[key])
        stored = np.asarray(canonical_edges, dtype=np.int64)
        stored.setflags(write=False)
        result[tag] = stored
    missing = sorted(set(canonical).difference(owners))
    if missing:
        raise InvalidPadMesh(
            "tagged boundaries must own every mesh boundary edge; "
            f"missing edges: {missing}"
        )
    return MappingProxyType(result)


@dataclass(frozen=True)
class PadMesh:
    """Reference compliant-pad coordinates, topology, and boundaries."""

    node_ids: np.ndarray
    reference_coordinates_mm: np.ndarray
    triangles: np.ndarray
    boundary_edges: np.ndarray
    boundary_edges_by_tag: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        node_ids = _integer_array(self.node_ids, name="node_ids")
        coordinates = np.array(self.reference_coordinates_mm, dtype=float, copy=True)
        triangles = _integer_array(self.triangles, name="triangles")
        boundary_edges = _integer_array(self.boundary_edges, name="boundary_edges")
        if node_ids.ndim != 1 or len(np.unique(node_ids)) != len(node_ids):
            raise InvalidPadMesh("node_ids must be one-dimensional and unique")
        node_count = len(node_ids)
        if coordinates.shape != (node_count, 2):
            raise InvalidPadMesh("coordinates must have shape (N, 2)")
        if not np.all(np.isfinite(coordinates)):
            raise InvalidPadMesh("coordinates must contain finite values")
        if triangles.ndim != 2 or triangles.shape[1:] != (3,) or not len(triangles):
            raise InvalidPadMesh("triangles must have nonempty shape (M, 3)")
        if boundary_edges.ndim != 2 or boundary_edges.shape[1:] != (2,):
            raise InvalidPadMesh("boundary_edges must have shape (B, 2)")
        for name, indices in (
            ("triangles", triangles),
            ("boundary_edges", boundary_edges),
        ):
            if np.any(indices < 0) or np.any(indices >= node_count):
                raise InvalidPadMesh(f"{name} indices must lie within [0, N)")
        if np.any(
            (triangles[:, 0] == triangles[:, 1])
            | (triangles[:, 1] == triangles[:, 2])
            | (triangles[:, 2] == triangles[:, 0])
        ):
            raise InvalidPadMesh("a triangle repeats a node")
        triangles = _normalized_triangles(coordinates, triangles)
        expected_boundary = _boundary_edges(triangles)
        if {tuple(edge) for edge in boundary_edges} != {
            tuple(edge) for edge in expected_boundary
        } or len(boundary_edges) != len(expected_boundary):
            raise InvalidPadMesh(
                "boundary_edges must be the canonical oriented triangle boundary"
            )
        boundaries = _immutable_boundaries(
            boundary_edges,
            self.boundary_edges_by_tag,
            node_count=node_count,
        )
        for array in (node_ids, coordinates, triangles, boundary_edges):
            array.setflags(write=False)
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "reference_coordinates_mm", coordinates)
        object.__setattr__(self, "triangles", triangles)
        object.__setattr__(self, "boundary_edges", boundary_edges)
        object.__setattr__(self, "boundary_edges_by_tag", boundaries)

    @classmethod
    def from_arrays(
        cls,
        *,
        node_ids: np.ndarray,
        reference_coordinates_mm: np.ndarray,
        element_connectivity_node_ids: np.ndarray,
        boundary_edge_node_ids_by_tag: Mapping[str, np.ndarray] | None = None,
    ) -> PadMesh:
        """Map global node IDs to canonical local pad topology."""
        global_ids = _integer_array(node_ids, name="node_ids")
        connectivity = _integer_array(
            element_connectivity_node_ids,
            name="element_connectivity_node_ids",
        )
        if global_ids.ndim != 1 or len(np.unique(global_ids)) != len(global_ids):
            raise InvalidPadMesh("node_ids must be one-dimensional and unique")
        if connectivity.ndim != 2 or connectivity.shape[1:] != (3,):
            raise InvalidPadMesh(
                "element_connectivity_node_ids must have shape (M, 3)"
            )
        coordinates = np.asarray(reference_coordinates_mm, dtype=float)
        if coordinates.shape != (len(global_ids), 2):
            raise InvalidPadMesh("coordinates must have shape (N, 2)")
        if not np.all(np.isfinite(coordinates)):
            raise InvalidPadMesh("coordinates must contain finite values")
        id_to_local = {
            int(node_id): index
            for index, node_id in enumerate(global_ids)
        }
        try:
            local_triangles = np.asarray(
                [
                    [id_to_local[int(node_id)] for node_id in triangle]
                    for triangle in connectivity
                ],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise InvalidPadMesh(
                f"triangle references unknown node ID {exc.args[0]}"
            ) from exc
        normalized = _normalized_triangles(coordinates, local_triangles)
        boundary_edges = _boundary_edges(normalized)
        local_boundaries: dict[str, np.ndarray] = {}
        for tag, values in dict(boundary_edge_node_ids_by_tag or {}).items():
            global_edges = _integer_array(values, name=f"boundaries[{tag!r}]")
            if global_edges.ndim != 2 or global_edges.shape[1:] != (2,):
                raise InvalidPadMesh(f"boundary tag {tag!r} must have shape (K, 2)")
            try:
                local_boundaries[tag] = np.asarray(
                    [
                        [id_to_local[int(first)], id_to_local[int(second)]]
                        for first, second in global_edges
                    ],
                    dtype=np.int64,
                ).reshape((-1, 2))
            except KeyError as exc:
                raise InvalidPadMesh(
                    f"boundary tag {tag!r} references unknown node ID "
                    f"{exc.args[0]}"
                ) from exc
        return cls(
            node_ids=global_ids,
            reference_coordinates_mm=coordinates,
            triangles=normalized,
            boundary_edges=boundary_edges,
            boundary_edges_by_tag=local_boundaries,
        )

    @classmethod
    def from_fingertip_mesh(cls, mesh: Any) -> PadMesh:
        """Extract the compliant-pad topology from a neutral fingertip mesh."""
        pad_node_ids = sorted(
            {
                int(node_id)
                for element in mesh.pad_elements
                for node_id in element.node_ids
            }
        )
        pad_node_set = set(pad_node_ids)
        boundaries = {
            tag: np.asarray(selected, dtype=np.int64)
            for tag, edges in mesh.boundary_edges.items()
            if (
                selected := [
                    edge.node_ids
                    for edge in edges
                    if edge.domain == "pad"
                    and set(edge.node_ids).issubset(pad_node_set)
                ]
            )
        }
        return cls.from_arrays(
            node_ids=np.asarray(pad_node_ids, dtype=np.int64),
            reference_coordinates_mm=np.asarray(
                [
                    [mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm]
                    for node_id in pad_node_ids
                ],
                dtype=float,
            ),
            element_connectivity_node_ids=np.asarray(
                [element.node_ids for element in mesh.pad_elements],
                dtype=np.int64,
            ),
            boundary_edge_node_ids_by_tag=boundaries,
        )

    @property
    def coordinates(self) -> np.ndarray:
        """Return current x-y coordinates in millimeters."""
        return self.reference_coordinates_mm

    @property
    def boundaries(self) -> Mapping[str, np.ndarray]:
        """Return immutable semantic boundary-edge groups."""
        return self.boundary_edges_by_tag

    @property
    def semantic_boundary_tags(self) -> tuple[str, ...]:
        return tuple(self.boundaries)

    def boundary_edges_for(self, tag: str) -> np.ndarray:
        try:
            return self.boundaries[tag]
        except KeyError as exc:
            available = ", ".join(self.semantic_boundary_tags) or "<none>"
            raise KeyError(
                f"unknown boundary tag {tag!r}; available tags: {available}"
            ) from exc

    def boundary_node_indices_for(self, tag: str) -> np.ndarray:
        indices = np.unique(self.boundary_edges_for(tag).reshape(-1))
        indices.setflags(write=False)
        return indices
