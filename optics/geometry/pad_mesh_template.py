"""Immutable reference topology for deformation-aware optical geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from optics.geometry.deformation_state import PadDeformationState2D


class InvalidPadMeshTemplate(ValueError):
    """Raised when reference pad topology is invalid or non-manifold."""


def _integer_array(values: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype == np.bool_:
        raise InvalidPadMeshTemplate(f"{name} must contain integer indices")
    if np.issubdtype(raw.dtype, np.integer):
        return np.array(raw, dtype=np.int64, copy=True)
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidPadMeshTemplate(
            f"{name} must contain integer-valued entries"
        ) from exc
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise InvalidPadMeshTemplate(
            f"{name} must contain finite integer-valued entries"
        )
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


def _normalize_triangles(
    coordinates: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    normalized = np.array(triangles, dtype=np.int64, copy=True)
    areas = _signed_double_areas(coordinates, normalized)
    tolerance = _area_tolerance(coordinates)
    if np.any(np.abs(areas) <= tolerance):
        raise InvalidPadMeshTemplate(
            "reference mesh contains a degenerate triangle"
        )
    clockwise = areas < 0.0
    normalized[clockwise, 1], normalized[clockwise, 2] = (
        normalized[clockwise, 2].copy(),
        normalized[clockwise, 1].copy(),
    )
    return normalized


def _derive_boundary_edges(triangles: np.ndarray) -> np.ndarray:
    edge_records: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for triangle in triangles:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            directed = (int(first), int(second))
            key = (min(directed), max(directed))
            count, inherited = edge_records.get(key, (0, directed))
            count += 1
            if count > 2:
                raise InvalidPadMeshTemplate(
                    f"non-manifold edge {key} belongs to more than two triangles"
                )
            edge_records[key] = (count, inherited)
    boundary = [
        inherited
        for key, (count, inherited) in sorted(edge_records.items())
        if count == 1
    ]
    if not boundary:
        raise InvalidPadMeshTemplate("reference mesh has no boundary edges")
    return np.asarray(boundary, dtype=np.int64)


def _immutable_semantic_boundary_partition(
    boundary_edges: np.ndarray,
    boundary_edges_by_tag: Mapping[str, np.ndarray],
    *,
    node_count: int,
) -> Mapping[str, np.ndarray]:
    """Canonicalize and validate a complete semantic boundary partition."""
    supplied = dict(boundary_edges_by_tag)
    if not supplied:
        return MappingProxyType({})
    if any(not isinstance(tag, str) or not tag for tag in supplied):
        raise InvalidPadMeshTemplate(
            "semantic boundary tags must be nonempty strings"
        )

    canonical_by_key = {
        (min(int(first), int(second)), max(int(first), int(second))): (
            int(first),
            int(second),
        )
        for first, second in boundary_edges
    }
    owner_by_key: dict[tuple[int, int], str] = {}
    normalized: dict[str, np.ndarray] = {}
    for tag in sorted(supplied):
        edges = _integer_array(
            supplied[tag],
            name=f"boundary_edges_by_tag[{tag!r}]",
        )
        if edges.ndim != 2 or edges.shape[1:] != (2,):
            raise InvalidPadMeshTemplate(
                f"semantic boundary tag {tag!r} must have shape (K, 2)"
            )
        if len(edges) == 0:
            raise InvalidPadMeshTemplate(
                f"semantic boundary tag {tag!r} must not be empty"
            )
        if np.any(edges < 0) or np.any(edges >= node_count):
            raise InvalidPadMeshTemplate(
                f"semantic boundary tag {tag!r} contains an invalid node index"
            )
        if np.any(edges[:, 0] == edges[:, 1]):
            raise InvalidPadMeshTemplate(
                f"semantic boundary tag {tag!r} contains a repeated node"
            )

        canonical_edges: list[tuple[int, int]] = []
        keys_in_tag: set[tuple[int, int]] = set()
        for first, second in edges:
            key = (min(int(first), int(second)), max(int(first), int(second)))
            if key not in canonical_by_key:
                raise InvalidPadMeshTemplate(
                    f"semantic boundary tag {tag!r} contains non-boundary "
                    f"edge {key}"
                )
            if key in keys_in_tag:
                raise InvalidPadMeshTemplate(
                    f"semantic boundary tag {tag!r} contains duplicate edge {key}"
                )
            if key in owner_by_key:
                raise InvalidPadMeshTemplate(
                    f"semantic boundary edge {key} belongs to both "
                    f"{owner_by_key[key]!r} and {tag!r}"
                )
            keys_in_tag.add(key)
            owner_by_key[key] = tag
            canonical_edges.append(canonical_by_key[key])

        canonical_array = np.asarray(canonical_edges, dtype=np.int64)
        canonical_array.setflags(write=False)
        normalized[tag] = canonical_array

    missing = sorted(set(canonical_by_key).difference(owner_by_key))
    if missing:
        raise InvalidPadMeshTemplate(
            "nonempty semantic boundary partition does not own every mesh "
            f"boundary edge; missing edges: {missing}"
        )
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class PadMeshTemplate2D:
    """Reference node coordinates and fixed triangular pad topology."""

    node_ids: np.ndarray
    reference_coordinates_mm: np.ndarray
    triangles: np.ndarray
    boundary_edges: np.ndarray
    boundary_edges_by_tag: Mapping[str, np.ndarray] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Validate, normalize, and own immutable topology arrays."""
        node_ids = _integer_array(self.node_ids, name="node_ids")
        coordinates = np.array(
            self.reference_coordinates_mm,
            dtype=float,
            copy=True,
        )
        triangles = _integer_array(self.triangles, name="triangles")
        boundary_edges = _integer_array(
            self.boundary_edges,
            name="boundary_edges",
        )
        if node_ids.ndim != 1:
            raise InvalidPadMeshTemplate("node_ids must be one-dimensional")
        if len(np.unique(node_ids)) != len(node_ids):
            raise InvalidPadMeshTemplate("node_ids must be unique")
        node_count = len(node_ids)
        if coordinates.shape != (node_count, 2):
            raise InvalidPadMeshTemplate(
                "reference_coordinates_mm must have shape (N, 2)"
            )
        if not np.all(np.isfinite(coordinates)):
            raise InvalidPadMeshTemplate(
                "reference_coordinates_mm must contain finite values"
            )
        if triangles.ndim != 2 or triangles.shape[1:] != (3,):
            raise InvalidPadMeshTemplate("triangles must have shape (M, 3)")
        if boundary_edges.ndim != 2 or boundary_edges.shape[1:] != (2,):
            raise InvalidPadMeshTemplate(
                "boundary_edges must have shape (B, 2)"
            )
        if len(triangles) == 0:
            raise InvalidPadMeshTemplate("triangles must not be empty")
        for name, indices in (
            ("triangles", triangles),
            ("boundary_edges", boundary_edges),
        ):
            if np.any(indices < 0) or np.any(indices >= node_count):
                raise InvalidPadMeshTemplate(
                    f"{name} indices must lie within [0, N)"
                )
        if np.any(
            (triangles[:, 0] == triangles[:, 1])
            | (triangles[:, 1] == triangles[:, 2])
            | (triangles[:, 2] == triangles[:, 0])
        ):
            raise InvalidPadMeshTemplate("a triangle repeats a node")
        if np.any(boundary_edges[:, 0] == boundary_edges[:, 1]):
            raise InvalidPadMeshTemplate("a boundary edge repeats a node")

        triangles = _normalize_triangles(coordinates, triangles)
        expected_boundary_edges = _derive_boundary_edges(triangles)
        boundary_keys = [
            (int(first), int(second)) for first, second in boundary_edges
        ]
        if len(boundary_keys) != len(set(boundary_keys)):
            raise InvalidPadMeshTemplate("boundary_edges contains duplicates")
        if set(boundary_keys) != {
            (int(first), int(second))
            for first, second in expected_boundary_edges
        }:
            raise InvalidPadMeshTemplate(
                "boundary_edges must be the oriented boundary of triangles"
            )
        semantic_boundaries = _immutable_semantic_boundary_partition(
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
        object.__setattr__(
            self,
            "boundary_edges_by_tag",
            semantic_boundaries,
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        node_ids: np.ndarray,
        reference_coordinates_mm: np.ndarray,
        element_connectivity_node_ids: np.ndarray,
        boundary_edge_node_ids_by_tag: Mapping[str, np.ndarray] | None = None,
    ) -> PadMeshTemplate2D:
        """Map global node IDs to normalized local triangle topology."""
        global_node_ids = _integer_array(node_ids, name="node_ids")
        connectivity = _integer_array(
            element_connectivity_node_ids,
            name="element_connectivity_node_ids",
        )
        if global_node_ids.ndim != 1:
            raise InvalidPadMeshTemplate("node_ids must be one-dimensional")
        if len(np.unique(global_node_ids)) != len(global_node_ids):
            raise InvalidPadMeshTemplate("node_ids must be unique")
        if connectivity.ndim != 2 or connectivity.shape[1:] != (3,):
            raise InvalidPadMeshTemplate(
                "element_connectivity_node_ids must have shape (M, 3)"
            )
        id_to_local = {
            int(node_id): index
            for index, node_id in enumerate(global_node_ids)
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
            raise InvalidPadMeshTemplate(
                f"element connectivity references unknown node ID {exc.args[0]}"
            ) from exc
        coordinates = np.asarray(reference_coordinates_mm, dtype=float)
        if coordinates.shape != (len(global_node_ids), 2):
            raise InvalidPadMeshTemplate(
                "reference_coordinates_mm must have shape (N, 2)"
            )
        normalized = _normalize_triangles(coordinates, local_triangles)
        boundary_edges = _derive_boundary_edges(normalized)
        local_edges_by_tag: dict[str, np.ndarray] = {}
        for tag, global_edges_value in dict(
            boundary_edge_node_ids_by_tag or {}
        ).items():
            global_edges = _integer_array(
                global_edges_value,
                name=f"boundary_edge_node_ids_by_tag[{tag!r}]",
            )
            if global_edges.ndim != 2 or global_edges.shape[1:] != (2,):
                raise InvalidPadMeshTemplate(
                    f"semantic boundary tag {tag!r} must have shape (K, 2)"
                )
            try:
                local_edges_by_tag[tag] = np.asarray(
                    [
                        [id_to_local[int(first)], id_to_local[int(second)]]
                        for first, second in global_edges
                    ],
                    dtype=np.int64,
                ).reshape((-1, 2))
            except KeyError as exc:
                raise InvalidPadMeshTemplate(
                    f"semantic boundary tag {tag!r} references unknown "
                    f"node ID {exc.args[0]}"
                ) from exc
        return cls(
            node_ids=global_node_ids,
            reference_coordinates_mm=coordinates,
            triangles=normalized,
            boundary_edges=boundary_edges,
            boundary_edges_by_tag=local_edges_by_tag,
        )

    @property
    def semantic_boundary_tags(self) -> tuple[str, ...]:
        """Return semantic boundary tags in deterministic sorted order."""
        return tuple(self.boundary_edges_by_tag)

    def boundary_edges_for(self, tag: str) -> np.ndarray:
        """Return immutable canonical local edges for one semantic tag."""
        try:
            return self.boundary_edges_by_tag[tag]
        except KeyError as exc:
            available = ", ".join(self.semantic_boundary_tags) or "<none>"
            raise KeyError(
                f"unknown semantic boundary tag {tag!r}; available tags: "
                f"{available}"
            ) from exc

    def boundary_node_indices_for(self, tag: str) -> np.ndarray:
        """Return immutable sorted local node indices used by one tag."""
        indices = np.unique(self.boundary_edges_for(tag).reshape(-1))
        indices.setflags(write=False)
        return indices

    def validate_state(self, state: PadDeformationState2D) -> None:
        """Reject displacement states that mismatch or invert this topology."""
        if state.displacement_mm.shape != self.reference_coordinates_mm.shape:
            raise InvalidPadMeshTemplate(
                "deformation state shape does not match the mesh template"
            )
        if not np.all(np.isfinite(state.displacement_mm)):
            raise InvalidPadMeshTemplate(
                "deformation state contains non-finite values"
            )
        loaded = self.reference_coordinates_mm + state.displacement_mm
        loaded_areas = _signed_double_areas(loaded, self.triangles)
        tolerance = _area_tolerance(loaded)
        if np.any(loaded_areas <= tolerance):
            raise InvalidPadMeshTemplate(
                "deformation state creates a degenerate or inverted triangle"
            )

    def coordinates_for(self, state: PadDeformationState2D) -> np.ndarray:
        """Return immutable loaded coordinates for ``state``."""
        self.validate_state(state)
        coordinates = np.array(
            self.reference_coordinates_mm + state.displacement_mm,
            dtype=float,
            copy=True,
        )
        coordinates.setflags(write=False)
        return coordinates
