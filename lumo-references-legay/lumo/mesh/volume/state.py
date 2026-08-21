"""Solver-neutral deformed state for one canonical fingertip volume mesh."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from lumo.mesh.volume.contracts import FingertipVolumeMesh, Tetrahedron


_TET_VOLUME_TOLERANCE_MM3 = 1.0e-12
_SURFACE_CROSS_TOLERANCE_MM2 = 1.0e-12


class InvalidDeformedFingertipState(ValueError):
    """Raised when solver output cannot define a valid deformed volume state."""


def _readonly_array(value: np.ndarray) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def _coordinates_for_node_ids(
    volume_mesh: FingertipVolumeMesh,
    node_ids: tuple[int, ...],
) -> np.ndarray:
    if not node_ids:
        raise ValueError("FingertipVolumeMesh must contain source nodes")
    if tuple(sorted(volume_mesh.nodes)) != node_ids:
        raise ValueError("volume mesh node order is not canonical")
    if any(int(node_id) != int(volume_mesh.nodes[node_id].id) for node_id in node_ids):
        raise ValueError("volume mesh node keys and source node IDs disagree")
    coordinates = np.asarray(
        [
            [
                volume_mesh.nodes[node_id].x_mm,
                volume_mesh.nodes[node_id].y_mm,
                volume_mesh.nodes[node_id].z_mm,
            ]
            for node_id in node_ids
        ],
        dtype=float,
    )
    if coordinates.shape != (len(node_ids), 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError("volume mesh reference coordinates must be finite with shape (N, 3)")
    return coordinates


def _validate_topology(
    volume_mesh: FingertipVolumeMesh,
    node_ids: tuple[int, ...],
    reference: np.ndarray,
    deformed: np.ndarray,
) -> None:
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    tetrahedra = tuple(volume_mesh.tetrahedra)
    if not tetrahedra:
        raise ValueError("FingertipVolumeMesh must contain tetrahedral topology")
    tetrahedron_ids = tuple(int(tetrahedron.id) for tetrahedron in tetrahedra)
    if len(set(tetrahedron_ids)) != len(tetrahedron_ids):
        raise ValueError("volume mesh tetrahedron IDs must be unique")
    if set(volume_mesh.volume_element_ids.get("pad", ())) != set(tetrahedron_ids):
        raise ValueError("volume mesh tetrahedral topology and element IDs disagree")

    for tetrahedron in tetrahedra:
        if len(tetrahedron.node_ids) != 4 or len(set(tetrahedron.node_ids)) != 4:
            raise ValueError("volume mesh contains an invalid tetrahedron")
        try:
            local = np.asarray([node_index[node_id] for node_id in tetrahedron.node_ids], dtype=np.int64)
        except KeyError as exc:
            raise ValueError("volume mesh tetrahedron references an unknown source node") from exc
        for coordinates, label in ((reference, "reference"), (deformed, "deformed")):
            points = coordinates[local]
            six_volume = float(
                np.dot(
                    np.cross(points[1] - points[0], points[2] - points[0]),
                    points[3] - points[0],
                )
            )
            if not np.isfinite(six_volume) or six_volume <= 6.0 * _TET_VOLUME_TOLERANCE_MM3:
                error_type = (
                    InvalidDeformedFingertipState
                    if label == "deformed"
                    else ValueError
                )
                raise error_type(
                    f"{label} volume mesh contains an inverted or degenerate tetrahedron"
                )

    tetrahedra_by_face: dict[tuple[int, int, int], list[Tetrahedron]] = {}
    for tetrahedron in tetrahedra:
        first, second, third, fourth = tetrahedron.node_ids
        for face in (
            (first, second, third),
            (first, second, fourth),
            (first, third, fourth),
            (second, third, fourth),
        ):
            tetrahedra_by_face.setdefault(tuple(sorted(face)), []).append(tetrahedron)

    if not volume_mesh.surface_triangles:
        raise ValueError("FingertipVolumeMesh must contain semantic surface topology")
    for tag, triangles in sorted(volume_mesh.surface_triangles.items()):
        if not isinstance(tag, str) or not tag or not triangles:
            raise ValueError("volume mesh semantic surface families must be non-empty")
        for triangle in triangles:
            if triangle.semantic_tag != tag or len(triangle.node_ids) != 3:
                raise ValueError(f"semantic surface {tag!r} contains invalid topology")
            if len(set(triangle.node_ids)) != 3:
                raise ValueError(f"semantic surface {tag!r} contains a degenerate triangle")
            try:
                local = np.asarray([node_index[node_id] for node_id in triangle.node_ids], dtype=np.int64)
            except KeyError as exc:
                raise ValueError(f"semantic surface {tag!r} references an unknown source node") from exc
            reference_cross = np.cross(
                reference[local[1]] - reference[local[0]],
                reference[local[2]] - reference[local[0]],
            )
            deformed_cross = np.cross(
                deformed[local[1]] - deformed[local[0]],
                deformed[local[2]] - deformed[local[0]],
            )
            reference_norm = float(np.linalg.norm(reference_cross))
            deformed_norm = float(np.linalg.norm(deformed_cross))
            if not np.isfinite(reference_norm) or reference_norm <= _SURFACE_CROSS_TOLERANCE_MM2:
                raise ValueError(f"semantic surface {tag!r} contains a degenerate triangle")
            if not np.isfinite(deformed_norm) or deformed_norm <= _SURFACE_CROSS_TOLERANCE_MM2:
                raise InvalidDeformedFingertipState(
                    f"deformed semantic surface {tag!r} contains a degenerate triangle"
                )
            if float(np.dot(reference_cross, deformed_cross)) <= 0.0:
                raise InvalidDeformedFingertipState(
                    f"semantic surface {tag!r} changed orientation"
                )
            adjacent = tetrahedra_by_face.get(tuple(sorted(triangle.node_ids)), [])
            if len(adjacent) != 1:
                raise ValueError(
                    f"semantic surface {tag!r} is not a unique tetrahedral boundary"
                )
            tetra_points = reference[
                np.asarray(
                    [node_index[node_id] for node_id in adjacent[0].node_ids],
                    dtype=np.int64,
                )
            ]
            outward = np.mean(reference[local], axis=0) - np.mean(
                tetra_points,
                axis=0,
            )
            if float(np.dot(reference_cross, outward)) <= _TET_VOLUME_TOLERANCE_MM3:
                raise ValueError(
                    f"semantic surface {tag!r} is not outward-oriented"
                )


@dataclass(frozen=True)
class FingertipVolumeState:
    """Canonical solver-neutral coordinates for one ``FingertipVolumeMesh``.

    The state owns only physical/discrete geometry.  Source node IDs and all
    topology are borrowed from the canonical volume mesh in deterministic
    sorted-node order; no state construction ever remeshes or reorders them.
    """

    volume_mesh: FingertipVolumeMesh
    deformed_coordinates_mm: np.ndarray
    _source_node_ids: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _reference_coordinates_mm: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.volume_mesh, FingertipVolumeMesh):
            raise TypeError("volume_mesh must be FingertipVolumeMesh")
        if not self.volume_mesh.validation.passed:
            raise ValueError(
                "refusing an invalid FingertipVolumeMesh: "
                + ", ".join(self.volume_mesh.validation.errors)
            )
        node_ids = tuple(sorted(self.volume_mesh.nodes))
        reference = _coordinates_for_node_ids(self.volume_mesh, node_ids)
        deformed = np.asarray(self.deformed_coordinates_mm, dtype=float)
        if deformed.shape != (len(node_ids), 3):
            raise InvalidDeformedFingertipState(
                "deformed_coordinates_mm must have shape (N, 3) in canonical node order"
            )
        if not np.all(np.isfinite(deformed)):
            raise InvalidDeformedFingertipState(
                "deformed_coordinates_mm must contain only finite coordinates"
            )
        _validate_topology(self.volume_mesh, node_ids, reference, deformed)
        object.__setattr__(self, "deformed_coordinates_mm", _readonly_array(deformed))
        object.__setattr__(self, "_source_node_ids", node_ids)
        object.__setattr__(self, "_reference_coordinates_mm", _readonly_array(reference))

    @classmethod
    def reference(cls, volume_mesh: FingertipVolumeMesh) -> "FingertipVolumeState":
        """Construct the identity state from the mesh's canonical coordinates."""
        if not isinstance(volume_mesh, FingertipVolumeMesh):
            raise TypeError("volume_mesh must be FingertipVolumeMesh")
        node_ids = tuple(sorted(volume_mesh.nodes))
        return cls(volume_mesh, _coordinates_for_node_ids(volume_mesh, node_ids))

    @classmethod
    def from_deformed_coordinates(
        cls,
        volume_mesh: FingertipVolumeMesh,
        deformed_coordinates_mm: np.ndarray,
    ) -> "FingertipVolumeState":
        """Construct and validate a state from canonical-order coordinates."""
        return cls(volume_mesh, deformed_coordinates_mm)

    @property
    def source_node_ids(self) -> tuple[int, ...]:
        """Return source Gmsh node IDs in exact canonical order."""
        return self._source_node_ids

    @property
    def canonical_node_ids(self) -> tuple[int, ...]:
        """Alias documenting that state rows follow sorted source IDs."""
        return self._source_node_ids

    @property
    def reference_coordinates_mm(self) -> np.ndarray:
        """Return immutable reference coordinates in canonical node order."""
        return self._reference_coordinates_mm

    @property
    def displacement_mm(self) -> np.ndarray:
        """Return deformed-minus-reference displacement in millimeters."""
        return _readonly_array(self.deformed_coordinates_mm - self._reference_coordinates_mm)

    @property
    def tetrahedra(self):
        """Return the canonical volume-mesh tetrahedral topology."""
        return self.volume_mesh.tetrahedra

    @property
    def surface_triangles(self):
        """Return the canonical semantic surface topology."""
        return MappingProxyType(self.volume_mesh.surface_triangles)

    @property
    def morphology_fingerprint(self) -> str:
        """Return the source morphology fingerprint."""
        return self.volume_mesh.morphology_fingerprint

    @property
    def settings(self):
        """Return volume-mesh tier/settings provenance."""
        return self.volume_mesh.settings


def make_fingertip_volume_state(
    volume_mesh: FingertipVolumeMesh,
    deformed_coordinates_mm: np.ndarray,
) -> FingertipVolumeState:
    """Create the one shared validated state representation."""
    return FingertipVolumeState.from_deformed_coordinates(
        volume_mesh,
        deformed_coordinates_mm,
    )


__all__ = [
    "FingertipVolumeState",
    "InvalidDeformedFingertipState",
    "make_fingertip_volume_state",
]
