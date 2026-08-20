"""Neutral data contracts for the optional 3D mechanics surrogate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _readonly_array(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TetMeshData:
    """A validated tetrahedral mesh represented only by NumPy arrays.

    Coordinates use the repository's millimetre convention. The optional
    Newton backend converts them to metres at its solver boundary.
    """

    vertices: np.ndarray
    tetrahedra: np.ndarray

    def __post_init__(self) -> None:
        raw_vertices = np.asarray(self.vertices)
        raw_tetrahedra = np.asarray(self.tetrahedra)

        if raw_vertices.ndim != 2 or raw_vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (n_vertices, 3)")
        if raw_tetrahedra.ndim != 2 or raw_tetrahedra.shape[1] != 4:
            raise ValueError("tetrahedra must have shape (n_tetrahedra, 4)")
        if raw_vertices.shape[0] < 4 or raw_tetrahedra.shape[0] < 1:
            raise ValueError("mesh must contain vertices and at least one tetrahedron")
        if not np.issubdtype(raw_tetrahedra.dtype, np.integer):
            raise ValueError("tetrahedra must contain integer vertex indices")

        vertices = np.asarray(raw_vertices, dtype=np.float32)
        tetrahedra = np.asarray(raw_tetrahedra, dtype=np.int32)
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices must be finite")
        if np.any(tetrahedra < 0) or np.any(tetrahedra >= vertices.shape[0]):
            raise ValueError("tetrahedra contain an out-of-range vertex index")

        a = vertices[tetrahedra[:, 0]]
        b = vertices[tetrahedra[:, 1]]
        c = vertices[tetrahedra[:, 2]]
        d = vertices[tetrahedra[:, 3]]
        six_volumes = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)
        if np.any(~np.isfinite(six_volumes)) or np.any(np.abs(six_volumes) <= 1.0e-12):
            raise ValueError("tetrahedra must be non-degenerate")

        object.__setattr__(self, "vertices", _readonly_array(vertices, dtype=np.float32))
        object.__setattr__(self, "tetrahedra", _readonly_array(tetrahedra, dtype=np.int32))


@dataclass(frozen=True)
class NewtonResult:
    """Neutral rest/deformed coordinates returned by a 3D mechanics solve."""

    rest_vertices: np.ndarray
    deformed_vertices: np.ndarray
    tetrahedra: np.ndarray
    steps: int

    def __post_init__(self) -> None:
        rest = np.asarray(self.rest_vertices, dtype=np.float32)
        deformed = np.asarray(self.deformed_vertices, dtype=np.float32)
        raw_tetrahedra = np.asarray(self.tetrahedra)
        tetrahedra = np.asarray(raw_tetrahedra, dtype=np.int32)
        if rest.shape != deformed.shape or rest.ndim != 2 or rest.shape[1] != 3:
            raise ValueError("rest_vertices and deformed_vertices must share shape (n_vertices, 3)")
        if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
            raise ValueError("tetrahedra must have shape (n_tetrahedra, 4)")
        if not np.issubdtype(raw_tetrahedra.dtype, np.integer):
            raise ValueError("tetrahedra must contain integer vertex indices")
        if np.any(tetrahedra < 0) or np.any(tetrahedra >= rest.shape[0]):
            raise ValueError("tetrahedra contain an out-of-range vertex index")
        if not np.all(np.isfinite(rest)) or not np.all(np.isfinite(deformed)):
            raise ValueError("mechanics result coordinates must be finite")
        if int(self.steps) < 1:
            raise ValueError("steps must be positive")
        object.__setattr__(self, "rest_vertices", _readonly_array(rest, dtype=np.float32))
        object.__setattr__(self, "deformed_vertices", _readonly_array(deformed, dtype=np.float32))
        object.__setattr__(self, "tetrahedra", _readonly_array(tetrahedra, dtype=np.int32))
        object.__setattr__(self, "steps", int(self.steps))

    @property
    def displacement(self) -> np.ndarray:
        """Return the deformed-minus-rest displacement as a fresh NumPy array."""

        return np.asarray(self.deformed_vertices - self.rest_vertices, dtype=np.float32)
