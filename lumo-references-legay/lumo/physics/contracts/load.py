"""Neutral particle-force contract for the 3D mechanics backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _readonly_array(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ParticleLoad:
    """Deterministic per-particle external forces in Newtons.

    ``vertex_indices`` are local zero-based mesh indices.  Each index occurs
    once; callers aggregate shared face contributions before constructing this
    contract.  ``load_steps`` controls linear force ramping in a session.
    """

    vertex_indices: np.ndarray
    forces_n: np.ndarray
    load_steps: int = 1

    def __post_init__(self) -> None:
        raw_indices = np.asarray(self.vertex_indices)
        if raw_indices.ndim != 1 or not np.issubdtype(raw_indices.dtype, np.integer):
            raise ValueError("vertex_indices must be a one-dimensional integer array")
        indices = np.asarray(raw_indices, dtype=np.int32)
        if np.any(indices < 0) or len(set(indices.tolist())) != len(indices):
            raise ValueError("vertex_indices must be unique and non-negative")

        forces = np.asarray(self.forces_n, dtype=np.float64)
        if forces.shape != (indices.shape[0], 3):
            raise ValueError("forces_n must have shape (len(vertex_indices), 3)")
        if not np.all(np.isfinite(forces)):
            raise ValueError("forces_n must contain only finite values")
        if int(self.load_steps) < 1:
            raise ValueError("load_steps must be positive")

        object.__setattr__(self, "vertex_indices", _readonly_array(indices, dtype=np.int32))
        object.__setattr__(self, "forces_n", _readonly_array(forces, dtype=np.float64))
        object.__setattr__(self, "load_steps", int(self.load_steps))

    @classmethod
    def zero(cls, *, load_steps: int = 1) -> "ParticleLoad":
        """Return an explicit zero-load contract for deterministic resets."""

        return cls(
            vertex_indices=np.empty(0, dtype=np.int32),
            forces_n=np.empty((0, 3), dtype=np.float64),
            load_steps=load_steps,
        )

    @property
    def resultant_force_n(self) -> np.ndarray:
        """Return the vector sum of the nodal forces as a fresh array."""

        return np.sum(np.asarray(self.forces_n), axis=0, dtype=np.float64)


__all__ = ["ParticleLoad"]
