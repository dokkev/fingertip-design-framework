"""Public neutral solver boundary for the Newton VBD backend.

The current backend/runtime is Newton and the selected solver is ``SolverVBD``.
Those implementation identities remain private to the backend; callers receive
only the repository-neutral NumPy result contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts.types import NewtonResult, TetMeshData


class PhysicsDependencyError(RuntimeError):
    """Raised when the Newton/Warp execution backend is unavailable."""


def _load_newton_backend():
    """Load the optional Newton/Warp backend at the execution boundary."""

    try:
        from . import vbd
    except (ImportError, OSError) as exc:
        raise PhysicsDependencyError(
            "Newton/Warp backend could not be imported: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return vbd


@dataclass(frozen=True)
class NewtonSettings:
    """Small, deterministic settings surface for the VBD backend."""

    device: str = "cuda:0"
    dt: float = 1.0 / 60.0
    steps: int = 1
    iterations: int = 5
    gravity: float = -9.81
    density: float = 1.0e3
    k_mu: float = 1.0e5
    k_lambda: float = 1.0e5
    k_damp: float = 10.0
    fixed_vertex_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty device string")
        for name in ("steps", "iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("dt", "gravity", "density", "k_mu", "k_lambda", "k_damp"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.density <= 0.0 or self.k_mu <= 0.0 or self.k_lambda <= 0.0:
            raise ValueError("density and elastic parameters must be positive")
        if self.k_damp < 0.0:
            raise ValueError("k_damp must be non-negative")
        for index in self.fixed_vertex_indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("fixed_vertex_indices must contain integers")
        fixed = tuple(self.fixed_vertex_indices)
        if any(index < 0 for index in fixed) or len(set(fixed)) != len(fixed):
            raise ValueError("fixed_vertex_indices must be unique and non-negative")
        object.__setattr__(self, "fixed_vertex_indices", fixed)


def solve(
    mesh: TetMeshData,
    *,
    settings: NewtonSettings | None = None,
) -> NewtonResult:
    """Run one small Newton ``SolverVBD`` solve and return neutral NumPy arrays.

    Importing :mod:`lumo.physics` does not import Warp or Newton. The optional
    backend is loaded only when this function is called.
    """

    if not isinstance(mesh, TetMeshData):
        raise TypeError("mesh must be a TetMeshData instance")
    if settings is None:
        settings = NewtonSettings()

    backend = _load_newton_backend()
    return backend.solve_newton_vbd(mesh, settings)
