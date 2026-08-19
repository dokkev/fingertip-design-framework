"""Public solver boundary for the optional Newton VBD surrogate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Mechanics3DResult, TetMeshData


@dataclass(frozen=True)
class Mechanics3DSettings:
    """Small, deterministic settings surface for the VBD prototype."""

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
        for name in ("dt", "gravity", "density", "k_mu", "k_lambda", "k_damp"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.dt <= 0.0 or self.steps < 1 or self.iterations < 1:
            raise ValueError("dt, steps, and iterations must be positive")
        if self.density <= 0.0 or self.k_mu <= 0.0 or self.k_lambda <= 0.0:
            raise ValueError("density and elastic parameters must be positive")
        if self.k_damp < 0.0:
            raise ValueError("k_damp must be non-negative")
        fixed = tuple(int(index) for index in self.fixed_vertex_indices)
        if any(index < 0 for index in fixed) or len(set(fixed)) != len(fixed):
            raise ValueError("fixed_vertex_indices must be unique and non-negative")
        object.__setattr__(self, "fixed_vertex_indices", fixed)


def solve(
    mesh: TetMeshData,
    *,
    settings: Mechanics3DSettings | None = None,
) -> Mechanics3DResult:
    """Run one small Newton VBD solve and return neutral NumPy arrays.

    Importing :mod:`mechanics3d` does not import Warp or Newton.  The optional
    backend is loaded only when this function is called.
    """

    if not isinstance(mesh, TetMeshData):
        raise TypeError("mesh must be a TetMeshData instance")
    if settings is None:
        settings = Mechanics3DSettings()

    from .backends.newton_vbd import solve_newton_vbd

    return solve_newton_vbd(mesh, settings)
