"""Persistent Newton VBD execution for one fixed neutral tetrahedral mesh."""

from __future__ import annotations

import time

import numpy as np

from ..contracts.load import ParticleLoad
from .solve import NewtonSettings, _load_newton_backend
from ..contracts.types import NewtonResult, TetMeshData


class NewtonSession:
    """Build one Newton model and run independent reset-and-solve evaluations."""

    def __init__(
        self,
        mesh: TetMeshData,
        settings: NewtonSettings | None = None,
    ) -> None:
        if not isinstance(mesh, TetMeshData):
            raise TypeError("mesh must be a TetMeshData instance")
        if settings is None:
            settings = NewtonSettings()

        backend = _load_newton_backend()
        wp = backend.wp

        started = time.perf_counter()
        context = backend._build_vbd_context(mesh, settings)
        wp.synchronize_device(context.device)
        self._context = context
        self._mesh = mesh
        self._settings = settings
        self._model_build_wall_s = time.perf_counter() - started
        self._session_creation_wall_s = self._model_build_wall_s

    @property
    def model_build_wall_s(self) -> float:
        """Synchronized one-time Newton model/solver construction time."""

        return float(self._model_build_wall_s)

    @property
    def session_creation_wall_s(self) -> float:
        """Synchronized model/solver construction time for this session."""

        return float(self._session_creation_wall_s)

    @property
    def settings(self) -> NewtonSettings:
        return self._settings

    @property
    def mesh(self) -> TetMeshData:
        return self._mesh

    def reset(self) -> None:
        """Restore both Newton states to the verified rest state."""

        backend = _load_newton_backend()
        backend._reset_vbd_context(self._context)

    def solve(self, load: ParticleLoad | None = None) -> NewtonResult:
        """Reset, ramp one load, and return a fresh neutral mechanics result."""

        if load is None:
            load = ParticleLoad.zero(load_steps=self._settings.steps)
        if not isinstance(load, ParticleLoad):
            raise TypeError("load must be ParticleLoad or None")
        if np.any(load.vertex_indices >= self._mesh.vertices.shape[0]):
            raise ValueError("ParticleLoad contains an out-of-range vertex index")

        backend = _load_newton_backend()
        result, _timing = backend._solve_vbd_context(
            self._context,
            self._mesh,
            self._settings,
            load,
        )
        return result

    def solve_with_timing(
        self,
        load: ParticleLoad | None = None,
    ) -> tuple[NewtonResult, dict[str, float | int | str]]:
        """Run ``solve`` while returning synchronized reset/solve timing."""

        if load is None:
            load = ParticleLoad.zero(load_steps=self._settings.steps)
        if not isinstance(load, ParticleLoad):
            raise TypeError("load must be ParticleLoad or None")
        if np.any(load.vertex_indices >= self._mesh.vertices.shape[0]):
            raise ValueError("ParticleLoad contains an out-of-range vertex index")

        backend = _load_newton_backend()
        return backend._solve_vbd_context(
            self._context,
            self._mesh,
            self._settings,
            load,
        )


__all__ = ["NewtonSession"]
