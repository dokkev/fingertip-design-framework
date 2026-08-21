from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lumo.ray_tracing.optical_mechanics.optix_backend import (
    OptixScene,
    Transport3DDependencyError,
)
from lumo.ray_tracing.optix.runtime import OptixRuntime, OptixRuntimeError


def _surface() -> SimpleNamespace:
    return SimpleNamespace(
        vertices=np.zeros((3, 3), dtype=np.float32),
        faces=np.zeros((1, 3), dtype=np.uint32),
    )


@pytest.mark.parametrize("error", (Transport3DDependencyError("CUDA unavailable"), RuntimeError("backend bug")))
def test_optix_scene_does_not_reclassify_runtime_errors(error: Exception) -> None:
    class _Runtime:
        def build_gas(self, vertices, faces):
            raise error

    with pytest.raises(type(error), match=str(error)):
        OptixScene(_Runtime(), _surface(), _surface(), _surface())


def test_optix_scene_translates_explicit_runtime_operation_failure() -> None:
    class _Runtime:
        def build_gas(self, vertices, faces):
            raise OptixRuntimeError(
                "device synchronization failed",
                stage="optix_runtime_execution",
            )

    with pytest.raises(Transport3DDependencyError, match="device synchronization"):
        OptixScene(_Runtime(), _surface(), _surface(), _surface())


def test_trace_translates_contiguous_cuda_allocation_failure() -> None:
    runtime = object.__new__(OptixRuntime)

    def fail_allocation(*_args, **_kwargs):
        raise RuntimeError("CUDA out of memory")

    runtime.cp = SimpleNamespace(
        float32=np.float32,
        ascontiguousarray=fail_allocation,
    )

    with pytest.raises(OptixRuntimeError, match="contiguous origin allocation"):
        runtime.trace(
            1,
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            tmin=1.0e-5,
        )


def test_trace_translates_empty_cuda_allocation_failure() -> None:
    runtime = object.__new__(OptixRuntime)

    def fail_allocation(*_args, **_kwargs):
        raise RuntimeError("CUDA runtime unavailable")

    runtime.cp = SimpleNamespace(
        float32=np.float32,
        uint32=np.uint32,
        empty=fail_allocation,
    )

    with pytest.raises(OptixRuntimeError, match="empty trace allocation"):
        runtime.trace(
            1,
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            tmin=1.0e-5,
        )
