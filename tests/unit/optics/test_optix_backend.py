from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from optics.transport3d.optix_backend import (
    OptixScene,
    Transport3DDependencyError,
)
from optics.optix.runtime import OptixRuntimeError


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
