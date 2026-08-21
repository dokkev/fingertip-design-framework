"""OptiX transport adapter over the shared optional runtime boundary."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from lumo.ray_tracing.optix.runtime import OptixRuntime, OptixRuntimeError
from lumo.ray_tracing.optical_mechanics.geometry import TriangleSurface


class Transport3DDependencyError(RuntimeError):
    """Raised when the optional CUDA/OptiX runtime is unavailable."""


class Transport3DTraceError(RuntimeError):
    """Raised when OptiX cannot return a valid traversal result."""


_DEVICE_SOURCE = Path(__file__).with_name("kernels").joinpath("transport.cu").read_text()
_PARAMS_DTYPE = np.dtype(
    [
        ("handle", np.uint64),
        ("origins", np.uint64),
        ("directions", np.uint64),
        ("distances", np.uint64),
        ("primitives", np.uint64),
        ("barycentrics", np.uint64),
        ("hits", np.uint64),
        ("count", np.uint32),
        ("tmin", np.float32),
    ],
    align=True,
)


def create_runtime() -> OptixRuntime:
    """Create the configured runtime for the deferred 3D transport kernel."""
    try:
        return OptixRuntime.create(
            device_source=_DEVICE_SOURCE,
            source_name="optical_mechanics.cu",
            raygen_entry="__raygen__transport",
            miss_entry="__miss__transport",
            hitgroup_entry="__closesthit__transport",
            params_dtype=_PARAMS_DTYPE,
            num_payload_values=0,
            num_attribute_values=2,
        )
    except OptixRuntimeError as exc:
        raise Transport3DDependencyError(str(exc)) from exc


class OptixScene:
    """One rebuilt GAS collection for one deformed volume state."""

    def __init__(
        self,
        runtime: OptixRuntime,
        silicone: TriangleSurface,
        rigid: TriangleSurface,
        envelope: TriangleSurface,
    ) -> None:
        import time

        started = time.perf_counter()
        self.runtime = runtime
        self.handles = {}
        self._owners: list[object] = []
        for name, surface in (
            ("silicone", silicone),
            ("rigid", rigid),
            ("envelope", envelope),
        ):
            try:
                handle, owners = runtime.build_gas(surface.vertices, surface.faces)
            except OptixRuntimeError as exc:
                raise Transport3DDependencyError(
                    f"OptiX GAS construction failed: {exc}"
                ) from exc
            self.handles[name] = handle
            self._owners.extend(owners)
        self.gas_build_seconds = time.perf_counter() - started

    def trace(
        self,
        name: str,
        origins: object,
        directions: object,
        *,
        tmin: float,
    ) -> tuple[object, object, object, object]:
        try:
            return self.runtime.trace(
                self.handles[name], origins, directions, tmin=tmin
            )
        except OptixRuntimeError as exc:
            raise Transport3DDependencyError(
                f"OptiX {name} traversal failed: {exc}"
            ) from exc

__all__ = [
    "OptixRuntime",
    "OptixScene",
    "Transport3DDependencyError",
    "Transport3DTraceError",
    "create_runtime",
]
