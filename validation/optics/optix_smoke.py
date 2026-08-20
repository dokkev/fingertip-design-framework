"""Real CUDA/OptiX environment verification with one deterministic launch."""

from __future__ import annotations

from dataclasses import dataclass
import sys
import time
from typing import Any, Mapping

import numpy as np

from optics.optix.runtime import OptixRuntime, OptixRuntimeError


_DEVICE_SOURCE = r"""
#include <optix.h>
#include <optix_device.h>

struct Params {
    OptixTraversableHandle handle;
    unsigned int* result;
    float3 origin;
    float3 direction;
};

extern "C" {
__constant__ Params params;

extern "C" __global__ void __raygen__raygen_program()
{
    unsigned int payload = 0;
    optixTrace(
        params.handle,
        params.origin,
        params.direction,
        0.0f,
        10.0f,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,
        1,
        0,
        payload
    );
}

extern "C" __global__ void __miss__miss_program()
{
    params.result[0] = 0;
    params.result[1] = 0;
    params.result[2] = 0;
}

extern "C" __global__ void __closesthit__closesthit_program()
{
    params.result[0] = 1;
    params.result[1] = optixGetPrimitiveIndex();
    params.result[2] = __float_as_uint(optixGetRayTmax());
}
}
"""

_PARAMS_DTYPE = np.dtype(
    [
        ("handle", np.uint64),
        ("result", np.uint64),
        ("origin", np.float32, (3,)),
        ("direction", np.float32, (3,)),
    ],
    align=True,
)


class OptixSmokeError(RuntimeError):
    """A staged failure from the real OptiX environment contract."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True)
class OptixSmokeResult:
    """Compact evidence from one deterministic hit/miss launch pair."""

    metadata: Mapping[str, Any]
    hit: tuple[int, int, float]
    miss: tuple[int, int, float]
    setup_time_seconds: float
    trace_time_seconds: float
    ray_count: int
    terminal_event_counts: Mapping[str, int]
    result_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "hit": list(self.hit),
            "miss": list(self.miss),
            "setup_time_seconds": self.setup_time_seconds,
            "trace_time_seconds": self.trace_time_seconds,
            "ray_count": self.ray_count,
            "terminal_event_counts": dict(self.terminal_event_counts),
            "result_counts": dict(self.result_counts),
        }


def _runtime_stage(exception: OptixRuntimeError) -> str:
    if exception.stage != "optix_runtime_initialization":
        return exception.stage
    message = str(exception)
    if "NVRTC" in message or "nvrtc" in message:
        return "nvrtc_compile"
    return exception.stage


def _decode_result(result: Any, *, name: str) -> tuple[int, int, float]:
    values = np.asarray(result)
    if values.shape != (3,):
        raise OptixSmokeError(
            "result_sanity",
            f"{name} result has unexpected shape {values.shape!r}",
        )
    distance = np.asarray(values[2], dtype=np.uint32).view(np.float32).item()
    decoded = (int(values[0]), int(values[1]), float(distance))
    if not np.isfinite(decoded[2]):
        raise OptixSmokeError(
            "result_sanity",
            f"{name} ray distance is not finite: {decoded}",
        )
    return decoded


def run() -> OptixSmokeResult:
    """Initialize the real runtime, build one GAS, and launch hit/miss rays."""
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - environment dependent
        raise OptixSmokeError("cupy_import", f"{type(exc).__name__}: {exc}") from exc
    try:
        import optix  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise OptixSmokeError("optix_import", f"{type(exc).__name__}: {exc}") from exc
    try:
        from cuda.bindings import nvrtc  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise OptixSmokeError(
            "cuda_nvrtc_import", f"{type(exc).__name__}: {exc}"
        ) from exc

    setup_started = time.perf_counter()
    try:
        runtime = OptixRuntime.create(
            device_source=_DEVICE_SOURCE,
            source_name="optix_smoke.cu",
            raygen_entry="__raygen__raygen_program",
            miss_entry="__miss__miss_program",
            hitgroup_entry="__closesthit__closesthit_program",
            params_dtype=_PARAMS_DTYPE,
            num_payload_values=1,
            num_attribute_values=2,
        )
    except OptixRuntimeError as exc:  # pragma: no cover - environment dependent
        raise OptixSmokeError(_runtime_stage(exc), f"{type(exc).__name__}: {exc}") from exc

    try:
        handle, _owners = runtime.build_gas(
            np.asarray(
                [(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.0, 0.5, 0.0)],
                dtype=np.float32,
            ),
            np.asarray([(0, 1, 2)], dtype=np.uint32),
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise OptixSmokeError(
            "gas_construction", f"{type(exc).__name__}: {exc}"
        ) from exc
    setup_time_seconds = time.perf_counter() - setup_started

    try:
        result_device = cp.zeros(3, dtype=cp.uint32)
        params_host = np.zeros(1, dtype=_PARAMS_DTYPE)
        params_host["handle"] = handle
        params_host["result"] = int(result_device.data.ptr)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise OptixSmokeError("optix_launch", f"{type(exc).__name__}: {exc}") from exc

    def launch(origin: tuple[float, float, float]) -> tuple[int, int, float]:
        params_host["origin"] = origin
        params_host["direction"] = (0.0, 0.0, -1.0)
        result_device.fill(0)
        runtime.launch(params_host, width=1, height=1, depth=1)
        return _decode_result(cp.asnumpy(result_device), name="launch")

    trace_started = time.perf_counter()
    try:
        hit = launch((0.0, 0.0, 1.0))
        miss = launch((2.0, 2.0, 1.0))
    except OptixSmokeError:
        raise
    except Exception as exc:  # pragma: no cover - environment dependent
        raise OptixSmokeError("optix_launch", f"{type(exc).__name__}: {exc}") from exc
    trace_time_seconds = time.perf_counter() - trace_started

    if hit[0] != 1 or hit[1] != 0 or hit[2] <= 0.0:
        raise OptixSmokeError("result_sanity", f"deterministic hit validation failed: {hit}")
    if miss[0] != 0 or miss[2] != 0.0:
        raise OptixSmokeError("result_sanity", f"deterministic miss validation failed: {miss}")

    return OptixSmokeResult(
        metadata=dict(runtime.metadata),
        hit=hit,
        miss=miss,
        setup_time_seconds=setup_time_seconds,
        trace_time_seconds=trace_time_seconds,
        ray_count=2,
        terminal_event_counts={"hit": 1, "miss": 1},
        result_counts={"hit": 1, "miss": 1},
    )


def _version(value: object) -> str:
    if isinstance(value, (tuple, list)):
        return ".".join(str(item) for item in value)
    return str(value)


def main() -> int:
    try:
        result = run()
    except OptixSmokeError as exc:
        print(f"FAIL: {exc.stage}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(
            "FAIL: optix_runtime_initialization: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    metadata = result.metadata
    terminal = ",".join(
        f"{name}={count}" for name, count in result.terminal_event_counts.items()
    )
    results = ",".join(
        f"{name}={count}" for name, count in result.result_counts.items()
    )
    print(
        "PASS: optix_smoke "
        f"GPU/device={metadata['cuda_device']} "
        f"OptiX={metadata['optix_version']} "
        f"CUDA_runtime={metadata['cuda_runtime_version']} "
        f"NVRTC={_version(metadata['nvrtc_version'])} "
        f"OptiX_include={metadata['optix_include']} "
        f"CUDA_include={metadata['cuda_include']} "
        f"setup={result.setup_time_seconds:.3f}s "
        f"trace={result.trace_time_seconds:.3f}s "
        f"rays={result.ray_count} "
        f"terminal={terminal} results={results}"
    )
    return 0


__all__ = ["OptixSmokeError", "OptixSmokeResult", "main", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
