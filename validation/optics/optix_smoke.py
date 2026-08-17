"""Run the minimal NVIDIA OptiX runtime smoke test."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any

import numpy as np

from optics.optix.runtime import OptixRuntime


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


@dataclass(frozen=True)
class SmokeResult:
    """Validated output of one hit and one miss launch."""

    hit: tuple[int, int, float]
    miss: tuple[int, int, float]


def run_smoke() -> tuple[SmokeResult, dict[str, Any]]:
    """Create shared OptiX resources, launch hit/miss rays, and verify results."""
    import cupy as cp

    dtype = np.dtype(
        [
            ("handle", np.uint64),
            ("result", np.uint64),
            ("origin", np.float32, (3,)),
            ("direction", np.float32, (3,)),
        ],
        align=True,
    )
    runtime = OptixRuntime.create(
        device_source=_DEVICE_SOURCE,
        source_name="optix_smoke.cu",
        raygen_entry="__raygen__raygen_program",
        miss_entry="__miss__miss_program",
        hitgroup_entry="__closesthit__closesthit_program",
        params_dtype=dtype,
        num_payload_values=1,
        num_attribute_values=2,
    )
    handle, _owners = runtime.build_gas(
        np.asarray(
            [(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.0, 0.5, 0.0)],
            dtype=np.float32,
        ),
        np.asarray([(0, 1, 2)], dtype=np.uint32),
    )
    result_device = cp.zeros(3, dtype=cp.uint32)
    params_host = np.zeros(1, dtype=dtype)
    params_host["handle"] = handle
    params_host["result"] = int(result_device.data.ptr)

    def launch(origin: tuple[float, float, float]) -> tuple[int, int, float]:
        params_host["origin"] = origin
        params_host["direction"] = (0.0, 0.0, -1.0)
        result_device.fill(0)
        runtime.launch(params_host, width=1, height=1, depth=1)
        result = cp.asnumpy(result_device)
        return int(result[0]), int(result[1]), float(
            np.asarray(result[2], dtype=np.uint32).view(np.float32)
        )

    hit = launch((0.0, 0.0, 1.0))
    miss = launch((2.0, 2.0, 1.0))
    if hit[0] != 1 or hit[1] != 0 or not np.isfinite(hit[2]) or hit[2] <= 0.0:
        raise RuntimeError(f"deterministic hit validation failed: {hit}")
    if miss[0] != 0:
        raise RuntimeError(f"deterministic miss validation failed: {miss}")
    return SmokeResult(hit=hit, miss=miss), dict(runtime.metadata)


def main() -> int:
    try:
        result, metadata = run_smoke()
    except Exception as exc:
        print(f"FAIL OptiX smoke: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        "PASS OptiX smoke: "
        f"OptiX {metadata['optix_version']}; "
        f"CUDA device {metadata['cuda_device']} "
        f"(compute capability {metadata['compute_capability']}); "
        f"OptiX include {metadata['optix_include']}; "
        f"CUDA include {metadata['cuda_include']}; "
        f"hit={result.hit}; miss={result.miss}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
