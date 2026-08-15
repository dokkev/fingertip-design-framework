"""Run the minimal NVIDIA OptiX runtime smoke test."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any

import numpy as np

from optics.optix.paths import OptixCudaPaths, discover_paths


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


def _require_cuda_result(result: tuple[Any, ...], operation: str) -> tuple[Any, ...]:
    status = result[0]
    if int(status) != 0:
        raise RuntimeError(f"{operation} failed with {status}")
    return result[1:]


def _nvrtc_log(nvrtc: Any, program: Any) -> str:
    (size,) = _require_cuda_result(
        nvrtc.nvrtcGetProgramLogSize(program),
        "nvrtcGetProgramLogSize",
    )
    if not size:
        return ""
    buffer = bytearray(size)
    _require_cuda_result(
        nvrtc.nvrtcGetProgramLog(program, buffer),
        "nvrtcGetProgramLog",
    )
    return bytes(buffer).rstrip(b"\0").decode(errors="replace")


def _compile_device_source(paths: OptixCudaPaths, compute_capability: str) -> str:
    from cuda.bindings import nvrtc

    (program,) = _require_cuda_result(
        nvrtc.nvrtcCreateProgram(
            _DEVICE_SOURCE.encode(),
            b"optix_smoke.cu",
            0,
            None,
            None,
        ),
        "nvrtcCreateProgram",
    )
    options = [
        b"--std=c++17",
        f"--gpu-architecture=compute_{compute_capability}".encode(),
        f"-I{paths.optix_include}".encode(),
        f"-I{paths.cuda_include}".encode(),
    ]
    compile_result = nvrtc.nvrtcCompileProgram(program, len(options), options)
    if int(compile_result[0]) != 0:
        raise RuntimeError(
            "nvrtcCompileProgram failed: "
            f"{compile_result[0]}; log: {_nvrtc_log(nvrtc, program)}"
        )
    (ptx_size,) = _require_cuda_result(
        nvrtc.nvrtcGetPTXSize(program),
        "nvrtcGetPTXSize",
    )
    ptx_buffer = bytearray(ptx_size)
    _require_cuda_result(nvrtc.nvrtcGetPTX(program, ptx_buffer), "nvrtcGetPTX")
    _require_cuda_result(
        nvrtc.nvrtcDestroyProgram(program),
        "nvrtcDestroyProgram",
    )
    return bytes(ptx_buffer).rstrip(b"\0").decode()


def _build_gas(context: Any, cp: Any) -> tuple[int, Any, Any, Any]:
    import optix

    vertices = cp.asarray(
        np.asarray(
            [
                (-0.5, -0.5, 0.0),
                (0.5, -0.5, 0.0),
                (0.0, 0.5, 0.0),
            ],
            dtype=np.float32,
        )
    )
    indices = cp.asarray(np.asarray([(0, 1, 2)], dtype=np.uint32))

    triangle = optix.BuildInputTriangleArray()
    triangle.vertexBuffers = [int(vertices.data.ptr)]
    triangle.numVertices = 3
    triangle.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
    triangle.vertexStrideInBytes = int(vertices.strides[0])
    triangle.indexBuffer = int(indices.data.ptr)
    triangle.numIndexTriplets = 1
    triangle.indexFormat = optix.INDICES_FORMAT_UNSIGNED_INT3
    triangle.indexStrideInBytes = int(indices.strides[0])
    triangle.numSbtRecords = 1
    triangle.flags = [optix.GEOMETRY_FLAG_NONE]

    build_options = optix.AccelBuildOptions()
    build_options.buildFlags = optix.BUILD_FLAG_NONE
    sizes = context.accelComputeMemoryUsage([build_options], [triangle])
    temporary = cp.empty(int(sizes.tempSizeInBytes), dtype=cp.uint8)
    output = cp.empty(int(sizes.outputSizeInBytes), dtype=cp.uint8)
    stream = int(cp.cuda.Stream.null.ptr)
    handle = context.accelBuild(
        stream,
        [build_options],
        [triangle],
        int(temporary.data.ptr),
        int(temporary.nbytes),
        int(output.data.ptr),
        int(output.nbytes),
        [],
    )
    cp.cuda.runtime.deviceSynchronize()
    return int(handle), vertices, indices, output


def _make_sbt(optix: Any, cp: Any, groups: list[Any]) -> tuple[Any, list[Any]]:
    records = []
    for group in groups:
        record = np.zeros(optix.SBT_RECORD_HEADER_SIZE, dtype=np.uint8)
        optix.sbtRecordPackHeader(group, record)
        records.append(cp.asarray(record))
    sbt = optix.ShaderBindingTable()
    sbt.raygenRecord = int(records[0].data.ptr)
    sbt.missRecordBase = int(records[1].data.ptr)
    sbt.missRecordStrideInBytes = optix.SBT_RECORD_HEADER_SIZE
    sbt.missRecordCount = 1
    sbt.hitgroupRecordBase = int(records[2].data.ptr)
    sbt.hitgroupRecordStrideInBytes = optix.SBT_RECORD_HEADER_SIZE
    sbt.hitgroupRecordCount = 1
    return sbt, records


def _launch(
    optix: Any,
    cp: Any,
    pipeline: Any,
    sbt: Any,
    params_device: Any,
    params_host: np.ndarray,
    result_device: Any,
    origin: tuple[float, float, float],
) -> tuple[int, int, float]:
    params_host["origin"] = origin
    params_host["direction"] = (0.0, 0.0, -1.0)
    params_device.set(params_host)
    result_device.fill(0)
    stream = int(cp.cuda.Stream.null.ptr)
    optix.launch(
        pipeline,
        stream,
        int(params_device.data.ptr),
        int(params_device.nbytes),
        sbt,
        1,
        1,
        1,
    )
    cp.cuda.runtime.deviceSynchronize()
    result = cp.asnumpy(result_device)
    return int(result[0]), int(result[1]), float(
        np.asarray(result[2], dtype=np.uint32).view(np.float32)
    )


def run_smoke() -> tuple[SmokeResult, dict[str, Any]]:
    """Create OptiX resources, launch hit/miss rays, and verify results."""
    import cupy as cp
    import optix
    from cuda.bindings import nvrtc

    paths = discover_paths()
    device = cp.cuda.Device()
    device.use()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    compute_capability = f"{properties['major']}{properties['minor']}"
    ptx = _compile_device_source(paths, compute_capability)

    context = optix.deviceContextCreate(0, optix.DeviceContextOptions())
    module_options = optix.ModuleCompileOptions()
    pipeline_options = optix.PipelineCompileOptions()
    pipeline_options.traversableGraphFlags = (
        optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS
    )
    pipeline_options.numPayloadValues = 1
    pipeline_options.numAttributeValues = 2
    pipeline_options.pipelineLaunchParamsVariableName = "params"
    module, module_log = context.moduleCreate(
        module_options,
        pipeline_options,
        ptx,
    )
    raygen = optix.ProgramGroupDesc()
    raygen.raygenModule = module
    raygen.raygenEntryFunctionName = "__raygen__raygen_program"
    miss = optix.ProgramGroupDesc()
    miss.missModule = module
    miss.missEntryFunctionName = "__miss__miss_program"
    hitgroup = optix.ProgramGroupDesc()
    hitgroup.hitgroupModuleCH = module
    hitgroup.hitgroupEntryFunctionNameCH = "__closesthit__closesthit_program"
    groups, group_log = context.programGroupCreate(
        [raygen, miss, hitgroup],
        optix.ProgramGroupOptions(),
    )
    pipeline = context.pipelineCreate(
        pipeline_options,
        optix.PipelineLinkOptions(maxTraceDepth=1),
        groups,
        "",
    )
    stack_sizes = [group.getStackSize(pipeline) for group in groups]
    pipeline.setStackSize(
        max(size.cssRG for size in stack_sizes),
        max(size.cssMS for size in stack_sizes),
        max(size.cssCH for size in stack_sizes),
        1,
    )
    handle, vertices, indices, gas_output = _build_gas(context, cp)
    result_device = cp.zeros(3, dtype=cp.uint32)
    dtype = np.dtype(
        [
            ("handle", np.uint64),
            ("result", np.uint64),
            ("origin", np.float32, (3,)),
            ("direction", np.float32, (3,)),
        ],
        align=True,
    )
    params_host = np.zeros(1, dtype=dtype)
    params_host["handle"] = handle
    params_host["result"] = int(result_device.data.ptr)
    params_device = cp.asarray(params_host)
    sbt, _sbt_records = _make_sbt(optix, cp, groups)
    hit = _launch(
        optix,
        cp,
        pipeline,
        sbt,
        params_device,
        params_host,
        result_device,
        (0.0, 0.0, 1.0),
    )
    miss_result = _launch(
        optix,
        cp,
        pipeline,
        sbt,
        params_device,
        params_host,
        result_device,
        (2.0, 2.0, 1.0),
    )
    if hit[0] != 1 or hit[1] != 0 or not np.isfinite(hit[2]) or hit[2] <= 0.0:
        raise RuntimeError(f"deterministic hit validation failed: {hit}")
    if miss_result[0] != 0:
        raise RuntimeError(f"deterministic miss validation failed: {miss_result}")
    metadata = {
        "optix_version": ".".join(map(str, optix.version())),
        "cuda_device": properties["name"].decode(),
        "compute_capability": compute_capability,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "nvrtc_version": _require_cuda_result(
            nvrtc.nvrtcVersion(),
            "nvrtcVersion",
        )[0:2],
        "optix_include": str(paths.optix_include),
        "cuda_include": str(paths.cuda_include),
        "module_log": module_log,
        "program_group_log": group_log,
    }
    return SmokeResult(hit=hit, miss=miss_result), metadata


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
