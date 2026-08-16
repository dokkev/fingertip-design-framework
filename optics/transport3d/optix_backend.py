"""Small OptiX traversal backend for the transport3d wavefront."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from optics.optix.paths import discover_paths
from optics.transport3d.geometry import TriangleSurface, Transport3DGeometryError


class Transport3DDependencyError(RuntimeError):
    """Raised when the optional CUDA/OptiX runtime is unavailable."""


class Transport3DTraceError(RuntimeError):
    """Raised when OptiX cannot return a valid traversal result."""


_DEVICE_SOURCE = (
    Path(__file__).with_name("kernels").joinpath("transport.cu").read_text()
)


def _require_cuda_result(result: tuple[Any, ...], operation: str) -> tuple[Any, ...]:
    if int(result[0]) != 0:
        raise Transport3DDependencyError(f"{operation} failed with CUDA status {result[0]}")
    return result[1:]


def _nvrtc_log(nvrtc: Any, program: Any) -> str:
    (size,) = _require_cuda_result(nvrtc.nvrtcGetProgramLogSize(program), "nvrtcGetProgramLogSize")
    if not size:
        return ""
    buffer = bytearray(size)
    _require_cuda_result(nvrtc.nvrtcGetProgramLog(program, buffer), "nvrtcGetProgramLog")
    return bytes(buffer).rstrip(b"\0").decode(errors="replace")


def _compile_source(paths: Any, compute_capability: str) -> str:
    from cuda.bindings import nvrtc

    (program,) = _require_cuda_result(
        nvrtc.nvrtcCreateProgram(
            _DEVICE_SOURCE.encode(),
            b"transport3d.cu",
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
    compiled = nvrtc.nvrtcCompileProgram(program, len(options), options)
    if int(compiled[0]) != 0:
        raise Transport3DDependencyError(
            f"NVRTC transport compilation failed: {compiled[0]}; "
            f"log: {_nvrtc_log(nvrtc, program)}"
        )
    (ptx_size,) = _require_cuda_result(nvrtc.nvrtcGetPTXSize(program), "nvrtcGetPTXSize")
    ptx = bytearray(ptx_size)
    _require_cuda_result(nvrtc.nvrtcGetPTX(program, ptx), "nvrtcGetPTX")
    _require_cuda_result(nvrtc.nvrtcDestroyProgram(program), "nvrtcDestroyProgram")
    return bytes(ptx).rstrip(b"\0").decode()


@dataclass
class _Runtime:
    optix: Any
    cp: Any
    context: Any
    pipeline: Any
    sbt: Any
    sbt_records: list[Any]
    metadata: dict[str, Any]
    params_dtype: np.dtype

    @classmethod
    def create(cls) -> _Runtime:
        started = time.perf_counter()
        try:
            import cupy as cp
            import optix
            from cuda.bindings import nvrtc
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            raise Transport3DDependencyError(
                "transport3d requires CuPy, PyOptiX, and cuda-python"
            ) from exc
        try:
            paths = discover_paths()
            device = cp.cuda.Device()
            device.use()
            properties = cp.cuda.runtime.getDeviceProperties(device.id)
            compute_capability = f"{properties['major']}{properties['minor']}"
            ptx = _compile_source(paths, compute_capability)
            context = optix.deviceContextCreate(0, optix.DeviceContextOptions())
            module_options = optix.ModuleCompileOptions()
            pipeline_options = optix.PipelineCompileOptions()
            pipeline_options.traversableGraphFlags = optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS
            pipeline_options.numPayloadValues = 0
            pipeline_options.numAttributeValues = 2
            pipeline_options.pipelineLaunchParamsVariableName = "params"
            module, module_log = context.moduleCreate(module_options, pipeline_options, ptx)
            raygen = optix.ProgramGroupDesc()
            raygen.raygenModule = module
            raygen.raygenEntryFunctionName = "__raygen__transport"
            miss = optix.ProgramGroupDesc()
            miss.missModule = module
            miss.missEntryFunctionName = "__miss__transport"
            hitgroup = optix.ProgramGroupDesc()
            hitgroup.hitgroupModuleCH = module
            hitgroup.hitgroupEntryFunctionNameCH = "__closesthit__transport"
            groups, group_log = context.programGroupCreate(
                [raygen, miss, hitgroup], optix.ProgramGroupOptions()
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
            params_dtype = np.dtype(
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
            metadata = {
                "optix_version": ".".join(map(str, optix.version())),
                "cuda_device": properties["name"].decode(),
                "compute_capability": compute_capability,
                "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
                "nvrtc_version": _require_cuda_result(nvrtc.nvrtcVersion(), "nvrtcVersion")[0:2],
                "optix_include": str(paths.optix_include),
                "cuda_include": str(paths.cuda_include),
                "module_setup_seconds": time.perf_counter() - started,
                "module_log": module_log,
                "program_group_log": group_log,
            }
            return cls(optix, cp, context, pipeline, sbt, records, metadata, params_dtype)
        except Transport3DDependencyError:
            raise
        except Exception as exc:  # pragma: no cover - GPU API failure is environment-specific
            raise Transport3DDependencyError(f"OptiX transport setup failed: {exc}") from exc

    def build_gas(self, surface: TriangleSurface) -> tuple[int, list[Any]]:
        vertices = self.cp.asarray(surface.vertices, dtype=self.cp.float32)
        faces = self.cp.asarray(surface.faces, dtype=self.cp.uint32)
        triangle = self.optix.BuildInputTriangleArray()
        triangle.vertexBuffers = [int(vertices.data.ptr)]
        triangle.numVertices = len(surface.vertices)
        triangle.vertexFormat = self.optix.VERTEX_FORMAT_FLOAT3
        triangle.vertexStrideInBytes = int(vertices.strides[0])
        triangle.indexBuffer = int(faces.data.ptr)
        triangle.numIndexTriplets = len(surface.faces)
        triangle.indexFormat = self.optix.INDICES_FORMAT_UNSIGNED_INT3
        triangle.indexStrideInBytes = int(faces.strides[0])
        triangle.numSbtRecords = 1
        triangle.flags = [self.optix.GEOMETRY_FLAG_NONE]
        build_options = self.optix.AccelBuildOptions()
        build_options.buildFlags = self.optix.BUILD_FLAG_NONE
        sizes = self.context.accelComputeMemoryUsage([build_options], [triangle])
        temporary = self.cp.empty(int(sizes.tempSizeInBytes), dtype=self.cp.uint8)
        output = self.cp.empty(int(sizes.outputSizeInBytes), dtype=self.cp.uint8)
        handle = self.context.accelBuild(
            int(self.cp.cuda.Stream.null.ptr),
            [build_options],
            [triangle],
            int(temporary.data.ptr),
            int(temporary.nbytes),
            int(output.data.ptr),
            int(output.nbytes),
            [],
        )
        self.cp.cuda.runtime.deviceSynchronize()
        return int(handle), [vertices, faces, temporary, output]

    def trace(
        self,
        handle: int,
        origins: Any,
        directions: Any,
        *,
        tmin: float,
    ) -> tuple[Any, Any, Any, Any]:
        count = int(origins.shape[0])
        if count == 0:
            return (
                self.cp.empty((0,), dtype=self.cp.float32),
                self.cp.empty((0,), dtype=self.cp.uint32),
                self.cp.empty((0, 2), dtype=self.cp.float32),
                self.cp.empty((0,), dtype=self.cp.uint32),
            )
        origins = self.cp.ascontiguousarray(origins, dtype=self.cp.float32)
        directions = self.cp.ascontiguousarray(directions, dtype=self.cp.float32)
        distances = self.cp.empty(count, dtype=self.cp.float32)
        primitives = self.cp.empty(count, dtype=self.cp.uint32)
        barycentrics = self.cp.empty((count, 2), dtype=self.cp.float32)
        hits = self.cp.empty(count, dtype=self.cp.uint32)
        params_host = np.zeros(1, dtype=self.params_dtype)
        params_host["handle"] = handle
        params_host["origins"] = int(origins.data.ptr)
        params_host["directions"] = int(directions.data.ptr)
        params_host["distances"] = int(distances.data.ptr)
        params_host["primitives"] = int(primitives.data.ptr)
        params_host["barycentrics"] = int(barycentrics.data.ptr)
        params_host["hits"] = int(hits.data.ptr)
        params_host["count"] = count
        params_host["tmin"] = tmin
        params_device = self.cp.asarray(params_host)
        self.optix.launch(
            self.pipeline,
            int(self.cp.cuda.Stream.null.ptr),
            int(params_device.data.ptr),
            int(params_device.nbytes),
            self.sbt,
            count,
            1,
            1,
        )
        self.cp.cuda.runtime.deviceSynchronize()
        return distances, primitives, barycentrics, hits


class OptixScene:
    """One rebuilt GAS collection for one deformed FEM state."""

    def __init__(self, runtime: _Runtime, silicone: TriangleSurface, rigid: TriangleSurface, envelope: TriangleSurface) -> None:
        self.runtime = runtime
        started = time.perf_counter()
        try:
            self.handles = {}
            self._owners: list[Any] = []
            for name, surface in (("silicone", silicone), ("rigid", rigid), ("envelope", envelope)):
                handle, owners = runtime.build_gas(surface)
                self.handles[name] = handle
                self._owners.extend(owners)
        except Exception as exc:
            raise Transport3DTraceError(f"OptiX GAS construction failed: {exc}") from exc
        self.gas_build_seconds = time.perf_counter() - started

    def trace(self, name: str, origins: Any, directions: Any, *, tmin: float) -> tuple[Any, Any, Any, Any]:
        try:
            return self.runtime.trace(self.handles[name], origins, directions, tmin=tmin)
        except Exception as exc:
            raise Transport3DTraceError(f"OptiX {name} traversal failed: {exc}") from exc


__all__ = [
    "OptixScene",
    "Transport3DDependencyError",
    "Transport3DTraceError",
]
