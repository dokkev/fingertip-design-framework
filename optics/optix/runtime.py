"""Shared low-level optional CUDA/OptiX runtime setup."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from optics.optix._paths import _discover_include_paths


class OptixRuntimeError(RuntimeError):
    """Raised when optional CUDA/OptiX setup or execution fails."""

    def __init__(
        self,
        message: str,
        *,
        stage: str = "optix_runtime_initialization",
    ) -> None:
        super().__init__(message)
        self.stage = stage


def _require_cuda_result(result: tuple[Any, ...], operation: str) -> tuple[Any, ...]:
    if int(result[0]) != 0:
        raise OptixRuntimeError(f"{operation} failed with CUDA status {result[0]}")
    return result[1:]


def _nvrtc_log(nvrtc: Any, program: Any) -> str:
    (size,) = _require_cuda_result(
        nvrtc.nvrtcGetProgramLogSize(program), "nvrtcGetProgramLogSize"
    )
    if not size:
        return ""
    buffer = bytearray(size)
    _require_cuda_result(
        nvrtc.nvrtcGetProgramLog(program, buffer), "nvrtcGetProgramLog"
    )
    return bytes(buffer).rstrip(b"\0").decode(errors="replace")


@dataclass
class OptixRuntime:
    """One configured OptiX context/pipeline and its CUDA owners."""

    optix: Any
    cp: Any
    context: Any
    pipeline: Any
    sbt: Any
    sbt_records: list[Any]
    metadata: dict[str, Any]
    params_dtype: np.dtype

    @classmethod
    def create(
        cls,
        *,
        device_source: str,
        source_name: str,
        raygen_entry: str,
        miss_entry: str,
        hitgroup_entry: str,
        params_dtype: np.dtype,
        num_payload_values: int,
        num_attribute_values: int,
    ) -> "OptixRuntime":
        started = time.perf_counter()
        stage = "dependency_import"
        try:
            import cupy as cp
            import optix
            from cuda.bindings import nvrtc
        except Exception as exc:  # pragma: no cover - optional environment
            raise OptixRuntimeError(
                "OptiX runtime requires CuPy, PyOptiX, and cuda-python: "
                f"{type(exc).__name__}: {exc}",
                stage=stage,
            ) from exc
        try:
            stage = "optix_header_resolution"
            paths = _discover_include_paths()
            stage = "cuda_device"
            device = cp.cuda.Device()
            device.use()
            properties = cp.cuda.runtime.getDeviceProperties(device.id)
            compute_capability = f"{properties['major']}{properties['minor']}"
            stage = "nvrtc_compile"
            (program,) = _require_cuda_result(
                nvrtc.nvrtcCreateProgram(
                    device_source.encode(),
                    source_name.encode(),
                    0,
                    None,
                    None,
                ),
                "nvrtcCreateProgram",
            )
            options = [
                b"--std=c++17",
                f"--gpu-architecture=compute_{compute_capability}".encode(),
                f"-I{paths.optix}".encode(),
                f"-I{paths.cuda}".encode(),
            ]
            compiled = nvrtc.nvrtcCompileProgram(program, len(options), options)
            if int(compiled[0]) != 0:
                raise OptixRuntimeError(
                    f"NVRTC compilation failed: {compiled[0]}; "
                    f"log: {_nvrtc_log(nvrtc, program)}",
                    stage=stage,
                )
            (ptx_size,) = _require_cuda_result(
                nvrtc.nvrtcGetPTXSize(program), "nvrtcGetPTXSize"
            )
            ptx_buffer = bytearray(ptx_size)
            _require_cuda_result(nvrtc.nvrtcGetPTX(program, ptx_buffer), "nvrtcGetPTX")
            _require_cuda_result(
                nvrtc.nvrtcDestroyProgram(program), "nvrtcDestroyProgram"
            )

            stage = "optix_context"
            context = optix.deviceContextCreate(0, optix.DeviceContextOptions())
            stage = "optix_pipeline"
            module_options = optix.ModuleCompileOptions()
            pipeline_options = optix.PipelineCompileOptions()
            pipeline_options.traversableGraphFlags = (
                optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS
            )
            pipeline_options.numPayloadValues = num_payload_values
            pipeline_options.numAttributeValues = num_attribute_values
            pipeline_options.pipelineLaunchParamsVariableName = "params"
            module, module_log = context.moduleCreate(
                module_options,
                pipeline_options,
                bytes(ptx_buffer).rstrip(b"\0").decode(),
            )
            raygen = optix.ProgramGroupDesc()
            raygen.raygenModule = module
            raygen.raygenEntryFunctionName = raygen_entry
            miss = optix.ProgramGroupDesc()
            miss.missModule = module
            miss.missEntryFunctionName = miss_entry
            hitgroup = optix.ProgramGroupDesc()
            hitgroup.hitgroupModuleCH = module
            hitgroup.hitgroupEntryFunctionNameCH = hitgroup_entry
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
            stage = "sbt_setup"
            sbt, sbt_records = cls._make_sbt(optix, cp, groups)
            metadata = {
                "optix_version": ".".join(map(str, optix.version())),
                "cuda_device": properties["name"].decode(),
                "compute_capability": compute_capability,
                "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
                "nvrtc_version": _require_cuda_result(
                    nvrtc.nvrtcVersion(), "nvrtcVersion"
                )[0:2],
                "optix_include": str(paths.optix),
                "cuda_include": str(paths.cuda),
                "optix_include_resolution_source": paths.optix_source,
                "cuda_include_resolution_source": paths.cuda_source,
                "module_setup_seconds": time.perf_counter() - started,
                "module_log": module_log,
                "program_group_log": group_log,
            }
            return cls(
                optix,
                cp,
                context,
                pipeline,
                sbt,
                sbt_records,
                metadata,
                np.dtype(params_dtype),
            )
        except OptixRuntimeError as exc:
            if exc.stage == "optix_runtime_initialization":
                raise OptixRuntimeError(str(exc), stage=stage) from exc
            raise
        except Exception as exc:  # pragma: no cover - optional API failure
            if stage == "optix_header_resolution":
                message = str(exc)
                if "CUDA" in message or "cuda" in message:
                    stage = "cuda_header_resolution"
            raise OptixRuntimeError(
                f"OptiX runtime setup failed: {exc}", stage=stage
            ) from exc

    @staticmethod
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

    @staticmethod
    def _backend_call(operation: str, callback: Callable[[], Any]) -> Any:
        """Translate direct CUDA/OptiX operation failures to runtime errors."""
        try:
            return callback()
        except OptixRuntimeError:
            raise
        except (MemoryError, RuntimeError) as exc:
            raise OptixRuntimeError(
                f"{operation} failed: {exc}",
                stage="optix_runtime_execution",
            ) from exc

    def make_sbt(self, groups: list[Any]) -> tuple[Any, list[Any]]:
        """Pack a shader binding table for an alternate caller-owned program set."""
        return self._make_sbt(self.optix, self.cp, groups)

    def build_gas(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> tuple[int, list[Any]]:
        """Build one triangle GAS and return handles plus device-memory owners."""
        vertices_device = self._backend_call(
            "CUDA vertex allocation",
            lambda: self.cp.asarray(vertices, dtype=self.cp.float32),
        )
        faces_device = self._backend_call(
            "CUDA index allocation",
            lambda: self.cp.asarray(faces, dtype=self.cp.uint32),
        )
        triangle = self._backend_call(
            "OptiX triangle description",
            self.optix.BuildInputTriangleArray,
        )
        triangle.vertexBuffers = [int(vertices_device.data.ptr)]
        triangle.numVertices = len(vertices_device)
        triangle.vertexFormat = self.optix.VERTEX_FORMAT_FLOAT3
        triangle.vertexStrideInBytes = int(vertices_device.strides[0])
        triangle.indexBuffer = int(faces_device.data.ptr)
        triangle.numIndexTriplets = len(faces_device)
        triangle.indexFormat = self.optix.INDICES_FORMAT_UNSIGNED_INT3
        triangle.indexStrideInBytes = int(faces_device.strides[0])
        triangle.numSbtRecords = 1
        triangle.flags = [self.optix.GEOMETRY_FLAG_NONE]
        build_options = self._backend_call(
            "OptiX acceleration-build options",
            self.optix.AccelBuildOptions,
        )
        build_options.buildFlags = self.optix.BUILD_FLAG_NONE
        sizes = self._backend_call(
            "OptiX GAS memory query",
            lambda: self.context.accelComputeMemoryUsage(
                [build_options], [triangle]
            ),
        )
        temporary = self._backend_call(
            "CUDA temporary GAS allocation",
            lambda: self.cp.empty(
                int(sizes.tempSizeInBytes), dtype=self.cp.uint8
            ),
        )
        output = self._backend_call(
            "CUDA output GAS allocation",
            lambda: self.cp.empty(
                int(sizes.outputSizeInBytes), dtype=self.cp.uint8
            ),
        )
        handle = self._backend_call(
            "OptiX GAS build",
            lambda: self.context.accelBuild(
                int(self.cp.cuda.Stream.null.ptr),
                [build_options],
                [triangle],
                int(temporary.data.ptr),
                int(temporary.nbytes),
                int(output.data.ptr),
                int(output.nbytes),
                [],
            ),
        )
        self._backend_call(
            "CUDA GAS synchronization",
            lambda: self.cp.cuda.runtime.deviceSynchronize(),
        )
        return int(handle), [vertices_device, faces_device, temporary, output]

    def launch(
        self,
        params_host: np.ndarray,
        *,
        width: int,
        height: int,
        depth: int,
        sbt: Any | None = None,
    ) -> None:
        """Launch the configured pipeline with a caller-owned parameter record."""
        params_device = self._backend_call(
            "CUDA launch-parameter allocation",
            lambda: self.cp.asarray(params_host),
        )
        self._backend_call(
            "OptiX launch",
            lambda: self.optix.launch(
                self.pipeline,
                int(self.cp.cuda.Stream.null.ptr),
                int(params_device.data.ptr),
                int(params_device.nbytes),
                self.sbt if sbt is None else sbt,
                width,
                height,
                depth,
            ),
        )
        self._backend_call(
            "CUDA launch synchronization",
            lambda: self.cp.cuda.runtime.deviceSynchronize(),
        )

    def trace(
        self,
        handle: int,
        origins: Any,
        directions: Any,
        *,
        tmin: float,
    ) -> tuple[Any, Any, Any, Any]:
        """Trace the standard vectorized transport parameter record."""
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
        self.launch(params_host, width=count, height=1, depth=1)
        return distances, primitives, barycentrics, hits


__all__ = ["OptixRuntime", "OptixRuntimeError"]
