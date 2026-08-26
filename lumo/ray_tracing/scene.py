"""OptiX fingertip scene for silicone and carrier."""

from __future__ import annotations

import ctypes
import os
import sys
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lumo.mesh import FingertipMesh


_PAYLOAD_WORD_COUNT = 6
_RESULT_WORD_COUNT = 15
_SILICONE_INSTANCE_ID = 1
_CARRIER_INSTANCE_ID = 2
_SILICONE_MASK = 0x01
_CARRIER_MASK = 0x02
_ALL_MASK = _SILICONE_MASK | _CARRIER_MASK
_RESULT_DTYPE = np.dtype(
    [
        ("hit", np.bool_),
        ("t", np.float32),
        ("instance_id", np.int32),
        ("primitive_id", np.int32),
        ("barycentrics", np.float32, (2,)),
        ("normal_W", np.float32, (3,)),
        ("spawn_front_W", np.float32, (3,)),
        ("spawn_back_W", np.float32, (3,)),
    ]
)


def _vertices(value: np.ndarray, *, name: str) -> np.ndarray:
    vertices = np.ascontiguousarray(value, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not vertices.size:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"{name} must be finite")
    return vertices


def _triangles(
    value: np.ndarray,
    *,
    vertex_count: int,
    name: str,
) -> np.ndarray:
    triangle_indices = np.asarray(value, dtype=np.int64)
    if triangle_indices.size % 3 != 0:
        raise ValueError(f"{name} must contain index triplets")
    triangle_indices = triangle_indices.reshape(-1, 3)
    if not triangle_indices.size:
        raise ValueError(f"{name} must not be empty")
    if int(triangle_indices.min()) < 0:
        raise ValueError(f"{name} contains a negative vertex index")
    if int(triangle_indices.max()) >= vertex_count:
        raise ValueError(f"{name} contains an out-of-range vertex index")
    return np.ascontiguousarray(triangle_indices, dtype=np.uint32)


def _visibility_mask(value: int, *, name: str) -> int:
    value = int(value)
    if not 1 <= value <= 0xFF:
        raise ValueError(f"{name} must be in [1, 0xFF]")
    return value


def _include_directory(
    explicit: str | Path | None,
    *,
    environment_name: str,
    header_name: str,
    fallback: Path | None = None,
) -> Path:
    candidate = explicit or os.environ.get(environment_name)
    directory = Path(candidate).expanduser() if candidate else fallback
    if directory is None or not (directory / header_name).is_file():
        raise RuntimeError(
            f"{header_name} was not found; set {environment_name} to its "
            "include directory"
        )
    return directory.resolve()


def _nvrtc_result(result, *, nvrtc, program=None):
    error = result[0]
    if error.value:
        log = ""
        if program is not None:
            log_size_result = nvrtc.nvrtcGetProgramLogSize(program)
            if not log_size_result[0].value:
                log_buffer = b" " * log_size_result[1]
                nvrtc.nvrtcGetProgramLog(program, log_buffer)
                log = log_buffer.decode("utf-8", errors="replace").rstrip("\0")
        error_name = nvrtc.nvrtcGetErrorString(error)[1].decode()
        detail = f"\n{log}" if log else ""
        raise RuntimeError(f"NVRTC failed: {error_name}{detail}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def _compile_trace_cuda(
    *,
    nvrtc,
    optix_include_dir: Path,
    otk_include_dir: Path,
    cuda_include_dir: Path,
) -> str:
    cuda_standard_include_dir = cuda_include_dir / "cccl" / "cuda" / "std"
    cuda_assert_include_dir = cuda_standard_include_dir / "__cccl"
    cuda_cccl_include_dir = cuda_include_dir / "cccl"
    if not (
        (cuda_standard_include_dir / "cassert").is_file()
        and (cuda_assert_include_dir / "assert.h").is_file()
    ):
        raise RuntimeError("CUDA's NVRTC-compatible cassert header was not found")
    source = files("lumo.ray_tracing").joinpath(
        "kernels",
        "trace.cu",
    ).read_bytes()
    program = _nvrtc_result(
        nvrtc.nvrtcCreateProgram(
            source,
            b"trace.cu",
            0,
            [],
            [],
        ),
        nvrtc=nvrtc,
    )
    options = [
        b"-use_fast_math",
        b"-lineinfo",
        b"-default-device",
        b"-std=c++17",
        b"-rdc",
        b"true",
        f"-I{optix_include_dir}".encode(),
        f"-I{otk_include_dir}".encode(),
        f"-I{cuda_standard_include_dir}".encode(),
        f"-I{cuda_assert_include_dir}".encode(),
        f"-I{cuda_cccl_include_dir}".encode(),
        f"-I{cuda_include_dir}".encode(),
    ]
    try:
        _nvrtc_result(
            nvrtc.nvrtcCompileProgram(program, len(options), options),
            nvrtc=nvrtc,
            program=program,
        )
        ptx_size = _nvrtc_result(
            nvrtc.nvrtcGetPTXSize(program),
            nvrtc=nvrtc,
            program=program,
        )
        ptx = b" " * ptx_size
        _nvrtc_result(
            nvrtc.nvrtcGetPTX(program, ptx),
            nvrtc=nvrtc,
            program=program,
        )
        return ptx.decode("utf-8").rstrip("\0")
    finally:
        nvrtc.nvrtcDestroyProgram(program)


def _sbt_header(optix, program_group) -> np.ndarray:
    return np.frombuffer(
        optix.sbtRecordGetHeader(program_group),
        dtype=np.uint8,
    )


class OptixScene:
    """Fingertip silicone and carrier IAS scene."""

    def __init__(
        self,
        fingertip_mesh: FingertipMesh,
        *,
        optix_include_dir: str | Path | None = None,
        otk_include_dir: str | Path | None = None,
    ) -> None:
        from lumo.mesh import FingertipMesh

        if not isinstance(fingertip_mesh, FingertipMesh):
            raise TypeError("fingertip_mesh must be a FingertipMesh")
        silicone_vertices = _vertices(
            fingertip_mesh.silicone.vertices,
            name="fingertip_mesh.silicone.vertices",
        )
        silicone_triangles = _triangles(
            fingertip_mesh.silicone.surface_tri_indices,
            vertex_count=len(silicone_vertices),
            name="fingertip_mesh.silicone.surface_tri_indices",
        )
        bonded_indices = np.asarray(
            fingertip_mesh.bonded_vertex_indices,
            dtype=np.int64,
        )
        if bonded_indices.size and int(bonded_indices.max()) >= len(
            silicone_vertices
        ):
            raise ValueError("bonded vertex index exceeds silicone vertex count")
        bonded_vertices = np.zeros(len(silicone_vertices), dtype=bool)
        bonded_vertices[bonded_indices] = True
        silicone_triangles = np.ascontiguousarray(
            silicone_triangles[
                ~np.all(bonded_vertices[silicone_triangles], axis=1)
            ]
        )

        carrier_vertices = _vertices(
            fingertip_mesh.carrier.vertices,
            name="fingertip_mesh.carrier.vertices",
        )
        carrier_triangles = _triangles(
            fingertip_mesh.carrier.indices,
            vertex_count=len(carrier_vertices),
            name="fingertip_mesh.carrier.indices",
        )
        try:
            import cupy as cp
            import optix
            from cuda.bindings import nvrtc
        except ImportError as exc:
            raise RuntimeError(
                "OptixScene requires pyoptix, cupy, and cuda-python"
            ) from exc
        if tuple(optix.version())[:2] != (9, 1):
            raise RuntimeError(
                "OptixScene currently targets PyOptiX 9.1; found "
                f"{optix.version()}"
            )

        optix_include_path = _include_directory(
            optix_include_dir,
            environment_name="OPTIX_INCLUDE_DIR",
            header_name="optix.h",
        )
        otk_include_path = _include_directory(
            otk_include_dir,
            environment_name="OTK_INCLUDE_DIR",
            header_name=(
                "OptiXToolkit/ShaderUtil/"
                "OptixSelfIntersectionAvoidance.h"
            ),
        )
        cuda_include_path = _include_directory(
            None,
            environment_name="CUDA_INCLUDE_DIR",
            header_name="cuda_runtime.h",
            fallback=(
                Path(sys.prefix)
                / "targets"
                / "x86_64-linux"
                / "include"
            ),
        )

        self._cp = cp
        self._optix = optix
        self._stream = cp.cuda.Stream(non_blocking=True)
        self._geometry_buffers: list[object] = []
        self._accel_buffers: list[object] = []
        self._build_temporaries: list[object] = []
        self._sbt_buffers: list[object] = []
        self._launch_buffers: list[object] = []
        self._log_messages: list[str] = []
        cp.cuda.runtime.free(0)
        context_options = optix.DeviceContextOptions(
            logCallbackFunction=self._log,
            logCallbackLevel=2,
            validationMode=optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL,
        )
        self._context = optix.deviceContextCreate(0, context_options)

        ptx = _compile_trace_cuda(
            nvrtc=nvrtc,
            optix_include_dir=optix_include_path,
            otk_include_dir=otk_include_path,
            cuda_include_dir=cuda_include_path,
        )
        self._create_pipeline(ptx)
        self._create_sbt()

        self._build_silicone_gas(
            silicone_vertices,
            silicone_triangles,
        )
        self._carrier_gas_handle = self._build_triangle_gas(
            carrier_vertices,
            carrier_triangles,
        )
        self._build_ias()

    def _log(self, level: int, tag: str, message: str) -> None:
        self._log_messages.append(f"[{level}][{tag}] {message}")

    def _device_array(self, host_array: np.ndarray):
        with self._stream:
            return self._cp.asarray(host_array)

    def _device_bytes(self, host_array: np.ndarray):
        device_memory = self._cp.cuda.alloc(host_array.nbytes)
        device_memory.copy_from_async(
            ctypes.c_void_p(host_array.ctypes.data),
            host_array.nbytes,
            self._stream,
        )
        return device_memory

    def _build_accel(
        self,
        build_input,
        *,
        build_flags: int,
    ) -> tuple[int, object, object]:
        optix = self._optix
        options = optix.AccelBuildOptions(
            buildFlags=build_flags,
            operation=optix.BUILD_OPERATION_BUILD,
        )
        sizes = self._context.accelComputeMemoryUsage(
            [options],
            [build_input],
        )
        temporary = self._cp.cuda.alloc(sizes.tempSizeInBytes)
        output = self._cp.cuda.alloc(sizes.outputSizeInBytes)
        handle = self._context.accelBuild(
            self._stream.ptr,
            [options],
            [build_input],
            temporary.ptr,
            sizes.tempSizeInBytes,
            output.ptr,
            sizes.outputSizeInBytes,
            [],
        )
        self._build_temporaries.append(temporary)
        self._accel_buffers.append(output)
        return int(handle), sizes, output

    def _build_silicone_gas(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
    ) -> None:
        optix = self._optix
        self._silicone_vertices = self._device_array(vertices)
        self._silicone_triangles = self._device_array(triangles)
        self._silicone_vertex_count = len(vertices)
        self._geometry_buffers.extend(
            (self._silicone_vertices, self._silicone_triangles)
        )

        self._silicone_build_input = optix.BuildInputTriangleArray(
            vertexBuffers_=[self._silicone_vertices.data.ptr],
            vertexFormat=optix.VERTEX_FORMAT_FLOAT3,
            vertexStrideInBytes=3 * np.dtype(np.float32).itemsize,
            indexBuffer=self._silicone_triangles.data.ptr,
            numIndexTriplets=len(triangles),
            indexFormat=optix.INDICES_FORMAT_UNSIGNED_INT3,
            indexStrideInBytes=3 * np.dtype(np.uint32).itemsize,
            flags_=[int(optix.GEOMETRY_FLAG_DISABLE_ANYHIT)],
            numSbtRecords=1,
        )
        self._silicone_build_input.numVertices = len(vertices)

        build_flags = int(optix.BUILD_FLAG_PREFER_FAST_TRACE) | int(
            optix.BUILD_FLAG_ALLOW_UPDATE
        )
        (
            self._silicone_gas_handle,
            sizes,
            self._silicone_gas_output,
        ) = self._build_accel(
            self._silicone_build_input,
            build_flags=build_flags,
        )
        self._silicone_gas_output_size = sizes.outputSizeInBytes
        self._silicone_update_options = optix.AccelBuildOptions(
            buildFlags=build_flags,
            operation=optix.BUILD_OPERATION_UPDATE,
        )
        self._silicone_update_scratch_size = sizes.tempUpdateSizeInBytes
        self._silicone_update_scratch = self._cp.cuda.alloc(
            self._silicone_update_scratch_size
        )

    def _build_triangle_gas(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
    ) -> int:
        optix = self._optix
        device_vertices = self._device_array(vertices)
        device_triangles = self._device_array(triangles)
        self._geometry_buffers.extend((device_vertices, device_triangles))

        build_input = optix.BuildInputTriangleArray(
            vertexBuffers_=[device_vertices.data.ptr],
            vertexFormat=optix.VERTEX_FORMAT_FLOAT3,
            vertexStrideInBytes=3 * np.dtype(np.float32).itemsize,
            indexBuffer=device_triangles.data.ptr,
            numIndexTriplets=len(triangles),
            indexFormat=optix.INDICES_FORMAT_UNSIGNED_INT3,
            indexStrideInBytes=3 * np.dtype(np.uint32).itemsize,
            flags_=[int(optix.GEOMETRY_FLAG_DISABLE_ANYHIT)],
            numSbtRecords=1,
        )
        build_input.numVertices = len(vertices)
        handle, _, _ = self._build_accel(
            build_input,
            build_flags=int(optix.BUILD_FLAG_PREFER_FAST_TRACE),
        )
        return handle

    def _instance_bytes(self) -> np.ndarray:
        optix = self._optix
        identity = [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        instance_specs = (
            (
                self._silicone_gas_handle,
                _SILICONE_INSTANCE_ID,
                _SILICONE_MASK,
            ),
            (
                self._carrier_gas_handle,
                _CARRIER_INSTANCE_ID,
                _CARRIER_MASK,
            ),
        )
        instances = [
            optix.Instance(
                transform=identity,
                instanceId=instance_id,
                sbtOffset=0,
                visibilityMask=visibility_mask,
                flags=int(optix.INSTANCE_FLAG_NONE),
                traversableHandle=handle,
            )
            for handle, instance_id, visibility_mask in instance_specs
        ]
        return np.frombuffer(
            optix.getDeviceRepresentation(instances),
            dtype=np.uint8,
        ).copy()

    def _build_ias(self) -> None:
        optix = self._optix
        instance_bytes = self._instance_bytes()
        self._ias_instances = self._device_array(instance_bytes)
        self._geometry_buffers.append(self._ias_instances)
        self._ias_build_input = optix.BuildInputInstanceArray(
            instances=self._ias_instances.data.ptr,
            numInstances=2,
        )
        build_flags = int(optix.BUILD_FLAG_PREFER_FAST_TRACE) | int(
            optix.BUILD_FLAG_ALLOW_UPDATE
        )
        self._ias_handle, sizes, self._ias_output = self._build_accel(
            self._ias_build_input,
            build_flags=build_flags,
        )
        self._ias_output_size = sizes.outputSizeInBytes
        self._ias_update_options = optix.AccelBuildOptions(
            buildFlags=build_flags,
            operation=optix.BUILD_OPERATION_UPDATE,
        )
        self._ias_update_scratch_size = sizes.tempUpdateSizeInBytes
        self._ias_update_scratch = self._cp.cuda.alloc(
            self._ias_update_scratch_size
        )

    def _create_pipeline(self, ptx: str) -> None:
        optix = self._optix
        self._pipeline_compile_options = optix.PipelineCompileOptions(
            usesMotionBlur=False,
            traversableGraphFlags=int(
                optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING
            ),
            numPayloadValues=_PAYLOAD_WORD_COUNT,
            numAttributeValues=2,
            exceptionFlags=int(optix.EXCEPTION_FLAG_NONE),
            pipelineLaunchParamsVariableName="params",
            usesPrimitiveTypeFlags=int(optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE),
        )
        module_options = optix.ModuleCompileOptions(
            maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
            debugLevel=optix.COMPILE_DEBUG_LEVEL_DEFAULT,
        )
        self._module, module_log = self._context.moduleCreate(
            module_options,
            self._pipeline_compile_options,
            ptx,
        )
        if module_log:
            self._log_messages.append(module_log)

        descriptions = (
            optix.ProgramGroupDesc(
                raygenModule=self._module,
                raygenEntryFunctionName="__raygen__trace_closest",
            ),
            optix.ProgramGroupDesc(
                missModule=self._module,
                missEntryFunctionName="__miss__trace_closest",
            ),
            optix.ProgramGroupDesc(
                hitgroupModuleCH=self._module,
                hitgroupEntryFunctionNameCH="__closesthit__triangle",
            ),
        )
        self._program_groups = []
        for description in descriptions:
            groups, log = self._context.programGroupCreate([description])
            self._program_groups.append(groups[0])
            if log:
                self._log_messages.append(log)

        self._pipeline = self._context.pipelineCreate(
            self._pipeline_compile_options,
            optix.PipelineLinkOptions(maxTraceDepth=1),
            self._program_groups,
            "",
        )
        stack_sizes = optix.StackSizes()
        for program_group in self._program_groups:
            optix.util.accumulateStackSizes(
                program_group,
                stack_sizes,
                self._pipeline,
            )
        stack_from_traversal, stack_from_state, continuation_stack = (
            optix.util.computeStackSizes(stack_sizes, 1, 0, 0)
        )
        self._pipeline.setStackSize(
            stack_from_traversal,
            stack_from_state,
            continuation_stack,
            2,
        )

    def _create_sbt(self) -> None:
        optix = self._optix
        header_size = optix.SBT_RECORD_HEADER_SIZE
        alignment = optix.SBT_RECORD_ALIGNMENT

        header_record_size = (
            (header_size + alignment - 1) // alignment * alignment
        )
        header_dtype = np.dtype(
            {
                "names": ["header"],
                "formats": [(np.uint8, header_size)],
                "offsets": [0],
                "itemsize": header_record_size,
            }
        )
        raygen_record = np.zeros(1, dtype=header_dtype)
        raygen_record["header"][0] = _sbt_header(
            optix,
            self._program_groups[0],
        )
        miss_record = np.zeros(1, dtype=header_dtype)
        miss_record["header"][0] = _sbt_header(
            optix,
            self._program_groups[1],
        )

        hit_records = np.zeros(1, dtype=header_dtype)
        hit_records["header"][0] = _sbt_header(
            optix,
            self._program_groups[2],
        )

        device_raygen = self._device_bytes(raygen_record)
        device_miss = self._device_bytes(miss_record)
        device_hits = self._device_bytes(hit_records)
        self._sbt_buffers.extend((device_raygen, device_miss, device_hits))
        self._sbt = optix.ShaderBindingTable(
            raygenRecord=device_raygen.ptr,
            missRecordBase=device_miss.ptr,
            missRecordStrideInBytes=miss_record.dtype.itemsize,
            missRecordCount=1,
            hitgroupRecordBase=device_hits.ptr,
            hitgroupRecordStrideInBytes=hit_records.dtype.itemsize,
            hitgroupRecordCount=1,
        )

    def update_silicone(self, vertices: np.ndarray) -> None:
        """Refit the silicone GAS and IAS after a vertex-position update."""
        vertices = _vertices(vertices, name="vertices")
        if len(vertices) != self._silicone_vertex_count:
            raise ValueError(
                "vertices must preserve the silicone vertex count "
                f"({self._silicone_vertex_count})"
            )

        self._silicone_vertices.set(vertices, stream=self._stream)
        self._silicone_gas_handle = int(
            self._context.accelBuild(
                self._stream.ptr,
                [self._silicone_update_options],
                [self._silicone_build_input],
                self._silicone_update_scratch.ptr,
                self._silicone_update_scratch_size,
                self._silicone_gas_output.ptr,
                self._silicone_gas_output_size,
                [],
            )
        )

        instance_bytes = self._instance_bytes()
        self._ias_instances.set(instance_bytes, stream=self._stream)
        self._ias_handle = int(
            self._context.accelBuild(
                self._stream.ptr,
                [self._ias_update_options],
                [self._ias_build_input],
                self._ias_update_scratch.ptr,
                self._ias_update_scratch_size,
                self._ias_output.ptr,
                self._ias_output_size,
                [],
            )
        )

    def trace_closest(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        *,
        mask: int = 0xFF,
    ) -> np.ndarray:
        """Trace normalized metric rays once and return a structured array."""
        origins = _vertices(origins, name="origins")
        directions = _vertices(directions, name="directions")
        if len(origins) != len(directions):
            raise ValueError("origins and directions must contain equal ray counts")
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms == 0.0):
            raise ValueError("ray directions must be nonzero")
        directions = np.ascontiguousarray(
            directions / norms[:, None],
            dtype=np.float32,
        )
        mask = _visibility_mask(mask, name="mask")

        device_origins = self._device_array(origins)
        device_directions = self._device_array(directions)
        with self._stream:
            device_results = self._cp.empty(
                (len(origins), _RESULT_WORD_COUNT),
                dtype=self._cp.uint32,
            )

        params_dtype = np.dtype(
            {
                "names": [
                    "origins",
                    "directions",
                    "results",
                    "handle",
                    "mask",
                ],
                "formats": [np.uint64, np.uint64, np.uint64, np.uint64, np.uint32],
                "offsets": [0, 8, 16, 24, 32],
                "itemsize": 40,
            }
        )
        host_params = np.zeros(1, dtype=params_dtype)
        host_params["origins"] = device_origins.data.ptr
        host_params["directions"] = device_directions.data.ptr
        host_params["results"] = device_results.data.ptr
        host_params["handle"] = self._ias_handle
        host_params["mask"] = mask
        device_params = self._device_bytes(host_params)
        self._launch_buffers = [
            device_origins,
            device_directions,
            device_results,
            device_params,
        ]

        self._optix.launch(
            self._pipeline,
            self._stream.ptr,
            device_params.ptr,
            host_params.dtype.itemsize,
            self._sbt,
            len(origins),
            1,
            1,
        )
        raw = device_results.get(stream=self._stream)

        results = np.empty(len(origins), dtype=_RESULT_DTYPE)
        results["hit"] = raw[:, 0] != 0
        results["t"] = raw[:, 1].view(np.float32)
        results["instance_id"] = raw[:, 2].view(np.int32)
        results["primitive_id"] = raw[:, 3].view(np.int32)
        results["barycentrics"][:, 0] = raw[:, 4].view(np.float32)
        results["barycentrics"][:, 1] = raw[:, 5].view(np.float32)
        results["normal_W"][:, 0] = raw[:, 6].view(np.float32)
        results["normal_W"][:, 1] = raw[:, 7].view(np.float32)
        results["normal_W"][:, 2] = raw[:, 8].view(np.float32)
        results["spawn_front_W"][:, 0] = raw[:, 9].view(np.float32)
        results["spawn_front_W"][:, 1] = raw[:, 10].view(np.float32)
        results["spawn_front_W"][:, 2] = raw[:, 11].view(np.float32)
        results["spawn_back_W"][:, 0] = raw[:, 12].view(np.float32)
        results["spawn_back_W"][:, 1] = raw[:, 13].view(np.float32)
        results["spawn_back_W"][:, 2] = raw[:, 14].view(np.float32)
        return results


def safe_secondary_origins(
    hits: np.ndarray,
    outgoing_directions: np.ndarray,
) -> np.ndarray:
    """Select each OTK safe spawn point from its outgoing direction."""
    if hits.ndim != 1 or hits.dtype.names is None:
        raise ValueError("hits must be a one-dimensional structured array")
    required_fields = {
        "hit",
        "normal_W",
        "spawn_front_W",
        "spawn_back_W",
    }
    if not required_fields.issubset(hits.dtype.names):
        raise ValueError("hits must be results from OptixScene.trace_closest()")

    outgoing_directions = _vertices(
        outgoing_directions,
        name="outgoing_directions",
    )
    if len(hits) != len(outgoing_directions):
        raise ValueError("hits and outgoing_directions must have equal lengths")
    if np.any(np.linalg.norm(outgoing_directions, axis=1) == 0.0):
        raise ValueError("outgoing_directions must be nonzero")
    if not np.all(hits["hit"]):
        raise ValueError("safe secondary origins require triangle hits")
    if not (
        np.all(np.isfinite(hits["normal_W"]))
        and np.all(np.isfinite(hits["spawn_front_W"]))
        and np.all(np.isfinite(hits["spawn_back_W"]))
    ):
        raise ValueError("safe secondary origins require triangle spawn data")

    use_front = np.sum(
        outgoing_directions * hits["normal_W"],
        axis=1,
    ) > 0.0
    return np.ascontiguousarray(
        np.where(
            use_front[:, None],
            hits["spawn_front_W"],
            hits["spawn_back_W"],
        ),
        dtype=np.float32,
    )


__all__ = ["OptixScene", "safe_secondary_origins"]
