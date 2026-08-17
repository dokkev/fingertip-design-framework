"""Optional NVIDIA OptiX environment helpers."""

from optics.optix.paths import (
    IncludeCandidate,
    IncludeResolution,
    OptixCudaPaths,
    diagnose_paths,
    discover_paths,
)
from optics.optix.runtime import OptixRuntime, OptixRuntimeError

__all__ = [
    "IncludeCandidate",
    "IncludeResolution",
    "OptixCudaPaths",
    "OptixRuntime",
    "OptixRuntimeError",
    "diagnose_paths",
    "discover_paths",
]
