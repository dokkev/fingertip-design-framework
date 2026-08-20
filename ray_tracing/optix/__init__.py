"""Optional NVIDIA OptiX execution boundary."""

from ray_tracing.optix.runtime import OptixRuntime, OptixRuntimeError

__all__ = ["OptixRuntime", "OptixRuntimeError"]
