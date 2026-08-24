"""Optional NVIDIA OptiX execution boundary."""

from lumo.ray_tracing.optix.runtime import OptixRuntime, OptixRuntimeError

__all__ = ["OptixRuntime", "OptixRuntimeError"]
