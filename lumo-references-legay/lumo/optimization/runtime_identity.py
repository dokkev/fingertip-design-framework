"""Exact GPU/runtime identity used by deterministic evaluation contracts."""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=None)
def runtime_identity_for_device(device: str) -> dict[str, Any]:
    """Resolve the architecture and numerical backend versions for ``device``."""

    if not isinstance(device, str) or not device.startswith("cuda:"):
        raise ValueError("runtime identity requires a cuda:<index> device")
    try:
        ordinal = int(device.split(":", 1)[1])
    except ValueError as exc:
        raise ValueError("runtime identity requires a cuda:<index> device") from exc
    try:
        import cupy as cp
        import newton
        import optix
        import warp as wp

        properties = cp.cuda.runtime.getDeviceProperties(ordinal)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        optix_version = tuple(int(value) for value in optix.version())
        return {
            "status": "available",
            "device": device,
            "device_ordinal": ordinal,
            "gpu_name": str(name),
            "compute_capability": (
                f"{int(properties['major'])}.{int(properties['minor'])}"
            ),
            "warp_version": str(wp.__version__),
            "newton_version": str(newton.__version__),
            "optix_binding_version": str(optix.__version__),
            "optix_runtime_version": ".".join(str(value) for value in optix_version),
            "cupy_version": str(cp.__version__),
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "status": "unavailable",
            "device": device,
            "device_ordinal": ordinal,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


__all__ = ["runtime_identity_for_device"]
