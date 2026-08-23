"""Ray-tracing ownership for the new LUMO implementation."""

from .scene import OptixScene, safe_secondary_origins
from .transport import interface_transport, lambertian_reflection

__all__ = [
    "OptixScene",
    "interface_transport",
    "lambertian_reflection",
    "safe_secondary_origins",
]
