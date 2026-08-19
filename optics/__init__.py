"""Public deterministic optical-transport API."""

from optics.contact_object import (
    CarrierOptics,
    IndenterOptics,
    InvalidIndenterOptics,
    ObjectBoundaryOptics,
)
from optics.transport import ExitEvent, RaySegment, TraceSettings, TransportResult, trace
from optics.metrics import evaluate, field_difference

__all__ = [
    "RaySegment",
    "ExitEvent",
    "TraceSettings",
    "TransportResult",
    "IndenterOptics",
    "CarrierOptics",
    "ObjectBoundaryOptics",
    "InvalidIndenterOptics",
    "evaluate",
    "field_difference",
    "trace",
]
