"""Typed optical configuration contracts used by optimization boundaries.

This module owns the stable inputs and fingerprints for FULL_3D transport.
Artifact files and JSON schema handling remain in :mod:`optical_artifact`.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any, Mapping

from lumo.finger.fingertip import Fingertip
from lumo.ray_tracing.optical_mechanics.settings import Transport3DSettings


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint_mapping(value: Mapping[str, Any]) -> str:
    """Return the stable fingerprint used by optimization contracts."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def optical_physics_parameters(tip: Fingertip) -> dict[str, float]:
    """Return exactly the optical values used by FULL_3D transport."""
    if not isinstance(tip, Fingertip):
        raise TypeError("tip must be a Fingertip")
    return {
        "refractive_index_air": float(tip.optical.refractive_index_air),
        "refractive_index_silicone": float(tip.optical.refractive_index_silicone),
        "absorption_per_mm": float(tip.optical.absorption_per_mm),
        "relative_radiant_power": float(tip.led.relative_radiant_power),
        "emission_half_angle_deg": float(tip.led.emission_half_angle_deg),
    }


def transport_configuration(
    settings: Transport3DSettings,
    *,
    material: Mapping[str, Any],
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize only inputs that can change FULL_3D transport."""
    configuration: dict[str, Any] = {
        "schema": "full3d-transport-configuration-v1",
        "settings": asdict(settings),
        "material": dict(material),
    }
    if source is not None:
        configuration["source"] = dict(source)
    return configuration


__all__ = [
    "fingerprint_mapping",
    "optical_physics_parameters",
    "transport_configuration",
]
