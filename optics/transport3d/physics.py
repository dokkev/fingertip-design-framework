"""Optical interface and attenuation conventions shared by focused checks."""

from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np


class Transport3DPhysicsError(RuntimeError):
    """Raised when a deterministic interface calculation is invalid."""


def _interface_split_indices(
    xp: Any,
    incident: Any,
    normal: Any,
    n1: Any,
    n2: Any,
) -> tuple[Any, Any, Any, Any]:
    """Return reflected/transmitted directions, reflectance, and TIR mask.

    ``normal`` points into the transmitted medium.  The expression is written
    against an array namespace so the same convention runs on NumPy for
    analytic checks and on CuPy for the OptiX wavefront.
    """
    alignment = xp.sum(incident * normal, axis=1)
    if bool(xp.any(alignment <= 1.0e-7)):
        bad = int(xp.asnumpy(xp.where(alignment <= 1.0e-7)[0][0])) if hasattr(xp, "asnumpy") else int(np.where(alignment <= 1.0e-7)[0][0])
        raise Transport3DPhysicsError(
            "interface normal does not point into transmitted medium: "
            f"alignment={float(xp.asnumpy(alignment[bad]) if hasattr(xp, 'asnumpy') else alignment[bad]):g}, "
            f"incident={np.asarray(xp.asnumpy(incident[bad]) if hasattr(xp, 'asnumpy') else incident[bad])}, "
            f"normal={np.asarray(xp.asnumpy(normal[bad]) if hasattr(xp, 'asnumpy') else normal[bad])}"
        )
    tangent = incident - alignment[:, None] * normal
    eta = n1 / n2
    tangent_squared = xp.sum(tangent * tangent, axis=1)
    transmitted_sine_squared = eta * eta * tangent_squared
    tir = transmitted_sine_squared > 1.0
    reflected = incident - 2.0 * alignment[:, None] * normal
    reflected /= xp.linalg.norm(reflected, axis=1)[:, None]
    transmitted_cosine = xp.sqrt(xp.maximum(0.0, 1.0 - transmitted_sine_squared))
    transmitted = eta[:, None] * tangent + transmitted_cosine[:, None] * normal
    transmitted_norm = xp.linalg.norm(transmitted, axis=1)
    transmitted /= xp.maximum(transmitted_norm, 1.0e-30)[:, None]
    cosine_incident = xp.abs(alignment)
    s_denominator = n1 * cosine_incident + n2 * transmitted_cosine
    p_denominator = n1 * transmitted_cosine + n2 * cosine_incident
    if bool(xp.any((s_denominator <= 0.0) | (p_denominator <= 0.0))):
        raise Transport3DPhysicsError("Fresnel denominator is nonpositive")
    reflectance_s = (
        (n1 * cosine_incident - n2 * transmitted_cosine) / s_denominator
    ) ** 2
    reflectance_p = (
        (n1 * transmitted_cosine - n2 * cosine_incident) / p_denominator
    ) ** 2
    reflectance = xp.clip(0.5 * (reflectance_s + reflectance_p), 0.0, 1.0)
    reflectance = xp.where(tir, 1.0, reflectance)
    transmitted = xp.where(tir[:, None], 0.0, transmitted)
    return reflected, transmitted, reflectance, tir


def interface_split(
    xp: Any,
    incident: Any,
    normal: Any,
    medium: Any,
    refractive_index_air: float,
    refractive_index_silicone: float,
) -> tuple[Any, Any, Any, Any]:
    """Split an air/silicone interface using the shared Fresnel convention."""
    n1 = xp.where(medium == 0, refractive_index_air, refractive_index_silicone)
    n2 = xp.where(medium == 0, refractive_index_silicone, refractive_index_air)
    return _interface_split_indices(xp, incident, normal, n1, n2)


def object_interface_split(
    xp: Any,
    incident: Any,
    normal: Any,
    refractive_index_silicone: float,
    refractive_index_object: float,
) -> tuple[Any, Any, Any, Any]:
    """Split silicone-to-object rays without tracing the object interior."""
    count = incident.shape[0]
    n1 = xp.full(count, refractive_index_silicone, dtype=incident.dtype)
    n2 = xp.full(count, refractive_index_object, dtype=incident.dtype)
    return _interface_split_indices(xp, incident, normal, n1, n2)


def attenuated_weight(
    weight: float,
    length_mm: float,
    *,
    medium: str,
    absorption_per_mm: float,
) -> tuple[float, float]:
    """Return post-segment weight and removed weight under reduced conventions."""
    if not all(isfinite(value) for value in (weight, length_mm, absorption_per_mm)):
        raise Transport3DPhysicsError("attenuation inputs must be finite")
    if weight < 0.0 or length_mm < 0.0 or absorption_per_mm < 0.0:
        raise Transport3DPhysicsError("attenuation inputs must be nonnegative")
    if medium == "air":
        end = weight
    elif medium == "silicone":
        end = weight * float(np.exp(-absorption_per_mm * length_mm))
    else:
        raise Transport3DPhysicsError(f"unsupported propagating medium: {medium!r}")
    return end, weight - end


def periodic_plane_distance(
    xp: Any,
    z: Any,
    direction_z: Any,
    *,
    z_min_mm: float,
    z_max_mm: float,
    epsilon_mm: float,
) -> Any:
    """Return forward travel to the next periodic z plane, or infinity."""
    infinity = xp.asarray(xp.inf)
    distance = xp.where(
        direction_z > epsilon_mm,
        (z_max_mm - z) / direction_z,
        xp.where(
            direction_z < -epsilon_mm,
            (z_min_mm - z) / direction_z,
            infinity,
        ),
    )
    return xp.where(distance > epsilon_mm, distance, infinity)


def wrapped_periodic_z(
    xp: Any,
    direction_z: Any,
    *,
    z_min_mm: float,
    z_max_mm: float,
    offset_mm: float,
) -> Any:
    """Map a crossed periodic plane to the opposite plane with an offset."""
    return xp.where(direction_z > 0.0, z_min_mm, z_max_mm) + offset_mm * direction_z


__all__ = [
    "Transport3DPhysicsError",
    "attenuated_weight",
    "interface_split",
    "object_interface_split",
    "periodic_plane_distance",
    "wrapped_periodic_z",
]
