"""Optical interface and attenuation conventions shared by focused checks."""

from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np


class Transport3DPhysicsError(RuntimeError):
    """Raised when a deterministic interface calculation is invalid."""


def interface_split(
    xp: Any,
    incident: Any,
    normal: Any,
    medium: Any,
    refractive_index_air: float,
    refractive_index_silicone: float,
) -> tuple[Any, Any, Any, Any]:
    """Return reflected/transmitted directions, reflectance, and TIR mask.

    ``normal`` points into the transmitted medium.  The expression is written
    against an array namespace so the same convention runs on NumPy for
    analytic checks and on CuPy for the OptiX wavefront.
    """
    n1 = xp.where(medium == 0, refractive_index_air, refractive_index_silicone)
    n2 = xp.where(medium == 0, refractive_index_silicone, refractive_index_air)
    alignment = xp.sum(incident * normal, axis=1)
    if bool(xp.any(alignment <= 1.0e-7)):
        bad = int(xp.asnumpy(xp.where(alignment <= 1.0e-7)[0][0])) if hasattr(xp, "asnumpy") else int(np.where(alignment <= 1.0e-7)[0][0])
        raise Transport3DPhysicsError(
            "interface normal does not point into transmitted medium: "
            f"alignment={float(xp.asnumpy(alignment[bad]) if hasattr(xp, 'asnumpy') else alignment[bad]):g}, "
            f"medium={int(xp.asnumpy(medium[bad]) if hasattr(xp, 'asnumpy') else medium[bad])}, "
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


def interface_directions_and_reflectance(
    incident_direction: np.ndarray,
    interface_normal: np.ndarray,
    refractive_index_1: float,
    refractive_index_2: float,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Scalar NumPy form matching the reduced 2D Fresnel convention."""
    incident = np.asarray(incident_direction, dtype=float)
    normal = np.asarray(interface_normal, dtype=float)
    incident /= np.linalg.norm(incident)
    normal /= np.linalg.norm(normal)
    if float(np.dot(incident, normal)) <= 1.0e-7:
        raise Transport3DPhysicsError("interface normal does not point forward")
    # Medium 0/1 is selected below only to reuse the vector formula; explicit
    # indices are passed through the equivalent air/silicone ratio.
    if refractive_index_1 <= 0.0 or refractive_index_2 <= 0.0:
        raise Transport3DPhysicsError("refractive indices must be positive")
    ratio = refractive_index_1 / refractive_index_2
    tangent = incident - float(np.dot(incident, normal)) * normal
    sin_transmitted_squared = ratio * ratio * float(np.dot(tangent, tangent))
    reflected = incident - 2.0 * float(np.dot(incident, normal)) * normal
    reflected /= np.linalg.norm(reflected)
    if sin_transmitted_squared > 1.0:
        return reflected, None, 1.0
    cos_i = abs(float(np.dot(incident, normal)))
    cos_t = max(0.0, 1.0 - sin_transmitted_squared) ** 0.5
    transmitted = ratio * tangent + cos_t * normal
    transmitted /= np.linalg.norm(transmitted)
    s_denominator = refractive_index_1 * cos_i + refractive_index_2 * cos_t
    p_denominator = refractive_index_1 * cos_t + refractive_index_2 * cos_i
    if s_denominator <= 0.0 or p_denominator <= 0.0:
        raise Transport3DPhysicsError("Fresnel denominator is nonpositive")
    reflectance_s = (
        (refractive_index_1 * cos_i - refractive_index_2 * cos_t)
        / s_denominator
    ) ** 2
    reflectance_p = (
        (refractive_index_1 * cos_t - refractive_index_2 * cos_i)
        / p_denominator
    ) ** 2
    return reflected, transmitted, float(np.clip(0.5 * (reflectance_s + reflectance_p), 0.0, 1.0))


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
    "interface_directions_and_reflectance",
    "interface_split",
    "periodic_plane_distance",
    "wrapped_periodic_z",
]
